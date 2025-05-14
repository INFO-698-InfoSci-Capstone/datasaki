import numpy as np
import json
from app.utils import grid_columns, grid_formatter, json_int, json_float
from fastapi import APIRouter, Depends, HTTPException
from app import models, crud
from app.database import get_db
from app.utils import get_dtypes, classify_type, find_dtype, build_formatters, build_string_metrics, build_sequential_diffs
from app.dependencies import get_current_user
from sqlalchemy.orm import Session
from app.arcticdb_utils import ArcticDBInstance
from app.query import load_filterable_data
from app.code_export import build_code_export
router = APIRouter()

def load_describe(column_series, additional_aggs=None):
    """
    Helper function for grabbing the output from :meth:`pandas:pandas.Series.describe` in a JSON serializable format

    :param column_series: data to describe
    :type column_series: :class:`pandas:pandas.Series`
    :return: JSON serializable dictionary of the output from calling :meth:`pandas:pandas.Series.describe`
    """
    desc = column_series.describe().to_frame().T
    code = [
        "# main statistics",
        "stats = df['{col}'].describe().to_frame().T".format(col=column_series.name),
    ]
    if additional_aggs:
        for agg in additional_aggs:
            if agg == "mode":
                mode = column_series.mode().values
                desc["mode"] = np.nan if len(mode) > 1 else mode[0]
                code.append(
                    (
                        "# mode\n"
                        "mode = df['{col}'].mode().values\n"
                        "stats['mode'] = np.nan if len(mode) > 1 else mode[0]"
                    ).format(col=column_series.name)
                )
                continue
            desc[agg] = getattr(column_series, agg)()
            code.append(
                "# {agg}\nstats['{agg}'] = df['{col}'].{agg}()".format(
                    col=column_series.name, agg=agg
                )
            )
    desc_f_overrides = {
        "I": lambda f, i, c: f.add_int(i, c, as_string=True),
        "F": lambda f, i, c: f.add_float(i, c, precision=4, as_string=True),
    }
    desc_f = grid_formatter(
        grid_columns(desc), nan_display="nan", overrides=desc_f_overrides
    )
    desc = desc_f.format_dict(next(desc.itertuples(), None))
    if "count" in desc:
        # pandas always returns 'count' as a float and it adds useless decimal points
        desc["count"] = desc["count"].split(".")[0]
    desc["total_count"] = json_int(len(column_series), as_string=True)
    missing_ct = column_series.isnull().sum()
    desc["missing_pct"] = json_float((missing_ct / len(column_series) * 100).round(2))
    desc["missing_ct"] = json_int(missing_ct, as_string=True)
    return desc, code

@router.get("/datasets/{dataset_id}/dtypes", response_model=list)
def fetch_dtypes(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    API endpoint to get column data types (dtypes) of the dataset
    :param dataset_id: Identifier for the dataset
    :param db: Database session
    :param current_user: The current authenticated user
    :return: Dictionary of column names and their corresponding data types
    """
    # Retrieve the dataset
    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied")
    data_instance = ArcticDBInstance(dataset_id=dataset_id,tenant_id=current_user.tenant_id)
    return data_instance.dtypes


@router.post("/datasets/{dataset_id}/describe/{column}")
def describe(
    dataset_id: int,
    column:str,
    request:dict,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    :class:`flask:flask.Flask` route which returns standard details about column data using
    :meth:`pandas:pandas.DataFrame.describe` to the front-end as JSON

    :param data_id: integer string identifier for a D-Tale process's data
    :type data_id: str
    :param column: required dash separated string "START-END" stating a range of row indexes to be returned
                   to the screen
    :return: JSON {
        describe: object representing output from :meth:`pandas:pandas.Series.describe`,
        unique_data: array of unique values when data has <= 100 unique values
        success: True/False
    }

    """
    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied")
    data_instance = ArcticDBInstance(dataset_id=dataset_id,tenant_id=current_user.tenant_id)
    columns_to_load = [column]
    curr_settings = data_instance.settings
    indexes = curr_settings.get("indexes", [])
    if column in indexes:
        column_to_load = next(
            (
                c
                for c in data_instance.dtypes
                if c["name"] not in indexes
            ),
            None,
        )
        columns_to_load = [column_to_load["name"]] if column_to_load else None
    data = load_filterable_data(dataset_id, current_user.tenant_id,req=request,query=None,columns=columns_to_load)
    additional_aggs = None
    dtype = data_instance.get_dtype_info(col=column)
    classification = classify_type(dtype["dtype"])
    if classification == "I":
        additional_aggs = ["sum", "median", "mode", "var", "sem"]
    elif classification == "F":
        additional_aggs = ["sum", "median", "var", "sem"]
    code = build_code_export(dataset_id,tenant_id=current_user.tenant_id)
    desc, desc_code = load_describe(data[column], additional_aggs=additional_aggs)
    code += desc_code
    return_data = dict(describe=desc, success=True)
    if "unique" not in return_data["describe"] and "unique_ct" in dtype:
        return_data["describe"]["unique"] = json_int(dtype["unique_ct"], as_string=True)
    for p in ["skew", "kurt"]:
        if p in dtype:
            return_data["describe"][p] = dtype[p]

    if classification != "F":
        uniq_vals = data[column].value_counts().sort_values(ascending=False)
        uniq_vals.index.name = "value"
        uniq_vals.name = "count"
        uniq_vals = uniq_vals.reset_index()

        # build top
        top_freq = uniq_vals["count"].values[0]
        top_freq_pct = (top_freq / uniq_vals["count"].sum()) * 100
        top_vals = (
            uniq_vals[uniq_vals["count"] == top_freq].sort_values("value").head(5)
        )
        top_vals_f = grid_formatter(grid_columns(top_vals), as_string=True)
        top_vals = top_vals_f.format_lists(top_vals)
        return_data["describe"]["top"] = "{} ({}%)".format(
            ", ".join(top_vals["value"]), json_float(top_freq_pct, as_string=True)
        )
        return_data["describe"]["freq"] = int(top_freq)

        code.append(
            (
                "uniq_vals = data['{}'].value_counts().sort_values(ascending=False)\n"
                "uniq_vals.index.name = 'value'\n"
                "uniq_vals.name = 'count'\n"
                "uniq_vals = uniq_vals.reset_index()"
            ).format(column)
        )

        if dtype["dtype"].startswith("mixed"):
            uniq_vals["type"] = apply(uniq_vals["value"], lambda i: type(i).__name__)
            dtype_counts = uniq_vals.groupby("type")["count"].sum().reset_index()
            dtype_counts.columns = ["dtype", "count"]
            return_data["dtype_counts"] = dtype_counts.to_dict(orient="records")
            code.append(
                (
                    "uniq_vals['type'] = uniq_vals['value'].apply( lambda i: type(i).__name__)\n"
                    "dtype_counts = uniq_vals.groupby('type')['count'].sum().reset_index()\n"
                    "dtype_counts.columns = ['dtype', 'count']"
                )
            )
        else:
            uniq_vals.loc[:, "type"] = find_dtype(uniq_vals["value"])
            code.append(
                "uniq_vals.loc[:, 'type'] = '{}'".format(uniq_vals["type"].values[0])
            )

        return_data["uniques"] = {}
        for uniq_type, uniq_grp in uniq_vals.groupby("type"):
            total = len(uniq_grp)
            top = total > 100
            uniq_grp = (
                uniq_grp[["value", "count"]]
                .sort_values(["count", "value"], ascending=[False, True])
                .head(100)
            )
            # pandas started supporting string dtypes in 1.1.0
            conversion_type = (
                uniq_type
                if uniq_type == "string"
                else "object"
            )
            uniq_grp["value"] = uniq_grp["value"].astype(conversion_type)
            uniq_f, _ = build_formatters(uniq_grp)
            return_data["uniques"][uniq_type] = dict(
                data=uniq_f.format_dicts(uniq_grp.itertuples()), total=total, top=top
            )

    if (
        classification in ["I", "F", "D"]
    ):
        sd_metrics, sd_code = build_sequential_diffs(data[column], column)
        return_data["sequential_diffs"] = sd_metrics
        code.append(sd_code)

    if classification == "S":
        str_col = data[column]
        sm_metrics, sm_code = build_string_metrics(
            str_col[~str_col.isnull()].astype("str").str, column
        )
        return_data["string_metrics"] = sm_metrics
        code += sm_code

    return_data["code"] = "\n".join(code)
    return return_data