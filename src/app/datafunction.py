import numpy as np
import json
from app.utils import grid_columns, grid_formatter, json_int, json_float
from fastapi import APIRouter, Depends, HTTPException
from app import models, crud
from app.database import get_db
from app.utils import get_dtypes, classify_type, find_dtype, build_formatters, build_string_metrics, build_sequential_diffs, unique_count, find_dtype_formatter
from app.dependencies import get_current_user
from sqlalchemy.orm import Session
from app.arcticdb_utils import ArcticDBInstance
from app.query import load_filterable_data
from app.code_export import build_code_export
router = APIRouter()

@router.get("/variance/{dataset_id}")
def variance(    dataset_id: int,
    request:dict,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)):
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
    column = get_str_arg(request, "col")
    data = load_filterable_data(data_id,tenant_id=current_user.tenant_id,req=request)
    s = data[column]
    code = ["s = df['{}']".format(column)]
    unique_ct = unique_count(s)
    code.append("unique_ct = s.unique().size")
    s_size = len(s)
    code.append("s_size = len(s)")
    check1 = bool((unique_ct / s_size) < 0.1)
    code.append("check1 = (unique_ct / s_size) < 0.1")
    return_data = dict(check1=dict(unique=unique_ct, size=s_size, result=check1))
    dtype = data_instance.get_dtype_info(col=column)
    if unique_ct >= 2:
        val_counts = s.value_counts()
        check2 = bool((val_counts.values[0] / val_counts.values[1]) > 20)
        fmt = find_dtype_formatter(dtype["dtype"])
        return_data["check2"] = dict(
            val1=dict(val=fmt(val_counts.index[0]), ct=int(val_counts.values[0])),
            val2=dict(val=fmt(val_counts.index[1]), ct=int(val_counts.values[1])),
            result=check2,
        )
    code += [
        "check2 = False",
        "if unique_ct > 1:",
        "\tval_counts = s.value_counts()",
        "\tcheck2 = (val_counts.values[0] / val_counts.values[1]) > 20",
        "low_variance = check1 and check2",
    ]

    return_data["size"] = len(s)
    return_data["outlierCt"] = dtype["hasOutliers"]
    return_data["missingCt"] = int(s.isnull().sum())

    jb_stat, jb_p = sts.jarque_bera(s)
    return_data["jarqueBera"] = dict(statistic=float(jb_stat), pvalue=float(jb_p))
    sw_stat, sw_p = sts.shapiro(s)
    return_data["shapiroWilk"] = dict(statistic=float(sw_stat), pvalue=float(sw_p))
    code += [
        "\nimport scipy.stats as sts\n",
        "jb_stat, jb_p = sts.jarque_bera(s)",
        "sw_stat, sw_p = sts.shapiro(s)",
    ]
    return_data["code"] = "\n".join(code)
    return jsonify(return_data)