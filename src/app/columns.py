import numpy as np
import json

from app.crud import create_user
from app.utils import grid_columns, grid_formatter, json_int, json_float, get_column_stats
from fastapi import APIRouter, Depends, HTTPException
from app import models, crud
from app.database import get_db
from app.utils import (get_dtypes, classify_type, find_dtype, build_formatters, build_string_metrics,
                       build_sequential_diffs, get_str_arg,dtype_formatter, get_column_stats, NpEncoder, make_list)
from app.dependencies import get_current_user
from sqlalchemy.orm import Session
from app.arcticdb_utils import ArcticDBInstance
from app.schemas import ColumnBuilderSchema, ColumnAnalysisSchema
from app.column_builders import ColumnBuilder
from app.column_analysis import ColumnAnalysis
import pandas as pd

router = APIRouter()


@router.post("/build-column/{dataset_id}")
def build_column( request:ColumnBuilderSchema,
                  dataset_id: int,
                  db: Session = Depends(get_db),
                  current_user: models.User = Depends(get_current_user)):
    """
    :class:`flask:flask.Flask` route to handle the building of new columns in a dataframe. Some of the operations the
    are available are:
     - numeric: sum/difference/multiply/divide any combination of two columns or static values
     - datetime: retrieving date properties (hour, minute, month, year...) or conversions of dates (month start, month
                 end, quarter start...)
     - bins: bucketing numeric data into bins using :meth:`pandas:pandas.cut` & :meth:`pandas:pandas.qcut`

    :param data_id: integer string identifier for a D-Tale process's data
    :type data_id: str
    :param name: string from flask.request.args['name'] of new column to create
    :param type: string from flask.request.args['type'] of the type of column to build (numeric/datetime/bins)
    :param cfg: dict from flask.request.args['cfg'] of how to calculate the new column
    :return: JSON {success: True/False}
    """
    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied")
    request = request.dict()
    data_instance = ArcticDBInstance(dataset_id=dataset_id,tenant_id=current_user.tenant_id)
    data = data_instance.get_data()
    # dtypes =lib_data.metadata["dtypes"]
    col_type = request['type']
    cfg = request['cfg']
    save_as = get_str_arg(request, "saveAs", "new")
    if save_as == "inplace":
        name = cfg["col"]
    else:
        name = get_str_arg(request, "name")
        if not name and col_type != "type_conversion":
            raise Exception("'name' is required for new column!")
        # non-type conversions cannot be done inplace and thus need a name and the name needs to be checked that it
        # won't overwrite something else
        name = str(name)
        if name in data.columns:
            raise Exception("A column named '{}' already exists!".format(name))

    def _build_column():
        builder = ColumnBuilder(data, col_type, name, cfg)
        new_col_data = builder.build_column()
        print(new_col_data)
        new_cols = []
        if isinstance(new_col_data, pd.Series):
            data[name] = new_col_data
            new_cols.append(name)
        else:
            for i in range(len(new_col_data.columns)):
                new_col = new_col_data.iloc[:, i]
                if pandas_util.is_pandas2():
                    data[str(new_col.name)] = new_col
                else:
                    data.loc[:, str(new_col.name)] = new_col

        new_types = {}
        data_ranges = {}
        for new_col in new_cols:
            dtype = find_dtype(data[new_col])
            if classify_type(dtype) == "F" and not data[new_col].isnull().all():
                new_ranges = calc_data_ranges(data[[new_col]])
                data_ranges[new_col] = new_ranges.get(new_col, data_ranges.get(new_col))
            new_types[new_col] = dtype
        dtype_f = dtype_formatter(data, new_types, data_ranges)
        print(dtype_f)
        data_instance.update_data(data=data)
        #curr_dtypes = data_instance.dtypes
        # if next((cdt for cdt in curr_dtypes if cdt["name"] in new_cols), None):
        #     curr_dtypes = [
        #         (
        #             dtype_f(len(curr_dtypes), cdt["name"])
        #             if cdt["name"] in new_cols
        #             else cdt
        #         )
        #         for cdt in curr_dtypes
        #     ]
        # else:
        #     curr_dtypes += [dtype_f(len(curr_dtypes), new_col) for new_col in new_cols]
        # global_state.set_dtypes(data_id, curr_dtypes)
        curr_history = data_instance.history
        curr_history += make_list(builder.build_code())
        data_instance.history = curr_history

    if cfg.get("applyAllType", False):
        cols = [
            dtype["name"]
            for dtype in curr_dtypes
            if dtype["dtype"] == cfg["from"]
        ]
        for col in cols:
            cfg = dict_merge(cfg, dict(col=col))
            name = col
            _build_column()
    else:
        _build_column()
    return {"success":True}


@router.post("/column-analysis/{dataset_id}")
def get_column_analysis(request:ColumnAnalysisSchema,
                  dataset_id: int,
                  db: Session = Depends(get_db),
                  current_user: models.User = Depends(get_current_user)):
    """
    :class:`flask:flask.Flask` route which returns output from numpy.histogram/pd.value_counts to front-end as JSON

    :param data_id: integer string identifier for a D-Tale process's data
    :type data_id: str
    :param col: string from flask.request.args['col'] containing name of a column in your dataframe
    :param type: string from flask.request.args['type'] to signify either a histogram or value counts
    :param query: string from flask.request.args['query'] which is applied to DATA using the query() function
    :param bins: the number of bins to display in your histogram, options on the front-end are 5, 10, 20, 50
    :param top: the number of top values to display in your value counts, default is 100
    :returns: JSON {results: DATA, desc: output from pd.DataFrame[col].describe(), success: True/False}
    """
    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")
    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied")
    request = request.dict()
    # data_instance = ArcticDBInstance(dataset_id=dataset_id,tenant_id=current_user.tenant_id)
    # data = data_instance.get_data()
    analysis = ColumnAnalysis(data_id=dataset_id,tenant_id=current_user.tenant_id, req=request)
    return analysis.build()
