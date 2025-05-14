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
from app.schemas import RelationshipSchema
import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr

router = APIRouter()

def FunctionAnova(inpData, TargetVariable, ContinuousPredictorList):
    res = {TargetVariable:{"Anova":{}}}
    for predictor in ContinuousPredictorList:
        CategoryGroupLists = inpData.groupby(TargetVariable)[predictor].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)
        res[TargetVariable]["Anova"][predictor] = AnovaResults
    return res

def FunctionChisq(inpData, TargetVariable, CategoricalVariablesList):
    res = {TargetVariable:{"Chisq": {}}}
    for predictor in CategoricalVariablesList:
        CrossTabResult = pd.crosstab(index=inpData[TargetVariable], columns=inpData[predictor])
        ChiSqResult = chi2_contingency(CrossTabResult)
        res[TargetVariable]["Chisq"][predictor] ={"statistic":ChiSqResult.statistic,
                                                            "pvalue":ChiSqResult.pvalue,
                                                            "dof":ChiSqResult.dof,
                                                            "expected_freq":ChiSqResult.expected_freq.tolist()}
    return res

def FunctionPearson(inpData, TargetVariable, ContinuousPredictorList):
    res = {TargetVariable:{"Pearson":{}}}
    for predictor in ContinuousPredictorList:
        CategoryGroupLists = inpData.groupby(TargetVariable)[predictor].apply(list)
        PearsonResults = pearsonr(*CategoryGroupLists)
        res[TargetVariable]["Pearson"][predictor] = PearsonResults
    return res

@router.post("/relationship/{dataset_id}/{type}")
def set_relationship( request:RelationshipSchema,
                  dataset_id: int,
                  type:str,
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
    if type == "Anova":
        res = FunctionAnova(data,request["target"],request["cols"])
    elif type == "Chisq":
        res = FunctionChisq(data, request["target"], request["cols"])
    data_instance.update_relationship(request["target"], res, type)
    return res


@router.get("/relationship/{dataset_id}")
def get_relationship( target:str,
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
    data_instance = ArcticDBInstance(dataset_id=dataset_id,tenant_id=current_user.tenant_id)
    return data_instance.relationship[target]


def set_relationship_based_on_column(data_instance,target=None):
    data = data_instance.get_data()
    target = target if target else data_instance._target
    dtypes = json.loads(data_instance._dtypes)
    anova_columns = []
    chisq_columns = []
    pearson_columns = []
    res1 = {}
    target_type = [k["is_categorical"] for k in dtypes if k["name"]==target]
    for k in dtypes:
        if k["name"] == target:
            continue
        if k["is_categorical"] and target_type:
            chisq_columns.append(k["name"])
        elif  k["is_categorical"] and not target_type:
            anova_columns.append(k["name"])
        elif  not k["is_categorical"] and target_type:
            anova_columns.append(k["name"])
        elif not k["is_categorical"] and  not target_type:
            pearson_columns.append(k["name"])
    res1 = FunctionAnova(data,target,anova_columns)
    res_feature = [k for k,v in res1[target]["Anova"].items() if v[1] < 0.05]
    res2 = FunctionChisq(data,target,chisq_columns)
    res_feature.extend([k for k, v in res2[target]["Chisq"].items()  if v["pvalue"] < 0.05])
    res3 = FunctionPearson(data, target,pearson_columns)
    res_feature.extend([k for k, v in res3[target]["Pearson"].items()  if v[1] < 0.05])
    res1[target].update(res2[target])
    res1[target].update(res3[target])

    return res1,res_feature
