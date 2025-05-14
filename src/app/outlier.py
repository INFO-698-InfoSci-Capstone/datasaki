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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Sample data
np.random.seed(42)
data = np.random.randn(100)  # Normally distributed data
data = np.append(data, [15, 16, 17])  # Adding some outliers
df = pd.DataFrame(data, columns=['value'])

# 1. Z-score method
def detect_outliers_z_score(data, threshold=3):
    z_scores = stats.zscore(data)
    abs_z_scores = np.abs(z_scores)
    outliers = abs_z_scores > threshold
    return np.where(outliers)

def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return np.where(outliers)

def visualize_boxplot(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data)
    plt.title('Boxplot')
    plt.show()

def visualize_scatter(data):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data)), data)
    plt.title('Scatter Plot')
    plt.show()

def detect_outliers_isolation_forest(data):
    iso_forest = IsolationForest(contamination=0.1)
    y_pred = iso_forest.fit_predict(data.reshape(-1, 1))
    outliers = np.where(y_pred == -1)
    return outliers


def detect_outliers_lof(data):
    lof = LocalOutlierFactor(n_neighbors=20)
    y_pred = lof.fit_predict(data.reshape(-1, 1))
    outliers = np.where(y_pred == -1)
    return outliers

def get_outlier_based_on_column(data_instance,target=None):
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
