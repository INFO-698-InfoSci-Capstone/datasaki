from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from app import schemas, crud, models
from app.database import get_db
from app.dependencies import get_current_user
from app.arcticdb_utils import ArcticDBInstance
import pandas as pd
from typing import Any, Dict, List, Optional
from io import StringIO
from app.utils import get_column_stats, NpEncoder
import json
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.tools as tls
import plotly.io as pio

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

router = APIRouter()
def pieChartBuilder(data_instance:ArcticDBInstance,column:str):
    """Read the uploaded file into a DataFrame."""
    data = data_instance.get_data()
    values = round(((data[column].value_counts() / data[column].count()) * 100))
    return {"data":[{"values":values.tolist(),"labels":values.keys().tolist(),"type": "pie"}]}

def barChartBuilder(data_instance:ArcticDBInstance,column:str):
    """Read the uploaded file into a DataFrame."""
    data = data_instance.get_data()
    sns.countplot(data=data,x=column)
    plotly_fig = tls.mpl_to_plotly(plt.gcf())
    categories = list(data[column].unique())
    plotly_fig.update_layout(
        xaxis=dict(
            tickvals=list(range(len(categories))),  # Use sequential numbers
            ticktext=categories  # Map sequential numbers to original category names
        )
    )
    json_plot = json.loads(pio.to_json(plotly_fig))
    return json_plot

def histChartBuilder(data_instance:ArcticDBInstance,column:str):
    """Read the uploaded file into a DataFrame."""
    data = data_instance.get_data()
    sns.histplot(data=data, x=column, kde=True)
    plotly_fig = tls.mpl_to_plotly(plt.gcf())
    json_plot = pio.to_json(plotly_fig)
    return json.loads(json_plot)

def boxPlotChartBuilder(data_instance:ArcticDBInstance,column:str):
    """Read the uploaded file into a DataFrame."""
    data = data_instance.get_data()
    sns.boxplot(data=data, x=column, y="default")
    plotly_fig = tls.mpl_to_plotly(plt.gcf())
    json_plot = pio.to_json(plotly_fig)
    return json.loads(json_plot)

@router.post("/chart/{dataset_id}/{chart_type}")
async def chart(
        dataset_id: int,
        chart_type:str,
        request: dict,
        db: Session = Depends(get_db),
        current_user: models.User = Depends(get_current_user)):

    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")

    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied")
    data_instance = ArcticDBInstance(dataset_id=dataset_id,tenant_id=current_user.tenant_id)
    if chart_type == 'hist':
        return histChartBuilder(data_instance,request.get('col'))
    if chart_type == 'count':
        return barChartBuilder(data_instance,request.get('col'))
    if chart_type == 'pie':
        return pieChartBuilder(data_instance,request.get('col'))

