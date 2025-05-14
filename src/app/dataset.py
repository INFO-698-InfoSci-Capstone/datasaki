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

router = APIRouter()
async def read_file_to_dataframe(file: UploadFile) -> pd.DataFrame:
    """Read the uploaded file into a DataFrame."""
    content_type = file.content_type
    if content_type == 'text/csv':
        return pd.read_csv(file.file)
    elif content_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/xlsx']:
        return pd.read_excel(file.file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

@router.post("/datasets/")
async def create_dataset(
    name: str = Form(...),
    description: str = Form(None),
    tenant_id: int = Form(...),
    file: UploadFile = File(None),
    current_user: models.User = Depends(get_current_user)
):
    if tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
    # Convert the uploaded file to a pandas DataFrame if provided
    data = None
    if file:
        file_content = StringIO(file.file.read().decode('utf-8'))
        data = pd.read_csv(file_content)

    # ArcticDB storage
    data_instance = ArcticDBInstance(data=data,tenant_id=current_user.tenant_id,dataset_name=name,description=description)
    return {"dataset_name":data_instance.dataset_name,"id":data_instance.dataset_id}

@router.put("/datasets/{dataset_id}", response_model=schemas.Dataset)
async def update_dataset(
    dataset_id: int,
    name: str = Form(...),
    description: str = Form(None),
    file: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied")

    # Convert the uploaded file to a pandas DataFrame if provided
    data = None
    if file:
        file_content = StringIO(file.file.read().decode('utf-8'))
        data = pd.read_csv(file_content)
    if data:
        data_instance = ArcticDBInstance(tenant_id=current_user.tenant_id, dataset_id=dataset_id)
        data_instance.update_data(data= data,metadata = {"history":[],"settings":{},"context_variables":{}})
    # Update dataset in DB
    dataset_update = schemas.DatasetUpdate(name=name, description=description)
    db_dataset = crud.update_dataset(db, dataset_id, dataset_update)
    return db_dataset

@router.get("/datasets/{dataset_id}")
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied")
    data_instance = ArcticDBInstance(tenant_id=current_user.tenant_id,dataset_id=dataset_id)
    return json.loads(data_instance.get_data().to_json(orient='records'))

@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    try:
        db_dataset = crud.get_dataset(db, dataset_id)
        if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied")
        data_instance = ArcticDBInstance(tenant_id=current_user.tenant_id,dataset_id=dataset_id)
        data_instance.delete_dataset()
        return {"Deleted":True}
    except Exception as err:
        return {"error":err}


@router.get("/datasets/", response_model=list[schemas.Dataset])
async def get_all_datasets_for_user(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get all datasets for the current user's tenant.
    """
    try:
        datasets = crud.get_datasets_by_tenant_id(db, tenant_id=current_user.tenant_id)
        return datasets
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dataset_count")
async def get_all_datasets_count_for_user(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get all datasets for the current user's tenant.
    """
    try:
        datasets = crud.get_datasets_by_tenant_id(db, tenant_id=current_user.tenant_id)
        return {"count":len(datasets)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/target/{dataset_id}")
def update_target( target:str,
                  dataset_id: int,
                  db: Session = Depends(get_db),
                  current_user: models.User = Depends(get_current_user)):
    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")
    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied")
    data_instance = ArcticDBInstance(dataset_id=dataset_id,tenant_id=current_user.tenant_id)
    data_instance.update_target(target)
    return target

@router.get("/target/{dataset_id}")
async def get_target(
        dataset_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Get all datasets for the current user's tenant.
    """
    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Dataset not found or access denied")
    db_dataset = crud.get_dataset(db, dataset_id)
    if db_dataset is None or db_dataset.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found or access denied")
    data_instance = ArcticDBInstance(dataset_id=dataset_id,tenant_id=current_user.tenant_id)
    return data_instance.target