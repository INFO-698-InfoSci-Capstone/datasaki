from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import pandas as pd
from app import arcticdb_utils, login
from sqlalchemy.orm import Session

router = APIRouter()

@router.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(login.get_db)):
    df = pd.read_csv(file.file)
    tenant_id = login.get_tenant_id_from_db(db)  # Replace with actual tenant id extraction logic
    # arcticdb_utils.save_dataframe(df, tenant_id)
    return {"filename": file.filename}
