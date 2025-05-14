from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


# User Schemas
class UserCreate(BaseModel):
    email: EmailStr
    name: str
    company_name: str
    company_department: str
    company_size: str
    company_industries: str
    password: str
    tenant_id: int


class User(BaseModel):
    email: EmailStr
    name: str
    company_name: str
    company_department: str
    company_size: str
    company_industries: str
    tenant_id: int

    class Config:
        from_attributes = True


# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class LoginRequest(BaseModel):
    username: EmailStr  # Use EmailStr for email validation
    password: str


# Dataset Schemas
class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    tenant_id: int


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class Dataset(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    tenant_id: int

    class Config:
        from_attributes = True

# Dataset Schemas
class ModelCreate(BaseModel):
    name: str
    f1_score: Optional[str] = None
    cv_score: Optional[str] = None
    confusion_matrix: Optional[str] = None
    tenant_id: int
    dataset_id: int

class ModelUpdate(BaseModel):
    name: Optional[str] = None
    f1_score: Optional[str] = None
    cv_score: Optional[str] = None
    confusion_matrix: Optional[str] = None

class Model(BaseModel):
    id: int
    name: str
    f1_score: Optional[str] = None
    cv_score: Optional[str] = None
    confusion_matrix: Optional[str] = None
    tenant_id: int
    dataset_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# Snapshot Schemas
class SnapshotCreate(BaseModel):
    description: Optional[str] = None
    dataset_id: int
    user_id: int


class Snapshot(BaseModel):
    id: int
    snapshot_timestamp: datetime
    description: Optional[str] = None
    dataset_id: int
    user_id: int

    class Config:
        from_attributes = True


# Tenant Schemas
class TenantCreate(BaseModel):
    name: str

class Tenant(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True

class Inclusive(BaseModel):
    lower: bool
    upper: bool

class ConfigSchema(BaseModel):
    cols: Optional[List[str]] = None
    col: Optional[str] = None
    group: Optional[List[str]] = None
    left: Optional[str] = None
    right: Optional[str] = None
    operations: Optional[str] = None
    joinChar: Optional[str] = None
    search: Optional[str] = None
    replacement: Optional[str] = None
    caseSensitive: Optional[bool] = False
    regex: Optional[bool] = False
    timeDifference: Optional[str] = None
    timeDifferenceCol: Optional[str] = None
    property: Optional[str] = None
    conversion: Optional[str] = None
    bins: Optional[int] = None
    labels: Optional[str] = None
    type: Optional[str] = None
    length: Optional[int] = None
    chars: Optional[int] = None
    low: Optional[int] = None
    high: Optional[int] = None
    start: Optional[str] = None
    end: Optional[str] = None
    businessDay: Optional[bool] = False
    timestamps: Optional[bool] = False
    choices: Optional[List[str]] = None
    from_: Optional[str] = None  # 'from' is a reserved keyword in Python, so use from_
    to: Optional[str] = None
    agg: Optional[str] = None
    limits: Optional[int] = None
    inclusive: Optional[Inclusive] = None
    algo: Optional[str] = None
    normalized: Optional[bool] = False
    cleaners: Optional[List[str]] = None
    periods: Optional[int] = None
    comp: Optional[str] = None
    window: Optional[int] = None
    center: Optional[bool] = False
    min_periods: Optional[int] = None
    win_type: Optional[str] = None
    on: Optional[str] = None
    closed: Optional[str] = None
    alpha: Optional[int] = None
    fillValue: Optional[str] = None
    dtype: Optional[str] = None  # Assuming `dtype` is a string here, adjust if necessary
    delimiter: Optional[str] = None

class ColumnBuilderSchema(BaseModel):
    name: str
    type: str
    cfg: ConfigSchema
    saveAs: str

class ColumnAnalysisSchema(BaseModel):
    col: str
    type: str
    query: str
    bins: int
    top: int
    density:bool
    filtered:bool
    target:str
    ordinalCol:str
    ordinalAgg:str
    splits:str
    categoryCol:str
    categoryAgg:str
    cleaner:str

class RelationshipSchema(BaseModel):
    target: str
    cols: list[str]

class ChatStreamSchema(BaseModel):
    id: str
    user_id: int

class ChatStreamSchemaCreate(BaseModel):
    id: str
    user_id: int

class ChatMessageSchema(BaseModel):
    stream_id: str
    message: str
    stage: str
    type: str
    current: bool
    user_id: int

class ChatMessageSchemaCreate(BaseModel):
    user_id: int
    stream_id: str
    message: str
    stage: str
    type: str
    current: bool


