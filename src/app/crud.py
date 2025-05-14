from sqlalchemy.orm import Session
from app import models, schemas
from app.hashing import hash_password


# User CRUD operations
def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(
        email=user.email,
        name=user.name,
        company_name=user.company_name,
        company_department=user.company_department,
        company_size=user.company_size,
        company_industries=user.company_industries,
        hashed_password=hash_password(user.password),
        tenant_id=user.tenant_id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

# Tenant CRUD operations
def create_tenant(db: Session, tenant: schemas.TenantCreate):
    db_tenant = models.Tenant(name=tenant.name)
    db.add(db_tenant)
    db.commit()
    db.refresh(db_tenant)
    return db_tenant


def get_tenant(db: Session, tenant_id: int):
    return db.query(models.Tenant).filter(models.Tenant.id == tenant_id).first()


# Dataset CRUD operations
def create_dataset(db: Session, dataset: schemas.DatasetCreate):
    db_dataset = models.Dataset(
        name=dataset.name,
        description=dataset.description,
        tenant_id=dataset.tenant_id
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset


def get_dataset(db: Session, dataset_id: int):
    return db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()

def get_dataset_by_name(db: Session, dataset_name: str,tenant_id:int):
    return db.query(models.Dataset).filter(models.Dataset.name == dataset_name and models.Dataset.tenant_id == tenant_id).first()

def get_datasets_by_tenant_id(db: Session, tenant_id: int):
    """
    Retrieve all datasets that belong to a tenant.

    :param db: Database session
    :param tenant_id: ID of the tenant
    :return: List of datasets
    """
    return db.query(models.Dataset).filter(models.Dataset.tenant_id == tenant_id).all()


def update_dataset(db: Session, dataset_id: int, dataset_update: schemas.DatasetUpdate):
    db_dataset = get_dataset(db, dataset_id)
    if db_dataset:
        for key, value in dataset_update.dict(exclude_unset=True).items():
            setattr(db_dataset, key, value)
        db.commit()
        db.refresh(db_dataset)
    return db_dataset


def delete_dataset(db: Session, dataset_id: int):
    db_dataset = get_dataset(db, dataset_id)
    if db_dataset:
        db.delete(db_dataset)
        db.commit()
    return db_dataset


# Snapshot CRUD operations
def create_snapshot(db: Session, snapshot: schemas.SnapshotCreate):
    db_snapshot = models.Snapshot(
        description=snapshot.description,
        dataset_id=snapshot.dataset_id,
        user_id=snapshot.user_id
    )
    db.add(db_snapshot)
    db.commit()
    db.refresh(db_snapshot)
    return db_snapshot

def get_snapshots_by_dataset_id(db: Session, dataset_id: int):
    return db.query(models.Snapshot).filter(models.Snapshot.dataset_id == dataset_id).all()

def get_all_snapshots(db: Session):
    return db.query(models.Snapshot).all()


# Dataset CRUD operations
def create_model(db: Session, model: schemas.ModelCreate):
    db_models = models.Models(
        name=model.name,
        dataset_id=model.dataset_id,
        tenant_id=model.tenant_id,
        f1_score=model.f1_score,
        cv_score= model.cv_score,
        confusion_matrix=model.confusion_matrix
    )
    db.add(db_models)
    db.commit()
    db.refresh(db_models)
    return db_models

def get_model(db: Session, model_id: int):
    return db.query(models.Models).filter(models.Models.id == model_id).first()

def delete_model(db: Session, model_id: int):
    db_models = get_model(db, models_id)
    if db_models:
        db.delete(db_models)
        db.commit()
    return db_models

def get_models_by_tenant_id(db: Session, tenant_id: int):
    """
    Retrieve all datasets that belong to a tenant.

    :param db: Database session
    :param tenant_id: ID of the tenant
    :return: List of datasets
    """
    return db.query(models.Models).filter(models.Models.tenant_id == tenant_id).all()

def get_models_by_dataset_id(db: Session, dataset_id: int):
    """
    Retrieve all datasets that belong to a tenant.

    :param db: Database session
    :param tenant_id: ID of the tenant
    :return: List of datasets
    """
    return db.query(models.Models).filter(models.Models.dataset_id == dataset_id).all()

def update_model(db: Session, model_id: int, model_update: schemas.ModelUpdate):
    db_models = get_dataset(db, model_id)
    if db_models:
        for key, value in model_update.dict(exclude_unset=True).items():
            setattr(db_models, key, value)
        db.commit()
        db.refresh(db_models)
    return db_models


def create_chat_stream(db: Session, chat_stream: schemas.ChatStreamSchemaCreate):
    db_models = models.ChatStream(
        id = chat_stream.id,
        user_id =  chat_stream.user_id,
        status = "Active"
    )
    db.add(db_models)
    db.commit()
    db.refresh(db_models)
    return db_models

def get_chat_stream_by_user_id(db: Session, user_id: int):
    """
    Retrieve all datasets that belong to a tenant.

    :param db: Database session
    :param tenant_id: ID of the tenant
    :return: List of datasets
    """
    return db.query(models.ChatStream).filter(models.ChatStream.user_id == user_id).all()


def create_chat_message(db: Session, chat_message: schemas.ChatMessageSchemaCreate):
    db_models = models.ChatMessages(
        user_id =  chat_message.user_id,
        stream_id = chat_message.stream_id,
        message = chat_message.message,
        stage=chat_message.message,
        type=chat_message.message,
        current = chat_message.current
    )
    db.add(db_models)
    db.commit()
    db.refresh(db_models)
    return db_models


def get_chat_messages_by_stream_id(db: Session, stream_id: str):
    """
    Retrieve all datasets that belong to a tenant.

    :param db: Database session
    :param tenant_id: ID of the tenant
    :return: List of datasets
    """
    return db.query(models.ChatStream).filter(models.ChatMessages.stream_id== stream_id).all()