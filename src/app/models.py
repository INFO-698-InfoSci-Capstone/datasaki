from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, Boolean
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime


class Tenant(Base):
    __tablename__ = 'tenants'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    users = relationship("User", back_populates="tenant")
    datasets = relationship("Dataset", back_populates="tenant")
    models = relationship("Models", back_populates="tenant")

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    company_name = Column(String)
    company_department = Column(String)
    company_size = Column(String)
    company_industries = Column(String)
    hashed_password = Column(String, nullable=False)
    tenant_id = Column(Integer, ForeignKey('tenants.id'), nullable=False)
    tenant = relationship("Tenant", back_populates="users")
    chatmessages = relationship("ChatMessages", back_populates="users")
    chatstreams = relationship("ChatStream", back_populates="users")

class Dataset(Base):
    __tablename__ = 'datasets'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    tenant_id = Column(Integer, ForeignKey('tenants.id'), nullable=False)
    tenant = relationship("Tenant", back_populates="datasets")
    snapshots = relationship("Snapshot", back_populates="datasets")
    models = relationship("Models", back_populates="datasets")


class Models(Base):
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    tenant_id = Column(Integer, ForeignKey('tenants.id'), nullable=False)
    tenant = relationship("Tenant", back_populates="models")
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    datasets = relationship("Dataset", back_populates="models")
    # models = relationship("Models", back_populates="models")
    f1_score = Column(String, nullable=True)
    cv_score = Column(String, nullable=True)
    confusion_matrix = Column(String, nullable=True)

class Snapshot(Base):
    __tablename__ = 'snapshots'

    id = Column(Integer, primary_key=True, index=True)
    snapshot_timestamp = Column(DateTime, default=datetime.utcnow)
    description = Column(Text)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    datasets = relationship("Dataset", back_populates="snapshots")
    user = relationship("User")

class ChatStream(Base):
    __tablename__ = 'chatstreams'
    id = Column(Text, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    status = Column(Text)
    chatmessages = relationship("ChatMessages", back_populates="chatstreams")
    users = relationship("User", back_populates="chatstreams")

class ChatMessages(Base):
    __tablename__ = 'chatmessages'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    stream_id =  Column(Text, ForeignKey('chatstreams.id'), nullable=False)
    message = Column(Text)
    stage = Column(Text)
    type = Column(Text)
    received_at = Column(DateTime, default=datetime.utcnow)
    current = Column(Boolean)
    chatstreams = relationship("ChatStream", back_populates="chatmessages")
    users = relationship("User", back_populates="chatmessages")
