# app/auth.py
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException
from app.config import settings
from pydantic import BaseModel

from passlib.context import CryptContext


SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str
    tenant_id: Optional[int] = None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        tenant_id: int = payload.get("tenant_id")
        if username is None or tenant_id is None:
            raise HTTPException(status_code=403, detail="Invalid token")
        return TokenData(username=username,tenant_id=tenant_id)
    except jwt.PyJWTError:
        raise HTTPException(status_code=403, detail="Invalid token")

# Set up password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify that a plain password matches the hashed password."""
    return pwd_context.verify(plain_password, hashed_password)