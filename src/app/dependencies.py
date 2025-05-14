# app/dependencies.py
from fastapi import Depends
from app.auth import verify_token, TokenData
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from app.config import settings

SECRET_KEY = settings.SECRET_KEY
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
ALGORITHM = "HS256"
async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        tenant_id: int = payload.get("tenant_id",1)  # Extract tenant_id
        if username is None or tenant_id is None:
            raise credentials_exception
        token_data = TokenData(username=username, tenant_id=tenant_id)
    except JWTError as err:
        raise credentials_exception
    return token_data
