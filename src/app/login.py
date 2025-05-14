from fastapi import APIRouter, HTTPException, Depends, status, Form
from sqlalchemy.orm import Session
from app import crud, schemas, email_utils
from app.database import SessionLocal
from app.hashing import verify_password, hash_password
from app.models import User  # Import your User model or any other relevant model
from app.schemas import Token, LoginRequest
from datetime import timedelta
from app.auth import TokenData, create_access_token
from app.dependencies import get_current_user
router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/register", response_model=schemas.User)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.create_user(db=db, user=user)
    return db_user
#
# @router.post("/login")
# def login(email: str, password: str, db: Session = Depends(get_db)):
#     user = crud.get_user_by_email(db, email)
#     if not user or not verify_password(password, user.hashed_password):
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
#     return {"access_token": "fake_token", "token_type": "bearer"}


def get_tenant_id_from_db(db: Session, user_id: str) -> str:
    """Retrieve the tenant ID for a user from the database."""
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        return user.tenant_id
    else:
        raise ValueError("User not found")


@router.post("/token1", response_model=Token)
async def login2(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.username).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    access_token_expires = timedelta(hours=1)
    access_token = create_access_token(
        data={"sub": user.email,"tenant_id": user.tenant_id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# @router.get("/some-protected-route/")
# async def some_protected_route(db: Session = Depends(get_db), current_user: TokenData = Depends(get_current_user)):
#     user_id = current_user.username  # Use the username or user ID as needed
#     # Proceed with your logic using user_id
#     return {"user_id": user_id}


@router.post("/token", response_model=Token)
async def login3(
        grant_type: str = Form(...),
        username: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_db)
):
    if grant_type != "password":
        raise HTTPException(status_code=400, detail="Invalid grant_type")

    user = db.query(User).filter(User.email == username).first()
    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token_expires = timedelta(hours=1)
    access_token = create_access_token(
        data={"sub": user.email,"tenant_id": user.tenant_id}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}