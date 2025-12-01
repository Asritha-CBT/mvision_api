from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from mvision.db.database import get_db 
from mvision.schemas import user_schema
from mvision.db import models

router = APIRouter()
 
@router.post("/user_register", response_model=user_schema.UserResponse)
def user_register(user: user_schema.UserRegister, db: Session = Depends(get_db)):
    # Additional validation for empty strings
    if not user.name.strip() or not user.department.strip():
        raise HTTPException(status_code=400, detail="Name and Department are required!")

    new_user = models.User(
        name=user.name,
        department=user.department
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user
