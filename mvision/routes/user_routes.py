from fastapi import APIRouter, Depends, HTTPException,status
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql import func
from sqlalchemy.exc import IntegrityError
from mvision.db.database import get_db
from mvision.schemas import user_schema
from mvision.db import models
from datetime import datetime


router = APIRouter() 

# ---------------- GET ALL DEPARTMENTS ---------------- 
@router.get("/departments", response_model=list[user_schema.DepartmentResponse])
def get_all_departments(db: Session = Depends(get_db)):
    departments = (
        db.query(models.Department)
        .order_by(func.lower(models.Department.name).asc())
        .all()
    )
    return departments


# ---------------- GET ALL USERS ----------------
@router.get("/users", response_model=list[user_schema.UserResponse])
def get_all_users(db: Session = Depends(get_db)):
    users = (
        db.query(models.User)
        .options(
            joinedload(models.User.department),
            joinedload(models.User.area_definition)
        )
        .order_by(func.lower(models.User.name).asc())
        .all()
    )
    # print("------This is user data-------",users)
    return users

# ---------------- CREATE USER ----------------
@router.post("/user_register", response_model=user_schema.UserResponse)
def user_register(user: user_schema.UserRegister, db: Session = Depends(get_db)):
    name = user.name.strip()
    gender = user.gender.strip()
    department_id = user.department_id

    exists = (
        db.query(models.User)
        .filter(
            models.User.name == name,
            models.User.department_id == department_id
        )
        .first()
    )

    if exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this name and department already exists"
        )

    new_user = models.User(
        name=name,
        gender=gender,
        department_id=department_id 
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user



# ---------------- UPDATE USER ----------------
@router.put("/update/{id}", response_model=user_schema.UserResponse)
def update_user(id: int, user: user_schema.UserRegister, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == id).first()

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    name = user.name.strip()
    gender = user.gender.strip()
    department = user.department_id

    # Check duplicate EXCLUDING current user
    exists = (
        db.query(models.User)
        .filter(
            models.User.name == name,
            models.User.gender == gender,
            models.User.department_id == department,
            models.User.id != id
        )
        .first()
    )

    if exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Another user with this name and department already exists"
        )

    db_user.name = name
    db_user.gender = gender
    db_user.department = department

    try:
        db.commit()
        db.refresh(db_user)
        return db_user

    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Duplicate user (name + department)"
        )
# ---------------- DELETE USER ----------------
@router.delete("/delete/{id}")
def delete_user(id: int, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == id).first()

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(db_user)
    db.commit()

    return {"message": "User deleted successfully"}

 