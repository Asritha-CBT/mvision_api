from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from mvision.db.database import get_db
from mvision.schemas import user_schema
from mvision.db import models
from datetime import datetime

router = APIRouter()


# ---------------- GET ALL USERS ----------------
@router.get("/users", response_model=list[user_schema.UserResponse])
def get_all_users(db: Session = Depends(get_db)):
    users = db.query(models.User).all()
    return users


# ---------------- CREATE USER ----------------
@router.post("/user_register", response_model=user_schema.UserResponse)
def user_register(user: user_schema.UserRegister, db: Session = Depends(get_db)):
    new_user = models.User(
        name=user.name.strip(),
        department=user.department.strip()
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

    db_user.name = user.name
    db_user.department = user.department

    db.commit()
    db.refresh(db_user)
    return db_user


# ---------------- DELETE USER ----------------
@router.delete("/delete/{id}")
def delete_user(id: int, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.id == id).first()

    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(db_user)
    db.commit()

    return {"message": "User deleted successfully"}


# ============================================================
# ✅ FIXED: UPDATE USER EMBEDDING PATH
# ============================================================
@router.put("/update_embedding/{id}")
def update_embedding(
    id: int,
    data: user_schema.EmbeddingUpdate,
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(models.User.id == id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    embeddings_path = data.embeddings_path

    # -------------------------------
    # ⭐ FIXED LOGIC
    # -------------------------------
    if embeddings_path and embeddings_path.strip():
        # Valid path → update both
        user.embeddings_path = embeddings_path.strip()
        user.last_embedding_update_ts = datetime.utcnow()
    else:
        # Empty or null → set both to NULL
        user.embeddings_path = None
        user.last_embedding_update_ts = None

    db.commit()
    db.refresh(user)

    return {
        "message": "Embedding path updated successfully",
        "user": user
    }
