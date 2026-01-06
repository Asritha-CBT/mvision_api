from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from mvision.db.database import get_db
from sqlalchemy.sql import func
from mvision.db import models
from mvision.schemas import category_schema, camera_schema

router = APIRouter()

# ---------------- GET ALL categories ----------------
@router.get("/categories", response_model=list[camera_schema.CameraCategoryResponse])
def get_all_categories(db: Session = Depends(get_db)):
    return db.query(models.CameraCategory).all()

# ---------------- CREATE CATEGORY ----------------
@router.post("/create", response_model=category_schema.CameraCategoryResponse)
def category_register(payload: category_schema.CameraCategoryCreate, db: Session = Depends(get_db)):
    exists = (
        db.query(models.CameraCategory)
        .filter(func.lower(models.CameraCategory.name) == payload.name.lower())
        .first()
    )
    if exists:
        raise HTTPException(status_code=400, detail="Category already exists")

    new_cat = models.CameraCategory(
        name=payload.name,
        description=payload.description.strip() if payload.description else None,
    )
    db.add(new_cat)
    db.commit()
    db.refresh(new_cat)
    return new_cat

# ---------------- UPDATE CATEGORY ----------------
@router.put("/update/{id}", response_model=category_schema.CameraCategoryResponse)
def update_category(id: int, payload: category_schema.CameraCategoryUpdate, db: Session = Depends(get_db)):
    cat = db.query(models.CameraCategory).filter(models.CameraCategory.id == id).first()
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found")

    # unique name check (ignore same row)
    exists = (
        db.query(models.CameraCategory)
        .filter(func.lower(models.CameraCategory.name) == payload.name.lower())
        .filter(models.CameraCategory.id != id)
        .first()
    )
    if exists:
        raise HTTPException(status_code=400, detail="Category already exists")

    cat.name = payload.name
    cat.description = payload.description.strip() if payload.description else None

    db.commit()
    db.refresh(cat)
    return cat

# ---------------- DELETE CATEGORY ----------------
@router.delete("/delete/{id}")
def delete_category(id: int, db: Session = Depends(get_db)):
    category = db.query(models.CameraCategory).filter(
        models.CameraCategory.id == id
    ).first()

    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    # Check Camera usage
    camera_in_use = db.query(models.Camera).filter(
        models.Camera.category_id == id
    ).first()

    if camera_in_use:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete, Category is in use by cameras."
        )

    # Check User usage
    user_in_use = db.query(models.User).filter(
        models.User.category_id == id
    ).first()

    if user_in_use:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete, Category is in use by users."
        )

    db.delete(category)
    db.commit()
    return {"message": "Category deleted successfully"}
