from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from mvision.db.database import get_db
from sqlalchemy.sql import func
from mvision.db import models
from mvision.schemas import area_definition_schema, camera_schema

router = APIRouter()

# ---------------- GET ALL area_definitions ----------------
@router.get("/area_definitions", response_model=list[camera_schema.AreaDefinitionResponse])
def get_all_area_definitions(db: Session = Depends(get_db)):
    return db.query(models.AreaDefinition).all()

# ---------------- CREATE CATEGORY ----------------
@router.post("/create", response_model=area_definition_schema.AreaDefinitionResponse)
def area_definition_register(payload: area_definition_schema.AreaDefinitionCreate, db: Session = Depends(get_db)):
    exists = (
        db.query(models.AreaDefinition)
        .filter(func.lower(models.AreaDefinition.name) == payload.name.lower())
        .first()
    )
    if exists:
        raise HTTPException(status_code=400, detail="AreaDefinition already exists")

    new_cat = models.AreaDefinition(
        name=payload.name,
        description=payload.description.strip() if payload.description else None,
    )
    db.add(new_cat)
    db.commit()
    db.refresh(new_cat)
    return new_cat

# ---------------- UPDATE CATEGORY ----------------
@router.put("/update/{id}", response_model=area_definition_schema.AreaDefinitionResponse)
def update_area_definition(id: int, payload: area_definition_schema.AreaDefinitionUpdate, db: Session = Depends(get_db)):
    cat = db.query(models.AreaDefinition).filter(models.AreaDefinition.id == id).first()
    if not cat:
        raise HTTPException(status_code=404, detail="AreaDefinition not found")

    # unique name check (ignore same row)
    exists = (
        db.query(models.AreaDefinition)
        .filter(func.lower(models.AreaDefinition.name) == payload.name.lower())
        .filter(models.AreaDefinition.id != id)
        .first()
    )
    if exists:
        raise HTTPException(status_code=400, detail="AreaDefinition already exists")

    cat.name = payload.name
    cat.description = payload.description.strip() if payload.description else None

    db.commit()
    db.refresh(cat)
    return cat

# ---------------- DELETE CATEGORY ----------------
@router.delete("/delete/{id}")
def delete_area_definition(id: int, db: Session = Depends(get_db)):
    area_definition = db.query(models.AreaDefinition).filter(
        models.AreaDefinition.id == id
    ).first()

    if not area_definition:
        raise HTTPException(status_code=404, detail="AreaDefinition not found")

    if not area_definition:
        raise HTTPException(status_code=404, detail="AreaDefinition not found")

    # Check Camera usage
    camera_in_use = db.query(models.Camera).filter(
        models.Camera.area_definition_id == id
    ).first()

    if camera_in_use:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete, AreaDefinition is in use by cameras."
        )

    # Check User usage
    user_in_use = db.query(models.User).filter(
        models.User.area_definition_id == id
    ).first()

    if user_in_use:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete, AreaDefinition is in use by users."
        )

    db.delete(area_definition)
    db.commit()
    return {"message": "AreaDefinition deleted successfully"}
