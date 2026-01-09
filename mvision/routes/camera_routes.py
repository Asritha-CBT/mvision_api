from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql import func
from mvision.db.database import get_db
from mvision.db import models
from mvision.schemas import camera_schema

router = APIRouter()

# ---------------- GET ALL CAMERAS ----------------
@router.get("/cameras", response_model=list[camera_schema.CameraResponse])
def get_all_cameras(db: Session = Depends(get_db)):
    cams = db.query(models.Camera).options(joinedload(models.Camera.area_definition)).order_by(func.lower(models.Camera.camera_name).asc()).all()
    return cams
# ---------------- GET ALL CAMERAS ----------------
@router.get("/camera_list", response_model=list[camera_schema.CameraListResponse])
def get_all_cameras(db: Session = Depends(get_db)):
    cams = db.query(models.Camera).order_by(func.lower(models.Camera.camera_name).asc()).all()
    return cams

# ---------------- CREATE CAMERA ----------------
@router.post("/create", response_model=camera_schema.CameraResponse)
def camera_register(cam: camera_schema.CameraCreate, db: Session = Depends(get_db)):
    # check unique camera_name
    exists = db.query(models.Camera).filter(models.Camera.camera_name == cam.camera_name).first()
    if exists:
        raise HTTPException(status_code=400, detail="Camera number already exists")

    # ensure area_definition exists
    cat = db.query(models.AreaDefinition).filter(models.AreaDefinition.id == cam.area_definition_id).first()
    if not cat:
        raise HTTPException(status_code=400, detail="Invalid area_definition_id")

    new_cam = models.Camera(
        camera_name=cam.camera_name.strip(), 
        area_definition_id=cam.area_definition_id
    )
    db.add(new_cam)
    db.commit()
    db.refresh(new_cam)
    return new_cam

# ---------------- UPDATE CAMERA ----------------
@router.put("/update/{id}", response_model=camera_schema.CameraResponse)
def update_camera(id: int, cam: camera_schema.CameraCreate, db: Session = Depends(get_db)):
    db_cam = db.query(models.Camera).filter(models.Camera.id == id).first()
    if not db_cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    # if camera_name is changed, ensure uniqueness
    if cam.camera_name.strip() != db_cam.camera_name:
        exists = db.query(models.Camera).filter(models.Camera.camera_name == cam.camera_name.strip()).first()
        if exists:
            raise HTTPException(status_code=400, detail="Camera name already exists")

    # ensure area_definition exists
    cat = db.query(models.AreaDefinition).filter(models.AreaDefinition.id == cam.area_definition_id).first()
    if not cat:
        raise HTTPException(status_code=400, detail="Invalid area_definition_id")

    db_cam.camera_name = cam.camera_name.strip() 
    db_cam.area_definition_id = cam.area_definition_id

    db.commit()
    db.refresh(db_cam)
    return db_cam

# ---------------- DELETE CAMERA ----------------
@router.delete("/delete/{id}")
def delete_camera(id: int, db: Session = Depends(get_db)):
    # Fetch camera
    db_cam = db.query(models.Camera).filter(
        models.Camera.id == id
    ).first()

    if not db_cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Check if camera's area_definition is used by any user
    area_definition_in_use_by_user = db.query(models.User).filter(
        models.User.area_definition_id == db_cam.area_definition_id
    ).first()

    if area_definition_in_use_by_user:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete, Camera is in use by users."
        )

    # Safe delete
    db.delete(db_cam)
    db.commit()

    return {"message": "Camera deleted successfully"}
