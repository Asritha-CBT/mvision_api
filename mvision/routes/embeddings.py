# app/routes/embeddings.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel 
from mvision.services import extract_service 
from mvision.db.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from mvision.db.models import CameraCategory

router = APIRouter(prefix="/api/extraction", tags=["extraction"])


class StartRequest(BaseModel):
    id: int


@router.get("/categories")
def get_categories(db: Session = Depends(get_db)):
    return db.query(CameraCategory).all()

@router.post("/start")
def start(req: StartRequest):
    """
    Start background extraction for all RTSP streams.
    Payload: { "id": 123 }
    """
    return extract_service.start_extraction(req.id)


@router.post("/stop")
def stop():
    """Stop background extraction."""
    return extract_service.stop_extraction()

@router.delete("/remove/{id}")
def remove(id: int):
    """Remove embeddings."""
    return extract_service.remove_embeddings(id)


@router.get("/status")
def status():
    """Return extract_service status."""
    return extract_service.get_status()
