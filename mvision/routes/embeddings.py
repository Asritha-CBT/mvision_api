# app/routes/embeddings.py
from fastapi import APIRouter
from pydantic import BaseModel 
from mvision.services import extract_service

router = APIRouter(prefix="/api/extraction", tags=["extraction"])


class StartRequest(BaseModel):
    id: int


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


@router.get("/status")
def status():
    """Return extract_service status."""
    return extract_service.get_status()
