 
# app/routes/embeddings.py
from __future__ import annotations 
from fastapi import APIRouter, Depends, HTTPException 
from mvision.services import extract_service 
from mvision.db.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from mvision.db.models import AreaDefinition 
from pydantic import BaseModel, Field, conint
from typing import Optional, List, Dict, Any
 

router = APIRouter(prefix="/api/extraction", tags=["extraction"])

# ---------- Schemas ----------
class StartRequest(BaseModel):
    id: conint(gt=0) = Field(..., description="User ID to capture for")
    name: str = Field("unknown", description="User name for folder/DB labeling")
    area_definition_id: Optional[int] = Field(None, description="AreaDefinition.id to store with embeddings/user")

class StartResponse(BaseModel):
    status: str
    num_cams: int
    id: Optional[int]
    name: Optional[str]
    area_definition_id: Optional[int] = None

class ExtractStartResponse(BaseModel):
    status: str
    id: int
    name: str
    area_definition_id: Optional[int] = None

class ProgressResponse(BaseModel):
    id: int
    stage: str
    percent: int
    message: str
    total_body: int
    total_face: int
    done_body: int
    done_face: int

class StopResponse(BaseModel):
    status: str
    id: Optional[int] = None

class ExtractionStatus(BaseModel):
    running: bool
    num_cams: int
    rtsp_streams: List[str]
    id: Optional[int]

# ---------- Routes ----------
@router.post("/start", response_model=StartResponse)
def start(req: StartRequest) -> StartResponse:
    """Start background capture & embedding pipeline (YOLO + OSNet)."""
    if not hasattr(extract_service, "start_extraction"):
        raise HTTPException(status_code=501, detail="start_extraction is not implemented in extract_service")

    try:
        res = extract_service.start_extraction(req.id, req.name, area_definition_id=req.area_definition_id)  # type: ignore
    except TypeError as e:
        raise HTTPException(status_code=500, detail=f"start_extraction signature mismatch: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"start_extraction failed: {e}") from e

    if not isinstance(res, dict):
        raise HTTPException(status_code=500, detail="start_extraction returned invalid response")
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "Unknown error"))

    num_cams = int(res.get("num_cams", res.get("streams", len(getattr(extract_service, "RTSP_STREAMS", [])))))

    return StartResponse(
        status=str(res.get("status", "unknown")),
        num_cams=num_cams,
        id=req.id,
        name=req.name,
        area_definition_id=req.area_definition_id,
    )

@router.get("/area_definitions")
def get_area_definitions(db: Session = Depends(get_db)):
    return db.query(AreaDefinition).all()
 
@router.post("/stop", response_model=StopResponse)
def stop() -> StopResponse:
    """Stop background capture/embedding."""
    if not hasattr(extract_service, "stop_extraction"):
        raise HTTPException(status_code=501, detail="stop_extraction is not implemented in extract_service")

    try:
        res = extract_service.stop_extraction()  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"stop_extraction failed: {e}") from e

    if not isinstance(res, dict):
        raise HTTPException(status_code=500, detail="stop_extraction returned invalid response")
    if res.get("status") in {"not_running", "error"}:
        raise HTTPException(status_code=409, detail=res.get("message", "extraction is not running"))

    return StopResponse(status=str(res.get("status", "unknown")), id=res.get("id"))

@router.post("/extract", response_model=ExtractStartResponse)
def extract_embeddings(req: StartRequest) -> ExtractStartResponse:
    """Compute embeddings from saved folders only; write to DB."""
    if not hasattr(extract_service, "extract_embeddings_async"):
        raise HTTPException(status_code=501, detail="extract_embeddings_async is not implemented in extract_service")

    try: 
        res = extract_service.extract_embeddings_async(req.id, req.area_definition_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"extract_embeddings failed: {e}") from e

    if not isinstance(res, dict):
        raise HTTPException(status_code=500, detail="extract_embeddings returned invalid response")
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "Unknown error"))

    return ExtractStartResponse(
        status=str(res.get("status", "unknown")),
        id=int(res.get("id", req.id)),
        name=str(res.get("name", req.name)),
        area_definition_id=req.area_definition_id,
    )

@router.get("/progress/{id}", response_model=ProgressResponse)
def progress(id: int) -> ProgressResponse:
    """Get current extraction progress."""
    if hasattr(extract_service, "get_progress_for_user"):
        try:
            st = extract_service.get_progress_for_user(id)  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"get_progress failed: {e}") from e
        st = st or {}
    else:
        if not hasattr(extract_service, "get_status"):
            raise HTTPException(status_code=501, detail="get_status is not implemented in extract_service")
        try:
            status = extract_service.get_status()  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"get_status failed: {e}") from e
        st = (status or {}).get("progress", {}) or {}

    return ProgressResponse(
        id=id,
        stage=str(st.get("stage", "unknown")),
        percent=int(st.get("percent", 0)),
        message=str(st.get("message", "")),
        total_body=int(st.get("total_body", 0)),
        total_face=int(st.get("total_face", 0)),
        done_body=int(st.get("done_body", 0)),
        done_face=int(st.get("done_face", 0)),
    )

@router.delete("/remove/{id}", response_model=Dict[str, Any])
def remove(id: int):
    """Remove stored embeddings (body+face) for a user."""
    if not hasattr(extract_service, "remove_embeddings"):
        raise HTTPException(status_code=501, detail="remove_embeddings is not implemented in extract_service")

    try:
        res = extract_service.remove_embeddings(id)  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"remove_embeddings failed: {e}") from e

    if isinstance(res, dict) and res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("message", "unknown error"))

    return res if isinstance(res, dict) else {"status": "ok", "id": id}

@router.get("/status", response_model=ExtractionStatus)
def status() -> ExtractionStatus:
    """Return current extract_service status."""
    if not hasattr(extract_service, "get_status"):
        raise HTTPException(status_code=501, detail="get_status is not implemented in extract_service")

    try:
        res = extract_service.get_status()  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get_status failed: {e}") from e

    if not isinstance(res, dict):
        raise HTTPException(status_code=500, detail="get_status returned invalid response")

    num_cams = int(res.get("num_cams", res.get("streams", len(getattr(extract_service, "RTSP_STREAMS", [])))))

    return ExtractionStatus(
        running=bool(res.get("running", False)),
        num_cams=num_cams,
        rtsp_streams=list(res.get("rtsp_streams", [])),
        id=res.get("id"),
    )
