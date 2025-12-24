from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime 
from mvision.db.database import get_db
from mvision.services.presence_report import get_user_presence_report 
from mvision.schemas import user_schema
from mvision.constants import CAMERAS  

router = APIRouter()
@router.get("/cameras")
def get_cameras():
    """
    Returns list of available cameras (hardcoded)
    """
    return CAMERAS

@router.get(
    "/user_presence",
    response_model=list[user_schema.UserPresenceResponse]
)
def user_presence_report(
    from_ts: datetime | None = Query(None, alias="from"),
    to_ts: datetime | None = Query(None, alias="to"),
    cam: str | None = None,
    user: int | None = None,
    db: Session = Depends(get_db),
):
    """
    User Presence Report API
    """

    return get_user_presence_report(
        db=db,
        from_ts=from_ts,
        to_ts=to_ts,
        cam=cam,
        user_id=user,
    )
