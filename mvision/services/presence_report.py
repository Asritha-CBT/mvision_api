from sqlalchemy.orm import Session
from sqlalchemy import and_,  or_, case, asc
from datetime import datetime 
from mvision.db.session import get_session, engine
from mvision.db.models import UserPresence, User
from mvision.constants import CAMERAS 

def get_user_presence_report(
    db: Session,
    from_ts: datetime | None = None,
    to_ts: datetime | None = None,
    cam: str | None = None,
    user_id: int | None = None,
):
    query = (
        db.query(UserPresence, User)
        .join(User, User.id == UserPresence.user_id)
    )

    filters = []
    CAMERA_MAP = {c["id"]: c["name"] for c in CAMERAS}


    if from_ts and to_ts:
        filters.append(UserPresence.entry_time.between(from_ts, to_ts))
    elif from_ts:
        filters.append(UserPresence.entry_time >= from_ts)
    elif to_ts:
        filters.append(UserPresence.entry_time <= to_ts)

    if cam:
        filters.append(UserPresence.cam_number == cam)

    if user_id:
        filters.append(UserPresence.user_id == user_id)

    if filters:
        query = query.filter(and_(*filters))

    rows = (
        query
        .order_by(
            case(
                (UserPresence.exit_time.is_(None), 0),
                else_=1
            ),
            User.name.asc()
        )
        .all()
    )
    result = []
    for presence, user in rows: 
        cam_name = CAMERA_MAP.get(str(presence.cam_number))
        result.append(
            {
                "id": presence.id,
                "user_id": presence.user_id,
                "user_name": user.name,
                "cam_number": presence.cam_number, 
                "cam_name": cam_name,
                "entry_time": presence.entry_time,
                "exit_time": presence.exit_time,
                "duration": (
                    presence.exit_time - presence.entry_time
                    if presence.exit_time and presence.entry_time
                    else None
                ),
            }
        )

    return result
