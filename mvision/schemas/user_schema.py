from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime, timedelta 

class UserRegister(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    department: str = Field(..., min_length=1, max_length=50)

    @field_validator('name', 'department')
    def not_empty(cls, value):
        if not value.strip():
            raise ValueError("cannot be empty")
        return value


class UserResponse(BaseModel):
    id: int
    name: str
    department: Optional[str] = None
    body_embedding: Optional[List[float]] = None   # <-- pgvector field
    face_embedding: Optional[List[float]] = None   # <-- pgvector field

    class Config:
        from_attributes = True


class EmbeddingUpdate(BaseModel):
    body_embedding: Optional[List[float]] = None
    face_embedding: Optional[List[float]] = None

class UserResponse(BaseModel):
    id: int
    name: str
    department: Optional[str] = None
    body_embedding: Optional[List[float]] = None  
    face_embedding: Optional[List[float]] = None  

    class Config:
        from_attributes = True


class EmbeddingUpdate(BaseModel):
    body_embedding: Optional[List[float]] = None
    face_embedding: Optional[List[float]] = None

#user presence
class UserPresenceBase(BaseModel):
    user_id: int
    cam_number: str
    entry_time: datetime
    exit_time: datetime | None = None
    time_spent: timedelta | None = None
    date_time: datetime | None = None


class UserPresenceCreate(UserPresenceBase):
    pass


class UserPresenceResponse(UserPresenceBase):
    id: int

    class Config:
        from_attributes = True    
