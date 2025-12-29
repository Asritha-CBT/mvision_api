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

class CategoryResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True

class UserResponse(BaseModel):
    id: int
    name: str
    department: Optional[str] = None
    body_embedding: Optional[List[float]] = None   # <-- pgvector field
    face_embedding: Optional[List[float]] = None   # <-- pgvector field
    category_id: Optional[int] = None
    category: Optional[CategoryResponse] = None  # nested object

    class Config:
        from_attributes = True


class EmbeddingUpdate(BaseModel):
    body_embedding: Optional[List[float]] = None
    face_embedding: Optional[List[float]] = None 

class EmbeddingUpdate(BaseModel):
    body_embedding: Optional[List[float]] = None
    face_embedding: Optional[List[float]] = None

#user presence
class UserPresenceBase(BaseModel):
    user_id: int
    cam_number: str


class UserPresenceCreate(UserPresenceBase):
    # entry_time is optional → server_default=now()
    entry_time: datetime | None = None

class UserPresenceResponse(BaseModel):
    id: int
    user_id: int
    user_name: str   
    cam_number: str
    cam_name: str
    entry_time: datetime
    exit_time: datetime | None
    duration: timedelta | None

    class Config:
        from_attributes = True