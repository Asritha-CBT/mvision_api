from pydantic import BaseModel, Field, field_validator
from typing import Optional

# Reuse same nested category structure style
class CameraCategoryResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True


class CameraCreate(BaseModel):
    camera_name: str = Field(..., min_length=1, max_length=80) 
    category_id: int = Field(..., gt=0)

    @field_validator("camera_name")
    def not_empty(cls, value: str):
        if not value.strip():
            raise ValueError("cannot be empty")
        return value.strip()


class CameraResponse(BaseModel):
    id: int
    camera_name: str 
    category_id: int
    category: Optional[CameraCategoryResponse] = None  # joinedload(Camera.category)

    class Config:
        from_attributes = True

class CameraListResponse(BaseModel):
    id: int
    camera_name: str 
      
    class Config:
        from_attributes = True
