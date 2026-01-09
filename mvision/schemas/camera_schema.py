from pydantic import BaseModel, Field, field_validator
from typing import Optional
from pydantic.networks import IPvAnyAddress

# Reuse same nested area_definition structure style
class AreaDefinitionResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True


class CameraCreate(BaseModel):
    camera_name: str = Field(..., min_length=1, max_length=80) 
    camera_ip: IPvAnyAddress 
    area_definition_id: int = Field(..., gt=0)

    @field_validator("camera_name")
    def not_empty(cls, value: str):
        if not value.strip():
            raise ValueError("cannot be empty")
        return value.strip()


class CameraResponse(BaseModel):
    id: int
    camera_name: str 
    camera_ip: IPvAnyAddress   
    area_definition_id: int
    area_definition: Optional[AreaDefinitionResponse] = None  # joinedload(Camera.area_definition)

    class Config:
        from_attributes = True

class CameraListResponse(BaseModel):
    id: int
    camera_name: str 
      
    class Config:
        from_attributes = True
