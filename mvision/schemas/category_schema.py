from pydantic import BaseModel, Field, field_validator
from typing import Optional

class CameraCategoryCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: Optional[str] = None

    @field_validator("name")
    def not_empty(cls, v: str):
        if not v.strip():
            raise ValueError("cannot be empty")
        return v.strip()

class CameraCategoryUpdate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: Optional[str] = None

    @field_validator("name")
    def not_empty(cls, v: str):
        if not v.strip():
            raise ValueError("cannot be empty")
        return v.strip()

class CameraCategoryResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

    class Config:
        from_attributes = True
