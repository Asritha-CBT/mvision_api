from pydantic import BaseModel, Field, validator

class UserRegister(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, description="User's name is required")
    department: str = Field(..., min_length=1, max_length=50, description="Department is required")

    # Ensure strings are not just spaces
    @validator('name', 'department')
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("This field cannot be empty")
        return v

class UserResponse(BaseModel):
    id: int
    name: str
    department: str

    class Config:
        from_attributes = True   # correct attribute for ORM support
