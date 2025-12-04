from pydantic import BaseModel, Field, field_validator

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
    department: str
    embeddings_path: str | None = None

    class Config:
        from_attributes = True



class EmbeddingUpdate(BaseModel):
    embeddings_path: str
