from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic.types import Base64Bytes

EMBED_DIM = 512


def _validate_embed_512(v: Optional[List[float]]) -> Optional[List[float]]:
    if v is None:
        return None
    if len(v) != EMBED_DIM:
        raise ValueError(f"embedding must be length {EMBED_DIM}")
    return v


class UserRegister(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    department: str = Field(..., min_length=1, max_length=50)

    @field_validator("name", "department")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("cannot be empty")
        return value


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    department: Optional[str] = None

    # centroids (pgvector -> list[float])
    body_embedding: Optional[List[float]] = None
    face_embedding: Optional[List[float]] = None

    # raw banks (bytea) returned as base64 in JSON
    body_embeddings_raw: Optional[Base64Bytes] = None
    face_embeddings_raw: Optional[Base64Bytes] = None

    @field_validator("body_embedding")
    @classmethod
    def _body_len(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        return _validate_embed_512(v)

    @field_validator("face_embedding")
    @classmethod
    def _face_len(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        return _validate_embed_512(v)


class EmbeddingUpdate(BaseModel):
    body_embedding: Optional[List[float]] = None
    face_embedding: Optional[List[float]] = None
    body_embeddings_raw: Optional[Base64Bytes] = None
    face_embeddings_raw: Optional[Base64Bytes] = None

    @field_validator("body_embedding")
    @classmethod
    def _body_len(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        return _validate_embed_512(v)

    @field_validator("face_embedding")
    @classmethod
    def _face_len(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        return _validate_embed_512(v)


# ---------------- user presence ----------------
class UserPresenceBase(BaseModel):
    user_id: int
    cam_number: str = Field(..., min_length=1, max_length=64)


class UserPresenceCreate(UserPresenceBase):
    entry_time: datetime | None = None


class UserPresenceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    user_name: str
    cam_number: str
    cam_name: str
    entry_time: datetime
    exit_time: datetime | None
    duration: timedelta | None
