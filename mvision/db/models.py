from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=True)
    embeddings_path = Column(String, nullable=True)
    last_embedding_update_ts = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="active")  # e.g., active/inactive
