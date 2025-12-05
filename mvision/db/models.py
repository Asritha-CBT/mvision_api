from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector


Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=False) 
    embedding = Column(Vector(512), nullable=True)
    last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True)
    status = Column(String, default="active")
