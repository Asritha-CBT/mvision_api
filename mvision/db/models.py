from sqlalchemy import Column, Integer, String, DateTime, Enum, Interval, ForeignKey, event
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector  
from datetime import datetime, timedelta


Base = declarative_base()

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    camera_name = Column(String, nullable=False)
    camera_number = Column(String, unique=True, nullable=False)

    category_id = Column(Integer, ForeignKey("camera_category.id"), nullable=False)

    category = relationship("CameraCategory", back_populates="cameras")

class CameraCategory(Base):
    __tablename__ = "camera_category"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)

    # relationships
    cameras = relationship("Camera", back_populates="category")
    users = relationship("User", back_populates="category")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=False) 
    body_embedding = Column(Vector(512), nullable=True)
    face_embedding = Column(Vector(512), nullable=True)
    last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True)
    category_id = Column(Integer, ForeignKey("camera_category.id"), nullable=False)
    category = relationship("CameraCategory", back_populates="users")
    status = Column(String, default="active")

class UserPresence(Base):
    __tablename__ = "user_presence"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    cam_number = Column(String, nullable=False)

    entry_time = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    exit_time = Column(
        DateTime(timezone=True),
        nullable=True
    )

    user = relationship("User", backref="presence_logs")
 
