from sqlalchemy import Column, Integer, String, DateTime, Enum, Interval, ForeignKey, event, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector  
from datetime import datetime, timedelta
from sqlalchemy.dialects.postgresql import INET


Base = declarative_base()

class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False, unique=True)
    status = Column(String(20), nullable=False, default="active")
    users = relationship("User", back_populates="department")

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    camera_name = Column(String, nullable=False) 
    camera_ip = Column(INET, nullable=False, unique=True)
    area_definition_id = Column(Integer, ForeignKey("area_definition.id"), nullable=False) 
    area_definition = relationship("AreaDefinition", back_populates="cameras")

class AreaDefinition(Base):
    __tablename__ = "area_definition"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(String, nullable=True)

    # relationships
    cameras = relationship("Camera", back_populates="area_definition")
    users = relationship("User", back_populates="area_definition")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    gender = Column(String, nullable=False)
    department_id = Column(
        Integer,
        ForeignKey("departments.id"),
        nullable=False
    )
    department = relationship("Department", back_populates="users")
    body_embedding = Column(Vector(512), nullable=True)
    face_embedding = Column(Vector(512), nullable=True)
    last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True)
    area_definition_id = Column(Integer, ForeignKey("area_definition.id"), nullable=True)
    area_definition = relationship("AreaDefinition", back_populates="users")
    status = Column(String, default="active")
    __table_args__ = (
        UniqueConstraint("name", "department_id", name="uq_user_name_department"),
    )

class UserPresence(Base):
    __tablename__ = "user_presence"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)

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
    camera = relationship("Camera")
 
