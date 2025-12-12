from sqlalchemy import Column, Integer, String, DateTime, Enum, Interval, ForeignKey, event
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector  
from datetime import datetime, timedelta


Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    department = Column(String, nullable=False) 
    body_embedding = Column(Vector(512), nullable=True)
    face_embedding = Column(Vector(512), nullable=True)
    last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True)
    status = Column(String, default="active")

class UserPresence(Base):
    __tablename__ = "user_presence"

    id = Column(Integer, primary_key=True, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    cam_number = Column(String, nullable=False)

    entry_time = Column(DateTime(timezone=True), nullable=False)
    exit_time = Column(DateTime(timezone=True), nullable=True)

    # time spent in seconds, minutes, etc. Use INTERVAL in PostgreSQL
    time_spent = Column(Interval, nullable=True)

    # For date-only or full timestamp — keeping full timestamp for reports
    date_time = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Optional relationship
    user = relationship("User", backref="presence_logs")  

@event.listens_for(UserPresence, "before_update")
def compute_time_spent(mapper, connection, target):
    if target.exit_time and target.entry_time:
        target.time_spent = target.exit_time - target.entry_time      
