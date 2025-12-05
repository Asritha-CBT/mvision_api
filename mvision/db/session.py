from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from mvision.constants import EMB_DB_URL


# Synchronous engine - works well with worker threads
engine = create_engine(EMB_DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session():
    return SessionLocal()