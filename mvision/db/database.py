from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from mvision.core.config import settings

# 🔥 Debug: confirm DATABASE_URL
print("\n==============================")
print(" LOADED DATABASE URL:", settings.database_url)
print("==============================\n")

if not settings.database_url:
    raise ValueError("❌ ERROR: DATABASE URL is EMPTY! Check your .env")

# Create SQLAlchemy engine
engine = create_engine(
    settings.database_url,
    echo=True,          # optional: prints all SQL queries
    pool_pre_ping=True  # avoids connection hang issues
)

# Create session
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
