from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mvision.routes.user_routes import router as user_router
from mvision.db.database import engine
from mvision.db import models

# Create database tables
models.Base.metadata.drop_all(bind=engine)
models.Base.metadata.create_all(bind=engine)


app = FastAPI(title="My FastAPI App")

# Enable CORS so frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],       # GET, POST, OPTIONS, DELETE, etc.
    allow_headers=["*"],       # allow all headers
)

# Include your user router
app.include_router(user_router, prefix="/users", tags=["Users"])

@app.get("/")
def root():
    return {"message": "FastAPI project is running!"}
