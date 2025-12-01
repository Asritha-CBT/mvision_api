from fastapi import FastAPI
from mvision.routes.user_routes import router as user_router
from mvision.db.database import engine
from mvision.db import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="My FastAPI App")

app.include_router(user_router, prefix="/users", tags=["Users"])

@app.get("/")
def root():
    return {"message": "FastAPI project is running!"}
