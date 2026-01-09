from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mvision.routes.user_routes import router as user_router
from mvision.routes.embeddings import router as embeddings_router
from mvision.routes.camera_routes import router as camera_router
from mvision.routes.area_definition_routes import router as area_definition_router
from mvision.routes.reports import router as report_router
from mvision.db.database import engine
from mvision.db import models
from mvision.services import extract_service

# Create database tables
# models.Base.metadata.drop_all(bind=engine)
# models.Base.metadata.create_all(bind=engine)


app = FastAPI(title="My FastAPI App")

# Enable CORS so frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all (can restrict later)
    allow_credentials=True,
    allow_methods=["*"],       # GET, POST, OPTIONS, DELETE, etc.
    allow_headers=["*"],       # allow all headers
)

# Include your user router
app.include_router(user_router, prefix="/users", tags=["Users"])
app.include_router(embeddings_router)  
app.include_router(camera_router, prefix="/camera", tags=["Camera"])  
app.include_router(area_definition_router, prefix="/area_definition", tags=["AreaDefinition"])  
app.include_router(report_router, prefix="/reports", tags=["Reports"])

@app.get("/")
def root():
    return {"message": "FastAPI project is running!"}

@app.on_event("startup")
def on_startup():
    extract_service.init_db()
    extract_service.init_models()
    extract_service.ensure_csv_header() 

@app.on_event("shutdown")
def on_shutdown():
    extract_service.stop_extraction()