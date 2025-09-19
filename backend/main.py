from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import os

from .models import QueryRequest, QueryResponse, HealthResponse
from .services import ReportAnalysisService
from .config import settings

# Global service instance
analysis_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global analysis_service
    
    # Startup
    print("Starting Guard Owl API...")
    analysis_service = ReportAnalysisService()
    
    # Initialize with mock reports
    reports_path = "guard_owl_mock_reports.json"
    if os.path.exists(reports_path):
        analysis_service.initialize(reports_path)
        print("Service initialized successfully")
    else:
        print(f"Warning: Reports file not found at {reports_path}")
    
    yield
    
    # Shutdown
    print("Shutting down Guard Owl API...")

app = FastAPI(
    title="Guard Owl API",
    description="A lightweight service for querying security reports using semantic search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Streamlit default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if analysis_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    status = analysis_service.get_health_status()
    return HealthResponse(**status)

@app.post("/query", response_model=QueryResponse)
async def query_reports(request: QueryRequest):
    """Query security reports using natural language"""
    if analysis_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        response = analysis_service.query(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Guard Owl API",
        "description": "Semantic search for security reports",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
