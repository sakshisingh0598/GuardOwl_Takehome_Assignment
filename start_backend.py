#!/usr/bin/env python3
"""
Startup script for Guard Owl API backend
"""
import uvicorn
from backend.config import settings

if __name__ == "__main__":
    print("Starting Guard Owl API...")
    print(f"API will be available at: http://{settings.API_HOST}:{settings.API_PORT}")
    print("API Documentation will be available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
