#!/usr/bin/env python3
"""
Startup script for Guard Owl Streamlit frontend
"""
import subprocess
import sys

if __name__ == "__main__":
    print("Starting Guard Owl Frontend...")
    print("Frontend will be available at: http://localhost:8501")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "frontend/app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nFrontend stopped.")
