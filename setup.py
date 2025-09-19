#!/usr/bin/env python3
"""
Setup script for Guard Owl project
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed: {e.stderr}")
        return False

def main():
    print("🦉 Guard Owl Setup")
    print("=" * 40)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"Python 3.8+ required. Current version: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    print("Created data directory")
    
    # Test imports
    print("Testing imports...")
    test_imports = [
        "fastapi",
        "streamlit", 
        "sentence_transformers",
        "faiss",
        "numpy",
        "pandas"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            print(f"  {module}")
        except ImportError:
            print(f"  {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"Failed to import: {', '.join(failed_imports)}")
        print("Try running: pip install -r requirements.txt")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start backend:  python start_backend.py")
    print("2. Start frontend: python start_frontend.py")
    print("3. Open browser:   http://localhost:8501")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
