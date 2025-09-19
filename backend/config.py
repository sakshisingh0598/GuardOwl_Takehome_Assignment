import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    
    # OpenAI Configuration (optional)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Vector Database Configuration
    VECTOR_INDEX_PATH = "data/vector_index.faiss"
    REPORTS_INDEX_PATH = "data/reports_index.pkl"
    
    # Embedding Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fallback model
    EMBEDDING_DIMENSION = 384  # Dimension for sentence-transformers model
    
    # Retrieval Configuration
    TOP_K_RESULTS = 3
    
settings = Settings()
