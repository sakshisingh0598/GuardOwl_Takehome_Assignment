---
Author: Sakshi Singh
Date: 18th September 2025
---

# Guard Owl - AI Security Report Analysis

An intelligent chatbot for querying security reports using natural language. Built with FastAPI (backend) and Streamlit (frontend), powered by semantic search and OpenAI's conversational AI.

## Overview

Guard Owl allows security teams to ask natural language questions about their security reports and get intelligent, contextual responses. The system uses:

- **Semantic Search**: Understands the meaning of queries, not just keywords
- **AI-Powered Responses**: Uses OpenAI GPT for conversational, contextual answers
- **Smart Filtering**: Filter by site, date range, and other criteria
- **Real-time Analysis**: Instant search across all security reports

## Features

- Natural language queries (e.g., "What happened at Site S01 last night?")
- AI-generated conversational responses
- Semantic similarity search using embeddings
- Site and date range filtering
- Clean web interface with example queries
- RESTful API for integration

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API Key (recommended for best results)

### Installation & Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure OpenAI (Optional but Recommended)**

   Create a `.env` file:

   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Start the Backend** (Terminal 1)

   ```bash
   python start_backend.py
   ```

   → API available at: http://localhost:8000

4. **Start the Frontend** (Terminal 2)
   ```bash
   python start_frontend.py
   ```
   → Web interface at: http://localhost:8501

## Usage

### Web Interface

1. Open http://localhost:8501
2. Ask questions like:
   - "Show me all geofence breaches"
   - "What incidents happened at S03 yesterday?"
   - "Any reports about a red Toyota Camry?"
3. Use sidebar filters for site and date range
4. Get AI-powered conversational responses

### API Usage

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "security incidents last week",
    "siteId": "S01"
  }'
```

## Architecture

- **Backend**: FastAPI with semantic search (FAISS + sentence-transformers)
- **Frontend**: Streamlit web interface
- **AI Responses**: OpenAI GPT-4 for conversational analysis
- **Data**: 52 mock security reports (Aug-Sep 2025)
- **Embedding**: sentence-transformers (fallback) or OpenAI embeddings

## Sample Data

The system includes mock security reports from 5 sites (S01-S05) covering:

- Security incidents and observations
- Vehicle sightings and violations
- Geofence breaches and exits
- Routine patrol reports

## Configuration

Optional environment variables in `.env`:

```
OPENAI_API_KEY=your_key_here
API_HOST=0.0.0.0
API_PORT=8000
```

## API Documentation

Interactive API docs available at: http://localhost:8000/docs

## Design Choices

### **Technology Stack**

- **FastAPI**: Chosen for async support, automatic OpenAPI docs, and fast development
- **Streamlit**: Rapid prototyping for frontend with minimal complexity
- **FAISS**: Lightweight vector database for fast similarity search without external dependencies
- **Sentence Transformers**: Offline embeddings as primary option, OpenAI as fallback for better quality
- **OpenAI GPT-4**: Conversational AI for natural language responses

### **Embedding Strategy**

- **Primary**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
  - Fast, lightweight, no API costs
  - Good performance for security text
- **Fallback**: OpenAI text-embedding-ada-002 when API key provided
  - Higher quality but requires API costs

### **Search Architecture**

- **Semantic Search**: Vector similarity using FAISS for meaning-based matching
- **Filtering**: Post-search filtering for site/date constraints (simple but effective)
- **Conversational AI**: LLM-only responses (removed all hardcoded templates)
- **Date Awareness**: Current date context for relative time queries

### **Conversational Design**

- **Natural Responses**: Eliminated formal email-style responses
- **Proactive Assistance**: AI suggests alternatives when no results found
- **Context Aware**: Understands available data range and suggests alternatives

## Scaling to 1M+ Reports

### **Key Scaling Strategies**

- **Vector Databases**: Replace FAISS with Pinecone (managed) or Qdrant (self-hosted)
- **Document Storage**: PostgreSQL for full reports with metadata
- **Caching**: Redis for frequent queries (1-hour TTL)
- **Batch Processing**: Process embeddings in batches of 100-1000 reports
- **Load Balancing**: Multiple API instances behind nginx

```python
# Example scaled architecture
class ScaledService:
    def __init__(self):
        self.vector_db = PineconeClient()  # Handles billions of vectors
        self.document_db = PostgreSQL()    # Full reports storage
        self.cache = Redis()               # Query caching
```

### **Infrastructure & Performance**

- **Containerization**: Docker with 3+ API replicas, load balancer
- **Databases**: Qdrant/Pinecone + PostgreSQL + Redis stack
- **Data Partitioning**: By date/site for optimized queries
- **Performance**: Async operations, connection pooling, incremental updates

## Extending to Timesheets & Geofence Events

### **Unified Event System**

Extend the current security report model to handle multiple event types:

```python
class EventType(str, Enum):
    SECURITY_REPORT = "security_report"
    TIMESHEET = "timesheet"
    GEOFENCE = "geofence"

class BaseEvent(BaseModel):
    id: str
    type: EventType
    site_id: str
    timestamp: datetime
    guard_id: str
```

### **Key Extension Strategies**

- **Multi-Type Search**: Different embedding models for each event type
- **Cross-Event Queries**: "Show me timesheet discrepancies near security incidents"
- **Pattern Detection**: Correlate geofence exits with timesheet breaks
- **Anomaly Detection**: Flag suspicious timing patterns across event types
- **Unified Interface**: Single API endpoint for all event types

### **Implementation Benefits**

- **Holistic View**: See all guard activities in one interface
- **Better Context**: Security incidents with timesheet/location context
- **Pattern Detection**: Identify correlations between different event types
- **Anomaly Detection**: Flag suspicious timing patterns across events
- **Unified Search**: Ask questions spanning multiple data types

---

**Note**: This is a prototype system. For production use, consider adding authentication, comprehensive logging, real-time streaming, and advanced analytics.
