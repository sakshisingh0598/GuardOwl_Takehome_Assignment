from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Report(BaseModel):
    id: str
    siteId: str
    date: str
    guardId: str
    text: str

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about security reports")
    siteId: Optional[str] = Field(None, description="Filter by specific site ID")
    dateRange: Optional[List[str]] = Field(None, description="Date range filter [start_date, end_date] in ISO format")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="Summary of the findings")
    sources: List[str] = Field(..., description="List of report IDs used as sources")
    reports: List[Report] = Field(..., description="Full report details for transparency")

class HealthResponse(BaseModel):
    status: str
    message: str
    reports_loaded: int
    vector_index_ready: bool
