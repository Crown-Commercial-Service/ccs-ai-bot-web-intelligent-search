from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
from models.MultiAgnet_model import MultiAgent_Answering as process_query
from auth import validate_api_key
import time
import uvicorn

app = FastAPI(
    title="Intelligent Search Engine API",
    description="API for multi-agent intelligent search engine built with LangGraph",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[list] = None
    processing_time: Optional[float] = None

@app.get("/")
async def home():
    """Health check endpoint"""
    return {"status": "Intelligent Search Engine API is Active"}

@app.post("/search", response_model=QueryResponse)
async def search(
    request: QueryRequest,
    api_key: str = Depends(validate_api_key)
):
    """
    Process a search query through the intelligent search engine
    
    Args:
        query: The search query string
        context: Additional context for the search (optional)
    
    Returns:
        Response containing the answer and any sources/references
    """
    try:
        start_time = time.time()
        answer, framework_numbers = process_query(request.query)
        processing_time = time.time() - start_time
        
        return QueryResponse(
            answer=answer,
            sources=framework_numbers, 
            processing_time=processing_time

        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
# gunicorn -k uvicorn.workers.UvicornWorker app:app --bind=0.0.0.0:8000