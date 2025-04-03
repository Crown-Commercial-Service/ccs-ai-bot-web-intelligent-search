from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
import uvicorn
import logging
from models.MultiAgnet_model import MultiAgent_Answering 


# configure logging
logging.basicConfig(level=logging.INFO)
logger =  logging.getLogger(__name__)

app = FastAPI(title="Webpilot Intelligent search")

# API key secuirty setup
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# API keys in env
API_KEYS = {os.environ.get("API_KEY", "default_dev_key"): "default"}

# request and response model
class SearchRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = 1000
    additional_context: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    result: str
    metadata: Optional[Dict[str, Any]] = None

# API key verfication function 
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header not in API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, 
                            detail="Invalid API key")
    return api_key_header

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest,
                 api_key: APIKey = Depends(get_api_key)):
    try:
        logger.info(f"Processing search Query: {request.query}")

        result = MultiAgent_Answering(request.query)

        # construct response 
        response = SearchResponse(result = result)

        return response
    except Exception as e:
        logger.error(f"Error processing search query: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error processing search query: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    