from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class SearchRequest(BaseModel): 
    query: str
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    cluster_ids: Optional[List[int]] = None

class ChatRequest(BaseModel): 
    paper_text: str
    history: List[Dict[str, str]] = []
    question: str
    provider: Optional[str] = "google"  # "ollama", "openrouter" или "google"

class TranslateRequest(BaseModel): 
    text: str
    provider: Optional[str] = "google"  # "google" или "ollama"

# Ответ для фронтенда (для документации)
class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    topics: List[Dict[str, Any]]