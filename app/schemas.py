from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class SearchRequest(BaseModel): 
    query: str

class ChatRequest(BaseModel): 
    paper_text: str
    history: List[Dict[str, str]] = []
    question: str

class TranslateRequest(BaseModel): 
    text: str

# Ответ для фронтенда (для документации)
class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    topics: List[Dict[str, Any]]