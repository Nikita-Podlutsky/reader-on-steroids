from fastapi import APIRouter, HTTPException
from app.schemas import SearchRequest, ChatRequest, TranslateRequest, GraphResponse
from app.services.engine import HybridEngine
from app.services.clusterizer import process_graph_analysis
from app.arxiv_loader import ArxivLoader
from app.config import ARXIV_MAX_RESULTS
import asyncio

router = APIRouter()

# Глобальные объекты (инициализируются в main.py, здесь просто ссылки)
engine: HybridEngine = None
arxiv_loader: ArxivLoader = None
search_cache = {}

def init_globals(eng, loader):
    global engine, arxiv_loader
    engine = eng
    arxiv_loader = loader

@router.post("/analyze_graph", response_model=GraphResponse)
async def analyze_graph(req: SearchRequest):
    query = req.query.strip()
    if query in search_cache:
        return search_cache[query]
    
    if not arxiv_loader or not engine:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # 1. Качаем статьи
    papers = await asyncio.to_thread(arxiv_loader.search_and_load, query, max_results=ARXIV_MAX_RESULTS)
    
    # 2. Запускаем пайплайн (Embed -> Scorer -> UMAP -> JSON)
    result = await process_graph_analysis(engine, query, papers)
    
    search_cache[query] = result
    return result

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = [
        {"role": "system", "content": f"Context:\n{req.paper_text[:3000]}"}
    ] + req.history + [
        {"role": "user", "content": req.question}
    ]
    
    ans = await asyncio.to_thread(engine.chat_ollama_sync, messages)
    if ans is None:
        raise HTTPException(status_code=500, detail="Ollama failed")
    return {"answer": ans}

@router.post("/translate")
async def translate_endpoint(req: TranslateRequest):
    sys_prompt = "Translate abstract to Russian. Keep terms. Return only translation."
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": req.text[:2000]}
    ]
    ans = await asyncio.to_thread(engine.chat_ollama_sync, messages)
    return {"translation": ans.strip() if ans else "Translation failed"}