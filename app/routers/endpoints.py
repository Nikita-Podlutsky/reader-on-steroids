from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from schemas import SearchRequest, ChatRequest, TranslateRequest, GraphResponse
from services.engine import HybridEngine
from services.clusterizer import process_graph_analysis
from arxiv_loader import ArxivLoader
from config import ARXIV_MAX_RESULTS
import asyncio
import json

router = APIRouter()


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
    cache_key = f"{query}_{req.year_min}_{req.year_max}_{req.cluster_ids}"
    
    if cache_key in search_cache:
        result = search_cache[cache_key]
    else:
        if not arxiv_loader or not engine:
            raise HTTPException(status_code=503, detail="Service not initialized")

        
        papers = await asyncio.to_thread(arxiv_loader.search_and_load, query, max_results=ARXIV_MAX_RESULTS)
        
        
        result = await process_graph_analysis(engine, query, papers)
        
        search_cache[cache_key] = result
    
    
    filtered_nodes = result["nodes"]
    
    
    if req.year_min is not None or req.year_max is not None:
        filtered_nodes = [
            n for n in filtered_nodes
            if n.get("year") is not None
            and (req.year_min is None or n["year"] >= req.year_min)
            and (req.year_max is None or n["year"] <= req.year_max)
        ]
    
    
    if req.cluster_ids is not None and len(req.cluster_ids) > 0:
        filtered_nodes = [
            n for n in filtered_nodes
            if n.get("cluster_id") in req.cluster_ids
        ]
    
    
    result["nodes"] = filtered_nodes
    
    return result

@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """Chat endpoint with support for Ollama and OpenRouter.ai"""
    if not engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    messages = [
        {"role": "system", "content": f"Говори на языке пользователя. Отвечай емко и по делу небойся задать уточняющие вопросы. Контекст:\n{req.paper_text[:3000]}"}
    ] + req.history + [
        {"role": "user", "content": req.question}
    ]
    
    
    provider = req.provider or "ollama"
    
    try:
        if provider == "openrouter":
            
            ans = await engine.chat_openrouter(messages)
            if ans is None:
                raise HTTPException(status_code=500, detail="OpenRouter failed - no response")
            return {"answer": ans}
        elif provider == "google":
            
            ans = await engine.chat_google_ai_studio(messages)
            if ans is None:
                raise HTTPException(status_code=500, detail="Google AI Studio failed - no response")
            return {"answer": ans}
        else:
            
            ans = await asyncio.to_thread(engine.chat_ollama_sync, messages)
            if ans is None:
                raise HTTPException(status_code=500, detail="Ollama failed - no response")
            return {"answer": ans}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error details: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.post("/translate")
async def translate_endpoint(req: TranslateRequest):
    """Translation endpoint with support for Google Translate and Ollama"""
    if not engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    
    provider = req.provider or "google"
    
    
    translation = await engine.translate_fast(req.text, provider=provider)
    return {"translation": translation}