from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.schemas import SearchRequest, ChatRequest, TranslateRequest, GraphResponse
from app.services.engine import HybridEngine
from app.services.clusterizer import process_graph_analysis
from app.arxiv_loader import ArxivLoader
from app.config import ARXIV_MAX_RESULTS
import asyncio
import json

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
    """Chat endpoint with streaming support and fallback"""
    if not engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    messages = [
        {"role": "system", "content": f"Говори на языке пользователя. Отвечай емко и по делу небойся задать уточняющие вопросы. Контекст:\n{req.paper_text[:3000]}"}
    ] + req.history + [
        {"role": "user", "content": req.question}
    ]
    
    # Проверяем, запрашивает ли клиент streaming (через заголовок или параметр)
    # Для обратной совместимости используем обычный режим по умолчанию
    use_streaming = False  # Можно добавить проверку заголовка
    
    if use_streaming:
        # Streaming режим
        async def generate():
            """Async generator for streaming responses"""
            try:
                async for chunk in engine.chat_ollama_stream(messages):
                    if chunk is None:
                        break
                    yield f"data: {json.dumps({'content': chunk, 'done': False})}\n\n"
                yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            except Exception as e:
                error_msg = json.dumps({'error': str(e), 'done': True})
                yield f"data: {error_msg}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Обычный режим (для обратной совместимости)
        try:
            ans = await asyncio.to_thread(engine.chat_ollama_sync, messages)
            if ans is None:
                raise HTTPException(status_code=500, detail="Ollama failed - no response")
            return {"answer": ans}
        except Exception as e:
            print(f"Chat error details: {e}")  # Логируем для отладки
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@router.post("/translate")
async def translate_endpoint(req: TranslateRequest):
    """Optimized translation endpoint using fast model"""
    if not engine:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Используем оптимизированный метод перевода
    translation = await engine.translate_fast(req.text)
    return {"translation": translation}