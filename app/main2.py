# ==============================================================================
# KOTODEX CORE API (BERTOPIC EDITION - FIX TOPIC NAMES)
# ==============================================================================

import os
import sys
import logging
import asyncio
import json
import colorsys
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache

# ML Libraries
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# FastAPI
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import ollama
import re

# BERTopic imports
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

# --- ИМПОРТЫ ТВОЕГО ПРОЕКТА ---
try:
    sys.path.append(str(Path(__file__).parent))
    from app.arxiv_loader import ArxivLoader
    from config import CONFIG
    from models import UniversalScorer
    from checkpoint_utils import load_checkpoint
except ImportError as e:
    print(f"Error importing project files: {e}")
    sys.exit(1)


# ==============================================================================
# CONFIGURATION
# ==============================================================================
OLLAMA_MODEL = "gemma3"
SENTENCE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_TOPICS = getattr(CONFIG, 'MAX_TOPICS', 5)

# Инициализируем лоадер
arxiv_loader = ArxivLoader(max_workers=10)

# --- ENGINE ---
class HybridEngine:
    def __init__(self):
        self.device = torch.device(CONFIG.DEVICE)
        print(f"INFO: Init Engine on {self.device}...")

        # 1. Custom Model (Score)
        self.custom_model = UniversalScorer().to(self.device).eval()
        ckpt = Path(__file__).parent.parent / "checkpoints" / "best_model.pt"
        if ckpt.exists():
            load_checkpoint(self.custom_model, None, None, None, ckpt)
        
        # 2. Standard Embedder
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(SENTENCE_MODEL_NAME, device=self.device)

    @torch.no_grad()
    def calculate_relevance(self, query, papers, doc_embs_tensor):
        # Hack for 384 -> 768
        q_emb = self.embedder.encode(query, convert_to_tensor=True)
        q_emb = F.normalize(q_emb, p=2, dim=0).unsqueeze(0)
        q_input = torch.cat([q_emb, q_emb], dim=1)
        
        doc_inputs = doc_embs_tensor.unsqueeze(1)
        q_expanded = q_input.expand(len(papers), -1)
        
        if hasattr(self.custom_model, 'document_encoder'):
            final = self.custom_model.document_encoder(
                doc_inputs, 
                torch.ones(len(papers), 1, device=self.device), 
                q_expanded
            )
            scores = (F.normalize(final, p=2, dim=1) * F.normalize(q_expanded, p=2, dim=1)).sum(dim=1)
            return scores.cpu().numpy()
        return np.random.rand(len(papers))

    def chat_ollama_sync(self, msgs):
        try: 
            return ollama.chat(model=OLLAMA_MODEL, messages=msgs)['message']['content']
        except Exception as e:
            print(f"Ollama error: {e}")
            return None

    async def name_clusters(self, clusters_data):
        """Генерация названий тем с улучшенным парсингом"""
        if not clusters_data:
            return {}
            
        # Более явный промпт
        prompt = f"""Ты - научный аналитик. Дай краткие названия для {len(clusters_data)} групп научных статей.
Каждое название должно быть 1-3 слова, отражать суть темы. Формат: ID: Название

"""
        for cid, titles in sorted(clusters_data.items()):
            prompt += f"ГРУППА {cid}:\n" + "\n".join([f"- {t}" for t in titles[:3]]) + "\n\n"
        
        prompt += "Отвечай ТОЛЬКО списком, без пояснений."
        
        print(f"DEBUG: Sending to Ollama:\n{prompt}\n")  # Debug log
        
        res = await asyncio.to_thread(self.chat_ollama_sync, [{"role": "user", "content": prompt}])
        
        topic_map = {}
        if res:
            print(f"DEBUG: Ollama raw response:\n{res}\n")  # Debug log
            
            for line in res.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Более гибкий regex
                m = re.search(r'^(\d+)\s*[:.-]\s*(.+?)(?:\s*(?:\*\*)?)?$', line)
                if m:
                    topic_id = int(m.group(1))
                    name = m.group(2).strip().strip('"*').strip()
                    if name and topic_id in clusters_data:
                        topic_map[topic_id] = name
        
        print(f"DEBUG: Parsed topic names: {topic_map}")  # Debug log
        
        # Создаем запасные названия, если Ollama не ответила
        if not topic_map:
            print("WARNING: Ollama failed to generate names, using fallbacks")
            topic_map = {cid: f"Topic {cid}" for cid in clusters_data.keys()}
        
        return topic_map

# --- API ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
engine = None
search_cache = {}

# Подключаем статические файлы
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def root():
    index_file = static_path / "index.html"
    if index_file.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(index_file))
    return {
        "status": "ok",
        "message": "Kotodex API is running",
        "endpoints": {
            "POST /analyze_graph": "Search and analyze papers",
            "POST /chat": "Chat with paper context",
            "POST /translate": "Translate abstracts"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "device": str(engine.device) if engine else "N/A",
        "cache_size": len(search_cache)
    }

class SearchRequest(BaseModel): 
    query: str

class ChatRequest(BaseModel): 
    paper_text: str
    history: List[dict]
    question: str

class TranslateRequest(BaseModel): 
    text: str

@app.on_event("startup")
async def startup():
    global engine
    engine = HybridEngine()

def generate_neon_colors(n):
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        rgb = colorsys.hls_to_rgb(hue, 0.6, 1.0)
        hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgb)
        colors.append(hex_color)
    return colors

@app.post("/analyze_graph")
async def handle_search(req: SearchRequest):
    query = req.query.strip()
    if query in search_cache: 
        return search_cache[query]

    # 1. Load papers
    papers = await asyncio.to_thread(arxiv_loader.search_and_load, query, max_results=30)
    if not papers: 
        return {"nodes": [], "links": [], "topics": []}

    # 2. Generate embeddings
    texts = [f"{p['title']}. {p['abstract']}"[:2000] for p in papers]
    doc_tensor = engine.embedder.encode(texts, convert_to_tensor=True)
    doc_np = doc_tensor.cpu().numpy()

    # 3. Calculate relevance scores
    scores = await asyncio.to_thread(engine.calculate_relevance, query, papers, doc_tensor)

    # 4. BERTopic clustering
    umap_model = UMAP(
        n_neighbors=min(15, len(papers) - 1) if len(papers) > 1 else 1,
        n_components=5,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    
    min_cluster_size = min(5, max(2, len(papers) // 3)) if len(papers) > 2 else 2
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    topic_model = BERTopic(
        embedding_model=engine.embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=False,
        nr_topics=MAX_TOPICS if MAX_TOPICS > 0 else None
    )
    
    # Fit model
    topics, probs = topic_model.fit_transform(texts)
    
    # Handle outliers
    unique_topics = sorted(set(topics))
    outlier_topic = None
    if -1 in unique_topics:
        outlier_topic = max(unique_topics) + 1
        topics = [outlier_topic if t == -1 else t for t in topics]
        unique_topics = sorted(set(topics))
    
    n_clusters = len(unique_topics)
    
    # 5. Generate 2D coordinates
    coords_2d = UMAP(
        n_neighbors=min(15, len(papers) - 1) if len(papers) > 1 else 1,
        n_components=2,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    ).fit_transform(doc_np)

    # 6. Create similarity links
    sim_matrix = cosine_similarity(doc_np)
    links = []
    THRESHOLD = 0.65 
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            if sim_matrix[i][j] > THRESHOLD:
                links.append({"source": papers[i]['id'], "target": papers[j]['id']})

    # 7. Prepare clusters data and generate names
    clusters_data = {}
    root_candidates = {}
    
    for i, topic_id in enumerate(topics):
        topic_id = int(topic_id)
        if topic_id not in clusters_data: 
            clusters_data[topic_id] = []
        clusters_data[topic_id].append(papers[i]['title'])
        
        s = float(scores[i])
        if topic_id not in root_candidates or s > root_candidates[topic_id][1]:
            root_candidates[topic_id] = (i, s)

    # Generate topic names
    topic_names = await engine.name_clusters(clusters_data)
    
    # Fallback names if Ollama failed
    if not topic_names:
        topic_names = {cid: f"Topic {cid}" for cid in clusters_data.keys()}
    
    palette = generate_neon_colors(n_clusters)
    
    # Sort topics by size
    topics_list = []
    topic_sizes = {cid: len(clusters_data[cid]) for cid in clusters_data.keys()}
    sorted_topics = sorted(clusters_data.keys(), key=lambda x: topic_sizes[x], reverse=True)
    
    for idx, cid in enumerate(sorted_topics):
        name = topic_names.get(cid, f"Topic {cid}")
        topics_list.append({
            "id": idx, 
            "original_id": cid, 
            "name": name, 
            "color": palette[idx],
            "size": topic_sizes[cid]
        })
    
    # Create ID mapping
    topic_id_mapping = {t["original_id"]: t["id"] for t in topics_list}

    # 8. Build nodes
    nodes = []
    for i, p in enumerate(papers):
        original_cid = int(topics[i])
        cid = topic_id_mapping[original_cid]
        is_root = (root_candidates[original_cid][0] == i)

        nodes.append({
            "id": p['id'],
            "label": p['title'],
            "abstract": p['abstract'],
            "full_text": p.get('full_text', ''),
            "x": float(coords_2d[i][0]) * 20, 
            "y": float(coords_2d[i][1]) * 20,
            "cluster": cid,
            "color": palette[cid],
            "val": float(scores[i]) * 15 + 5,
            "is_root": is_root
        })

    # 9. Prepare final response
    final_topics = [{"id": t["id"], "name": t["name"], "color": t["color"]} for t in topics_list]
    
    res = {"nodes": nodes, "links": links, "topics": final_topics}
    search_cache[query] = res
    return res

@app.post("/chat")
async def handle_chat(req: ChatRequest):
    messages = [
        {"role": "system", "content": f"Контекст:\n{req.paper_text[:3000]}"}
    ] + req.history + [
        {"role": "user", "content": req.question}
    ]
    
    ans = await asyncio.to_thread(engine.chat_ollama_sync, messages)
    if ans is None:
        raise HTTPException(status_code=500, detail="Ollama chat failed")
    return {"answer": ans}

@app.post("/translate")
async def handle_translate(req: TranslateRequest):
    system_prompt = """Ты - профессиональный переводчик научных текстов. 
Твоя задача: перевести аннотацию научной статьи с английского на русский язык.
- Переводи только содержание, без пояснений
- Сохраняй технические термины, формулы, ссылки
- Сохраняй академический стиль
- Если текст уже на русском, верни его без изменений"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Переведи на русский:\n\n{req.text[:2000]}"}
    ]
    
    ans = await asyncio.to_thread(engine.chat_ollama_sync, messages)
    if ans is None:
        raise HTTPException(status_code=500, detail="Translation failed")
    
    # Очистка markdown
    cleaned = ans.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split('\n')
        if len(lines) > 2 and lines[-1] == "```":
            cleaned = '\n'.join(lines[1:-1])
    
    return {"translation": cleaned.strip()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)