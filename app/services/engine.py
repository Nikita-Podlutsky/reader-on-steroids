import sys
import torch
import torch.nn.functional as F
import numpy as np
import ollama
import asyncio
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

from app.config import DEVICE, SENTENCE_MODEL_NAME, OLLAMA_MODEL, CHECKPOINT_PATH, BASE_DIR

# Пытаемся импортировать твои файлы из корня проекта

try:
    from app.models import UniversalScorer
    from app.checkpoint_utils import load_checkpoint
    HAS_CUSTOM_MODEL = True
except ImportError:
    print("WARNING: 'models.py' or 'checkpoint_utils.py' not found. Using fallback mode.")
    HAS_CUSTOM_MODEL = False

class HybridEngine:
    def __init__(self):
        self.device = torch.device(DEVICE)
        print(f"INFO: Init Engine on {self.device}...")

        # 1. Загрузка Embedder (всегда нужна)
        self.embedder = SentenceTransformer(SENTENCE_MODEL_NAME, device=self.device)

        # 2. Загрузка Твоей модели (UniversalScorer)
        self.custom_model = None
        if HAS_CUSTOM_MODEL:
            try:
                self.custom_model = UniversalScorer().to(self.device).eval()
                if CHECKPOINT_PATH.exists():
                    print(f"INFO: Loading checkpoint from {CHECKPOINT_PATH}")
                    
                    # БЫЛО: str(CHECKPOINT_PATH) -> ОШИБКА
                    # СТАЛО: CHECKPOINT_PATH (передаем объект Path)
                    load_checkpoint(self.custom_model, None, None, None, CHECKPOINT_PATH)
                    
                else:
                    print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
            except Exception as e:
                print(f"ERROR loading custom model: {e}")
                # Чтобы не крашилось совсем, модель остается None, но сервер работает
                self.custom_model = None

    @torch.no_grad()
    def get_custom_embeddings(self, query, papers, doc_embs_tensor):
        """
        ВАЖНО: Извлекает векторы из UniversalScorer для построения карты (UMAP).
        Если модели нет, возвращает обычные векторы BGE.
        """
        if not self.custom_model or not hasattr(self.custom_model, 'document_encoder'):
            return doc_embs_tensor.cpu().numpy()

        # 1. Вектор запроса (Query)
        q_emb = self.embedder.encode(query, convert_to_tensor=True)
        q_emb = F.normalize(q_emb, p=2, dim=0).unsqueeze(0)
        q_input = torch.cat([q_emb, q_emb], dim=1) # [1, 768*2]
        q_expanded = q_input.expand(len(papers), -1) # Расширяем на кол-во статей

        # 2. Векторы документов (Docs)
        doc_inputs = doc_embs_tensor.unsqueeze(1) 
        mask = torch.ones(len(papers), 1, device=self.device)

        # 3. Прогон через энкодер твоей модели
        # Получаем "умное" представление статьи в контексте запроса
        custom_vectors = self.custom_model.document_encoder(
            doc_inputs, 
            mask, 
            q_expanded
        )
        
        # Нормализация улучшает работу UMAP/Cosine
        return F.normalize(custom_vectors, p=2, dim=1).float().cpu().numpy()

    @torch.no_grad()
    def calculate_relevance(self, query, papers, doc_embs_tensor):
        """Считает Score (важность) для размера узлов"""
        if not self.custom_model:
            return np.random.rand(len(papers)) # Fallback

        q_emb = self.embedder.encode(query, convert_to_tensor=True)
        q_emb = F.normalize(q_emb, p=2, dim=0).unsqueeze(0)
        q_input = torch.cat([q_emb, q_emb], dim=1)
        q_expanded = q_input.expand(len(papers), -1)
        
        doc_inputs = doc_embs_tensor.unsqueeze(1)
        mask = torch.ones(len(papers), 1, device=self.device)

        final_emb = self.custom_model.document_encoder(doc_inputs, mask, q_expanded)
        
        # Скалярное произведение (Cosine Similarity) между запросом и документом в пространстве модели
        scores = (F.normalize(final_emb, p=2, dim=1) * F.normalize(q_expanded, p=2, dim=1)).sum(dim=1)
        return scores.float().cpu().numpy()

    def chat_ollama_sync(self, msgs):
        try: 
            return ollama.chat(model=OLLAMA_MODEL, messages=msgs)['message']['content']
        except Exception as e:
            print(f"Ollama error: {e}")
            return None

    async def name_clusters(self, clusters_data):
        """Генерация названий тем"""
        if not clusters_data: return {}
        
        prompt = f"Give short (1-3 words) titles for {len(clusters_data)} scientific paper clusters.\n"
        for cid, titles in sorted(clusters_data.items()):
            prompt += f"Cluster {cid}:\n" + "\n".join([f"- {t}" for t in titles[:3]]) + "\n\n"
        prompt += "Format: ID: Title\nReply ONLY with the list."
        
        res = await asyncio.to_thread(self.chat_ollama_sync, [{"role": "user", "content": prompt}])
        
        topic_map = {}
        if res:
            for line in res.strip().split('\n'):
                m = re.search(r'^(\d+)\s*[:.-]\s*(.+?)(?:\s*(?:\*\*)?)?$', line.strip())
                if m:
                    try:
                        topic_map[int(m.group(1))] = m.group(2).strip().strip('"*')
                    except: pass
        
        # Если Ollama молчит, даем дефолтные имена
        for cid in clusters_data:
            if cid not in topic_map:
                topic_map[cid] = f"Topic {cid}"
        
        return topic_map