# ==============================================================================
#
#                   FastAPI Сервер для Иерархического Ранжирования
#
# ==============================================================================
#
#   Что делает:
#   1. Запускает HTTP API сервер на FastAPI для семантического ранжирования документов.
#   2. Предоставляет эндпоинт /encode-query для преобразования текста запроса в вектор.
#   3. Предоставляет эндпоинт /rank для ранжирования списка документов по релевантности запросу.
#   4. Загружает обученную модель UniversalScorer с поддержкой чекпоинтов и LoRA.
#   5. Обрабатывает PDF и текстовые документы, извлекает текст и кодирует предложения.
#   6. Возвращает скоры релевантности для каждого документа в запросе.
#
#   Запуск:
#   > python api.py
#
#   API будет доступно по адресу http://127.0.0.1:8000 (по умолчанию)
#
# ==============================================================================

# ==============================================================================
# 0. ИМПОРТЫ
# ==============================================================================
import base64
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback

import fitz # PyMuPDF
import nltk
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# ==============================================================================
# 1. ИМПОРТЫ КОМПОНЕНТОВ ПРОЕКТА
# ==============================================================================
from config import CONFIG
from models import UniversalScorer # Наша основная модель
from checkpoint_utils import find_latest_checkpoint, load_checkpoint

# ==============================================================================
# 2. СХЕМЫ Pydantic (валидация запросов/ответов)
# ==============================================================================
class Base64Vector(BaseModel):
    shape: List[int]
    dtype: str
    data: str

class QueryRequest(BaseModel):
    text: str

class DocumentInput(BaseModel):
    path: Optional[str] = None
    content: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def check_one_source(cls, data: Any) -> Any:
        if not isinstance(data, dict): raise ValueError('Input must be a dictionary')
        if data.get('path') is None and data.get('content') is None:
            raise ValueError("Either 'path' or 'content' must be provided.")
        if data.get('path') is not None and data.get('content') is not None:
            raise ValueError("Provide either 'path' or 'content', not both.")
        return data

class RankingRequest(BaseModel):
    query: str
    documents: List[DocumentInput]

class QueryResponse(BaseModel):
    vector: Base64Vector

class RankingResponse(BaseModel):
    scores: List[float]

# ==============================================================================
# 3. УТИЛИТЫ
# ==============================================================================
def safe_resolve_path(user_path: str) -> Path:
    resolved_path = CONFIG.API_ALLOWED_DOCS_DIR.joinpath(user_path).resolve()
    if not str(resolved_path).startswith(str(CONFIG.API_ALLOWED_DOCS_DIR)):
        raise PermissionError(f"Access to path '{user_path}' is denied.")
    if not resolved_path.exists():
        raise FileNotFoundError(f"File not found: '{user_path}'")
    return resolved_path

def extract_text_from_path(file_path: Path) -> str:
    try:
        if file_path.suffix.lower() == ".pdf":
            with fitz.open(file_path) as doc:
                text = " ".join(page.get_text() for page in doc)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print(f"Warning: Could not extract text from {file_path}. Error: {e}")
        return ""

def vector_to_base64(vec: torch.Tensor) -> Dict[str, Any]:
    arr = vec.detach().cpu().numpy().astype("float32")
    return {
        "shape": list(arr.shape), "dtype": "float32",
        "data": base64.b64encode(arr.tobytes()).decode("ascii")
    }

# ==============================================================================
# 4. ДВИЖОК ИНФЕРЕНСА (InferenceEngine)
# ==============================================================================
class InferenceEngine:
    def __init__(self):
        print(f"INFO: Initializing Inference Engine on device: {CONFIG.DEVICE}")
        self.device = torch.device(CONFIG.DEVICE)
        
        print("INFO: Initializing model architecture...")
        self.model = UniversalScorer().to(self.device).eval()

        best_model_path = CONFIG.CHECKPOINT_DIR / CONFIG.CHECKPOINT_FILENAME_BEST
        checkpoint_path_to_load = best_model_path if best_model_path.is_dir() else find_latest_checkpoint(CONFIG.CHECKPOINT_DIR)

        if checkpoint_path_to_load:
            print(f"INFO: Found checkpoint to load at: {checkpoint_path_to_load}")
            load_checkpoint(model=self.model, optimizer=None, scheduler=None, scaler=None, path=checkpoint_path_to_load)
        else:
            print("WARNING: No checkpoint found. Using initial, untrained model weights!")
        
        print(f"INFO: Loading sentence encoder for on-the-fly processing: {CONFIG.SENTENCE_EMBEDDING_MODEL}")
        self.sentence_encoder = SentenceTransformer(CONFIG.SENTENCE_EMBEDDING_MODEL, device=self.device)
        
        print(f"INFO: Loading tokenizer: {CONFIG.QUERY_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.QUERY_MODEL_NAME, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        tokenized = self.tokenizer(
            text, max_length=CONFIG.QUERY_MODEL_MAX_LEN,
            padding='max_length', truncation=True, return_tensors='pt'
        ).to(self.device)
        query_vec = self.model.query_encoder(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
            token_type_ids=tokenized.get('token_type_ids')
        )
        return query_vec

    @torch.no_grad()
    def rank_documents(self, query: str, documents: List[DocumentInput]) -> List[float]:
        if not documents: return []

        model_dtype = next(self.model.parameters()).dtype
        query_vec = self.encode_query(query)
        
        texts = []
        for doc in documents:
            if doc.content: texts.append(doc.content)
            elif doc.path:
                try:
                    resolved_path = safe_resolve_path("arxiv_pdfs/" + doc.path)
                    texts.append(extract_text_from_path(resolved_path))
                except (FileNotFoundError, PermissionError) as e:
                    raise HTTPException(status_code=404, detail=str(e))

        docs_sentences = [nltk.sent_tokenize(text) for text in texts]
        
        sentence_embeddings = []
        for sents in docs_sentences:
            if sents:
                sents_truncated = sents[:CONFIG.MAX_SENTENCES]
                embs = self.sentence_encoder.encode(sents_truncated, convert_to_tensor=True, batch_size=CONFIG.API_SENTENCE_ENCODING_BATCH)
                sentence_embeddings.append(embs)
            else:
                sentence_embeddings.append(torch.empty(0, CONFIG.SENTENCE_EMBEDDING_DIM, device=self.device))
        
        max_sents = max(len(s) for s in sentence_embeddings) if sentence_embeddings else 0
        if max_sents == 0: return [0.0] * len(documents)

        batch_embs = torch.zeros(len(documents), max_sents, CONFIG.SENTENCE_EMBEDDING_DIM, device=self.device, dtype=model_dtype)
        batch_mask = torch.zeros(len(documents), max_sents, device=self.device, dtype=torch.long)

        for i, embs in enumerate(sentence_embeddings):
            if embs.shape[0] > 0:
                batch_embs[i, :embs.shape[0], :] = embs
                batch_mask[i, :embs.shape[0]] = 1
        
        query_vec_expanded = query_vec.expand(len(documents), -1)
        doc_vecs = self.model.document_encoder(batch_embs.to(model_dtype), batch_mask, query_vec_expanded.to(model_dtype))
        
        query_norm = F.normalize(query_vec.to(torch.float32), p=2, dim=1)
        doc_norms = F.normalize(doc_vecs.to(torch.float32), p=2, dim=1)
        
        scores = (query_norm @ doc_norms.T).squeeze().cpu().tolist()
        
        return scores if isinstance(scores, list) else [scores]

# ==============================================================================
# 5. ПРИЛОЖЕНИЕ FastAPI
# ==============================================================================
app = FastAPI(
    title="Hierarchical Semantic Ranking API",
    version="3.0-lora-final",
    description="API for ranking documents using the trained hierarchical model.",
    docs_url="/",
)

ml_engine: Optional[InferenceEngine] = None

@app.on_event("startup")
def startup_event():
    global ml_engine
    ml_engine = InferenceEngine()

@app.post("/encode-query", response_model=QueryResponse, tags=["Encoding"])
def handle_encode_query(req: QueryRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    query_vec = ml_engine.encode_query(req.text)
    return QueryResponse(vector=vector_to_base64(query_vec.squeeze()))

@app.post("/rank", response_model=RankingResponse, tags=["Ranking"])
def handle_rank(req: RankingRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    if not req.documents:
        raise HTTPException(status_code=400, detail="Documents list cannot be empty.")
    try:
        scores = ml_engine.rank_documents(req.query, req.documents)
        return RankingResponse(scores=scores)
    except (FileNotFoundError, PermissionError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print("\n" + "="*20 + " INTERNAL SERVER ERROR " + "="*20)
        traceback.print_exc()
        print("="*65 + "\n")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# ==============================================================================
# 6. ЗАПУСК СЕРВЕРА
# ==============================================================================
if __name__ == "__main__":
    # NLTK-загрузчик для токенизации предложений (нужно сделать один раз)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("INFO: Downloading NLTK 'punkt' model for sentence tokenization...")
        nltk.download('punkt')

    print(f"Starting API server on http://{CONFIG.API_HOST}:{CONFIG.API_PORT}")
    uvicorn.run(app, host=CONFIG.API_HOST, port=CONFIG.API_PORT)