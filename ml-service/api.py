"""
ML-сервис Kotodex
─────────────────
Эндпоинты:
  GET  /health          — проверка живости
  POST /encode-query    — вектор запроса
  POST /rank            — скоры релевантности документов
"""
import base64
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback

import fitz
import nltk
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from config import CONFIG
from models import UniversalScorer
from checkpoint_utils import find_latest_checkpoint, load_checkpoint

# ── Pydantic-схемы ────────────────────────────────────────────────────────────

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
        if not isinstance(data, dict):
            raise ValueError('Input must be a dictionary')
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

# ── Утилиты ───────────────────────────────────────────────────────────────────

def safe_resolve_path(user_path: str) -> Path:
    resolved = CONFIG.API_ALLOWED_DOCS_DIR.joinpath(user_path).resolve()
    if not str(resolved).startswith(str(CONFIG.API_ALLOWED_DOCS_DIR)):
        raise PermissionError(f"Access to path '{user_path}' is denied.")
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: '{user_path}'")
    return resolved

def extract_text(file_path: Path) -> str:
    try:
        if file_path.suffix.lower() == ".pdf":
            with fitz.open(file_path) as doc:
                text = " ".join(page.get_text() for page in doc)
        else:
            text = file_path.read_text(encoding="utf-8")
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print(f"Warning: could not extract text from {file_path}: {e}")
        return ""

def vec_to_b64(vec: torch.Tensor) -> Dict[str, Any]:
    arr = vec.detach().cpu().numpy().astype("float32")
    return {"shape": list(arr.shape), "dtype": "float32",
            "data": base64.b64encode(arr.tobytes()).decode("ascii")}

# ── Движок инференса ──────────────────────────────────────────────────────────

class InferenceEngine:
    def __init__(self):
        print(f"[ML] Device: {CONFIG.DEVICE}")
        self.device = torch.device(CONFIG.DEVICE)

        print("[ML] Loading model architecture…")
        self.model = UniversalScorer().to(self.device).eval()

        best = CONFIG.CHECKPOINT_DIR / CONFIG.CHECKPOINT_FILENAME_BEST
        ckpt = best if best.is_dir() else find_latest_checkpoint(CONFIG.CHECKPOINT_DIR)
        if ckpt:
            load_checkpoint(model=self.model, optimizer=None,
                            scheduler=None, scaler=None, path=ckpt)
        else:
            print("[ML] WARNING: No checkpoint found — using random weights!")

        print(f"[ML] Loading sentence encoder: {CONFIG.SENTENCE_EMBEDDING_MODEL}")
        self.sentence_enc = SentenceTransformer(CONFIG.SENTENCE_EMBEDDING_MODEL,
                                                device=self.device)

        print(f"[ML] Loading tokenizer: {CONFIG.QUERY_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.QUERY_MODEL_NAME,
                                                       trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[ML] Ready ✓")

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        tok = self.tokenizer(
            text, max_length=CONFIG.QUERY_MODEL_MAX_LEN,
            padding='max_length', truncation=True, return_tensors='pt'
        ).to(self.device)
        return self.model.query_encoder(
            input_ids=tok['input_ids'],
            attention_mask=tok['attention_mask'],
            token_type_ids=tok.get('token_type_ids')
        )

    @torch.no_grad()
    def rank_documents(self, query: str, documents: List[DocumentInput]) -> List[float]:
        if not documents:
            return []

        dtype = next(self.model.parameters()).dtype
        q_vec = self.encode_query(query)

        texts = []
        for doc in documents:
            if doc.content:
                texts.append(doc.content)
            else:
                try:
                    texts.append(extract_text(safe_resolve_path("arxiv_pdfs/" + doc.path)))
                except (FileNotFoundError, PermissionError) as e:
                    raise HTTPException(status_code=404, detail=str(e))

        sents_list = [nltk.sent_tokenize(t) for t in texts]
        embs_list = []
        for sents in sents_list:
            if sents:
                s = sents[:CONFIG.MAX_SENTENCES]
                embs_list.append(
                    self.sentence_enc.encode(s, convert_to_tensor=True,
                                             batch_size=CONFIG.API_SENTENCE_ENCODING_BATCH)
                )
            else:
                embs_list.append(torch.empty(0, CONFIG.SENTENCE_EMBEDDING_DIM,
                                             device=self.device))

        max_s = max(e.shape[0] for e in embs_list) if embs_list else 0
        if max_s == 0:
            return [0.0] * len(documents)

        n = len(documents)
        batch_e = torch.zeros(n, max_s, CONFIG.SENTENCE_EMBEDDING_DIM,
                              device=self.device, dtype=dtype)
        batch_m = torch.zeros(n, max_s, device=self.device, dtype=torch.long)

        for i, e in enumerate(embs_list):
            if e.shape[0] > 0:
                batch_e[i, :e.shape[0]] = e
                batch_m[i, :e.shape[0]] = 1

        q_exp = q_vec.expand(n, -1)
        doc_vecs = self.model.document_encoder(batch_e, batch_m, q_exp.to(dtype))

        q_n = F.normalize(q_vec.float(), p=2, dim=1)
        d_n = F.normalize(doc_vecs.float(), p=2, dim=1)
        scores = (q_n @ d_n.T).squeeze().cpu().tolist()
        return scores if isinstance(scores, list) else [scores]

# ── FastAPI-приложение ────────────────────────────────────────────────────────

app = FastAPI(title="Kotodex ML Service", version="1.0")
engine: Optional[InferenceEngine] = None

@app.on_event("startup")
def startup():
    global engine
    engine = InferenceEngine()

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}

@app.post("/encode-query", response_model=QueryResponse, tags=["ML"])
def encode_query(req: QueryRequest):
    if not req.text.strip():
        raise HTTPException(400, "Query text cannot be empty.")
    return QueryResponse(vector=vec_to_b64(engine.encode_query(req.text).squeeze()))

@app.post("/rank", response_model=RankingResponse, tags=["ML"])
def rank(req: RankingRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query text cannot be empty.")
    if not req.documents:
        raise HTTPException(400, "Documents list cannot be empty.")
    try:
        return RankingResponse(scores=engine.rank_documents(req.query, req.documents))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Internal error: {e}")

if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    uvicorn.run(app, host="0.0.0.0", port=8001)
