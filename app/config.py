import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === ML SETTINGS ===
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_MODEL_FAST = "gemma3:4b"
OLLAMA_MODEL_CHAT = "gemma3:12b"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SENTENCE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_TOPICS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === UniversalScorer (бывший ml-service) ===
QUERY_MODEL_NAME = SENTENCE_MODEL_NAME
SENTENCE_EMBEDDING_MODEL = SENTENCE_MODEL_NAME
LONGFORMER_MODEL = 'allenai/longformer-base-4096'

USE_QLORA_QUERY = True
QLORA_QUERY_R = 64
QLORA_QUERY_ALPHA = 128
QLORA_QUERY_DROPOUT = 0.05
QLORA_QUERY_TARGET_MODULES = [
    "query", "key", "value", "attention.output.dense",
    "intermediate.dense", "output.dense"
]

USE_LORA_DOC = True
LORA_DOC_R = 32
LORA_DOC_ALPHA = 64
LORA_DOC_DROPOUT = 0.05
LORA_DOC_TARGET_MODULES = ["query", "key", "value"]

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
USE_GRADIENT_CHECKPOINTING = False

CHECKPOINT_FILENAME_BEST = "best_model.pt"
CHECKPOINT_FILENAME_AUTOSAVE = "autosave.pt"

MAX_SENTENCES = 256


def _load_model_dims():
    """Подтягивает размерности моделей из HuggingFace-конфигов (как раньше делал ml-service)."""
    from transformers import AutoConfig
    try:
        sc = AutoConfig.from_pretrained(SENTENCE_EMBEDDING_MODEL)
        sentence_dim = sc.hidden_size

        lc = AutoConfig.from_pretrained(LONGFORMER_MODEL)
        longformer_dim = lc.hidden_size

        qc = AutoConfig.from_pretrained(QUERY_MODEL_NAME)
        query_input_dim = qc.hidden_size

        return sentence_dim, longformer_dim, query_input_dim
    except OSError as e:
        print(f"CRITICAL: Cannot load model configs: {e}")
        raise


SENTENCE_EMBEDDING_DIM, LONGFORMER_DIM, QUERY_MODEL_INPUT_DIM = _load_model_dims()
FINAL_QUERY_DIM = LONGFORMER_DIM
FINAL_DOC_DIM = LONGFORMER_DIM

# OpenRouter.ai
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemma-2-2b-it:free"
OPENROUTER_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
    "HTTP-Referer": "",
    "X-Title": "Kotodex Research Explorer"
}

# Переводчик
USE_GOOGLE_TRANSLATE = True

# Google AI Studio (Gemini)
GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")
GOOGLE_AI_STUDIO_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
GOOGLE_AI_STUDIO_MODEL = "gemini-2.0-flash-lite"

# === PATHS ===
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
CHECKPOINT_PATH = APP_DIR / "checkpoints" / "best_model.pt"
STATIC_PATH = APP_DIR / "static"

ARXIV_MAX_RESULTS = 50