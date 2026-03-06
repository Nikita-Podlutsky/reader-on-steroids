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
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pt"
STATIC_PATH = APP_DIR / "static"

ARXIV_MAX_RESULTS = 50