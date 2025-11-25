import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === ML SETTINGS ===
# (тут твои настройки ключей и моделей оставляем как есть)
# Используем быстрые модели для разных задач
OLLAMA_MODEL = "gemma3:12b"  # Мощная модель для общего использования
OLLAMA_MODEL_FAST = "gemma3:12b"  # Мощная модель для переводов
OLLAMA_MODEL_CHAT = "gemma3:12b"  # Мощная модель для чата
OLLAMA_HOST = "http://localhost:11434"  # Локальный Ollama сервер
SENTENCE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_TOPICS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OpenRouter.ai настройки (бесплатный анонимный API)
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemma-2-2b-it:free"  # Бесплатная модель
OPENROUTER_HEADERS = {
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8000",  # Для анонимного доступа
    "X-Title": "Kotodex Research Explorer"
}

# Переводчик настройки
USE_GOOGLE_TRANSLATE = True  # Использовать Google Translate вместо LLM

# Google AI Studio (Gemini) настройки
GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")
GOOGLE_AI_STUDIO_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
GOOGLE_AI_STUDIO_MODEL = "gemini-flash-lite-latest"

# === PATHS (ИСПРАВЛЕНО) ===

# 1. Получаем папку, где лежит ЭТОТ файл (app/config.py -> папка app)
APP_DIR = Path(__file__).resolve().parent

# 2. Корень проекта (на уровень выше app)
BASE_DIR = APP_DIR.parent

# 3. Путь к чекпоинту (project/checkpoints/best_model.pt)
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pt"

# 4. Путь к статике (app/static) - теперь он строится от APP_DIR
STATIC_PATH = APP_DIR / "static"

ARXIV_MAX_RESULTS = 10