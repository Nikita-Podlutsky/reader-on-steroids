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

# === PATHS (ИСПРАВЛЕНО) ===

# 1. Получаем папку, где лежит ЭТОТ файл (app/config.py -> папка app)
APP_DIR = Path(__file__).resolve().parent

# 2. Корень проекта (на уровень выше app)
BASE_DIR = APP_DIR.parent

# 3. Путь к чекпоинту (project/checkpoints/best_model.pt)
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pt"

# 4. Путь к статике (app/static) - теперь он строится от APP_DIR
STATIC_PATH = APP_DIR / "static"

ARXIV_MAX_RESULTS = 50