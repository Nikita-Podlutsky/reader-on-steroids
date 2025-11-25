import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse # <-- Добавили

from app.config import STATIC_PATH
from app.services.engine import HybridEngine
from app.arxiv_loader import ArxivLoader
from app.routers import endpoints

app = FastAPI(title="Kotodex Core API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные для хранения ресурсов
engine_instance = None
loader_instance = None

@app.on_event("startup")
async def startup_event():
    global engine_instance, loader_instance
    print("=== STARTING KOTODEX CORE ===")
    
    # 1. Инициализируем движок
    engine_instance = HybridEngine()
    
    # 2. Инициализируем лоадер
    loader_instance = ArxivLoader()
    
    # 3. Передаем их в роутер
    endpoints.init_globals(engine_instance, loader_instance)

# Подключаем API роуты
app.include_router(endpoints.router)

# === ИСПРАВЛЕНИЕ СТАТИКИ ===

# 1. Явный роут для главной страницы
@app.get("/")
async def read_root():
    index_file = STATIC_PATH / "index.html"
    
    # ДЕБАГ: Пишем в консоль, где ищем файл
    if not index_file.exists():
        print(f"❌ ОШИБКА: Файл не найден по пути: {index_file.absolute()}")
        return {"error": f"index.html not found at: {index_file.absolute()}"}
    
    return FileResponse(str(index_file))

# 2. Монтируем assets для JS/CSS файлов
assets_path = STATIC_PATH / "assets"
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

# 3. Монтируем остальную статику (если есть другие файлы)
if STATIC_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)