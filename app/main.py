import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import STATIC_PATH
from services.engine import HybridEngine
from arxiv_loader import ArxivLoader
from routers import endpoints

app = FastAPI(title="Kotodex API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check —──────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    engine = HybridEngine()
    loader = ArxivLoader()
    endpoints.init_globals(engine, loader)

app.include_router(endpoints.router)

# ── Статика ───────────────────────────────────────────────────────────────────
@app.get("/")
async def read_root():
    index = STATIC_PATH / "index.html"
    if not index.exists():
        return {"error": "Frontend not built. See README."}
    return FileResponse(str(index))

assets = STATIC_PATH / "assets"
if assets.exists():
    app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

if STATIC_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)