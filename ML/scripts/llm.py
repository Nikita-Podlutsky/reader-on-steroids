import os, time, random
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

PRIORITY = os.getenv("LLM_PRIORITY", "ollama,gemini,openrouter,moonshot,ollama,hf").split(",")

ENDPOINTS = {
    "gemini":  ("https://generativelanguage.googleapis.com/v1beta/openai/", os.getenv("GEMINI_API_KEY")),
    "openrouter": ("https://openrouter.ai/api/v1", os.getenv("OPENROUTER_API_KEY")),
    "moonshot": ("https://api.moonshot.ai/v1", os.getenv("MOONSHOT_API_KEY")),
    "ollama":  (os.getenv("OLLAMA_HOST", "http://localhost:11434/v1"), "ollama"),
    "hf":      ("https://api-inference.huggingface.co/models/", os.getenv("HF_API_KEY"))
}

MODELS = {
    "gemini":  "gemini-2.0-flash-exp",
    "openrouter": "google/gemini-2.0-flash-exp:free",
    "moonshot": "kimi-v1",
    "ollama":  "gemma3:latest",
    "hf":      "microsoft/DialoGPT-medium"
}

class LLM:
    def __init__(self):
        self.client = None
        self.name = None
        self._rotate()

    def _rotate(self, skip=set()):
        for p in PRIORITY:
            print(p)
            if p in skip: continue
            base, key = ENDPOINTS[p]
            if not key: continue
            try:
                self.client = OpenAI(api_key=key, base_url=base)
                self.name = p
                return
            except Exception: continue
        raise RuntimeError("Нет рабочих LLM")

    def chat(self, messages, **kw):
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(
                    model=MODELS[self.name], messages=messages, **kw
                ).choices[0].message.content
            except Exception as e:
                time.sleep(2 ** attempt + random.random())
                self._rotate(skip={self.name})
        raise RuntimeError("Все модели отказали")

