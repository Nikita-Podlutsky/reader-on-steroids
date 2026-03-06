import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import ollama
import asyncio
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
import json
import hashlib
import aiohttp
from google import genai
from google.genai import types
from ollama import Client

from config import (
    DEVICE, SENTENCE_MODEL_NAME, OLLAMA_MODEL, OLLAMA_MODEL_FAST, OLLAMA_MODEL_CHAT, 
    OLLAMA_HOST, CHECKPOINT_PATH, BASE_DIR, OPENROUTER_API_URL, OPENROUTER_MODEL, 
    OPENROUTER_HEADERS, USE_GOOGLE_TRANSLATE, GOOGLE_AI_STUDIO_API_KEY, 
    GOOGLE_AI_STUDIO_API_URL, GOOGLE_AI_STUDIO_MODEL
)

# Пытаемся импортировать твои файлы из корня проекта

try:
    from models import UniversalScorer
    from checkpoint_utils import load_checkpoint
    HAS_CUSTOM_MODEL = True
except ImportError:
    print("WARNING: 'models.py' or 'checkpoint_utils.py' not found. Using fallback mode.")
    HAS_CUSTOM_MODEL = False


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

def ensure_ollama_model():
    print(f"🤖 Checking Ollama connection at {OLLAMA_HOST}...")
    client = Client(host=OLLAMA_HOST)
    
    try:
        # 1. Проверяем, есть ли модель
        models_list = client.list()
        # ollama library возвращает объекты, нужно правильно проверить
        models = [m['name'] for m in models_list.get('models', [])]
        
        # У Ollama имена могут быть с тегом 'latest', проверяем гибко
        found = any(OLLAMA_MODEL in m for m in models)
        
        if not found:
            print(f"📥 Model {OLLAMA_MODEL} not found inside Docker. Pulling now... (This may take time)")
            # Это синхронный вызов, он заблокирует старт, пока не скачает
            # Для первого запуска это нормально
            client.pull(OLLAMA_MODEL)
            print(f"✅ Model {OLLAMA_MODEL} downloaded successfully!")
        else:
            print(f"✅ Model {OLLAMA_MODEL} is ready.")
            
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to Ollama inside Docker: {e}")


ensure_ollama_model()
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
        
        # 3. Инициализация кэша переводов
        self.translation_cache_path = BASE_DIR / "translations_cache.json"
        self.translation_cache = self._load_translation_cache()
        
        # 4. Настройка Ollama клиента (обход прокси)
        self._setup_ollama_client()

    def _load_translation_cache(self):
        """Загружает кэш переводов из JSON"""
        if self.translation_cache_path.exists():
            try:
                with open(self.translation_cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading translation cache: {e}")
        return {}
    
    def _save_translation_cache(self):
        """Сохраняет кэш переводов в JSON"""
        try:
            with open(self.translation_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving translation cache: {e}")
    
    def _get_text_hash(self, text):
        """Создает хэш текста для использования как ключ"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _setup_ollama_client(self):
        """Настраивает Ollama клиент для обхода прокси"""
        # Отключаем прокси для Ollama
        os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
        os.environ['no_proxy'] = 'localhost,127.0.0.1'
        
        # Настраиваем Ollama клиент с явным указанием хоста
        try:
            # Импортируем Client из ollama
            from ollama import Client
            self.ollama_client = Client(host=OLLAMA_HOST)
            print(f"INFO: Ollama client configured for {OLLAMA_HOST}")
        except Exception as e:
            print(f"WARNING: Could not configure Ollama client: {e}")
            print("INFO: Using default ollama client")
            self.ollama_client = None

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

    def chat_ollama_sync(self, msgs, model=None):
        """Синхронный чат с Ollama"""
        try:
            model = model or OLLAMA_MODEL
            print(f"DEBUG: Using model {model} for chat")  # Логируем модель
            
            # Используем настроенный клиент или создаем новый с явным указанием хоста
            if hasattr(self, 'ollama_client') and self.ollama_client:
                response = self.ollama_client.chat(model=model, messages=msgs)
            else:
                # Используем глобальный клиент с явным указанием хоста
                import os
                # Временно отключаем прокси
                old_no_proxy = os.environ.get('NO_PROXY', '')
                old_no_proxy_lower = os.environ.get('no_proxy', '')
                os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
                os.environ['no_proxy'] = 'localhost,127.0.0.1'
                try:
                    # Используем Client с явным указанием хоста
                    from ollama import Client
                    client = Client(host=OLLAMA_HOST)
                    response = client.chat(model=model, messages=msgs)
                finally:
                    # Восстанавливаем старые значения
                    if old_no_proxy:
                        os.environ['NO_PROXY'] = old_no_proxy
                    if old_no_proxy_lower:
                        os.environ['no_proxy'] = old_no_proxy_lower
            
            if 'message' in response and 'content' in response['message']:
                return response['message']['content']
            else:
                print(f"WARNING: Unexpected response format: {response}")
                return None
        except Exception as e:
            print(f"Ollama error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def chat_openrouter(self, msgs, model=None):
        """Асинхронный чат через OpenRouter.ai (бесплатный анонимный API)"""
        try:
            model = model or OPENROUTER_MODEL
            print(f"DEBUG: Using OpenRouter model {model} for chat")
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model,
                    "messages": msgs,
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
                
                async with session.post(
                    OPENROUTER_API_URL,
                    json=payload,
                    headers=OPENROUTER_HEADERS,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'choices' in data and len(data['choices']) > 0:
                            return data['choices'][0]['message']['content']
                        else:
                            print(f"Unexpected OpenRouter response: {data}")
                            return None
                    else:
                        error_text = await response.text()
                        print(f"OpenRouter error {response.status}: {error_text}")
                        return None
        except Exception as e:
            print(f"OpenRouter error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def chat_google_ai_studio(self, msgs, model=None):
        """Асинхронный чат через Google AI Studio (Gemini API) с использованием SDK"""
        try:
            if not GOOGLE_AI_STUDIO_API_KEY:
                print("ERROR: GOOGLE_AI_STUDIO_API_KEY not set in environment")
                return None
            
            model = model or GOOGLE_AI_STUDIO_MODEL
            print(f"DEBUG: Using Google AI Studio model {model} for chat")
            
            # Инициализация клиента
            client = genai.Client(api_key=GOOGLE_AI_STUDIO_API_KEY)

            # Преобразуем сообщения в формат, ожидаемый SDK
            formatted_contents = []
            for msg in msgs:
                print(f"DEBUG: Google AI Studio message: {msg}")
                role = "user" if msg["role"] == "user" else "model"
                formatted_contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])]
                    )
                )

            # Конфигурация генерации
            config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=1000
            )

            # Асинхронный вызов через свойство .aio
            response = await client.aio.models.generate_content(
                model=model,
                contents=formatted_contents,
                config=config
            )

            # Возвращаем текст
            return response.text

        except Exception as e:
            print(f"Google AI Studio error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def translate_google(self, text):
        """Перевод через Google Translate API (бесплатный)"""
        try:
            # Используем бесплатный API Google Translate через googletrans
            try:
                from googletrans import Translator
                import inspect  # Добавляем для проверки типа функции
                
                translator = Translator()
                
                # Проверяем, является ли метод асинхронным (googletrans==4.0.0rc1+)
                if inspect.iscoroutinefunction(translator.translate):
                    result = await translator.translate(text, dest='ru')
                else:
                    # Для старых синхронных версий (googletrans==3.0.0)
                    result = await asyncio.to_thread(translator.translate, text, dest='ru')
                
                return result.text
                
            except ImportError:
                # Если googletrans не установлен, используем альтернативный метод
                # Используем бесплатный веб-API Google Translate
                import urllib.parse
                import urllib.request
                
                url = "https://translate.googleapis.com/translate_a/single"
                params = {
                    'client': 'gtx',
                    'sl': 'en',
                    'tl': 'ru',
                    'dt': 't',
                    'q': text[:5000]  # Ограничение длины
                }
                
                query_string = urllib.parse.urlencode(params)
                full_url = f"{url}?{query_string}"
                
                def _translate():
                    with urllib.request.urlopen(full_url, timeout=10) as response:
                        data = json.loads(response.read().decode('utf-8'))
                        if data and len(data) > 0 and len(data[0]) > 0:
                            return ''.join([item[0] for item in data[0] if item[0]])
                        return text
                
                return await asyncio.to_thread(_translate)
        except Exception as e:
            print(f"Google Translate error: {e}")
            # Fallback - возвращаем оригинал
            return text

    async def chat_ollama_stream(self, msgs, model=None):
        """Асинхронный генератор для streaming чата с Ollama"""
        import queue
        import threading
        model = model or OLLAMA_MODEL_CHAT
        q = queue.Queue()
        done = threading.Event()
        
        def _read_stream():
            """Читает stream в отдельном потоке и кладет в очередь"""
            try:
                import os
                # Временно отключаем прокси
                old_no_proxy = os.environ.get('NO_PROXY', '')
                old_no_proxy_lower = os.environ.get('no_proxy', '')
                os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
                os.environ['no_proxy'] = 'localhost,127.0.0.1'
                try:
                    # Используем Client с явным указанием хоста
                    from ollama import Client
                    client = Client(host=OLLAMA_HOST)
                    stream = client.chat(model=model, messages=msgs, stream=True)
                    for chunk in stream:
                        if 'message' in chunk and 'content' in chunk['message']:
                            content = chunk['message']['content']
                            if content:
                                q.put(content)
                    q.put(None)  # Сигнал завершения
                finally:
                    # Восстанавливаем старые значения
                    if old_no_proxy:
                        os.environ['NO_PROXY'] = old_no_proxy
                    if old_no_proxy_lower:
                        os.environ['no_proxy'] = old_no_proxy_lower
            except Exception as e:
                print(f"Stream read error: {e}")
                import traceback
                traceback.print_exc()
                q.put(None)
            finally:
                done.set()
        
        # Запускаем чтение stream в отдельном потоке
        thread = threading.Thread(target=_read_stream, daemon=True)
        thread.start()
        
        # Читаем из очереди и отдаем chunks
        while True:
            try:
                # Используем timeout для неблокирующего чтения
                chunk = q.get(timeout=0.1)
                if chunk is None:
                    break
                yield chunk
            except queue.Empty:
                if done.is_set():
                    # Проверяем, не осталось ли что-то в очереди
                    try:
                        chunk = q.get_nowait()
                        if chunk is None:
                            break
                        yield chunk
                    except queue.Empty:
                        break
                await asyncio.sleep(0.01)  # Небольшая задержка перед следующей попыткой
            except Exception as e:
                print(f"Stream yield error: {e}")
                break

    async def translate_fast(self, text, provider="google"):
        """Быстрый перевод с кэшированием и выбором провайдера"""
        # Проверяем кэш
        text_hash = self._get_text_hash(text)
        if text_hash in self.translation_cache:
            return self.translation_cache[text_hash]
        
        try:
            translation = None
            
            # Выбираем провайдера
            if provider == "google" or (provider is None and USE_GOOGLE_TRANSLATE):
                # Используем Google Translate
                translation = await self.translate_google(text)
            else:
                # Используем Ollama (старый метод)
                prompt = f"""Переведи следующий текст на русский язык. Сохрани технические термины и формулы без изменений. Верни только перевод без пояснений.

Текст:
{text[:1500]}"""
                msgs = [{"role": "user", "content": prompt}]
                result = await asyncio.to_thread(self.chat_ollama_sync, msgs, OLLAMA_MODEL_FAST)
                if result:
                    # Очистка markdown
                    cleaned = result.strip()
                    if cleaned.startswith("```"):
                        lines = cleaned.split('\n')
                        if len(lines) > 2 and lines[-1] == "```":
                            cleaned = '\n'.join(lines[1:-1])
                    translation = cleaned.strip()
                    
                    # Убираем возможные префиксы
                    for prefix in ["Перевод:", "Translation:", "Переведенный текст:", "Translated text:"]:
                        if translation.startswith(prefix):
                            translation = translation[len(prefix):].strip()
            
            if translation:
                # Сохраняем в кэш
                self.translation_cache[text_hash] = translation
                self._save_translation_cache()
                return translation
            
            return text  # Fallback - возвращаем оригинал
        except Exception as e:
            print(f"Translation error: {e}")
            import traceback
            traceback.print_exc()
            return text

    async def name_clusters(self, clusters_data):
        """Оптимизированная генерация названий тем с быстрой моделью"""
        if not clusters_data: 
            return {}
        
        # Более простой и понятный промпт на русском
        prompt = f"""Дай короткие названия (1-3 слова) для {len(clusters_data)} групп научных статей.

Формат: каждая строка должна быть "ID: Название"

Пример:
0: Нейронные сети
1: Глубокое обучение
2: Этика ИИ

Группы:
"""
        for cid, titles in sorted(clusters_data.items()):
            if titles and len(titles) > 0:  # Проверка на пустой список
                prompt += f"{cid}: {titles[0][:40]}\n"
            else:
                prompt += f"{cid}: (пусто)\n"
        
        prompt += "\nОтветь ТОЛЬКО в формате 'ID: Название', по одной строке на группу. Без пояснений."
        
        msgs = [{"role": "user", "content": prompt}]
        res = await asyncio.to_thread(self.chat_ollama_sync, msgs, OLLAMA_MODEL_FAST)
        
        topic_map = {}
        if res:
            print(f"DEBUG: Ollama response for topics: {res[:200]}")  # Логируем для отладки
            for line in res.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                # Пропускаем строки с примерами и пояснениями
                if any(skip in line.lower() for skip in ['example', 'пример', 'format', 'формат', 'ответь', 'reply']):
                    continue
                # Более гибкий regex - ищем ID: Название
                m = re.search(r'^(\d+)\s*[:.-]\s*(.+?)(?:\s*$|\.|,|;|\n)', line)
                if m:
                    try:
                        topic_id = int(m.group(1))
                        name = m.group(2).strip().strip('"*').strip()
                        # Ограничиваем длину названия - берем только первые 3 слова
                        if name and topic_id in clusters_data:
                            words = name.split()[:3]
                            topic_map[topic_id] = ' '.join(words)
                    except Exception as e:
                        print(f"Error parsing topic name: {e}")
                        pass
        
        # Fallback - дефолтные имена
        for cid in clusters_data:
            if cid not in topic_map:
                topic_map[cid] = f"Topic {cid}"
        
        print(f"DEBUG: Final topic_map: {topic_map}")  # Логируем результат
        return topic_map