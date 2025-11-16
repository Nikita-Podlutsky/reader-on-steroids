# ==============================================================================
#
#                    Единый Файл Конфигурации Проекта
#
# ==============================================================================

import torch
from pathlib import Path
from transformers import AutoConfig
import huggingface_hub
from dotenv import load_dotenv
import os

load_dotenv()


debug = os.getenv('DEBUG', 'False')
hf_key = os.environ['HF_KEY']

huggingface_hub.login(token=hf_key)

class _Config:
    def __init__(self):
        """
        При создании объекта конфига мы сначала задаем основные, "ручные" параметры,
        а затем вызываем метод для автоматической загрузки зависимых параметров.
        """
        # =============================================================================
        # 1. ОСНОВНЫЕ ПАРАМЕТРЫ
        # =============================================================================
        
        # --- Модели  ---
        self.QUERY_MODEL_NAME = 'BAAI/bge-small-en-v1.5'
        self.SENTENCE_EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
        self.LONGFORMER_MODEL = 'allenai/longformer-base-4096'

        # --- Пути ---
        self.BASE_DATA_DIR = Path('triplets_data')
        self.PDF_DIR = Path('arxiv_pdfs')
        self.CHECKPOINT_DIR = Path('checkpoints')
        self.METADATA_FILE = 'arxiv-metadata-oai-snapshot.json'
        self.ENRICHED_PAPERS_CACHE_FILE = self.BASE_DATA_DIR / 'enriched_papers_cache.json'
        self.OUTPUT_TRIPLETS_FILE = self.BASE_DATA_DIR / 'conditional_triplets.json'
        self.EMBEDDINGS_PT_DIR = self.BASE_DATA_DIR / 'sentence_embeddings_pt'
        self.FINAL_METADATA_FILE = self.BASE_DATA_DIR / 'metadata_for_hierarchical.json'
        
        # --- Параметры предобработки ---
        self.CATEGORIES_CONFIG = {'cs.LG': 100, 'cs.CV': 400, 'cs.CL': 500, 'hep-ph': 100, 'quant-ph': 100}
        self.YEAR_SINCE = 2020
        self.NUM_TRIPLETS_TO_GENERATE = 20000
        self.OLLAMA_MODEL = "gemma3:latest"
        self.OLLAMA_API_URL = "http://localhost:11434/api/generate"
        self.OLLAMA_REQUEST_TIMEOUT = 45
        self.MIN_SENTENCE_LENGTH = 15
        self.PDF_DOWNLOAD_WORKERS = 40
        self.TEXT_EXTRACTION_WORKERS = 8
        self.LLM_WORKERS = 4
        self.HARD_NEGATIVE_SEARCH_ATTEMPTS = 50

        # --- Параметры обучения ---
        self.BATCH_SIZE = 2
        self.ACCUMULATION_STEPS = 1
        self.LEARNING_RATE = 2e-4
        self.WEIGHT_DECAY = 0.01
        self.NUM_EPOCHS = 10
        self.WARMUP_STEPS = 100
        self.MARGIN = 0.5
        self.SAVE_EVERY_EPOCH = 1
        
        # --- QLoRA для QueryEncoder ---
        self.USE_QLORA_QUERY = True # Переименуем для ясности
        self.QLORA_QUERY_R = 64
        self.QLORA_QUERY_ALPHA = 128
        self.QLORA_QUERY_DROPOUT = 0.05
        self.QLORA_QUERY_TARGET_MODULES = [
            "query", "key", "value", "attention.output.dense",
            "intermediate.dense", "output.dense"
        ]

        # ===> НОВАЯ СЕКЦИЯ: LoRA для DocumentEncoder <===
        self.USE_LORA_DOC = True # Включаем/выключаем LoRA для Longformer
        # QLoRA (4-bit) для Longformer'а может быть нестабильной,
        # поэтому будем использовать обычную LoRA (16-bit).
        self.LORA_DOC_R = 32  # Ранг можно сделать поменьше, т.к. модель проще
        self.LORA_DOC_ALPHA = 64
        self.LORA_DOC_DROPOUT = 0.05
        # Целевые модули для Longformer'а обычно те же, что и для BERT-подобных моделей
        self.LORA_DOC_TARGET_MODULES = ["query", "key", "value"]

        # --- Технические настройки ---
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.USE_GRADIENT_CHECKPOINTING = True



        # ===> НОВАЯ СЕКЦИЯ: "МЕГА" СИСТЕМА ЧЕКПОИНТОВ <===
        self.RESUME_TRAINING = False  # Главный рубильник: пытаться ли возобновить обучение
        self.CHECKPOINT_FILENAME_EPOCH = "checkpoint_epoch_{epoch}.pt"
        self.CHECKPOINT_FILENAME_BEST = "best_model.pt"
        self.CHECKPOINT_FILENAME_AUTOSAVE = "autosave.pt"
        self.AUTOSAVE_EVERY_N_STEPS = 50 # Как часто делать "автосохранение". 0 - отключить.

        # API
        self.API_PORT = 8000
        self.API_HOST = "127.0.0.1"
        self.API_ALLOWED_DOCS_DIR = Path(".").resolve()
        self.API_SENTENCE_ENCODING_BATCH = 64



        # =============================================================================
        # 2. ВЫЗОВ ДИНАМИЧЕСКОЙ ЗАГРУЗКИ
        # =============================================================================
        # Инициализируем динамические параметры
        self._load_dynamic_properties()

    def _load_dynamic_properties(self):
        """
        Загружает конфигурации моделей с Hugging Face для автоматического
        определения размерностей. Это предотвращает ошибки при смене моделей.
        """
        print("INFO: Loading dynamic properties from model configs...")
        try:
            # --- Загрузка конфига для Sentence Encoder ---
            sentence_model_config = AutoConfig.from_pretrained(self.SENTENCE_EMBEDDING_MODEL)
            self.SENTENCE_EMBEDDING_DIM = sentence_model_config.hidden_size
            
            # --- Загрузка конфига для Longformer ---
            longformer_config = AutoConfig.from_pretrained(self.LONGFORMER_MODEL)
            self.LONGFORMER_DIM = longformer_config.hidden_size
            
            # --- Загрузка конфига для Query Encoder ---
            query_model_config = AutoConfig.from_pretrained(self.QUERY_MODEL_NAME)
            self.QUERY_MODEL_INPUT_DIM = query_model_config.hidden_size
            
            # Получаем максимальную поддерживаемую длину из конфига модели
            # Если в конфиге нет max_position_embeddings, используем стандартное значение 512
            self.QUERY_MODEL_MAX_LEN = getattr(query_model_config, 'max_position_embeddings', 512)
            
            print(f"INFO: -> Sentence Embedding Dim: {self.SENTENCE_EMBEDDING_DIM} (from '{self.SENTENCE_EMBEDDING_MODEL}')")
            print(f"INFO: -> Longformer Dim: {self.LONGFORMER_DIM} (from '{self.LONGFORMER_MODEL}')")
            print(f"INFO: -> Query Model Input Dim: {self.QUERY_MODEL_INPUT_DIM} (from '{self.QUERY_MODEL_NAME}')")
            print(f"INFO: -> Query Model Max Length: {self.QUERY_MODEL_MAX_LEN} (from model config)")

        except OSError as e:
            print(f"CRITICAL: Could not fetch model configs from Hugging Face Hub: {e}")
            print("CRITICAL: Please check your internet connection and model names.")
            # В случае ошибки, можно либо завершить работу, либо установить значения по умолчанию
            # Для надежности лучше завершать, чтобы избежать скрытых ошибок.
            exit(1)

        # =============================================================================
        # 3. ЗАВИСИМЫЕ ПАРАМЕТРЫ (теперь они тоже "умные")
        # =============================================================================
        # Эти параметры зависят от других, поэтому мы определяем их в конце.
        
        # Максимальная длина последовательности для Query Encoder
        self.QUERY_MODEL_MAX_LEN = 512
        
        # Максимальное количество предложений для Document Encoder
        self.MAX_SENTENCES = 1024
        
        # Размерность выходов Query и Document энкодеров. 
        # Проектируем так, чтобы они всегда были равны размерности Longformer для совместимости.
        self.FINAL_QUERY_DIM = self.LONGFORMER_DIM
        self.FINAL_DOC_DIM = self.LONGFORMER_DIM
        
        # Батч для кодирования предложений при предобработке
        self.SENTENCE_EMBEDDING_BATCH_SIZE = 128


# ==============================================================================
# СОЗДАЕМ ЕДИНСТВЕННЫЙ ЭКЗЕМПЛЯР КОНФИГА
# В других файлах вы будете импортировать именно `CONFIG`, а не `_Config`
# ==============================================================================
CONFIG = _Config()