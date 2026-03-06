"""
Конфиг ML-сервиса. Урезанная версия config4model.py —
только то, что нужно для инференса (не для обучения).
"""
import os
import torch
from pathlib import Path
from transformers import AutoConfig
from dotenv import load_dotenv

load_dotenv()

class _Config:
    def __init__(self):
        # ── Модели ────────────────────────────────────────────────────────────
        self.QUERY_MODEL_NAME        = 'BAAI/bge-small-en-v1.5'
        self.SENTENCE_EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
        self.LONGFORMER_MODEL        = 'allenai/longformer-base-4096'

        # ── Пути (в Docker всё в /app) ────────────────────────────────────────
        self.CHECKPOINT_DIR          = Path('/app/checkpoints')
        self.API_ALLOWED_DOCS_DIR    = Path('/app/arxiv_pdfs').resolve()

        # ── LoRA ──────────────────────────────────────────────────────────────
        self.USE_QLORA_QUERY         = True
        self.QLORA_QUERY_R           = 64
        self.QLORA_QUERY_ALPHA       = 128
        self.QLORA_QUERY_DROPOUT     = 0.05
        self.QLORA_QUERY_TARGET_MODULES = [
            "query", "key", "value", "attention.output.dense",
            "intermediate.dense", "output.dense"
        ]

        self.USE_LORA_DOC            = True
        self.LORA_DOC_R              = 32
        self.LORA_DOC_ALPHA          = 64
        self.LORA_DOC_DROPOUT        = 0.05
        self.LORA_DOC_TARGET_MODULES = ["query", "key", "value"]

        # ── Техническое ───────────────────────────────────────────────────────
        self.DEVICE                  = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.USE_BF16                = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.USE_GRADIENT_CHECKPOINTING = False  # инференс, не обучение

        # ── Чекпоинт ──────────────────────────────────────────────────────────
        self.CHECKPOINT_FILENAME_BEST      = "best_model.pt"
        self.CHECKPOINT_FILENAME_AUTOSAVE  = "autosave.pt"

        # ── API ───────────────────────────────────────────────────────────────
        self.API_SENTENCE_ENCODING_BATCH   = 64
        self.MAX_SENTENCES                 = 256
        self.MAX_TOPICS                    = 10

        self._load_dims()

    def _load_dims(self):
        print("INFO: Loading model dims from HuggingFace configs...")
        try:
            sc = AutoConfig.from_pretrained(self.SENTENCE_EMBEDDING_MODEL)
            self.SENTENCE_EMBEDDING_DIM = sc.hidden_size

            lc = AutoConfig.from_pretrained(self.LONGFORMER_MODEL)
            self.LONGFORMER_DIM = lc.hidden_size

            qc = AutoConfig.from_pretrained(self.QUERY_MODEL_NAME)
            self.QUERY_MODEL_INPUT_DIM = qc.hidden_size
            self.QUERY_MODEL_MAX_LEN   = getattr(qc, 'max_position_embeddings', 512)

            self.FINAL_QUERY_DIM = self.LONGFORMER_DIM
            self.FINAL_DOC_DIM   = self.LONGFORMER_DIM

            print(f"  sentence_dim={self.SENTENCE_EMBEDDING_DIM}, longformer_dim={self.LONGFORMER_DIM}")
        except OSError as e:
            print(f"CRITICAL: Cannot load model configs: {e}")
            exit(1)


CONFIG = _Config()