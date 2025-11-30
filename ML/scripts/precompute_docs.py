# ==============================================================================
#
#                   Предварительное Вычисление Эмбеддингов Документов
#
# ==============================================================================
#
#   Что делает:
#   1. Загружает все .pt файлы с эмбеддингами предложений документов.
#   2. Усредняет эмбеддинги по предложениям для каждого документа.
#   3. Создает файл doc_pool.pt с векторами размерности N×384.
#   4. Ускоряет онлайн поиск сложных негативов в HierarchicalTripletDataset2.
#   5. Выполняется один раз после prepare_data.py и перед train.py.
#
#   Запуск:
#   > python precompute_docs.py
#
#   Требует triplets_data/emebddings_pt/ и metadata_for_hierarchical.json
#
# ==============================================================================

#!/usr/bin/env python3
import json, torch, nltk
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import CONFIG
from data_prepair import get_hashed_path

device = "cuda"
model = SentenceTransformer(CONFIG.SENTENCE_EMBEDDING_MODEL, device=device)

meta = json.load(open(CONFIG.FINAL_METADATA_FILE))
pool = []
for d in meta:
    pt_path = get_hashed_path(CONFIG.EMBEDDINGS_PT_DIR, Path(d["anchor_path"]).stem)
    embs = torch.load(pt_path)           # S×384
    pool.append(embs.mean(0))            # 384
torch.save(torch.stack(pool), "triplets_data/doc_pool.pt")
print(f"Сохранено {len(pool)} док-эмбеддингов в triplets_data/doc_pool.pt")