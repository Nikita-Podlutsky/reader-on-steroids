# ==============================================================================
# 0. ИМПОРТЫ
# ==============================================================================
import torch
from torch.utils.data import Dataset
import json
from transformers import AutoTokenizer
from pathlib import Path
# ==============================================================================
# 1. КОНФИГУРАЦИЯ
# ==============================================================================

from config import CONFIG
from sklearn.metrics.pairwise import cosine_similarity
# ==============================================================================
# 2.1 HIERARCHICAL DATASET
# ==============================================================================

class HierarchicalTripletDataset(Dataset):
    def __init__(self, metadata_file: str):
        """
        Инициализатор. Теперь ему нужен только путь к файлу метаданных,
        так как пути к эмбеддингам содержатся внутри него.
        """
        # Загрузка метаданных, которые теперь содержат ПУТИ к .pt файлам
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Токенизатор для запросов (BGE-M3 или его аналог из конфига)
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG.QUERY_MODEL_NAME,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def __len__(self):
        """Возвращает общее количество обучающих примеров."""
        # return len(self.metadata)
        return 1000
    
    def __getitem__(self, idx: int) -> dict:
        """
        Получает один элемент данных (триплет) по его индексу.
        """
        item_meta = self.metadata[idx]
        
        # 1. Токенизация запроса (query)
        tokenized_query = self.tokenizer(
            item_meta['query'],
            max_length=CONFIG.QUERY_MODEL_MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )
        
        query_data = {
            'input_ids': tokenized_query['input_ids'].squeeze(0),
            'attention_mask': tokenized_query['attention_mask'].squeeze(0),
        }
        # Добавляем token_type_ids, если они были возвращены токенизатором
        if 'token_type_ids' in tokenized_query:
            query_data['token_type_ids'] = tokenized_query['token_type_ids'].squeeze(0)
        
        # 2. ГЛАВНОЕ ИЗМЕНЕНИЕ: Загрузка эмбеддингов из .pt файлов по путям
        # Вместо обращения к HDF5, мы используем простую и надежную torch.load()
        # try:
        anchor_embeddings = torch.load(item_meta['anchor_path'])
        positive_embeddings = torch.load(item_meta['positive_path'])
        negative_embeddings = torch.load(item_meta['negative_path'])
        # except:
        #     print("Файла не существует!")
        #     return self.__getitem__(idx+1)
        # 3. Создание масок внимания для предложений
        # Маска создается на основе реального размера загруженного тензора
        anchor_mask = torch.ones(anchor_embeddings.shape[0], dtype=torch.long)
        positive_mask = torch.ones(positive_embeddings.shape[0], dtype=torch.long)
        negative_mask = torch.ones(negative_embeddings.shape[0], dtype=torch.long)
        
        return {
            'query': query_data,
            'anchor': {'embeddings': anchor_embeddings, 'attention_mask': anchor_mask},
            'positive': {'embeddings': positive_embeddings, 'attention_mask': positive_mask},
            'negative': {'embeddings': negative_embeddings, 'attention_mask': negative_mask}
        }





# ==============================================================================
# 2.2 HIERARCHICAL DATASET (Переписанная версия)
# ==============================================================================

POOL_FILE = Path("doc_pool.pt")          # 10k×384


class HierarchicalTripletDataset2(Dataset):
    """
    Online Hard Negative Mining прямо в __getitem__:
    - anchor & positive берём из файла метаданных (как раньше)
    - negative = самый далёкий из 500 случайных кандидатов (argmin)
    """
    def __init__(self,
                 metadata_file: str,
                 pool_file: str = str(POOL_FILE),
                 pool_size: int = None,
                 num_cand: int = 5000,
                 device: str = "cpu"):
        super().__init__()

        # 1. метаданные (triplets пути)
        with open(metadata_file, encoding="utf-8") as fp:
            self.metadata = json.load(fp)

        # 2. предкодированный пул документов (mean по предложениям)
        self.doc_pool = torch.load(pool_file).to(device)        # N×384
        self.pool_size = len(self.doc_pool) if pool_size is None else pool_size
        self.num_cand = num_cand
        self.device = device

        # 3. токенизатор запросов (BGE-small)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG.QUERY_MODEL_NAME, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ---------------------------------------------------------------------------------
    def __len__(self):
        return len(self.metadata)

    # ---------------------------------------------------------------------------------
    def __getitem__(self, idx: int):
        """Возвращает dict с query, anchor, positive, negative (все тензоры)"""
        item = self.metadata[idx]

        # 1. query (токенизируем на месте)
        query_text = item["query"]
        tok = self.tokenizer(
            query_text,
            max_length=CONFIG.QUERY_MODEL_MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        query = {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in tok:
            query["token_type_ids"] = tok["token_type_ids"].squeeze(0)

        # 2. anchor & positive (загружаем .pt файлы)
        anchor = torch.load(item["anchor_path"]).to(self.device)
        positive = torch.load(item["positive_path"]).to(self.device)

        # 3. ONLINE HARD NEGATIVE MINING
        negative = self._get_hard_negative(anchor)

        # 4. маски внимания (просто 1, без паддинга)
        anchor_mask = torch.ones(anchor.shape[0], dtype=torch.long, device=self.device)
        pos_mask = torch.ones(positive.shape[0], dtype=torch.long, device=self.device)
        neg_mask = torch.ones(negative.shape[0], dtype=torch.long, device=self.device)

        return {
            "query": query,
            "anchor": {"embeddings": anchor, "attention_mask": anchor_mask},
            "positive": {"embeddings": positive, "attention_mask": pos_mask},
            "negative": {"embeddings": negative, "attention_mask": neg_mask},
        }

    # ---------------------------------------------------------------------------------
    @torch.no_grad()
    def _get_hard_negative(self, anchor: torch.Tensor) -> torch.Tensor:
        """
        Выбираем самого ДАЛЁКОГO (argmin cos_sim) из num_cand случайных документов.
        anchor: S×384  (mean pooling делаем на лету)
        return: N×384 тензор предложений hardest-документа
        """
        # mean-представление anchor (1×384)
        anc_mean = anchor.mean(0, keepdim=True)                       # 1×384

        # 500 случайных индексов (без повторов)
        cand_idx = torch.randperm(self.pool_size, device=self.device)[:self.num_cand]
        cand_embs = self.doc_pool[cand_idx]                           # 500×384

        # косинусное сходство (чем МЕНЬШЕ - тем ДАЛЬШЕ)
        scores = cosine_similarity(anc_mean.cpu().numpy(), cand_embs.cpu().numpy())[0]  # 500
        hardest_abs_idx = cand_idx[scores.argmin()].item()            # индекс в pool

        # загружаем сам .pt файл hardest-документа
        hardest_meta = self.metadata[hardest_abs_idx]
        negative = torch.load(hardest_meta["anchor_path"]).to(self.device)
        return negative

# ==============================================================================
# 3. COLLATE FUNCTION (Остается БЕЗ ИЗМЕНЕНИЙ)
# ==============================================================================

def collate_for_hierarchical(batch: list) -> dict:
    """
    Динамический паддинг до максимальной длины в батче.
    Эта функция НЕ МЕНЯЕТСЯ, так как она работает с уже загруженными в память
    тензорами и ей не важно, пришли они из HDF5 или из .pt файлов.
    """
    # Обработка запросов
    queries = {
        'input_ids': torch.stack([item['query']['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['query']['attention_mask'] for item in batch])
    }
    # Добавляем token_type_ids, если они есть
    if 'token_type_ids' in batch[0]['query']:
        queries['token_type_ids'] = torch.stack([item['query']['token_type_ids'] for item in batch])
    
    # Обработка документов с динамическим паддингом
    padded_docs = {}
    for doc_type in ['anchor', 'positive', 'negative']:
        # Находим максимальное количество предложений в текущем батче
        max_sentences = max(item[doc_type]['embeddings'].shape[0] for item in batch)
        # Ограничиваем максимальным значением из конфига
        max_sentences = min(max_sentences, CONFIG.MAX_SENTENCES)
        
        padded_embeddings = []
        padded_masks = []
        
        for item in batch:
            # Обрезаем слишком длинные документы уже здесь, до паддинга
            embeddings = item[doc_type]['embeddings'][:CONFIG.MAX_SENTENCES]
            mask = item[doc_type]['attention_mask'][:CONFIG.MAX_SENTENCES]
            
            current_len = embeddings.shape[0]
            pad_len = max_sentences - current_len
            
            if pad_len > 0:
                # Паддинг (добавление нулей) до max_sentences
                padded_emb = torch.nn.functional.pad(
                    embeddings, (0, 0, 0, pad_len), "constant", 0
                )
                padded_mask = torch.nn.functional.pad(mask, (0, pad_len), "constant", 0)
            else:
                padded_emb = embeddings
                padded_mask = mask
            
            padded_embeddings.append(padded_emb)
            padded_masks.append(padded_mask)
        
        padded_docs[doc_type] = {
            'embeddings': torch.stack(padded_embeddings),
            'attention_mask': torch.stack(padded_masks)
        }
    
    return {
        'query': queries,
        **padded_docs
    }