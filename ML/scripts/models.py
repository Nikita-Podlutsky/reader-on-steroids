# ==============================================================================
# 0. ИМПОРТЫ
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# ==============================================================================
# 1. КОНФИГУРАЦИЯ
# ==============================================================================
from config import CONFIG
# ==============================================================================
# 2. QUERY ENCODER (BGE-M3 + QLoRA) - СУПЕР-СТАБИЛЬНАЯ ВЕРСИЯ
# ==============================================================================

class QueryEncoderBGE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Главное: использовать float16 для всего
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.transformer = AutoModel.from_pretrained(
            CONFIG.QUERY_MODEL_NAME,
            quantization_config=quantization_config,
            dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if CONFIG.USE_QLORA_QUERY:
            self.transformer = prepare_model_for_kbit_training(self.transformer)
            
            lora_config = LoraConfig(
                r=CONFIG.QLORA_QUERY_R,
                lora_alpha=CONFIG.QLORA_QUERY_ALPHA,
                target_modules=CONFIG.QLORA_QUERY_TARGET_MODULES,
                lora_dropout=CONFIG.QLORA_QUERY_DROPOUT,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            
            self.transformer = get_peft_model(self.transformer, lora_config)
            print(f"✓ QLoRA: {self.transformer.get_nb_trainable_parameters()[0]:,} параметров")
        
        # ВСЕГДА на GPU перед projection
        if CONFIG.DEVICE == 'cuda':
            self.transformer = self.transformer.cuda()
        
        # Получаем актуальный dtype от модели
        with torch.no_grad():
            dummy = torch.randint(0, 100, (1, 10)).cuda() if CONFIG.DEVICE == 'cuda' else torch.randint(0, 100, (1, 10))
            dummy_mask = torch.ones(1, 10).cuda() if CONFIG.DEVICE == 'cuda' else torch.ones(1, 10)
            out = self.transformer(dummy, dummy_mask).last_hidden_state
            self.output_dtype = out.dtype  # Сохраняем как атрибут
        
        input_dim = CONFIG.QUERY_MODEL_INPUT_DIM
        intermediate_dim = input_dim // 2
        output_dim = CONFIG.FINAL_QUERY_DIM

        self.projection = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim, dtype=self.output_dtype),
            nn.LayerNorm(intermediate_dim, dtype=self.output_dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, output_dim, dtype=self.output_dtype)
        ).to(CONFIG.DEVICE)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None): # Добавляем token_type_ids
        device = next(self.transformer.parameters()).device
        
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device, non_blocking=True) # Переносим на GPU
        
        # Отключаем autocast
        # with torch.no_grad():
            # Передаем token_type_ids в модель
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids 
        )
        last_hidden_state = outputs.last_hidden_state
        
        # Mean pooling в float32 (более стабильно)
        expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * expanded_mask, 1)
        sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Приводим к типу projection
        return self.projection(pooled_output.to(self.output_dtype))


# ==============================================================================
# 3. DOCUMENT ENCODER (Версия с LoRA)
# ==============================================================================

class DocumentEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Определяем dtype для Longformer'а
        model_dtype = torch.bfloat16 if CONFIG.USE_BF16 else torch.float16

        # Загружаем Longformer
        self.longformer = AutoModel.from_pretrained(
            CONFIG.LONGFORMER_MODEL,
            torch_dtype=model_dtype  # Используем bf16/f16
        )

        # ===> ПРИМЕНЯЕМ LoRA К LONGFORMER <===
        if CONFIG.USE_LORA_DOC:
            print("INFO: Applying LoRA to DocumentEncoder (Longformer)...")
            # Создаем конфиг для LoRA
            lora_config_doc = LoraConfig(
                r=CONFIG.LORA_DOC_R,
                lora_alpha=CONFIG.LORA_DOC_ALPHA,
                target_modules=CONFIG.LORA_DOC_TARGET_MODULES,
                lora_dropout=CONFIG.LORA_DOC_DROPOUT,
                bias="none",
                # Указываем, что это не классификация, а извлечение признаков
                task_type=TaskType.FEATURE_EXTRACTION
            )
            # "Оборачиваем" нашу модель в LoRA
            self.longformer = get_peft_model(self.longformer, lora_config_doc)
            print("LoRA enabled for DocumentEncoder. Trainable parameters:")
            self.longformer.print_trainable_parameters()

        # Переносим на GPU
        if CONFIG.DEVICE == 'cuda':
            self.longformer = self.longformer.cuda()
        
        if CONFIG.USE_GRADIENT_CHECKPOINTING:
            # Важно: для моделей с PEFT нужно вызывать специальный метод
            self.longformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Остальная часть __init__ остается почти без изменений
        # Но теперь мы должны использовать model_dtype для всех слоев
        
        self.sentence_projection = nn.Linear(
            CONFIG.SENTENCE_EMBEDDING_DIM, 
            CONFIG.LONGFORMER_DIM,
            dtype=model_dtype
        ).to(CONFIG.DEVICE)
        
        self.query_projection = nn.Linear(
            CONFIG.FINAL_QUERY_DIM, 
            CONFIG.LONGFORMER_DIM,
            dtype=model_dtype
        ).to(CONFIG.DEVICE)
        
        self.attention = nn.Linear(
            CONFIG.LONGFORMER_DIM * 2, 
            1,
            dtype=model_dtype
        ).to(CONFIG.DEVICE)
        
        self.layer_norm = nn.LayerNorm(CONFIG.LONGFORMER_DIM, dtype=model_dtype).to(CONFIG.DEVICE)
        
        self.register_buffer(
            'global_attention_mask_template',
            torch.zeros(CONFIG.MAX_SENTENCES, dtype=torch.long, device=CONFIG.DEVICE),
            persistent=False
        )
        self.global_attention_mask_template[0] = 1
        
    def forward(self, embeddings, attention_mask, query_vec):
        # Forward pass остается практически без изменений,
        # но теперь нужно следить за типами данных (dtype)
        device = next(self.longformer.parameters()).device
        dtype = next(self.longformer.parameters()).dtype # Получаем dtype от модели (bf16/f16)
        
        # Перемещаем и приводим к нужному типу
        embeddings = embeddings.to(device, non_blocking=True, dtype=dtype)
        attention_mask = attention_mask.to(device, non_blocking=True)
        query_vec = query_vec.to(device, non_blocking=True, dtype=dtype)
        
        projected = self.sentence_projection(embeddings)
        global_attention_mask = attention_mask * self.global_attention_mask_template[:embeddings.shape[1]]
        
        # Вызов longformer'а остается таким же
        transformer_output = self.longformer(
            inputs_embeds=projected,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=False
        ).last_hidden_state
        
        # ... (остальная часть forward pass без изменений) ...
        query_projected = self.query_projection(query_vec)
        query_expanded = query_projected.unsqueeze(1).expand(-1, embeddings.shape[1], -1)
        combined = torch.cat([transformer_output, query_expanded], dim=-1)
        energy = self.attention(combined).squeeze(-1)
        energy_masked = torch.where(attention_mask == 0, -1e9, energy)
        attention_weights = F.softmax(energy_masked, dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)
        
        return self.layer_norm(context_vector)

# ==============================================================================
# 4. UNIVERSAL SCORER - ЦЕНТРАЛЬНЫЙ КОНТРОЛЬ
# ==============================================================================

class UniversalScorer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.query_encoder = QueryEncoderBGE()
        self.document_encoder = DocumentEncoder()
        
        assert CONFIG.FINAL_QUERY_DIM == CONFIG.LONGFORMER_DIM
        
        # Однократный перенос всей модели
        if CONFIG.DEVICE == 'cuda':
            self.cuda()
        
    def forward(self, batch):
        # Явный девайс-фикс
        device = next(self.parameters()).device
        
        for key in batch:
            if key == 'query':
                batch[key]['input_ids'] = batch[key]['input_ids'].to(device, non_blocking=True)
                batch[key]['attention_mask'] = batch[key]['attention_mask'].to(device, non_blocking=True)
            else:
                batch[key]['embeddings'] = batch[key]['embeddings'].to(device, non_blocking=True)
                batch[key]['attention_mask'] = batch[key]['attention_mask'].to(device, non_blocking=True)
        
        anchor_vec = self.query_encoder(
            input_ids=batch['query']['input_ids'],
            attention_mask=batch['query']['attention_mask'],
            token_type_ids=batch['query'].get('token_type_ids')
        )
        
        positive_vec = self.document_encoder(
            embeddings=batch['positive']['embeddings'],
            attention_mask=batch['positive']['attention_mask'],
            query_vec=anchor_vec
        )
        
        negative_vec = self.document_encoder(
            embeddings=batch['negative']['embeddings'],
            attention_mask=batch['negative']['attention_mask'],
            query_vec=anchor_vec
        )
        
        # Нормализация
        anchor_vec = F.normalize(anchor_vec, p=2, dim=1)
        positive_vec = F.normalize(positive_vec, p=2, dim=1)
        negative_vec = F.normalize(negative_vec, p=2, dim=1)
        
        return anchor_vec, positive_vec, negative_vec