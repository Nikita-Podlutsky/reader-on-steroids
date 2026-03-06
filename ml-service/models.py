import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

from config import CONFIG

try:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("WARNING: bitsandbytes не установлен — QLoRA отключён, используем обычный LoRA")


class QueryEncoderBGE(nn.Module):
    def __init__(self):
        super().__init__()

        use_qlora = CONFIG.USE_QLORA_QUERY and HAS_BNB

        if use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.transformer = AutoModel.from_pretrained(
                CONFIG.QUERY_MODEL_NAME,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.transformer = prepare_model_for_kbit_training(self.transformer)
        else:
            # Обычная загрузка без квантизации
            self.transformer = AutoModel.from_pretrained(
                CONFIG.QUERY_MODEL_NAME,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        if CONFIG.USE_QLORA_QUERY:
            lora_config = LoraConfig(
                r=CONFIG.QLORA_QUERY_R,
                lora_alpha=CONFIG.QLORA_QUERY_ALPHA,
                target_modules=CONFIG.QLORA_QUERY_TARGET_MODULES,
                lora_dropout=CONFIG.QLORA_QUERY_DROPOUT,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.transformer = get_peft_model(self.transformer, lora_config)
            print(f"✓ LoRA QueryEncoder: {self.transformer.get_nb_trainable_parameters()[0]:,} параметров")

        if CONFIG.DEVICE == 'cuda':
            self.transformer = self.transformer.cuda()

        
        with torch.no_grad():
            dummy = torch.randint(0, 100, (1, 10))
            dummy_mask = torch.ones(1, 10)
            if CONFIG.DEVICE == 'cuda':
                dummy = dummy.cuda()
                dummy_mask = dummy_mask.cuda()
            out = self.transformer(dummy, dummy_mask).last_hidden_state
            self.output_dtype = out.dtype

        input_dim = CONFIG.QUERY_MODEL_INPUT_DIM
        output_dim = CONFIG.FINAL_QUERY_DIM

        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, dtype=self.output_dtype),
            nn.LayerNorm(input_dim // 2, dtype=self.output_dtype),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim, dtype=self.output_dtype)
        ).to(CONFIG.DEVICE)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        device = next(self.transformer.parameters()).device

        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device, non_blocking=True)

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state = outputs.last_hidden_state

        expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * expanded_mask, 1)
        sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        return self.projection(pooled_output.to(self.output_dtype))


class DocumentEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        model_dtype = torch.bfloat16 if CONFIG.USE_BF16 else torch.float16

        self.longformer = AutoModel.from_pretrained(
            CONFIG.LONGFORMER_MODEL,
            torch_dtype=model_dtype
        )

        if CONFIG.USE_LORA_DOC:
            print("INFO: Applying LoRA to DocumentEncoder...")
            lora_config_doc = LoraConfig(
                r=CONFIG.LORA_DOC_R,
                lora_alpha=CONFIG.LORA_DOC_ALPHA,
                target_modules=CONFIG.LORA_DOC_TARGET_MODULES,
                lora_dropout=CONFIG.LORA_DOC_DROPOUT,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            self.longformer = get_peft_model(self.longformer, lora_config_doc)
            self.longformer.print_trainable_parameters()

        if CONFIG.DEVICE == 'cuda':
            self.longformer = self.longformer.cuda()

        if CONFIG.USE_GRADIENT_CHECKPOINTING:
            self.longformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.sentence_projection = nn.Linear(
            CONFIG.SENTENCE_EMBEDDING_DIM, CONFIG.LONGFORMER_DIM, dtype=model_dtype
        ).to(CONFIG.DEVICE)

        self.query_projection = nn.Linear(
            CONFIG.FINAL_QUERY_DIM, CONFIG.LONGFORMER_DIM, dtype=model_dtype
        ).to(CONFIG.DEVICE)

        self.attention = nn.Linear(
            CONFIG.LONGFORMER_DIM * 2, 1, dtype=model_dtype
        ).to(CONFIG.DEVICE)

        self.layer_norm = nn.LayerNorm(CONFIG.LONGFORMER_DIM, dtype=model_dtype).to(CONFIG.DEVICE)

        self.register_buffer(
            'global_attention_mask_template',
            torch.zeros(CONFIG.MAX_SENTENCES, dtype=torch.long, device=CONFIG.DEVICE),
            persistent=False
        )
        self.global_attention_mask_template[0] = 1

    def forward(self, embeddings, attention_mask, query_vec):
        device = next(self.longformer.parameters()).device
        dtype = next(self.longformer.parameters()).dtype

        embeddings = embeddings.to(device, non_blocking=True, dtype=dtype)
        attention_mask = attention_mask.to(device, non_blocking=True)
        query_vec = query_vec.to(device, non_blocking=True, dtype=dtype)

        projected = self.sentence_projection(embeddings)
        global_attention_mask = attention_mask * self.global_attention_mask_template[:embeddings.shape[1]]

        transformer_output = self.longformer(
            inputs_embeds=projected,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=False
        ).last_hidden_state

        query_projected = self.query_projection(query_vec)
        query_expanded = query_projected.unsqueeze(1).expand(-1, embeddings.shape[1], -1)
        combined = torch.cat([transformer_output, query_expanded], dim=-1)
        energy = self.attention(combined).squeeze(-1)
        energy_masked = torch.where(attention_mask == 0, -1e9, energy)
        attention_weights = F.softmax(energy_masked, dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * transformer_output, dim=1)

        return self.layer_norm(context_vector)


class UniversalScorer(nn.Module):
    def __init__(self):
        super().__init__()

        self.query_encoder = QueryEncoderBGE()
        self.document_encoder = DocumentEncoder()

        assert CONFIG.FINAL_QUERY_DIM == CONFIG.LONGFORMER_DIM

        if CONFIG.DEVICE == 'cuda':
            self.cuda()

    def forward(self, batch):
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

        anchor_vec = F.normalize(anchor_vec, p=2, dim=1)
        positive_vec = F.normalize(positive_vec, p=2, dim=1)
        negative_vec = F.normalize(negative_vec, p=2, dim=1)

        return anchor_vec, positive_vec, negative_vec