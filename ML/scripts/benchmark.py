# ==============================================================================
#
#                   Скрипт Комплексного Тестирования Модели
#
# ==============================================================================
#
#   Что делает:
#   1. Оценивает качество модели ранжирования по расширенному набору метрик IR.
#   2. Вычисляет MRR, NDCG@K, Hit@K, MAP, Precision/Recall@K для K=1,3,5,10.
#   3. Сравнивает производительность Custom модели с бейзлайнами (Mean, MaxSim).
#   4. Анализирует результаты в зависимости от длины документов (короткие/длинные).
#   5. Измеряет латентность (latency) с процентилями P50, P95, P99.
#   6. Сохраняет подробные результаты в JSON файл с метаданными.
#
#   Запуск:
#   > python benchmark.py
#
#   Требует предобученной модели в checkpoints/best_model.pt
#
# ==============================================================================

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import ndcg_score
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime

# ==============================================================================
# ИМПОРТЫ ПРОЕКТА
# ==============================================================================
from config import CONFIG
from dataset import HierarchicalTripletDataset2, collate_for_hierarchical
from models import UniversalScorer

# ==============================================================================
# НАСТРОЙКИ БЕНЧМАРКА
# ==============================================================================
BATCH_SIZE = 4
NUM_NEGATIVES = 100
EVAL_LIMIT = 200

# ==============================================================================
# РАСШИРЕННЫЕ МЕТРИКИ
# ==============================================================================

def compute_metrics_extended(y_true, y_score, k_values=[1, 3, 5, 10]):
    """
    Полный набор метрик для IR систем
    
    Args:
        y_true: [1, 0, 0, ...] - бинарные метки релевантности
        y_score: [0.95, 0.1, 0.3 ...] - предсказанные скоры
        k_values: список K для вычисления Precision@K, Recall@K
    
    Returns:
        dict с метриками
    """
    metrics = {}
    
    # Сортируем по убыванию скора
    sorted_indices = np.argsort(y_score)[::-1]
    
    # Позиция первого релевантного документа
    relevant_positions = np.where(y_true[sorted_indices] == 1)[0]
    first_relevant_pos = relevant_positions[0] + 1 if len(relevant_positions) > 0 else len(y_true) + 1
    
    # 1. MRR (Mean Reciprocal Rank)
    metrics['mrr'] = 1.0 / first_relevant_pos if first_relevant_pos <= len(y_true) else 0.0
    
    # 2. NDCG@K для разных K
    y_true_2d = np.expand_dims(y_true, axis=0)
    y_score_2d = np.expand_dims(y_score, axis=0)
    
    for k in k_values:
        try:
            metrics[f'ndcg@{k}'] = ndcg_score(y_true_2d, y_score_2d, k=k)
        except (ValueError, ZeroDivisionError):
            metrics[f'ndcg@{k}'] = 0.0
    
    # 3. Hit@K (нашелся ли релевантный документ в топ-K?)
    for k in k_values:
        top_k_indices = sorted_indices[:k]
        metrics[f'hit@{k}'] = 1.0 if np.any(y_true[top_k_indices] == 1) else 0.0
    
    # 4. Precision@K и Recall@K
    total_relevant = np.sum(y_true)
    
    for k in k_values:
        top_k_indices = sorted_indices[:k]
        relevant_in_top_k = np.sum(y_true[top_k_indices])
        
        metrics[f'precision@{k}'] = relevant_in_top_k / k if k > 0 else 0.0
        metrics[f'recall@{k}'] = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0
    
    # 5. MAP (Mean Average Precision)
    # Для одного запроса это просто Average Precision
    relevant_ranks = relevant_positions + 1  # +1 потому что позиции с 1
    if len(relevant_ranks) > 0:
        precisions_at_relevant = [(i + 1) / rank for i, rank in enumerate(relevant_ranks)]
        metrics['map'] = np.mean(precisions_at_relevant)
    else:
        metrics['map'] = 0.0
    
    # 6. Rank of first relevant document
    metrics['first_relevant_rank'] = first_relevant_pos
    
    return metrics


def compute_latency_stats(latencies):
    """Статистика по latency"""
    if len(latencies) == 0:
        return {}
    
    return {
        'mean_ms': np.mean(latencies),
        'median_ms': np.median(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p90_ms': np.percentile(latencies, 90),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'std_ms': np.std(latencies)
    }


def analyze_by_document_length(results_by_length):
    """
    Анализ качества в зависимости от длины документа
    
    Args:
        results_by_length: dict[length_bucket] -> list[metrics]
    """
    print("\n" + "="*80)
    print("📏 ANALYSIS BY DOCUMENT LENGTH")
    print("="*80)
    print(f"{'Length Range':<20} | {'Count':<8} | {'Hit@1':<10} | {'NDCG@10':<10} | {'MRR':<10}")
    print("-" * 80)
    
    for length_range in sorted(results_by_length.keys()):
        metrics_list = results_by_length[length_range]
        if len(metrics_list) == 0:
            continue
        
        count = len(metrics_list)
        avg_hit1 = np.mean([m['hit@1'] for m in metrics_list])
        avg_ndcg = np.mean([m['ndcg@10'] for m in metrics_list])
        avg_mrr = np.mean([m['mrr'] for m in metrics_list])
        
        print(f"{length_range:<20} | {count:<8} | {avg_hit1:.4f}     | {avg_ndcg:.4f}     | {avg_mrr:.4f}")
    
    print("="*80)


def categorize_document_length(num_sentences):
    """Категоризация документа по длине"""
    if num_sentences < 10:
        return "Short (<10 sent)"
    elif num_sentences < 30:
        return "Medium (10-30)"
    elif num_sentences < 50:
        return "Long (30-50)"
    else:
        return "Very Long (50+)"


def print_comparison_table(results_dict, metric_keys):
    """
    Печать таблицы сравнения методов с improvement
    
    Args:
        results_dict: dict[method_name] -> dict[metric] -> list[values]
        metric_keys: список метрик для отображения
    """
    print("\n" + "="*100)
    print("📊 DETAILED COMPARISON TABLE")
    print("="*100)
    
    # Собираем средние значения
    method_stats = {}
    for method, metrics in results_dict.items():
        method_stats[method] = {}
        for metric in metric_keys:
            if metric in metrics and len(metrics[metric]) > 0:
                method_stats[method][metric] = np.mean(metrics[metric])
            else:
                method_stats[method][metric] = 0.0
    
    # Определяем baseline (обычно MaxSim или первый метод)
    baseline_name = "Baseline_MaxSim" if "Baseline_MaxSim" in method_stats else list(method_stats.keys())[0]
    
    # Печатаем заголовок
    header = f"{'Metric':<15}"
    for method in method_stats.keys():
        header += f" | {method:<12}"
        if method != baseline_name:
            header += f" | Δ vs {baseline_name[:4]:<6}"
    print(header)
    print("-" * len(header))
    
    # Печатаем строки метрик
    for metric in metric_keys:
        row = f"{metric:<15}"
        baseline_value = method_stats[baseline_name][metric]
        
        for method in method_stats.keys():
            value = method_stats[method][metric]
            row += f" | {value:.4f}      "
            
            if method != baseline_name and baseline_value > 0:
                improvement = ((value / baseline_value) - 1) * 100
                emoji = "🚀" if improvement > 10 else "✅" if improvement > 0 else "⚠️"
                row += f" | {improvement:+6.1f}% {emoji}"
            elif method != baseline_name:
                row += f" |      N/A    "
        
        print(row)
    
    print("="*100)


def save_benchmark_results(results, output_path):
    """Сохранение результатов в JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Конвертируем numpy в native Python types
    results_serializable = {}
    for method, metrics in results.items():
        results_serializable[method] = {}
        for metric_name, values in metrics.items():
            if isinstance(values, list):
                results_serializable[method][metric_name] = {
                    'mean': float(np.mean(values)) if len(values) > 0 else 0.0,
                    'std': float(np.std(values)) if len(values) > 0 else 0.0,
                    'count': len(values),
                    'values': [float(v) for v in values]  # сохраняем сырые данные
                }
    
    # Добавляем метаданные
    results_serializable['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'batch_size': BATCH_SIZE,
        'num_negatives': NUM_NEGATIVES,
        'eval_limit': EVAL_LIMIT,
        'config': {
            'device': CONFIG.DEVICE,
            'query_model': CONFIG.QUERY_MODEL_NAME,
            'longformer_model': CONFIG.LONGFORMER_MODEL
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_path}")


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЧЕКПОИНТОВ
# ==============================================================================

def load_checkpoint(model, optimizer, scaler, scheduler, path: Path):
    """Загружает чекпоинт, включая LoRA-адаптеры."""
    if not path.is_dir():
        print(f"INFO: Директория чекпоинта не найдена: {path}")
        return None
        
    try:
        print(f"INFO: Загрузка чекпоинта из директории {path}...")
        
        # 1. Загружаем LoRA-адаптеры QueryEncoder
        query_lora_path = path / "query_encoder_lora"
        if CONFIG.USE_QLORA_QUERY and query_lora_path.is_dir():
            model.query_encoder.transformer.load_adapter(str(query_lora_path), adapter_name="default")
            print("INFO: -> LoRA адаптеры для QueryEncoder загружены.")

        # 2. Загружаем LoRA-адаптеры DocumentEncoder
        doc_lora_path = path / "document_encoder_lora"
        if CONFIG.USE_LORA_DOC and doc_lora_path.is_dir():
            model.document_encoder.longformer.load_adapter(str(doc_lora_path), adapter_name="default")
            print("INFO: -> LoRA адаптеры для DocumentEncoder загружены.")
        
        # 3. Загружаем остальные веса
        model.load_state_dict(torch.load(path / "model_state.pt", map_location=CONFIG.DEVICE), strict=False)
        print("INFO: -> Остальные веса модели загружены.")

        # 4. Загружаем состояние обучения
        training_state_path = path / "training_state.pt"
        if training_state_path.exists():
            state = torch.load(training_state_path, map_location=CONFIG.DEVICE)
            
            if optimizer is not None and 'optimizer_state_dict' in state:
                optimizer.load_state_dict(state['optimizer_state_dict'])
            
            if scheduler is not None and 'scheduler_state_dict' in state:
                scheduler.load_state_dict(state['scheduler_state_dict'])
            
            if scaler is not None and 'scaler_state_dict' in state and scaler.is_enabled():
                scaler.load_state_dict(state['scaler_state_dict'])
            
            print(f"✓ Чекпоинт успешно загружен! Метаданные обучения восстановлены.")
            return state
        else:
            print("WARNING: Файл training_state.pt не найден. Загружены только веса.")
            return None

    except Exception as e:
        print(f"❌ Ошибка загрузки чекпоинта: {e}. Обучение начнется с нуля.")
        return None


# ==============================================================================
# ОСНОВНАЯ ФУНКЦИЯ БЕНЧМАРКА
# ==============================================================================

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    
    # Структура для хранения результатов
    results = {
        "Custom_Model":    defaultdict(list),
        "Baseline_Mean":   defaultdict(list),
        "Baseline_MaxSim": defaultdict(list)
    }
    
    # Дополнительный анализ
    results_by_length = {
        "Custom_Model": defaultdict(list),
        "Baseline_MaxSim": defaultdict(list)
    }
    
    latencies = {
        "Custom_Model": [],
        "Baseline_Mean": [],
        "Baseline_MaxSim": []
    }
    
    k_values = [1, 3, 5, 10]  # K для метрик
    
    print(f"🚀 Starting Extended Benchmark on {device}...")
    print(f"📊 Computing metrics for K = {k_values}")
    
    count = 0
    pbar = tqdm(dataloader, total=min(len(dataloader), EVAL_LIMIT))
    
    for batch in pbar:
        if count >= EVAL_LIMIT:
            break
        count += 1
        
        bs = batch['query']['input_ids'].shape[0]
        if bs < 2:
            continue
        
        # ----------------------------------------------------------------------
        # 1. Подготовка данных
        # ----------------------------------------------------------------------
        q_input = batch['query']['input_ids'].to(device)
        q_mask = batch['query']['attention_mask'].to(device)
        q_token_type = batch['query'].get('token_type_ids')
        if q_token_type is not None:
            q_token_type = q_token_type.to(device)

        docs_emb = batch['positive']['embeddings'].to(device)
        docs_mask = batch['positive']['attention_mask'].to(device)

        # ----------------------------------------------------------------------
        # 2. Энкодинг запросов
        # ----------------------------------------------------------------------
        q_vecs = model.query_encoder(q_input, q_mask, q_token_type)
        q_vecs = F.normalize(q_vecs, p=2, dim=1)

        # ----------------------------------------------------------------------
        # 3. Проекция документов для baseline'ов
        # ----------------------------------------------------------------------
        model_dtype = next(model.document_encoder.parameters()).dtype
        docs_emb_casted = docs_emb.to(dtype=model_dtype)
        docs_emb_proj = model.document_encoder.sentence_projection(docs_emb_casted)

        # ----------------------------------------------------------------------
        # 4. Evaluation loop для каждого запроса
        # ----------------------------------------------------------------------
        for i in range(bs):
            current_q_vec = q_vecs[i].unsqueeze(0)
            
            # Формируем кандидатов
            cand_indices = [i] + [j for j in range(bs) if j != i]
            cand_indices = cand_indices[:NUM_NEGATIVES+1]
            num_cands = len(cand_indices)
            
            cand_raw_emb = docs_emb[cand_indices]
            cand_masks = docs_mask[cand_indices]
            cand_proj_emb = docs_emb_proj[cand_indices]
            
            # Определяем длину документа (для анализа)
            doc_length = int(cand_masks[0].sum().item())
            length_category = categorize_document_length(doc_length)
            
            # ==================================================================
            # METHOD 1: CUSTOM MODEL
            # ==================================================================
            import time
            start = time.perf_counter()
            
            q_vec_expanded = current_q_vec.expand(num_cands, -1)
            doc_vecs_custom = model.document_encoder(
                embeddings=cand_raw_emb,
                attention_mask=cand_masks,
                query_vec=q_vec_expanded
            )
            doc_vecs_custom = F.normalize(doc_vecs_custom, p=2, dim=1)
            scores_custom = torch.sum(q_vec_expanded * doc_vecs_custom, dim=1).float().cpu().numpy()
            
            latency_custom = (time.perf_counter() - start) * 1000  # ms
            latencies["Custom_Model"].append(latency_custom)

            # ==================================================================
            # METHOD 2: BASELINE MEAN
            # ==================================================================
            start = time.perf_counter()
            
            mask_expanded = cand_masks.unsqueeze(-1)
            sum_emb = torch.sum(cand_proj_emb * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_emb = sum_emb / sum_mask
            mean_emb = F.normalize(mean_emb, p=2, dim=1)
            scores_mean = torch.sum(current_q_vec * mean_emb, dim=1).float().cpu().numpy()
            
            latency_mean = (time.perf_counter() - start) * 1000
            latencies["Baseline_Mean"].append(latency_mean)

            # ==================================================================
            # METHOD 3: BASELINE MAXSIM
            # ==================================================================
            start = time.perf_counter()
            
            cand_proj_norm = F.normalize(cand_proj_emb, p=2, dim=2)
            sim_matrix = torch.matmul(cand_proj_norm.float(), current_q_vec.transpose(0, 1)).squeeze(-1)
            sim_matrix = sim_matrix.masked_fill(cand_masks == 0, -1e9)
            scores_max, _ = torch.max(sim_matrix, dim=1)
            scores_max = scores_max.float().cpu().numpy()
            
            latency_max = (time.perf_counter() - start) * 1000
            latencies["Baseline_MaxSim"].append(latency_max)

            # ==================================================================
            # COMPUTE EXTENDED METRICS
            # ==================================================================
            y_true = np.zeros(num_cands)
            y_true[0] = 1
            
            # Custom Model
            metrics_custom = compute_metrics_extended(y_true, scores_custom, k_values)
            for metric_name, value in metrics_custom.items():
                results["Custom_Model"][metric_name].append(value)
            results_by_length["Custom_Model"][length_category].append(metrics_custom)
            
            # Baseline Mean
            metrics_mean = compute_metrics_extended(y_true, scores_mean, k_values)
            for metric_name, value in metrics_mean.items():
                results["Baseline_Mean"][metric_name].append(value)
            
            # Baseline MaxSim
            metrics_max = compute_metrics_extended(y_true, scores_max, k_values)
            for metric_name, value in metrics_max.items():
                results["Baseline_MaxSim"][metric_name].append(value)
            results_by_length["Baseline_MaxSim"][length_category].append(metrics_max)
            
        # Update progress bar
        pbar.set_postfix({
            "C_H@1": f"{np.mean(results['Custom_Model']['hit@1'][-bs:]):.2f}",
            "B_H@1": f"{np.mean(results['Baseline_MaxSim']['hit@1'][-bs:]):.2f}",
            "Lat_C": f"{np.mean(latencies['Custom_Model'][-bs:]):.0f}ms"
        })

    # ==========================================================================
    # ИТОГОВЫЙ ВЫВОД РЕЗУЛЬТАТОВ
    # ==========================================================================
    
    # 1. Основная таблица сравнения
    main_metrics = ['hit@1', 'hit@3', 'hit@10', 'mrr', 'ndcg@10', 'map', 
                    'precision@10', 'recall@10']
    print_comparison_table(results, main_metrics)
    
    # 2. Детальные метрики для разных K
    print("\n" + "="*80)
    print("📈 METRICS ACROSS DIFFERENT K VALUES")
    print("="*80)
    for k in k_values:
        metrics_at_k = [f'hit@{k}', f'ndcg@{k}', f'precision@{k}', f'recall@{k}']
        available_metrics = [m for m in metrics_at_k if m in results["Custom_Model"]]
        if available_metrics:
            print(f"\n--- K = {k} ---")
            print_comparison_table(results, available_metrics)
    
    # 3. Latency Analysis
    print("\n" + "="*80)
    print("⚡ LATENCY ANALYSIS")
    print("="*80)
    print(f"{'Method':<20} | {'Mean':<10} | {'P50':<10} | {'P95':<10} | {'P99':<10}")
    print("-" * 80)
    
    for method in latencies.keys():
        stats = compute_latency_stats(latencies[method])
        if stats:
            print(f"{method:<20} | {stats['mean_ms']:>8.1f}ms | {stats['p50_ms']:>8.1f}ms | "
                  f"{stats['p95_ms']:>8.1f}ms | {stats['p99_ms']:>8.1f}ms")
    print("="*80)
    
    # 4. Analysis by document length
    analyze_by_document_length(results_by_length["Custom_Model"])
    analyze_by_document_length(results_by_length["Baseline_MaxSim"])
    
    # 5. Rank distribution analysis
    print("\n" + "="*80)
    print("🎯 FIRST RELEVANT DOCUMENT RANK DISTRIBUTION")
    print("="*80)
    
    for method in ["Custom_Model", "Baseline_MaxSim"]:
        ranks = results[method]["first_relevant_rank"]
        if len(ranks) > 0:
            print(f"\n{method}:")
            print(f"  Rank 1 (Hit@1):     {np.sum(np.array(ranks) == 1) / len(ranks) * 100:.1f}%")
            print(f"  Rank 2-3:           {np.sum((np.array(ranks) >= 2) & (np.array(ranks) <= 3)) / len(ranks) * 100:.1f}%")
            print(f"  Rank 4-10:          {np.sum((np.array(ranks) >= 4) & (np.array(ranks) <= 10)) / len(ranks) * 100:.1f}%")
            print(f"  Rank 11+:           {np.sum(np.array(ranks) > 10) / len(ranks) * 100:.1f}%")
            print(f"  Median Rank:        {np.median(ranks):.1f}")
            print(f"  Mean Rank:          {np.mean(ranks):.1f}")
    
    print("="*80)
    
    # 6. Save results to JSON
    output_path = Path("benchmark_results") / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_benchmark_results(results, output_path)
    
    return results, latencies


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("🚀 KOTODEX EXTENDED BENCHMARK")
    print("="*80)
    
    print("\n📦 Загрузка датасета...")
    dataset = HierarchicalTripletDataset2(CONFIG.FINAL_METADATA_FILE, device="cpu")
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_for_hierarchical,
        num_workers=0
    )
    
    print("🧠 Инициализация модели...")
    model = UniversalScorer()
    
    # Путь к чекпоинту
    CHECKPOINT_PATH = Path(r"checkpoints\best_model.pt")
    
    print(f"📂 Загрузка весов из: {CHECKPOINT_PATH}")
    load_checkpoint(model, None, None, None, CHECKPOINT_PATH)
    
    print("\n" + "="*80)
    print("▶️  Starting Evaluation...")
    print("="*80 + "\n")
    
    results, latencies = evaluate_model(model, loader, CONFIG.DEVICE)
    
    print("\n✅ Benchmark completed successfully!")
    print("="*80)
