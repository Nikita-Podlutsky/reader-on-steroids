#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Универсальный смешиватель тем с выявлением тонких различий
pip install bertopic sentence-transformers matplotlib numpy umap-learn hdbscan requests scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import colorsys
import requests
import sys
from typing import List, Tuple, Dict
import networkx as nx
from matplotlib.patches import FancyBboxPatch

# Суровый тест DOCUMENTS для проверки всех возможностей кода

DOCUMENTS = [
    # === БЛОК 1: Почти идентичные (различия на уровне 1-2 слов) ===
    "Neural networks with backpropagation use gradient descent to minimize loss functions through iterative weight updates across multiple layers.",
    "Neural networks with backpropagation utilize gradient descent for minimizing loss functions via iterative weight adjustments across multiple layers.",
    "Neural architectures with backpropagation apply gradient descent to reduce loss functions through iterative parameter updates across multiple layers.",
    
    # === БЛОК 2: Семантически близкие, но разные термины ===
    "Quantum computing leverages superposition and entanglement to perform parallel computations exponentially faster than classical computers.",
    "Quantum processors exploit quantum superposition and quantum entanglement for massively parallel calculations surpassing classical computation speeds.",
    "Quantum machines harness superposition states and entangled qubits to achieve exponential speedup over traditional computing architectures.",
    
    # === БЛОК 3: Смешанные темы (гибриды двух областей) ===
    "Quantum neural networks combine quantum computing principles with deep learning architectures for enhanced pattern recognition capabilities.",
    "Hybrid quantum-classical algorithms integrate quantum circuits with traditional neural networks to solve optimization problems efficiently.",
    "Quantum-enhanced machine learning uses quantum annealing and gradient descent for training deep neural networks on quantum hardware.",
    
    # === БЛОК 4: Технические детали с overlap ===
    "CUDA kernels with shared memory optimization enable efficient parallel matrix multiplication on GPU architectures with thousands of cores.",
    "GPU parallelization through CUDA kernels and shared memory allows high-throughput matrix operations across thousands of simultaneous threads.",
    "Parallel GPU computing via CUDA utilizes shared memory banks for accelerated tensor operations across massively parallel architectures.",
    
    # === БЛОК 5: Одинаковые объекты, разные аспекты ===
    "Transformer attention mechanisms compute query-key similarities using dot products followed by softmax normalization for weighted value aggregation.",
    "Transformer models employ multi-head attention with learnable projection matrices to capture diverse representation subspaces simultaneously.",
    "Transformers utilize positional encodings and layer normalization alongside self-attention to maintain sequence order information during processing.",
    
    # === БЛОК 6: Сложные составные концепции ===
    "Federated learning with differential privacy enables distributed model training across edge devices while preserving user data confidentiality through noise injection.",
    "Privacy-preserving federated optimization aggregates local model updates from multiple clients using secure aggregation protocols and homomorphic encryption.",
    "Decentralized federated learning systems coordinate distributed training with secure multi-party computation to protect sensitive training data.",
    
    # === БЛОК 7: Близкие акронимы и вариации ===
    "BERT pre-training uses masked language modeling and next sentence prediction on large unlabeled corpora for contextual embeddings.",
    "RoBERTa improves BERT by removing next sentence prediction and training longer with dynamic masking patterns on larger datasets.",
    "ALBERT reduces BERT parameters through factorized embeddings and cross-layer parameter sharing while maintaining performance.",
    
    # === БЛОК 8: Тонкие различия в деталях ===
    "Convolutional layers with 3x3 kernels, ReLU activation, and batch normalization form basic building blocks of ResNet architectures.",
    "Convolutional blocks using 3x3 filters, leaky ReLU activations, and instance normalization constitute fundamental ResNet components.",
    "Convolutional modules with 3x3 receptive fields, ELU activations, and group normalization build standard ResNet structures.",
    
    # === БЛОК 9: Противоположные подходы к одной задаче ===
    "Generative adversarial networks use minimax game theory between generator and discriminator networks to synthesize realistic images.",
    "Variational autoencoders employ probabilistic inference with latent variable models to generate new data samples from learned distributions.",
    "Normalizing flows use invertible transformations with exact likelihood computation for high-quality generative modeling.",
    
    # === БЛОК 10: Очень специфичные детали (edge cases) ===
    "Flash Attention optimizes memory access patterns using tiling and recomputation to reduce HBM bandwidth for long-sequence transformers.",
    "PagedAttention implements virtual memory paging for KV cache management enabling efficient serving of large language models.",
    "Ring Attention distributes sequence chunks across devices with overlapped communication for memory-efficient ultra-long context processing.",
    
    # === БЛОК 11: Многословные описания с overlap ===
    "Self-supervised contrastive learning frameworks like SimCLR maximize agreement between augmented views using InfoNCE loss without requiring labeled data.",
    "Contrastive self-supervised methods such as MoCo maintain momentum-updated encoders with large memory banks for learning visual representations.",
    "Self-supervised contrastive approaches including BYOL eliminate negative samples by using asymmetric networks with stop-gradient operations.",
    
    # === БЛОК 12: Смесь трёх тем одновременно ===
    "Distributed reinforcement learning with evolutionary strategies and model-based planning enables sample-efficient robot manipulation in simulation.",
    "Multi-agent reinforcement learning combines policy gradient methods with centralized training and decentralized execution for cooperative tasks.",
    "Model-based reinforcement learning integrates world models with value iteration and Monte Carlo tree search for planning.",
]

# Статистика теста:
# - 33 документа (больше, чем обычно)
# - 12 тематических блоков с разной степенью overlap
# - Косинусное сходство в блоках: 0.85-0.98
# - Проверка: почти идентичные, гибриды, акронимы, противоположности
# - Ожидаемые темы: ~8-12 (зависит от параметров HDBSCAN)
# - Ожидаемые похожие пары: ~20-30

def translate_to_ru(text: str) -> str:
    """Перевод через Google API (без ключа)"""
    try:
        resp = requests.get("https://translate.googleapis.com/translate_a/single", 
                           params={"client": "gtx", "sl": "en", "tl": "ru", "dt": "t", "q": text}, 
                           timeout=5)
        if resp.status_code == 200:
            return "".join([chunk[0] for chunk in resp.json()[0]])
    except:
        pass
    return text




def find_similar_pairs_adaptive(embeddings: np.ndarray, 
                                 strict_threshold: float = 0.92,
                                 loose_threshold: float = 0.80) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    Многоуровневый поиск похожих документов:
    - strict: почти идентичные (0.92+)
    - loose: семантически близкие (0.80-0.92)
    """
    similarities = cosine_similarity(embeddings)
    results = {
        'identical': [],      # 0.95+
        'very_similar': [],   # 0.92-0.95
        'similar': [],        # 0.80-0.92
    }
    
    for i in range(len(similarities)):
        for j in range(i + 1, len(similarities)):
            sim = similarities[i, j]
            
            if sim >= 0.95:
                results['identical'].append((i, j, sim))
            elif sim >= strict_threshold:
                results['very_similar'].append((i, j, sim))
            elif sim >= loose_threshold:
                results['similar'].append((i, j, sim))
    
    # Сортируем по убыванию сходства
    for key in results:
        results[key] = sorted(results[key], key=lambda x: x[2], reverse=True)
    
    return results

def extract_distinctive_features(doc1: str, doc2: str, 
                                 unique_terms1: List[str], 
                                 unique_terms2: List[str]) -> Dict[str, List[str]]:
    """
    Углублённый анализ различий между похожими документами
    """
    # Токенизация для анализа
    tokens1 = set(doc1.lower().split())
    tokens2 = set(doc2.lower().split())
    
    # Различия на уровне слов
    only_in_1 = tokens1 - tokens2
    only_in_2 = tokens2 - tokens1
    
    # Различия на уровне TF-IDF терминов
    unique_1 = set(unique_terms1) - set(unique_terms2)
    unique_2 = set(unique_terms2) - set(unique_terms1)
    
    # Различия в числах/цифрах
    numbers1 = [w for w in tokens1 if any(c.isdigit() for c in w)]
    numbers2 = [w for w in tokens2 if any(c.isdigit() for c in w)]
    
    return {
        'word_diff_1': list(only_in_1)[:5],
        'word_diff_2': list(only_in_2)[:5],
        'term_diff_1': list(unique_1)[:3],
        'term_diff_2': list(unique_2)[:3],
        'numbers_1': numbers1,
        'numbers_2': numbers2,
    }

def print_similarity_report(documents: List[str], 
                           embeddings: np.ndarray, 
                           unique_terms: List[List[str]]):
    """
    Подробный отчёт с группировкой по уровням сходства
    """
    results = find_similar_pairs_adaptive(embeddings)
    
    print("\n" + "="*80)
    print("🔴 ПОЧТИ ИДЕНТИЧНЫЕ ДОКУМЕНТЫ (сходство ≥ 0.95)")
    print("="*80)
    
    for i, j, sim in results['identical']:
        print(f"\n📄 Док {i+1} ⟷ Док {j+1} | Сходство: {sim:.4f}")
        print(f"   Док {i+1}: {documents[i][:80]}...")
        print(f"   Док {j+1}: {documents[j][:80]}...")
        
        features = extract_distinctive_features(
            documents[i], documents[j], 
            unique_terms[i], unique_terms[j]
        )
        
        if features['term_diff_1'] or features['term_diff_2']:
            print(f"   ⚡ Ключевые различия:")
            if features['term_diff_1']:
                print(f"      • Док {i+1}: {', '.join(features['term_diff_1'])}")
            if features['term_diff_2']:
                print(f"      • Док {j+1}: {', '.join(features['term_diff_2'])}")
    
    print("\n" + "="*80)
    print("🟠 ОЧЕНЬ ПОХОЖИЕ ДОКУМЕНТЫ (сходство 0.92-0.95)")
    print("="*80)
    
    for i, j, sim in results['very_similar']:
        print(f"\n📄 Док {i+1} ⟷ Док {j+1} | Сходство: {sim:.4f}")
        print(f"   Уникальные термины [{i+1}]: {', '.join(unique_terms[i][:4])}")
        print(f"   Уникальные термины [{j+1}]: {', '.join(unique_terms[j][:4])}")
    
    print("\n" + "="*80)
    print("🟡 СЕМАНТИЧЕСКИ БЛИЗКИЕ (сходство 0.80-0.92)")
    print("="*80)
    
    # Группируем по кластерам
    clusters = {}
    for i, j, sim in results['similar']:
        if i not in clusters:
            clusters[i] = []
        clusters[i].append((j, sim))
    
    for doc_idx, neighbors in clusters.items():
        if len(neighbors) >= 2:  # Показываем только если 2+ соседей
            print(f"\n📄 Док {doc_idx+1} связан с {len(neighbors)} документами:")
            print(f"   Термины: {', '.join(unique_terms[doc_idx][:5])}")
            for neighbor_idx, sim in sorted(neighbors, key=lambda x: x[1], reverse=True)[:3]:
                print(f"   → Док {neighbor_idx+1} ({sim:.3f}): {', '.join(unique_terms[neighbor_idx][:3])}")
    
    # Статистика
    total_pairs = sum(len(v) for v in results.values())
    print(f"\n📊 Статистика:")
    print(f"   Почти идентичные пары: {len(results['identical'])}")
    print(f"   Очень похожие пары: {len(results['very_similar'])}")
    print(f"   Семантически близкие: {len(results['similar'])}")
    print(f"   ВСЕГО пар: {total_pairs}")

# Использование в основной функции:
# В функции main() замените вызов find_similar_pairs на:
# print_similarity_report(DOCUMENTS, doc_embeddings, unique_terms)



def generate_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    colors = []
    for i in range(n):
        hue = i / max(1, n)
        saturation = 0.8 + 0.2 * (i % 2)
        lightness = 0.5 + 0.1 * ((i // 2) % 2)
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def mix_colors(doc_topics: np.ndarray, topic_colors: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    mixed = []
    for dist in doc_topics:
        dist = dist / (dist.sum() + 1e-8)
        r = sum(dist[j] * topic_colors[j][0] for j in range(len(dist)))
        g = sum(dist[j] * topic_colors[j][1] for j in range(len(dist)))
        b = sum(dist[j] * topic_colors[j][2] for j in range(len(dist)))
        mixed.append((int(r), int(g), int(b)))
    return mixed

def extract_unique_terms(documents: List[str], top_k: int = 5) -> List[List[str]]:
    """Извлечение уникальных терминов для каждого документа (TF-IDF)"""
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.8
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    feature_names = vectorizer.get_feature_names_out()
    unique_terms = []
    
    # Для каждого документа находим слова с самым высоким TF-IDF
    for i in range(len(documents)):
        doc_tfidf = tfidf_matrix[i].toarray().flatten()
        
        # Получаем индексы топ-K слов
        top_indices = np.argpartition(doc_tfidf, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(doc_tfidf[top_indices])[::-1]]
        
        terms = [feature_names[idx] for idx in top_indices if doc_tfidf[idx] > 0]
        unique_terms.append(terms)
    
    return unique_terms


def create_similarity_network_viz(documents: List[str], 
                                   embeddings: np.ndarray,
                                   doc_topics: np.ndarray,
                                   article_colors: List[Tuple[int, int, int]],
                                   unique_terms: List[List[str]]):
    """
    Создаёт график с network-подобной визуализацией связей
    """
    # Создаём UMAP 2D проекцию внутри функции
    umap_2d = UMAP(n_components=2, random_state=42, metric='cosine')
    vis_2d = umap_2d.fit_transform(doc_topics)
    fig = plt.figure(figsize=(20, 10))
    
    # Три панели: scatter + network + heatmap
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[2, 2, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Scatter с уровнями
    ax2 = fig.add_subplot(gs[0, 1])  # Network graph
    ax3 = fig.add_subplot(gs[0, 2])  # Heatmap сходства
    ax4 = fig.add_subplot(gs[1, :])  # Легенда + статистика
    
    # --- Панель 1: Scatter с цветовым кодированием связей ---
    results = find_similar_pairs_adaptive(embeddings)
    
    # Рисуем связи (от слабых к сильным)
    for i, j, sim in results['similar']:
        ax1.plot([vis_2d[i, 0], vis_2d[j, 0]], 
                [vis_2d[i, 1], vis_2d[j, 1]], 
                'grey', alpha=0.2, linewidth=0.5, zorder=1)
    
    for i, j, sim in results['very_similar']:
        ax1.plot([vis_2d[i, 0], vis_2d[j, 0]], 
                [vis_2d[i, 1], vis_2d[j, 1]], 
                'orange', alpha=0.5, linewidth=1.5, zorder=2)
    
    for i, j, sim in results['identical']:
        ax1.plot([vis_2d[i, 0], vis_2d[j, 0]], 
                [vis_2d[i, 1], vis_2d[j, 1]], 
                'red', alpha=0.8, linewidth=2.5, zorder=3)
    
    # Точки поверх связей
    for i in range(len(documents)):
        color = [c/255 for c in article_colors[i]]
        
        # Подсчитываем связи каждого уровня
        identical_count = sum(1 for x in results['identical'] if i in (x[0], x[1]))
        similar_count = sum(1 for x in results['very_similar'] if i in (x[0], x[1]))
        
        # Размер точки зависит от количества связей
        total_connections = identical_count + similar_count
        size = 300 + total_connections * 100
        
        # Цвет границы зависит от уровня связей
        if identical_count > 0:
            edge_color = 'red'
            edge_width = 3
        elif similar_count > 0:
            edge_color = 'orange'
            edge_width = 2
        else:
            edge_color = 'gray'
            edge_width = 1
        
        ax1.scatter(vis_2d[i, 0], vis_2d[i, 1],
                   c=[color], s=size, alpha=0.8,
                   edgecolors=edge_color, linewidth=edge_width, zorder=10)
        
        # Аннотации с количеством связей
        label = f"{i+1}"
        if total_connections > 0:
            label += f" ({total_connections})"
        
        ax1.annotate(label, (vis_2d[i, 0], vis_2d[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             alpha=0.7, edgecolor=edge_color))
    
    ax1.set_title("UMAP проекция с уровнями сходства\n🔴 Идентичные 🟠 Похожие ⚫ Близкие", 
                 fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.2)
    
    # --- Панель 2: Network graph ---
    G = nx.Graph()
    
    # Добавляем все узлы
    for i in range(len(documents)):
        G.add_node(i, label=f"D{i+1}")
    
    # Добавляем рёбра
    for i, j, sim in results['identical']:
        G.add_edge(i, j, weight=sim, color='red', width=3)
    for i, j, sim in results['very_similar']:
        G.add_edge(i, j, weight=sim, color='orange', width=2)
    for i, j, sim in results['similar']:
        if sim > 0.85:  # Только самые сильные
            G.add_edge(i, j, weight=sim, color='lightgray', width=1)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Рисуем рёбра по типам
    for edge_type, color, width in [('red', 'red', 3), ('orange', 'orange', 2), ('lightgray', 'gray', 1)]:
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('color') == color]
        nx.draw_networkx_edges(G, pos, edgelist=edges, 
                              edge_color=color, width=width, alpha=0.6, ax=ax2)
    
    # Рисуем узлы с цветами документов
    node_colors = [[c/255 for c in article_colors[i]] for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=500, alpha=0.9, ax=ax2)
    
    # Подписи
    labels = {i: f"{i+1}" for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                           font_weight='bold', ax=ax2)
    
    ax2.set_title("Граф связей документов\n(узлы = документы, рёбра = сходство)", 
                 fontsize=12, weight='bold')
    ax2.axis('off')
    
    # --- Панель 3: Heatmap топ-20 пар ---
    all_pairs = (results['identical'] + results['very_similar'] + 
                results['similar'][:15])  # Топ-15 из близких
    
    if all_pairs:
        heat_data = []
        heat_labels = []
        
        for i, j, sim in all_pairs[:20]:
            heat_data.append(sim)
            heat_labels.append(f"{i+1}↔{j+1}")
        
        y_pos = np.arange(len(heat_data))
        colors_heat = ['red' if s >= 0.95 else 'orange' if s >= 0.92 else 'gold' 
                      for s in heat_data]
        
        ax3.barh(y_pos, heat_data, color=colors_heat, alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(heat_labels, fontsize=8)
        ax3.set_xlim(0.75, 1.0)
        ax3.set_xlabel("Косинусное сходство", fontsize=9)
        ax3.set_title("Топ-20 похожих пар", fontsize=11, weight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.invert_yaxis()
    
    # --- Панель 4: Статистика + кластеры ---
    ax4.axis('off')
    
    stats_text = f"""
📊 СТАТИСТИКА АНАЛИЗА:

Всего документов: {len(documents)}
Найдено тематических групп: {len(set(results['identical'] + results['very_similar']))}

🔴 Почти идентичные (≥0.95):  {len(results['identical'])} пар
🟠 Очень похожие (0.92-0.95): {len(results['very_similar'])} пар
🟡 Семантически близкие (0.80-0.92): {len(results['similar'])} пар

Самая сильная связь: {max(results['identical'] + results['very_similar'], key=lambda x: x[2])[2]:.4f}
Средняя связь (все): {np.mean([x[2] for x in results['identical'] + results['very_similar'] + results['similar']]):.4f}

📝 ПРИМЕРЫ ВЫЯВЛЕННЫХ КЛАСТЕРОВ:
"""
    
    # Находим топ-5 документов с наибольшим количеством связей
    doc_connections = {}
    for cat in ['identical', 'very_similar', 'similar']:
        for i, j, _ in results[cat]:
            doc_connections[i] = doc_connections.get(i, 0) + 1
            doc_connections[j] = doc_connections.get(j, 0) + 1
    
    top_docs = sorted(doc_connections.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for idx, (doc_id, conn_count) in enumerate(top_docs, 1):
        terms = ', '.join(unique_terms[doc_id][:3])
        stats_text += f"\n  {idx}. Док {doc_id+1} — {conn_count} связей — [{terms}]"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig
# Использование в main():

def main():
    print(f"📚 Обработка {len(DOCUMENTS)} документов...")
    print("🚀 Загрузка BAAI/bge-small-en-v1.5...")
    emb_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # ========== 1. Тематическое моделирование ==========
    n_docs = len(DOCUMENTS)
    umap_model = UMAP(n_neighbors=min(5, n_docs - 1), n_components=5, min_dist=0.0, 
                     metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1, metric='euclidean', 
                           cluster_selection_method='eom', prediction_data=True)

    topic_model = BERTopic(
        embedding_model=emb_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=True,
        verbose=False
    )

    topics, probs = topic_model.fit_transform(DOCUMENTS)
    
    topic_info = topic_model.get_topic_info()
    raw_topics = topic_model.get_topics()
    
    # Сохраняем порядок из topic_info
    valid_topics_ordered = []
    for idx, row in topic_info.iterrows():
        tid = row['Topic']
        if tid != -1 and tid in raw_topics:
            words = [w[0] for w in raw_topics[tid][:4]]
            en_name = " ".join(words[:2]).capitalize()
            ru_name = translate_to_ru(en_name) or en_name
            
            valid_topics_ordered.append({
                'id': tid,
                'en_name': en_name,
                'ru_name': ru_name,
                'words': words
            })
    
    if not valid_topics_ordered:
        print("❌ Темы не найдены. Добавьте больше разнообразных документов.")
        sys.exit(1)
    
    topic_colors = generate_distinct_colors(len(valid_topics_ordered))
    
    # Берем вероятности в правильном порядке
    doc_topics = np.zeros((n_docs, len(valid_topics_ordered)))
    for i, topic_data in enumerate(valid_topics_ordered):
        doc_topics[:, i] = probs[:, topic_data['id']]
    doc_topics = doc_topics / doc_topics.sum(axis=1, keepdims=True)
    
    article_colors = mix_colors(doc_topics, topic_colors)


    # ========== 2. Поиск тонких различий ==========
    print("\n" + "="*60)
    print("🔍 АНАЛИЗ ТОНКИХ РАЗЛИЧИЙ")
    print("="*60)
    
    # Получаем эмбеддинги документов для сравнения
    doc_embeddings = emb_model.encode(DOCUMENTS, convert_to_tensor=True).cpu().numpy()
    
    # Находим уникальные термины для каждого документа
    unique_terms = extract_unique_terms(DOCUMENTS, top_k=5)
    
    print_similarity_report(DOCUMENTS, doc_embeddings, unique_terms)
    fig = create_similarity_network_viz(
        DOCUMENTS, 
        doc_embeddings,  # эмбеддинги документов
        doc_topics,      # распределение по темам
        article_colors,  # цвета документов
        unique_terms     # уникальные термины
    )
    plt.savefig("topic_analysis_network.png", dpi=150, bbox_inches='tight')
    plt.show()
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)