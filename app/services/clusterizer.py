import asyncio
import colorsys
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

from app.config import MAX_TOPICS

def generate_neon_colors(n):
    """Генерирует пастельные цвета в стиле Anthropic"""
    colors = []
    # Пастельная палитра: мягкие оттенки
    pastel_palette = [
        '#fce7f3',  # мягкий розовый
        '#dbeafe',  # мягкий голубой
        '#ede9fe',  # мягкий фиолетовый
        '#f0fdf4',  # мягкий зеленый
        '#fffbeb',  # мягкий желтый
        '#fef2f2',  # мягкий красный
        '#f0f9ff',  # небесно-голубой
        '#faf5ff',  # лавандовый
        '#ecfdf5',  # мятный
        '#fff7ed',  # персиковый
    ]
    
    for i in range(n):
        # Используем палитру с циклическим повторением
        color = pastel_palette[i % len(pastel_palette)]
        colors.append(color)
    
    return colors

async def process_graph_analysis(engine, query, papers):
    if not papers:
        return {"nodes": [], "links": [], "topics": []}

    # 1. Получаем сырые тексты
    texts = [f"{p['title']}. {p['abstract']}"[:2000] for p in papers]
    
    # 2. Базовые эмбеддинги (нужны как вход для Scorer)
    base_tensor = engine.embedder.encode(texts, convert_to_tensor=True)

    # 3. === КЛЮЧЕВОЙ МОМЕНТ ===
    # Получаем "Умные векторы" из твоей модели UniversalScorer.
    # Именно они будут определять координаты X/Y.
    print("custom_vectors1")
    import time
    st = time.time()
    custom_vectors = await asyncio.to_thread(engine.get_custom_embeddings, query, papers, base_tensor)
    print(time.time()-st)
    
    # 4. Получаем Scores (важность статьи)
    scores = await asyncio.to_thread(engine.calculate_relevance, query, papers, base_tensor)
    print("scores")
    
    # Добавляем логирование времени
    import time
    t0 = time.time()
    
    # 5. Кластеризация (BERTopic)
    print("Starting BERTopic clustering...")
    umap_model = UMAP(n_neighbors=min(15, len(papers)-1), n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=MAX_TOPICS
    )
    
    # Обучаем BERTopic на твоих векторах
    topics, _ = topic_model.fit_transform(texts, embeddings=custom_vectors)
    t1 = time.time()
    print(f"BERTopic clustering took: {t1-t0:.2f}s")
    
    # Обработка выбросов (-1)
    unique_topics = sorted(set(topics))
    if -1 in unique_topics:
        max_topic = max(unique_topics) if max(unique_topics) > -1 else 0
        topics = [max_topic + 1 if t == -1 else t for t in topics]

    # 6. Проекция UMAP (2D координаты)
    print("Starting 2D UMAP projection...")
    t2 = time.time()
    coords_2d = UMAP(
        n_neighbors=min(15, len(papers)-1),
        n_components=2,
        min_dist=0.2,
        metric='cosine',
        random_state=42
    ).fit_transform(custom_vectors)
    t3 = time.time()
    print(f"2D UMAP projection took: {t3-t2:.2f}s")

    # 7. Анализ иерархии (Кто главный?)
    clusters_map = {}
    for i, t_id in enumerate(topics):
        t_id = int(t_id)
        if t_id not in clusters_map: clusters_map[t_id] = []
        clusters_map[t_id].append(i)

    # Находим Roots (лидеров с макс скором)
    roots_map = {} 
    for t_id, indices in clusters_map.items():
        leader_idx = max(indices, key=lambda idx: scores[idx])
        roots_map[t_id] = leader_idx

    # Нейминг тем
    print("Generating cluster names...")
    t4 = time.time()
    cluster_titles = {tid: [papers[i]['title'] for i in idxs] for tid, idxs in clusters_map.items()}
    topic_names = await engine.name_clusters(cluster_titles)
    t5 = time.time()
    print(f"Cluster naming took: {t5-t4:.2f}s")
    
    palette = generate_neon_colors(len(clusters_map))
    sorted_topics = sorted(clusters_map.keys())
    topic_to_color_idx = {t: i for i, t in enumerate(sorted_topics)}

    # 8. Сборка Nodes JSON
    nodes = []
    for i, p in enumerate(papers):
        t_id = int(topics[i])
        root_idx = roots_map[t_id]
        is_root = (i == root_idx)
        
        # Считаем дистанцию от текущей точки до Лидера её группы
        dist_to_root = float(np.linalg.norm(coords_2d[i] - coords_2d[root_idx]))

        # Извлекаем год из published (формат: "YYYY-MM-DD" или просто год)
        year = None
        if 'published' in p:
            try:
                year = int(p['published'].split('-')[0]) if '-' in p['published'] else int(p['published'][:4])
            except:
                year = None
        
        nodes.append({
            "id": p['id'],
            "label": p['title'],
            "abstract": p['abstract'],
            "year": year,  # Год публикации для фильтрации
            
            # Координаты (посчитаны один раз навечно)
            "x": float(coords_2d[i][0]) * 20,
            "y": float(coords_2d[i][1]) * 20,
            
            # Группировка
            "cluster_id": t_id,
            "cluster_name": topic_names.get(t_id, f"Topic {t_id}"),
            "color": palette[topic_to_color_idx[t_id]],
            "val": float(scores[i]) * 15 + 5, # Размер узла
            
            # Данные для фронтенда (Туман войны)
            "is_root": is_root,                 # True = показывать сразу
            "parent_id": papers[root_idx]['id'],# ID родителя
            "dist_to_root": dist_to_root        # Для сортировки при раскрытии
        })

    # Легенда
    final_topics = []
    for t_id in sorted_topics:
        final_topics.append({
            "id": t_id,
            "name": topic_names.get(t_id, f"Topic {t_id}"),
            "color": palette[topic_to_color_idx[t_id]],
            "count": len(clusters_map[t_id])
        })

    return {"nodes": nodes, "links": [], "topics": final_topics}