import asyncio
import colorsys
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

from config import MAX_TOPICS

def generate_neon_colors(n):
    """Генерирует контрастные цвета для темного фона"""
    colors = []
    # Контрастная палитра для темного фона: яркие насыщенные оттенки
    contrast_palette = [
        '#3498db',  # синий
        '#e74c3c',  # красный
        '#2ecc71',  # зеленый
        '#f39c12',  # оранжевый
        '#9b59b6',  # фиолетовый
        '#1abc9c',  # бирюзовый
        '#e67e22',  # темно-оранжевый
        '#34495e',  # серо-синий
        '#16a085',  # темно-бирюзовый
        '#c0392b',  # темно-красный
        '#27ae60',  # темно-зеленый
        '#8e44ad',  # темно-фиолетовый
    ]
    
    for i in range(n):
        
        color = contrast_palette[i % len(contrast_palette)]
        colors.append(color)
    
    return colors

async def process_graph_analysis(engine, query, papers):
    if not papers:
        return {"nodes": [], "links": [], "topics": []}

    
    texts = [f"{p['title']}. {p['abstract']}"[:2000] for p in papers]
    
    
    base_tensor = engine.embedder.encode(texts, convert_to_tensor=True)


    
    print("custom_vectors1")
    import time
    st = time.perf_counter()
    custom_vectors = await asyncio.to_thread(engine.get_custom_embeddings, query, papers, base_tensor)
    print(time.perf_counter()-st)
    
    
    scores = await asyncio.to_thread(engine.calculate_relevance, query, papers, base_tensor)
    print("scores")
    
    
    import time
    t0 = time.perf_counter()
    
    print("Starting BERTopic clustering...")
    umap_model = UMAP(n_neighbors=min(15, len(papers)-1), n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=MAX_TOPICS
    )
    
    
    topics, _ = topic_model.fit_transform(texts, embeddings=custom_vectors)
    t1 = time.perf_counter()
    print(f"BERTopic clustering took: {t1-t0:.2f}s")
    
    
    unique_topics = sorted(set(topics))
    if -1 in unique_topics:
        max_topic = max(unique_topics) if max(unique_topics) > -1 else 0
        topics = [max_topic + 1 if t == -1 else t for t in topics]

    
    print("Starting 2D UMAP projection...")
    t2 = time.perf_counter()
    coords_2d = UMAP(
        n_neighbors=min(15, len(papers)-1),
        n_components=2,
        min_dist=0.2,
        metric='cosine',
        random_state=42
    ).fit_transform(custom_vectors)
    t3 = time.perf_counter()
    print(f"2D UMAP projection took: {t3-t2:.2f}s")

    
    clusters_map = {}
    for i, t_id in enumerate(topics):
        t_id = int(t_id)
        if t_id not in clusters_map: clusters_map[t_id] = []
        clusters_map[t_id].append(i)

    
    roots_map = {} 
    for t_id, indices in clusters_map.items():
        leader_idx = max(indices, key=lambda idx: scores[idx])
        roots_map[t_id] = leader_idx

    
    print("Generating cluster names...")
    t4 = time.perf_counter()
    cluster_titles = {tid: [papers[i]['title'] for i in idxs] for tid, idxs in clusters_map.items()}
    topic_names = await engine.name_clusters(cluster_titles)
    t5 = time.perf_counter()
    print(f"Cluster naming took: {t5-t4:.2f}s")
    
    palette = generate_neon_colors(len(clusters_map))
    sorted_topics = sorted(clusters_map.keys())
    topic_to_color_idx = {t: i for i, t in enumerate(sorted_topics)}

    
    nodes = []
    for i, p in enumerate(papers):
        t_id = int(topics[i])
        root_idx = roots_map[t_id]
        is_root = (i == root_idx)
        
        dist_to_root = float(np.linalg.norm(coords_2d[i] - coords_2d[root_idx]))

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
            "year": year,
            

            "x": float(coords_2d[i][0]) * 20,
            "y": float(coords_2d[i][1]) * 20,
            
            "cluster_id": t_id,
            "cluster_name": topic_names.get(t_id, f"Topic {t_id}"),
            "color": palette[topic_to_color_idx[t_id]],
            "val": float(scores[i]) * 15 + 5,
            
            
            "is_root": is_root,
            "parent_id": papers[root_idx]['id'],
            "dist_to_root": dist_to_root
        })

    final_topics = []
    for t_id in sorted_topics:
        final_topics.append({
            "id": t_id,
            "name": topic_names.get(t_id, f"Topic {t_id}"),
            "color": palette[topic_to_color_idx[t_id]],
            "count": len(clusters_map[t_id])
        })

    return {"nodes": nodes, "links": [], "topics": final_topics}