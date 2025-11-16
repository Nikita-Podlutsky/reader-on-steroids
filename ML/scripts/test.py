#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScholarMap API - Автономный сервис для анализа научных статей
Запуск: python scholarmap_api.py
Swagger UI: http://localhost:8001/
"""

import traceback
from typing import List, Dict, Optional, Tuple
import colorsys

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================

class Config:
    """Настройки приложения"""
    API_HOST = "0.0.0.0"
    API_PORT = 8001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    MIN_DOCUMENTS = 5
    MAX_DOCUMENTS = 500
    
CONFIG = Config()

# ==============================================================================
# PYDANTIC СХЕМЫ
# ==============================================================================

class DocumentInput(BaseModel):
    """Входной документ для анализа"""
    id: str = Field(..., description="Уникальный ID документа")
    title: str = Field(..., description="Заголовок")
    abstract: str = Field(..., description="Аннотация")
    authors: Optional[List[str]] = Field(default=None, description="Авторы")
    year: Optional[int] = Field(default=None, description="Год публикации")
    keywords: Optional[List[str]] = Field(default=None, description="Ключевые слова")

class TopicInfo(BaseModel):
    """Информация о теме"""
    id: int
    name: str
    color_rgb: List[int]
    keywords: List[str]
    doc_count: int

class DocumentNode(BaseModel):
    """Узел документа для графа"""
    id: str
    title: str
    position: List[float]  # [x, y]
    color: List[int]  # RGB
    size: float
    topics: List[int]
    similarity_score: float
    unique_terms: List[str]

class DocumentEdge(BaseModel):
    """Связь между документами"""
    source: str
    target: str
    similarity: float
    level: str  # 'identical', 'very_similar', 'similar'

class ClusterGroup(BaseModel):
    """Кластер документов"""
    id: int
    name: str
    doc_ids: List[str]
    centroid: List[float]
    size: int
    diversity_score: float

class AnalysisRequest(BaseModel):
    """Запрос на анализ"""
    documents: List[DocumentInput]
    min_similarity: Optional[float] = Field(default=0.75, ge=0.0, le=1.0)
    max_topics: Optional[int] = Field(default=10, ge=2, le=50)

class AnalysisResponse(BaseModel):
    """Результат анализа"""
    topics: List[TopicInfo]
    nodes: List[DocumentNode]
    edges: List[DocumentEdge]
    clusters: List[ClusterGroup]
    statistics: Dict

class HealthResponse(BaseModel):
    """Статус здоровья сервиса"""
    status: str
    device: str
    model: str

# ==============================================================================
# АНАЛИЗАТОР SCHOLARMAP
# ==============================================================================

class ScholarMapAnalyzer:
    """Движок анализа научных статей"""
    
    def __init__(self, model_name: str, device: str):
        print(f"🚀 Initializing ScholarMap Analyzer...")
        print(f"   Device: {device}")
        print(f"   Model: {model_name}")
        
        self.device = torch.device(device)
        self.model = SentenceTransformer(model_name, device=device)
        
        print("✅ Analyzer ready!")
    
    def generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Генерация различимых цветов"""
        colors = []
        for i in range(n):
            hue = i / max(1, n)
            saturation = 0.8 + 0.2 * (i % 2)
            lightness = 0.5 + 0.1 * ((i // 2) % 2)
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors
    
    def mix_colors(self, doc_topics: np.ndarray, 
                   colors: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """Смешивание цветов для документов"""
        mixed = []
        for dist in doc_topics:
            dist = dist / (dist.sum() + 1e-8)
            r = sum(dist[j] * colors[j][0] for j in range(len(dist)))
            g = sum(dist[j] * colors[j][1] for j in range(len(dist)))
            b = sum(dist[j] * colors[j][2] for j in range(len(dist)))
            mixed.append((int(r), int(g), int(b)))
        return mixed
    
    def extract_unique_terms(self, texts: List[str], top_k: int = 5) -> List[List[str]]:
        """TF-IDF для уникальных терминов"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            unique_terms = []
            for i in range(len(texts)):
                doc_tfidf = tfidf_matrix[i].toarray().flatten()
                top_indices = np.argpartition(doc_tfidf, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(doc_tfidf[top_indices])[::-1]]
                terms = [feature_names[idx] for idx in top_indices if doc_tfidf[idx] > 0]
                unique_terms.append(terms)
            
            return unique_terms
        except Exception as e:
            print(f"⚠️  Warning: TF-IDF extraction failed: {e}")
            return [[] for _ in texts]
    
    def find_similar_pairs(self, embeddings: np.ndarray, 
                          min_sim: float = 0.75) -> Dict[str, List[Tuple[int, int, float]]]:
        """Многоуровневый поиск похожих документов"""
        similarities = cosine_similarity(embeddings)
        results = {
            'identical': [],      # 0.95+
            'very_similar': [],   # 0.85-0.95
            'similar': [],        # min_sim - 0.85
        }
        
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                sim = similarities[i, j]
                
                if sim >= 0.95:
                    results['identical'].append((i, j, sim))
                elif sim >= 0.85:
                    results['very_similar'].append((i, j, sim))
                elif sim >= min_sim:
                    results['similar'].append((i, j, sim))
        
        # Сортировка по убыванию сходства
        for key in results:
            results[key] = sorted(results[key], key=lambda x: x[2], reverse=True)
        
        return results
    
    @torch.no_grad()
    def analyze(self, documents: List[DocumentInput], 
                min_similarity: float = 0.75,
                max_topics: int = 10) -> AnalysisResponse:
        """Основной метод анализа"""
        

    
        n_docs = len(documents)
        print(f"\n📚 Analyzing {n_docs} documents...")
        
        # Безопасная валидация размера данных
        if n_docs < 3:
            raise ValueError("Need at least 3 documents for meaningful analysis")
        
        if n_docs < 5 and max_topics > 5:
            print(f"⚠️  Adjusting max_topics from {max_topics} to {n_docs - 1} for small dataset")
            max_topics = n_docs - 1
        
        # 1. Подготовка текстов
        texts = [f"{doc.title}. {doc.abstract}" for doc in documents]
        doc_ids = [doc.id for doc in documents]
        
        # 2. Создание эмбеддингов
        print("   🔄 Creating embeddings...")
        embeddings = self.model.encode(
            texts, 
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=False
        ).cpu().numpy()
        
        # 3. Тематическое моделирование
        print("   🔄 Topic modeling...")
        
        # Безопасные параметры для малых корпусов
        if n_docs <= 10:
            # Для очень малых наборов данных
            n_neighbors = min(3, n_docs - 1)  # Не более 3 соседей
            n_components = min(2, n_docs - 1)  # Строго 2 компоненты
        else:
            # Для нормальных наборов данных
            n_neighbors = max(2, min(5, n_docs - 1))
            n_components = max(2, min(5, n_docs - 1))
        
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        if n_docs <= 10:
            min_cluster_size = 2  # Минимальный возможный кластер
        else:
            min_cluster_size = max(2, n_docs // 10)

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        topic_model = BERTopic(
            embedding_model=self.model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            verbose=False,
            nr_topics=max_topics
        )
        
        topics, probs = topic_model.fit_transform(texts)
        topic_info = topic_model.get_topic_info()
        raw_topics = topic_model.get_topics()
        
        # 4. Извлечение валидных тем
        valid_topics = []
        for idx, row in topic_info.iterrows():
            tid = row['Topic']
            if tid != -1 and tid in raw_topics:
                words = [w[0] for w in raw_topics[tid][:4]]
                valid_topics.append({
                    'id': tid,
                    'name': " ".join(words[:2]).capitalize(),
                    'words': words,
                    'count': int(row['Count'])
                })
        
        if not valid_topics:
            raise ValueError("Could not identify topics. Documents may be too similar or too few.")
        
        print(f"   ✅ Found {len(valid_topics)} topics")
        
        # 5. Генерация цветов
        topic_colors = self.generate_colors(len(valid_topics))
        
        # 6. Распределение документов по темам
        doc_topics = np.zeros((n_docs, len(valid_topics)))
        for i, topic_data in enumerate(valid_topics):
            doc_topics[:, i] = probs[:, topic_data['id']]
        doc_topics = doc_topics / (doc_topics.sum(axis=1, keepdims=True) + 1e-8)
        
        article_colors = self.mix_colors(doc_topics, topic_colors)
        
        # 7. UMAP для 2D визуализации
        print("   🔄 Creating 2D projection...")
        umap_2d = UMAP(n_components=2, random_state=42, metric='cosine')
        vis_2d = umap_2d.fit_transform(doc_topics)
        
        # 8. Поиск похожих пар
        print("   🔄 Finding similar pairs...")
        similar_pairs = self.find_similar_pairs(embeddings, min_similarity)
        
        # 9. Извлечение уникальных терминов
        print("   🔄 Extracting unique terms...")
        unique_terms = self.extract_unique_terms(texts, top_k=5)
        
        # 10. Подсчёт связей
        doc_connections = {}
        for level_pairs in similar_pairs.values():
            for i, j, _ in level_pairs:
                doc_connections[i] = doc_connections.get(i, 0) + 1
                doc_connections[j] = doc_connections.get(j, 0) + 1
        
        # 11. Формирование узлов
        nodes = []
        for i in range(n_docs):
            connections = doc_connections.get(i, 0)
            
            # Средняя схожесть с соседями
            neighbor_sims = []
            for level_pairs in similar_pairs.values():
                for idx1, idx2, sim in level_pairs:
                    if idx1 == i or idx2 == i:
                        neighbor_sims.append(sim)
            
            avg_sim = float(np.mean(neighbor_sims)) if neighbor_sims else 0.0
            
            nodes.append(DocumentNode(
                id=doc_ids[i],
                title=documents[i].title,
                position=[float(vis_2d[i, 0]), float(vis_2d[i, 1])],
                color=list(article_colors[i]),
                size=1.0 + connections * 0.3,
                topics=[t['id'] for j, t in enumerate(valid_topics) if doc_topics[i, j] > 0.1],
                similarity_score=avg_sim,
                unique_terms=unique_terms[i][:5]
            ))
        
        # 12. Формирование рёбер
        edges = []
        for level_name, pairs in similar_pairs.items():
            for i, j, sim in pairs:
                edges.append(DocumentEdge(
                    source=doc_ids[i],
                    target=doc_ids[j],
                    similarity=float(sim),
                    level=level_name
                ))
        
        # 13. Формирование кластеров
        clusters = []
        for topic_idx, topic_data in enumerate(valid_topics):
            cluster_docs = [
                doc_ids[i] for i in range(n_docs)
                if doc_topics[i, topic_idx] > 0.3
            ]
            
            if cluster_docs:
                cluster_positions = vis_2d[[i for i in range(n_docs) if doc_ids[i] in cluster_docs]]
                centroid = cluster_positions.mean(axis=0)
                diversity = float(np.std(cluster_positions))
                
                clusters.append(ClusterGroup(
                    id=topic_idx,
                    name=topic_data['name'],
                    doc_ids=cluster_docs,
                    centroid=[float(centroid[0]), float(centroid[1])],
                    size=len(cluster_docs),
                    diversity_score=min(1.0, diversity / 5.0)
                ))
        
        # 14. Статистика
        statistics = {
            'total_documents': n_docs,
            'total_topics': len(valid_topics),
            'total_edges': len(edges),
            'identical_pairs': len(similar_pairs['identical']),
            'very_similar_pairs': len(similar_pairs['very_similar']),
            'similar_pairs': len(similar_pairs['similar']),
            'avg_connections_per_doc': float(np.mean(list(doc_connections.values()))) if doc_connections else 0.0,
            'max_connections': max(doc_connections.values()) if doc_connections else 0
        }
        
        # 15. Формирование ответа
        topics_response = [
            TopicInfo(
                id=t['id'],
                name=t['name'],
                color_rgb=list(topic_colors[i]),
                keywords=t['words'],
                doc_count=t['count']
            )
            for i, t in enumerate(valid_topics)
        ]
        
        print(f"   ✅ Analysis complete!")
        print(f"      Topics: {len(topics_response)}")
        print(f"      Edges: {len(edges)}")
        print(f"      Clusters: {len(clusters)}")
        
        return AnalysisResponse(
            topics=topics_response,
            nodes=nodes,
            edges=edges,
            clusters=clusters,
            statistics=statistics
        )

# ==============================================================================
# FASTAPI ПРИЛОЖЕНИЕ
# ==============================================================================

app = FastAPI(
    title="ScholarMap API",
    version="1.0.0",
    description="API for analyzing scientific papers and building knowledge graphs",
    docs_url="/",
)

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный анализатор
analyzer: Optional[ScholarMapAnalyzer] = None

@app.on_event("startup")
def startup_event():
    """Инициализация при запуске"""
    global analyzer
    print("\n" + "="*60)
    print("🚀 SCHOLARMAP API STARTING")
    print("="*60)
    analyzer = ScholarMapAnalyzer(
        model_name=CONFIG.EMBEDDING_MODEL,
        device=CONFIG.DEVICE
    )
    print("="*60)
    print(f"✅ Server ready at http://{CONFIG.API_HOST}:{CONFIG.API_PORT}")
    print("="*60 + "\n")

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Проверка здоровья сервиса"""
    return HealthResponse(
        status="healthy",
        device=CONFIG.DEVICE,
        model=CONFIG.EMBEDDING_MODEL
    )

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
def analyze_documents(request: AnalysisRequest):
    """
    Анализ научных статей:
    - Тематическое моделирование
    - Поиск похожих документов
    - Кластеризация
    - Построение графа связей
    """
    
    n_docs = len(request.documents)
    
    # Валидация
    if n_docs < CONFIG.MIN_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {CONFIG.MIN_DOCUMENTS} documents (got {n_docs})"
        )
    
    if n_docs > CONFIG.MAX_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many documents. Max allowed: {CONFIG.MAX_DOCUMENTS} (got {n_docs})"
        )
    
    # Проверка на дубликаты ID
    doc_ids = [d.id for d in request.documents]
    if len(doc_ids) != len(set(doc_ids)):
        raise HTTPException(
            status_code=400,
            detail="Duplicate document IDs found"
        )
    
    try:
        result = analyzer.analyze(
            documents=request.documents,
            min_similarity=request.min_similarity,
            max_topics=request.max_topics
        )
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        print("\n" + "="*20 + " ERROR " + "="*20)
        traceback.print_exc()
        print("="*47 + "\n")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

# ==============================================================================
# ЗАПУСК
# ==============================================================================

if __name__ == "__main__":
    print(f"""
    ╔══════════════════════════════════════════════╗
    ║          ScholarMap API Server              ║
    ║                                              ║
    ║  Swagger UI: http://localhost:{CONFIG.API_PORT}/       ║
    ║  Health:     http://localhost:{CONFIG.API_PORT}/health ║
    ║                                              ║
    ║  Device: {CONFIG.DEVICE:<35} ║
    ║  Model:  {CONFIG.EMBEDDING_MODEL:<35} ║
    ╚══════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        app,
        host=CONFIG.API_HOST,
        port=CONFIG.API_PORT,
        log_level="info"
    )