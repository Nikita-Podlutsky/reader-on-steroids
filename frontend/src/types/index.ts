// Типы для работы с API бэкенда тимлида

export interface GraphNode {
  id: string;
  label: string;           // Используется как title
  abstract: string;
  
  // Координаты (фиксированные с бэкенда)
  x: number;
  y: number;
  
  // Группировка
  cluster_id: number;
  cluster_name: string;
  color: string;
  val: number;             // Размер узла
  
  // Туман войны
  is_root: boolean;
  parent_id: string;       // ID родителя
  dist_to_root: number;    // Дистанция до root-узла
  
  // Дополнительные поля (опционально)
  title?: string;          // Дубликат label для совместимости
  year?: number;
  topic?: string;
  full_text?: string;
  cluster?: number;        // Дубликат cluster_id для совместимости
}

export interface GraphLink {
  source: string;
  target: string;
  value?: number;
}

export interface Graph {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface ClusterInfo {
  id: number;
  name: string;
  color: string;
  count: number;  // Количество статей в кластере
}

export interface SearchResponse {
  nodes: GraphNode[];
  links: GraphLink[];
  topics: ClusterInfo[];  // Информация о кластерах/темах
  popular_topics?: string[];
  metadata?: {
    min_year: number;
    max_year: number;
  };
}

export interface FilterParams {
  years?: number[];
  clusters?: number[];
}

// Типы для состояния приложения
export interface AppState {
  searchQuery: string;
  selectedArticleId: string | null;
  filters: FilterParams;
}

// Типы для D3 визуализации
export interface D3Node extends GraphNode {
  fx?: number | null;
  fy?: number | null;
}

export interface D3Link {
  source: D3Node | string;
  target: D3Node | string;
  value?: number;
}
