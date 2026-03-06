import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Header } from './components/layout/Header';
import { SidebarLeft } from './components/layout/SidebarLeft';
import { SidebarRight } from './components/layout/SidebarRight';
import { SearchBar } from './components/ui/SearchBar';
import { GraphVisualization } from './components/graph/GraphVisualization';
import { useSearch } from './hooks/useApi';
import type { FilterParams, Graph, GraphNode } from './types';

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchCounter, setSearchCounter] = useState(0); // Счетчик для принудительного обновления
  const [hasSearched, setHasSearched] = useState(false);
  const [selectedArticleId, setSelectedArticleId] = useState<string | null>(null);
  const [selectedArticle, setSelectedArticle] = useState<GraphNode | null>(null);
  const [currentGraph, setCurrentGraph] = useState<Graph | null>(null);
  const [activeFilters, setActiveFilters] = useState<FilterParams>({});
  const [visibleNodesCount, setVisibleNodesCount] = useState<number>(0);
  
  // Мемоизируем filters чтобы не триггерить useEffect в GraphVisualization
  const filtersKey = JSON.stringify(activeFilters);
  const memoizedFilters = useMemo(() => activeFilters, [filtersKey]);
  
  // Храним все узлы для клиентской фильтрации
  const allNodesRef = useRef<GraphNode[]>([]);
  const allLinksRef = useRef<{ source: string; target: string }[]>([]);

  // Единственный запрос к API (включаем счетчик в ключ для принудительного обновления)
  const { data: searchData, isLoading: isSearchLoading } = useSearch(searchQuery, searchCounter, hasSearched);

  // Обновляем граф при получении данных поиска
  useEffect(() => {
    if (searchData) {
      
      // Сохраняем все узлы и связи
      allNodesRef.current = searchData.nodes || [];
      allLinksRef.current = searchData.links || [];
      
      // Вычисляем metadata (годы) если её нет
      if (!searchData.metadata && searchData.nodes.length > 0) {
        const years = searchData.nodes
          .map(n => n.year)
          .filter((y): y is number => y !== undefined);
        
        if (years.length > 0) {
          searchData.metadata = {
            min_year: Math.min(...years),
            max_year: Math.max(...years)
          };
        }
      }
      
      // Устанавливаем полный граф ОДИН РАЗ (не пересоздаём при каждом рендере)
      setCurrentGraph(searchData);
    }
  }, [searchData]);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    setSearchCounter(prev => prev + 1); // Увеличиваем счетчик для принудительного обновления
    setHasSearched(true);
    setSelectedArticleId(null);
    setSelectedArticle(null); // Сбрасываем выбранную статью
    // Сбрасываем фильтры при новом поиске
    setActiveFilters({});
  };

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedArticleId(nodeId);
    // Находим полные данные узла
    const node = allNodesRef.current.find(n => n.id === nodeId);
    setSelectedArticle(node || null);
  }, []); // Нет зависимостей - функция стабильна

  const handleFilterChange = (filters: FilterParams) => {
    setActiveFilters(filters);
  };

  const handleTopicClick = (topic: string) => {
    setSearchQuery(topic);
    setHasSearched(true);
  };

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0a0e17] text-gray-200 overflow-hidden">
      {!hasSearched ? (
        // Начальное состояние: пустая страница с поиском по центру
        <>
          <Header />
          <div className="flex-1 flex items-center justify-center">
            <div className="w-full max-w-3xl px-4">
              <div className="text-center mb-8">
                <div className="text-6xl mb-4">📚</div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-secondary to-accent bg-clip-text text-transparent mb-2">
                  Kotodex
                </h1>
                <p className="text-gray-400 text-lg">Карта научных идей</p>
              </div>
              <SearchBar onSearch={handleSearch} />
            </div>
          </div>
        </>
      ) : (
        // Состояние после поиска: полный интерфейс
        <>
          <Header hasSearched={hasSearched} onSearch={handleSearch} />

          <div className="flex flex-1 overflow-hidden bg-[#0a0e17]">
            <SidebarLeft
              onFilterChange={handleFilterChange}
              clusters={searchData?.topics}
              popularTopics={searchData?.popular_topics}
              onTopicClick={handleTopicClick}
              metadata={searchData?.metadata}
              visibleCount={visibleNodesCount}
              totalCount={allNodesRef.current.length}
            />

            <div className="flex-1 relative bg-[#0a0e17]">
              {isSearchLoading ? (
                <div className="flex items-center justify-center h-full bg-[#0a0e17]">
                  <div className="text-xl text-gray-400">Загрузка графа...</div>
                </div>
              ) : currentGraph ? (
                <GraphVisualization
                  data={currentGraph}
                  onNodeClick={handleNodeClick}
                  selectedNodeId={selectedArticleId}
                  filters={memoizedFilters}
                  onVisibleCountChange={setVisibleNodesCount}
                />
              ) : (
                <div className="flex items-center justify-center h-full bg-[#0a0e17]">
                  <div className="text-xl text-gray-400">Нет данных для отображения</div>
                </div>
              )}
            </div>

            {selectedArticle && (
              <SidebarRight 
                selectedArticle={selectedArticle} 
                onFindSimilar={handleSearch}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default App;
