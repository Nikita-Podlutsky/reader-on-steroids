import { useQuery } from '@tanstack/react-query';
import { scholarApi } from '@/services/api';

// Хук для поиска и загрузки графа (единственный запрос к бэкенду)
export const useSearch = (query: string, counter: number = 0, enabled: boolean = true) => {
  return useQuery({
    queryKey: ['search', query, counter], // Добавляем счетчик в ключ
    queryFn: () => scholarApi.analyzeGraph(query),
    enabled: enabled && query.length > 0,
    staleTime: 0, // Отключаем кэш - всегда делаем новый запрос
  });
};
