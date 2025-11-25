import axios from 'axios';
import type { SearchResponse } from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const scholarApi = {
  // Единственный эндпоинт для поиска и получения графа (как у тимлида)
  analyzeGraph: async (query: string): Promise<SearchResponse> => {
    const response = await api.post<SearchResponse>('/analyze_graph', { query });
    return response.data;
  },

  // Перевод текста
  translate: async (text: string): Promise<{ translation: string }> => {
    const response = await api.post<{ translation: string }>('/translate', { text });
    return response.data;
  },

  // Чат с AI (используется в AIChat компоненте)
  chat: async (paperText: string, history: Array<{ role: string; content: string }>, question: string): Promise<{ answer: string }> => {
    const response = await api.post<{ answer: string }>('/chat', {
      paper_text: paperText,
      history,
      question,
    });
    return response.data;
  },
};
