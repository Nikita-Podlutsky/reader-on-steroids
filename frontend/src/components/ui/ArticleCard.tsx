import React from 'react';
import type { GraphNode } from '@/types';

interface ArticleCardProps {
  article: GraphNode;
}

export const ArticleCard: React.FC<ArticleCardProps> = ({ article }) => {
  const articleUrl = `https://scholar.google.com/scholar?q=${encodeURIComponent(article.label)}`;
  
  return (
    <div className="bg-[#1e283c]/60 rounded-lg p-4 mb-4 border border-white/10">
      <a 
        href={articleUrl} 
        target="_blank" 
        rel="noopener noreferrer"
        className="text-lg font-semibold mb-3 text-secondary hover:text-blue-400 transition-colors block"
        title="Найти в Google Scholar"
      >
        {article.label}
      </a>
      
      <div className="flex justify-between text-xs text-gray-400 mb-2">
        {article.year && <span>{article.year}</span>}
        <span>Кластер: {article.cluster_name}</span>
      </div>

      <p className="text-sm text-gray-400 leading-relaxed mb-4">
        {article.abstract}
      </p>

      <div className="flex flex-wrap gap-2">
        <span
          className="px-2 py-1 rounded-full text-xs"
          style={{ backgroundColor: article.color + '40', color: article.color }}
        >
          {article.topic || article.cluster_name}
        </span>
      </div>
    </div>
  );
};
