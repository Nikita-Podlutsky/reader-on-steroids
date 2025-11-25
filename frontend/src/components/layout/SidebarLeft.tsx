import React from 'react';
import { Filters } from '../ui/Filters';
import type { FilterParams, ClusterInfo } from '@/types';

interface SidebarLeftProps {
  onFilterChange: (filters: FilterParams) => void;
  clusters?: ClusterInfo[];
  popularTopics?: string[];
  onTopicClick?: (topic: string) => void;
  metadata?: {
    min_year: number;
    max_year: number;
  };
  visibleCount?: number;
  totalCount?: number;
}

export const SidebarLeft: React.FC<SidebarLeftProps> = ({
  onFilterChange,
  clusters = [],
  popularTopics = [],
  onTopicClick,
  metadata,
  visibleCount = 0,
  totalCount = 0,
}) => {
  return (
    <div className="w-64 bg-[#141e30]/80 p-5 border-r border-white/10 flex flex-col overflow-y-auto">
      {totalCount > 0 && (
        <div className="mb-4 p-3 bg-[#1e283c]/60 rounded-lg border border-white/10">
          <div className="text-sm text-gray-400">
            Показано {visibleCount}/{totalCount} статей
          </div>
        </div>
      )}
      <Filters onFilterChange={onFilterChange} clusters={clusters} metadata={metadata} />

      {popularTopics.length > 0 && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-4 pb-3 border-b border-white/10 text-secondary">
            Популярные темы
          </h2>
          <ul className="space-y-0">
            {popularTopics.map((topic) => (
              <li
                key={topic}
                onClick={() => onTopicClick?.(topic)}
                className="py-3 border-b border-white/5 cursor-pointer hover:text-secondary transition-colors last:border-0"
              >
                {topic}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
