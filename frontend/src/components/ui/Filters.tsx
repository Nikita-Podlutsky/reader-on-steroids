import React, { useState, useEffect } from 'react';
import type { FilterParams, ClusterInfo } from '@/types';

interface FiltersProps {
  onFilterChange: (filters: FilterParams) => void;
  clusters?: ClusterInfo[];
  metadata?: {
    min_year: number;
    max_year: number;
  };
}

export const Filters: React.FC<FiltersProps> = ({ onFilterChange, clusters = [], metadata }) => {
  const minYear = metadata?.min_year || 2000;
  const maxYear = metadata?.max_year || 2025;

  const [selectedClusters, setSelectedClusters] = useState<number[]>([]);
  const [yearRange, setYearRange] = useState<number[]>([minYear, maxYear]);

  // Сбрасываем все фильтры при новом поиске (когда приходят новые кластеры)
  useEffect(() => {
    setSelectedClusters([]);
    setYearRange([minYear, maxYear]);
    // Сразу сообщаем родителю что фильтры пустые
    onFilterChange({});
    console.log('🔄 Reset filters UI on new data');
  }, [clusters, minYear, maxYear]); // Зависит от clusters - при новом поиске они меняются

  const applyFilters = () => {
    const filters: FilterParams = {};
    
    // Добавляем годы только если они не полный диапазон
    if (yearRange[0] > minYear || yearRange[1] < maxYear) {
      const years = [];
      for (let year = yearRange[0]; year <= yearRange[1]; year++) {
        years.push(year);
      }
      filters.years = years;
    }

    // Добавляем кластеры если выбраны
    if (selectedClusters.length > 0) {
      filters.clusters = selectedClusters;
    }

    onFilterChange(filters);
  };

  const handleClusterToggle = (clusterId: number) => {
    setSelectedClusters(prev => 
      prev.includes(clusterId) 
        ? prev.filter(id => id !== clusterId)
        : [...prev, clusterId]
    );
  };

  const handleYearRangeChange = (newRange: number[]) => {
    setYearRange(newRange);
  };



  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold mb-4 pb-3 border-b border-white/10 text-secondary">
        Фильтры
      </h2>

      <div className="filter-group">
        <label className="block mb-3 text-sm text-gray-400">Кластеры</label>
        <div className="space-y-2 max-h-40 overflow-y-auto">
          {clusters.map((cluster) => (
            <label key={cluster.id} className="flex items-center space-x-2 cursor-pointer hover:bg-white/5 p-2 rounded">
              <input
                type="checkbox"
                checked={selectedClusters.includes(cluster.id)}
                onChange={() => handleClusterToggle(cluster.id)}
                className="w-4 h-4 rounded border-white/10 text-secondary focus:ring-secondary"
              />
              <div className="flex items-center space-x-2 flex-1">
                <div 
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: cluster.color }}
                />
                <span className="text-sm text-white">{cluster.name}</span>
              </div>
            </label>
          ))}
        </div>
      </div>

      <div className="filter-group">
        <label className="block mb-2 text-sm text-gray-400">
          Годы: {yearRange[0]} - {yearRange[1]}
        </label>
        <div className="relative pt-1 pb-4">
          <div className="relative h-8 flex items-center">
            <div className="absolute w-full h-2 bg-gray-700 rounded" style={{ pointerEvents: 'none' }}>
              <div
                className="absolute h-2 bg-secondary rounded"
                style={{
                  left: `${((yearRange[0] - minYear) / (maxYear - minYear)) * 100}%`,
                  right: `${100 - ((yearRange[1] - minYear) / (maxYear - minYear)) * 100}%`,
                  pointerEvents: 'none',
                }}
              />
            </div>
            <input
              type="range"
              min={minYear}
              max={maxYear}
              value={yearRange[1]}
              onChange={(e) => {
                const val = Number(e.target.value);
                if (val >= yearRange[0]) {
                  handleYearRangeChange([yearRange[0], val]);
                }
              }}
              className="absolute w-full bg-transparent appearance-none cursor-pointer range-slider"
            />
            <input
              type="range"
              min={minYear}
              max={maxYear}
              value={yearRange[0]}
              onChange={(e) => {
                const val = Number(e.target.value);
                if (val <= yearRange[1]) {
                  handleYearRangeChange([val, yearRange[1]]);
                }
              }}
              className="absolute w-full bg-transparent appearance-none cursor-pointer range-slider"
            />
          </div>
        </div>
      </div>

      <button
        onClick={applyFilters}
        className="w-full bg-secondary hover:bg-blue-600 text-white py-2 rounded-md transition-colors"
      >
        Применить фильтры
      </button>
    </div>
  );
};
