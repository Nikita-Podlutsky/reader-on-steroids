import React, { useState } from 'react';

interface SearchBarProps {
  onSearch: (query: string) => void;
  isCompact?: boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({ onSearch, isCompact = false }) => {
  const [query, setQuery] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim());
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && query.trim()) {
      onSearch(query.trim());
    }
  };

  return (
    <div className={`search-container ${isCompact ? 'max-w-2xl' : 'max-w-4xl'} mx-auto`}>
      <form onSubmit={handleSubmit} className="search-box flex bg-white/10 rounded-full overflow-hidden border border-white/20">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Поиск по идее (например: 'graph neural networks for chemistry')"
          className="flex-1 px-6 py-3 bg-transparent text-white placeholder-gray-400 focus:outline-none text-base"
        />
        <button
          type="submit"
          className="bg-secondary hover:bg-blue-600 text-white px-6 py-3 transition-colors"
        >
          🔍
        </button>
      </form>
    </div>
  );
};
