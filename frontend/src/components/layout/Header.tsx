import React from 'react';

interface HeaderProps {
  hasSearched?: boolean;
  onSearch?: (query: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ hasSearched, onSearch }) => {
  const [searchValue, setSearchValue] = React.useState('');

  const handleSearch = () => {
    if (searchValue.trim() && onSearch) {
      onSearch(searchValue.trim());
      setSearchValue('');
    }
  };

  return (
    <header className="bg-primary px-5 py-4 flex justify-between items-center shadow-lg z-[100]">
      <div className="flex items-center gap-3">
        <div className="text-3xl">📚</div>
        <h1 className="text-2xl font-bold bg-gradient-to-r from-secondary to-accent bg-clip-text text-transparent">
          ScholarMap
        </h1>
      </div>
      
      {hasSearched && onSearch && (
        <div className="flex-1 max-w-2xl mx-8 flex gap-2">
          <input
            type="text"
            value={searchValue}
            onChange={(e) => setSearchValue(e.target.value)}
            placeholder="Поиск научных статей..."
            className="flex-1 bg-[#1a1f2e] text-gray-200 px-4 py-2 rounded-lg border border-gray-700 focus:outline-none focus:border-secondary"
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                handleSearch();
              }
            }}
          />
          <button
            onClick={handleSearch}
            className="bg-secondary hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors flex items-center justify-center"
            title="Поиск"
          >
            🔍
          </button>
        </div>
      )}
      
      <div className="w-10"></div>
    </header>
  );
};
