import React, { useState, useEffect } from 'react';
import { AIChat } from '../ui/AIChat';
import type { GraphNode } from '@/types';
import axios from 'axios';

interface SidebarRightProps {
  selectedArticle: GraphNode | null;
  onFindSimilar?: (title: string) => void;
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export const SidebarRight: React.FC<SidebarRightProps> = ({ selectedArticle, onFindSimilar }) => {
  const [translatedAbstract, setTranslatedAbstract] = useState<string>('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [messages, setMessages] = useState<Array<{id: string, text: string, sender: 'user' | 'ai', timestamp: Date}>>([]);
  const [provider, setProvider] = useState<'ollama' | 'openrouter' | 'google'>('ollama');
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Автоматическое изменение размера textarea
  React.useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      const maxHeight = 40 * 2.5;
      textareaRef.current.style.height = Math.min(scrollHeight, maxHeight) + 'px';
    }
  }, [inputValue]);

  const handleSend = async () => {
    if (!inputValue.trim() || isSending || !selectedArticle) return;

    const userMessage = {
      id: Date.now().toString(),
      text: inputValue,
      sender: 'user' as const,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsSending(true);

    const currentInput = inputValue;
    setInputValue('');

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        paper_text: selectedArticle.full_text || selectedArticle.abstract,
        history: messages.map(m => ({
          role: m.sender === 'user' ? 'user' : 'assistant',
          content: m.text,
        })),
        question: currentInput,
        provider: provider,
      });

      const aiMessage = {
        id: (Date.now() + 1).toString(),
        text: response.data.answer,
        sender: 'ai' as const,
        timestamp: new Date(),
      };
      
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: 'Ошибка при отправке сообщения. Попробуйте позже.',
        sender: 'ai' as const,
        timestamp: new Date(),
      };
      
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsSending(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (!selectedArticle) {
      setTranslatedAbstract('');
      return;
    }

    // Сбрасываем перевод и начинаем новый запрос
    setTranslatedAbstract('');
    setIsTranslating(true);

    const translateAbstract = async () => {
      try {
        const response = await axios.post(`${API_BASE_URL}/translate`, {
          text: selectedArticle.abstract
        });
        setTranslatedAbstract(response.data.translation);
      } catch (error) {
        console.error('Translation error:', error);
        setTranslatedAbstract(selectedArticle.abstract); // Fallback к оригиналу
      } finally {
        setIsTranslating(false);
      }
    };

    translateAbstract();
  }, [selectedArticle]);
  if (!selectedArticle) {
    return (
      <div className="w-[28rem] bg-[#141e30]/80 p-5 border-l border-white/10 flex flex-col overflow-y-auto h-full">
        <div className="text-center text-gray-400 mt-10">
          Выберите статью для просмотра деталей
        </div>
      </div>
    );
  }

  return (
    <div className="w-[28rem] bg-[#141e30]/80 border-l border-white/10 flex flex-col h-full relative">
      <div className="flex-1 overflow-y-auto p-5 pb-[100px]">
      <div className="mb-4 pb-3 border-b border-white/10">
        <h2 className="text-xl font-semibold text-secondary break-words leading-tight">
          {selectedArticle.label}
        </h2>
      </div>

      {/* Кнопка поиска похожих */}
      {onFindSimilar && (
        <button
          onClick={() => onFindSimilar(selectedArticle.label)}
          className="w-full mb-4 bg-accent hover:bg-orange-600 text-white py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
        >
          <span>🔍</span>
          <span>Найти похожие статьи</span>
        </button>
      )}

      {/* Аннотация */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <div className="text-xs text-gray-500 font-bold">АННОТАЦИЯ</div>
        </div>
        <div className="bg-[#1e283c]/60 rounded-lg p-4 border border-white/10">
          {isTranslating ? (
            <p className="text-sm text-gray-500 leading-relaxed">
              Перевод...
            </p>
          ) : (
            <p className="text-sm text-gray-300 leading-relaxed">
              {translatedAbstract || selectedArticle.abstract}
            </p>
          )}
        </div>
      </div>

      {/* Метаданные */}
      <div className="flex flex-wrap gap-2 text-xs mb-4">
        <span 
          className="px-3 py-1 rounded-full"
          style={{ backgroundColor: selectedArticle.color + '40', color: selectedArticle.color }}
        >
          {selectedArticle.cluster_name}
        </span>
        {selectedArticle.year && (
          <span className="px-3 py-1 bg-primary/20 rounded-full text-gray-300">
            Год: {selectedArticle.year}
          </span>
        )}
      </div>

      {/* Чат с AI */}
      <div className="mt-4">
        <AIChat paperText={selectedArticle.full_text || selectedArticle.abstract} messages={messages} provider={provider} onProviderChange={setProvider} isSending={isSending} />
      </div>
      </div>

      {/* Фиксированный footer с textarea */}
      <div className="absolute bottom-0 left-0 right-0 p-3 border-t border-white/10 bg-[#141e30]/95 backdrop-blur-sm flex-shrink-0">
        <div className="flex gap-2">
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Задай вопрос о статье"
            disabled={isSending}
            rows={1}
            className="flex-1 bg-gray-800 text-gray-200 px-3 py-2 rounded-lg border border-gray-600 focus:outline-none focus:border-secondary disabled:opacity-50 disabled:cursor-not-allowed text-sm resize-none overflow-y-auto"
            style={{ minHeight: '40px', maxHeight: '100px' }}
          />
          <button
            onClick={handleSend}
            disabled={!inputValue.trim() || isSending}
            className="px-4 py-2 bg-secondary text-white rounded-lg hover:opacity-90 transition-all disabled:opacity-50 disabled:cursor-not-allowed self-end"
          >
            Отправить
          </button>
        </div>
      </div>
    </div>
  );
};
