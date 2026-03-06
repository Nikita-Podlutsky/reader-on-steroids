import React from 'react';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'ai';
  timestamp: Date;
}

interface ChatProps {
  paperText?: string;
  messages?: Message[];
  provider?: 'ollama' | 'openrouter' | 'google';
  onProviderChange?: (provider: 'ollama' | 'openrouter' | 'google') => void;
  isSending?: boolean;
}

export const AIChat: React.FC<ChatProps> = ({ messages = [], provider = 'ollama', onProviderChange, isSending = false }) => {
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  // Автоскролл до последнего сообщения
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col h-[400px] bg-[#1a1f2e] rounded-lg border border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-gray-700 flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-xl">🤖</span>
          <span className="font-semibold text-gray-200">AI Ассистент</span>
        </div>
      </div>

      {/* Provider Selector */}
      <div className="p-3 border-b border-gray-700 flex-shrink-0">
        <select
          value={provider}
          onChange={(e) => onProviderChange?.(e.target.value as 'ollama' | 'openrouter' | 'google')}
          className="w-full bg-gray-800 text-gray-200 px-3 py-2 rounded-lg border border-gray-600 focus:outline-none focus:border-secondary text-sm"
        >
          <option value="ollama">Ollama (локальный)</option>
          <option value="openrouter">OpenRouter.ai (бесплатный)</option>
          <option value="google">Google AI Studio</option>
        </select>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 text-sm mt-8">
            Начните диалог с AI ассистентом
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={message.sender === 'user' ? 'flex justify-end' : 'w-full'}
            >
              <div
                className={`${
                  message.sender === 'user'
                    ? 'max-w-[80%] bg-secondary text-white rounded-lg p-3'
                    : 'w-full text-gray-200'
                }`}
              >
                <p className="text-sm whitespace-pre-wrap">{message.text}</p>
                <span className="text-xs opacity-70 mt-1 block">
                  {message.timestamp.toLocaleTimeString('ru-RU', {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </span>
              </div>
            </div>
          ))
        )}
        {isSending && (
          <div className="w-full">
            <div className="text-gray-200">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};
