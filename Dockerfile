# ==========================================
# STAGE 1: Сборка Фронтенда
# ==========================================
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Копируем только файлы зависимостей для кэширования слоев
COPY frontend/package*.json ./
RUN npm ci

# Копируем исходники и собираем
COPY frontend/ .
RUN npm run build

# ==========================================
# STAGE 2: Сборка Бэкенда и Финальный образ
# ==========================================
FROM python:3.12-slim

WORKDIR /app

# Установка системных зависимостей (нужны для сборки некоторых python пакетов)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости Python
COPY pyproject.toml .

# Устанавливаем зависимости. 
# Используем --no-cache-dir, чтобы образ был легче.
# Точка в конце означает установку текущего проекта (чтение pyproject.toml)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Копируем код бэкенда
COPY app ./app

# ! МАГИЯ !: Забираем собранный фронтенд из первого стейджа
# Кладем его туда, где FastAPI ожидает статику
COPY --from=frontend-builder /app/frontend/dist ./app/static

# Создаем папки для монтирования томов (чтобы избежать проблем с правами)
RUN mkdir -p arxiv_pdfs checkpoints triplets_data

# Переменные окружения для Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Открываем порт
EXPOSE 8000

# Запуск
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]