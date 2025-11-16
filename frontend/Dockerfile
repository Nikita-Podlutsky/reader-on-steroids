# frontend/Dockerfile

# --- Этап 1: Сборка статики ---
FROM node:20 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
# Замените 'npm run build' на вашу команду сборки, если она другая
# Если у вас пока нет сборки, можно закомментировать строку ниже
# RUN npm run build

# --- Этап 2: Сервер ---
FROM nginx:1.25-alpine
# ВАЖНО: Если у вас нет шага сборки, копируйте исходники
# Если есть, замените ./src на путь к собранным файлам (например, /app/dist)
COPY ./src /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]