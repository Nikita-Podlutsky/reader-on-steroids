# Kotodex
  

Интерактивная карта научных публикаций. Вводишь тему — получаешь граф связанных статей с кластеризацией по темам и AI-чатом по содержанию каждой работы.

  

---

  

## Как это работает

  

Запрос уходит на ArXiv, статьи скачиваются и кодируются через `bge-small-en-v1.5`. Обученная модель (`UniversalScorer`) ранжирует их по релевантности запросу. Затем `BERTopic + UMAP + HDBSCAN` кластеризуют результаты и раскладывают по координатам графа. Ollama (или Google/OpenRouter) называет кластеры человекочитаемыми именами.

  

```

Браузер → nginx → backend → ArXiv

                           → ml-service  (ранжирование)

                           → BERTopic    (кластеризация)

                           → Ollama/LLM  (названия кластеров)

                  ← граф {nodes, links, topics}

```

  

---

  

## Быстрый старт

  

**Требования:** Docker, Docker Compose v2, NVIDIA GPU + [Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

  

```bash

cp .env.example .env


  

docker compose up --build

```

  

Открыть: **http://localhost**

  

### Без GPU

  

ML-сервис не поднимется, но поиск и граф работают через fallback (BGE + косинусное сходство):

  

```bash

docker compose up --build frontend backend

```

  

---

  

## Конфигурация

  

Файл `.env` в корне проекта:

  

```env

# AI-чат — достаточно одного

GOOGLE_AI_STUDIO_API_KEY=   # https://aistudio.google.com/

OPENROUTER_API_KEY=          # https://openrouter.ai/

  

# Ollama на хосте (по умолчанию уже настроен)

# OLLAMA_HOST=http://host.docker.internal:11434

```

  

Провайдер выбирается в интерфейсе чата. Если ключи не заданы — работает только Ollama (если запущен локально).

  

---

  

## Структура

  

```

kotodex/

├── app/                        # Backend (FastAPI)

│   ├── routers/endpoints.py    # /analyze_graph, /chat, /translate

│   ├── services/

│   │   ├── engine.py           # Embeddings, scoring, AI-чат

│   │   └── clusterizer.py      # BERTopic + UMAP + HDBSCAN

│   ├── arxiv_loader.py         # Загрузка статей и PDF

│   ├── config.py

│   └── main.py

├── ml-service/                 # Ранжирование (отдельный контейнер)

│   ├── api.py                  # /rank, /encode-query

│   ├── models.py               # UniversalScorer

│   └── config.py

├── frontend/                   # React + TypeScript

│   ├── src/

│   ├── Dockerfile              # node → nginx (multi-stage)

│   └── nginx.conf

├── checkpoints/

│   └── best_model.pt/          # Веса модели (LoRA адаптеры)

├── docker-compose.yml

└── .env.example

```

  

---

  

## API

  

| Метод | Путь | Описание |

|-------|------|----------|

| `POST` | `/analyze_graph` | Поиск + граф. Тело: `{ query, year_min?, year_max?, cluster_ids? }` |

| `POST` | `/chat` | AI-чат. Тело: `{ paper_text, question, history, provider }` |

| `POST` | `/translate` | Перевод EN→RU. Тело: `{ text }` |

  

ML-сервис (порт `8001`) не доступен снаружи — только для внутреннего взаимодействия с backend.

  

---

  

## Модель

[Подробно](README_MODEL.md)

`UniversalScorer` — двухэнкодерная архитектура:

  

- **QueryEncoderBGE** — `bge-small-en-v1.5` + QLoRA (r=64). Кодирует запрос.

- **DocumentEncoder** — `longformer-base-4096` + LoRA (r=32). Кодирует документ в контексте запроса.

  

Чекпоинт монтируется из `./checkpoints` как read-only volume. Если файл не найден — ml-service стартует с нулевыми весами и логирует предупреждение.

  

---

  

## Решение проблем


**`invalid load key 'v'`** — файл модели это Git LFS pointer, не реальные веса:

```bash

git lfs install && git lfs pull

```

  

**Изменения не применяются** — кэш Docker:

```bash

docker compose build --no-cache && docker compose up -d

```