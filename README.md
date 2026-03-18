# NLP Semantic Lab

A hands-on learning laboratory for exploring modern NLP and semantic search technologies. Built with production-grade architecture featuring async FastAPI, vector embeddings, RAG, SEO generation, and event-driven design.

## Overview

This system scrapes Arabic articles, generates vector embeddings, stores them in PostgreSQL with pgvector, and enables semantic similarity search. It includes a full RAG (Retrieval-Augmented Generation) pipeline with pluggable LLM providers and an SEO content generation module powered by a fine-tuned Arabic language model.

### Key Features

- **Semantic Search** — HNSW-indexed vector similarity using pgvector (1024-dim embeddings)
- **RAG Pipeline** — Retrieve context from your data, call LLMs (ChatGPT, Claude, DeepSeek, Ollama), return grounded answers with cost tracking
- **SEO Generation** — Generate Arabic SEO meta descriptions using a QLoRA fine-tuned Command-R7B model
- **Article Scraping** — Async extraction of content, metadata, and SEO data from Arabic article URLs
- **Arabic NLP** — Full text normalization pipeline: hidden Unicode removal, diacritic stripping (CAMeL Tools), character normalization
- **Queue Processing** — Optional RabbitMQ integration for async embedding generation
- **Caching** — Optional Redis caching for similarity search results
- **API Security** — Optional API key authentication, CORS, rate limiting

### Architecture Highlights

- Modular feature-based design with clean separation of concerns
- Pure FastAPI dependency injection with `Annotated` type aliases
- Interface-based abstractions (`ILLMProvider`, `IWebScraper`) for testability
- Fully async I/O: database (asyncpg), HTTP (httpx), embeddings (Ollama), queue (aio-pika)
- Repository pattern for data access, service layer for orchestration
- Structured logging with correlation IDs (Seq integration)

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI 0.116.1, Uvicorn |
| Database | PostgreSQL + pgvector (async via asyncpg) |
| ORM | SQLAlchemy 2.0.43 (async) |
| Migrations | Alembic 1.16.5 |
| Embeddings | Ollama (nomic-embed-text, 1024-dim) |
| Scraping | httpx (async) + BeautifulSoup4 |
| LLM Providers | OpenAI SDK (ChatGPT, DeepSeek), Ollama, Claude (template) |
| Queue | RabbitMQ + aio-pika (optional) |
| Cache | Redis (optional) |
| Logging | seqlog to Seq (optional) |
| Arabic NLP | camel-tools |
| Fine-tuning | Transformers, PEFT, bitsandbytes, torch |
| Rate Limiting | slowapi |

## Getting Started

### Prerequisites

- Python 3.12+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension
- Ollama with `nomic-embed-text` model

Optional:
- RabbitMQ (for async embeddings)
- Redis (for caching)
- Seq (for structured logging)

### Installation

```bash
# Clone
git clone https://github.com/IbrahimMNada/nlp-semantic-lab.git
cd nlp-semantic-lab

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Dependencies
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env with your database URL and settings

# Database migrations
alembic upgrade head
```

### Ollama Setup

```bash
# Install Ollama: https://ollama.ai
# Pull the embedding model
ollama pull nomic-embed-text

# For local LLM chat (optional, for Ollama RAG provider)
ollama pull llama3.2
```

### Infrastructure (Docker)

The project includes a Docker Compose file for optional services:

```bash
cd docker
docker compose up -d
```

This starts: RabbitMQ (5672, management UI on 15672), Redis (6379), RedisInsight (8081), and Seq (5341).

### Running

```bash
# Start the API server
uvicorn src.main:app --reload
```

- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

#### With Async Embeddings (Optional)

Set `QUEUE_ENABLED=true` in `.env`, then run the consumer in a separate terminal:

```bash
python -m src.modules.data.consumers.embeddings_consumer
```

## Configuration

All settings are configured via environment variables (`.env` file). See `.env.example` for all options.

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_URL` | PostgreSQL connection string | *(required)* |
| `OLLAMA_URL` | Ollama embedding server | `http://localhost:11434` |
| `OLLAMA_MODEL_NAME` | Embedding model | `nomic-embed-text` |
| `LLM_PROVIDER` | RAG LLM provider (`chatgpt`, `claude`, `deepseek`, `ollama`) | `chatgpt` |
| `OPENAI_API_KEY` | OpenAI API key (if using ChatGPT) | |
| `DEEPSEEK_API_KEY` | DeepSeek API key (if using DeepSeek) | |
| `QUEUE_ENABLED` | Enable RabbitMQ async embeddings | `false` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `API_KEY_ENABLED` | Require API key authentication | `false` |
| `API_KEY` | API key value (sent via `X-API-Key` header) | |
| `CORS_ALLOWED_ORIGINS` | Allowed CORS origins | `["*"]` |
| `SEQ_ENABLED` | Enable Seq structured logging | `false` |
| `HF_TOKEN` | HuggingFace token (for SEO model) | |

## API Reference

### Health & Info

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/ping` | App name and version |

### Data Module — `/api/data`

Scraping, embedding generation, and vector search.

| Method | Path | Rate Limit | Description |
|--------|------|-----------|-------------|
| `POST` | `/process` | 10/min | Scrape URL, save article, generate embeddings |
| `POST` | `/search-similar` | 30/min | Find similar articles by URL (article-level vectors) |
| `POST` | `/search-similar-paragraphs` | 30/min | Find similar paragraphs by text (paragraph-level vectors) |
| `POST` | `/rebuild-index` | 2/min | Recreate HNSW vector indexes |
| `POST` | `/compute-article-embeddings` | — | Precompute article vectors from paragraph averages |
| `POST` | `/process-articles-without-embeddings` | — | Backfill embeddings for articles missing them |
| `GET` | `/random-articles?limit=10` | — | Get random articles with SEO metadata |

#### Process Article

```http
POST /api/data/process
Content-Type: application/json

{
  "url": "https://example.com/article"
}
```

Returns scraped content: title, author, paragraphs, and SEO metadata.

#### Search Similar Articles

```http
POST /api/data/search-similar
Content-Type: application/json

{
  "url": "https://example.com/article",
  "limit": 10,
  "threshold": 0.7
}
```

Returns similar articles with similarity scores (0.0–1.0).

#### Search Similar Paragraphs

```http
POST /api/data/search-similar-paragraphs
Content-Type: application/json

{
  "text": "your search text here",
  "limit": 10,
  "threshold": 0.5,
  "min_words": 10
}
```

Returns individual paragraphs ranked by similarity. Optimized for RAG context retrieval.

### RAG Module — `/api/rag`

Retrieval-Augmented Generation with pluggable LLM providers.

| Method | Path | Rate Limit | Description |
|--------|------|-----------|-------------|
| `POST` | `/search-context` | 30/min | Search paragraphs via the data module |
| `POST` | `/ask-with-context` | 20/min | Search context → build prompt → call LLM → return answer |

#### Ask With Context

```http
POST /api/rag/ask-with-context
Content-Type: application/json

{
  "question": "What is the article about?",
  "limit": 3,
  "similarity_threshold": 0.5,
  "temperature": 0.7,
  "max_tokens": 1000
}
```

Response:

```json
{
  "data": {
    "message": "AI-generated answer grounded in your data...",
    "tokens_used": 450,
    "context_used": true,
    "sources": ["https://example.com/article1", "https://example.com/article2"],
    "cost": 0.0012
  },
  "status_code": 0,
  "error_description": null
}
```

### SEO Generation Module — `/api/seo`

Arabic SEO content generation using a fine-tuned model.

| Method | Path | Rate Limit | Description |
|--------|------|-----------|-------------|
| `POST` | `/generate` | 10/min | Generate SEO meta description from input text |
| `GET` | `/dataset/random-samples?num_samples=10` | — | Get samples from the xyz SEO dataset |

#### Generate SEO Content

```http
POST /api/seo/generate
Content-Type: application/json

{
  "text": "Arabic article text...",
  "max_length": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

Response:

```json
{
  "data": {
    "generated_text": "Generated SEO description...",
    "input_text": "Original input...",
    "model_name": "ibrahim-nada/cmdr7b-ar-seo-qlora-v1-2025-12-20_19.08.11"
  },
  "status_code": 200,
  "error_description": null
}
```

### Authentication

When `API_KEY_ENABLED=true`, all module routes require an API key header:

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/api/data/random-articles
```

## Project Structure

```
nlp-semantic-lab/
├── src/
│   ├── main.py                     # FastAPI app, middleware, lifespan
│   ├── core/                       # Cross-cutting infrastructure
│   │   ├── config.py               # Pydantic settings (env vars)
│   │   ├── database.py             # Async SQLAlchemy session factory
│   │   ├── security.py             # API key auth, rate limiter
│   │   ├── cache_service.py        # Redis caching wrapper
│   │   ├── base.py                 # SQLAlchemy declarative base
│   │   ├── base_dtos/              # Generic ResponseDto[T]
│   │   └── exceptions/             # Custom exceptions
│   ├── abstractions/interfaces/    # ILLMProvider, IWebScraper
│   ├── contracts/data/             # Shared DTOs between modules
│   ├── shared/                     # Utilities
│   │   ├── arabic_text_processor.py  # CAMeL Tools integration
│   │   ├── text_utils.py           # Unicode cleanup, normalization
│   │   └── modules_http_client.py  # (legacy) HTTP client, replaced by event bus
│   │   └── event_bus.py            # Blinker-based in-process event bus
│   ├── app_routes/                 # Health/ping endpoints
│   └── modules/
│       ├── data/                   # Scraping, embeddings, vector search
│       │   ├── routes.py
│       │   ├── dependencies.py
│       │   ├── services/
│       │   │   ├── data_service.py           # Orchestration
│       │   │   ├── embedding_service.py      # Ollama embeddings
│       │   │   ├── article_repository.py     # Data access
│       │   │   └── web_scraper.py            # Article extraction
│       │   ├── consumers/
│       │   │   └── embeddings_consumer.py    # RabbitMQ consumer
│       │   ├── entities/                     # SQLAlchemy models
│       │   └── dtos/                         # Request/Response DTOs
│       ├── rag/                    # RAG pipeline
│       │   ├── routes.py
│       │   ├── dependencies.py     # LLM provider factory
│       │   ├── services/
│       │   │   └── rag_service.py            # Context retrieval + LLM
│       │   └── remote_models/      # LLM provider implementations
│       │       ├── chatgpt_consumer.py
│       │       ├── claude_consumer.py
│       │       ├── deepseek_consumer.py
│       │       └── ollama_consumer.py
│       ├── seo_generation/         # SEO content generation
│       │   ├── routes.py
│       │   ├── dependencies.py
│       │   └── services/
│       │       ├── seo_service.py            # Fine-tuned model inference
│       │       └── dataset_service.py        # HuggingFace dataset loading
│       └── model_training/         # Fine-tuning scripts
│           ├── fine-tuner.py
│           └── data_set_services.py
├── alembic/                        # Database migrations
├── docker/                         # Docker Compose (RabbitMQ, Redis, Seq)
├── .env.example                    # Environment template
├── requirements.txt
└── LICENSE
```

## Database Schema

PostgreSQL with pgvector extension. Four tables:

- **`articles`** — Scraped content with SEO metadata (title, author, meta tags, Open Graph, Twitter Cards as JSON). Includes computed `word_count` and `paragraph_count`.
- **`article_paragraphs`** — Individual paragraphs with `order_index`, FK to articles.
- **`paragraph_embeddings_1024`** — 1024-dim vectors per paragraph, HNSW indexed (cosine similarity).
- **`article_embedding_1024`** — 1024-dim vectors per article (average of paragraph vectors), HNSW indexed.

HNSW index parameters: `m=16`, `ef_construction=64`.

## Data Flow

### Scrape → Embed → Search

```
POST /api/data/process  { url: "..." }
  → WebScraper: fetch HTML, extract title/author/paragraphs/SEO
  → ArticleRepository: upsert article + save paragraphs
  → EmbeddingService: generate 1024-dim vector per paragraph (Ollama)
  → Compute article embedding (average of paragraph vectors)
  → Store in pgvector tables with HNSW index
```

### RAG Pipeline

```
POST /api/rag/ask-with-context  { question: "..." }
  → Search similar paragraphs via data module (blinker event bus)
  → Build provider-specific prompt with retrieved context
  → Call LLM (ChatGPT / DeepSeek / Ollama / Claude)
  → Return answer with sources and cost
```

## Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| Dependency Injection | `dependencies.py` per module | Testable wiring via `Annotated` + `Depends` |
| Repository | `ArticleRepository` | Isolates all SQL/ORM data access |
| Service Layer | `DataService`, `RagService`, `SeoService` | Business orchestration |
| Factory | `get_llm_provider()` | Runtime LLM provider selection |
| Interface Abstraction | `ILLMProvider`, `IWebScraper` | Pluggable implementations |
| Singleton | `@lru_cache()` on dependency factories | One instance per service |
| Generic DTO | `ResponseDto[T]` | Type-safe API responses |

### Adding a New Module

1. Create `src/modules/your_module/` with `routes.py`, `dependencies.py`, `__init__.py`
2. Define services and DTOs
3. Add a `register_your_module(app)` function in `__init__.py`
4. Call it from the lifespan function in `src/main.py`

## Development

```bash
# Run migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"

# Start API with hot reload
uvicorn src.main:app --reload

# Run tests
pytest tests/
```

### Monitoring

| Service | URL |
|---------|-----|
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| RabbitMQ Management | http://localhost:15672 (guest/guest) |
| RedisInsight | http://localhost:8081 |
| Seq (logs) | http://localhost:5341 |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `greenlet_spawn` errors | Ensure all DB operations use `await session.execute()`, not sync `session.query()` |
| Embeddings slow | Check Ollama server: `curl http://localhost:11434/api/tags`. Consider GPU. |
| Redis not connecting | App works without Redis. Check `redis-cli ping` returns PONG. |
| RabbitMQ connection refused | Verify it's running on port 5672. Check `RABBITMQ_URL` in `.env`. |
| SEO model not loading | Ensure `HF_TOKEN` is set and has access to the model repository. |

## Known Limitations

- Web scraper uses hard-coded CSS selectors — will break if target site structure changes
- Claude LLM provider is a template stub (requires `anthropic` package)
- No test suite yet (DI setup makes testing straightforward to add)
- Embedding batch processing is sequential (one paragraph at a time)

## License

MIT License — see [LICENSE](LICENSE) for details.

