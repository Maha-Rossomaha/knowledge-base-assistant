# Knowledge Base Assistant

## M0: Search Baseline

Система индексирует Markdown-документы из внешнего Git-репозитория
и выполняет полнотекстовый поиск по структурным чанкам.

### In scope

- сканирование Markdown-файлов;
- парсинг структуры заголовков;
- создание стабильных document_id и chunk_id;
- сохранение чанков в JSONL;
- BM25-поиск;
- CLI-команды index и search;
- минимальный retrieval benchmark;
- unit-тесты parser/chunker/search.

### Out of scope

- LLM generation;
- RAG;
- Qdrant;
- OpenSearch;
- PostgreSQL;
- FastAPI;
- agents;
- MCP;
- Kubernetes.