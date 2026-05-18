# Architecture

The application is a Flask app factory in `src/egov_bot/app.py`. Runtime state is stored in `app.extensions`:

- settings from environment variables
- loaded data resources
- hybrid retriever
- RAG pipeline
- in-memory session manager
- SQLite repository

The request path is:

```text
UI -> Flask API -> follow-up detection -> retrieval -> context builder -> Gemini/fallback answer -> sources -> SQLite logs
```

The default data source is Hugging Face. `DATA_SOURCE=local` makes the loader read files from `DATA_DIR`.

