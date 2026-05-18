# Deployment

## Docker

```bash
cp .env.example .env
# edit .env and set GOOGLE_API_KEY
docker build -t egov-bot .
docker run --env-file .env -p 7860:7860 egov-bot
```

Use Compose for the development/demo workflow:

```bash
docker compose up --build
```

The Compose file persists `.cache` and `user_data` as host-mounted volumes. This
keeps Hugging Face resources, embedding model cache, SQLite logs, and feedback
between container restarts.

If the expected Hugging Face files have already been cached and you want startup
to avoid network checks, set:

```env
HF_LOCAL_FILES_ONLY=true
```

If you build a local index with `scripts/build_local_index.py`, set:

```env
DATA_SOURCE=local
DATA_DIR=.cache/egov_data
```

## Render

Use the Docker environment and set these variables:

```text
GOOGLE_API_KEY
HF_REPO_ID=HungBB/egov-bot-data
HF_REPO_TYPE=dataset
EMB_MODEL=AITeamVN/Vietnamese_Embedding
GENAI_MODEL=gemini-2.5-flash
APP_ENV=production
```

The Dockerfile binds Gunicorn to `${PORT:-7860}`.

## Hugging Face Spaces

Use Docker SDK and expose port `7860`. Add `GOOGLE_API_KEY` as a Space secret.
