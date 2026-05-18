FROM python:3.11-slim

LABEL org.opencontainers.image.title="Vietnamese eGov RAG Assistant" \
      org.opencontainers.image.description="Dockerized Flask RAG chatbot for Vietnamese administrative procedures"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CACHE_DIR=/app/.cache \
    DATA_DIR=/app/.cache/egov_data \
    SQLITE_PATH=/app/user_data/egov_bot.db \
    HF_LOCAL_FILES_ONLY=false \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/user_data /app/.cache

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

CMD ["sh", "-c", "gunicorn -w 1 --threads 4 --timeout 180 -b 0.0.0.0:${PORT:-7860} 'egov_bot.app:create_app()'"]
