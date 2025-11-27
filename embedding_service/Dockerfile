FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn fastembed sentence-transformers

COPY embedding_service.py app.py

EXPOSE 11434

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "11434"]
