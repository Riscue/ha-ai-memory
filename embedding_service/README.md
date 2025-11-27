# External Embedding Service

This service provides an Ollama-compatible API for generating embeddings using **FastEmbed** or **SentenceTransformers**. It is designed to run as a microservice, allowing Home Assistant to offload embedding generation.

## Features

-   **Ollama-Compatible API**: Drop-in replacement for Ollama embeddings.
-   **Multiple Engines**: Support for `fastembed` (default) and `sentence_transformer`.
-   **Auto-Download**: Automatically downloads the default model on startup.
-   **Caching**: Caches models to a volume for persistence.

## Running with Docker

1.  Build the image:
    ```bash
    docker build -t ha-ai-memory-embedding .
    ```

2.  Run the container:
    ```bash
    docker run -d \
      -p 8000:11434 \
      -v $(pwd)/cache:/app/cache \
      -e EMBEDDING_ENGINE=fastembed \
      -e MODEL_NAME=qllama/bge-small-en-v1.5:q8_0 \
      --name embedding-service \
      ha-ai-memory-embedding
    ```

    The service will be available at `http://localhost:8000`.

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `EMBEDDING_ENGINE` | `fastembed` | Engine to use: `fastembed` or `sentence_transformer` |
| `MODEL_NAME` | `qllama/bge-small-en-v1.5:q8_0` | Default model to load on startup |
| `CACHE_DIR` | `/app/cache` | Directory to cache models |

## API Endpoints

-   `POST /api/pull`: Download a model.
    -   Body: `{"name": "qllama/bge-small-en-v1.5:q8_0"}`
-   `POST /api/embed`: Generate embeddings.
    -   Body: `{"model": "qllama/bge-small-en-v1.5:q8_0", "prompt": "Your text here"}`
    -   Response: `{"embedding": [0.1, 0.2, ...]}`

## Home Assistant Configuration

1.  Go to **Settings > Devices & Services**.
2.  Add **AI Memory** integration.
3.  Select **Remote Service (Ollama/FastEmbed Service)** as the embedding engine.
4.  Enter the URL (e.g., `http://localhost:8000` or `http://<ip>:8000`) and Model Name.
