"""Constants for AI Memory integration."""

DOMAIN = "ai_memory"

# Embedding model configuration
MEMORY_MAX_ENTRIES = 1000

# Embedding model configuration
EMBEDDINGS_VECTOR_DIM = 384

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_REMOTE_URL = "http://localhost:11434"

# Embedding engine types
ENGINE_REMOTE = "remote"
ENGINE_TFIDF = "tfidf"

# User-friendly names for UI
ENGINE_NAMES = {
    ENGINE_REMOTE: "Remote Service (Recommended - Docker/Ollama)",
    ENGINE_TFIDF: "TF-IDF (Fallback - No Dependencies)",
}

# Default storage path (relative to HA config)
DEFAULT_STORAGE_PATH = "ai_memory.db"
