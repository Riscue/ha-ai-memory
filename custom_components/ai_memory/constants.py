"""Constants for AI Memory integration."""

DOMAIN = "ai_memory"

# Embedding model configuration
MEMORY_MAX_ENTRIES = 1000

# Embedding model configuration
EMBEDDINGS_VECTOR_DIM = 384

# Embedding engine types
ENGINE_TFIDF = "tfidf"
ENGINE_REMOTE = "remote"

# User-friendly names for UI
ENGINE_NAMES = {
    ENGINE_REMOTE: "Remote Service (Recommended - Docker/Ollama)",
    ENGINE_TFIDF: "TF-IDF (Fallback - No Dependencies)",
}

# Default storage path (relative to HA config)
DEFAULT_STORAGE_PATH = "ai_memory.db"
