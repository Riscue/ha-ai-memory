"""Constants for AI Memory integration."""

DOMAIN = "ai_memory"

# Embedding model configuration
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_VECTOR_DIM = 384

# Embedding engine types
ENGINE_SENTENCE_TRANSFORMER = "sentence_transformer"
ENGINE_FASTEMBED = "fastembed"
ENGINE_TFIDF = "tfidf"
ENGINE_AUTO = "auto"

# Fallback order (try in sequence)
ENGINE_FALLBACK_ORDER = [
    ENGINE_SENTENCE_TRANSFORMER,
    ENGINE_FASTEMBED,
    ENGINE_TFIDF,
]

# User-friendly names for UI
ENGINE_NAMES = {
    ENGINE_SENTENCE_TRANSFORMER: "SentenceTransformer (Best Quality, ~500MB)",
    ENGINE_FASTEMBED: "FastEmbed (Good Quality, RPi4 Optimized, ~100MB)",
    ENGINE_TFIDF: "TF-IDF (Lightweight, No Dependencies, ~1MB)",
    ENGINE_AUTO: "Auto (Recommended - Try all, fallback to TF-IDF)",
}

# Default storage path (relative to HA config)
DEFAULT_STORAGE_PATH = "ai_memory.db"
