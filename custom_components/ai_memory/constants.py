"""Constants for AI Memory integration."""

DOMAIN = "ai_memory"

# Embedding model configuration
MEMORY_MAX_ENTRIES = 1000
SIMILARITY_THRESHOLD = 0.45
MEMORY_LIMIT = 5

# Embedding model configuration
EMBEDDINGS_VECTOR_DIM = 384  # Default; auto-detected from model at runtime

DEFAULT_MODEL = "bge-m3"
DEFAULT_REMOTE_URL = "http://127.0.0.1:11434"

# Embedding engine types (internal)
ENGINE_REMOTE = "remote"
ENGINE_TFIDF = "tfidf"

# Embedding providers (config-flow facing values stored in entry data)
PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai_compatible"
PROVIDER_TFIDF = "tfidf"

# User-friendly names for UI
PROVIDER_NAMES = {
    PROVIDER_OLLAMA: "Ollama",
    PROVIDER_OPENAI: "OpenAI-compatible (llama.cpp, LM Studio, vLLM, TEI)",
    PROVIDER_TFIDF: "TF-IDF (Fallback - No Dependencies)",
}

# Remote API flavors
API_FLAVOR_OLLAMA = "ollama"
API_FLAVOR_OPENAI = "openai"


def resolve_engine_type(provider: str) -> str:
    """Map a stored provider value to an internal engine type.

    Historically the only values were "remote" and "tfidf", and the
    migration rewrites "remote" to "ollama" — so anything that isn't
    TF-IDF is a remote-flavored provider.
    """
    if provider == PROVIDER_TFIDF:
        return ENGINE_TFIDF
    return ENGINE_REMOTE


def resolve_flavor(provider: str) -> str:
    """Map a stored provider value to a remote API flavor."""
    if provider == PROVIDER_OPENAI:
        return API_FLAVOR_OPENAI
    return API_FLAVOR_OLLAMA

# Default storage path (relative to HA config)
DEFAULT_STORAGE_PATH = "ai_memory.db"

# Palace defaults
DEFAULT_WING = "general"
DEFAULT_ROOM = "general"
DEFAULT_LAYER = 2

# L1 Promotion thresholds
L1_PROMOTION_THRESHOLD = 10
L1_DEMOTION_DAYS = 90

# Scope constants
SCOPE_PRIVATE = "private"
SCOPE_COMMON = "common"

# Database schema version
DB_VERSION = 1
