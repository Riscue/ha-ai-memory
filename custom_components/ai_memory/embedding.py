"""Embedding Engine for AI Memory with multiple backend support."""
import logging
from typing import List, Optional

from homeassistant.core import HomeAssistant

from .constants import (
    ENGINE_REMOTE,
    ENGINE_TFIDF,
    EMBEDDINGS_VECTOR_DIM,
)

_LOGGER = logging.getLogger(__name__)


class EmbeddingEngine:
    """Engine to generate vector embeddings from text with multiple backends.
    
    Supports multiple embedding engines with automatic fallback:
    1. Remote Service (Recommended)
    2. TF-IDF (Lightweight, No Dependencies)
    """

    def __init__(self, hass: HomeAssistant, engine_type: str = ENGINE_TFIDF, config_data: dict = None):
        """Initialize the embedding engine.
        
        Args:
            hass: Home Assistant instance
            engine_type: Engine type to use (remote, tfidf)
            config_data: Configuration data (for remote engine)
        """
        self.hass = hass
        self._engine_type = engine_type
        self._config_data = config_data or {}
        self._engine = None
        self._engine_name = None
        self._initialized = False

    def _create_engine(self, engine_type: str):
        """Create specific engine instance."""
        try:
            if engine_type == ENGINE_TFIDF:
                from .embedding_tfidf import TFIDFEmbeddingEngine
                return TFIDFEmbeddingEngine(self.hass, EMBEDDINGS_VECTOR_DIM)

            elif engine_type == ENGINE_REMOTE:
                from .embedding_remote import RemoteEmbeddingEngine
                return RemoteEmbeddingEngine(self.hass, self._config_data)

        except ImportError as e:
            _LOGGER.debug("Engine %s import failed: %s", engine_type, e)
            return None
        except Exception as e:
            _LOGGER.warning("Engine %s creation failed: %s", engine_type, e)
            return None

        return None

    def _try_initialize_engine(self, engine_type: str) -> bool:
        """Try to initialize a specific engine."""
        _LOGGER.debug("Attempting to initialize engine: %s", engine_type)

        engine = self._create_engine(engine_type)
        if not engine:
            return False

        try:
            # Test generation (lightweight check)
            # For some engines this triggers model load
            if hasattr(engine, '_load_model'):
                engine._load_model()

            self._engine = engine
            self._engine_name = engine_type
            _LOGGER.info("Embedding engine initialized: %s", engine_type)
            return True

        except Exception as e:
            _LOGGER.warning("Failed to initialize %s: %s", engine_type, e)
            return False

    def _initialize_engine(self):
        """Initialize the embedding engine with strict fallback to TF-IDF."""
        if self._initialized:
            return

        _LOGGER.debug("Initializing embedding engine (requested: %s)", self._engine_type)

        success = False

        # 1. Try requested engine
        if self._try_initialize_engine(self._engine_type):
            success = True
        else:
            _LOGGER.warning(
                "Requested engine '%s' failed. Falling back to TF-IDF.",
                self._engine_type
            )

            # 2. Strict Fallback to TF-IDF
            # Only if the requested engine was NOT TF-IDF (to avoid infinite loop or redundant check)
            if self._engine_type != ENGINE_TFIDF:
                if self._try_initialize_engine(ENGINE_TFIDF):
                    success = True
                else:
                    _LOGGER.error("Fallback to TF-IDF failed.")

        if not success:
            raise RuntimeError("No embedding engine available. Please check logs.")

        self._initialized = True

    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Generate embedding synchronously."""
        if not self._initialized:
            self._initialize_engine()

        if not self._engine:
            raise RuntimeError("Embedding engine not initialized")

        return self._engine.generate_embedding(text)

    async def async_generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text asynchronously."""
        if not text:
            return [0.0] * EMBEDDINGS_VECTOR_DIM

        return await self.hass.async_add_executor_job(
            self._generate_embedding_sync,
            text
        )

    async def async_update_vocabulary(self, text: str):
        """Update vocabulary for engines that need it."""
        if not self._initialized:
            self._initialize_engine()

        if hasattr(self._engine, 'update_vocabulary'):
            await self.hass.async_add_executor_job(
                self._engine.update_vocabulary,
                text
            )

    async def async_initialize(self):
        """Initialize the engine asynchronously (non-blocking)."""
        if self._initialized:
            return

        await self.hass.async_add_executor_job(self._initialize_engine)

    @property
    def engine_name(self) -> Optional[str]:
        """Get the name of the active engine."""
        return self._engine_name
