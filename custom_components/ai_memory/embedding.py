"""Embedding Engine for AI Memory."""
import logging
from typing import List

from homeassistant.core import HomeAssistant

from custom_components.ai_memory.constants import EMBEDDINGS_MODEL_NAME

_LOGGER = logging.getLogger(__name__)


class EmbeddingEngine:
    """Engine to generate vector embeddings from text."""

    def __init__(self, hass: HomeAssistant):
        """Initialize the embedding engine."""
        self.hass = hass
        self._model = None
        self._model_loaded = False

    def _load_model(self):
        """Load the sentence-transformer model.
        
        This is a blocking operation and should be run in an executor.
        """
        if self._model_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer
            _LOGGER.debug(f"Loading embedding model: {EMBEDDINGS_MODEL_NAME}...")
            self._model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)
            self._model_loaded = True
            _LOGGER.debug("Embedding model loaded successfully")
        except ImportError:
            _LOGGER.error(
                "sentence-transformers not installed. "
                "Please wait for Home Assistant to install dependencies or install manually."
            )
            raise
        except Exception as e:
            _LOGGER.error(f"Failed to load embedding model: {e}")
            raise

    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Generate embedding synchronously."""
        if not self._model_loaded:
            self._load_model()

        if not self._model:
            raise RuntimeError("Embedding model not available")

        # Generate embedding
        # encode returns a numpy array, convert to list
        embedding = self._model.encode(text)
        return embedding.tolist()

    async def async_generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text asynchronously.
        
        This runs the heavy model inference in the executor to avoid blocking the loop.
        """
        if not text:
            return []

        return await self.hass.async_add_executor_job(
            self._generate_embedding_sync,
            text
        )
