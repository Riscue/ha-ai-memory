"""SentenceTransformer embedding engine."""
import logging
from typing import List

from custom_components.ai_memory.constants import EMBEDDINGS_MODEL_NAME

_LOGGER = logging.getLogger(__name__)


from homeassistant.util.package import install_package

class SentenceTransformerEngine:
    """SentenceTransformer embedding engine (Best Quality)."""

    def __init__(self, hass):
        """Initialize the engine."""
        self.hass = hass
        self.model = None
        self._model_loaded = False

    def _load_model(self):
        """Load the sentence-transformer model."""
        if self._model_loaded:
            return

        try:
            import sentence_transformers
        except ImportError:
            _LOGGER.info("sentence-transformers not found, installing...")
            try:
                install_package("sentence-transformers")
                import sentence_transformers
            except Exception as e:
                raise RuntimeError(f"Failed to install sentence-transformers: {e}")

        try:
            from sentence_transformers import SentenceTransformer
            _LOGGER.debug(f"Loading SentenceTransformer model: {EMBEDDINGS_MODEL_NAME}...")
            self.model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)
            self._model_loaded = True
            _LOGGER.debug("SentenceTransformer model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding."""
        if not self._model_loaded:
            self._load_model()

        if not self.model:
            raise RuntimeError("SentenceTransformer model not available")

        # encode returns a numpy array, convert to list
        embedding = self.model.encode(text)
        return embedding.tolist()

    def update_vocabulary(self, text: str):
        """No-op for SentenceTransformer."""
        pass
