"""FastEmbed embedding engine."""
import logging
import os
from typing import List

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class FastEmbedEngine:
    """FastEmbed embedding engine (ONNX-based, optimized for RPi4)."""

    def __init__(self, hass: HomeAssistant):
        """Initialize the engine."""
        self.hass = hass
        self.model = None
        self._model_loaded = False

    def _load_model(self):
        """Load FastEmbed model (downloads on first use)."""
        if self._model_loaded:
            return

        try:
            from fastembed import TextEmbedding
            
            cache_dir = self.hass.config.path(".storage", "fastembed_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            _LOGGER.debug("Loading FastEmbed model (cache: %s)...", cache_dir)
            
            # Auto-downloads to cache_dir
            self.model = TextEmbedding(
                model_name="BAAI/bge-small-en-v1.5",  # Good default, or use "sentence-transformers/all-MiniLM-L6-v2" if supported by library version
                cache_dir=cache_dir
            )
            self._model_loaded = True
            _LOGGER.debug("FastEmbed model loaded successfully")
            
        except ImportError:
            raise RuntimeError(
                "fastembed not installed. "
                "Please install it manually: pip install fastembed"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FastEmbed: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding."""
        if not self._model_loaded:
            self._load_model()

        if not self.model:
            raise RuntimeError("FastEmbed model not available")

        # embed() returns generator of numpy arrays
        # We pass a list with one text, get the first result
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

    def update_vocabulary(self, text: str):
        """No-op for FastEmbed."""
        pass
