"""Remote embedding engine (Ollama/FastEmbed Service)."""
import logging
from typing import List, Dict, Any

import aiohttp
import requests
from homeassistant.core import HomeAssistant

from ..constants import DEFAULT_MODEL, DEFAULT_REMOTE_URL

_LOGGER = logging.getLogger(__name__)


class RemoteEmbeddingEngine:
    """Remote embedding engine using Ollama-compatible API."""

    def __init__(self, hass: HomeAssistant, config_data: Dict[str, Any]):
        """Initialize the engine."""
        self.hass = hass
        self.remote_url = config_data.get("remote_url", DEFAULT_REMOTE_URL)
        self.model_name = config_data.get("model_name", DEFAULT_MODEL)
        self._model_loaded = False

    def _load_model(self):
        """Trigger model load on remote server."""
        pass

    async def async_get_version(self) -> bool:
        """Check if remote service is available."""
        url = f"{self.remote_url}/api/version"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception:
            return False

    async def async_load_model(self):
        """Async load model (pull)."""
        url = f"{self.remote_url}/api/pull"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"name": self.model_name}) as response:
                    if response.status == 200:
                        _LOGGER.info("Remote model %s loaded/ready", self.model_name)
                        self._model_loaded = True
                    else:
                        _LOGGER.error("Failed to load remote model: %s", await response.text())
        except Exception as e:
            _LOGGER.error("Failed to connect to remote service during pull: %s", e)
            self._model_loaded = False

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding synchronously (blocking).

        Called by EmbeddingEngine._generate_embedding_sync which runs in executor.
        Uses requests for sync HTTP since we're already in an executor thread.
        Dimension validation is handled by MemoryStore, not here — different models
        produce different dimensions (e.g. bge-m3=1024, all-minilm=384).
        """
        url = f"{self.remote_url}/api/embed"
        try:
            response = requests.post(
                url,
                json={"model": self.model_name, "input": [text]},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["embeddings"][0]
            return embedding
        except Exception as e:
            _LOGGER.error("Remote embedding generation failed: %s", e)
            raise RuntimeError(f"Remote embedding failed: {e}")

    def update_vocabulary(self, text: str):
        """No-op for Remote Engine."""
        pass
