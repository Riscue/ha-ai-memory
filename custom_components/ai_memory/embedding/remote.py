"""Remote embedding engine (Ollama or OpenAI-compatible API)."""
import logging
from typing import List, Dict, Any, Optional

import aiohttp
import requests
from homeassistant.core import HomeAssistant

from ..constants import (
    API_FLAVOR_OPENAI,
    DEFAULT_MODEL,
    DEFAULT_REMOTE_URL,
    resolve_flavor,
)

_LOGGER = logging.getLogger(__name__)


class RemoteEmbeddingEngine:
    """Remote embedding engine using an Ollama or OpenAI-compatible API."""

    def __init__(self, hass: HomeAssistant, config_data: Dict[str, Any]):
        """Initialize the engine."""
        self.hass = hass
        self.remote_url = config_data.get("remote_url", DEFAULT_REMOTE_URL)
        self.model_name = config_data.get("model_name", DEFAULT_MODEL)
        self.api_key = config_data.get("api_key")
        self.flavor = resolve_flavor(config_data.get("embedding_engine"))
        self._model_loaded = False

    def _headers(self) -> Optional[Dict[str, str]]:
        """Auth headers, or None when no API key is configured."""
        if not self.api_key:
            return None
        return {"Authorization": f"Bearer {self.api_key}"}

    def _is_openai(self) -> bool:
        """Whether the backend speaks the OpenAI-compatible flavor."""
        return self.flavor == API_FLAVOR_OPENAI

    def _load_model(self):
        """Trigger model load on remote server."""
        pass

    async def async_get_version(self) -> bool:
        """Check if remote service is available."""
        if self._is_openai():
            url = f"{self.remote_url}/v1/models"
        else:
            url = f"{self.remote_url}/api/version"
        kwargs: Dict[str, Any] = {"timeout": aiohttp.ClientTimeout(total=5)}
        headers = self._headers()
        if headers:
            kwargs["headers"] = headers
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, **kwargs) as response:
                    return response.status == 200
        except Exception:
            return False

    async def async_load_model(self):
        """Async load model (pull). No-op for OpenAI-compatible servers,
        which load the model at startup."""
        if self._is_openai():
            _LOGGER.debug(
                "Skipping model pull for OpenAI-compatible endpoint (%s)",
                self.model_name,
            )
            self._model_loaded = True
            return

        url = f"{self.remote_url}/api/pull"
        kwargs = {"json": {"name": self.model_name}}
        headers = self._headers()
        if headers:
            kwargs["headers"] = headers
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, **kwargs) as response:
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
        if self._is_openai():
            url = f"{self.remote_url}/v1/embeddings"
        else:
            url = f"{self.remote_url}/api/embed"

        kwargs: Dict[str, Any] = {
            "json": {"model": self.model_name, "input": [text]},
            "timeout": 30,
        }
        headers = self._headers()
        if headers:
            kwargs["headers"] = headers

        try:
            response = requests.post(url, **kwargs)
            response.raise_for_status()
            data = response.json()
            if self._is_openai():
                # Map per input position so a future batched path stays correct.
                items = sorted(data["data"], key=lambda d: d.get("index", 0))
                embedding = items[0]["embedding"]
            else:
                embedding = data["embeddings"][0]
            return embedding
        except Exception as e:
            _LOGGER.error("Remote embedding generation failed: %s", e)
            raise RuntimeError(f"Remote embedding failed: {e}")

    def update_vocabulary(self, text: str):
        """No-op for Remote Engine."""
        pass
