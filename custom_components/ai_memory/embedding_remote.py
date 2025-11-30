"""Remote embedding engine (Ollama/FastEmbed Service)."""
import logging
from typing import List, Dict, Any

import aiohttp
import requests
from homeassistant.core import HomeAssistant

from .constants import DEFAULT_MODEL, DEFAULT_REMOTE_URL

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
        # This is called synchronously by EmbeddingEngine, but we need async.
        # Since we can't await here easily without blocking, we'll rely on
        # async_initialize or lazy loading in generate_embedding.
        # However, EmbeddingEngine calls this to verify the engine works.
        # We can try a sync request or just skip if we trust the config.
        # For now, we'll implement a sync wrapper or just set a flag.

        # Ideally, we should check connection.
        # But since _load_model is called in executor, we can run async code via run_coroutine_threadsafe
        # or just use requests (but we should avoid blocking calls if possible, though executor is fine).
        # Let's use a simple check.
        pass

    async def async_get_version(self) -> bool:
        """Check if remote service is available."""
        url = f"{self.remote_url}/api/version"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
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
            # Don't raise here, just log, so we don't crash startup
            self._model_loaded = False

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding synchronously (blocking).
        
        This is called by EmbeddingEngine._generate_embedding_sync which runs in executor.
        So we can block here, but it's better to use async if possible.
        However, the interface is sync.
        We can use asyncio.run or run_coroutine_threadsafe, but since we are already in an executor,
        we can just use a sync library like requests or run async loop.
        
        Actually, since we are in an executor, we can use `requests`.
        But HA prefers `aiohttp`.
        
        If we are in an executor, we can create a new event loop or use `asyncio.run`.
        """

        url = f"{self.remote_url}/api/embed"
        try:
            response = requests.post(
                url,
                json={"model": self.model_name, "input": [text]},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
        except Exception as e:
            _LOGGER.error("Remote embedding generation failed: %s", e)
            raise RuntimeError(f"Remote embedding failed: {e}")

    async def async_generate_embedding(self, text: str) -> List[float]:
        """Generate embedding asynchronously."""
        url = f"{self.remote_url}/api/embed"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                        url,
                        json={"model": self.model_name, "input": text}
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data["embeddings"][0]
        except Exception as e:
            _LOGGER.error("Remote embedding generation failed: %s", e)
            raise RuntimeError(f"Remote embedding failed: {e}")

    def update_vocabulary(self, text: str):
        """No-op for Remote Engine."""
        pass
