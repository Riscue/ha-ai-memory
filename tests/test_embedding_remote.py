"""Tests for Remote embedding engine."""
from unittest.mock import Mock, patch, AsyncMock

import pytest

from custom_components.ai_memory.embedding.remote import RemoteEmbeddingEngine


class TestRemoteEmbeddingEngine:
    """Test Remote engine."""

    @pytest.fixture
    def mock_hass(self):
        """Mock Home Assistant."""
        hass = Mock()
        return hass

    @pytest.fixture
    def config_data(self):
        """Mock config data."""
        return {
            "remote_url": "http://test-server:8000",
            "model_name": "test-model"
        }

    @pytest.fixture
    def openai_config_data(self):
        """Mock config data for the OpenAI-compatible flavor."""
        return {
            "remote_url": "http://test-server:8000",
            "model_name": "test-model",
            "embedding_engine": "openai_compatible",
            "api_key": "secret-key",
        }

    def test_initialization(self, mock_hass, config_data):
        """Test initialization."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)
        assert engine.remote_url == "http://test-server:8000"
        assert engine.model_name == "test-model"

    def test_sync_generate_embedding_success(self, mock_hass, config_data):
        """Test successful sync embedding generation."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)

        # Generate a 384-dimensional embedding to match EMBEDDINGS_VECTOR_DIM
        fake_embedding = [0.1] * 384

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embeddings": [fake_embedding]}
            mock_post.return_value = mock_response

            embedding = engine.generate_embedding("test")

            assert embedding == fake_embedding
            mock_post.assert_called_with(
                "http://test-server:8000/api/embed",
                json={"model": "test-model", "input": ["test"]},
                timeout=30,
            )

    @pytest.mark.asyncio
    async def test_async_get_version_success(self, mock_hass):
        engine = RemoteEmbeddingEngine(mock_hass, {"remote_url": "http://localhost:11434"})
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response

            assert await engine.async_get_version() is True

    @pytest.mark.asyncio
    async def test_async_get_version_failure(self, mock_hass):
        engine = RemoteEmbeddingEngine(mock_hass, {"remote_url": "http://localhost:11434"})
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            assert await engine.async_get_version() is False

    @pytest.mark.asyncio
    async def test_async_load_model_success(self, mock_hass, config_data):
        """Test successful model loading."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response

            await engine.async_load_model()

            assert engine._model_loaded is True
            mock_post.assert_called_with(
                "http://test-server:8000/api/pull",
                json={"name": "test-model"}
            )

    def test_generate_embedding_sync_success(self, mock_hass, config_data):
        """Test successful sync embedding generation."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)

        fake_embedding = [0.1] * 384

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embeddings": [fake_embedding]}
            mock_post.return_value = mock_response

            embedding = engine.generate_embedding("test")

            assert embedding == fake_embedding
            mock_post.assert_called_with(
                "http://test-server:8000/api/embed",
                json={"model": "test-model", "input": ["test"]},
                timeout=30,
            )

    @pytest.mark.asyncio
    async def test_async_load_model_failure(self, mock_hass, config_data):
        """Test model loading failure."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text.return_value = "Internal Server Error"
            mock_post.return_value.__aenter__.return_value = mock_response

            await engine.async_load_model()

            assert engine._model_loaded is False

    @pytest.mark.asyncio
    async def test_async_load_model_exception(self, mock_hass, config_data):
        """Test model loading exception."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)

        with patch("aiohttp.ClientSession.post", side_effect=Exception("Connection failed")):
            await engine.async_load_model()
            assert engine._model_loaded is False

    def test_generate_embedding_sync_failure(self, mock_hass, config_data):
        """Test sync embedding generation failure."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)

        with patch("requests.post", side_effect=Exception("Request failed")):
            with pytest.raises(RuntimeError, match="Remote embedding failed"):
                engine.generate_embedding("test")

    @pytest.mark.asyncio
    async def test_async_generate_embedding_failure(self, mock_hass, config_data):
        """Test async embedding generation failure."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)

        with patch("aiohttp.ClientSession.post", side_effect=Exception("Request failed")):
            with pytest.raises(RuntimeError, match="Remote embedding failed"):
                engine.generate_embedding("test")

    def test_load_model_sync(self, mock_hass, config_data):
        """Test sync load model (no-op)."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)
        engine._load_model()
        # Should just pass without error

    def test_update_vocabulary(self, mock_hass, config_data):
        """Test update vocabulary (no-op)."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)
        engine.update_vocabulary("test")
        # Should just pass without error

    def test_openai_generate_embedding_maps_data_list(
            self, mock_hass, openai_config_data):
        """OpenAI flavor posts to /v1/embeddings and maps data[i].embedding
        by input position."""
        engine = RemoteEmbeddingEngine(mock_hass, openai_config_data)

        fake_embedding = [0.1] * 1024
        other_embedding = [0.9] * 1024

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            # Out of order on purpose: index drives the mapping
            mock_response.json.return_value = {
                "data": [
                    {"index": 1, "embedding": other_embedding},
                    {"index": 0, "embedding": fake_embedding},
                ]
            }
            mock_post.return_value = mock_response

            embedding = engine.generate_embedding("test")

            assert embedding == fake_embedding
            mock_post.assert_called_with(
                "http://test-server:8000/v1/embeddings",
                json={"model": "test-model", "input": ["test"]},
                timeout=30,
                headers={"Authorization": "Bearer secret-key"},
            )

    def test_ollama_headers_with_api_key(self, mock_hass, config_data):
        """Ollama flavor sends the bearer token when an API key is set."""
        config_data["api_key"] = "ollama-secret"
        engine = RemoteEmbeddingEngine(mock_hass, config_data)

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embeddings": [[0.1] * 384]}
            mock_post.return_value = mock_response

            engine.generate_embedding("test")

            mock_post.assert_called_with(
                "http://test-server:8000/api/embed",
                json={"model": "test-model", "input": ["test"]},
                timeout=30,
                headers={"Authorization": "Bearer ollama-secret"},
            )

    @pytest.mark.asyncio
    async def test_openai_get_version_uses_v1_models(
            self, mock_hass, openai_config_data):
        """OpenAI flavor health check hits /v1/models with auth."""
        engine = RemoteEmbeddingEngine(mock_hass, openai_config_data)
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response

            assert await engine.async_get_version() is True
            assert "v1/models" in mock_get.call_args[0][0]

    @pytest.mark.asyncio
    async def test_openai_load_model_noop(self, mock_hass, openai_config_data):
        """Pull is skipped for OpenAI-compatible servers."""
        engine = RemoteEmbeddingEngine(mock_hass, openai_config_data)
        with patch("aiohttp.ClientSession.post") as mock_post:
            await engine.async_load_model()

            mock_post.assert_not_called()
            assert engine._model_loaded is True

    def test_legacy_remote_normalizes_to_ollama(self, mock_hass, config_data):
        """Legacy embedding_engine="remote" behaves as the Ollama flavor."""
        config_data["embedding_engine"] = "remote"
        engine = RemoteEmbeddingEngine(mock_hass, config_data)

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embeddings": [[0.1] * 384]}
            mock_post.return_value = mock_response

            engine.generate_embedding("test")

            assert "api/embed" in mock_post.call_args[0][0]
