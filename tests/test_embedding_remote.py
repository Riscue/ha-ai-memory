"""Tests for Remote embedding engine."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from custom_components.ai_memory.embedding_remote import RemoteEmbeddingEngine

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

    def test_initialization(self, mock_hass, config_data):
        """Test initialization."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)
        assert engine.remote_url == "http://test-server:8000"
        assert engine.model_name == "test-model"

    @pytest.mark.asyncio
    async def test_async_generate_embedding_success(self, mock_hass, config_data):
        """Test successful async embedding generation."""
        engine = RemoteEmbeddingEngine(mock_hass, config_data)
        
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            mock_response.raise_for_status = Mock()
            mock_post.return_value.__aenter__.return_value = mock_response
            
            embedding = await engine.async_generate_embedding("test")
            
            assert embedding == [0.1, 0.2, 0.3]
            mock_post.assert_called_with(
                "http://test-server:8000/api/embeddings",
                json={"model": "test-model", "prompt": "test"}
            )

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
        
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
            mock_post.return_value = mock_response
            
            embedding = engine.generate_embedding("test")
            
            assert embedding == [0.1, 0.2, 0.3]
            mock_post.assert_called_with(
                "http://test-server:8000/api/embeddings",
                json={"model": "test-model", "prompt": "test"},
                timeout=10
            )
