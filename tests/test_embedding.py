"""Tests for EmbeddingEngine selector."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from homeassistant.core import HomeAssistant

from custom_components.ai_memory.embedding import EmbeddingEngine
from custom_components.ai_memory.constants import (
    ENGINE_TFIDF,
    ENGINE_REMOTE,
)

class TestEmbeddingEngine:
    """Test EmbeddingEngine selector logic."""

    @pytest.fixture
    def mock_hass(self):
        """Mock Home Assistant."""
        hass = MagicMock(spec=HomeAssistant)
        hass.async_add_executor_job = AsyncMock(side_effect=lambda f, *args: f(*args))
        return hass

    def test_init_defaults(self, mock_hass):
        """Test initialization with defaults."""
        engine = EmbeddingEngine(mock_hass)
        assert engine._engine_type == ENGINE_TFIDF
        assert engine._initialized is False

    def test_init_specific_engine(self, mock_hass):
        """Test initialization with specific engine."""
        engine = EmbeddingEngine(mock_hass, ENGINE_REMOTE)
        assert engine._engine_type == ENGINE_REMOTE
        assert engine._initialized is False

    @patch("custom_components.ai_memory.embedding.EmbeddingEngine._create_engine")
    async def test_initialize_engine_success(self, mock_create, mock_hass):
        """Test successful initialization."""
        mock_engine_instance = MagicMock()
        mock_create.return_value = mock_engine_instance
        
        engine = EmbeddingEngine(mock_hass, ENGINE_TFIDF)
        await engine.async_initialize()
        
        assert engine._initialized is True
        assert engine._engine == mock_engine_instance
        assert engine._engine_name == ENGINE_TFIDF
        mock_create.assert_called_once_with(ENGINE_TFIDF)

    @patch("custom_components.ai_memory.embedding.EmbeddingEngine._create_engine")
    async def test_initialize_engine_fallback(self, mock_create, mock_hass):
        """Test strict fallback to TF-IDF."""
        # First attempt (Remote) fails, Second (TF-IDF) succeeds
        mock_tfidf = MagicMock()
        
        def side_effect(engine_type):
            if engine_type == ENGINE_REMOTE:
                return None
            if engine_type == ENGINE_TFIDF:
                return mock_tfidf
            return None
            
        mock_create.side_effect = side_effect
        
        engine = EmbeddingEngine(mock_hass, ENGINE_REMOTE)
        await engine.async_initialize()
        
        assert engine._initialized is True
        assert engine._engine == mock_tfidf
        assert engine._engine_name == ENGINE_TFIDF
        mock_create.assert_any_call(ENGINE_REMOTE)
        mock_create.assert_called_with(ENGINE_TFIDF)

    @patch("custom_components.ai_memory.embedding.EmbeddingEngine._create_engine")
    async def test_initialize_engine_all_fail(self, mock_create, mock_hass):
        """Test when all engines fail."""
        mock_create.return_value = None
        
        engine = EmbeddingEngine(mock_hass, ENGINE_TFIDF)
        
        with pytest.raises(RuntimeError, match="No embedding engine available"):
            await engine.async_initialize()
        
        assert engine._initialized is False
        assert engine._engine is None

    def test_create_engine_unknown(self, mock_hass):
        """Test unknown engine creation."""
        engine = EmbeddingEngine(mock_hass)
        result = engine._create_engine("unknown_engine")
        assert result is None

    @patch("custom_components.ai_memory.embedding_tfidf.TFIDFEmbeddingEngine")
    async def test_generate_embedding_delegates(self, mock_tfidf, mock_hass):
        """Test generate_embedding delegates to selected engine."""
        engine = EmbeddingEngine(mock_hass, ENGINE_TFIDF)
        
        mock_instance = MagicMock()
        mock_instance.generate_embedding.return_value = [0.1, 0.2]
        mock_tfidf.return_value = mock_instance
        
        # Initialize
        await engine.async_initialize()
        
        result = await engine.async_generate_embedding("test")
        
        assert result == [0.1, 0.2]
        mock_instance.generate_embedding.assert_called_with("test")

    async def test_generate_embedding_not_initialized(self, mock_hass):
        """Test generating embedding before initialization."""
        engine = EmbeddingEngine(mock_hass)
        # Should implicitly initialize
        with patch.object(engine, '_initialize_engine') as mock_init:
            # Mock internal engine to avoid actual generation error
            engine._engine = MagicMock()
            engine._engine.generate_embedding.return_value = [0.1, 0.2]
            # We don't set initialized=True here because we want to trigger the check
            # But we need to ensure _generate_embedding_sync proceeds if we mock init
            
            # Actually, _generate_embedding_sync calls _initialize_engine then checks _engine.
            # So we need to ensure _engine is set.
            
            await engine.async_generate_embedding("test")
            mock_init.assert_called_once()

    async def test_generate_embedding_error(self, mock_hass):
        """Test error during embedding generation."""
        engine = EmbeddingEngine(mock_hass)
        engine._engine = MagicMock()
        engine._engine.generate_embedding.side_effect = Exception("Generation failed")
        engine._initialized = True
        
        with pytest.raises(Exception, match="Generation failed"):
            await engine.async_generate_embedding("test")

    async def test_async_update_vocabulary(self, mock_hass):
        """Test vocabulary update delegation."""
        engine = EmbeddingEngine(mock_hass)
        engine._engine = MagicMock()
        engine._initialized = True
        
        await engine.async_update_vocabulary("new word")
        
        engine._engine.update_vocabulary.assert_called_with("new word")
