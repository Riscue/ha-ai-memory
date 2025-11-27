"""Tests for FastEmbed embedding engine."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from custom_components.ai_memory.embedding_fastembed import FastEmbedEngine

class TestFastEmbedEngine:
    """Test FastEmbed engine."""

    @pytest.fixture(autouse=True)
    def mock_fastembed_module(self):
        """Mock fastembed module."""
        mock_module = MagicMock()
        mock_cls = MagicMock()
        mock_module.TextEmbedding = mock_cls
        
        with patch.dict(sys.modules, {"fastembed": mock_module}):
            yield mock_cls

    def test_initialization(self):
        """Test initialization."""
        hass = Mock()
        hass.config.path.return_value = "/tmp/cache"
        engine = FastEmbedEngine(hass)
        assert engine.model is None
        assert engine._model_loaded is False

    def test_load_model_success(self, mock_fastembed_module):
        """Test successful model loading."""
        hass = Mock()
        hass.config.path.return_value = "/tmp/cache"
        engine = FastEmbedEngine(hass)
        
        engine._load_model()
        
        assert engine._model_loaded is True
        assert engine.model is not None
        mock_fastembed_module.assert_called_once()

    def test_generate_embedding(self, mock_fastembed_module):
        """Test embedding generation."""
        hass = Mock()
        hass.config.path.return_value = "/tmp/cache"
        engine = FastEmbedEngine(hass)
        
        # Mock model
        mock_model = mock_fastembed_module.return_value
        import numpy as np
        # embed returns generator
        mock_model.embed.return_value = (np.array([0.1, 0.2, 0.3]) for _ in range(1))
        
        embedding = engine.generate_embedding("test")
        
        assert embedding == [0.1, 0.2, 0.3]
        mock_model.embed.assert_called_with(["test"])

    def test_import_error(self):
        """Test handling of missing dependency."""
        hass = Mock()
        engine = FastEmbedEngine(hass)
        
        with patch.dict(sys.modules, {"fastembed": None}):
             with pytest.raises(RuntimeError, match="fastembed not installed"):
                 engine._load_model()

    def test_load_model_generic_exception(self, mock_fastembed_module):
        """Test generic exception during load."""
        hass = Mock()
        hass.config.path.return_value = "/tmp/cache"
        engine = FastEmbedEngine(hass)
        
        # Configure the mock class to raise exception when instantiated
        mock_fastembed_module.side_effect = Exception("Generic Error")
        
        with pytest.raises(RuntimeError, match="Failed to load FastEmbed"):
            engine._load_model()

    def test_generate_embedding_auto_load(self, mock_fastembed_module):
        """Test generate_embedding automatically loads model."""
        hass = Mock()
        hass.config.path.return_value = "/tmp/cache"
        engine = FastEmbedEngine(hass)
        
        # Mock model
        mock_model = mock_fastembed_module.return_value
        import numpy as np
        mock_model.embed.return_value = (np.array([0.1]) for _ in range(1))
        
        # Should call _load_model internally
        engine.generate_embedding("test")
        
        assert engine._model_loaded is True
        mock_fastembed_module.assert_called_once()

    def test_update_vocabulary(self):
        """Test update_vocabulary (no-op)."""
        hass = Mock()
        engine = FastEmbedEngine(hass)
        engine.update_vocabulary("test")
        # Should not raise
