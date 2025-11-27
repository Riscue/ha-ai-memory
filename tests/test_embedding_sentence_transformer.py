"""Tests for SentenceTransformer embedding engine."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys

# Mock the module before importing the class under test if it had top-level imports
# But it doesn't, so we can just mock it for the tests
from custom_components.ai_memory.embedding_sentence_transformer import SentenceTransformerEngine

class TestSentenceTransformerEngine:
    """Test SentenceTransformer engine."""

    @pytest.fixture(autouse=True)
    def mock_sentence_transformers_module(self):
        """Mock sentence_transformers module."""
        mock_module = MagicMock()
        mock_cls = MagicMock()
        mock_module.SentenceTransformer = mock_cls
        
        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            yield mock_cls

    def test_initialization(self):
        """Test initialization."""
        hass = Mock()
        engine = SentenceTransformerEngine(hass)
        assert engine.model is None
        assert engine._model_loaded is False

    def test_load_model_success(self, mock_sentence_transformers_module):
        """Test successful model loading."""
        hass = Mock()
        engine = SentenceTransformerEngine(hass)
        
        engine._load_model()
        
        assert engine._model_loaded is True
        assert engine.model is not None
        mock_sentence_transformers_module.assert_called_once()

    def test_generate_embedding(self, mock_sentence_transformers_module):
        """Test embedding generation."""
        hass = Mock()
        engine = SentenceTransformerEngine(hass)
        
        # Mock model instance
        mock_model = mock_sentence_transformers_module.return_value
        # encode returns numpy array usually, so we need something with tolist()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embedding
        
        embedding = engine.generate_embedding("test")
        
        assert embedding == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_with("test")

    def test_import_error_installs_package(self):
        """Test that missing dependency triggers installation."""
        hass = Mock()
        engine = SentenceTransformerEngine(hass)
        
        # Mock install_package
        with patch("custom_components.ai_memory.embedding_sentence_transformer.install_package") as mock_install:
            # We verify that if sentence_transformers is present (mocked), install_package is NOT called.
            # This is the regression test part.
            
            # Mock successful import
            mock_module = MagicMock()
            mock_module.SentenceTransformer = MagicMock()
            
            with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
                engine._load_model()
                mock_install.assert_not_called()

    def test_load_model_generic_exception(self, mock_sentence_transformers_module):
        """Test generic exception during load."""
        hass = Mock()
        engine = SentenceTransformerEngine(hass)
        
        # Configure the mock class to raise exception when instantiated
        mock_sentence_transformers_module.side_effect = Exception("Generic Error")
        
        with pytest.raises(RuntimeError, match="Failed to load SentenceTransformer model"):
            engine._load_model()

    def test_generate_embedding_auto_load(self, mock_sentence_transformers_module):
        """Test generate_embedding automatically loads model."""
        hass = Mock()
        engine = SentenceTransformerEngine(hass)
        
        # Mock model instance
        mock_model = mock_sentence_transformers_module.return_value
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1]
        mock_model.encode.return_value = mock_embedding
        
        # Should call _load_model internally
        engine.generate_embedding("test")
        
        assert engine._model_loaded is True
        mock_sentence_transformers_module.assert_called_once()

    def test_update_vocabulary(self):
        """Test update_vocabulary (no-op)."""
        hass = Mock()
        engine = SentenceTransformerEngine(hass)
        engine.update_vocabulary("test")
        # Should not raise

    def test_load_model_idempotency(self, mock_sentence_transformers_module):
        """Test that _load_model returns early if already loaded."""
        hass = Mock()
        engine = SentenceTransformerEngine(hass)
        
        # First load
        engine._load_model()
        assert engine._model_loaded is True
        mock_sentence_transformers_module.assert_called_once()
        
        # Second load
        engine._load_model()
        # Should not call init again
        mock_sentence_transformers_module.assert_called_once()

    def test_generate_embedding_missing_model(self, mock_sentence_transformers_module):
        """Test RuntimeError if model is missing after load."""
        hass = Mock()
        engine = SentenceTransformerEngine(hass)
        
        # Mock load success but model remains None
        engine._model_loaded = True
        engine.model = None
        
        with pytest.raises(RuntimeError, match="SentenceTransformer model not available"):
            engine.generate_embedding("test")

    def test_install_package_failure(self):
        """Test that installation failure raises RuntimeError."""
        hass = Mock()
        engine = SentenceTransformerEngine(hass)
        
        with patch("custom_components.ai_memory.embedding_sentence_transformer.install_package") as mock_install:
            mock_install.side_effect = Exception("Install Failed")
            
            # Simulate import error to trigger install
            with patch.dict(sys.modules, {"sentence_transformers": None}):
                with pytest.raises(RuntimeError, match="Failed to install sentence-transformers"):
                    engine._load_model()
