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

    def test_import_error(self):
        """Test handling of missing dependency."""
        hass = Mock()
        engine = SentenceTransformerEngine(hass)
        
        # Remove from sys.modules to simulate missing package
        with patch.dict(sys.modules):
            sys.modules.pop("sentence_transformers", None)
            
            # Also need to ensure builtins.__import__ fails for it
            # or just rely on the fact that it's not in sys.modules and not in path
            # But since we are in a venv where it might not be installed, it might just work.
            # If it IS installed, we need to force failure.
            
            with patch("builtins.__import__", side_effect=ImportError):
                # This is too aggressive, it breaks all imports.
                # We need a side_effect that only fails for sentence_transformers
                pass

        # Better way: Mock the import inside the method by patching sys.modules with None?
        # No, that might cause issues.
        # Let's just patch the class to raise ImportError when instantiated? No, the import happens before.
        
        # We can patch `importlib.import_module` if used, but it uses `import ...` statement.
        
        # Let's try patching the class in the module, but the module isn't imported yet.
        
        # Simplest: Use a side_effect on the mock module if we keep it mocked, 
        # or unmock it and assume it's missing (which it is in this env).
        
        # Since we know it's missing in this env, we can just run it?
        # But we want the test to pass even if it IS installed.
        
        # Let's use the fixture but make it raise ImportError
        with patch.dict(sys.modules, {"sentence_transformers": None}):
             # When sys.modules has None, import raises ImportError
             with pytest.raises(RuntimeError, match="sentence-transformers not installed"):
                 engine._load_model()

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
