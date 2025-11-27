"""Tests for Embedding Engine."""
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ai_memory.embedding import EmbeddingEngine


@pytest.fixture
def mock_hass():
    """Mock Home Assistant."""
    hass = MagicMock(spec=HomeAssistant)
    hass.async_add_executor_job = MagicMock()

    async def mock_async_add_executor_job(target, *args):
        return target(*args)

    hass.async_add_executor_job.side_effect = mock_async_add_executor_job
    return hass


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer."""
    # Mock the module in sys.modules to prevent actual import
    mock_module = MagicMock()
    mock_cls = MagicMock()
    mock_module.SentenceTransformer = mock_cls

    with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
        # Mock encode to return a numpy-like object with tolist()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_cls.return_value.encode.return_value = mock_embedding
        yield mock_cls


async def test_embedding_engine_load_model(mock_hass, mock_sentence_transformer):
    """Test loading the model."""
    engine = EmbeddingEngine(mock_hass)
    assert engine._model_loaded is False

    # Trigger load via generate
    await engine.async_generate_embedding("test")

    assert engine._model_loaded is True
    mock_sentence_transformer.assert_called_once()


async def test_embedding_generation(mock_hass, mock_sentence_transformer):
    """Test embedding generation."""
    engine = EmbeddingEngine(mock_hass)

    embedding = await engine.async_generate_embedding("test text")

    assert embedding == [0.1, 0.2, 0.3]
    mock_sentence_transformer.return_value.encode.assert_called_with("test text")


async def test_embedding_generation_empty(mock_hass):
    """Test empty text."""
    engine = EmbeddingEngine(mock_hass)
    embedding = await engine.async_generate_embedding("")
    assert embedding == []


async def test_embedding_model_import_error(mock_hass):
    """Test handling of import error."""
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        # Force reload to trigger import
        with patch("builtins.__import__", side_effect=ImportError):
            engine = EmbeddingEngine(mock_hass)
            with pytest.raises(ImportError):
                await engine.async_generate_embedding("test")


async def test_model_load_generic_exception(mock_hass):
    """Test generic exception during model loading."""
    mock_module = MagicMock()
    mock_module.SentenceTransformer.side_effect = Exception("Load Error")

    with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
        engine = EmbeddingEngine(mock_hass)
        # Should raise exception when trying to generate
        with pytest.raises(Exception, match="Load Error"):
            await engine.async_generate_embedding("test")


async def test_generate_embedding_model_not_available(mock_hass):
    """Test generation when model is not available."""
    engine = EmbeddingEngine(mock_hass)
    # Simulate model load failure but flag set (should not happen in normal flow but good for coverage)
    engine._model_loaded = True
    engine._model = None

    with pytest.raises(RuntimeError, match="Embedding model not available"):
        await engine.async_generate_embedding("test")
