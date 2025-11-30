import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Set CACHE_DIR to a temporary directory for tests
os.environ["CACHE_DIR"] = "/tmp/embedding_service_cache"

from embedding_service.embedding_service import app, get_model, get_fastembed_model, get_sentence_transformer_model, loaded_models

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "engine" in data
    assert "default_model" in data

@patch("embedding_service.embedding_service.get_model")
@patch("os.makedirs")
@pytest.mark.asyncio
async def test_lifespan(mock_makedirs, mock_get_model):
    from embedding_service.embedding_service import lifespan
    # lifespan is an async context manager
    async with lifespan(MagicMock()):
        mock_makedirs.assert_called()
        mock_get_model.assert_called()

@patch("embedding_service.embedding_service.get_model")
@patch("os.makedirs")
@pytest.mark.asyncio
async def test_lifespan_failure(mock_makedirs, mock_get_model):
    from embedding_service.embedding_service import lifespan
    mock_get_model.side_effect = Exception("Download failed")
    # Should not raise exception, just log warning
    async with lifespan(MagicMock()):
        mock_get_model.assert_called()

def test_get_model_sentence_transformer():
    with patch("embedding_service.embedding_service.ENGINE_TYPE", "sentence_transformer"):
        with patch("embedding_service.embedding_service.get_sentence_transformer_model") as mock_get_st:
            get_model("test_model")
            mock_get_st.assert_called_with("test_model")

def test_get_model_fastembed():
    with patch("embedding_service.embedding_service.ENGINE_TYPE", "fastembed"):
        with patch("embedding_service.embedding_service.get_fastembed_model") as mock_get_fe:
            get_model("test_model")
            mock_get_fe.assert_called_with("test_model")

def test_get_model_invalid_engine():
    with patch("embedding_service.embedding_service.ENGINE_TYPE", "invalid"):
        with pytest.raises(ValueError, match="Engine type invalid not supported"):
            get_model("test_model")

@patch.dict("sys.modules", {"fastembed": MagicMock()})
def test_get_fastembed_model_success_mocked():
    loaded_models.clear()
    # Setup the mock module
    mock_fastembed = sys.modules["fastembed"]
    mock_fastembed.TextEmbedding = MagicMock()
    
    model = get_fastembed_model("test_model")
    assert model is not None
    assert "test_model" in loaded_models

@patch.dict("sys.modules", {"sentence_transformers": MagicMock()})
def test_get_sentence_transformer_model_success_mocked():
    loaded_models.clear()
    # Setup the mock module
    mock_st = sys.modules["sentence_transformers"]
    mock_st.SentenceTransformer = MagicMock()
    
    model = get_sentence_transformer_model("test_model")
    assert model is not None
    assert "test_model" in loaded_models

def test_api_embeddings_sentence_transformer():
    with patch("embedding_service.embedding_service.ENGINE_TYPE", "sentence_transformer"):
        with patch("embedding_service.embedding_service.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_vector = MagicMock()
            mock_vector.tolist.return_value = [[0.1, 0.2]]
            mock_model.encode.return_value = mock_vector
            mock_get_model.return_value = mock_model
            
            response = client.post("/api/embed", json={"model": "test", "input": "hello"})
            assert response.status_code == 200
            assert response.json()["embeddings"] == [[0.1, 0.2]]

def test_api_embeddings_fastembed():
    with patch("embedding_service.embedding_service.ENGINE_TYPE", "fastembed"):
        with patch("embedding_service.embedding_service.get_model") as mock_get_model:
            mock_model = MagicMock()
            # FastEmbed returns a generator of vectors, each vector has tolist()
            mock_vector = MagicMock()
            mock_vector.tolist.return_value = [0.1, 0.2]
            mock_model.embed.return_value = [mock_vector]
            mock_get_model.return_value = mock_model
            
            response = client.post("/api/embed", json={"model": "test", "input": "hello"})
            assert response.status_code == 200
            assert response.json()["embeddings"] == [[0.1, 0.2]]
