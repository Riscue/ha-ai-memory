import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Set CACHE_DIR to a temporary directory for tests
os.environ["CACHE_DIR"] = "/tmp/embedding_service_cache"

from embedding_service.embedding_service import app, SENTENCE_TRANSFORMER, FASTEMBED

client = TestClient(app)

@pytest.fixture
def mock_model():
    with patch("embedding_service.embedding_service.get_model") as mock:
        model = MagicMock()
        # Mock for SentenceTransformer
        model.encode.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock for FastEmbed - needs to return objects with tolist()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        model.embed.return_value = [mock_embedding]
        
        mock.return_value = model
        yield mock

def test_api_pull(mock_model):
    response = client.post("/api/pull", json={"name": "BAAI/bge-small-en-v1.5"})
    assert response.status_code == 200
    assert response.json() == {"status": "success"}

def test_api_embed_string(mock_model):
    # Setup mock to return specific embedding for string input
    # Note: The implementation handles list conversion, so result is always list of lists
    mock_model.return_value.encode.return_value = [[0.1, 0.2, 0.3]]
    
    mock_embedding = MagicMock()
    mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
    mock_model.return_value.embed.return_value = [mock_embedding]
    
    response = client.post("/api/embed", json={
        "model": "BAAI/bge-small-en-v1.5",
        "input": "Hello world"
    })
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 1
    assert len(data["embeddings"][0]) == 3
    assert data["model"] == "BAAI/bge-small-en-v1.5"

def test_api_embed_list(mock_model):
    # Setup mock to return multiple embeddings
    mock_model.return_value.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    
    mock_embedding1 = MagicMock()
    mock_embedding1.tolist.return_value = [0.1, 0.2, 0.3]
    mock_embedding2 = MagicMock()
    mock_embedding2.tolist.return_value = [0.4, 0.5, 0.6]
    mock_model.return_value.embed.return_value = [mock_embedding1, mock_embedding2]

    response = client.post("/api/embed", json={
        "model": "BAAI/bge-small-en-v1.5",
        "input": ["Hello", "World"]
    })
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2
    assert data["model"] == "BAAI/bge-small-en-v1.5"
