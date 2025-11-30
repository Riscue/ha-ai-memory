from fastapi.testclient import TestClient
from embedding_service.embedding_service import app
import sys

client = TestClient(app)

def test_read_version():
    response = client.get("/api/version")
    assert response.status_code == 200
    assert response.json() == {"version": "0.1.0"}

def test_pull_model():
    # Mocking get_model to avoid actual download
    from unittest.mock import patch
    with patch("embedding_service.embedding_service.get_model") as mock_get_model:
        response = client.post("/api/pull", json={"name": "test-model"})
        assert response.status_code == 200
        assert response.json() == {"status": "success"}
        mock_get_model.assert_called_with("test-model")

def test_embed_input_string():
    from unittest.mock import patch, Mock
    with patch("embedding_service.embedding_service.get_model") as mock_get_model:
        mock_model = Mock()
        # Mock sentence transformer encode
        mock_model.encode.return_value = Mock(tolist=lambda: [[0.1, 0.2]])
        mock_get_model.return_value = mock_model
        
        # Force ENGINE_TYPE to sentence_transformer for test if needed, 
        # but mocking get_model might be enough if we mock the return value correctly based on engine type.
        # Let's assume default engine or mock the engine type check if possible.
        # Actually, the service code checks ENGINE_TYPE global. We might need to patch it or ensure mock behaves.
        # For simplicity, let's patch the global ENGINE_TYPE in the service module if possible,
        # or just ensure our mock model works for the active engine.
        # The code does:
        # if ENGINE_TYPE == SENTENCE_TRANSFORMER: embeddings = model.encode(inputs).tolist()
        # elif ENGINE_TYPE == FASTEMBED: vectors = list(model.embed(inputs)); embeddings = ...
        
        # Let's patch ENGINE_TYPE to SENTENCE_TRANSFORMER for deterministic testing
        with patch("embedding_service.embedding_service.ENGINE_TYPE", "sentence_transformer"):
             response = client.post("/api/embed", json={"model": "test-model", "input": "hello"})
             assert response.status_code == 200
             data = response.json()
             assert "embeddings" in data
             assert len(data["embeddings"]) == 1
             assert data["embeddings"][0] == [0.1, 0.2]
             assert data["prompt_eval_count"] == 0

def test_embed_input_list():
    from unittest.mock import patch, Mock
    with patch("embedding_service.embedding_service.get_model") as mock_get_model:
        mock_model = Mock()
        mock_model.encode.return_value = Mock(tolist=lambda: [[0.1, 0.2], [0.3, 0.4]])
        mock_get_model.return_value = mock_model
        
        with patch("embedding_service.embedding_service.ENGINE_TYPE", "sentence_transformer"):
             response = client.post("/api/embed", json={"model": "test-model", "input": ["hello", "world"]})
             assert response.status_code == 200
             data = response.json()
             assert len(data["embeddings"]) == 2
             assert data["embeddings"][0] == [0.1, 0.2]
             assert data["embeddings"][1] == [0.3, 0.4]

def test_tags():
    from unittest.mock import patch, MagicMock
    # Mock TextEmbedding.list_supported_models
    with patch("embedding_service.embedding_service.ENGINE_TYPE", "fastembed"):
        with patch.dict("sys.modules", {"fastembed": MagicMock()}):
            mock_fastembed = sys.modules["fastembed"]
            mock_fastembed.TextEmbedding.list_supported_models.return_value = [
                {"model": "test-model", "dim": 384}
            ]
            
            response = client.get("/api/tags")
            assert response.status_code == 200
            data = response.json()
            assert "models" in data
            assert len(data["models"]) == 1
            assert data["models"][0]["name"] == "test-model"
