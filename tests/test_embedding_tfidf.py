"""Tests for TF-IDF embedding engine."""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

from custom_components.ai_memory.embedding_tfidf import TFIDFEmbeddingEngine


@pytest.fixture
def mock_hass():
    """Create a mock Home Assistant instance."""
    hass = Mock()
    hass.config = Mock()
    
    # Use temporary directory for storage
    temp_dir = tempfile.mkdtemp()
    hass.config.path = Mock(return_value=temp_dir)
    
    # Mock executor
    async def mock_executor(func, *args):
        return func(*args)
    
    hass.async_add_executor_job = mock_executor
    
    return hass


class TestTFIDFEmbeddingEngine:
    """Test TF-IDF embedding engine."""
    
    def test_initialization(self, mock_hass):
        """Test engine initialization."""
        engine = TFIDFEmbeddingEngine(mock_hass, vector_dim=384)
        
        assert engine.vector_dim == 384
        assert engine._document_count == 0
        assert len(engine._term_document_freq) == 0
    
    def test_tokenization(self, mock_hass):
        """Test text tokenization."""
        engine = TFIDFEmbeddingEngine(mock_hass)
        
        # Test basic tokenization
        tokens = engine._tokenize("Hello world, this is a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        
        # Test empty string
        tokens = engine._tokenize("")
        assert tokens == []
        
        # Test special characters
        tokens = engine._tokenize("user@example.com and #hashtag")
        assert "user" in tokens
        assert "example" in tokens
    
    def test_tf_calculation(self, mock_hass):
        """Test term frequency calculation."""
        engine = TFIDFEmbeddingEngine(mock_hass)
        
        tokens = ["hello", "world", "hello"]
        tf = engine._calculate_tf(tokens)
        
        # "hello" appears twice (max count), so TF = 2/2 = 1.0
        assert tf["hello"] == 1.0
        # "world" appears once, so TF = 1/2 = 0.5
        assert tf["world"] == 0.5
    
    def test_idf_calculation(self, mock_hass):
        """Test inverse document frequency calculation."""
        engine = TFIDFEmbeddingEngine(mock_hass)
        
        # Empty corpus
        idf = engine._calculate_idf("test")
        assert idf > 0
        
        # Add documents
        engine._document_count = 10
        engine._term_document_freq["common"] = 8
        engine._term_document_freq["rare"] = 1
        
        # Common term should have lower IDF
        idf_common = engine._calculate_idf("common")
        idf_rare = engine._calculate_idf("rare")
        
        assert idf_rare > idf_common
    
    def test_term_hashing(self, mock_hass):
        """Test term to index hashing."""
        engine = TFIDFEmbeddingEngine(mock_hass, vector_dim=384)
        
        # Same term should always hash to same index
        idx1 = engine._hash_term_to_index("test")
        idx2 = engine._hash_term_to_index("test")
        assert idx1 == idx2
        
        # Index should be within bounds
        assert 0 <= idx1 < 384
    
    def test_vector_creation(self, mock_hass):
        """Test vector creation and normalization."""
        engine = TFIDFEmbeddingEngine(mock_hass, vector_dim=384)
        
        tf_idf = {"hello": 0.5, "world": 0.3}
        vector = engine._create_vector(tf_idf)
        
        # Check dimension
        assert len(vector) == 384
        
        # Check normalization (L2 norm should be ~1)
        import math
        magnitude = math.sqrt(sum(x * x for x in vector))
        assert abs(magnitude - 1.0) < 0.001
    
    def test_update_vocabulary(self, mock_hass):
        """Test vocabulary update."""
        engine = TFIDFEmbeddingEngine(mock_hass)
        
        # Add first document
        engine.update_vocabulary("hello world")
        assert engine._document_count == 1
        assert engine._term_document_freq["hello"] == 1
        assert engine._term_document_freq["world"] == 1
        
        # Add second document with overlap
        engine.update_vocabulary("hello universe")
        assert engine._document_count == 2
        assert engine._term_document_freq["hello"] == 2
        assert engine._term_document_freq["world"] == 1
        assert engine._term_document_freq["universe"] == 1
    
    def test_embedding_generation_sync(self, mock_hass):
        """Test synchronous embedding generation."""
        engine = TFIDFEmbeddingEngine(mock_hass, vector_dim=384)
        
        # Generate embedding
        embedding = engine._generate_embedding_sync("hello world test")
        
        # Check output
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
        
        # Test empty text
        embedding = engine._generate_embedding_sync("")
        assert embedding == [0.0] * 384
    
    @pytest.mark.asyncio
    async def test_embedding_generation_async(self, mock_hass):
        """Test asynchronous embedding generation."""
        engine = TFIDFEmbeddingEngine(mock_hass, vector_dim=384)
        
        # Generate embedding
        embedding = await engine.async_generate_embedding("hello world test")
        
        # Check output
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_embedding_similarity(self, mock_hass):
        """Test that similar texts have similar embeddings."""
        engine = TFIDFEmbeddingEngine(mock_hass, vector_dim=384)
        
        # Use larger text samples for more reliable similarity testing
        emb1 = await engine.async_generate_embedding("I love programming in Python and building web applications")
        emb2 = await engine.async_generate_embedding("I enjoy coding in Python and creating web apps")
        emb3 = await engine.async_generate_embedding("The weather is nice today with sunshine")
        
        # Calculate cosine similarity
        def cosine_similarity(v1, v2):
            import math
            dot = sum(a * b for a, b in zip(v1, v2))
            mag1 = math.sqrt(sum(a * a for a in v1))
            mag2 = math.sqrt(sum(b * b for b in v2))
            if mag1 == 0 or mag2 == 0:
                return 0.0
            return dot / (mag1 * mag2)
        
        # Similar programming texts should be more similar than unrelated weather text
        sim_12 = cosine_similarity(emb1, emb2)
        sim_13 = cosine_similarity(emb1, emb3)
        
        # Check that embeddings are meaningful (non-zero similarity for similar texts)
        assert sim_12 > 0.1, "Similar texts should have positive similarity"
        assert sim_12 > sim_13, "More similar texts should have higher similarity score"
    
    def test_vocabulary_persistence(self, mock_hass):
        """Test vocabulary saving and loading."""
        engine1 = TFIDFEmbeddingEngine(mock_hass, vector_dim=384)
        
        # Add some documents
        engine1.update_vocabulary("hello world")
        engine1.update_vocabulary("test document")
        engine1._save_vocabulary()
        
        # Create new engine instance (should load vocabulary)
        engine2 = TFIDFEmbeddingEngine(mock_hass, vector_dim=384)
        
        # Check loaded vocabulary
        assert engine2._document_count == engine1._document_count
        assert engine2._term_document_freq == engine1._term_document_freq
