"""TF-IDF based embedding engine for AI Memory (stdlib only, no dependencies)."""
import logging
import math
import re
from collections import Counter, defaultdict
from typing import List, Dict
import json
import os

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class TFIDFEmbeddingEngine:
    """Lightweight embedding engine using TF-IDF (no external dependencies).
    
    This engine provides semantic search capabilities without requiring
    sentence-transformers or any ML libraries. Perfect for resource-constrained
    devices like Raspberry Pi 4.
    
    Generates 384-dimensional vectors (same dimension as all-MiniLM-L6-v2)
    using TF-IDF weighting scheme with vocabulary hashing.
    """

    def __init__(self, hass: HomeAssistant, vector_dim: int = 384):
        """Initialize the TF-IDF embedding engine.
        
        Args:
            hass: Home Assistant instance
            vector_dim: Dimension of output vectors (default: 384)
        """
        self.hass = hass
        self.vector_dim = vector_dim
        self._document_count = 0
        self._term_document_freq: Dict[str, int] = defaultdict(int)
        self._vocabulary_file = os.path.join(
            hass.config.path(), ".storage", "ai_memory_tfidf_vocab.json"
        )
        self._load_vocabulary()
        _LOGGER.info("TF-IDF embedding engine initialized (dimension: %d)", vector_dim)

    def _load_vocabulary(self):
        """Load vocabulary and IDF statistics from storage."""
        try:
            if os.path.exists(self._vocabulary_file):
                with open(self._vocabulary_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._document_count = data.get('document_count', 0)
                    self._term_document_freq = defaultdict(int, data.get('term_df', {}))
                _LOGGER.debug(
                    "Loaded TF-IDF vocabulary: %d docs, %d terms",
                    self._document_count,
                    len(self._term_document_freq)
                )
        except Exception as e:
            _LOGGER.warning("Failed to load TF-IDF vocabulary: %s", e)

    def _save_vocabulary(self):
        """Save vocabulary and IDF statistics to storage."""
        try:
            os.makedirs(os.path.dirname(self._vocabulary_file), exist_ok=True)
            with open(self._vocabulary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'document_count': self._document_count,
                    'term_df': dict(self._term_document_freq)
                }, f)
        except Exception as e:
            _LOGGER.error("Failed to save TF-IDF vocabulary: %s", e)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into terms.
        
        Args:
            text: Input text
            
        Returns:
            List of lowercase tokens
        """
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        # Keep alphanumeric and basic punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate term frequency.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary mapping terms to their TF scores
        """
        if not tokens:
            return {}
        
        term_counts = Counter(tokens)
        max_count = max(term_counts.values())
        
        # Normalized TF: term_count / max_count_in_doc
        tf = {term: count / max_count for term, count in term_counts.items()}
        return tf

    def _calculate_idf(self, term: str) -> float:
        """Calculate inverse document frequency for a term.
        
        Args:
            term: The term to calculate IDF for
            
        Returns:
            IDF score
        """
        if self._document_count == 0:
            return 1.0
        
        # IDF = log(N / df(t))
        # Add smoothing: log((N + 1) / (df(t) + 1))
        df = self._term_document_freq.get(term, 0)
        idf = math.log((self._document_count + 1) / (df + 1))
        return idf

    def _hash_term_to_index(self, term: str) -> int:
        """Hash a term to a vector index.
        
        Args:
            term: Term to hash
            
        Returns:
            Index in range [0, vector_dim)
        """
        # Simple hash function that maps terms to indices
        return hash(term) % self.vector_dim

    def _create_vector(self, tf_idf: Dict[str, float]) -> List[float]:
        """Create a fixed-dimension vector from TF-IDF scores.
        
        Args:
            tf_idf: Dictionary of term -> TF-IDF score
            
        Returns:
            Fixed-dimension vector
        """
        vector = [0.0] * self.vector_dim
        
        # Map each term to an index and accumulate scores
        for term, score in tf_idf.items():
            idx = self._hash_term_to_index(term)
            vector[idx] += score
        
        # Normalize vector to unit length (L2 normalization)
        magnitude = math.sqrt(sum(x * x for x in vector))
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector

    def update_vocabulary(self, text: str):
        """Update vocabulary with a new document (for IDF calculation).
        
        This should be called when adding new memories to improve
        future embedding quality.
        
        Args:
            text: Document text
        """
        tokens = self._tokenize(text)
        if not tokens:
            return
        
        # Update document count
        self._document_count += 1
        
        # Update document frequency for each unique term
        unique_terms = set(tokens)
        for term in unique_terms:
            self._term_document_freq[term] += 1
        
        # Save vocabulary periodically (every 10 documents)
        if self._document_count % 10 == 0:
            self._save_vocabulary()

    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Generate TF-IDF embedding synchronously.
        
        Args:
            text: Input text
            
        Returns:
            384-dimensional embedding vector
        """
        # Tokenize
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.vector_dim
        
        # Calculate TF
        tf = self._calculate_tf(tokens)
        
        # Calculate TF-IDF
        tf_idf = {}
        for term, tf_score in tf.items():
            idf_score = self._calculate_idf(term)
            tf_idf[term] = tf_score * idf_score
        
        # Create fixed-dimension vector
        vector = self._create_vector(tf_idf)
        
        return vector

    async def async_generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text asynchronously.
        
        Args:
            text: Input text
            
        Returns:
            384-dimensional embedding vector
        """
        if not text:
            return [0.0] * self.vector_dim
        
        # Run in executor to avoid blocking the event loop
        return await self.hass.async_add_executor_job(
            self._generate_embedding_sync,
            text
        )
