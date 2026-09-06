"""Tests for provider constants and normalization helpers."""
from custom_components.ai_memory.constants import (
    ENGINE_REMOTE,
    ENGINE_TFIDF,
    API_FLAVOR_OLLAMA,
    API_FLAVOR_OPENAI,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_TFIDF,
    resolve_engine_type,
    resolve_flavor,
)


class TestResolveHelpers:
    def test_resolve_engine_type(self):
        assert resolve_engine_type(PROVIDER_OLLAMA) == ENGINE_REMOTE
        assert resolve_engine_type(PROVIDER_OPENAI) == ENGINE_REMOTE
        assert resolve_engine_type(PROVIDER_TFIDF) == ENGINE_TFIDF
        # Anything that isn't TF-IDF was remote-flavored: the legacy value,
        # a missing key, or a corrupted value.
        assert resolve_engine_type("remote") == ENGINE_REMOTE
        assert resolve_engine_type(None) == ENGINE_REMOTE
        assert resolve_engine_type("bogus") == ENGINE_REMOTE

    def test_resolve_flavor(self):
        assert resolve_flavor(PROVIDER_OPENAI) == API_FLAVOR_OPENAI
        assert resolve_flavor(PROVIDER_OLLAMA) == API_FLAVOR_OLLAMA
        # Legacy and unknown values default to the Ollama flavor
        assert resolve_flavor("remote") == API_FLAVOR_OLLAMA
        assert resolve_flavor(None) == API_FLAVOR_OLLAMA
