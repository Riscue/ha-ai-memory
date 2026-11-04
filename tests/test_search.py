"""Tests for MemorySearch engine."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from homeassistant.core import HomeAssistant

from custom_components.ai_memory.memory.store import MemoryStore
from custom_components.ai_memory.memory.migration import MigrationManager
from custom_components.ai_memory.memory.search import MemorySearch


@pytest.fixture
def store():
    """Create in-memory store with migration."""
    s = MemoryStore(":memory:")
    MigrationManager(s).migrate()
    yield s
    s.close()


@pytest.fixture
def mock_embedding_engine():
    """Mock embedding engine."""
    engine = MagicMock()
    engine._generate_embedding_sync = MagicMock(return_value=[1.0] + [0.0] * 383)
    engine.async_generate_embedding = AsyncMock(return_value=[1.0] + [0.0] * 383)
    return engine


@pytest.fixture
def mock_hass():
    """Mock Home Assistant."""
    hass = MagicMock(spec=HomeAssistant)
    async def mock_executor(target, *args):
        return target(*args)
    hass.async_add_executor_job = AsyncMock(side_effect=mock_executor)
    return hass


@pytest.fixture
def search(store, mock_embedding_engine):
    """Create MemorySearch instance."""
    return MemorySearch(store, mock_embedding_engine)


def _insert_memory(store, content, scope="common", agent_id=None, wing=None, room=None, embedding=None, summary=None):
    """Helper to insert a memory directly."""
    import uuid
    from datetime import datetime
    mem_id = str(uuid.uuid4())
    emb_json = json.dumps(embedding) if embedding else None
    store.execute_commit(
        """INSERT INTO memories
           (id, content, embedding, scope, agent_id, created_at, wing, room, layer, updated_at, access_count, summary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 2, ?, 0, ?)""",
        (mem_id, content, emb_json, scope, agent_id, datetime.now().isoformat(),
         wing or "general", room or "general", datetime.now().isoformat(), summary),
    )
    return mem_id


async def test_basic_search(search, store, mock_hass):
    """Test basic search returns matching memories."""
    emb = [1.0] + [0.0] * 383
    _insert_memory(store, "Kitchen light is on", "common", embedding=emb)
    _insert_memory(store, "Garage door is closed", "common", embedding=[0.0, 1.0] + [0.0] * 382)

    results = await search.async_search("kitchen light", "agent_1", hass=mock_hass)
    assert len(results) >= 1
    assert any("Kitchen" in r["content"] for r in results)


async def test_search_with_wing_filter(search, store, mock_hass):
    """Test search with wing filter."""
    emb = [1.0] + [0.0] * 383
    _insert_memory(store, "Light is on", "common", wing="household", room="devices", embedding=emb)
    _insert_memory(store, "I like coffee", "common", wing="personal", room="preferences", embedding=emb)

    results = await search.async_search("light", "agent_1", wing="household", hass=mock_hass)
    assert all(r["wing"] == "household" for r in results)


async def test_search_with_room_filter(search, store, mock_hass):
    """Test search with room filter."""
    emb = [1.0] + [0.0] * 383
    _insert_memory(store, "Light is on", "common", wing="household", room="devices", embedding=emb)
    _insert_memory(store, "Light broken", "common", wing="household", room="maintenance", embedding=emb)

    results = await search.async_search("light", "agent_1", room="devices", hass=mock_hass)
    assert all(r["room"] == "devices" for r in results)


async def test_search_private_scope_isolation(search, store, mock_hass):
    """Test private memories are only visible to owner."""
    emb = [1.0] + [0.0] * 383
    _insert_memory(store, "Secret 1", "private", "agent_1", embedding=emb)
    _insert_memory(store, "Secret 2", "private", "agent_2", embedding=emb)

    results = await search.async_search("secret", "agent_1", hass=mock_hass)
    assert all(r["agent_id"] == "agent_1" for r in results)


async def test_search_empty_query(search, mock_hass):
    """Test search with empty query returns empty."""
    results = await search.async_search("", "agent_1", hass=mock_hass)
    assert results == []


async def test_search_no_results(search, store, mock_hass, mock_embedding_engine):
    """Test search with no matching memories."""
    _insert_memory(store, "Kitchen light is on", "common", embedding=[0.5] + [0.0] * 383)

    # Use an embedding that produces low similarity: query=[0,-1,0,...], stored=[0.5,0,0,...]
    mock_embedding_engine._generate_embedding_sync.return_value = [0.0, -1.0] + [0.0] * 382
    mock_embedding_engine.async_generate_embedding.return_value = [0.0, -1.0] + [0.0] * 382

    results = await search.async_search("something completely different", "agent_1", min_score=0.99, hass=mock_hass)
    assert results == []


async def test_search_access_count_updated(search, store, mock_hass):
    """Test access_count is incremented for returned results."""
    emb = [1.0] + [0.0] * 383
    mem_id = _insert_memory(store, "Test memory", "common", embedding=emb)

    await search.async_search("test", "agent_1", hass=mock_hass)

    rows = store.execute_query("SELECT access_count FROM memories WHERE id = ?", (mem_id,))
    assert rows[0][0] == 1


async def test_text_fallback_when_semantic_fails(search, store, mock_hass, mock_embedding_engine):
    """Test text fallback when semantic search returns no results."""
    # No embedding — semantic search can't match
    _insert_memory(store, "Kullanıcı Fenerbahçe taraftarıdır", "common", embedding=None)

    # Make embedding engine return something that won't match
    mock_embedding_engine.async_generate_embedding.return_value = [1.0] + [0.0] * 383

    results = await search.async_search("fenerbahçe futbol", "agent_1", hass=mock_hass)
    assert len(results) == 1
    assert results[0]["match_type"] == "text"
    assert "Fenerbahçe" in results[0]["content"]


async def test_text_fallback_with_summary(search, store, mock_hass, mock_embedding_engine):
    """Test text fallback searches summary field too."""
    _insert_memory(
        store, "Some content", "common",
        summary="fenerbahçe, futbol, taraftar", embedding=None
    )

    mock_embedding_engine.async_generate_embedding.return_value = [1.0] + [0.0] * 383

    results = await search.async_search("futbol", "agent_1", hass=mock_hass)
    assert len(results) == 1
    assert results[0]["match_type"] == "text"


async def test_text_fallback_no_match(search, store, mock_hass, mock_embedding_engine):
    """Test text fallback returns empty when no text match either."""
    _insert_memory(store, "Completely unrelated content", "common", embedding=None)

    mock_embedding_engine.async_generate_embedding.return_value = [1.0] + [0.0] * 383

    results = await search.async_search("fenerbahçe futbol", "agent_1", hass=mock_hass)
    assert results == []


async def test_text_fallback_ignores_stopwords(search, store, mock_hass, mock_embedding_engine):
    """Common question words must not match unrelated memories."""
    _insert_memory(store, "The user does not want the blinds opened before 8 AM.", "common", embedding=None)
    _insert_memory(store, "The user likes pizza.", "common", embedding=None)

    results = await search.async_search("Do I like jazz music?", "agent_1", hass=mock_hass)
    assert results == []


async def test_text_fallback_only_stopwords(search, store, mock_hass, mock_embedding_engine):
    """A query made only of stopwords returns nothing."""
    _insert_memory(store, "The user likes pizza.", "common", embedding=None)

    results = await search.async_search("What do I like?", "agent_1", hass=mock_hass)
    assert results == []


async def test_text_fallback_requires_half_of_keywords(search, store, mock_hass, mock_embedding_engine):
    """A single keyword hit is not enough when the query has several keywords."""
    _insert_memory(store, "The user plays guitar.", "common", embedding=None)

    results = await search.async_search("guitar piano violin drums", "agent_1", hass=mock_hass)
    assert results == []

    results = await search.async_search("guitar piano", "agent_1", hass=mock_hass)
    assert len(results) == 1


async def test_text_fallback_strips_punctuation(search, store, mock_hass, mock_embedding_engine):
    """Punctuation attached to words does not prevent a match."""
    _insert_memory(store, "The user supports Real Madrid.", "common", embedding=None)

    results = await search.async_search("Madrid?", "agent_1", hass=mock_hass)
    assert len(results) == 1


async def test_text_fallback_unicode_case_insensitive(search, store, mock_hass, mock_embedding_engine):
    """Non-ASCII uppercase letters are folded (SQLite LOWER() would not)."""
    _insert_memory(store, "Kullanıcı ÇAY sever.", "common", embedding=None)

    results = await search.async_search("çay", "agent_1", hass=mock_hass)
    assert len(results) == 1


async def test_text_fallback_ranks_by_hits(search, store, mock_hass, mock_embedding_engine):
    """Memories matching more keywords come first."""
    _insert_memory(store, "The user drinks coffee.", "common", embedding=None)
    _insert_memory(store, "The user drinks coffee every morning.", "common", embedding=None)

    results = await search.async_search("coffee morning", "agent_1", hass=mock_hass)
    assert [r["content"] for r in results] == [
        "The user drinks coffee every morning.",
        "The user drinks coffee.",
    ]


async def test_text_fallback_respects_scope(search, store, mock_hass, mock_embedding_engine):
    """Private memories of other agents are not returned by the text fallback."""
    _insert_memory(store, "Secret guitar notes", "private", agent_id="agent_2", embedding=None)

    results = await search.async_search("guitar", "agent_1", hass=mock_hass)
    assert results == []


async def test_text_fallback_short_terms_and_numbers(search, store, mock_hass, mock_embedding_engine):
    """Two-letter terms and numbers are kept, and match whole words only."""
    _insert_memory(store, "The TV in the living room is a Sony.", "common", embedding=None)
    _insert_memory(store, "The user's bank account is at ING.", "common", embedding=None)
    _insert_memory(store, "The alarm is set for 8 AM.", "common", embedding=None)

    results = await search.async_search("Is the TV on?", "agent_1", hass=mock_hass)
    assert [r["content"] for r in results] == ["The TV in the living room is a Sony."]

    results = await search.async_search("Is the AC on?", "agent_1", hass=mock_hass)
    assert results == []

    results = await search.async_search("8", "agent_1", hass=mock_hass)
    assert [r["content"] for r in results] == ["The alarm is set for 8 AM."]


async def test_text_fallback_matches_word_prefix_not_substring(search, store, mock_hass, mock_embedding_engine):
    """Keywords match word prefixes (Turkish suffixes) but not inner substrings."""
    _insert_memory(store, "The user wears a red scarf.", "common", embedding=None)
    _insert_memory(store, "Kullanıcı sabahları çayı şekersiz içer.", "common", embedding=None)

    results = await search.async_search("car", "agent_1", hass=mock_hass)
    assert results == []

    results = await search.async_search("çay", "agent_1", hass=mock_hass)
    assert len(results) == 1


async def test_text_fallback_turkish_dotted_i(search, store, mock_hass, mock_embedding_engine):
    """Turkish I/ı and İ/i fold together regardless of case."""
    _insert_memory(store, "Salondaki ısıtıcı akşam açılır.", "common", embedding=None)
    _insert_memory(store, "Kullanıcı İzmir'de yaşıyor.", "common", embedding=None)

    results = await search.async_search("ISITICI", "agent_1", hass=mock_hass)
    assert len(results) == 1
    assert "ısıtıcı" in results[0]["content"]

    results = await search.async_search("izmir", "agent_1", hass=mock_hass)
    assert len(results) == 1
    assert "İzmir" in results[0]["content"]

    # Uppercase stopword is still recognized after folding
    results = await search.async_search("KULLANICI", "agent_1", hass=mock_hass)
    assert results == []


async def test_text_fallback_respects_wing_filter(search, store, mock_hass, mock_embedding_engine):
    """Text fallback honors wing/room filters."""
    _insert_memory(store, "Guitar lessons on Monday", "common", wing="personal", room="hobbies", embedding=None)
    _insert_memory(store, "Guitar amp is in the garage", "common", wing="household", room="devices", embedding=None)

    results = await search.async_search("guitar", "agent_1", wing="Household", hass=mock_hass)
    assert [r["content"] for r in results] == ["Guitar amp is in the garage"]


async def test_text_fallback_updates_access_count(search, store, mock_hass, mock_embedding_engine):
    """Results returned by the text fallback also bump access_count."""
    mem_id = _insert_memory(store, "The user plays guitar.", "common", embedding=None)

    results = await search.async_search("guitar", "agent_1", hass=mock_hass)
    assert results[0]["match_type"] == "text"

    rows = store.execute_query("SELECT access_count, accessed_at FROM memories WHERE id = ?", (mem_id,))
    assert rows[0][0] == 1
    assert rows[0][1] is not None


async def test_cosine_similarity():
    """Test cosine similarity calculation."""
    s = MemorySearch.__new__(MemorySearch)
    # Identical vectors
    assert s._cosine_similarity(np.array([1, 0]), np.array([1, 0])) == pytest.approx(1.0)
    # Orthogonal vectors
    assert s._cosine_similarity(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)
    # Opposite vectors
    assert s._cosine_similarity(np.array([1, 0]), np.array([-1, 0])) == pytest.approx(-1.0)
    # Zero vectors
    assert s._cosine_similarity(np.array([0, 0]), np.array([1, 0])) == 0.0
