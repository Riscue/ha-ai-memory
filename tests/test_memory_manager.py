"""Tests for Memory Manager with SQLite."""
import json
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from custom_components.ai_memory.embedding.tfidf import TFIDFEmbeddingEngine
from custom_components.ai_memory.memory.manager import MemoryManager
from custom_components.ai_memory.memory.store import MemoryStore
from custom_components.ai_memory.memory.search import MemorySearch


@pytest.fixture
def mock_hass():
    """Mock Home Assistant."""
    hass = MagicMock(spec=HomeAssistant)
    # Mock executor to run function immediately
    async def mock_async_add_executor_job(target, *args):
        return target(*args)

    hass.async_add_executor_job = AsyncMock(side_effect=mock_async_add_executor_job)
    hass.bus = MagicMock()
    hass.config = MagicMock()
    hass.config.path.return_value = "/tmp/test_ha_config"
    return hass


@pytest.fixture
def mock_embedding_engine():
    """Mock EmbeddingEngine."""
    with patch("custom_components.ai_memory.embedding.engine.EmbeddingEngine") as mock_cls:
        instance = mock_cls.return_value
        instance.async_generate_embedding = AsyncMock(return_value=[1.0] + [0.0] * 383)
        instance._generate_embedding_sync = MagicMock(return_value=[1.0] + [0.0] * 383)
        instance.engine_name = "mock"
        yield instance


@pytest.fixture
def memory_manager(mock_hass, mock_embedding_engine):
    """Create MemoryManager instance with in-memory database."""
    with patch("custom_components.ai_memory.memory.manager.EmbeddingEngine") as mock_engine_cls:
        mock_engine_cls.return_value = mock_embedding_engine
        manager = MemoryManager(mock_hass, db_path=":memory:")
        manager.initialize_store()
        return manager


async def test_init_creates_tables(memory_manager):
    """Test database initialization creates tables."""
    # Verify memories table exists with new columns
    rows = memory_manager._store.execute_query("PRAGMA table_info(memories)")
    column_names = [row[1] for row in rows]
    assert "summary" in column_names
    assert "wing" in column_names
    assert "room" in column_names
    assert "layer" in column_names
    assert "access_count" in column_names


async def test_add_memory_basic(memory_manager):
    """Test adding a memory entry."""
    await memory_manager.async_add_memory("Test content", "private", "agent_1")

    rows = memory_manager._store.execute_query("SELECT content, scope, agent_id, wing, room, layer FROM memories")
    assert len(rows) == 1
    assert rows[0][0] == "Test content"
    assert rows[0][1] == "private"
    assert rows[0][2] == "agent_1"
    # wing/room auto-detected by RoomDetector
    assert rows[0][3] is not None
    assert rows[0][4] is not None
    assert rows[0][5] == 2  # default layer


async def test_add_memory_with_wing_room(memory_manager):
    """Test adding memory with explicit wing/room."""
    await memory_manager.async_add_memory(
        "Test", "common", "agent_1",
        wing="household", room="devices"
    )

    rows = memory_manager._store.execute_query("SELECT wing, room FROM memories")
    assert rows[0][0] == "household"
    assert rows[0][1] == "devices"


async def test_add_memory_with_summary(memory_manager, mock_embedding_engine):
    """Test adding memory with summary uses summary for embedding."""
    await memory_manager.async_add_memory(
        "Mutfaktaki ampul patladi",
        "common",
        summary="mutfak ampul patlak ariza"
    )

    # Verify embedding was generated (from summary)
    mock_embedding_engine.async_generate_embedding.assert_called()
    rows = memory_manager._store.execute_query("SELECT summary, content FROM memories")
    assert rows[0][0] == "mutfak ampul patlak ariza"
    assert rows[0][1] == "Mutfaktaki ampul patladi"


async def test_add_memory_invalid_input(memory_manager):
    """Test adding memory with invalid input."""
    # Empty content
    await memory_manager.async_add_memory("", "private", "agent_1")
    rows = memory_manager._store.execute_query("SELECT COUNT(*) FROM memories")
    assert rows[0][0] == 0

    # Invalid scope
    with pytest.raises(ValueError, match="Invalid scope"):
        await memory_manager.async_add_memory("Test", "invalid_scope")

    # Private scope without agent_id
    with pytest.raises(ValueError, match="Agent ID required"):
        await memory_manager.async_add_memory("Test", "private", None)


async def test_async_get_memory_counts(memory_manager):
    """Test getting memory counts."""
    await memory_manager.async_add_memory("Common 1", "common")
    await memory_manager.async_add_memory("Private 1", "private", "agent_1")
    await memory_manager.async_add_memory("Private 2", "private", "agent_1")

    counts = await memory_manager.async_get_memory_counts()
    assert counts["common"] == 1
    assert counts["private"] == 2
    assert counts["total"] == 3


async def test_async_search_memory(memory_manager):
    """Test search returns matching memories."""
    await memory_manager.async_add_memory("Kitchen light is on", "common")
    await memory_manager.async_add_memory("Garage door is closed", "common")

    results = await memory_manager.async_search_memory("kitchen light", "agent_1")
    assert len(results) >= 1
    assert any("Kitchen" in r["content"] for r in results)


async def test_async_search_memory_empty_query(memory_manager):
    """Test search with empty query returns empty."""
    results = await memory_manager.async_search_memory("", "agent_1")
    assert results == []


async def test_async_search_memory_embedding_failure(memory_manager, mock_embedding_engine):
    """Test search when query embedding generation fails."""
    mock_embedding_engine.async_generate_embedding.side_effect = Exception("Embedding Error")
    await memory_manager.async_add_memory("Test", "common")

    results = await memory_manager.async_search_memory("query", "agent_1")
    assert results == []


async def test_add_memory_embedding_failure(memory_manager, mock_embedding_engine):
    """Test adding memory when embedding generation fails."""
    mock_embedding_engine.async_generate_embedding.side_effect = Exception("Embedding Error")

    await memory_manager.async_add_memory("Test", "common")

    # Should still insert with NULL embedding
    rows = memory_manager._store.execute_query("SELECT embedding FROM memories")
    assert len(rows) == 1
    assert rows[0][0] is None


async def test_cosine_similarity_edge_cases():
    """Test cosine similarity edge cases."""
    search = MemorySearch.__new__(MemorySearch)
    # Zero vectors
    assert search._cosine_similarity(np.array([0, 0]), np.array([0, 0])) == 0.0
    # Mismatched length
    assert search._cosine_similarity(np.array([1]), np.array([1, 2])) == 0.0
    # Empty
    assert search._cosine_similarity(np.array([]), np.array([])) == 0.0


async def test_async_delete_memory_own_private(memory_manager):
    """Test agent can delete its own private memory."""
    await memory_manager.async_add_memory("Secret", "private", "agent_1", room="living_room")

    result = await memory_manager.async_delete_memory(agent_id="agent_1", room="living_room")
    assert result == 1

    rows = memory_manager._store.execute_query("SELECT COUNT(*) FROM memories")
    assert rows[0][0] == 0


async def test_async_delete_memory_cannot_delete_other_private(memory_manager):
    """Test agent cannot delete another agent's private memory."""
    await memory_manager.async_add_memory("Secret", "private", "agent_1", room="living_room")

    result = await memory_manager.async_delete_memory(agent_id="agent_2", room="living_room")
    assert result == 0

    # Memory should still exist
    rows = memory_manager._store.execute_query("SELECT COUNT(*) FROM memories")
    assert rows[0][0] == 1


async def test_async_delete_memory_any_agent_can_delete_common(memory_manager):
    """Test any agent can delete common memory."""
    await memory_manager.async_add_memory("Shared fact", "common", room="kitchen")

    result = await memory_manager.async_delete_memory(agent_id="any_agent", room="kitchen")
    assert result == 1

    rows = memory_manager._store.execute_query("SELECT COUNT(*) FROM memories")
    assert rows[0][0] == 0


async def test_close(memory_manager):
    """Test closing manager releases resources."""
    memory_manager.close()
    assert memory_manager._store._conn is None


async def test_async_initialize_failure(mock_hass, mock_embedding_engine):
    """Test initialization failure when remote is down."""
    mock_embedding_engine.async_initialize = AsyncMock()
    mock_embedding_engine._engine = AsyncMock()
    mock_embedding_engine._engine.async_get_version.return_value = False

    with patch("custom_components.ai_memory.memory.manager.EmbeddingEngine") as mock_engine_cls:
        mock_engine_cls.return_value = mock_embedding_engine
        manager = MemoryManager(mock_hass, db_path=":memory:")
        manager._embedding_engine = mock_embedding_engine

        with pytest.raises(ConfigEntryNotReady, match="Remote embedding service is not reachable"):
            await manager.async_initialize()


async def test_init_does_no_io(mock_hass, mock_embedding_engine):
    """Constructing the manager must not touch the database (event loop safety)."""
    with patch("custom_components.ai_memory.memory.manager.EmbeddingEngine") as mock_engine_cls:
        mock_engine_cls.return_value = mock_embedding_engine
        manager = MemoryManager(mock_hass, db_path=":memory:")
    assert manager._store._conn is None


async def test_default_db_path_uses_config_dir(mock_hass, mock_embedding_engine):
    """Without an explicit path the DB lives in the HA config directory."""
    mock_hass.config.path.side_effect = lambda *p: "/config/" + "/".join(p)
    with patch("custom_components.ai_memory.memory.manager.EmbeddingEngine") as mock_engine_cls:
        mock_engine_cls.return_value = mock_embedding_engine
        manager = MemoryManager(mock_hass)
    assert manager._db_path == "/config/ai_memory.db"


async def test_async_get_palace_stats(memory_manager):
    """Palace stats are available through the manager."""
    stats = await memory_manager.async_get_palace_stats()
    assert stats["wings"] > 0
    assert stats["rooms"] > 0


async def test_tfidf_reembeds_legacy_vectors_once(mock_hass):
    """Vectors from the old randomized hash are rebuilt on TF-IDF startup."""
    manager = MemoryManager(mock_hass, engine_type="tfidf", db_path=":memory:")
    with patch.object(TFIDFEmbeddingEngine, "_load_vocabulary"):
        await manager.async_initialize()

    stale = json.dumps([0.0] * 10)
    manager._store.execute_commit(
        "INSERT INTO memories (id, content, embedding, scope) VALUES (?, ?, ?, ?)",
        ("m1", "living room lamp", stale, "common"),
    )
    manager._store.execute_commit("DELETE FROM _meta WHERE key = 'tfidf_hash'")

    assert await mock_hass.async_add_executor_job(manager._reembed_tfidf_if_needed) == 1
    embedding = json.loads(
        manager._store.execute_query("SELECT embedding FROM memories")[0][0]
    )
    assert embedding == manager._embedding_engine._engine.generate_embedding("living room lamp")
    assert manager._store.get_embedding_dim() == 384

    # Marker is set, so a second run is a no-op
    assert manager._reembed_tfidf_if_needed() == 0


async def test_remote_engine_clears_tfidf_marker(mock_hass, mock_embedding_engine):
    """Running a non-TF-IDF engine forgets the marker so TF-IDF re-embeds later."""
    mock_embedding_engine.async_initialize = AsyncMock()
    mock_embedding_engine._engine = MagicMock(spec=[])
    with patch("custom_components.ai_memory.memory.manager.EmbeddingEngine") as mock_engine_cls:
        mock_engine_cls.return_value = mock_embedding_engine
        manager = MemoryManager(mock_hass, engine_type="remote", db_path=":memory:")
    manager.initialize_store()
    manager._store.execute_commit(
        "INSERT INTO _meta (key, value) VALUES ('tfidf_hash', 'md5')"
    )

    await manager.async_initialize()

    assert manager._store.execute_query(
        "SELECT value FROM _meta WHERE key = 'tfidf_hash'"
    ) == []
