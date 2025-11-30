"""Tests for Memory Manager with SQLite."""
import json
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ai_memory.memory_manager import MemoryManager


@pytest.fixture
def mock_hass():
    """Mock Home Assistant."""
    hass = MagicMock(spec=HomeAssistant)
    hass.async_add_executor_job = AsyncMock()

    # Mock executor to run function immediately
    async def mock_async_add_executor_job(target, *args):
        return target(*args)

    hass.async_add_executor_job.side_effect = mock_async_add_executor_job
    hass.bus = MagicMock()
    return hass


@pytest.fixture
def mock_embedding_engine():
    """Mock EmbeddingEngine."""
    with patch("custom_components.ai_memory.embedding.EmbeddingEngine") as mock_cls:
        instance = mock_cls.return_value
        instance.async_generate_embedding = AsyncMock(return_value=[1.0, 0.0, 0.0])
        yield instance


@pytest.fixture
def mock_db():
    """Mock SQLite connection."""
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        yield mock_cursor


@pytest.fixture
def memory_manager(mock_hass, mock_embedding_engine):
    """Create MemoryManager instance."""
    with patch("os.makedirs"), \
            patch("sqlite3.connect"), \
            patch("custom_components.ai_memory.memory_manager.EmbeddingEngine") as mock_engine_cls:
        mock_engine_cls.return_value = mock_embedding_engine
        manager = MemoryManager(mock_hass)
        return manager


async def test_init_db(mock_hass, mock_embedding_engine):
    """Test database initialization."""
    with patch("sqlite3.connect") as mock_connect, \
            patch("custom_components.ai_memory.memory_manager.EmbeddingEngine") as mock_engine_cls:
        mock_engine_cls.return_value = mock_embedding_engine
        mock_cursor = MagicMock()
        mock_connect.return_value.__enter__.return_value.cursor.return_value = mock_cursor

        MemoryManager(mock_hass)

        # Verify table creation
        mock_cursor.execute.assert_called()
        assert "CREATE TABLE IF NOT EXISTS memories" in mock_cursor.execute.call_args[0][0]


async def test_add_memory_sql(memory_manager, mock_db):
    """Test adding memory executes INSERT."""
    # Mock count query
    mock_db.fetchall.return_value = [[0]]

    await memory_manager.async_add_memory("Test", "private", "agent_1")

    # Verify INSERT
    insert_call = [c for c in mock_db.execute.call_args_list if "INSERT INTO" in c[0][0]]
    assert len(insert_call) == 1
    args = insert_call[0][0][1]
    assert args[1] == "Test"  # content
    assert args[3] == "private"  # scope
    assert args[4] == "agent_1"  # agent_id


async def test_async_get_memory_counts(mock_hass, memory_manager, mock_db):
    """Test getting memory counts."""
    # Mock return for SELECT scope, COUNT(*)
    mock_db.fetchall.return_value = [
        ("common", 1),
        ("private", 1)
    ]
    
    counts = await memory_manager.async_get_memory_counts()
    assert counts["common"] == 1
    assert counts["private"] == 1
    assert counts["total"] == 2

async def test_async_search_memory(memory_manager, mock_db):
    """Test search executes SELECT and filters."""
    # Mock SELECT return
    mock_db.fetchall.return_value = [
        ("Content A", json.dumps([1.0, 0.0, 0.0]), json.dumps({"scope": "private"})),
        ("Content B", json.dumps([-1.0, 0.0, 0.0]), json.dumps({"scope": "common"}))
    ]

    results = await memory_manager.async_search_memory("query", "agent_1")

    # Verify SELECT
    select_call = [c for c in mock_db.execute.call_args_list if "SELECT" in c[0][0]]
    assert len(select_call) == 1
    assert "scope = 'common'" in select_call[0][0][0]

    # Verify filtering (Content A should match [1,0,0], Content B should not [-1,0,0])
    assert len(results) == 1
    assert results[0]["content"] == "Content A"


async def test_get_all_memories_sql(memory_manager, mock_db):
    """Test get all executes SELECT."""
    mock_db.fetchall.return_value = [
        ("Content A", json.dumps({"scope": "private"}))
    ]

    results = await memory_manager.async_get_all_memories("agent_1")

    assert len(results) == 1
    assert results[0]["content"] == "Content A"

    # Verify SELECT
    select_call = [c for c in mock_db.execute.call_args_list if "SELECT" in c[0][0]]
    assert len(select_call) == 1


async def test_init_db_failure(mock_hass):
    """Test database initialization failure."""
    with patch("sqlite3.connect", side_effect=Exception("DB Error")):
        MemoryManager(mock_hass)
        # Should log error but not crash


async def test_execute_query_failure(memory_manager, mock_db):
    """Test query execution failure."""
    mock_db.execute.side_effect = Exception("Query Error")
    results = await memory_manager.async_search_memory("query", "agent_1")
    assert results == []


async def test_add_memory_invalid_input(memory_manager, mock_db):
    """Test adding memory with invalid input."""
    # Empty content
    await memory_manager.async_add_memory("", "private", "agent_1")
    # Verify NO INSERT
    insert_call = [c for c in mock_db.execute.call_args_list if "INSERT INTO" in c[0][0]]
    assert len(insert_call) == 0

    # Invalid scope
    with pytest.raises(ValueError, match="Invalid scope"):
        await memory_manager.async_add_memory("Test", "invalid_scope")

    # Private scope without agent_id
    with pytest.raises(ValueError, match="Agent ID required"):
        await memory_manager.async_add_memory("Test", "private", None)


async def test_add_memory_embedding_failure(memory_manager, mock_embedding_engine, mock_db):
    """Test adding memory when embedding generation fails."""
    mock_embedding_engine.async_generate_embedding.side_effect = Exception("Embedding Error")
    mock_db.fetchall.return_value = [[0]]  # Mock count

    await memory_manager.async_add_memory("Test", "common")

    # Should still insert with empty embedding
    insert_call = [c for c in mock_db.execute.call_args_list if "INSERT INTO" in c[0][0]]
    assert len(insert_call) == 1
    args = insert_call[0][0][1]
    assert args[2] == "[]"  # Empty embedding json


async def test_search_memory_embedding_failure(memory_manager, mock_embedding_engine, mock_db):
    """Test search when query embedding generation fails."""
    mock_embedding_engine.async_generate_embedding.side_effect = Exception("Embedding Error")

    results = await memory_manager.async_search_memory("query", "agent_1")
    assert results == []


async def test_search_memory_malformed_db_data(memory_manager, mock_db):
    """Test search with malformed JSON in DB."""
    mock_db.fetchall.return_value = [
        ("Content A", "invalid_json", "{}"),
        ("Content B", "[1.0, 0.0]", "{}")  # Mismatched dimension (assuming query is 3D)
    ]

    results = await memory_manager.async_search_memory("query", "agent_1")
    # Should skip invalid rows
    assert len(results) == 0


async def test_cosine_similarity_edge_cases(memory_manager):
    """Test cosine similarity edge cases."""
    # Zero vectors
    assert memory_manager._cosine_similarity([0, 0], [0, 0]) == 0.0
    # Mismatched length
    assert memory_manager._cosine_similarity([1], [1, 2]) == 0.0
    # Empty
    assert memory_manager._cosine_similarity([], []) == 0.0


async def test_clear_memory(memory_manager, mock_db):
    """Test clearing memory."""
    await memory_manager.async_clear_memory()
    mock_db.execute.assert_called_with("DELETE FROM memories", ())


async def test_ensure_directory_exists_failure(mock_hass):
    """Test directory creation failure."""
    with patch("os.makedirs", side_effect=Exception("Permission Denied")):
        # We need to instantiate MemoryManager which calls _ensure_directory_exists in __init__
        # We also need to mock other init calls to isolate the failure
        with patch("custom_components.ai_memory.embedding.EmbeddingEngine"), \
                patch("sqlite3.connect"):
            manager = MemoryManager(mock_hass)
            # It catches exception and logs error, so initialization should proceed (or at least not crash)
            # But wait, _ensure_directory_exists returns False. The init doesn't check return value.
            # So it just logs.
    async def test_async_initialize_failure(memory_manager, mock_embedding_engine):
        """Test initialization failure when remote is down."""
        # Mock engine to have async_get_version returning False
        mock_embedding_engine._engine = AsyncMock()
        mock_embedding_engine._engine.async_get_version.return_value = False
        
        # We need to re-assign the engine to the manager's internal engine wrapper
        memory_manager._embedding_engine = mock_embedding_engine
        
        with pytest.raises(RuntimeError, match="Remote embedding service is not reachable"):
            await memory_manager.async_initialize()
