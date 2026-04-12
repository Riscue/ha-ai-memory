"""Tests for MemoryStore (SQLite connection manager)."""
import sqlite3
from unittest.mock import patch

import pytest

from custom_components.ai_memory.memory.store import MemoryStore


@pytest.fixture
def store():
    """Create in-memory MemoryStore."""
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_connection_lazy(store):
    """Test connection is created lazily."""
    assert store._conn is None
    store.execute_query("SELECT 1")
    assert store._conn is not None


def test_connection_reuse(store):
    """Test connection is reused across calls."""
    store.execute_query("SELECT 1")
    conn1 = store._conn
    store.execute_query("SELECT 2")
    assert store._conn is conn1


def test_wal_mode(store):
    """Test WAL mode is set (returns 'memory' for in-memory DBs)."""
    store.execute_query("SELECT 1")
    rows = store.execute_query("PRAGMA journal_mode")
    # In-memory databases report 'memory', file-based report 'wal'
    assert rows[0][0] in ("wal", "memory")


def test_busy_timeout(store):
    """Test busy timeout is set."""
    store.execute_query("SELECT 1")
    rows = store.execute_query("PRAGMA busy_timeout")
    assert rows[0][0] == 5000


def test_execute_query(store):
    """Test basic read query."""
    store.execute_commit("CREATE TABLE test (id INTEGER, value TEXT)")
    store.execute_commit("INSERT INTO test VALUES (1, 'hello')")
    rows = store.execute_query("SELECT * FROM test")
    assert rows == [(1, "hello")]


def test_execute_commit_transaction(store):
    """Test write with explicit transaction."""
    store.execute_commit("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT)")
    store.execute_commit("INSERT INTO test VALUES (1, 'a')")
    rows = store.execute_query("SELECT val FROM test WHERE id = 1")
    assert rows[0][0] == "a"


def test_execute_commit_rollback_on_error(store):
    """Test transaction rollback on error."""
    store.execute_commit("CREATE TABLE test (id INTEGER PRIMARY KEY, val TEXT NOT NULL)")
    store.execute_commit("INSERT INTO test VALUES (1, 'a')")

    # This should fail (NULL constraint violation)
    with pytest.raises(Exception):
        store.execute_commit("INSERT INTO test VALUES (2, NULL)")

    # Original data should still exist
    rows = store.execute_query("SELECT COUNT(*) FROM test")
    assert rows[0][0] == 1


def test_execute_commit_many(store):
    """Test batch write."""
    store.execute_commit("CREATE TABLE test (id INTEGER, val TEXT)")
    params_list = [(1, "a"), (2, "b"), (3, "c")]
    store.execute_commit_many("INSERT INTO test VALUES (?, ?)", params_list)

    rows = store.execute_query("SELECT COUNT(*) FROM test")
    assert rows[0][0] == 3


def test_execute_commit_many_empty(store):
    """Test batch write with empty list does nothing."""
    store.execute_commit_many("SELECT 1", [])  # Should not raise


def test_validate_embedding_valid():
    """Test valid embedding is serialized to JSON."""
    import json
    embedding = [0.1] * 384
    result = MemoryStore.validate_embedding(embedding)
    assert result is not None
    assert json.loads(result) == embedding


def test_validate_embedding_empty():
    """Test empty embedding returns None."""
    assert MemoryStore.validate_embedding([]) is None
    assert MemoryStore.validate_embedding(None) is None


def test_validate_embedding_wrong_dimension():
    """Test wrong dimension returns None when expected_dim is specified."""
    assert MemoryStore.validate_embedding([0.1] * 128, expected_dim=384) is None
    assert MemoryStore.validate_embedding([0.1] * 383, expected_dim=384) is None


def test_validate_embedding_any_dimension():
    """Test without expected_dim accepts any valid vector."""
    import json
    result = MemoryStore.validate_embedding([0.1] * 128)
    assert result is not None
    assert len(json.loads(result)) == 128


def test_close(store):
    """Test close releases connection."""
    store.execute_query("SELECT 1")
    assert store._conn is not None
    store.close()
    assert store._conn is None


def test_close_idempotent(store):
    """Test close can be called multiple times."""
    store.execute_query("SELECT 1")
    store.close()
    store.close()  # Should not raise
    assert store._conn is None
