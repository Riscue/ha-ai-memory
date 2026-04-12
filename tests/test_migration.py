"""Tests for MigrationManager."""
import pytest

from custom_components.ai_memory.memory.store import MemoryStore
from custom_components.ai_memory.memory.migration import MigrationManager


@pytest.fixture
def store():
    """Create in-memory store."""
    s = MemoryStore(":memory:")
    yield s
    s.close()


def test_fresh_install(store):
    """Test migration on a fresh database creates all tables."""
    MigrationManager(store).migrate()

    # Check _meta table
    rows = store.execute_query("SELECT value FROM _meta WHERE key = 'db_version'")
    assert rows[0][0] == "1"

    # Check memories table columns
    columns = [row[1] for row in store.execute_query("PRAGMA table_info(memories)")]
    expected = ["id", "content", "embedding", "scope", "agent_id", "created_at",
                "summary", "wing", "room", "layer", "updated_at", "accessed_at", "access_count"]
    for col in expected:
        assert col in columns, f"Missing column: {col}"

    # Check knowledge_graph table
    rows = store.execute_query("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_graph'")
    assert len(rows) == 1

    # Check palace_structure table
    rows = store.execute_query("SELECT name FROM sqlite_master WHERE type='table' AND name='palace_structure'")
    assert len(rows) == 1


def test_migration_idempotent(store):
    """Test running migration twice is safe."""
    mgr = MigrationManager(store)
    mgr.migrate()
    mgr.migrate()  # Should not raise

    # Version should still be 1
    rows = store.execute_query("SELECT value FROM _meta WHERE key = 'db_version'")
    assert rows[0][0] == "1"


def test_v0_to_v1_upgrade(store):
    """Test upgrade from v0 schema (original table) to v1."""
    # Create original v0 table (no new columns)
    store.execute_commit(
        """CREATE TABLE memories (
            id TEXT PRIMARY KEY,
            content TEXT,
            embedding TEXT,
            scope TEXT,
            agent_id TEXT,
            created_at TEXT
        )"""
    )
    # Insert a v0 record
    store.execute_commit(
        "INSERT INTO memories (id, content, scope, agent_id, created_at) VALUES (?, ?, ?, ?, ?)",
        ("test-id", "Hello", "common", "agent_1", "2025-01-01"),
    )

    # Run migration
    MigrationManager(store).migrate()

    # New columns should exist
    columns = [row[1] for row in store.execute_query("PRAGMA table_info(memories)")]
    assert "summary" in columns
    assert "wing" in columns
    assert "room" in columns
    assert "layer" in columns
    assert "access_count" in columns

    # Original data preserved
    rows = store.execute_query("SELECT content, agent_id FROM memories WHERE id = 'test-id'")
    assert len(rows) == 1
    assert rows[0][0] == "Hello"
    assert rows[0][1] == "agent_1"

    # Default values applied
    rows = store.execute_query("SELECT wing, room, layer FROM memories WHERE id = 'test-id'")
    assert rows[0][0] == "general"
    assert rows[0][1] == "general"
    assert rows[0][2] == 2


def test_indexes_created(store):
    """Test that required indexes are created."""
    MigrationManager(store).migrate()

    indexes = store.execute_query(
        "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
    )
    index_names = [row[0] for row in indexes]
    assert "idx_memories_wing_room" in index_names
    assert "idx_memories_layer" in index_names
    assert "idx_memories_scope_agent" in index_names
    assert "idx_kg_subject" in index_names
