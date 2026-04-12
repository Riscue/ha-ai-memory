"""Database schema migration for AI Memory integration."""
import logging
from typing import List

from ..constants import DB_VERSION
from .store import MemoryStore

_LOGGER = logging.getLogger(__name__)


class MigrationManager:
    """Manages database schema migrations with version tracking."""

    def __init__(self, store: MemoryStore):
        """Initialize migration manager.

        Args:
            store: MemoryStore instance to run migrations on.
        """
        self._store = store

    def _get_version(self) -> int:
        """Get current database schema version."""
        rows = self._store.execute_query(
            "SELECT value FROM _meta WHERE key = 'db_version'"
        )
        if rows:
            try:
                return int(rows[0][0])
            except (ValueError, IndexError):
                return 0
        return 0

    def _set_version(self, version: int):
        """Set database schema version."""
        self._store.execute_commit(
            "INSERT OR REPLACE INTO _meta (key, value) VALUES ('db_version', ?)",
            (str(version),),
        )

    def _ensure_meta_table(self):
        """Create _meta table if it doesn't exist."""
        self._store.execute_commit(
            """CREATE TABLE IF NOT EXISTS _meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )"""
        )

    def _get_existing_columns(self, table: str) -> List[str]:
        """Get list of existing column names for a table."""
        rows = self._store.execute_query(f"PRAGMA table_info({table})")
        return [row[1] for row in rows]  # row[1] is column name

    def _table_exists(self, table: str) -> bool:
        """Check if a table exists."""
        rows = self._store.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return len(rows) > 0

    def _migrate_v0_to_v1(self):
        """Migrate from v0 (original schema) to v1 (Faz 1 schema).

        Adds: summary, wing, room, layer, updated_at, accessed_at, access_count
        Creates: knowledge_graph table, palace_structure table, new indexes.
        """
        _LOGGER.info("Running migration: v0 → v1")

        # Add new columns to memories table if they don't exist
        existing = self._get_existing_columns("memories")
        new_columns = {
            "summary": "TEXT",
            "wing": "TEXT DEFAULT 'general'",
            "room": "TEXT DEFAULT 'general'",
            "layer": "INTEGER DEFAULT 2",
            "updated_at": "TEXT",
            "accessed_at": "TEXT",
            "access_count": "INTEGER DEFAULT 0",
        }

        for col_name, col_type in new_columns.items():
            if col_name not in existing:
                self._store.execute_commit(
                    f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}"
                )
                _LOGGER.debug("Added column: memories.%s", col_name)

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_wing_room ON memories(wing, room)",
            "CREATE INDEX IF NOT EXISTS idx_memories_layer ON memories(layer)",
            "CREATE INDEX IF NOT EXISTS idx_memories_scope_agent ON memories(scope, agent_id)",
        ]
        for idx_sql in indexes:
            self._store.execute_commit(idx_sql)

        # Create knowledge_graph table
        self._store.execute_commit(
            """CREATE TABLE IF NOT EXISTS knowledge_graph (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                relationship TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                confidence REAL DEFAULT 1.0,
                source_memory_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""
        )

        kg_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_kg_subject ON knowledge_graph(subject)",
            "CREATE INDEX IF NOT EXISTS idx_kg_object ON knowledge_graph(object)",
            "CREATE INDEX IF NOT EXISTS idx_kg_valid ON knowledge_graph(valid_from, valid_to)",
            "CREATE INDEX IF NOT EXISTS idx_kg_relationship ON knowledge_graph(relationship)",
        ]
        for idx_sql in kg_indexes:
            self._store.execute_commit(idx_sql)

        # Create palace_structure table
        self._store.execute_commit(
            """CREATE TABLE IF NOT EXISTS palace_structure (
                wing TEXT NOT NULL,
                room TEXT NOT NULL,
                scope TEXT DEFAULT 'common',
                hall_connections TEXT DEFAULT '[]',
                tunnel_connections TEXT DEFAULT '[]',
                description TEXT,
                auto_assign_keywords TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (wing, room)
            )"""
        )

        # Set version
        self._set_version(1)
        _LOGGER.info("Migration v0 → v1 complete")

    def migrate(self):
        """Run all pending migrations."""
        self._ensure_meta_table()
        current_version = self._get_version()

        _LOGGER.info("Database schema version: %d, target: %d", current_version, DB_VERSION)

        if current_version < 1:
            # Ensure base memories table exists (for fresh installs)
            if not self._table_exists("memories"):
                self._store.execute_commit(
                    """CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT,
                        embedding TEXT,
                        scope TEXT,
                        agent_id TEXT,
                        created_at TEXT,
                        summary TEXT,
                        wing TEXT DEFAULT 'general',
                        room TEXT DEFAULT 'general',
                        layer INTEGER DEFAULT 2,
                        updated_at TEXT,
                        accessed_at TEXT,
                        access_count INTEGER DEFAULT 0
                    )"""
                )
            self._migrate_v0_to_v1()

        _LOGGER.info("Database schema up to date (v%d)", DB_VERSION)
