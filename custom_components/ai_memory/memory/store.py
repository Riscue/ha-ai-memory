"""SQLite store with WAL mode, connection reuse, and transaction safety."""
import json
import logging
import sqlite3
from typing import List, Any, Optional

from ..constants import EMBEDDINGS_VECTOR_DIM

_LOGGER = logging.getLogger(__name__)


class MemoryStore:
    """Thread-safe SQLite store with WAL mode and connection reuse.

    All database access is serialized through HA's async_add_executor_job,
    so the connection is created lazily on the executor thread and reused.
    """

    def __init__(self, db_path: str):
        """Initialize store.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for testing.
        """
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._embedding_dim: Optional[int] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the SQLite connection (lazy, on executor thread)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            _LOGGER.debug("SQLite connection established (WAL mode, db=%s)", self._db_path)
        return self._conn

    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        """Execute a read query and return results.

        Args:
            query: SQL query string.
            params: Query parameters.

        Returns:
            List of result tuples.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except Exception as e:
            _LOGGER.error("Database read error: %s", e)
            return []

    def execute_commit(self, query: str, params: tuple = ()):
        """Execute a write query with explicit transaction.

        Args:
            query: SQL query string.
            params: Query parameters.
        """
        conn = self._get_connection()
        try:
            conn.execute("BEGIN")
            conn.execute(query, params)
            conn.execute("COMMIT")
        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            _LOGGER.error("Database write error: %s", e)
            raise

    def execute_commit_many(self, query: str, params_list: List[tuple]):
        """Execute a write query with multiple parameter sets in a single transaction.

        Args:
            query: SQL query string.
            params_list: List of parameter tuples.
        """
        if not params_list:
            return

        conn = self._get_connection()
        try:
            conn.execute("BEGIN")
            conn.executemany(query, params_list)
            conn.execute("COMMIT")
        except Exception as e:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            _LOGGER.error("Database batch write error: %s", e)
            raise

    @staticmethod
    def validate_embedding(embedding: Any, expected_dim: int = None) -> Optional[str]:
        """Validate and serialize an embedding vector.

        Args:
            embedding: List of floats or None.
            expected_dim: Expected dimension. If None, accepts any valid vector.

        Returns:
            JSON string of the embedding, or None if invalid/empty.
        """
        if not embedding or not isinstance(embedding, list):
            return None

        if expected_dim and len(embedding) != expected_dim:
            _LOGGER.warning(
                "Embedding dimension mismatch: expected %d, got %d",
                expected_dim,
                len(embedding),
            )
            return None

        return json.dumps(embedding)

    def get_embedding_dim(self) -> int:
        """Get the detected embedding dimension.

        Checks in order: 1) cached value, 2) _meta table, 3) existing embeddings in DB,
        4) falls back to EMBEDDINGS_VECTOR_DIM constant.
        """
        if self._embedding_dim is not None:
            return self._embedding_dim

        # Check _meta table
        rows = self.execute_query(
            "SELECT value FROM _meta WHERE key = 'embedding_dim'"
        )
        if rows:
            try:
                self._embedding_dim = int(rows[0][0])
                return self._embedding_dim
            except (ValueError, IndexError):
                pass

        # Check existing embeddings in DB
        rows = self.execute_query(
            "SELECT embedding FROM memories WHERE embedding IS NOT NULL LIMIT 1"
        )
        if rows and rows[0][0]:
            try:
                existing = json.loads(rows[0][0])
                if existing:
                    self._embedding_dim = len(existing)
                    self._persist_embedding_dim(self._embedding_dim)
                    _LOGGER.info("Auto-detected embedding dimension: %d (from existing data)", self._embedding_dim)
                    return self._embedding_dim
            except (json.JSONDecodeError, IndexError):
                pass

        # Fallback to constant
        self._embedding_dim = EMBEDDINGS_VECTOR_DIM
        return self._embedding_dim

    def set_embedding_dim(self, dim: int):
        """Set and persist the embedding dimension (called when first embedding is generated)."""
        if dim <= 0:
            return
        self._embedding_dim = dim
        self._persist_embedding_dim(dim)
        _LOGGER.info("Embedding dimension set to: %d", dim)

    def _persist_embedding_dim(self, dim: int):
        """Persist embedding dimension to _meta table."""
        try:
            self.execute_commit(
                "INSERT OR REPLACE INTO _meta (key, value) VALUES ('embedding_dim', ?)",
                (str(dim),),
            )
        except Exception as e:
            _LOGGER.debug("Could not persist embedding_dim: %s", e)

    def close(self):
        """Close the SQLite connection."""
        if self._conn is not None:
            try:
                self._conn.close()
                _LOGGER.debug("SQLite connection closed")
            except Exception as e:
                _LOGGER.warning("Error closing SQLite connection: %s", e)
            self._conn = None
