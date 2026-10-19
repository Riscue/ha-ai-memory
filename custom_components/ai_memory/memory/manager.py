"""Memory Manager for AI Memory integration (refactored)."""
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .migration import MigrationManager
from .search import MemorySearch
from .store import MemoryStore
from ..constants import (
    MEMORY_MAX_ENTRIES,
    DEFAULT_STORAGE_PATH,
    DEFAULT_LAYER,
    ENGINE_TFIDF,
    META_TFIDF_HASH,
    TFIDF_HASH_VERSION,
)
from ..embedding.engine import EmbeddingEngine
from ..palace.metadata import RoomDetector
from ..palace.structure import PalaceStructure

_LOGGER = logging.getLogger(__name__)


class MemoryManager:
    """Manages the memory storage using SQLite with wing/room support."""

    def __init__(
        self,
        hass: HomeAssistant,
        engine_type: str = "tfidf",
        max_entries: int = MEMORY_MAX_ENTRIES,
        db_path: Optional[str] = None,
        config_data: dict = None,
    ):
        """Create the manager. No I/O happens here; call async_initialize()."""
        self.hass = hass
        self._max_entries = max_entries
        self._config_data = config_data or {}
        self._engine_type = engine_type
        self._db_path = db_path or hass.config.path(DEFAULT_STORAGE_PATH)

        self._store = MemoryStore(self._db_path)
        self._palace = PalaceStructure(self._store)
        self._room_detector = RoomDetector(self._store)

        # Initialize embedding engine
        self._embedding_engine = EmbeddingEngine(hass, engine_type, config_data)

        # Initialize search engine
        self._search = MemorySearch(self._store, self._embedding_engine)

    def initialize_store(self):
        """Run migrations and seed palace defaults (blocking, run in executor)."""
        MigrationManager(self._store).migrate()
        self._palace.initialize_defaults()

    def _reembed_tfidf_if_needed(self) -> int:
        """Re-embed all memories if they predate the stable TF-IDF hash (blocking).

        TF-IDF vectors used to be built with Python's per-process randomized
        hash(), so vectors stored before a restart never match new queries.
        """
        rows = self._store.execute_query(
            "SELECT value FROM _meta WHERE key = ?", (META_TFIDF_HASH,)
        )
        if rows and rows[0][0] == TFIDF_HASH_VERSION:
            return 0

        engine = self._embedding_engine._engine
        rows = self._store.execute_query("SELECT id, content, summary FROM memories")
        updates = []
        for mem_id, content, summary in rows:
            embedding = engine.generate_embedding(summary or content or "")
            updates.append((self._store.validate_embedding(embedding), mem_id))

        self._store.execute_commit_many(
            "UPDATE memories SET embedding = ? WHERE id = ?", updates
        )
        self._store.set_embedding_dim(engine.vector_dim)
        self._store.execute_commit(
            "INSERT OR REPLACE INTO _meta (key, value) VALUES (?, ?)",
            (META_TFIDF_HASH, TFIDF_HASH_VERSION),
        )
        return len(updates)

    def _clear_tfidf_marker(self):
        """Forget the TF-IDF marker so switching back to TF-IDF re-embeds (blocking)."""
        self._store.execute_commit(
            "DELETE FROM _meta WHERE key = ?", (META_TFIDF_HASH,)
        )

    async def async_initialize(self):
        """Initialize the database and embedding engine."""
        await self.hass.async_add_executor_job(self.initialize_store)

        if self._embedding_engine:
            await self._embedding_engine.async_initialize()

            # If remote, check connection and pull
            if hasattr(self._embedding_engine._engine, "async_get_version"):
                is_ready = await self._embedding_engine._engine.async_get_version()
                if is_ready:
                    _LOGGER.info("Remote embedding service is reachable.")
                    if hasattr(self._embedding_engine._engine, "async_load_model"):
                        await self._embedding_engine._engine.async_load_model()
                else:
                    # HA retries setup with backoff until the service comes up
                    raise ConfigEntryNotReady("Remote embedding service is not reachable")

            if self._engine_type == ENGINE_TFIDF:
                count = await self.hass.async_add_executor_job(
                    self._reembed_tfidf_if_needed
                )
                if count:
                    _LOGGER.info("Re-embedded %d memories with stable TF-IDF hashing", count)
            else:
                await self.hass.async_add_executor_job(self._clear_tfidf_marker)

    async def async_get_palace_stats(self) -> Dict[str, int]:
        """Get palace structure statistics."""
        return await self.hass.async_add_executor_job(self._palace.get_stats)

    async def async_get_memory_counts(self) -> Dict[str, int]:
        """Get counts of memories by scope."""
        counts = {"common": 0, "private": 0, "total": 0}
        try:
            rows = await self.hass.async_add_executor_job(
                self._store.execute_query,
                "SELECT scope, COUNT(*) FROM memories GROUP BY scope",
            )
            for scope, count in rows:
                counts[scope] = count
                counts["total"] += count
        except Exception as e:
            _LOGGER.error("Failed to get memory counts: %s", e)
        return counts

    async def async_get_layer_counts(self) -> Dict[str, int]:
        """Get counts of memories by layer."""
        counts = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
        try:
            rows = await self.hass.async_add_executor_job(
                self._store.execute_query,
                "SELECT layer, COUNT(*) FROM memories GROUP BY layer",
            )
            for layer, count in rows:
                key = f"L{layer}"
                if key in counts:
                    counts[key] = count
        except Exception as e:
            _LOGGER.error("Failed to get layer counts: %s", e)
        return counts

    async def async_get_wing_counts(self) -> Dict[str, int]:
        """Get counts of memories by wing."""
        counts = {}
        try:
            rows = await self.hass.async_add_executor_job(
                self._store.execute_query,
                "SELECT wing, COUNT(*) FROM memories GROUP BY wing",
            )
            for wing, count in rows:
                counts[wing] = count
        except Exception as e:
            _LOGGER.error("Failed to get wing counts: %s", e)
        return counts

    async def async_get_memories(
        self,
        limit: int = 50,
        room: Optional[str] = None,
        wing: Optional[str] = None,
        scope: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict]:
        """Get memories with optional filtering."""
        conditions = []
        params = []
        
        if room:
            conditions.append("room = ?")
            params.append(room)
        if wing:
            conditions.append("wing = ?")
            params.append(wing)
        if scope:
            conditions.append("scope = ?")
            params.append(scope)
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
            
        query = "SELECT id, content, scope, agent_id, created_at, summary, wing, room, layer FROM memories"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        memories = []
        try:
            rows = await self.hass.async_add_executor_job(
                self._store.execute_query,
                query,
                tuple(params),
            )
            for row in rows:
                memories.append({
                    "id": row[0],
                    "content": row[1],
                    "scope": row[2],
                    "agent_id": row[3],
                    "created_at": row[4],
                    "summary": row[5],
                    "wing": row[6],
                    "room": row[7],
                    "layer": row[8],
                })
        except Exception as e:
            _LOGGER.error("Failed to get memories: %s", e)
        return memories

    async def async_add_memory(
        self,
        content: str,
        scope: str,
        agent_id: Optional[str] = None,
        summary: Optional[str] = None,
        wing: Optional[str] = None,
        room: Optional[str] = None,
    ):
        """Add new memory entry.

        Args:
            content: The original text to store (Drawer).
            scope: Memory scope ('private' or 'common').
            agent_id: Agent ID (required for private scope).
            summary: Optional summary/keywords (Closet). If provided, embedding uses this.
            wing: Optional wing assignment. Auto-detected if empty.
            room: Optional room assignment. Auto-detected if empty.
        """
        if not content or not content.strip():
            _LOGGER.warning("Cannot add empty memory")
            return

        if scope not in ["common", "private"]:
            raise ValueError(f"Invalid scope: {scope}")

        if scope == "private" and not agent_id:
            raise ValueError("Agent ID required for private scope")

        # Check limit
        count_res = await self.hass.async_add_executor_job(
            self._store.execute_query,
            "SELECT COUNT(*) FROM memories",
        )
        if count_res and count_res[0][0] >= self._max_entries:
            await self.hass.async_add_executor_job(
                self._store.execute_commit,
                "DELETE FROM memories WHERE id = (SELECT id FROM memories ORDER BY created_at ASC LIMIT 1)",
            )

        # Determine wing/room (auto-detect if not provided)
        if not wing or not room:
            detected_wing, detected_room = await self.hass.async_add_executor_job(
                self._room_detector.detect, content, scope
            )
            wing = wing or detected_wing
            room = room or detected_room

        # Normalize and validate wing/room (lowercase, auto-create if unknown)
        if wing and room:
            wing, room = await self.hass.async_add_executor_job(
                self._palace.validate_or_create_room, wing, room, scope
            )

        # Generate embedding from summary (if available) or content
        embedding_text = summary if summary else content
        embedding = None
        try:
            raw_embedding = await self._embedding_engine.async_generate_embedding(embedding_text)
            if raw_embedding:
                # Auto-detect and persist embedding dimension on first success
                current_dim = await self.hass.async_add_executor_job(
                    self._store.get_embedding_dim
                )
                if current_dim != len(raw_embedding):
                    await self.hass.async_add_executor_job(
                        self._store.set_embedding_dim, len(raw_embedding)
                    )
                embedding = self._store.validate_embedding(
                    raw_embedding, expected_dim=len(raw_embedding)
                )
        except Exception as e:
            _LOGGER.error("Failed to generate embedding: %s", e)

        # Prepare data
        mem_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()

        await self.hass.async_add_executor_job(
            self._store.execute_commit,
            """INSERT INTO memories
               (id, content, embedding, scope, agent_id, created_at,
                summary, wing, room, layer, updated_at, accessed_at, access_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mem_id,
                content.strip(),
                embedding,
                scope,
                agent_id,
                created_at,
                summary,
                wing,
                room,
                DEFAULT_LAYER,
                created_at,
                None,
                0,
            ),
        )

        # Update vocabulary for TF-IDF engine
        if self._embedding_engine:
            try:
                await self._embedding_engine.async_update_vocabulary(content)
            except Exception as e:
                _LOGGER.debug("Vocabulary update skipped: %s", e)

        if hasattr(self.hass, "bus"):
            self.hass.bus.async_fire("ai_memory_updated")

    async def async_search_memory(
        self,
        query: str,
        agent_id: Optional[str],
        limit: int = 5,
        min_score: float = 0.55,
        wing: Optional[str] = None,
        room: Optional[str] = None,
    ) -> List[Dict]:
        """Search memory using semantic similarity.

        Args:
            query: Search query text.
            agent_id: Agent ID for scope filtering.
            limit: Maximum results.
            min_score: Minimum similarity threshold.
            wing: Optional wing filter.
            room: Optional room filter.

        Returns:
            List of matching memory dictionaries.
        """
        return await self._search.async_search(
            query=query,
            agent_id=agent_id,
            limit=limit,
            min_score=min_score,
            wing=wing,
            room=room,
            hass=self.hass,
        )

    async def async_delete_memory(
            self,
            agent_id: Optional[str] = None,
            room: Optional[str] = None,
            wing: Optional[str] = None,
            scope: Optional[str] = None,
    ) -> int:
        """Delete memory entries by filter.

        Authorization: Private memories can only be deleted by their owner agent.
        Common memories can be deleted by any agent.

        Args:
            agent_id: The requesting agent's ID.
            room: Optional room name to delete all memories in that room.
            wing: Optional wing name to delete all memories in that wing.
            scope: Optional scope filter ('private', 'common', or None for both).

        Returns:
            Number of memories deleted.
        """
        try:
            # Build query conditions
            conditions = []
            params = []

            # Bulk deletion by room/wing
            if room:
                conditions.append("room = ?")
                params.append(room)
            if wing:
                conditions.append("wing = ?")
                params.append(wing)
            if scope:
                conditions.append("scope = ?")
                params.append(scope)

            if not conditions:
                _LOGGER.warning("No deletion criteria provided")
                return 0

            # Authorization for bulk deletion
            if agent_id:
                conditions.append("(scope = 'common' OR agent_id = ?)")
                params.append(agent_id)
            else:
                conditions.append("scope = 'common'")

            query = f"DELETE FROM memories WHERE {' AND '.join(conditions)}"

            # Get count before deletion
            count_query = f"SELECT COUNT(*) FROM memories WHERE {' AND '.join(conditions)}"
            count_res = await self.hass.async_add_executor_job(
                self._store.execute_query,
                count_query,
                tuple(params),
            )
            deleted_count = count_res[0][0] if count_res else 0

            # Execute deletion
            await self.hass.async_add_executor_job(
                self._store.execute_commit,
                query,
                tuple(params),
            )

            if hasattr(self.hass, "bus"):
                self.hass.bus.async_fire("ai_memory_updated")

            _LOGGER.info("Deleted %d memory(s)", deleted_count)
            return deleted_count
        except Exception as e:
            _LOGGER.error("Failed to delete memory: %s", e)
            return 0

    def close(self):
        """Close database connection and release resources."""
        self._store.close()
