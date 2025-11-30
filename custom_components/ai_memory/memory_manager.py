import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any

from homeassistant.core import HomeAssistant

from . import ENGINE_TFIDF
from .constants import MEMORY_MAX_ENTRIES, DEFAULT_STORAGE_PATH
from .embedding import EmbeddingEngine

_LOGGER = logging.getLogger(__name__)


class MemoryManager:
    """Manages the memory storage using SQLite."""

    def __init__(
            self,
            hass: HomeAssistant,
            engine_type: str = ENGINE_TFIDF,
            max_entries: int = MEMORY_MAX_ENTRIES,
            config_data: dict = None
    ):
        self.hass = hass
        self._max_entries = max_entries
        self._embedding_engine = EmbeddingEngine(hass, engine_type, config_data)

        # Initialize DB
        self._db_path = DEFAULT_STORAGE_PATH
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                               CREATE TABLE IF NOT EXISTS memories
                               (
                                   id
                                   TEXT
                                   PRIMARY
                                   KEY,
                                   content
                                   TEXT,
                                   embedding
                                   TEXT,
                                   scope
                                   TEXT,
                                   agent_id
                                   TEXT,
                                   created_at
                                   TEXT,
                                   metadata
                                   TEXT
                               )
                               """)
                conn.commit()
        except Exception as e:
            _LOGGER.error(f"Failed to initialize database: {e}")

    def _execute_query(self, query: str, params: tuple = ()) -> List[Any]:
        """Execute a read query."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            _LOGGER.error(f"Database read error: {e}")
            return []

    def _execute_commit(self, query: str, params: tuple = ()):
        """Execute a write query."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
        except Exception as e:
            _LOGGER.error(f"Database write error: {e}")

    async def async_load_memories(self):
        """No-op for SQLite (data is on disk)."""
        pass

    async def async_initialize(self):
        """Initialize the memory manager and embedding engine."""
        if self._embedding_engine:
            await self._embedding_engine.async_initialize()

    async def async_add_memory(self, content: str, scope: str, agent_id: Optional[str] = None):
        """Add new memory entry."""
        if not content or not content.strip():
            _LOGGER.warning("Cannot add empty memory")
            return

        if scope not in ["common", "private"]:
            raise ValueError(f"Invalid scope: {scope}")

        if scope == "private" and not agent_id:
            raise ValueError("Agent ID required for private scope")

        # Check limit (simplistic check)
        count_res = await self.hass.async_add_executor_job(
            self._execute_query, "SELECT COUNT(*) FROM memories"
        )
        if count_res and count_res[0][0] >= self.max_entries:
            # Remove oldest
            await self.hass.async_add_executor_job(
                self._execute_commit,
                "DELETE FROM memories WHERE id = (SELECT id FROM memories ORDER BY created_at ASC LIMIT 1)"
            )

        # Generate embedding
        embedding = []
        try:
            embedding = await self._embedding_engine.async_generate_embedding(content)
        except Exception as e:
            _LOGGER.error(f"Failed to generate embedding: {e}")

        # Prepare data
        mem_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        metadata = {
            "type": "memory",
            "scope": scope,
            "agent_id": agent_id,
            "created_at": created_at
        }

        await self.hass.async_add_executor_job(
            self._execute_commit,
            """
            INSERT INTO memories (id, content, embedding, scope, agent_id, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mem_id,
                content.strip(),
                json.dumps(embedding),
                scope,
                agent_id,
                created_at,
                json.dumps(metadata)
            )
        )
        _LOGGER.debug(f"Added memory to DB (scope={scope}): {content[:30]}...")

        # Update vocabulary for TF-IDF engine (improves future embeddings)
        if self._embedding_engine:
            try:
                await self._embedding_engine.async_update_vocabulary(content)
            except Exception as e:
                _LOGGER.debug(f"Vocabulary update skipped: {e}")

        if hasattr(self.hass, 'bus'):
            self.hass.bus.async_fire("ai_memory_updated")

    async def async_search_memory(self, query: str, agent_id: Optional[str], limit: int = 5) -> List[Dict]:
        """Search memory using SQL filter + Vector Similarity."""
        if not query:
            return []

        # 1. SQL Filter (Hard Filter)
        # Select rows where scope is common OR (private AND agent_id matches)
        rows = await self.hass.async_add_executor_job(
            self._execute_query,
            """
            SELECT content, embedding, metadata
            FROM memories
            WHERE scope = 'common'
               OR (scope = 'private' AND agent_id = ?)
            """,
            (agent_id,)
        )

        if not rows:
            return []

        # Generate query embedding
        try:
            query_embedding = await self._embedding_engine.async_generate_embedding(query)
        except Exception as e:
            _LOGGER.error(f"Failed to generate query embedding: {e}")
            return []

        if not query_embedding:
            return []

        # 2. Vector Similarity (Soft Filter)
        scored_memories = []
        for content, emb_json, meta_json in rows:
            try:
                mem_embedding = json.loads(emb_json)
                if not mem_embedding:
                    continue

                score = self._cosine_similarity(query_embedding, mem_embedding)
                scored_memories.append({
                    "content": content,
                    "score": score,
                    "metadata": json.loads(meta_json)
                })
            except Exception as e:
                _LOGGER.warning(f"Error processing memory row: {e}")
                continue

        # Sort and limit
        scored_memories.sort(key=lambda x: x["score"], reverse=True)

        # Filter by threshold
        results = [m for m in scored_memories[:limit] if m["score"] > 0.1]
        return results

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity."""
        import math
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)

    async def async_get_all_memories(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Get all accessible memories."""
        rows = await self.hass.async_add_executor_job(
            self._execute_query,
            """
            SELECT content, metadata
            FROM memories
            WHERE scope = 'common'
               OR (scope = 'private' AND agent_id = ?)
            ORDER BY created_at DESC
            """,
            (agent_id,)
        )

        results = []
        for content, meta_json in rows:
            try:
                results.append({
                    "content": content,
                    "metadata": json.loads(meta_json)
                })
            except:
                pass
        return results

    async def async_clear_memory(self):
        """Clear all memories."""
        await self.hass.async_add_executor_job(
            self._execute_commit, "DELETE FROM memories"
        )
