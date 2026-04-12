"""Semantic search engine for AI Memory integration."""
import json
import logging
from typing import List, Dict, Optional

import numpy as np

from ..constants import SIMILARITY_THRESHOLD, MEMORY_LIMIT, EMBEDDINGS_VECTOR_DIM
from .store import MemoryStore

_LOGGER = logging.getLogger(__name__)


class MemorySearch:
    """Semantic search using vector similarity (NumPy optimized)."""

    def __init__(self, store: MemoryStore, embedding_engine):
        """Initialize search engine.

        Args:
            store: MemoryStore instance for database access.
            embedding_engine: EmbeddingEngine instance for generating query embeddings.
        """
        self._store = store
        self._embedding_engine = embedding_engine

    def _generate_embedding_sync(self, text: str) -> List[float]:
        """Generate embedding synchronously (called via executor)."""
        return self._embedding_engine._generate_embedding_sync(text)

    async def async_generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the embedding engine."""
        if not text:
            dim = self._store.get_embedding_dim()
            return [0.0] * dim

        from homeassistant.core import HomeAssistant
        # Delegate to the embedding engine's async method
        return await self._embedding_engine.async_generate_embedding(text)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity using NumPy."""
        if vec1.shape != vec2.shape:
            return 0.0

        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm_v1 * norm_v2))

    async def async_search(
        self,
        query: str,
        agent_id: Optional[str],
        limit: int = MEMORY_LIMIT,
        min_score: float = SIMILARITY_THRESHOLD,
        wing: Optional[str] = None,
        room: Optional[str] = None,
        hass=None,
    ) -> List[Dict]:
        """Search memory using SQL filter + Vector Similarity.

        Args:
            query: Search query text.
            agent_id: Agent ID for scope filtering.
            limit: Maximum number of results.
            min_score: Minimum cosine similarity threshold.
            wing: Optional wing filter.
            room: Optional room filter.
            hass: HomeAssistant instance for executor jobs.

        Returns:
            List of matching memory dictionaries sorted by score.
        """
        if not query:
            return []

        # Normalize wing/room filters to lowercase
        if wing:
            wing = wing.lower().strip()
        if room:
            room = room.lower().strip()

        # Build SQL WHERE clause with optional wing/room filters
        sql = """SELECT id, content, embedding, scope, agent_id, created_at,
                        summary, wing, room, layer, access_count
                 FROM memories
                 WHERE (scope = 'common' OR (scope = 'private' AND agent_id = ?))"""
        params: list = [agent_id]

        if wing:
            sql += " AND wing = ?"
            params.append(wing)

        if room:
            sql += " AND room = ?"
            params.append(room)

        # Execute query
        rows = await hass.async_add_executor_job(
            self._store.execute_query, sql, tuple(params)
        )

        if not rows:
            return []

        # Generate query embedding
        try:
            query_embedding = await self.async_generate_embedding(query)
        except Exception as e:
            _LOGGER.error("Failed to generate query embedding: %s", e)
            return []

        if not query_embedding:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)

        # Score memories using cosine similarity
        scored_memories = []
        result_ids = []

        for row in rows:
            memory_id, content, emb_json, scope, row_agent_id, created_at, \
                summary, mem_wing, mem_room, layer, access_count = row

            try:
                mem_embedding_list = json.loads(emb_json) if emb_json else None
                if not mem_embedding_list:
                    continue

                mem_vec = np.array(mem_embedding_list, dtype=np.float32)
                score = self._cosine_similarity(query_vec, mem_vec)

                if score > min_score:
                    _LOGGER.debug("[%.3f] %s", score, content)
                    result_ids.append(memory_id)
                    scored_memories.append({
                        "id": memory_id,
                        "content": content,
                        "score": float(score),
                        "scope": scope,
                        "agent_id": row_agent_id,
                        "created_at": created_at,
                        "summary": summary,
                        "wing": mem_wing,
                        "room": mem_room,
                        "layer": layer,
                    })
            except Exception as e:
                _LOGGER.warning("Error processing memory row: %s", e)
                continue

        # Sort by score descending
        scored_memories.sort(key=lambda x: x["score"], reverse=True)
        result = scored_memories[:limit]

        # Text fallback when semantic search returns nothing
        if not result:
            result = await self._text_fallback_search(
                query, agent_id, wing, room, limit, hass
            )
            if result:
                return result

        # Update access_count for returned results (batch)
        if result_ids and hass:
            from datetime import datetime
            now = datetime.now().isoformat()
            # Only update IDs that made it into the result
            update_ids = [m["id"] for m in result]
            if update_ids:
                placeholders = ",".join("?" for _ in update_ids)
                try:
                    await hass.async_add_executor_job(
                        self._store.execute_commit,
                        f"UPDATE memories SET access_count = access_count + 1, accessed_at = ? WHERE id IN ({placeholders})",
                        (now, *update_ids),
                    )
                except Exception as e:
                    _LOGGER.debug("Access count update skipped: %s", e)

        return result

    async def _text_fallback_search(
        self,
        query: str,
        agent_id: Optional[str],
        wing: Optional[str],
        room: Optional[str],
        limit: int,
        hass,
    ) -> List[Dict]:
        """Fallback text search when semantic search returns nothing.

        Searches content and summary fields using LIKE matching.
        Each token in the query is matched independently (OR logic).
        """
        tokens = [t.lower() for t in query.split() if len(t) > 1]
        if not tokens or not hass:
            return []

        conditions = []
        params: list = [agent_id]
        for token in tokens:
            conditions.append("(LOWER(content) LIKE ? OR LOWER(summary) LIKE ?)")
            params.extend([f"%{token}%", f"%{token}%"])

        sql = f"""SELECT id, content, scope, agent_id, created_at,
                         summary, wing, room, layer
                  FROM memories
                  WHERE (scope = 'common' OR (scope = 'private' AND agent_id = ?))
                  AND ({" OR ".join(conditions)})"""

        if wing:
            sql += " AND wing = ?"
            params.append(wing)
        if room:
            sql += " AND room = ?"
            params.append(room)

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        try:
            rows = await hass.async_add_executor_job(
                self._store.execute_query, sql, tuple(params)
            )
        except Exception as e:
            _LOGGER.debug("Text fallback search failed: %s", e)
            return []

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "content": row[1],
                "score": 0.0,
                "scope": row[2],
                "agent_id": row[3],
                "created_at": row[4],
                "summary": row[5],
                "wing": row[6],
                "room": row[7],
                "layer": row[8],
                "match_type": "text",
            })

        if results:
            _LOGGER.debug("Text fallback found %d results for: %s", len(results), query)

        return results
