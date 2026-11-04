"""Semantic search engine for AI Memory integration."""
import json
import logging
import math
import re
from typing import List, Dict, Optional

import numpy as np

from ..constants import SIMILARITY_THRESHOLD, MEMORY_LIMIT
from .store import MemoryStore

_LOGGER = logging.getLogger(__name__)

TEXT_FALLBACK_MIN_TOKEN_LEN = 2

# Keywords shorter than this must match a whole word ("tv", "ac"); longer
# ones match word prefixes so Turkish suffixes still hit ("çay" -> "çayı").
TEXT_FALLBACK_PREFIX_MIN_LEN = 3


def _normalize(text: str) -> str:
    """Lowercase and fold Turkish dotted/dotless i to plain "i".

    str.lower() maps "İ" to "i" + U+0307 and "I" to "i" (never "ı"), so both
    sides are folded to the same form instead of relying on locale rules.
    """
    return text.lower().replace("\u0307", "").replace("ı", "i")


def _words(text: str) -> List[str]:
    """Split normalized text into word tokens."""
    return re.findall(r"\w+", _normalize(text))


# Question/filler words that would match unrelated memories in the text
# fallback. Memories are written in third person, so "user" is noise too.
# Matched against query tokens only, as normalized whole words.
TEXT_FALLBACK_STOPWORDS = frozenset(_normalize(w) for w in {
    # English
    "about", "all", "am", "an", "and", "any", "are", "as", "at", "be", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "how", "if", "in", "is", "it", "its", "know", "like", "liked",
    "likes", "me", "mine", "my", "no", "not", "of", "on", "or", "our",
    "remember", "should", "so", "some", "tell", "that", "the", "their",
    "them", "there", "these", "they", "this", "those", "to", "user", "users",
    "was", "we", "were", "what", "when", "where", "which", "who", "whom",
    "whose", "why", "will", "with", "would", "you", "your", "yours",
    # Turkish
    "acaba", "ama", "bana", "ben", "benim", "beni", "bir", "bu", "çok", "da",
    "daha", "de", "değil", "en", "gibi", "hangi", "için", "ile", "kadar", "ki",
    "kullanıcı", "kullanıcıya", "kullanıcıyı", "kullanıcının", "mı", "mi",
    "mu", "mü", "mıyım", "miyim", "muyum", "müyüm", "nasıl", "ne", "neden",
    "nedir", "neyi", "niye", "ona", "onu", "onun", "sana", "sen", "senin",
    "seni", "şu", "var", "ve", "veya", "ya", "yok",
})


def _keywords(query: str) -> List[str]:
    """Extract unique, meaningful normalized keywords from a query.

    Numbers are always kept ("8 AM", "22 derece").
    """
    seen: Dict[str, None] = {}
    for token in _words(query):
        if token in TEXT_FALLBACK_STOPWORDS:
            continue
        if len(token) >= TEXT_FALLBACK_MIN_TOKEN_LEN or token.isdigit():
            seen.setdefault(token, None)
    return list(seen)


def _keyword_hits(keywords: List[str], words: set) -> int:
    """Count keywords that appear in a set of words."""
    hits = 0
    for keyword in keywords:
        if len(keyword) < TEXT_FALLBACK_PREFIX_MIN_LEN:
            hits += keyword in words
        else:
            hits += any(word.startswith(keyword) for word in words)
    return hits


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

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity using NumPy."""
        if vec1.shape != vec2.shape:
            return 0.0

        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm_v1 * norm_v2))

    def _score_rows(
        self, rows: list, query_vec: np.ndarray, min_score: float
    ) -> List[Dict]:
        """Parse embeddings and score memory rows (called via executor)."""
        scored_memories = []

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

        scored_memories.sort(key=lambda x: x["score"], reverse=True)
        return scored_memories

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
            query_embedding = await self._embedding_engine.async_generate_embedding(query)
        except Exception as e:
            _LOGGER.error("Failed to generate query embedding: %s", e)
            return []

        if not query_embedding:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)

        # Parse embeddings and score memories off the event loop
        scored_memories = await hass.async_add_executor_job(
            self._score_rows, rows, query_vec, min_score
        )
        result = scored_memories[:limit]

        # Text fallback when semantic search returns nothing; reuses the
        # already scope/wing/room-filtered rows
        if not result:
            result = await hass.async_add_executor_job(
                self._text_match_rows, rows, query, limit
            )

        # Update access_count for returned results (batch)
        if result:
            from datetime import datetime
            now = datetime.now().isoformat()
            update_ids = [m["id"] for m in result]
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

    def _text_match_rows(self, rows: list, query: str, limit: int) -> List[Dict]:
        """Fallback text match when semantic search returns nothing (executor).

        Matches query keywords against content and summary. Precision is
        preferred over recall: stopwords are dropped, keywords match whole
        words or word prefixes (not arbitrary substrings), and a memory must
        contain at least half of the keywords.
        """
        keywords = _keywords(query)
        if not keywords:
            return []

        required = math.ceil(len(keywords) / 2)
        matches = []
        for row in rows:
            memory_id, content, _emb, scope, row_agent_id, created_at, \
                summary, mem_wing, mem_room, layer, _access_count = row
            words = set(_words(f"{content or ''} {summary or ''}"))
            hits = _keyword_hits(keywords, words)
            if hits >= required:
                matches.append((hits, created_at or "", {
                    "id": memory_id,
                    "content": content,
                    "score": 0.0,
                    "scope": scope,
                    "agent_id": row_agent_id,
                    "created_at": created_at,
                    "summary": summary,
                    "wing": mem_wing,
                    "room": mem_room,
                    "layer": layer,
                    "match_type": "text",
                }))

        matches.sort(key=lambda m: (m[0], m[1]), reverse=True)
        results = [m[2] for m in matches[:limit]]

        if results:
            _LOGGER.debug("Text fallback found %d results for: %s", len(results), query)

        return results
