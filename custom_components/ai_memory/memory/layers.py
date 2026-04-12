"""Memory layer management (L0-L3).

L0: Identity — Config-defined core knowledge about the user (not stored in DB).
L1: Critical — Frequently accessed, high-priority memories (layer=1 in DB).
L2: Standard — Normal memories (layer=2 in DB, default).
L3: Archive — Old, rarely accessed memories (layer=3 in DB).

Phase 1 stub: context retrieval only.
Full promotion/demotion logic planned for Phase 3.
"""
import logging
from typing import Dict, List, Optional

from homeassistant.core import HomeAssistant

from .store import MemoryStore

_LOGGER = logging.getLogger(__name__)


class LayerManager:
    """Manages the 4-layer memory stack."""

    def __init__(self, store: MemoryStore, config_data: dict = None):
        """Initialize layer manager.

        Args:
            store: MemoryStore instance for database access.
            config_data: Config entry data containing identity_text (L0).
        """
        self._store = store
        self._config_data = config_data or {}

    async def async_get_context(self, hass: HomeAssistant, agent_id: Optional[str] = None) -> Dict:
        """Get memory context for an agent: L0 identity + L1 critical memories.

        Args:
            hass: Home Assistant instance for executor jobs.
            agent_id: Optional agent ID for filtering private L1 memories.

        Returns:
            Dict with 'identity' (L0 text) and 'critical_memories' (L1 list).
        """
        context = {
            "identity": self._config_data.get("identity_text", ""),
            "critical_memories": [],
        }

        # Retrieve L1 (layer=1) memories
        try:
            rows = await hass.async_add_executor_job(
                self._store.execute_query,
                """SELECT content, scope, wing, room, summary
                   FROM memories WHERE layer = 1
                   AND (scope = 'common' OR agent_id = ? OR agent_id IS NULL)
                   ORDER BY access_count DESC""",
                (agent_id,),
            )
            context["critical_memories"] = [
                {
                    "content": row[0],
                    "scope": row[1],
                    "wing": row[2],
                    "room": row[3],
                    "summary": row[4],
                }
                for row in rows
            ]
        except Exception as e:
            _LOGGER.error("Failed to get L1 context: %s", e)

        return context
