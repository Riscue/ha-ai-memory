"""Palace structure management (Wing/Room/Hall/Tunnel)."""
import json
import logging
from typing import Dict, List, Optional

from .defaults import DEFAULT_PALACE, ROOM_KEYWORDS
from ..memory.store import MemoryStore

_LOGGER = logging.getLogger(__name__)


class PalaceStructure:
    """Manages the palace structure: wings, rooms, halls, tunnels."""

    def __init__(self, store: MemoryStore):
        """Initialize palace structure.

        Args:
            store: MemoryStore instance for database access.
        """
        self._store = store

    def validate_or_create_room(self, wing: str, room: str, scope: str = "common") -> tuple:
        """Normalize wing/room to lowercase and auto-create if unknown.

        LLMs may send values in any case or language. We normalize to lowercase
        and auto-create unknown rooms in the palace structure.

        Args:
            wing: Wing name (will be lowercased).
            room: Room name (will be lowercased).
            scope: Default scope if auto-creating.

        Returns:
            Tuple of (normalized_wing, normalized_room).
        """
        wing = wing.lower().strip()
        room = room.lower().strip()

        if not wing or not room:
            return wing, room

        # Check if room exists in palace_structure
        rows = self._store.execute_query(
            "SELECT COUNT(*) FROM palace_structure WHERE wing = ? AND room = ?",
            (wing, room),
        )
        if not rows or rows[0][0] == 0:
            # Auto-create unknown room
            self.add_room(wing, room, scope)
            _LOGGER.info("Auto-created room %s/%s (LLM-provided)", wing, room)

        return wing, room

    def initialize_defaults(self):
        """Insert default wing/room structure if palace_structure table is empty."""
        rows = self._store.execute_query("SELECT COUNT(*) FROM palace_structure")
        if rows and rows[0][0] > 0:
            _LOGGER.debug("Palace structure already initialized (%d rooms)", rows[0][0])
            return

        for wing_data in DEFAULT_PALACE:
            wing = wing_data["wing"]
            scope = wing_data.get("scope", "common")
            for room in wing_data["rooms"]:
                keywords = ROOM_KEYWORDS.get(room, [])
                self._store.execute_commit(
                    """INSERT OR IGNORE INTO palace_structure
                       (wing, room, scope, auto_assign_keywords)
                       VALUES (?, ?, ?, ?)""",
                    (wing, room, scope, json.dumps(keywords)),
                )

        _LOGGER.info("Initialized default palace structure (%d wings)", len(DEFAULT_PALACE))

    def get_structure(self) -> Dict[str, List[Dict]]:
        """Get full palace structure as wing -> rooms mapping.

        Returns:
            Dict mapping wing names to lists of room info dicts.
        """
        rows = self._store.execute_query(
            "SELECT wing, room, scope, auto_assign_keywords FROM palace_structure ORDER BY wing, room"
        )

        structure: Dict[str, List[Dict]] = {}
        for wing, room, scope, keywords_json in rows:
            if wing not in structure:
                structure[wing] = []
            structure[wing].append({
                "room": room,
                "scope": scope,
                "keywords": json.loads(keywords_json) if keywords_json else [],
            })

        return structure

    def add_room(self, wing: str, room: str, scope: str = "common", keywords: Optional[List[str]] = None):
        """Add a custom room to the palace.

        Args:
            wing: Wing name.
            room: Room name.
            scope: Memory scope ('private' or 'common').
            keywords: Optional list of keywords for auto-assignment.
        """
        self._store.execute_commit(
            """INSERT OR IGNORE INTO palace_structure
               (wing, room, scope, auto_assign_keywords)
               VALUES (?, ?, ?, ?)""",
            (wing, room, scope, json.dumps(keywords or [])),
        )
        _LOGGER.info("Added room %s/%s (scope: %s)", wing, room, scope)

    def remove_room(self, wing: str, room: str):
        """Remove a room from the palace.

        Fails if there are memories in this room.

        Args:
            wing: Wing name.
            room: Room name.

        Raises:
            ValueError: If memories exist in this room.
        """
        # Check for existing memories
        rows = self._store.execute_query(
            "SELECT COUNT(*) FROM memories WHERE wing = ? AND room = ?",
            (wing, room),
        )
        if rows and rows[0][0] > 0:
            raise ValueError(f"Cannot remove {wing}/{room}: {rows[0][0]} memories exist")

        self._store.execute_commit(
            "DELETE FROM palace_structure WHERE wing = ? AND room = ?",
            (wing, room),
        )
        _LOGGER.info("Removed room %s/%s", wing, room)

    def get_all_keywords(self) -> Dict[str, Dict]:
        """Get all room keywords for detection.

        Returns:
            Dict mapping keyword -> {wing, room} for quick lookup.
        """
        rows = self._store.execute_query(
            "SELECT wing, room, auto_assign_keywords FROM palace_structure"
        )

        keyword_map: Dict[str, Dict] = {}
        for wing, room, keywords_json in rows:
            keywords = json.loads(keywords_json) if keywords_json else []
            for kw in keywords:
                keyword_map[kw.lower()] = {"wing": wing, "room": room}

        return keyword_map

    def get_stats(self) -> Dict:
        """Get palace structure statistics."""
        rows = self._store.execute_query("SELECT COUNT(DISTINCT wing), COUNT(*) FROM palace_structure")
        wings, rooms = rows[0] if rows else (0, 0)
        return {"wings": wings, "rooms": rooms}
