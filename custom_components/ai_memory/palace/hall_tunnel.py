"""Hall and Tunnel connections for Palace structure.

Hall: connections between rooms within the same wing.
Tunnel: connections between rooms across different wings.

Phase 1 stub: manual configuration only.
Automatic discovery planned for Phase 5.
"""
import json
import logging
from typing import Dict, List, Optional

from ..memory.store import MemoryStore

_LOGGER = logging.getLogger(__name__)


class HallTunnelManager:
    """Manages hall and tunnel connections in the palace.

    Halls connect rooms within the same wing.
    Tunnels connect rooms across different wings.
    """

    def __init__(self, store: MemoryStore):
        """Initialize hall/tunnel manager.

        Args:
            store: MemoryStore instance for database access.
        """
        self._store = store

    def set_hall_connection(self, wing: str, room_a: str, room_b: str):
        """Create a hall connection between two rooms in the same wing.

        Args:
            wing: Wing name (both rooms must be in this wing).
            room_a: First room name.
            room_b: Second room name.
        """
        self._update_connections(wing, room_a, "hall_connections", room_b)
        self._update_connections(wing, room_b, "hall_connections", room_a)
        _LOGGER.info("Hall connection: %s/%s <-> %s/%s", wing, room_a, wing, room_b)

    def set_tunnel_connection(
        self, wing_a: str, room_a: str, wing_b: str, room_b: str
    ):
        """Create a tunnel connection between rooms in different wings.

        Args:
            wing_a: First wing name.
            room_a: Room in first wing.
            wing_b: Second wing name.
            room_b: Room in second wing.
        """
        self._update_connections(wing_a, room_a, "tunnel_connections", f"{wing_b}/{room_b}")
        self._update_connections(wing_b, room_b, "tunnel_connections", f"{wing_a}/{room_a}")
        _LOGGER.info("Tunnel connection: %s/%s <-> %s/%s", wing_a, room_a, wing_b, room_b)

    def remove_hall_connection(self, wing: str, room_a: str, room_b: str):
        """Remove a hall connection between two rooms."""
        self._remove_connection(wing, room_a, "hall_connections", room_b)
        self._remove_connection(wing, room_b, "hall_connections", room_a)
        _LOGGER.info("Removed hall: %s/%s <-> %s/%s", wing, room_a, wing, room_b)

    def remove_tunnel_connection(
        self, wing_a: str, room_a: str, wing_b: str, room_b: str
    ):
        """Remove a tunnel connection between rooms in different wings."""
        self._remove_connection(wing_a, room_a, "tunnel_connections", f"{wing_b}/{room_b}")
        self._remove_connection(wing_b, room_b, "tunnel_connections", f"{wing_a}/{room_a}")
        _LOGGER.info("Removed tunnel: %s/%s <-> %s/%s", wing_a, room_a, wing_b, room_b)

    def get_connections(self, wing: str, room: str) -> Dict[str, List[str]]:
        """Get all connections for a room.

        Args:
            wing: Wing name.
            room: Room name.

        Returns:
            Dict with 'halls' and 'tunnels' lists.
        """
        rows = self._store.execute_query(
            "SELECT hall_connections, tunnel_connections FROM palace_structure WHERE wing = ? AND room = ?",
            (wing, room),
        )
        if not rows:
            return {"halls": [], "tunnels": []}

        halls = json.loads(rows[0][0]) if rows[0][0] else []
        tunnels = json.loads(rows[0][1]) if rows[0][1] else []
        return {"halls": halls, "tunnels": tunnels}

    def _update_connections(self, wing: str, room: str, column: str, target: str):
        """Add a connection to a room's connection list."""
        rows = self._store.execute_query(
            f"SELECT {column} FROM palace_structure WHERE wing = ? AND room = ?",
            (wing, room),
        )
        if not rows:
            _LOGGER.warning("Room %s/%s not found in palace structure", wing, room)
            return

        connections = json.loads(rows[0][0]) if rows[0][0] else []
        if target not in connections:
            connections.append(target)
            self._store.execute_commit(
                f"UPDATE palace_structure SET {column} = ? WHERE wing = ? AND room = ?",
                (json.dumps(connections), wing, room),
            )

    def _remove_connection(self, wing: str, room: str, column: str, target: str):
        """Remove a connection from a room's connection list."""
        rows = self._store.execute_query(
            f"SELECT {column} FROM palace_structure WHERE wing = ? AND room = ?",
            (wing, room),
        )
        if not rows:
            return

        connections = json.loads(rows[0][0]) if rows[0][0] else []
        if target in connections:
            connections.remove(target)
            self._store.execute_commit(
                f"UPDATE palace_structure SET {column} = ? WHERE wing = ? AND room = ?",
                (json.dumps(connections), wing, room),
            )
