"""Tests for Palace structure and Room detection."""
import json

import pytest

from custom_components.ai_memory.memory.store import MemoryStore
from custom_components.ai_memory.memory.migration import MigrationManager
from custom_components.ai_memory.palace.structure import PalaceStructure
from custom_components.ai_memory.palace.metadata import RoomDetector
from custom_components.ai_memory.palace.hall_tunnel import HallTunnelManager


@pytest.fixture
def store():
    """Create in-memory store with migration."""
    s = MemoryStore(":memory:")
    MigrationManager(s).migrate()
    yield s
    s.close()


@pytest.fixture
def palace(store):
    """Create PalaceStructure with defaults initialized."""
    p = PalaceStructure(store)
    p.initialize_defaults()
    return p


# --- PalaceStructure Tests ---

def test_initialize_defaults(palace, store):
    """Test default wings and rooms are created."""
    rows = store.execute_query("SELECT COUNT(*) FROM palace_structure")
    assert rows[0][0] > 0  # Should have several rooms


def test_initialize_defaults_idempotent(palace, store):
    """Test calling initialize_defaults twice doesn't duplicate."""
    palace.initialize_defaults()
    rows1 = store.execute_query("SELECT COUNT(*) FROM palace_structure")
    count1 = rows1[0][0]

    palace.initialize_defaults()
    rows2 = store.execute_query("SELECT COUNT(*) FROM palace_structure")
    count2 = rows2[0][0]

    assert count1 == count2


def test_get_structure(palace):
    """Test getting full palace structure."""
    structure = palace.get_structure()
    assert "household" in structure
    assert "personal" in structure
    assert "automation" in structure
    assert "general" in structure

    # Check a room has required fields
    rooms = structure["household"]
    room_names = [r["room"] for r in rooms]
    assert "devices" in room_names


def test_add_room(palace, store):
    """Test adding a custom room."""
    palace.add_room("household", "pets", "common", ["dog", "cat", "pet"])
    rows = store.execute_query(
        "SELECT scope, auto_assign_keywords FROM palace_structure WHERE wing='household' AND room='pets'"
    )
    assert len(rows) == 1
    assert rows[0][0] == "common"
    assert json.loads(rows[0][1]) == ["dog", "cat", "pet"]


def test_remove_room(palace, store):
    """Test removing a room with no memories."""
    palace.add_room("general", "temp_room", "common")
    palace.remove_room("general", "temp_room")

    rows = store.execute_query(
        "SELECT COUNT(*) FROM palace_structure WHERE wing='general' AND room='temp_room'"
    )
    assert rows[0][0] == 0


def test_remove_room_with_memories_fails(palace, store):
    """Test removing a room that has memories fails."""
    store.execute_commit(
        "INSERT INTO memories (id, content, scope, wing, room, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("test-id", "Test", "common", "household", "devices", "2025-01-01"),
    )

    with pytest.raises(ValueError, match="Cannot remove"):
        palace.remove_room("household", "devices")


def test_get_all_keywords(palace):
    """Test keyword map for room detection."""
    keyword_map = palace.get_all_keywords()
    assert len(keyword_map) > 0
    # 'light' should map to devices room
    assert "light" in keyword_map
    assert keyword_map["light"]["room"] == "devices"


def test_get_stats(palace):
    """Test palace structure stats."""
    stats = palace.get_stats()
    assert stats["wings"] == 4  # household, personal, automation, general
    assert stats["rooms"] > 4


# --- ValidateOrCreate Tests ---

def test_validate_normalizes_case(store, palace):
    """Test that wing/room are lowercased."""
    palace.initialize_defaults()
    wing, room = palace.validate_or_create_room("Household", "Devices", "common")
    assert wing == "household"
    assert room == "devices"


def test_validate_auto_creates_unknown_room(store, palace):
    """Test that unknown rooms are auto-created in palace_structure."""
    palace.initialize_defaults()
    wing, room = palace.validate_or_create_room("household", "mutfak", "common")
    assert wing == "household"
    assert room == "mutfak"

    # Verify it was added to palace_structure
    rows = store.execute_query(
        "SELECT scope FROM palace_structure WHERE wing='household' AND room='mutfak'"
    )
    assert len(rows) == 1
    assert rows[0][0] == "common"


def test_validate_existing_room_no_duplicate(store, palace):
    """Test existing room is not duplicated."""
    palace.initialize_defaults()
    initial = store.execute_query("SELECT COUNT(*) FROM palace_structure")[0][0]

    palace.validate_or_create_room("household", "devices", "common")
    after = store.execute_query("SELECT COUNT(*) FROM palace_structure")[0][0]

    assert initial == after


# --- RoomDetector Tests ---

def test_detect_keyword_match(store, palace):
    """Test room detection via keyword matching."""
    palace.initialize_defaults()
    detector = RoomDetector(store)
    # "light" matches devices, "broken" matches maintenance
    # "light" alone should match devices
    wing, room = detector.detect("The kitchen light is on", "common")
    assert wing == "household"
    assert room == "devices"


def test_detect_keyword_match_maintenance(store, palace):
    """Test room detection for maintenance keywords."""
    palace.initialize_defaults()
    detector = RoomDetector(store)
    wing, room = detector.detect("The kitchen light is broken", "common")
    assert wing == "household"
    assert room == "devices"  # Both "light" and "broken" match, "light" -> devices wins with 1 each


def test_detect_scope_default_private(store, palace):
    """Test scope-based default for private memories."""
    palace.initialize_defaults()
    detector = RoomDetector(store)
    wing, room = detector.detect("Some random text with no keywords", "private")
    assert wing == "personal"
    assert room == "preferences"


def test_detect_scope_default_common(store, palace):
    """Test scope-based default for common memories."""
    palace.initialize_defaults()
    detector = RoomDetector(store)
    wing, room = detector.detect("Some random text with no keywords", "common")
    assert wing == "household"
    assert room == "general"


def test_detect_fallback(store, palace):
    """Test fallback for unknown scope."""
    palace.initialize_defaults()
    detector = RoomDetector(store)
    wing, room = detector.detect("Something", "unknown_scope")
    assert wing == "general"
    assert room == "general"


def test_detect_empty_content(store, palace):
    """Test detection with empty content."""
    palace.initialize_defaults()
    detector = RoomDetector(store)
    wing, room = detector.detect("", "common")
    assert wing == "general"
    assert room == "general"


def test_detect_custom_keywords(store, palace):
    """Test detection with user-added custom keywords."""
    palace.initialize_defaults()
    palace.add_room("household", "pets", "common", ["dog", "cat", "pet"])
    detector = RoomDetector(store)
    detector.refresh_keywords()

    wing, room = detector.detect("My cat is sick", "common")
    assert wing == "household"
    assert room == "pets"


# --- HallTunnel Tests ---

def test_hall_connection(store, palace):
    """Test creating a hall connection between rooms in the same wing."""
    palace.initialize_defaults()
    ht = HallTunnelManager(store)

    ht.set_hall_connection("household", "devices", "maintenance")
    connections = ht.get_connections("household", "devices")
    assert "maintenance" in connections["halls"]

    # Bidirectional
    connections = ht.get_connections("household", "maintenance")
    assert "devices" in connections["halls"]


def test_tunnel_connection(store, palace):
    """Test creating a tunnel connection between wings."""
    palace.initialize_defaults()
    ht = HallTunnelManager(store)

    ht.set_tunnel_connection("household", "devices", "personal", "preferences")
    connections = ht.get_connections("household", "devices")
    assert "personal/preferences" in connections["tunnels"]

    # Bidirectional
    connections = ht.get_connections("personal", "preferences")
    assert "household/devices" in connections["tunnels"]


def test_remove_hall_connection(store, palace):
    """Test removing a hall connection."""
    palace.initialize_defaults()
    ht = HallTunnelManager(store)

    ht.set_hall_connection("household", "devices", "maintenance")
    ht.remove_hall_connection("household", "devices", "maintenance")

    connections = ht.get_connections("household", "devices")
    assert "maintenance" not in connections["halls"]


def test_get_connections_empty(store, palace):
    """Test get connections for room with none."""
    palace.initialize_defaults()
    ht = HallTunnelManager(store)

    connections = ht.get_connections("household", "devices")
    assert connections["halls"] == []
    assert connections["tunnels"] == []


def test_get_connections_nonexistent_room(store):
    """Test get connections for room that doesn't exist."""
    ht = HallTunnelManager(store)
    connections = ht.get_connections("nonexistent", "room")
    assert connections["halls"] == []
    assert connections["tunnels"] == []
