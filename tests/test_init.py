"""Comprehensive tests for AI Memory Init and MemoryManager."""
import unittest
from unittest.mock import patch, mock_open, MagicMock, AsyncMock

from homeassistant.core import HomeAssistant

from custom_components.ai_memory import (
    async_setup,
    async_setup_entry,
    async_reload_entry,
    async_unload_entry,
    async_setup_device_linking,
    _get_device_info_for_agent,
    DOMAIN,
)
from custom_components.ai_memory.memory_manager import MemoryManager


# Basic setup tests
async def test_setup_entry_creates_managers(hass: HomeAssistant, mock_config_entry):
    """Test that setup creates at least common memory manager."""
    mock_config_entry.add_to_hass(hass)

    # Mock conversation module to avoid import errors
    with patch("custom_components.ai_memory.memory_manager.MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"), \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):
        assert await async_setup_entry(hass, mock_config_entry)

        # Verify common manager created
        assert "common" in hass.data[DOMAIN]["memory_managers"]


async def test_setup_entry_missing_entry_id(hass: HomeAssistant, mock_config_entry):
    """Test setup entry when entry_id is missing."""
    mock_config_entry.add_to_hass(hass)
    # Remove entry_id attribute
    delattr(mock_config_entry, "entry_id")

    with patch("custom_components.ai_memory.memory_manager.MemoryManager"), \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):
        result = await async_setup_entry(hass, mock_config_entry)
        assert result is False


# MemoryManager tests
async def test_memory_manager_operations(hass: HomeAssistant):
    """Test MemoryManager operations."""
    manager = MemoryManager(
        hass, "test_mem", "Test Memory", "Desc", "/tmp/test", 10
    )

    # Test adding memory
    with patch("os.makedirs"), \
            patch("builtins.open", mock_open()) as mock_file, \
            patch("json.dump") as mock_dump:
        await manager.async_add_memory("Test entry")

        assert len(manager._memories) == 1
        assert manager._memories[0]["text"] == "Test entry"
        mock_dump.assert_called()

    # Test limit enforcement
    manager.max_entries = 1
    with patch("os.makedirs"), \
            patch("builtins.open", mock_open()), \
            patch("json.dump"):
        await manager.async_add_memory("Entry 2")
        assert len(manager._memories) == 1
        assert manager._memories[0]["text"] == "Entry 2"  # Old one removed

    # Test clearing memory
    with patch("os.makedirs"), \
            patch("builtins.open", mock_open()), \
            patch("json.dump") as mock_dump:
        await manager.async_clear_memory()
        assert len(manager._memories) == 0
        mock_dump.assert_called_with([], unittest.mock.ANY, ensure_ascii=False, indent=2)


async def test_memory_manager_add_empty_memory():
    """Test adding empty memory to manager."""
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock()
    manager = MemoryManager(hass, "test", "Test", "Desc", "/tmp", 10)

    # Mock file operations
    manager._save_to_file = MagicMock()

    # Add empty memory
    await manager.async_add_memory("")

    # Should not save anything
    manager._save_to_file.assert_not_called()
    assert len(manager._memories) == 0


async def test_memory_manager_add_whitespace_only_memory():
    """Test adding whitespace-only memory to manager."""
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock()
    manager = MemoryManager(hass, "test", "Test", "Desc", "/tmp", 10)

    # Mock file operations
    manager._save_to_file = MagicMock()

    # Add whitespace-only memory
    await manager.async_add_memory("   \n\t  ")

    # Should not save anything
    manager._save_to_file.assert_not_called()
    assert len(manager._memories) == 0


async def test_memory_manager_memory_limit_reached():
    """Test behavior when memory limit is reached."""
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock()
    manager = MemoryManager(hass, "test", "Test", "Desc", "/tmp", 2)  # Small limit
    manager._memories = [
        {"date": "2023-01-01 00:00:00", "text": "Old memory"},
        {"date": "2023-01-02 00:00:00", "text": "Another memory"}
    ]

    # Mock event firing
    manager.hass.bus = MagicMock()

    # Add new memory (should remove oldest)
    await manager.async_add_memory("New memory")

    # Should have removed oldest entry
    assert len(manager._memories) == 2
    assert manager._memories[0]["text"] == "Another memory"
    assert manager._memories[1]["text"] == "New memory"


async def test_memory_manager_clear_empty_memory():
    """Test clearing already empty memory."""
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock()
    manager = MemoryManager(hass, "test", "Test", "Desc", "/tmp", 10)
    manager._memories = []  # Already empty

    # Mock event firing
    manager.hass.bus = MagicMock()

    # Clear memory
    await manager.async_clear_memory()

    # Should have been called with empty list
    hass.async_add_executor_job.assert_called()


async def test_memory_manager_with_device_info():
    """Test MemoryManager with device info."""
    hass = MagicMock()
    device_info = {
        "identifiers": {"test_device"},
        "name": "Test Device"
    }

    manager = MemoryManager(
        hass,
        "test_memory",
        "Test Memory",
        "Description",
        "/tmp",
        100,
        device_info=device_info
    )

    assert manager.device_info == device_info
    assert manager.memory_id == "test_memory"
    assert manager.memory_name == "Test Memory"


async def test_memory_manager_without_device_info():
    """Test MemoryManager without device info."""
    hass = MagicMock()

    manager = MemoryManager(
        hass,
        "common",
        "Common Memory",
        "Shared memory",
        "/tmp",
        100
    )

    assert manager.device_info is None


async def test_memory_manager_get_memory_file_path():
    """Test getting memory file path."""
    hass = MagicMock()
    manager = MemoryManager(hass, "test_id", "Test", "Desc", "/config/ai_memory", 100)

    expected_path = "/config/ai_memory/test_id.json"
    assert manager.get_memory_file_path() == expected_path


async def test_memory_manager_load_memories():
    """Test MemoryManager async_load_memories."""
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock()
    manager = MemoryManager(hass, "test_mem", "Test Memory", "Desc", "/tmp", 100)

    # Mock the file reading
    test_memories = [{"date": "2023-01-01", "text": "Test memory"}]
    hass.async_add_executor_job.return_value = test_memories

    await manager.async_load_memories()
    assert len(manager._memories) == 1
    assert manager._memories[0]["text"] == "Test memory"


async def test_memory_manager_read_and_save_methods():
    """Test MemoryManager internal file methods."""
    hass = MagicMock()
    manager = MemoryManager(hass, "test_mem", "Test Memory", "Desc", "/tmp", 100)

    # Test _read_file method
    with patch("custom_components.ai_memory.platform_helpers.read_json_file", return_value=[{"test": "data"}]):
        result = manager._read_file("test_path.json")
        assert result == [{"test": "data"}]

    # Test _save_to_file method
    with patch("custom_components.ai_memory.platform_helpers.write_json_file", return_value=True):
        manager._save_to_file([{"test": "data"}], "test_path.json")


async def test_memory_manager_save_to_file_logging(hass: HomeAssistant):
    """Test MemoryManager _save_to_file logging on success."""
    hass = MagicMock()
    manager = MemoryManager(hass, "test_mem", "Test Memory", "Desc", "/tmp", 100)

    with patch("custom_components.ai_memory.platform_helpers.write_json_file", return_value=True) as mock_write:
        manager._save_to_file([{"test": "data"}], "test_path.json")
        mock_write.assert_called_once()


async def test_memory_manager_event_firing_on_add():
    """Test that events are fired when memory is added."""
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock()
    manager = MemoryManager(hass, "test", "Test", "Desc", "/tmp", 10)

    # Mock event firing
    manager.hass.bus = MagicMock()

    # Add memory
    await manager.async_add_memory("Test memory")

    # Should fire event
    manager.hass.bus.async_fire.assert_called_once_with(
        "ai_memory_updated",
        {"memory_id": "test"}
    )


async def test_memory_manager_event_firing_on_clear():
    """Test that events are fired when memory is cleared."""
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock()
    manager = MemoryManager(hass, "test", "Test", "Desc", "/tmp", 10)
    manager._memories = [{"date": "2023-01-01", "text": "Test"}]

    # Mock event firing
    manager.hass.bus = MagicMock()

    # Clear memory
    await manager.async_clear_memory()

    # Should fire event
    manager.hass.bus.async_fire.assert_called_once_with(
        "ai_memory_updated",
        {"memory_id": "test"}
    )


async def test_memory_manager_no_bus_available():
    """Test MemoryManager when hass.bus is not available."""
    hass = MagicMock()
    hass.async_add_executor_job = AsyncMock()
    del hass.bus  # Remove bus attribute

    manager = MemoryManager(hass, "test", "Test", "Desc", "/tmp", 10)

    # Should not crash when bus is not available
    await manager.async_add_memory("Test memory")
    await manager.async_clear_memory()


# Device info tests
async def test_get_device_info_for_agent(hass: HomeAssistant):
    """Test device info extraction for conversation agents."""
    # Mock entity registry
    mock_entity_reg = MagicMock()
    mock_reg_entry = MagicMock()
    mock_reg_entry.domain = "conversation"
    mock_reg_entry.entity_id = "conversation.test_agent"
    mock_reg_entry.device_id = "test_device_id"

    mock_entity_reg.entities.values.return_value = [mock_reg_entry]

    # Mock device registry
    mock_device_reg = MagicMock()
    mock_device = MagicMock()
    mock_device.identifiers = {("test", "device")}
    mock_device.name = "Test Device"
    mock_device.connections = set()
    mock_device_reg.async_get.return_value = mock_device

    # Mock state
    mock_state = MagicMock()
    mock_state.attributes = {"friendly_name": "Test Agent"}

    with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_entity_reg), \
            patch("homeassistant.helpers.device_registry.async_get", return_value=mock_device_reg), \
            patch.object(hass, "states") as mock_states:
        mock_states.get.return_value = mock_state

        result = _get_device_info_for_agent(hass, "Test Agent")

        assert result is not None
        assert result["name"] == "Test Device"
        assert result["identifiers"] == {("test", "device")}


async def test_get_device_info_for_agent_not_found(hass: HomeAssistant):
    """Test device info extraction when agent not found."""
    # Mock empty entity registry
    mock_entity_reg = MagicMock()
    mock_entity_reg.entities.values.return_value = []

    with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_entity_reg):
        result = _get_device_info_for_agent(hass, "Nonexistent Agent")
        assert result is None


async def test_get_device_info_for_agent_with_connections(hass: HomeAssistant):
    """Test device info extraction with device connections."""
    mock_reg_entry = MagicMock()
    mock_reg_entry.domain = "conversation"
    mock_reg_entry.entity_id = "conversation.test_agent"
    mock_reg_entry.device_id = "test_device_id"

    mock_entity_reg = MagicMock()
    mock_entity_reg.entities.values.return_value = [mock_reg_entry]

    mock_device = MagicMock()
    mock_device.identifiers = {("test", "device")}
    mock_device.name = "Test Device"
    mock_device.connections = {("mac", "aa:bb:cc:dd:ee:ff")}

    mock_device_reg = MagicMock()
    mock_device_reg.async_get.return_value = mock_device

    mock_state = MagicMock()
    mock_state.attributes = {"friendly_name": "Test Agent"}

    with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_entity_reg), \
            patch("homeassistant.helpers.device_registry.async_get", return_value=mock_device_reg), \
            patch.object(hass, "states", MagicMock(get=MagicMock(return_value=mock_state))):
        result = _get_device_info_for_agent(hass, "Test Agent")

        assert result is not None
        assert result["connections"] == {("mac", "aa:bb:cc:dd:ee:ff")}


async def test_get_device_info_for_agent_entity_registry_error(hass: HomeAssistant):
    """Test device info extraction with entity registry error."""
    with patch("homeassistant.helpers.entity_registry.async_get", side_effect=Exception("Registry error")):
        result = _get_device_info_for_agent(hass, "Test Agent")
        assert result is None


async def test_get_device_info_for_agent_device_registry_error(hass: HomeAssistant):
    """Test device info extraction with device registry error."""
    mock_reg_entry = MagicMock()
    mock_reg_entry.domain = "conversation"
    mock_reg_entry.entity_id = "conversation.test_agent"
    mock_reg_entry.device_id = "test_device_id"

    mock_entity_reg = MagicMock()
    mock_entity_reg.entities.values.return_value = [mock_reg_entry]

    with patch("homeassistant.helpers.entity_registry.async_get", return_value=mock_entity_reg), \
            patch("homeassistant.helpers.device_registry.async_get", side_effect=Exception("Device registry error")):
        result = _get_device_info_for_agent(hass, "Test Agent")
        assert result is None


# Device linking tests
async def test_async_setup_device_linking(hass: HomeAssistant):
    """Test late device linking functionality."""
    # Setup mock managers
    common_manager = MemoryManager(hass, "common", "Common", "Desc", "/tmp", 10)
    private_manager = MemoryManager(hass, "private_test", "Private Memory: Test Agent", "Desc", "/tmp", 10)

    hass.data[DOMAIN] = {"memory_managers": {"common": common_manager, "private_test": private_manager}}

    # Mock device info function to return device info for private manager
    mock_device_info = {"identifiers": {("test", "device")}, "name": "Test Device", "connections": set()}

    with patch("custom_components.ai_memory._get_device_info_for_agent", return_value=mock_device_info), \
            patch.object(hass.config_entries, "async_entries") as mock_async_entries, \
            patch.object(hass.config_entries, "async_reload") as mock_reload:
        mock_entry = MagicMock()
        mock_entry.entry_id = "test_entry"
        mock_async_entries.return_value = [mock_entry]

        await async_setup_device_linking(hass)

        # Verify device info was assigned
        assert private_manager.device_info == mock_device_info
        # Verify reload was called
        mock_reload.assert_called_once_with("test_entry")


async def test_async_setup_device_linking_no_managers(hass: HomeAssistant):
    """Test late device linking when no managers exist."""
    hass.data[DOMAIN] = {"memory_managers": {}}

    with patch("custom_components.ai_memory._get_device_info_for_agent") as mock_get_device:
        await async_setup_device_linking(hass)

        # Should not call device info function
        mock_get_device.assert_not_called()


async def test_async_setup_device_linking_reload_error(hass: HomeAssistant):
    """Test device linking when reload fails."""
    common_manager = MemoryManager(hass, "common", "Common", "Desc", "/tmp", 10)
    private_manager = MemoryManager(hass, "private_test", "Private Memory: Test Agent", "Desc", "/tmp", 10)

    hass.data[DOMAIN] = {"memory_managers": {"common": common_manager, "private_test": private_manager}}

    mock_device_info = {"identifiers": {("test", "device")}, "name": "Test Device", "connections": set()}

    with patch("custom_components.ai_memory._get_device_info_for_agent", return_value=mock_device_info), \
            patch.object(hass.config_entries, "async_entries") as mock_async_entries, \
            patch.object(hass.config_entries, "async_reload", side_effect=Exception("Reload error")) as mock_reload:
        mock_entry = MagicMock()
        mock_entry.entry_id = "test_entry"
        mock_async_entries.return_value = [mock_entry]

        await async_setup_device_linking(hass)

        # Should still attempt reload even if it fails
        mock_reload.assert_called_once_with("test_entry")


async def test_async_setup_device_linking_no_private_managers(hass: HomeAssistant):
    """Test device linking when only common manager exists."""
    common_manager = MemoryManager(hass, "common", "Common", "Desc", "/tmp", 10)
    hass.data[DOMAIN] = {"memory_managers": {"common": common_manager}}

    with patch("custom_components.ai_memory._get_device_info_for_agent") as mock_get_device:
        await async_setup_device_linking(hass)
        mock_get_device.assert_not_called()


async def test_async_setup_device_linking_already_has_device_info(hass: HomeAssistant):
    """Test device linking when private manager already has device_info."""
    common_manager = MemoryManager(hass, "common", "Common", "Desc", "/tmp", 10)
    private_manager = MemoryManager(hass, "private_test", "Private Memory: Test Agent", "Desc", "/tmp", 10)

    # Pre-set device info
    existing_device_info = {"identifiers": {("existing", "device")}, "name": "Existing Device", "connections": set()}
    private_manager.device_info = existing_device_info

    hass.data[DOMAIN] = {"memory_managers": {"common": common_manager, "private_test": private_manager}}

    with patch("custom_components.ai_memory._get_device_info_for_agent") as mock_get_device:
        await async_setup_device_linking(hass)

        # Should not call device info for managers that already have device_info
        mock_get_device.assert_not_called()


# Config entry lifecycle tests
async def test_async_setup(hass: HomeAssistant):
    """Test basic async_setup function."""
    result = await async_setup(hass, {})

    assert result is True
    assert DOMAIN in hass.data


async def test_async_reload_entry(hass: HomeAssistant, mock_config_entry):
    """Test reload entry functionality."""
    mock_config_entry.add_to_hass(hass)

    with patch.object(hass.config_entries, "async_reload") as mock_reload:
        await async_reload_entry(hass, mock_config_entry)

        mock_reload.assert_called_once_with(mock_config_entry.entry_id)


async def test_async_unload_entry_success(hass: HomeAssistant, mock_config_entry):
    """Test unload entry success."""
    mock_config_entry.add_to_hass(hass)
    hass.data[DOMAIN] = {"memory_managers": {"test": MagicMock()}}

    with patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_unload",
               return_value=True) as mock_unload, \
            patch(
                "custom_components.ai_memory.extended_openai_helper.async_unregister_from_conversation") as mock_unregister, \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):
        result = await async_unload_entry(hass, mock_config_entry)
        assert result is True

        assert mock_unload.call_count == 3
        mock_unregister.assert_called()


async def test_async_unload_entry_failure(hass: HomeAssistant, mock_config_entry):
    """Test unload entry failure."""
    mock_config_entry.add_to_hass(hass)

    with patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_unload", return_value=False), \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):
        result = await async_unload_entry(hass, mock_config_entry)
        assert result is False


# Service handler tests
async def test_service_add_memory_success(hass: HomeAssistant):
    """Test add_memory service with valid parameters."""
    # Setup
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    mock_manager.async_add_memory = AsyncMock()
    
    hass.data[DOMAIN] = {"memory_managers": {"test_memory": mock_manager}}
    
    # Mock entity state
    mock_state = MagicMock()
    mock_state.attributes = {"memory_id": "test_memory"}
    hass.states.async_set("sensor.test_memory", "state", mock_state.attributes)
    
    # Register services
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service
    await hass.services.async_call(
        DOMAIN,
        "add_memory",
        {"text": "Test memory text", "memory_id": "sensor.test_memory"},
        blocking=True
    )
    
    # Verify
    mock_manager.async_add_memory.assert_called_once_with("Test memory text")


async def test_service_add_memory_no_text(hass: HomeAssistant):
    """Test add_memory service without text parameter."""
    hass.data[DOMAIN] = {"memory_managers": {}}
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service without text
    await hass.services.async_call(
        DOMAIN,
        "add_memory",
        {"memory_id": "sensor.test_memory"},
        blocking=True
    )
    # Should log error and return early


async def test_service_add_memory_no_entity_id(hass: HomeAssistant):
    """Test add_memory service without entity_id parameter."""
    hass.data[DOMAIN] = {"memory_managers": {}}
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service without entity_id
    await hass.services.async_call(
        DOMAIN,
        "add_memory",
        {"text": "Test memory"},
        blocking=True
    )
    # Should log error and return early


async def test_service_add_memory_entity_not_found(hass: HomeAssistant):
    """Test add_memory service when entity doesn't exist."""
    hass.data[DOMAIN] = {"memory_managers": {}}
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service with non-existent entity
    await hass.services.async_call(
        DOMAIN,
        "add_memory",
        {"text": "Test memory", "memory_id": "sensor.nonexistent"},
        blocking=True
    )
    # Should log error and return early


async def test_service_add_memory_no_memory_id_attribute(hass: HomeAssistant):
    """Test add_memory service when entity has no memory_id attribute."""
    hass.data[DOMAIN] = {"memory_managers": {}}
    
    # Mock entity state without memory_id attribute
    hass.states.async_set("sensor.test_memory", "state", {})
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service
    await hass.services.async_call(
        DOMAIN,
        "add_memory",
        {"text": "Test memory", "memory_id": "sensor.test_memory"},
        blocking=True
    )
    # Should log error and return early


async def test_service_add_memory_manager_not_found(hass: HomeAssistant):
    """Test add_memory service when memory manager doesn't exist."""
    mock_manager = MagicMock()
    mock_manager.memory_id = "other_memory"
    
    hass.data[DOMAIN] = {"memory_managers": {"other_memory": mock_manager}}
    
    # Mock entity state with different memory_id
    mock_state = MagicMock()
    mock_state.attributes = {"memory_id": "nonexistent_memory"}
    hass.states.async_set("sensor.test_memory", "state", mock_state.attributes)
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service
    await hass.services.async_call(
        DOMAIN,
        "add_memory",
        {"text": "Test memory", "memory_id": "sensor.test_memory"},
        blocking=True
    )
    # Should log error about manager not found


async def test_service_clear_memory_success(hass: HomeAssistant):
    """Test clear_memory service with valid parameters."""
    # Setup
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    mock_manager.async_clear_memory = AsyncMock()
    
    hass.data[DOMAIN] = {"memory_managers": {"test_memory": mock_manager}}
    
    # Mock entity state
    mock_state = MagicMock()
    mock_state.attributes = {"memory_id": "test_memory"}
    hass.states.async_set("sensor.test_memory", "state", mock_state.attributes)
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service
    await hass.services.async_call(
        DOMAIN,
        "clear_memory",
        {"memory_id": "sensor.test_memory"},
        blocking=True
    )
    
    # Verify
    mock_manager.async_clear_memory.assert_called_once()


async def test_service_clear_memory_no_entity_id(hass: HomeAssistant):
    """Test clear_memory service without entity_id parameter."""
    hass.data[DOMAIN] = {"memory_managers": {}}
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service without entity_id
    await hass.services.async_call(
        DOMAIN,
        "clear_memory",
        {},
        blocking=True
    )
    # Should log error and return early


async def test_service_clear_memory_entity_not_found(hass: HomeAssistant):
    """Test clear_memory service when entity doesn't exist."""
    hass.data[DOMAIN] = {"memory_managers": {}}
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service with non-existent entity
    await hass.services.async_call(
        DOMAIN,
        "clear_memory",
        {"memory_id": "sensor.nonexistent"},
        blocking=True
    )
    # Should log error and return early


async def test_service_clear_memory_no_memory_id_attribute(hass: HomeAssistant):
    """Test clear_memory service when entity has no memory_id attribute."""
    hass.data[DOMAIN] = {"memory_managers": {}}
    
    # Mock entity state without memory_id attribute
    hass.states.async_set("sensor.test_memory", "state", {})
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service
    await hass.services.async_call(
        DOMAIN,
        "clear_memory",
        {"memory_id": "sensor.test_memory"},
        blocking=True
    )
    # Should log error and return early


async def test_service_clear_memory_manager_not_found(hass: HomeAssistant):
    """Test clear_memory service when memory manager doesn't exist."""
    mock_manager = MagicMock()
    mock_manager.memory_id = "other_memory"
    
    hass.data[DOMAIN] = {"memory_managers": {"other_memory": mock_manager}}
    
    # Mock entity state with different memory_id
    mock_state = MagicMock()
    mock_state.attributes = {"memory_id": "nonexistent_memory"}
    hass.states.async_set("sensor.test_memory", "state", mock_state.attributes)
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service
    await hass.services.async_call(
        DOMAIN,
        "clear_memory",
        {"memory_id": "sensor.test_memory"},
        blocking=True
    )
    # Should log error about manager not found


async def test_service_list_memories_success(hass: HomeAssistant):
    """Test list_memories service."""
    # Setup managers
    mock_manager1 = MagicMock()
    mock_manager1.memory_id = "common"
    mock_manager1.memory_name = "Common Memory"
    mock_manager1.description = "Shared memory"
    mock_manager1.max_entries = 1000
    mock_manager1.storage_location = "/tmp"
    mock_manager1._memories = [{"date": "2023-01-01", "text": "Test"}]
    
    mock_manager2 = MagicMock()
    mock_manager2.memory_id = "private_test"
    mock_manager2.memory_name = "Private Memory"
    mock_manager2.description = "Private memory"
    mock_manager2.max_entries = 500
    mock_manager2.storage_location = "/tmp"
    mock_manager2._memories = []
    
    hass.data[DOMAIN] = {
        "memory_managers": {
            "common": mock_manager1,
            "private_test": mock_manager2
        }
    }
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service
    response = await hass.services.async_call(
        DOMAIN,
        "list_memories",
        {},
        blocking=True,
        return_response=True
    )
    
    # Verify
    assert "memories" in response
    assert len(response["memories"]) == 2
    assert response["memories"][0]["memory_id"] == "common"
    assert response["memories"][0]["entry_count"] == 1
    assert response["memories"][1]["memory_id"] == "private_test"
    assert response["memories"][1]["entry_count"] == 0


async def test_service_list_memories_no_managers(hass: HomeAssistant):
    """Test list_memories service with no managers."""
    hass.data[DOMAIN] = {"memory_managers": {}}
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service
    response = await hass.services.async_call(
        DOMAIN,
        "list_memories",
        {},
        blocking=True,
        return_response=True
    )
    
    # Verify
    assert "memories" in response
    assert len(response["memories"]) == 0


async def test_service_get_context_specific_memory(hass: HomeAssistant):
    """Test get_context service for specific memory."""
    # Setup manager
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    mock_manager._memories = [{"date": "2023-01-01", "text": "Test memory"}]
    
    hass.data[DOMAIN] = {"memory_managers": {"test_memory": mock_manager}}
    
    # Mock entity state
    mock_state = MagicMock()
    mock_state.attributes = {"memory_id": "test_memory"}
    hass.states.async_set("sensor.test_memory", "state", mock_state.attributes)
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Mock get_memory_context_for_llm
    with patch("custom_components.ai_memory.extended_openai_helper.get_memory_context_for_llm",
               return_value="Memory context"):
        response = await hass.services.async_call(
            DOMAIN,
            "get_context",
            {"memory_id": "sensor.test_memory"},
            blocking=True,
            return_response=True
        )
    
    # Verify
    assert "memory_id" in response
    assert response["memory_id"] == "test_memory"
    assert "context" in response


async def test_service_get_context_all_memories(hass: HomeAssistant):
    """Test get_context service for all memories."""
    # Setup managers
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    
    hass.data[DOMAIN] = {"memory_managers": {"test_memory": mock_manager}}
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Mock get_all_memory_contexts
    with patch("custom_components.ai_memory.extended_openai_helper.get_all_memory_contexts",
               return_value="All contexts"):
        response = await hass.services.async_call(
            DOMAIN,
            "get_context",
            {},
            blocking=True,
            return_response=True
        )
    
    # Verify
    assert "context" in response
    assert "memory_count" in response
    assert response["memory_count"] == 1


async def test_service_get_context_entity_not_found(hass: HomeAssistant):
    """Test get_context service when entity doesn't exist."""
    hass.data[DOMAIN] = {"memory_managers": {}}
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service with non-existent entity
    response = await hass.services.async_call(
        DOMAIN,
        "get_context",
        {"memory_id": "sensor.nonexistent"},
        blocking=True,
        return_response=True
    )
    
    # Verify error response
    assert "error" in response
    assert "available" in response


async def test_service_get_context_no_memory_id_attribute(hass: HomeAssistant):
    """Test get_context service when entity has no memory_id attribute."""
    hass.data[DOMAIN] = {"memory_managers": {}}
    
    # Mock entity state without memory_id attribute
    hass.states.async_set("sensor.test_memory", "state", {})
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service
    response = await hass.services.async_call(
        DOMAIN,
        "get_context",
        {"memory_id": "sensor.test_memory"},
        blocking=True,
        return_response=True
    )
    
    # Verify error response
    assert "error" in response
    assert "available" in response


async def test_service_get_context_memory_not_found(hass: HomeAssistant):
    """Test get_context service when memory manager doesn't exist."""
    mock_manager = MagicMock()
    mock_manager.memory_id = "other_memory"
    
    hass.data[DOMAIN] = {"memory_managers": {"other_memory": mock_manager}}
    
    # Mock entity state with different memory_id
    mock_state = MagicMock()
    mock_state.attributes = {"memory_id": "nonexistent_memory"}
    hass.states.async_set("sensor.test_memory", "state", mock_state.attributes)
    
    from custom_components.ai_memory import _register_services
    _register_services(hass)
    
    # Call service
    response = await hass.services.async_call(
        DOMAIN,
        "get_context",
        {"memory_id": "sensor.test_memory"},
        blocking=True,
        return_response=True
    )
    
    # Verify error response
    assert "error" in response
    assert "available" in response


# Agent discovery tests
async def test_setup_entry_with_conversation_entities(hass: HomeAssistant, mock_config_entry):
    """Test setup with conversation entities in hass.data."""
    mock_config_entry.add_to_hass(hass)
    
    # Mock conversation entity
    mock_entity = MagicMock()
    mock_entity.name = "Test Agent"
    mock_entity.entity_id = "conversation.test_agent"
    
    mock_conversation_data = MagicMock()
    mock_conversation_data.entities = [mock_entity]
    
    with patch("custom_components.ai_memory.memory_manager.MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"), \
            patch("homeassistant.helpers.entity_registry.async_get") as mock_er, \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}), \
            patch("homeassistant.components.conversation.DOMAIN", "conversation"):
        # Setup hass.data with conversation using the mocked DOMAIN
        hass.data["conversation"] = mock_conversation_data
        
        # Mock empty entity registry
        mock_er.return_value.entities.values.return_value = []
        
        result = await async_setup_entry(hass, mock_config_entry)
        assert result is True
        
        # Verify private manager was created for discovered agent
        assert "private_test_agent" in hass.data[DOMAIN]["memory_managers"]


async def test_setup_entry_entity_with_no_name(hass: HomeAssistant, mock_config_entry):
    """Test setup skips entities with no name."""
    mock_config_entry.add_to_hass(hass)
    
    # Mock conversation entity without name
    mock_entity = MagicMock()
    mock_entity.name = None
    mock_entity.entity_id = "conversation.test_agent"
    
    mock_conversation_data = MagicMock()
    mock_conversation_data.entities = [mock_entity]
    
    hass.data["conversation"] = mock_conversation_data
    
    with patch("custom_components.ai_memory.memory_manager.MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"), \
            patch("homeassistant.helpers.entity_registry.async_get") as mock_er, \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):
        mock_er.return_value.entities.values.return_value = []
        
        result = await async_setup_entry(hass, mock_config_entry)
        assert result is True
        
        # Should only have common manager, no private manager for unnamed entity
        assert "common" in hass.data[DOMAIN]["memory_managers"]


async def test_setup_entry_conversation_error(hass: HomeAssistant, mock_config_entry):
    """Test setup handles conversation.DOMAIN access errors."""
    mock_config_entry.add_to_hass(hass)
    
    # Mock conversation data that raises exception
    mock_conversation_data = MagicMock()
    mock_conversation_data.entities = MagicMock(side_effect=Exception("Access error"))
    
    hass.data["conversation"] = mock_conversation_data
    
    with patch("custom_components.ai_memory.memory_manager.MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"), \
            patch("homeassistant.helpers.entity_registry.async_get") as mock_er, \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):
        mock_er.return_value.entities.values.return_value = []
        
        result = await async_setup_entry(hass, mock_config_entry)
        assert result is True


async def test_setup_entry_no_conversation_domain(hass: HomeAssistant, mock_config_entry):
    """Test setup when conversation.DOMAIN not in hass.data."""
    mock_config_entry.add_to_hass(hass)
    
    # Don't add conversation to hass.data
    
    with patch("custom_components.ai_memory.memory_manager.MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"), \
            patch("homeassistant.helpers.entity_registry.async_get") as mock_er, \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):
        mock_er.return_value.entities.values.return_value = []
        
        result = await async_setup_entry(hass, mock_config_entry)
        assert result is True


async def test_setup_entry_entity_registry_fallback(hass: HomeAssistant, mock_config_entry):
    """Test setup uses entity registry fallback."""
    mock_config_entry.add_to_hass(hass)
    
    # Mock entity registry entry
    mock_reg_entry = MagicMock()
    mock_reg_entry.domain = "conversation"
    mock_reg_entry.entity_id = "conversation.registry_agent"
    mock_reg_entry.original_name = "Registry Agent"
    mock_reg_entry.platform = None
    mock_reg_entry.device_id = None
    
    # Mock state
    mock_state = MagicMock()
    mock_state.attributes = {"friendly_name": "Registry Agent"}
    
    with patch("custom_components.ai_memory.memory_manager.MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"), \
            patch("homeassistant.helpers.entity_registry.async_get") as mock_er, \
            patch.object(hass, "states") as mock_states, \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):
        mock_er.return_value.entities.values.return_value = [mock_reg_entry]
        mock_states.get.return_value = mock_state
        
        result = await async_setup_entry(hass, mock_config_entry)
        assert result is True
        
        # Verify private manager created from registry
        assert "private_registry_agent" in hass.data[DOMAIN]["memory_managers"]


async def test_setup_entry_name_from_platform(hass: HomeAssistant, mock_config_entry):
    """Test setup generates name from platform when no friendly_name."""
    mock_config_entry.add_to_hass(hass)
    
    # Mock entity registry entry
    mock_reg_entry = MagicMock()
    mock_reg_entry.domain = "conversation"
    mock_reg_entry.entity_id = "conversation.test_agent"
    mock_reg_entry.original_name = None
    mock_reg_entry.platform = "google_generative_ai_conversation"
    mock_reg_entry.device_id = None
    
    with patch("custom_components.ai_memory.memory_manager.MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"), \
            patch("homeassistant.helpers.entity_registry.async_get") as mock_er, \
            patch.object(hass, "states") as mock_states, \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):
        mock_er.return_value.entities.values.return_value = [mock_reg_entry]
        mock_states.get.return_value = None
        
        result = await async_setup_entry(hass, mock_config_entry)
        assert result is True
        
        # Name should be generated from platform: "Google Generative Ai"
        assert "private_google_generative_ai" in hass.data[DOMAIN]["memory_managers"]


async def test_setup_entry_duplicate_agent_handling(hass: HomeAssistant, mock_config_entry):
    """Test setup handles duplicate agents correctly."""
    mock_config_entry.add_to_hass(hass)
    
    # Mock conversation entity
    mock_entity = MagicMock()
    mock_entity.name = "Test Agent"
    mock_entity.entity_id = "conversation.test_agent"
    
    mock_conversation_data = MagicMock()
    mock_conversation_data.entities = [mock_entity]
    
    hass.data["conversation"] = mock_conversation_data
    
    # Mock entity registry with same agent
    mock_reg_entry = MagicMock()
    mock_reg_entry.domain = "conversation"
    mock_reg_entry.entity_id = "conversation.test_agent"  # Same entity_id
    mock_reg_entry.original_name = "Test Agent"
    
    with patch("custom_components.ai_memory.memory_manager.MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"), \
            patch("homeassistant.helpers.entity_registry.async_get") as mock_er, \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):
        mock_er.return_value.entities.values.return_value = [mock_reg_entry]
        
        result = await async_setup_entry(hass, mock_config_entry)
        assert result is True
        
        # Should only create one private manager (no duplicates)
        private_managers = [k for k in hass.data[DOMAIN]["memory_managers"].keys() if k.startswith("private_")]
        assert len(private_managers) == 1


async def test_setup_entry_wrong_domain(hass: HomeAssistant):
    """Test setup fails when entry has wrong domain."""
    # Create a mock config entry with wrong domain
    mock_entry = MagicMock()
    mock_entry.entry_id = "test_entry"
    mock_entry.domain = "wrong_domain"
    mock_entry.data = {}
    
    result = await async_setup_entry(hass, mock_entry)
    assert result is False


async def test_setup_entry_already_initialized(hass: HomeAssistant, mock_config_entry):
    """Test setup when memory managers already exist."""
    mock_config_entry.add_to_hass(hass)
    
    # Pre-populate memory managers
    existing_manager = MagicMock()
    hass.data[DOMAIN] = {
        "entries": {},
        "memory_managers": {"existing": existing_manager}
    }
    
    with patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups") as mock_forward:
        result = await async_setup_entry(hass, mock_config_entry)
        assert result is True
        
        # Should skip manager creation but still forward to platforms
        mock_forward.assert_called_once()

