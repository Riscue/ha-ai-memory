"""Test AI Memory Init and MemoryManager."""
import unittest
from unittest.mock import patch, mock_open, MagicMock, AsyncMock

from homeassistant.core import HomeAssistant

from custom_components.ai_memory import (
    DOMAIN,
    MemoryManager,
    async_setup_entry,
    async_setup_device_linking,
    _get_device_info_for_agent,
)


async def test_setup_entry_creates_managers(hass: HomeAssistant, mock_config_entry):
    """Test that setup creates at least common memory manager."""
    mock_config_entry.add_to_hass(hass)

    # Mock conversation module to avoid import errors
    with patch("custom_components.ai_memory.MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"), \
            patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):
        assert await async_setup_entry(hass, mock_config_entry)

        # Verify common manager created
        assert "common" in hass.data[DOMAIN]["memory_managers"]


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


async def test_setup_entry_missing_entry_id(hass: HomeAssistant, mock_config_entry):
    """Test setup entry when entry_id is missing."""
    mock_config_entry.add_to_hass(hass)
    # Remove entry_id attribute
    delattr(mock_config_entry, "entry_id")

    with patch("custom_components.ai_memory.MemoryManager"), \
         patch.dict("sys.modules", {"homeassistant.components.conversation": MagicMock()}):

        result = await async_setup_entry(hass, mock_config_entry)
        assert result is False
