"""Test AI Memory Init and MemoryManager."""
import unittest
from unittest.mock import patch, mock_open, MagicMock

from homeassistant.core import HomeAssistant

from custom_components.ai_memory import (
    DOMAIN,
    MemoryManager,
    async_setup_entry,
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
