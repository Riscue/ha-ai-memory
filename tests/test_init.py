"""Test AI Memory Init and MemoryManager."""
from unittest.mock import patch, MagicMock, mock_open
import unittest
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.const import Platform

from custom_components.ai_memory import (
    DOMAIN,
    MemoryManager,
    async_setup_entry,
    async_unload_entry,
)

async def test_setup_entry_creates_managers(hass: HomeAssistant, mock_config_entry, mock_agent_manager):
    """Test that setup creates common and private memory managers."""
    mock_config_entry.add_to_hass(hass)
    
    # Mock platform forwarding to bypass state check in HA 2025.4+
    with patch("custom_components.ai_memory.MemoryManager") as mock_manager_cls, \
         patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"):
        assert await async_setup_entry(hass, mock_config_entry)
        
        # Verify common manager created
        assert "common" in hass.data[DOMAIN]["memory_managers"]
        
        # Verify private manager created for test agent
        # The agent name is "Test Agent", so ID should be "private_test_agent"
        assert "private_test_agent" in hass.data[DOMAIN]["memory_managers"]

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
        assert manager._memories[0]["text"] == "Entry 2" # Old one removed

    # Test clearing memory
    with patch("os.makedirs"), \
         patch("builtins.open", mock_open()), \
         patch("json.dump") as mock_dump:
        
        await manager.async_clear_memory()
        assert len(manager._memories) == 0
        mock_dump.assert_called_with([], unittest.mock.ANY, ensure_ascii=False, indent=2)

async def test_services(hass: HomeAssistant, mock_config_entry, mock_agent_manager):
    """Test service calls."""
    mock_config_entry.add_to_hass(hass)
    
    # Mock platform forwarding to bypass state check in HA 2025.4+
    with patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"):
        await async_setup_entry(hass, mock_config_entry)
    
    # Mock manager methods
    common_manager = hass.data[DOMAIN]["memory_managers"]["common"]
    common_manager.async_add_memory = MagicMock(side_effect=lambda x: None) # AsyncMock not needed if we don't await the mock directly in test, but the service awaits it.
    # Wait, the service calls await manager.async_add_memory(). So it MUST be awaitable.
    # unittest.mock.AsyncMock is best.
    from unittest.mock import AsyncMock
    common_manager.async_add_memory = AsyncMock()
    common_manager.async_clear_memory = AsyncMock()
    
    # Test add_memory
    await hass.services.async_call(
        DOMAIN, "add_memory", {"memory_id": "common", "text": "Service test"}, blocking=True
    )
    common_manager.async_add_memory.assert_called_with("Service test")
    
    # Test clear_memory
    await hass.services.async_call(
        DOMAIN, "clear_memory", {"memory_id": "common"}, blocking=True
    )
    common_manager.async_clear_memory.assert_called()
    
    # Test list_memories
    response = await hass.services.async_call(
        DOMAIN, "list_memories", return_response=True, blocking=True
    )
    assert "memories" in response
    assert len(response["memories"]) >= 2 # Common + Private
