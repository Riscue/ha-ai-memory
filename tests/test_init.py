"""Tests for AI Memory Init."""
from unittest.mock import patch, MagicMock, AsyncMock

from homeassistant.core import HomeAssistant

from custom_components.ai_memory import (
    async_setup,
    async_setup_entry,
    async_unload_entry,
    async_reload_entry,
    DOMAIN,
)


async def test_async_setup(hass: HomeAssistant):
    """Test async_setup."""
    assert await async_setup(hass, {})
    assert DOMAIN in hass.data


async def test_setup_entry_creates_single_manager(hass: HomeAssistant, mock_config_entry):
    """Test that setup creates a single memory manager."""
    mock_config_entry.add_to_hass(hass)

    # Mock dependencies
    mock_memory_api = MagicMock()
    mock_memory_api.async_setup = AsyncMock()

    import custom_components.ai_memory
    with patch.object(custom_components.ai_memory, "MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"), \
            patch.dict("sys.modules", {
                "custom_components.ai_memory.memory_llm_api": mock_memory_api
            }):
        mock_instance = mock_manager_cls.return_value
        mock_instance.async_load_memories = AsyncMock()

        assert await async_setup_entry(hass, mock_config_entry)

        # Verify single manager created and stored
        assert "manager" in hass.data[DOMAIN]
        assert hass.data[DOMAIN]["manager"] == mock_instance
        mock_instance.async_load_memories.assert_called_once()


async def test_setup_entry_already_initialized(hass: HomeAssistant, mock_config_entry):
    """Test setup when already initialized."""
    mock_config_entry.add_to_hass(hass)
    hass.data[DOMAIN] = {"manager": MagicMock()}

    import custom_components.ai_memory
    with patch.object(custom_components.ai_memory, "MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"):
        assert await async_setup_entry(hass, mock_config_entry)

        # Should not create new manager
        mock_manager_cls.assert_not_called()


async def test_unload_entry(hass: HomeAssistant, mock_config_entry):
    """Test unload entry."""
    mock_config_entry.add_to_hass(hass)
    hass.data[DOMAIN] = {"manager": MagicMock()}

    with patch("homeassistant.config_entries.ConfigEntries.async_unload_platforms", return_value=True):
        assert await async_unload_entry(hass, mock_config_entry)

        # Manager should be removed
        assert "manager" not in hass.data[DOMAIN]


async def test_unload_entry_failure(hass: HomeAssistant, mock_config_entry):
    """Test unload entry failure."""
    mock_config_entry.add_to_hass(hass)
    hass.data[DOMAIN] = {"manager": MagicMock()}

    with patch("homeassistant.config_entries.ConfigEntries.async_unload_platforms", return_value=False):
        assert not await async_unload_entry(hass, mock_config_entry)

        # Manager should still exist
        assert "manager" in hass.data[DOMAIN]


async def test_reload_entry(hass: HomeAssistant, mock_config_entry):
    """Test reload entry."""
    with patch("homeassistant.config_entries.ConfigEntries.async_reload", new_callable=AsyncMock) as mock_reload:
        await async_reload_entry(hass, mock_config_entry)
        mock_reload.assert_called_once_with(mock_config_entry.entry_id)
