"""Comprehensive tests for AI Memory Button Platform."""
from unittest.mock import MagicMock, AsyncMock

from homeassistant.core import HomeAssistant

from custom_components.ai_memory import DOMAIN
from custom_components.ai_memory.button import AIMemoryClearButton, async_setup_entry


async def test_button_setup(hass: HomeAssistant, mock_config_entry):
    """Test that buttons are created for memory managers."""
    # Create mock memory managers
    mock_common_manager = MagicMock()
    mock_common_manager.memory_id = "common"
    mock_common_manager.memory_name = "Common Memory"
    mock_common_manager.async_clear_memory = AsyncMock()
    mock_common_manager.device_info = None

    mock_private_manager = MagicMock()
    mock_private_manager.memory_id = "private_test_agent"
    mock_private_manager.memory_name = "Test Agent"
    mock_private_manager.async_clear_memory = AsyncMock()
    mock_private_manager.device_info = {
        "identifiers": {("test", "device")},
        "name": "Test Device"
    }

    hass.data[DOMAIN] = {
        "memory_managers": {
            "common": mock_common_manager,
            "private_test_agent": mock_private_manager
        }
    }

    async_add_entities = MagicMock()
    await async_setup_entry(hass, mock_config_entry, async_add_entities)

    # Verify buttons were created
    assert async_add_entities.called
    buttons = async_add_entities.call_args[0][0]
    assert len(buttons) == 2

    # Check common memory button
    common_button = next((b for b in buttons if b.memory_id == "common"), None)
    assert common_button is not None
    assert isinstance(common_button, AIMemoryClearButton)
    assert common_button.state == "ok"
    assert common_button._attr_name == "Clear Common Memory"

    # Check private memory button
    private_button = next((b for b in buttons if b.memory_id == "private_test_agent"), None)
    assert private_button is not None
    assert isinstance(private_button, AIMemoryClearButton)
    assert private_button.state == "ok"
    assert private_button._attr_name == "Clear Memory"
    assert private_button._attr_device_info == mock_private_manager.device_info


async def test_button_properties(hass: HomeAssistant, mock_config_entry):
    """Test button properties and attributes."""
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_mem"
    mock_manager.memory_name = "Test Memory"
    mock_manager.async_clear_memory = AsyncMock()
    mock_manager.device_info = {"identifiers": {("test", "device")}, "name": "Test Device"}

    button = AIMemoryClearButton(hass, mock_config_entry, mock_manager)

    # Test basic properties
    assert button.memory_id == "test_mem"
    assert button.unique_id == "ai_memory_clear_test_mem"
    assert button.name == "Clear Memory"
    assert button.icon == "mdi:delete-sweep"
    assert button.device_info == mock_manager.device_info


async def test_button_press_success(hass: HomeAssistant, mock_config_entry):
    """Test successful button press."""
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_mem"
    mock_manager.async_clear_memory = AsyncMock()
    mock_manager.memory_name = "Test Memory"

    button = AIMemoryClearButton(hass, mock_config_entry, mock_manager)

    # Press button
    await button.async_press()

    # Verify clear_memory was called
    mock_manager.async_clear_memory.assert_called_once()


async def test_button_press_error(hass: HomeAssistant, mock_config_entry):
    """Test button press when clear_memory fails."""
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_mem"
    mock_manager.async_clear_memory = AsyncMock(side_effect=Exception("Clear failed"))
    mock_manager.memory_name = "Test Memory"

    button = AIMemoryClearButton(hass, mock_config_entry, mock_manager)

    # Press button should not raise exception
    await button.async_press()

    # Verify clear_memory was attempted
    mock_manager.async_clear_memory.assert_called_once()


async def test_button_extra_state_attributes(hass: HomeAssistant, mock_config_entry):
    """Test button extra state attributes."""
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_mem"
    mock_manager.memory_name = "Test Memory"
    mock_manager.description = "Test Description"
    mock_manager.async_clear_memory = AsyncMock()

    button = AIMemoryClearButton(hass, mock_config_entry, mock_manager)

    attributes = button.extra_state_attributes
    assert attributes["memory_id"] == "test_mem"
    assert attributes["memory_name"] == "Test Memory"
    assert attributes["description"] == "Test Description"


async def test_button_no_managers(hass: HomeAssistant, mock_config_entry):
    """Test button setup when no memory managers exist."""
    hass.data[DOMAIN] = {"memory_managers": {}}

    async_add_entities = MagicMock()
    await async_setup_entry(hass, mock_config_entry, async_add_entities)

    # Should not call async_add_entities when no managers exist
    assert not async_add_entities.called


async def test_button_device_linking_private(hass: HomeAssistant, mock_config_entry):
    """Test button device linking for private memory."""
    mock_manager = MagicMock()
    mock_manager.memory_id = "private_agent"
    mock_manager.memory_name = "Private Agent"
    mock_manager.async_clear_memory = AsyncMock()
    mock_manager.device_info = {
        "identifiers": {("private", "device")},
        "name": "Private Device"
    }

    button = AIMemoryClearButton(hass, mock_config_entry, mock_manager)

    # Should have device info
    assert button.device_info == mock_manager.device_info


async def test_button_device_linking_common(hass: HomeAssistant, mock_config_entry):
    """Test button device linking for common memory (no device)."""
    mock_manager = MagicMock()
    mock_manager.memory_id = "common"
    mock_manager.memory_name = "Common Memory"
    mock_manager.async_clear_memory = AsyncMock()
    mock_manager.device_info = None

    button = AIMemoryClearButton(hass, mock_config_entry, mock_manager)

    # Should not have device info
    assert button.device_info is None


async def test_button_common_memory_disabled(hass: HomeAssistant, mock_config_entry):
    """Test button setup when common memory is disabled."""
    # Create only private manager (no common)
    mock_private_manager = MagicMock()
    mock_private_manager.memory_id = "private_test_agent"
    mock_private_manager.memory_name = "Test Agent"
    mock_private_manager.async_clear_memory = AsyncMock()

    hass.data[DOMAIN] = {
        "memory_managers": {
            "private_test_agent": mock_private_manager
        }
    }

    async_add_entities = MagicMock()
    await async_setup_entry(hass, mock_config_entry, async_add_entities)

    # Should create only one button for private memory
    assert async_add_entities.called
    buttons = async_add_entities.call_args[0][0]
    assert len(buttons) == 1
    assert buttons[0].memory_id == "private_test_agent"
