"""Test AI Memory Button Platform."""
from unittest.mock import patch, MagicMock, AsyncMock

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
    mock_private_manager.memory_name = "Private: Test Agent"
    mock_private_manager.async_clear_memory = AsyncMock()
    mock_private_manager.device_info = {"identifiers": {"test_device"}}

    # Mock hass.data with memory managers
    hass.data[DOMAIN] = {
        "memory_managers": {
            "common": mock_common_manager,
            "private_test_agent": mock_private_manager,
        }
    }

    # Mock async_add_entities
    async_add_entities = MagicMock()

    # Call setup entry
    await async_setup_entry(hass, mock_config_entry, async_add_entities)

    # Verify entities were added
    assert async_add_entities.called
    buttons = async_add_entities.call_args[0][0]
    assert len(buttons) == 2  # One clear button for each manager

    # Check button types
    clear_buttons = [b for b in buttons if isinstance(b, AIMemoryClearButton)]
    assert len(clear_buttons) == 2

    # Verify button names
    button_names = [b.name for b in clear_buttons]
    assert "Clear Common Memory" in button_names
    assert "Clear Memory" in button_names


async def test_clear_button_press(hass: HomeAssistant, mock_config_entry):
    """Test clear button press functionality."""
    # Create mock memory manager
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    mock_manager.memory_name = "Test Memory"
    mock_manager.async_clear_memory = AsyncMock()

    # Create clear button
    button = AIMemoryClearButton(hass, mock_config_entry, mock_manager)

    # Press the button
    await button.async_press()

    # Verify clear memory was called
    mock_manager.async_clear_memory.assert_called_once()


async def test_clear_button_press_error(hass: HomeAssistant, mock_config_entry):
    """Test clear button press with error."""
    # Create mock memory manager with error
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    mock_manager.memory_name = "Test Memory"
    mock_manager.async_clear_memory = AsyncMock(side_effect=Exception("Test error"))

    # Create clear button
    button = AIMemoryClearButton(hass, mock_config_entry, mock_manager)

    # Press the button (should not raise exception)
    await button.async_press()

    # Verify clear memory was called
    mock_manager.async_clear_memory.assert_called_once()


async def test_clear_button_properties():
    """Test clear button properties."""
    # Create mock objects
    hass = MagicMock()
    entry = MagicMock()
    entry.entry_id = "test_entry"
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    mock_manager.memory_name = "Test Memory"

    # Mock device registry for device linking
    with patch('homeassistant.helpers.entity_registry.async_get') as mock_entity_reg, \
            patch('homeassistant.helpers.device_registry.async_get') as mock_device_reg:

        # Setup mock responses for private memory
        mock_entity_reg.return_value.async_get.return_value = None  # No device found

        # Create clear button
        button = AIMemoryClearButton(hass, entry, mock_manager)

        # Test properties - based on device_info presence
        if hasattr(mock_manager, 'device_info') and mock_manager.device_info:
            assert button.name == "Clear Memory"
        else:
            assert button.name == "Clear Test Memory"
        assert button.unique_id == "ai_memory_clear_test_memory"
        assert button.icon == "mdi:delete-sweep"


async def test_clear_button_device_linking_private():
    """Test clear button device linking for private memory."""
    # Create mock objects
    hass = MagicMock()
    entry = MagicMock()
    mock_manager = MagicMock()
    mock_manager.memory_id = "private_google_generative_ai"
    mock_manager.memory_name = "Private: Google Generative AI"

    # Mock device registry for private memory device linking
    with patch('homeassistant.helpers.entity_registry.async_get') as mock_entity_reg, \
            patch('homeassistant.helpers.device_registry.async_get') as mock_device_reg:
        # Setup mock device
        mock_device = MagicMock()
        mock_device.identifiers = {"test_device"}
        mock_device.name = "Test Device"
        mock_device_reg.return_value.async_get.return_value = mock_device

        # Setup mock entity with device
        mock_agent_entry = MagicMock()
        mock_agent_entry.device_id = "test_device_id"
        mock_entity_reg.return_value.async_get.return_value = mock_agent_entry

        # Mock device info on manager
        mock_manager.device_info = {
            "identifiers": {"test_device"},
            "name": "Test Device"
        }

        # Create clear button
        button = AIMemoryClearButton(hass, entry, mock_manager)

        # Check device info was set from manager
        assert button._attr_device_info == mock_manager.device_info

        # Check device-linked memory has short name
        assert button.name == "Clear Memory"
        # All buttons should be disabled by default
        assert button._attr_entity_registry_enabled_default == False


async def test_clear_button_common_memory_disabled():
    """Test clear button for common memory is disabled by default."""
    # Create mock objects
    hass = MagicMock()
    entry = MagicMock()
    mock_manager = MagicMock()
    mock_manager.memory_id = "common"
    mock_manager.memory_name = "Common Memory"
    mock_manager.device_info = None  # Explicitly set to None

    # Create clear button
    button = AIMemoryClearButton(hass, entry, mock_manager)

    # Check common memory has full name
    assert button.name == "Clear Common Memory"
    # Common memory should be disabled by default (no device_info)
    assert button._attr_entity_registry_enabled_default == False
