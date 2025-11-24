"""Test AI Memory Text Platform."""
from unittest.mock import patch, MagicMock, AsyncMock

from homeassistant.core import HomeAssistant

from custom_components.ai_memory import DOMAIN
from custom_components.ai_memory.text import AIMemoryTextInput, async_setup_entry


async def test_text_setup(hass: HomeAssistant, mock_config_entry):
    """Test that text inputs are created for memory managers."""
    # Create mock memory managers
    mock_common_manager = MagicMock()
    mock_common_manager.memory_id = "common"
    mock_common_manager.memory_name = "Common Memory"
    mock_common_manager.async_add_memory = AsyncMock()
    mock_common_manager.device_info = None

    mock_private_manager = MagicMock()
    mock_private_manager.memory_id = "private_test_agent"
    mock_private_manager.memory_name = "Private: Test Agent"
    mock_private_manager.async_add_memory = AsyncMock()
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
    text_inputs = async_add_entities.call_args[0][0]
    assert len(text_inputs) == 2  # One text input for each manager

    # Check text input types
    ai_memory_texts = [t for t in text_inputs if isinstance(t, AIMemoryTextInput)]
    assert len(ai_memory_texts) == 2

    # Verify text input names
    text_names = [t.name for t in ai_memory_texts]
    assert "Add to Common Memory" in text_names
    assert "Add Memory" in text_names


async def test_text_set_value_valid(hass: HomeAssistant, mock_config_entry):
    """Test text input with valid value."""
    # Create mock memory manager
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    mock_manager.memory_name = "Test Memory"
    mock_manager.async_add_memory = AsyncMock()

    # Create text input
    text_input = AIMemoryTextInput(hass, mock_config_entry, mock_manager)

    # Set a valid value
    test_text = "This is a test memory entry"
    await text_input.async_set_value(test_text)

    # Verify add_memory was called
    mock_manager.async_add_memory.assert_called_once_with(test_text)

    # Verify the text input was cleared after successful submission
    assert text_input.native_value == ""


async def test_text_set_value_empty(hass: HomeAssistant, mock_config_entry):
    """Test text input with empty value."""
    # Create mock memory manager
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    mock_manager.memory_name = "Test Memory"
    mock_manager.async_add_memory = AsyncMock()

    # Create text input
    text_input = AIMemoryTextInput(hass, mock_config_entry, mock_manager)

    # Set an empty value
    await text_input.async_set_value("")
    await text_input.async_set_value("   ")

    # Verify add_memory was not called
    mock_manager.async_add_memory.assert_not_called()


async def test_text_set_value_whitespace_only(hass: HomeAssistant, mock_config_entry):
    """Test text input with whitespace-only value."""
    # Create mock memory manager
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    mock_manager.memory_name = "Test Memory"
    mock_manager.async_add_memory = AsyncMock()

    # Create text input
    text_input = AIMemoryTextInput(hass, mock_config_entry, mock_manager)

    # Set whitespace-only value
    await text_input.async_set_value("   \n\t   ")

    # Verify add_memory was not called
    mock_manager.async_add_memory.assert_not_called()


async def test_text_set_value_error(hass: HomeAssistant, mock_config_entry):
    """Test text input with error in add_memory."""
    # Create mock memory manager with error
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    mock_manager.memory_name = "Test Memory"
    mock_manager.async_add_memory = AsyncMock(side_effect=Exception("Test error"))

    # Create text input
    text_input = AIMemoryTextInput(hass, mock_config_entry, mock_manager)

    # Set a value (should not raise exception)
    test_text = "This is a test memory entry"
    await text_input.async_set_value(test_text)

    # Verify add_memory was called
    mock_manager.async_add_memory.assert_called_once_with(test_text)


async def test_text_properties():
    """Test text input properties."""
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

        # Setup mock responses
        mock_entity_reg.return_value.async_get.return_value = None  # No device found

        # Create text input
        text_input = AIMemoryTextInput(hass, entry, mock_manager)

        # Test properties - based on device_info presence
        if hasattr(mock_manager, 'device_info') and mock_manager.device_info:
            assert text_input.name == "Add Memory"
        else:
            assert text_input.name == "Add to Test Memory"
        assert text_input.unique_id == "ai_memory_text_test_memory"
        assert text_input.icon == "mdi:brain-plus"
        assert text_input._attr_max == 1000
        assert text_input._attr_min == 1
        assert text_input._attr_mode.value == "text"
        assert text_input.native_value == ""


async def test_text_extra_state_attributes():
    """Test text input extra state attributes."""
    # Create mock objects
    hass = MagicMock()
    entry = MagicMock()
    mock_manager = MagicMock()
    mock_manager.memory_id = "test_memory"
    mock_manager.memory_name = "Test Memory"

    # Mock device registry
    with patch('homeassistant.helpers.entity_registry.async_get'):
        # Create text input
        text_input = AIMemoryTextInput(hass, entry, mock_manager)

        # Check attributes
        attrs = text_input.extra_state_attributes
        assert attrs["memory_id"] == "test_memory"
        assert attrs["memory_name"] == "Test Memory"
        assert attrs["memory_sensor"] == "sensor.ai_memory_test_memory"


async def test_text_device_linking_private():
    """Test text input device linking for private memory."""
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

        # Create text input
        text_input = AIMemoryTextInput(hass, entry, mock_manager)

        # Check device info was set from manager
        assert text_input._attr_device_info == mock_manager.device_info

        # Check device-linked memory has short name and is disabled
        assert text_input.name == "Add Memory"
        assert text_input._attr_entity_registry_enabled_default == False


async def test_text_device_linking_common():
    """Test text input has no device linking for common memory."""
    # Create mock objects
    hass = MagicMock()
    entry = MagicMock()
    mock_manager = MagicMock()
    mock_manager.memory_id = "common"
    mock_manager.memory_name = "Common Memory"
    mock_manager.device_info = None  # No device info for common memory

    # Create text input
    text_input = AIMemoryTextInput(hass, entry, mock_manager)

    # Check no device info was set
    assert not hasattr(text_input, '_attr_device_info') or text_input._attr_device_info is None
    # Check common memory has full name and is disabled
    assert text_input.name == "Add to Common Memory"
    assert text_input._attr_entity_registry_enabled_default == False


async def test_text_setup_no_managers(hass: HomeAssistant, mock_config_entry):
    """Test text setup with no memory managers."""
    # Mock hass.data with no memory managers
    hass.data[DOMAIN] = {"memory_managers": {}}

    # Mock async_add_entities
    async_add_entities = MagicMock()

    # Call setup entry
    await async_setup_entry(hass, mock_config_entry, async_add_entities)

    # Verify no entities were added (early return when no managers)
    async_add_entities.assert_not_called()


async def test_text_common_memory_disabled():
    """Test text input for common memory is disabled by default."""
    # Create mock objects
    hass = MagicMock()
    entry = MagicMock()
    mock_manager = MagicMock()
    mock_manager.memory_id = "common"
    mock_manager.memory_name = "Common Memory"
    mock_manager.device_info = None

    # Create text input
    text_input = AIMemoryTextInput(hass, entry, mock_manager)

    # Check common memory has full name
    assert text_input.name == "Add to Common Memory"
    # Common memory should be disabled by default (no device_info)
    assert text_input._attr_entity_registry_enabled_default == False
