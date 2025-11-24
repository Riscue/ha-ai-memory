"""Comprehensive tests for AI Memory Sensor and event-based updates."""
from unittest.mock import patch, MagicMock, AsyncMock

from homeassistant.core import HomeAssistant

from custom_components.ai_memory import DOMAIN
from custom_components.ai_memory.memory_manager import MemoryManager
from custom_components.ai_memory.sensor import AIMemorySensor, async_setup_entry as sensor_setup


# Sensor creation tests
async def test_sensor_creation(hass: HomeAssistant, mock_config_entry, mock_agent_manager):
    """Test that sensors are created for memory managers."""
    mock_config_entry.add_to_hass(hass)

    # Setup entry to create managers
    with patch("custom_components.ai_memory.memory_manager.MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"):
        # Create a mock manager instance
        mock_manager = MagicMock()
        mock_manager.async_load_memories = AsyncMock()
        mock_manager.memory_id = "test_mem"
        mock_manager.memory_name = "Test Memory"
        mock_manager.description = "Test Desc"
        mock_manager.max_entries = 10
        mock_manager.storage_location = "/tmp"
        mock_manager._memories = [{"date": "2023-01-01", "text": "Test"}]

        mock_manager_cls.return_value = mock_manager

        # Setup the integration
        from custom_components.ai_memory import async_setup_entry
        await async_setup_entry(hass, mock_config_entry)
        await hass.async_block_till_done()

        # Verify sensor entity exists
        # Note: The unique ID logic in sensor.py uses "ai_memory_{memory_id}"
        # But entity ID might be "sensor.ai_memory_test_memory" or similar.
        # We'll check if any sensor is added.

        # Since we mocked async_setup_entry in conftest (mock_setup_entry fixture),
        # we need to be careful. The fixture mocks the *integration* setup.
        # Here we want to test the *sensor platform* setup.

        # Let's manually trigger sensor setup
        # Mock the manager in hass.data
        hass.data[DOMAIN] = {"memory_managers": {"test": mock_manager}}

        async_add_entities = MagicMock()
        await sensor_setup(hass, mock_config_entry, async_add_entities)

        assert async_add_entities.called
        sensors = async_add_entities.call_args[0][0]
        assert len(sensors) == 1
        sensor = sensors[0]
        assert isinstance(sensor, AIMemorySensor)
        assert sensor.state == 1
        assert sensor.extra_state_attributes["memory_id"] == "test_mem"
        assert "Test" in sensor.extra_state_attributes["full_text"]


# Event-based update tests
async def test_sensor_listens_to_memory_events(hass: HomeAssistant, mock_config_entry):
    """Test that sensors listen to memory update events."""
    manager = MemoryManager(
        hass,
        "test_mem",
        "Test Memory",
        "Desc",
        "/tmp",
        10
    )

    # Mock file operations
    manager._read_file = MagicMock(return_value=[])
    manager._save_to_file = MagicMock(return_value=True)

    # Create sensor
    sensor = AIMemorySensor(hass, mock_config_entry, manager)

    # Mock async_update to track if it's called
    sensor.async_update = AsyncMock()

    # Add sensor to hass to set up event listeners
    await sensor.async_added_to_hass()

    # Fire a memory update event
    hass.bus.async_fire("ai_memory_updated", {"memory_id": "test_mem"})
    await hass.async_block_till_done()

    # Verify sensor's async_update was called
    assert sensor.async_update.call_count == 1

    # Fire event for different memory ID - should not update sensor
    hass.bus.async_fire("ai_memory_updated", {"memory_id": "other_mem"})
    await hass.async_block_till_done()

    # Call count should still be 1 (not called for different memory)
    assert sensor.async_update.call_count == 1


async def test_sensor_ignores_unrelated_events(hass: HomeAssistant, mock_config_entry):
    """Test that sensors ignore events for other memory managers."""
    manager = MemoryManager(
        hass,
        "my_mem",
        "My Memory",
        "Desc",
        "/tmp",
        10
    )

    # Create sensor
    sensor = AIMemorySensor(hass, mock_config_entry, manager)
    sensor.async_update = AsyncMock()

    # Add sensor to hass
    await sensor.async_added_to_hass()

    # Fire event for different memory ID
    hass.bus.async_fire("ai_memory_updated", {"memory_id": "other_mem"})
    await hass.async_block_till_done()

    # Sensor should not have been updated
    assert sensor.async_update.call_count == 0


async def test_sensor_setup_conversation_registration_error(hass: HomeAssistant, mock_config_entry):
    """Test sensor setup when conversation registration fails."""
    # Setup mock managers
    mock_manager = MagicMock()
    mock_manager.memory_id = "test"
    mock_manager.async_load_memories = AsyncMock()

    hass.data[DOMAIN] = {"memory_managers": {"test": mock_manager}}
    async_add_entities = MagicMock()

    # Mock the conversation registration to fail
    with patch("custom_components.ai_memory.extended_openai_helper.async_register_with_conversation",
               side_effect=Exception("Registration failed")):
        await sensor_setup(hass, mock_config_entry, async_add_entities)

        # Should still create sensors despite registration error
        async_add_entities.assert_called_once()


async def test_sensor_setup_conversation_registration_import_error(hass: HomeAssistant, mock_config_entry):
    """Test sensor setup when extended_openai_helper import fails."""
    # Setup mock managers
    mock_manager = MagicMock()
    mock_manager.memory_id = "test"
    mock_manager.async_load_memories = AsyncMock()

    hass.data[DOMAIN] = {"memory_managers": {"test": mock_manager}}
    async_add_entities = MagicMock()

    # Mock import to fail
    with patch("builtins.__import__", side_effect=ImportError("Cannot import")):
        await sensor_setup(hass, mock_config_entry, async_add_entities)

        # Should still create sensors despite import error
        async_add_entities.assert_called_once()


# Additional sensor tests for coverage
async def test_sensor_device_info_linking(hass: HomeAssistant, mock_config_entry):
    """Test sensor device info linking."""
    # Create a manager with device info
    manager = MagicMock()
    manager.memory_id = "test_mem"
    manager.memory_name = "Test Memory"
    manager.description = "Test Description"
    manager.async_load_memories = AsyncMock()
    manager.device_info = {
        "identifiers": {("test", "device")},
        "name": "Test Device"
    }
    manager._memories = [{"date": "2023-01-01", "text": "Test memory"}]

    hass.data[DOMAIN] = {"memory_managers": {"test_mem": manager}}
    async_add_entities = MagicMock()

    await sensor_setup(hass, mock_config_entry, async_add_entities)

    # Verify sensor was created
    assert async_add_entities.called
    sensors = async_add_entities.call_args[0][0]
    assert len(sensors) == 1

    sensor = sensors[0]
    assert sensor._attr_device_info == manager.device_info


async def test_sensor_state_with_empty_memories(hass: HomeAssistant, mock_config_entry):
    """Test sensor state when no memories exist."""
    manager = MagicMock()
    manager.memory_id = "test_mem"
    manager.memory_name = "Test Memory"
    manager.description = "Test Description"
    manager.async_load_memories = AsyncMock()
    manager._memories = []  # Empty memories

    hass.data[DOMAIN] = {"memory_managers": {"test_mem": manager}}
    async_add_entities = MagicMock()

    await sensor_setup(hass, mock_config_entry, async_add_entities)

    # Verify sensor was created with state 0
    assert async_add_entities.called
    sensors = async_add_entities.call_args[0][0]
    assert len(sensors) == 1

    sensor = sensors[0]
    assert sensor.state == 0


async def test_sensor_extra_state_attributes_formatting(hass: HomeAssistant, mock_config_entry):
    """Test sensor extra state attributes formatting."""
    manager = MagicMock()
    manager.memory_id = "test_mem"
    manager.memory_name = "Test Memory"
    manager.description = "Test Description"
    manager.async_load_memories = AsyncMock()
    manager._memories = [
        {"date": "2023-01-01 12:00:00", "text": "First memory"},
        {"date": "2023-01-02 15:30:00", "text": "Second memory"}
    ]
    manager.storage_location = "/config/ai_memory"
    manager.max_entries = 1000

    hass.data[DOMAIN] = {"memory_managers": {"test_mem": manager}}
    async_add_entities = MagicMock()

    await sensor_setup(hass, mock_config_entry, async_add_entities)

    # Verify sensor attributes
    assert async_add_entities.called
    sensors = async_add_entities.call_args[0][0]
    assert len(sensors) == 1

    sensor = sensors[0]
    attributes = sensor.extra_state_attributes

    # Check basic attributes
    assert attributes["memory_id"] == "test_mem"
    assert attributes["memory_name"] == "Test Memory"
    assert attributes["description"] == "Test Description"
    assert attributes["max_entries"] == 1000
    assert attributes["entry_count"] == 2
    assert attributes["storage_location"] == "/config/ai_memory"

    # Check full_text formatting
    full_text = attributes["full_text"]
    assert "- First memory" in full_text
    assert "- Second memory" in full_text

    # Check prompt context snippet
    context = attributes["prompt_context_snippet"]
    assert "LONG-TERM MEMORY" in context
    assert "First memory" in context
    assert "Second memory" in context
    assert "--- MEMORY START ---" in context
    assert "--- MEMORY END ---" in context


async def test_sensor_with_no_description(hass: HomeAssistant, mock_config_entry):
    """Test sensor when memory manager has no description."""
    manager = MagicMock()
    manager.memory_id = "test_mem"
    manager.memory_name = "Test Memory"
    manager.description = None  # No description
    manager.async_load_memories = AsyncMock()
    manager._memories = [{"date": "2023-01-01", "text": "Test memory"}]

    hass.data[DOMAIN] = {"memory_managers": {"test_mem": manager}}
    async_add_entities = MagicMock()

    await sensor_setup(hass, mock_config_entry, async_add_entities)

    # Verify sensor was created
    assert async_add_entities.called
    sensors = async_add_entities.call_args[0][0]
    assert len(sensors) == 1

    sensor = sensors[0]
    attributes = sensor.extra_state_attributes
    assert attributes["description"] is None


async def test_sensor_update_method(hass: HomeAssistant, mock_config_entry):
    """Test sensor async_update method."""
    manager = MagicMock()
    manager.memory_id = "test_mem"
    manager.memory_name = "Test Memory"
    manager.async_load_memories = AsyncMock()

    sensor = AIMemorySensor(hass, mock_config_entry, manager)

    # Call async_update
    await sensor.async_update()

    # Should have called manager.async_load_memories
    manager.async_load_memories.assert_called_once()


async def test_sensor_handle_memory_update_event(hass: HomeAssistant, mock_config_entry):
    """Test sensor memory update event handler."""
    manager = MagicMock()
    manager.memory_id = "test_mem"
    manager.memory_name = "Test Memory"
    manager.async_load_memories = AsyncMock()

    sensor = AIMemorySensor(hass, mock_config_entry, manager)

    # Add sensor to hass to set entity_id and platform
    sensor.entity_id = "sensor.test_memory"
    sensor.async_schedule_update_ha_state = MagicMock()

    # Mock async_update to track if it's called
    async_update_mock = AsyncMock()
    original_async_update = sensor.async_update
    sensor.async_update = async_update_mock

    # Create mock event
    event = MagicMock()
    event.data = {"memory_id": "test_mem"}

    # Call event handler
    await sensor._handle_memory_update(event)

    # Should have called async_update
    async_update_mock.assert_called_once()
    sensor.async_schedule_update_ha_state.assert_called_once()

    # Reset mocks
    async_update_mock.reset_mock()
    sensor.async_schedule_update_ha_state.reset_mock()

    # Restore original async_update
    sensor.async_update = original_async_update

    # Test with different memory ID
    event.data = {"memory_id": "other_mem"}
    await sensor._handle_memory_update(event)

    # Should not have called schedule_update_ha_state
    sensor.async_schedule_update_ha_state.assert_not_called()
