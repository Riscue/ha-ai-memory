"""Test AI Memory event-based update mechanism."""
from unittest.mock import MagicMock, AsyncMock

from homeassistant.core import HomeAssistant

from custom_components.ai_memory import MemoryManager
from custom_components.ai_memory.sensor import AIMemorySensor


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
