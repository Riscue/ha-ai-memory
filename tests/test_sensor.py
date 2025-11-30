"""Tests for AI Memory Sensor."""
from unittest.mock import MagicMock, AsyncMock

from homeassistant.core import HomeAssistant

from custom_components.ai_memory.constants import DOMAIN
from custom_components.ai_memory.sensor import AIMemorySensor, async_setup_entry as sensor_setup


async def test_sensor_creation(hass: HomeAssistant, mock_config_entry):
    """Test that sensor is created for memory manager."""
    mock_config_entry.add_to_hass(hass)

    # Mock manager
    mock_manager = MagicMock()
    mock_manager._max_entries = 100
    mock_manager._embedding_engine.engine_name = "test_engine"
    mock_manager.async_get_memory_counts = AsyncMock(return_value={"total": 10})

    hass.data[DOMAIN] = {"manager": mock_manager}

    async_add_entities = MagicMock()
    await sensor_setup(hass, mock_config_entry, async_add_entities)

    assert async_add_entities.called
    sensors = async_add_entities.call_args[0][0]
    assert len(sensors) == 1
    sensor = sensors[0]
    assert isinstance(sensor, AIMemorySensor)
    assert sensor.state == "Active"
    
    # Test update
    await sensor.async_update()
    assert sensor.extra_state_attributes["memory_counts"] == {"total": 10}
    assert sensor.extra_state_attributes["max_entries"] == 500


async def test_sensor_update_event(hass: HomeAssistant, mock_config_entry):
    """Test that sensor updates on event."""
    mock_manager = MagicMock()
    mock_manager._max_entries = 100
    mock_manager._embedding_engine.engine_name = "test_engine"
    
    hass.data[DOMAIN] = {"manager": mock_manager}

    sensor = AIMemorySensor(hass, mock_config_entry, mock_manager)
    sensor.async_schedule_update_ha_state = MagicMock()

    # Simulate adding to HA
    await sensor.async_added_to_hass()

    # Fire event
    await sensor._handle_memory_update({})

    sensor.async_schedule_update_ha_state.assert_called_once_with(force_refresh=True)
