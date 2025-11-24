"""Test AI Memory Sensor."""
from unittest.mock import patch, MagicMock

from homeassistant.core import HomeAssistant

from custom_components.ai_memory import DOMAIN
from custom_components.ai_memory.sensor import AIMemorySensor


async def test_sensor_creation(hass: HomeAssistant, mock_config_entry, mock_agent_manager):
    """Test that sensors are created for memory managers."""
    mock_config_entry.add_to_hass(hass)

    # Setup entry to create managers
    with patch("custom_components.ai_memory.MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"):
        # Create a mock manager instance
        from unittest.mock import AsyncMock
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
        from custom_components.ai_memory.sensor import async_setup_entry as sensor_setup

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
