"""Sensor platform for AI Memory integration."""
import logging
from datetime import timedelta, datetime

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .constants import DOMAIN
from .memory_manager import MemoryManager

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(minutes=15)


async def async_setup_entry(
        hass: HomeAssistant,
        entry: ConfigEntry,
        async_add_entities: AddEntitiesCallback
):
    manager = hass.data[DOMAIN].get("manager")
    if not manager:
        _LOGGER.error("Memory manager not found")
        return

    sensor = AIMemorySensor(hass, entry, manager)
    async_add_entities([sensor], True)
    _LOGGER.debug("Created AI Memory sensor")


class AIMemorySensor(SensorEntity):
    def __init__(
            self,
            hass: HomeAssistant,
            entry: ConfigEntry,
            memory_manager: MemoryManager
    ):
        self.hass = hass
        self.entry = entry
        self.memory_manager = memory_manager
        self._attr_name = "AI Memory"
        self._attr_unique_id = "ai_memory_store"
        self._attr_icon = "mdi:brain"
        self._attr_entity_registry_enabled_default = True
        self._memory_counts = {}

    @property
    def state(self):
        return "Active"

    @property
    def extra_state_attributes(self):
        attrs = {
            "embedding_engine": self.memory_manager._embedding_engine.engine_name,
            "max_entries": self.memory_manager._max_entries,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "memory_counts": self._memory_counts,
        }

        # Add config data
        if self.entry.data:
            attrs.update(self.entry.data)

        return attrs

    async def async_update(self):
        """Update sensor state."""
        self._memory_counts = await self.memory_manager.async_get_memory_counts()

    async def async_added_to_hass(self):
        self.async_on_remove(
            self.hass.bus.async_listen(
                "ai_memory_updated",
                self._handle_memory_update
            )
        )

    async def _handle_memory_update(self, event):
        """Handle memory update event."""
        self.async_schedule_update_ha_state(force_refresh=True)
