"""Sensor platform for AI Memory integration."""
import logging
from datetime import datetime

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .constants import DOMAIN
from .memory_manager import MemoryManager

_LOGGER = logging.getLogger(__name__)


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

    @property
    def state(self):
        # We can't easily get total count without a query in SQLite, 
        # but for now let's assume we might want to expose something else or 
        # just return "Active" or query count if cheap.
        # Let's return "Active" for now as count requires async query.
        return "Active"

    @property
    def extra_state_attributes(self):
        return {
            "storage_location": self.memory_manager.storage_location,
            "max_entries": self.memory_manager.max_entries,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    async def async_update(self):
        # No-op for now as state is static "Active"
        pass

    async def async_added_to_hass(self):
        self.async_on_remove(
            self.hass.bus.async_listen(
                "ai_memory_updated",
                self._handle_memory_update
            )
        )

    async def _handle_memory_update(self, event):
        self.async_schedule_update_ha_state(force_refresh=True)
