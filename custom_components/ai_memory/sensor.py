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
    memory_managers = hass.data[DOMAIN].get("memory_managers", {})

    sensors = []
    for manager in memory_managers.values():
        sensor = AIMemorySensor(hass, entry, manager)
        sensors.append(sensor)
        await manager.async_load_memories()

    async_add_entities(sensors, True)

    try:
        from .extended_openai_helper import async_register_with_conversation, create_template_helper
        for manager in memory_managers.values():
            await async_register_with_conversation(hass, manager)

        create_template_helper(hass)
    except Exception as e:
        _LOGGER.warning(f"Could not register with conversation agents: {e}")

    _LOGGER.debug(f"Created {len(sensors)} AI Memory sensors")


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
        self._attr_name = f"AI Memory - {memory_manager.memory_name}"
        self._attr_unique_id = f"ai_memory_{memory_manager.memory_id}"
        self._attr_icon = "mdi:brain"

        if hasattr(memory_manager, 'device_info') and memory_manager.device_info:
            self._attr_device_info = memory_manager.device_info

        self._attr_entity_registry_enabled_default = True

    @property
    def state(self):
        return len(self.memory_manager._memories)

    @property
    def extra_state_attributes(self):
        memories = self.memory_manager._memories
        full_text = "\n".join([f"- {m['text']}" for m in memories])

        if full_text.strip():
            prompt_context_snippet = (
                "## LONG-TERM MEMORY\n"
                "Below are stable preferences inferred from past conversations. "
                "Use these only when relevant and do not over-personalize responses.\n"
                "--- MEMORY START ---\n"
                f"{full_text}\n"
                "--- MEMORY END ---\n"
            )
        else:
            prompt_context_snippet = ""

        return {
            "full_text": full_text,
            "prompt_context_snippet": prompt_context_snippet,
            "memory_id": self.memory_manager.memory_id,
            "memory_name": self.memory_manager.memory_name,
            "description": self.memory_manager.description,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "max_entries": self.memory_manager.max_entries,
            "entry_count": len(memories),
            "storage_location": self.memory_manager.storage_location,
            "created_at": self.entry.data.get("created_at", "Unknown")
        }

    async def async_update(self):
        await self.memory_manager.async_load_memories()

    async def async_added_to_hass(self):
        await self.memory_manager.async_load_memories()

        self.async_on_remove(
            self.hass.bus.async_listen(
                "ai_memory_updated",
                self._handle_memory_update
            )
        )

    async def _handle_memory_update(self, event):
        if event.data.get("memory_id") == self.memory_manager.memory_id:
            await self.async_update()
            self.async_schedule_update_ha_state()
