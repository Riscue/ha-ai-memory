import logging
from datetime import datetime

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

_LOGGER = logging.getLogger(__name__)

DOMAIN = "ai_memory"

from . import MemoryManager


async def async_setup_entry(
        hass: HomeAssistant,
        entry: ConfigEntry,
        async_add_entities: AddEntitiesCallback
):
    """Set up AI Memory sensors."""
    # Get all memory managers created in __init__
    memory_managers = hass.data[DOMAIN].get("memory_managers", {})

    sensors = []
    for manager in memory_managers.values():
        # Create sensor for each manager
        sensor = AIMemorySensor(hass, entry, manager)
        sensors.append(sensor)

        # Initialize memory
        await manager.async_load_memories()

    async_add_entities(sensors, True)

    # Register with conversation agents
    try:
        from .extended_openai_helper import async_register_with_conversation, create_template_helper
        # Register all managers
        for manager in memory_managers.values():
            await async_register_with_conversation(hass, manager)

        create_template_helper(hass)
    except Exception as e:
        _LOGGER.warning(f"Could not register with conversation agents: {e}")

    _LOGGER.debug(f"Created {len(sensors)} AI Memory sensors")


class AIMemorySensor(SensorEntity):
    """Representation of an AI Memory sensor."""

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

        # Link to conversation agent device if this is a private memory
        if memory_manager.memory_id.startswith("private_"):
            # Extract entity_id from memory_id (e.g., private_google_generative_ai -> conversation.google_generative_ai)
            agent_entity_id = memory_manager.memory_id.replace("private_", "")
            # Handle both formats: entity_id style and sanitized style
            if not agent_entity_id.startswith("conversation."):
                agent_entity_id = f"conversation.{agent_entity_id}"

            # Try to get device_id from entity registry
            from homeassistant.helpers import entity_registry as er
            entity_reg = er.async_get(hass)
            agent_entry = entity_reg.async_get(agent_entity_id)

            if agent_entry and agent_entry.device_id:
                from homeassistant.helpers import device_registry as dr
                device_reg = dr.async_get(hass)
                device = device_reg.async_get(agent_entry.device_id)

                if device:
                    self._attr_device_info = {
                        "identifiers": device.identifiers,
                        "name": device.name,
                    }

    @property
    def state(self):
        """State: Memory count."""
        return len(self.memory_manager._memories)

    @property
    def extra_state_attributes(self):
        """Attributes: Full text, prompt context, and metadata."""
        memories = self.memory_manager._memories
        full_text = "\n".join([f"- {m['date']}: {m['text']}" for m in memories])

        if full_text.strip():
            prompt_context_snippet = (
                "## LONG-TERM MEMORY\n"
                "This section contains permanent facts and preferences noted down from past conversations. "
                "Use this context to personalize your responses and guide your actions:\n"
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
        """Fetch new state data."""
        await self.memory_manager.async_load_memories()

    async def async_added_to_hass(self):
        """When entity is added to hass."""
        await self.memory_manager.async_load_memories()
