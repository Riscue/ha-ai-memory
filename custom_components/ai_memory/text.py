"""Text input platform for AI Memory integration."""
import logging

from homeassistant.components.text import TextEntity, TextMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import DOMAIN, MemoryManager

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
        hass: HomeAssistant,
        entry: ConfigEntry,
        async_add_entities: AddEntitiesCallback
) -> None:
    """Set up AI Memory text inputs."""
    memory_managers = hass.data[DOMAIN].get("memory_managers", {})

    text_inputs = []
    for manager in memory_managers.values():
        text_inputs.append(AIMemoryTextInput(hass, entry, manager))

    _LOGGER.debug(f"Creating {len(text_inputs)} AI Memory text inputs")
    async_add_entities(text_inputs, True)


class AIMemoryTextInput(TextEntity):
    """Text input to add memory entries."""

    def __init__(
            self,
            hass: HomeAssistant,
            entry: ConfigEntry,
            memory_manager: MemoryManager
    ) -> None:
        self.hass = hass
        self.entry = entry
        self.memory_manager = memory_manager
        if hasattr(memory_manager, 'device_info') and memory_manager.device_info:
            # Short name for device-linked memories
            self._attr_name = "Add Memory"
        else:
            # Full name for stand-alone memories
            self._attr_name = f"Add to {memory_manager.memory_name}"

        self._attr_unique_id = f"ai_memory_text_{memory_manager.memory_id}"
        self._attr_icon = "mdi:brain-plus"
        self._attr_entity_category = EntityCategory.CONFIG
        self._attr_mode = TextMode.TEXT
        self._attr_max = 1000  # Maximum 1000 characters
        self._attr_min = 1  # Minimum 1 character
        self._attr_pattern = None
        self._native_value = ""

        # All text inputs should be disabled by default
        self._attr_entity_registry_enabled_default = False

        # Link to device if device info is available - use precomputed device info
        if hasattr(memory_manager, 'device_info') and memory_manager.device_info:
            self._attr_device_info = memory_manager.device_info

    @property
    def native_value(self) -> str:
        """Return the value reported by the text."""
        return self._native_value or ""

    async def async_set_value(self, value: str) -> None:
        """Set new value."""
        if value and value.strip():
            try:
                await self.memory_manager.async_add_memory(value.strip())
                _LOGGER.debug(f"Added memory entry to {self.memory_manager.memory_name}: {value[:50]}...")
                self._native_value = ""  # Clear after successful submission
                self.async_write_ha_state()
            except Exception as e:
                _LOGGER.error(f"Failed to add memory to {self.memory_manager.memory_id}: {e}")
        else:
            _LOGGER.warning("Empty memory text provided")

    @property
    def extra_state_attributes(self):
        """Add additional attributes for the text input."""
        return {
            "memory_id": self.memory_manager.memory_id,
            "memory_name": self.memory_manager.memory_name,
            "memory_sensor": f"sensor.ai_memory_{self.memory_manager.memory_id}"
        }
