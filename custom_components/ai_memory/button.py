"""Button platform for AI Memory integration."""
import logging

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .memory_manager import MemoryManager
from .platform_helpers import async_setup_platform_entities

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
        hass: HomeAssistant,
        entry: ConfigEntry,
        async_add_entities: AddEntitiesCallback
) -> None:
    await async_setup_platform_entities(
        hass, entry, async_add_entities,
        AIMemoryClearButton, "button"
    )


class AIMemoryClearButton(ButtonEntity):
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
            self._attr_name = "Clear Memory"
        else:
            self._attr_name = f"Clear {memory_manager.memory_name}"

        self._attr_unique_id = f"ai_memory_clear_{memory_manager.memory_id}"
        self._attr_icon = "mdi:delete-sweep"
        self._attr_entity_category = EntityCategory.CONFIG
        self._attr_entity_registry_enabled_default = False

        if hasattr(memory_manager, 'device_info') and memory_manager.device_info:
            self._attr_device_info = memory_manager.device_info

    @property
    def state(self) -> str:
        """Return the state of the button."""
        return "ok"

    @property
    def memory_id(self) -> str:
        """Return the memory ID."""
        return self.memory_manager.memory_id

    @property
    def name(self) -> str:
        """Return the name of the button."""
        return self._attr_name

    @property
    def icon(self) -> str:
        """Return the icon of the button."""
        return self._attr_icon

    @property
    def device_info(self):
        """Return device info."""
        return self._attr_device_info if hasattr(self, '_attr_device_info') else None

    @property
    def extra_state_attributes(self):
        """Return extra state attributes."""
        return {
            "memory_id": self.memory_manager.memory_id,
            "memory_name": self.memory_manager.memory_name,
            "description": self.memory_manager.description,
        }

    async def async_press(self) -> None:
        try:
            await self.memory_manager.async_clear_memory()
            _LOGGER.debug(f"Cleared memory: {self.memory_manager.memory_id}")
        except Exception as e:
            _LOGGER.error(f"Failed to clear memory {self.memory_manager.memory_id}: {e}")
