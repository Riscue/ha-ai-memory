"""Button platform for AI Memory integration."""
import logging

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import MemoryManager
from .platform_helpers import async_setup_platform_entities

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
        hass: HomeAssistant,
        entry: ConfigEntry,
        async_add_entities: AddEntitiesCallback
) -> None:
    """Set up AI Memory buttons."""
    await async_setup_platform_entities(
        hass, entry, async_add_entities,
        AIMemoryClearButton, "button"
    )


class AIMemoryClearButton(ButtonEntity):
    """Button to clear AI memory."""

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
        # All buttons should be disabled by default
        self._attr_entity_registry_enabled_default = False

        # Link to device if device info is available - use precomputed device info
        if hasattr(memory_manager, 'device_info') and memory_manager.device_info:
            self._attr_device_info = memory_manager.device_info

    async def async_press(self) -> None:
        """Handle the button press."""
        try:
            await self.memory_manager.async_clear_memory()
            _LOGGER.debug(f"Cleared memory: {self.memory_manager.memory_id}")
        except Exception as e:
            _LOGGER.error(f"Failed to clear memory {self.memory_manager.memory_id}: {e}")
