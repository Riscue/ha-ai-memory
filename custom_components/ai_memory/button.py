"""Button platform for AI Memory integration."""
import logging

from homeassistant.components.button import ButtonEntity
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
    """Set up AI Memory buttons."""
    memory_managers = hass.data[DOMAIN].get("memory_managers", {})

    _LOGGER.debug(f"Setting up buttons, found {len(memory_managers)} memory managers")

    buttons = []

    # Add clear button for each memory manager (add functionality will be via text input)
    for manager in memory_managers.values():
        _LOGGER.debug(f"Creating clear button for {manager.memory_name}")
        buttons.append(AIMemoryClearButton(hass, entry, manager))

    _LOGGER.info(f"Creating {len(buttons)} AI Memory buttons: {[b.name for b in buttons]}")
    async_add_entities(buttons, True)


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
            # Short name for device-linked memories
            self._attr_name = "Clear Memory"
        else:
            # Full name for stand-alone memories
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
            _LOGGER.info(f"Cleared memory: {self.memory_manager.memory_id}")
        except Exception as e:
            _LOGGER.error(f"Failed to clear memory {self.memory_manager.memory_id}: {e}")
