"""AI Long Term Memory component."""
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from custom_components.ai_memory.constants import DOMAIN, ENGINE_AUTO
from custom_components.ai_memory.memory_manager import MemoryManager

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: dict):
    """Set up the AI Memory component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Set up AI Memory from a config entry."""
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}

    if "manager" in hass.data[DOMAIN]:
        _LOGGER.debug("AI Memory already initialized")
        await hass.config_entries.async_forward_entry_setups(entry, [Platform.SENSOR])
        return True

    # Initialize Single Memory Manager
    engine_type = entry.data.get("embedding_engine", ENGINE_AUTO)
    max_entries = entry.data.get("max_entries", 1000)

    manager = MemoryManager(hass, engine_type, max_entries)
    await manager.async_load_memories()

    hass.data[DOMAIN]["manager"] = manager
    _LOGGER.debug("Initialized Single Memory Manager")

    # Initialize LLM API
    from . import memory_llm_api
    await memory_llm_api.async_setup(hass)
    _LOGGER.debug("Initialized Memory LLM API")

    # Forward setup
    await hass.config_entries.async_forward_entry_setups(entry, [Platform.SENSOR])

    # Register services (Simplified)
    # _register_services(hass) # TODO: Update services to use new manager

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Reload config entry."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Unload AI Memory config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, [Platform.SENSOR])

    if unload_ok:
        hass.data[DOMAIN].pop("manager", None)

    return unload_ok
