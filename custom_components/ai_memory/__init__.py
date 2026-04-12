"""AI Long Term Memory component."""
import logging
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall

from . import memory_llm_api
from .constants import DOMAIN, ENGINE_TFIDF, MEMORY_MAX_ENTRIES
from .memory.manager import MemoryManager

_LOGGER = logging.getLogger(__name__)

SERVICE_ADD_MEMORY = "add_memory"
SERVICE_CLEAR_MEMORY = "clear_memory"
SERVICE_LIST_MEMORIES = "list_memories"

ADD_MEMORY_SCHEMA = vol.Schema({
    vol.Required("text"): str,
})


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
    engine_type = entry.data.get("embedding_engine", ENGINE_TFIDF)
    max_entries = entry.data.get("max_entries", MEMORY_MAX_ENTRIES)

    manager = MemoryManager(hass, engine_type, max_entries, config_data=entry.data)

    # Initialize embedding engine
    await manager.async_initialize()

    hass.data[DOMAIN]["manager"] = manager
    _LOGGER.debug("Initialized Single Memory Manager")

    # Initialize LLM API
    await memory_llm_api.async_setup(hass)
    _LOGGER.debug("Initialized Memory LLM API")

    # Forward setup
    await hass.config_entries.async_forward_entry_setups(entry, [Platform.SENSOR])

    # Register services
    _register_services(hass)

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


def _register_services(hass: HomeAssistant):
    """Register HA services for AI Memory."""

    async def handle_add_memory(call: ServiceCall):
        """Handle add_memory service call."""
        manager = hass.data.get(DOMAIN, {}).get("manager")
        if not manager:
            _LOGGER.error("Memory manager not initialized")
            return

        text = call.data.get("text", "")
        await manager.async_add_memory(text, "common")

    async def handle_clear_memory(call: ServiceCall):
        """Handle clear_memory service call."""
        manager = hass.data.get(DOMAIN, {}).get("manager")
        if not manager:
            _LOGGER.error("Memory manager not initialized")
            return

        await hass.async_add_executor_job(
            manager._store.execute_commit,
            "DELETE FROM memories"
        )
        _LOGGER.info("All memories cleared")

    async def handle_list_memories(call: ServiceCall):
        """Handle list_memories service call."""
        manager = hass.data.get(DOMAIN, {}).get("manager")
        if not manager:
            _LOGGER.error("Memory manager not initialized")
            return

        counts = await manager.async_get_memory_counts()
        _LOGGER.info("Memory counts: %s", counts)

    hass.services.async_register(DOMAIN, SERVICE_ADD_MEMORY, handle_add_memory, schema=ADD_MEMORY_SCHEMA)
    hass.services.async_register(DOMAIN, SERVICE_CLEAR_MEMORY, handle_clear_memory)
    hass.services.async_register(DOMAIN, SERVICE_LIST_MEMORIES, handle_list_memories)


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Reload config entry."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Unload AI Memory config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, [Platform.SENSOR])

    if unload_ok:
        manager = hass.data[DOMAIN].pop("manager", None)
        if manager:
            manager.close()

        # Remove services
        for service in [SERVICE_ADD_MEMORY, SERVICE_CLEAR_MEMORY, SERVICE_LIST_MEMORIES]:
            hass.services.async_remove(DOMAIN, service)

    return unload_ok
