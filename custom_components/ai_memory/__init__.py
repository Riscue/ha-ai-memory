"""AI Long Term Memory component."""
import logging

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse

from . import memory_llm_api
from .constants import DOMAIN, ENGINE_TFIDF, MEMORY_MAX_ENTRIES
from .memory.manager import MemoryManager

_LOGGER = logging.getLogger(__name__)

SERVICE_ADD_MEMORY = "add_memory"
SERVICE_LIST_MEMORIES = "list_memories"
SERVICE_DELETE_MEMORY = "delete_memory"

ADD_MEMORY_SCHEMA = vol.Schema({
    vol.Required("text"): str,
    vol.Optional("room"): str,
    vol.Optional("wing"): str,
})

DELETE_MEMORY_SCHEMA = vol.Schema({
    vol.Optional("room"): str,
    vol.Optional("wing"): str,
    vol.Optional("scope"): vol.In(["private", "common"]),
    vol.Optional("agent_id"): str,
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
            return {"error": "Memory manager not initialized"}

        text = call.data.get("text", "")
        room = call.data.get("room")
        wing = call.data.get("wing")
        await manager.async_add_memory(text, "common", room=room, wing=wing)
        return {"success": True}

    async def handle_list_memories(call: ServiceCall):
        """Handle list_memories service call."""
        manager = hass.data.get(DOMAIN, {}).get("manager")
        if not manager:
            _LOGGER.error("Memory manager not initialized")
            return {"error": "Memory manager not initialized"}

        counts = await manager.async_get_memory_counts()
        _LOGGER.info("Memory counts: %s", counts)
        return counts

    async def handle_delete_memory(call: ServiceCall):
        """Handle delete_memory service call."""
        manager = hass.data.get(DOMAIN, {}).get("manager")
        if not manager:
            _LOGGER.error("Memory manager not initialized")
            return {"error": "Memory manager not initialized"}

        room = call.data.get("room")
        wing = call.data.get("wing")
        scope = call.data.get("scope")
        agent_id = call.data.get("agent_id")

        count = await manager.async_delete_memory(
            agent_id=agent_id,
            room=room,
            wing=wing,
            scope=scope,
        )
        _LOGGER.info("Deleted %d memory(s)", count)
        return {"deleted_count": count}

    hass.services.async_register(DOMAIN, SERVICE_ADD_MEMORY, handle_add_memory, schema=ADD_MEMORY_SCHEMA, supports_response=SupportsResponse.OPTIONAL)
    hass.services.async_register(DOMAIN, SERVICE_LIST_MEMORIES, handle_list_memories, supports_response=SupportsResponse.OPTIONAL)
    hass.services.async_register(DOMAIN, SERVICE_DELETE_MEMORY, handle_delete_memory, schema=DELETE_MEMORY_SCHEMA, supports_response=SupportsResponse.OPTIONAL)


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
        for service in [SERVICE_ADD_MEMORY, SERVICE_LIST_MEMORIES, SERVICE_DELETE_MEMORY]:
            hass.services.async_remove(DOMAIN, service)

    return unload_ok
