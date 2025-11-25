"""AI Long Term Memory component."""
import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse

from custom_components.ai_memory.constants import DOMAIN
from custom_components.ai_memory.memory_manager import MemoryManager

_LOGGER = logging.getLogger(__name__)


async def async_setup(hass: HomeAssistant, config: dict):
    """Set up the AI Memory component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Set up AI Memory from a config entry."""
    if not hasattr(entry, 'entry_id'):
        _LOGGER.error("Config entry missing entry_id attribute")
        return False

    if entry.domain != DOMAIN:
        _LOGGER.error(f"CRITICAL: entry.domain is '{entry.domain}' but should be '{DOMAIN}'!")
        return False

    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}

    if "entries" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["entries"] = {}

    hass.data[DOMAIN]["entries"][entry.entry_id] = entry

    if "memory_managers" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["memory_managers"] = {}

    if hass.data[DOMAIN]["memory_managers"]:
        _LOGGER.debug("AI Memory already initialized, skipping manager creation for additional entry")
        await hass.config_entries.async_forward_entry_setups(entry, [Platform.SENSOR, Platform.TEXT, Platform.BUTTON])
        return True

    storage_location = entry.data.get("storage_location", "/config/ai_memory/")
    max_entries = entry.data.get("max_entries", 1000)
    common_manager = MemoryManager(
        hass,
        "common",
        "Common Memory",
        "Shared memory for all agents",
        storage_location,
        max_entries,
        device_info=None
    )
    hass.data[DOMAIN]["memory_managers"]["common"] = common_manager
    _LOGGER.debug("Initialized Common Memory")

    from homeassistant.components import conversation

    agent_infos = []

    _LOGGER.debug("Starting agent discovery for AI Memory...")
    if conversation.DOMAIN in hass.data:
        try:
            entities = hass.data[conversation.DOMAIN].entities
            _LOGGER.debug(f"Found {len(entities)} conversation entities in hass.data")
            for entity in entities:
                # Skip if entity.name is None
                if not entity.name:
                    _LOGGER.debug(f"Skipping entity with no name: {entity.entity_id}")
                    continue
                agent_infos.append({"name": entity.name, "id": entity.entity_id})
                _LOGGER.debug(f"Discovered agent from entities: {entity.name} ({entity.entity_id})")
        except Exception as e:
            _LOGGER.warning(f"Error accessing conversation entities: {e}")
    else:
        _LOGGER.warning("conversation.DOMAIN not found in hass.data")

    # Fallback: Check entity registry for conversation entities
    try:
        from homeassistant.helpers import entity_registry as er
        entity_reg = er.async_get(hass)

        # Iterate through all entities and filter by domain
        conversation_entities = [
            reg_entry for reg_entry in entity_reg.entities.values()
            if reg_entry.domain == "conversation"
        ]
        _LOGGER.debug(f"Entity registry has {len(conversation_entities)} conversation entities")

        for entity_entry in conversation_entities:
            entity_id = entity_entry.entity_id
            # Get friendly name from state, fallback to original_name or platform-based name
            state = hass.states.get(entity_id)
            if state and state.attributes.get("friendly_name"):
                name = state.attributes.get("friendly_name")
            elif entity_entry.original_name:
                name = entity_entry.original_name
            else:
                # Generate friendly name from platform
                # e.g., "google_generative_ai_conversation" -> "Google Generative AI"
                platform = entity_entry.platform
                if platform:
                    # Remove "_conversation" suffix if present
                    platform = platform.replace("_conversation", "")
                    # Convert snake_case to Title Case
                    name = platform.replace("_", " ").title()
                else:
                    # Last resort: use entity_id
                    name = entity_id

            # Avoid duplicates
            if not any(a["id"] == entity_id for a in agent_infos):
                agent_infos.append({"name": name, "id": entity_id})
                _LOGGER.debug(f"Discovered agent from entity registry: {name} ({entity_id})")
            else:
                _LOGGER.debug(f"Skipping duplicate from registry: {name} ({entity_id})")
    except Exception as e:
        _LOGGER.warning(f"Could not check entity registry: {e}")

    _LOGGER.debug(f"Total agents discovered: {len(agent_infos)}")

    for agent in agent_infos:
        name = agent["name"]

        # Skip agents with no name
        if not name:
            _LOGGER.warning(f"Skipping agent with no name: {agent['id']}")
            continue

        # Sanitize agent name for ID
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        memory_id = f"private_{safe_name}"

        # Get device info for this agent
        device_info = _get_device_info_for_agent(hass, name)

        manager = MemoryManager(
            hass,
            memory_id,
            f"Private Memory: {name}",
            f"Private memory for {name}",
            storage_location,
            max_entries,
            device_info=device_info,
            agent_id=agent["id"]
        )
        hass.data[DOMAIN]["memory_managers"][memory_id] = manager
        _LOGGER.debug(f"Initialized Private Memory for agent: {name} (device: {device_info is not None})")

    # Forward setup to platforms
    await hass.config_entries.async_forward_entry_setups(entry, [Platform.SENSOR, Platform.TEXT, Platform.BUTTON])

    # Register services
    _register_services(hass)
    _LOGGER.debug("AI Memory services registered")

    # Setup entry update listener
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    # Setup late device linking after Home Assistant is fully started
    async def late_init(_):
        await async_setup_device_linking(hass)

    hass.bus.async_listen_once("homeassistant_started", late_init)
    _LOGGER.debug("Scheduled late device linking after Home Assistant startup")

    return True


async def async_setup_device_linking(hass: HomeAssistant):
    """Late initialization for device linking after all components are ready."""
    _LOGGER.debug("Starting late device linking for AI Memory...")

    memory_managers = hass.data[DOMAIN].get("memory_managers", {})
    updated_count = 0

    for memory_id, manager in memory_managers.items():
        if memory_id.startswith("private_") and not manager.device_info:
            # Extract agent name from memory manager
            agent_name = manager.memory_name.replace("Private Memory: ", "")

            # Try to get device info
            device_info = _get_device_info_for_agent(hass, agent_name)
            if device_info:
                _LOGGER.debug(f"Linked {agent_name} to device: {device_info['name']}")
                manager.device_info = device_info
                updated_count += 1

    if updated_count > 0:
        _LOGGER.debug(f"Successfully linked {updated_count} AI Memory managers to devices")
        # Trigger entity registry update by reloading platforms
        try:
            # Get the config entry
            entries = hass.config_entries.async_entries(DOMAIN)
            if entries:
                entry = entries[0]  # Assume single entry
                await hass.config_entries.async_reload(entry.entry_id)
                _LOGGER.debug("Reloaded AI Memory entry to apply device linking")
        except Exception as e:
            _LOGGER.warning(f"Could not reload entry after device linking: {e}")
    else:
        _LOGGER.debug("No new device links found")


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Reload config entry."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Unload AI Memory config entry."""
    _LOGGER.debug(f"Unloading AI Memory: {entry.title}")

    # Unregister from conversation agents
    try:
        from .extended_openai_helper import async_unregister_from_conversation
        await async_unregister_from_conversation(hass, entry.entry_id)
    except Exception as e:
        _LOGGER.debug(f"Could not unregister from conversation: {e}")

    # Unload platforms
    unload_ok = await hass.config_entries.async_unload_platforms(entry,
                                                                 [Platform.SENSOR, Platform.TEXT, Platform.BUTTON])

    if unload_ok:
        # Remove entry reference
        if "entries" in hass.data[DOMAIN]:
            hass.data[DOMAIN]["entries"].pop(entry.entry_id, None)

        # Don't remove memory_managers here - they're shared across all entries (singleton)
        # Only clear them when the last entry is removed
        remaining_entries = [
            e for e in hass.config_entries.async_entries(DOMAIN)
            if e.entry_id != entry.entry_id
        ]

        if not remaining_entries:
            # This is the last entry, clean up everything
            if "memory_managers" in hass.data[DOMAIN]:
                hass.data[DOMAIN]["memory_managers"].clear()

            hass.services.async_remove(DOMAIN, "add_memory")
            hass.services.async_remove(DOMAIN, "clear_memory")
            hass.services.async_remove(DOMAIN, "list_memories")
            hass.services.async_remove(DOMAIN, "get_context")
            _LOGGER.debug("AI Memory services unregistered")

        _LOGGER.debug(f"AI Memory entry '{entry.title}' successfully removed")

    return unload_ok


def _get_device_info_for_agent(hass: HomeAssistant, agent_name: str) -> dict:
    """Get device info for a conversation agent."""
    try:
        from homeassistant.helpers import entity_registry as er
        entity_reg = er.async_get(hass)

        for reg_entry in entity_reg.entities.values():
            if reg_entry.domain == "conversation":
                state = hass.states.get(reg_entry.entity_id)
                if state and state.attributes.get("friendly_name") == agent_name:
                    if reg_entry.device_id:
                        from homeassistant.helpers import device_registry as dr
                        device_reg = dr.async_get(hass)
                        device = device_reg.async_get(reg_entry.device_id)
                        if device:
                            return {
                                "identifiers": device.identifiers,
                                "name": device.name,
                                "connections": device.connections,
                            }
    except Exception as e:
        _LOGGER.debug(f"Could not get device info for agent {agent_name}: {e}")

    return None


def _register_services(hass: HomeAssistant):
    """Register AI Memory services."""

    async def handle_add_memory(call: ServiceCall):
        """Handle add memory service call."""
        text = call.data.get("text")
        entity_id = call.data.get("memory_id")

        if not text:
            _LOGGER.error("No text provided for add_memory service")
            return

        if not entity_id:
            _LOGGER.error("No memory_id (entity) provided for add_memory service")
            return

        # Get memory_id from entity attributes
        entity_state = hass.states.get(entity_id)
        if not entity_state:
            _LOGGER.error(f"Entity '{entity_id}' not found")
            return

        memory_id = entity_state.attributes.get("memory_id")
        if not memory_id:
            _LOGGER.error(f"Entity '{entity_id}' does not have memory_id attribute")
            return

        # Find the memory manager for this memory_id
        memory_managers = hass.data[DOMAIN].get("memory_managers", {})

        manager_found = False
        for manager in memory_managers.values():
            if manager.memory_id == memory_id:
                await manager.async_add_memory(text)
                manager_found = True
                break

        if not manager_found:
            available_memories = [m.memory_id for m in memory_managers.values()]
            _LOGGER.error(
                f"Memory '{memory_id}' not found. Available memories: {available_memories}"
            )

    async def handle_clear_memory(call: ServiceCall):
        """Handle clear memory service call."""
        entity_id = call.data.get("memory_id")

        if not entity_id:
            _LOGGER.error("No memory_id (entity) provided for clear_memory service")
            return

        # Get memory_id from entity attributes
        entity_state = hass.states.get(entity_id)
        if not entity_state:
            _LOGGER.error(f"Entity '{entity_id}' not found")
            return

        memory_id = entity_state.attributes.get("memory_id")
        if not memory_id:
            _LOGGER.error(f"Entity '{entity_id}' does not have memory_id attribute")
            return

        # Find the memory manager for this memory_id
        memory_managers = hass.data[DOMAIN].get("memory_managers", {})

        manager_found = False
        for manager in memory_managers.values():
            if manager.memory_id == memory_id:
                await manager.async_clear_memory()
                _LOGGER.debug(f"Memory '{memory_id}' cleared")
                manager_found = True
                break

        if not manager_found:
            available_memories = [m.memory_id for m in memory_managers.values()]
            _LOGGER.error(
                f"Memory '{memory_id}' not found. Available memories: {available_memories}"
            )

    async def handle_list_memories(call: ServiceCall) -> ServiceResponse:
        """Handle list memories service call."""
        memory_managers = hass.data[DOMAIN].get("memory_managers", {})

        memories = []

        for manager in memory_managers.values():
            memories.append({
                "memory_id": manager.memory_id,
                "memory_name": manager.memory_name,
                "description": manager.description,
                "entry_count": len(manager._memories),
                "max_entries": manager.max_entries,
                "storage_location": manager.storage_location,
            })

        return {"memories": memories}

    async def handle_get_context(call: ServiceCall) -> ServiceResponse:
        """Handle get context service call - returns formatted context for debugging."""
        entity_id = call.data.get("memory_id")

        memory_managers = hass.data[DOMAIN].get("memory_managers", {})

        if entity_id:
            # Get memory_id from entity attributes
            entity_state = hass.states.get(entity_id)
            if not entity_state:
                return {
                    "error": f"Entity '{entity_id}' not found",
                    "available": [m.memory_id for m in memory_managers.values()]
                }

            memory_id = entity_state.attributes.get("memory_id")
            if not memory_id:
                return {
                    "error": f"Entity '{entity_id}' does not have memory_id attribute",
                    "available": [m.memory_id for m in memory_managers.values()]
                }

            # Get specific memory context
            for manager in memory_managers.values():
                if manager.memory_id == memory_id:
                    from .extended_openai_helper import get_memory_context_for_llm
                    context = get_memory_context_for_llm(manager)
                    return {
                        "memory_id": memory_id,
                        "context": context or "No memories yet"
                    }
            return {
                "error": f"Memory '{memory_id}' not found",
                "available": [m.memory_id for m in memory_managers.values()]
            }
        else:
            # Get all contexts
            from .extended_openai_helper import get_all_memory_contexts
            all_contexts = get_all_memory_contexts(hass)
            return {
                "context": all_contexts or "No memories available",
                "memory_count": len(memory_managers)
            }

    # Register services
    hass.services.async_register(DOMAIN, "add_memory", handle_add_memory)
    hass.services.async_register(DOMAIN, "clear_memory", handle_clear_memory)
    hass.services.async_register(
        DOMAIN,
        "list_memories",
        handle_list_memories,
        supports_response=SupportsResponse.ONLY
    )
    hass.services.async_register(
        DOMAIN,
        "get_context",
        handle_get_context,
        supports_response=SupportsResponse.ONLY
    )
