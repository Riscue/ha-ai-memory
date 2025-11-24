"""AI Long Term Memory component."""
import json
import logging
import os
from datetime import datetime
from typing import Dict, List

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse

_LOGGER = logging.getLogger(__name__)

DOMAIN = "ai_memory"


async def async_setup(hass: HomeAssistant, config: dict):
    """Set up the AI Memory component."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Set up AI Memory from a config entry."""
    _LOGGER.debug(f"Setting up AI Memory: {entry.title}")
    _LOGGER.debug(f"Entry ID: {entry.entry_id}, Domain: {entry.domain}, Source: {entry.source}")

    # Ensure domain data exists
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}

    # Store entry reference
    if "entries" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["entries"] = {}

    hass.data[DOMAIN]["entries"][entry.entry_id] = entry

    # Initialize memory managers storage
    if "memory_managers" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["memory_managers"] = {}

    # Singleton check: If managers already exist, skip initialization but still forward platform
    if hass.data[DOMAIN]["memory_managers"]:
        _LOGGER.debug("AI Memory already initialized, skipping manager creation for additional entry")
        # Still forward to platforms to avoid unload errors
        await hass.config_entries.async_forward_entry_setups(entry, [Platform.SENSOR, Platform.TEXT, Platform.BUTTON])
        return True

    # Get global settings
    storage_location = entry.data.get("storage_location", "/config/ai_memory/")
    max_entries = entry.data.get("max_entries", 1000)

    # 1. Create Common Memory
    common_manager = MemoryManager(
        hass,
        "common",
        "Common Memory",
        "Shared memory for all agents",
        storage_location,
        max_entries,
        device_info=None  # No device for common memory
    )
    hass.data[DOMAIN]["memory_managers"]["common"] = common_manager
    _LOGGER.debug("Initialized Common Memory")

    # 2. Scan for Conversation Agents and Create Private Memories
    # We need to import here to avoid circular deps if any
    from homeassistant.components import conversation

    agent_infos = []

    # Get Conversation Entities
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
        device_info = None
        try:
            from homeassistant.helpers import entity_registry as er
            entity_reg = er.async_get(hass)

            # Try to find the device for this conversation agent
            for reg_entry in entity_reg.entities.values():
                if reg_entry.domain == "conversation":
                    state = hass.states.get(reg_entry.entity_id)
                    if state and state.attributes.get("friendly_name") == name:
                        if reg_entry.device_id:
                            from homeassistant.helpers import device_registry as dr
                            device_reg = dr.async_get(hass)
                            device = device_reg.async_get(reg_entry.device_id)
                            if device:
                                device_info = {
                                    "identifiers": device.identifiers,
                                    "name": device.name,
                                    "connections": device.connections,
                                }
                                break
        except Exception as e:
            _LOGGER.debug(f"Could not get device info for agent {name}: {e}")

        manager = MemoryManager(
            hass,
            memory_id,
            f"Private Memory: {name}",
            f"Private memory for {name}",
            storage_location,
            max_entries,
            device_info=device_info
        )
        hass.data[DOMAIN]["memory_managers"][memory_id] = manager
        _LOGGER.debug(f"Initialized Private Memory for agent: {name} (device: {device_info is not None})")

    # Debug: Log all memory managers
    _LOGGER.debug(f"Total memory managers created: {len(hass.data[DOMAIN]['memory_managers'])}")
    for mid, mgr in hass.data[DOMAIN]["memory_managers"].items():
        _LOGGER.debug(f"  - {mid}: {mgr.memory_name}")

    # Forward setup to sensor platform
    _LOGGER.debug(f"About to forward - Entry domain: {entry.domain}, Entry ID: {getattr(entry, 'entry_id', 'N/A')}")
    _LOGGER.debug(f"Entry object type: {type(entry).__name__}, Entry title: {getattr(entry, 'title', 'N/A')}")

    # Sanity check
    if entry.domain != DOMAIN:
        _LOGGER.error(f"CRITICAL: entry.domain is '{entry.domain}' but should be '{DOMAIN}'!")
        _LOGGER.error(f"Entry type: {type(entry)}, this might be a Home Assistant bug!")
        return False

    await hass.config_entries.async_forward_entry_setups(entry, [Platform.SENSOR, Platform.TEXT, Platform.BUTTON])

    # Register services
    _register_services(hass)
    _LOGGER.debug("AI Memory services registered")

    # Setup entry update listener
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    return True


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


class MemoryManager:
    """Manages a single memory storage."""

    def __init__(
            self,
            hass: HomeAssistant,
            memory_id: str,
            memory_name: str,
            description: str,
            storage_location: str,
            max_entries: int,
            device_info: dict = None
    ):
        self.hass = hass
        self.memory_id = memory_id
        self.memory_name = memory_name
        self.description = description
        self.storage_location = storage_location
        self.max_entries = max_entries
        self.device_info = device_info  # Device info for UI components
        self._memories: List[Dict[str, str]] = []

        # Ensure memory directory exists
        try:
            os.makedirs(self.storage_location, exist_ok=True)
            _LOGGER.debug(f"Memory directory ready: {self.storage_location}")
        except Exception as e:
            _LOGGER.error(f"Failed to create memory directory: {e}")

    def get_memory_file_path(self) -> str:
        """Get file path for this memory."""
        return os.path.join(self.storage_location, f"{self.memory_id}.json")

    def _read_file(self, file_path: str) -> List[Dict[str, str]]:
        """Read memories from file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        # _LOGGER.debug(f"Loaded {len(content)} memories from {file_path}")
                        return content
                    else:
                        _LOGGER.warning(f"Invalid memory format in {file_path}, expected list")
                        return []
        except json.JSONDecodeError as e:
            _LOGGER.error(f"JSON decode error in {file_path}: {e}")
            # Backup corrupted file
            try:
                if os.path.exists(file_path):
                    os.rename(file_path, f"{file_path}.bak")
                    _LOGGER.warning(f"Backed up corrupted memory file to {file_path}.bak")
            except Exception as backup_error:
                _LOGGER.error(f"Failed to backup corrupted file: {backup_error}")
        except Exception as e:
            _LOGGER.error(f"File read error ({file_path}): {e}")
        return []

    def _save_to_file(self, data: List[Dict[str, str]], file_path: str):
        """Save memories to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            _LOGGER.debug(f"Saved {len(data)} memories to {file_path}")
        except Exception as e:
            _LOGGER.error(f"File write error ({file_path}): {e}")
            # Attempt to save to a temp file to avoid data loss
            try:
                temp_path = f"{file_path}.tmp"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                _LOGGER.warning(f"Saved memory to temporary file {temp_path} due to write error")
            except Exception as temp_error:
                _LOGGER.error(f"Failed to save to temporary file: {temp_error}")

    async def async_load_memories(self):
        """Load memories from file."""
        file_path = self.get_memory_file_path()
        self._memories = await self.hass.async_add_executor_job(self._read_file, file_path)

    async def async_add_memory(self, text: str):
        """Add new memory entry."""
        if not text or not text.strip():
            _LOGGER.warning("Cannot add empty memory")
            return

        # Check memory limit
        if len(self._memories) >= self.max_entries:
            # Remove oldest entry if limit reached
            removed = self._memories.pop(0)
            _LOGGER.warning(
                f"Memory limit ({self.max_entries}) reached for '{self.memory_id}', "
                f"removed oldest entry: {removed['date']}"
            )

        entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text.strip()
        }
        self._memories.append(entry)

        file_path = self.get_memory_file_path()
        await self.hass.async_add_executor_job(self._save_to_file, self._memories, file_path)
        _LOGGER.debug(f"Memory added to '{self.memory_id}': {text[:50]}...")

    async def async_clear_memory(self):
        """Clear all memories."""
        count = len(self._memories)
        self._memories = []
        file_path = self.get_memory_file_path()
        await self.hass.async_add_executor_job(self._save_to_file, [], file_path)
        _LOGGER.debug(f"Memory '{self.memory_id}' cleared ({count} entries removed)")
