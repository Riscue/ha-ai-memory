"""Helper for conversation agent integration."""
import logging
from typing import Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent

from .constants import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_register_with_conversation(hass: HomeAssistant, manager):
    """Register memory with conversation agents."""

    # Manager is passed directly now
    if not manager:
        _LOGGER.error("No memory manager provided for registration")
        return

    # Register conversation context provider (for all agents)
    try:
        await _register_conversation_context(hass, manager)
        _LOGGER.debug(f"Registered memory '{manager.memory_name}' with conversation agents")
    except Exception as e:
        _LOGGER.debug(f"Could not register conversation context: {e}")

    # Register intents for voice control
    try:
        await async_ensure_add_memory_intent(hass)
        _LOGGER.debug(f"Ensured AddMemory intent is registered")
    except Exception as e:
        _LOGGER.debug(f"Could not register intents: {e}")


async def _register_conversation_context(hass: HomeAssistant, manager):
    """Register as conversation context provider."""

    # Create our own storage in hass.data for memory contexts
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}

    if "conversation_contexts" not in hass.data[DOMAIN]:
        hass.data[DOMAIN]["conversation_contexts"] = {}

    # Store context provider
    hass.data[DOMAIN]["conversation_contexts"][manager.memory_id] = {
        "manager": manager,
        "get_context": lambda: get_memory_context_for_llm(manager)
    }


async def async_ensure_add_memory_intent(hass: HomeAssistant):
    class AddMemoryIntent(intent.IntentHandler):
        """Handle adding to a specific memory."""
        intent_type = "AddMemory"
        description = "Add content to a specific long-term memory"
        slot_schema = {
            "text": str,
            "shared": bool
        }

        async def async_handle(self, intent_obj: intent.Intent):
            text = intent_obj.slots.get("text", {}).get("value", "")
            shared_slot = intent_obj.slots.get("shared", {}).get("value")

            if not text:
                response = intent_obj.create_response()
                response.async_set_speech("I didn't understand what to remember")
                return response

            memory_managers = hass.data.get(DOMAIN, {}).get("memory_managers", {})
            if not memory_managers:
                response = intent_obj.create_response()
                response.async_set_speech("No memories are available")
                return response

            target_manager = None

            # 1. Check shared parameter
            # If shared is explicitly True, target common memory
            if shared_slot:
                if "common" in memory_managers:
                    target_manager = memory_managers["common"]
                else:
                    response = intent_obj.create_response()
                    response.async_set_speech("Shared memory is not available")
                    return response

            # 2. Default to private memory (shared is False or missing)
            else:
                # Try to find memory for the calling agent
                agent_id = getattr(intent_obj, "conversation_agent_id", None)

                # DEBUG: Dump intent object to find where agent info is
                _LOGGER.debug(f"DEBUG: intent_obj dir: {dir(intent_obj)}")
                _LOGGER.debug(f"DEBUG: intent_obj vars: {vars(intent_obj) if hasattr(intent_obj, '__dict__') else 'No __dict__'}")
                _LOGGER.debug(f"DEBUG: intent_obj context: {intent_obj.context if hasattr(intent_obj, 'context') else 'No context'}")
                _LOGGER.debug(f"DEBUG: intent_obj device_id: {intent_obj.device_id}")
                _LOGGER.debug(f"DEBUG: intent_obj assistant: {intent_obj.assistant}")
                _LOGGER.debug(f"DEBUG: intent_obj platform: {intent_obj.platform}")
                _LOGGER.debug(f"DEBUG: intent_obj satellite_id: {intent_obj.satellite_id}")

                if intent_obj.context:
                    _LOGGER.debug(f"DEBUG: context user_id: {intent_obj.context.user_id}")
                # Fallback: Try to resolve agent_id from platform using Entity Registry
                if not agent_id and hasattr(intent_obj, "platform") and intent_obj.platform:
                    from homeassistant.helpers import entity_registry as er
                    ent_reg = er.async_get(hass)
                    
                    # Look for conversation entities with matching platform
                    matching_entities = [
                        entry for entry in ent_reg.entities.values()
                        if entry.domain == "conversation" and entry.platform == intent_obj.platform
                    ]
                    
                    if len(matching_entities) == 1:
                        agent_id = matching_entities[0].entity_id
                        _LOGGER.debug(f"Resolved platform '{intent_obj.platform}' to agent_id '{agent_id}' via Entity Registry")
                    elif len(matching_entities) > 1:
                        # Ambiguous: multiple agents match the platform.
                        # Filter by agents that actually have a private memory configured.
                        candidates = [
                            e for e in matching_entities 
                            if any(m.agent_id == e.entity_id for m in memory_managers.values())
                        ]
                        
                        if len(candidates) == 1:
                            agent_id = candidates[0].entity_id
                            _LOGGER.debug(f"Resolved ambiguous platform '{intent_obj.platform}' to agent_id '{agent_id}' (only one with memory)")
                        elif len(candidates) > 1:
                            _LOGGER.warning(
                                f"Ambiguous platform '{intent_obj.platform}': found {len(candidates)} matching agents with memory "
                                f"({[e.entity_id for e in candidates]}). Cannot resolve default memory."
                            )
                        else:
                             _LOGGER.debug(f"Ambiguous platform '{intent_obj.platform}': no matching agents have memory configured.")
                    else:
                        _LOGGER.debug(f"No agent found for platform '{intent_obj.platform}' in Entity Registry")

                if agent_id:
                    for manager in memory_managers.values():
                        if manager.agent_id == agent_id:
                            target_manager = manager
                            break
                
                # Fallback: If no private memory found, fail?
                # User said "default kişisel olan memoryye eklesin" (default to private)
                # If private doesn't exist, we should probably inform the user.
                if not target_manager:
                    _LOGGER.warning(f"AddMemory: Private memory not found for agent_id='{agent_id}' (platform='{getattr(intent_obj, 'platform', 'N/A')}'). Available managers: {[(m.memory_id, m.agent_id) for m in memory_managers.values()]}")
                    response = intent_obj.create_response()
                    response.async_set_speech("I couldn't find your private memory")
                    return response

            if target_manager:
                await target_manager.async_add_memory(text)
                response = intent_obj.create_response()
                response.async_set_speech(f"Added to {target_manager.memory_name}")
                return response

            response = intent_obj.create_response()
            response.async_set_speech("I couldn't find a suitable memory to add to")
            return response

    # Register the intent
    intent.async_register(hass, AddMemoryIntent())


def get_memory_context_for_llm(manager) -> Optional[str]:
    """Get formatted memory context for any LLM."""

    if not manager._memories:
        return None

    # Format memories
    memory_entries = []
    for mem in manager._memories[-20:]:  # Last 20 entries for context limit
        memory_entries.append(f"[{mem['date']}] {mem['text']}")

    full_text = "\n".join(memory_entries)

    description_text = f"\nDescription: {manager.description}" if manager.description else ""

    context = (
        f"\n## 📝 LONG-TERM MEMORY: {manager.memory_name}{description_text}\n"
        f"This memory bank contains {len(manager._memories)} entries. "
        f"Showing the most recent {min(len(manager._memories), 20)}:\n\n"
        f"{full_text}\n\n"
    )

    return context


def get_all_memory_contexts(hass: HomeAssistant) -> str:
    """Get all memory contexts combined for conversation agents."""

    memory_contexts = hass.data.get(DOMAIN, {}).get("conversation_contexts", {})

    if not memory_contexts:
        return ""

    all_contexts = []

    for memory_id, context_data in memory_contexts.items():
        get_context = context_data.get("get_context")
        if get_context:
            context = get_context()
            if context:
                all_contexts.append(context)

    if not all_contexts:
        return ""

    combined = (
            "\n# 🧠 AVAILABLE LONG-TERM MEMORIES\n"
            + "\n---\n".join(all_contexts)
            + "\n---\n"
    )

    return combined


def get_memory_prompt_injection(hass: HomeAssistant, memory_ids: list = None) -> str:
    """
    Get memory context for prompt injection.
    Use this in your conversation agent's system prompt.

    Args:
        hass: Home Assistant instance
        memory_ids: List of specific memory IDs to include (None = all)

    Returns:
        Formatted memory context string
    """

    memory_contexts = hass.data.get(DOMAIN, {}).get("conversation_contexts", {})

    if not memory_contexts:
        return ""

    contexts_to_include = []

    for memory_id, context_data in memory_contexts.items():
        # Filter by memory_ids if specified
        if memory_ids and memory_id not in memory_ids:
            continue

        get_context = context_data.get("get_context")
        if get_context:
            context = get_context()
            if context:
                contexts_to_include.append(context)

    if not contexts_to_include:
        return ""

    return "\n".join(contexts_to_include)


async def async_unregister_from_conversation(hass: HomeAssistant, entry_id: str):
    """Unregister memory from conversation agents."""

    memory_managers = hass.data[DOMAIN].get("memory_managers", {})
    manager = memory_managers.get(entry_id)

    if not manager:
        return

    # Remove from conversation contexts
    conversation_contexts = hass.data.get(DOMAIN, {}).get("conversation_contexts", {})

    if manager.memory_id in conversation_contexts:
        conversation_contexts.pop(manager.memory_id)
        _LOGGER.debug(f"Unregistered memory '{manager.memory_name}' from conversation agents")


def create_template_helper(hass: HomeAssistant) -> None:
    """Create Jinja2 template helpers for easy use in prompts."""

    def get_memory_context(memory_id: str = None) -> str:
        """Template helper to get memory context."""
        if memory_id:
            memory_contexts = hass.data.get(DOMAIN, {}).get("conversation_contexts", {})
            context_data = memory_contexts.get(memory_id)
            if context_data:
                get_context = context_data.get("get_context")
                if get_context:
                    return get_context() or ""
        else:
            # Return all memories
            return get_all_memory_contexts(hass)
        return ""

    def list_available_memories() -> list:
        """Template helper to list available memory IDs."""
        memory_contexts = hass.data.get(DOMAIN, {}).get("conversation_contexts", {})
        return list(memory_contexts.keys())

    # Register as template functions in domain data
    if "template_helpers" not in hass.data.get(DOMAIN, {}):
        if DOMAIN not in hass.data:
            hass.data[DOMAIN] = {}
        hass.data[DOMAIN]["template_helpers"] = {
            "get_memory": get_memory_context,
            "list_memories": list_available_memories
        }
        _LOGGER.debug("Template helpers registered for AI Memory")
