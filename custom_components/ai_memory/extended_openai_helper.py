"""Helper for conversation agent integration."""
import logging
from typing import Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent

_LOGGER = logging.getLogger(__name__)

DOMAIN = "ai_memory"


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
        await _register_memory_intents(hass, manager)
        _LOGGER.debug(f"Registered conversation intents for '{manager.memory_name}'")
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


async def _register_memory_intents(hass: HomeAssistant, manager):
    """Register intents for this memory."""

    # Create custom intents for this memory
    class MemoryAddIntent(intent.IntentHandler):
        """Handle adding to specific memory."""
        intent_type = f"Add{manager.memory_id.title()}Memory"
        description = manager.description or f"Add to {manager.memory_name} memory"

        async def async_handle(self, intent_obj: intent.Intent):
            text = intent_obj.slots.get("text", {}).get("value", "")
            if text:
                await manager.async_add_memory(text)
                return intent_obj.create_response(
                    speech={"plain": {"speech": f"Added to {manager.memory_name} memory"}}
                )
            return intent_obj.create_response(
                speech={"plain": {"speech": "I didn't understand what to remember"}}
            )

    # Register the intent
    intent.async_register(hass, MemoryAddIntent())


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
        "Use these established facts and preferences naturally in your responses.\n"
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
            "You have access to these persistent memories from past conversations:\n"
            + "\n---\n".join(all_contexts) +
            "\n---\n"
            "Remember: Reference these memories naturally when relevant to the conversation.\n"
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
