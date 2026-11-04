"""LLM Tool definitions for AI Memory integration."""
import logging

import homeassistant.helpers.llm as llm
import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.util.json import JsonObjectType

from ..constants import DOMAIN
from ..utils import format_date

_LOGGER = logging.getLogger(__name__)

# ToolResult was added in HA 2026.10; older versions expect a plain JSON object.
_ToolResult = getattr(llm, "ToolResult", None)


def _result(data: JsonObjectType, error: bool = False):
    """Wrap a tool response in a ToolResult when Home Assistant supports it."""
    if _ToolResult is None:
        return data
    return _ToolResult(data=data, error=error)


class AddMemoryTool(llm.Tool):
    """Tool to add information to memory."""

    name = "add_memory"
    integration = DOMAIN
    description = (
        "Save information to long-term memory."
        "Use 'room' to group related memories (e.g., 'camera_events', 'user_preferences', 'home_status')."
        "Use 'wing' for broader categories (e.g., 'security', 'automation', 'settings')."
    )

    parameters = vol.Schema({
        vol.Required("content"): str,
        vol.Required("scope"): vol.In(["private", "common"]),
        vol.Optional("summary", default=""): str,
        vol.Optional("wing", default=""): str,
        vol.Optional("room", default=""): str,
    })

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> "llm.ToolResult | JsonObjectType":
        content = tool_input.tool_args.get("content")
        scope = tool_input.tool_args.get("scope")
        summary = tool_input.tool_args.get("summary", "") or None
        wing = tool_input.tool_args.get("wing", "") or None
        room = tool_input.tool_args.get("room", "") or None

        # Determine agent_id
        agent_id = llm_context.platform
        if scope == "private" and not agent_id:
            agent_id = "unknown_agent"

        try:
            await self.memory_manager.async_add_memory(
                content, scope, agent_id,
                summary=summary, wing=wing, room=room,
            )
            _LOGGER.debug("Saved to %s (%s) memory: %s", scope, agent_id, content[:50])
            return _result({"success": True, "message": f"Saved to {scope} ({agent_id}) memory."})
        except Exception as e:
            _LOGGER.error("Error adding memory: %s", e)
            raise


class SearchMemoryTool(llm.Tool):
    """Tool to search memory."""

    name = "search_memory"
    integration = DOMAIN
    description = "Search long-term memory for relevant information"

    parameters = vol.Schema({
        vol.Required("query"): str,
        vol.Optional("wing", default=""): str,
        vol.Optional("room", default=""): str,
        vol.Optional("limit", default=5): vol.All(int, vol.Range(min=1, max=20)),
    })

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> "llm.ToolResult | JsonObjectType":
        query = tool_input.tool_args.get("query")
        wing = tool_input.tool_args.get("wing", "") or None
        room = tool_input.tool_args.get("room", "") or None
        limit = tool_input.tool_args.get("limit", 5)
        agent_id = llm_context.platform
        _LOGGER.debug("AI Memory (search_memory): %s", query)

        try:
            results = await self.memory_manager.async_search_memory(
                query, agent_id, limit=limit, wing=wing, room=room,
            )
            if not results:
                return _result({"success": True, "message": "No matching memories found."})

            formatted = ""
            for memory in results:
                formatted += f"[{format_date(memory['created_at'])}] {memory['content']}\n"

            return _result({
                "success": True,
                "results": formatted.strip(),
            })
        except Exception as e:
            _LOGGER.error("Error searching memory: %s", e)
            raise


class DeleteMemoryTool(llm.Tool):
    """Tool to delete memories by room, or wing."""

    name = "delete_memory"
    integration = DOMAIN
    description = (
        "Delete memories by room, wing, or scope. "
        "At least one filter (room, wing, or scope) should be provided."
    )

    parameters = vol.Schema({
        vol.Optional("room"): str,
        vol.Optional("wing"): str,
        vol.Optional("scope"): vol.In(["private", "common"]),
    })

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> "llm.ToolResult | JsonObjectType":
        room = tool_input.tool_args.get("room", "") or None
        wing = tool_input.tool_args.get("wing", "") or None
        scope = tool_input.tool_args.get("scope", "") or None
        agent_id = llm_context.platform

        if not any([room, wing, scope]):
            return _result(
                {"success": False, "message": "You must provide at least one filter (room, wing, or scope)."},
                error=True,
            )

        try:
            count = await self.memory_manager.async_delete_memory(
                agent_id=agent_id,
                room=room,
                wing=wing,
                scope=scope,
            )

            return _result({
                "success": True,
                "message": f"Deleted {count} memor(y/ies) from {room or wing or scope or 'all'}."
            })
        except Exception as e:
            _LOGGER.error("Error deleting memory: %s", e)
            raise
