"""LLM Tool definitions for AI Memory integration."""
import logging

import homeassistant.helpers.llm as llm
import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.util.json import JsonObjectType

from ..utils import format_date

_LOGGER = logging.getLogger(__name__)


class AddMemoryTool(llm.Tool):
    """Tool to add information to memory."""

    name = "add_memory"
    description = "Save information to long-term memory"

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
    ) -> JsonObjectType:
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
            return {"success": True, "message": f"Saved to {scope} ({agent_id}) memory."}
        except Exception as e:
            _LOGGER.error("Error adding memory: %s", e)
            raise


class SearchMemoryTool(llm.Tool):
    """Tool to search memory."""

    name = "search_memory"
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
    ) -> JsonObjectType:
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
                return {"success": True, "message": "No matching memories found."}

            formatted = ""
            for memory in results:
                formatted += f"[{format_date(memory['created_at'])}] {memory['content']}\n"

            return {
                "success": True,
                "results": formatted.strip(),
            }
        except Exception as e:
            _LOGGER.error("Error searching memory: %s", e)
            raise


class DeleteMemoryTool(llm.Tool):
    """Tool to delete a specific memory."""

    name = "delete_memory"
    description = "Delete a specific memory by ID"

    parameters = vol.Schema({
        vol.Required("memory_id"): str,
    })

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        memory_id = tool_input.tool_args.get("memory_id")
        agent_id = llm_context.platform

        try:
            result = await self.memory_manager.async_delete_memory(memory_id, agent_id)
            if result:
                return {"success": True, "message": f"Memory {memory_id} deleted."}
            else:
                return {"success": False, "message": "Memory not found or not authorized."}
        except Exception as e:
            _LOGGER.error("Error deleting memory: %s", e)
            raise
