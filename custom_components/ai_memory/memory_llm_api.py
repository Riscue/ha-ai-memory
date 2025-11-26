"""LLM API for Memory Management."""
import logging

import homeassistant.components.llm as llm
import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.util.json import JsonObjectType

from .constants import DOMAIN

_LOGGER = logging.getLogger(__name__)

API_ID = "memory_api"


async def async_setup(hass: HomeAssistant):
    """Set up the Memory LLM API."""
    llm.async_register_api(hass, MemoryAPI(hass))


class AddMemoryTool(llm.Tool):
    """Tool to add information to memory."""

    name = "add_memory"
    description = "Add information to long-term memory. Use 'private' scope for user-specific facts, 'common' for general facts shared with other assistants."
    parameters = vol.Schema({
        vol.Required("content"): str,
        vol.Optional("scope", default="private"): vol.In(["private", "common"]),
    })

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def async_call(
            self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        content = tool_input.tool_args.get("content")
        scope = tool_input.tool_args.get("scope", "private")

        # Determine agent_id
        agent_id = llm_context.assistant
        if scope == "private" and not agent_id:
            # Fallback for testing or non-agent calls
            agent_id = "unknown_agent"

        try:
            await self.memory_manager.async_add_memory(content, scope, agent_id)
            return {"success": True, "message": f"Saved to {scope} memory."}
        except Exception as e:
            _LOGGER.error(f"Error adding memory: {e}")
            raise llm.ToolError(f"Error: {e}")


class SearchMemoryTool(llm.Tool):
    """Tool to search memory."""

    name = "search_memory"
    description = "Search through long-term memory (both private and common)."
    parameters = vol.Schema({
        vol.Required("query"): str,
    })

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def async_call(
            self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        query = tool_input.tool_args.get("query")
        agent_id = llm_context.assistant

        try:
            results = await self.memory_manager.async_search_memory(query, agent_id)
            if not results:
                return {"success": True, "message": "No matching memories found."}

            formatted = "\n".join([
                f"- {r['content']} (Scope: {r['metadata']['scope']})"
                for r in results
            ])
            return {
                "success": True,
                "results": formatted
            }
        except Exception as e:
            _LOGGER.error(f"Error searching memory: {e}")
            raise llm.ToolError(f"Error: {e}")


class MemoryAPI(llm.API):
    """Memory Management LLM API."""

    def __init__(self, hass: HomeAssistant) -> None:
        super().__init__(hass=hass, id=API_ID, name="Memory Management")

    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        manager = self.hass.data.get(DOMAIN, {}).get("manager")

        if not manager:
            _LOGGER.error("Memory Manager not initialized")
            return llm.APIInstance(self, "Error: Memory system unavailable", llm_context, [])

        tools = [
            AddMemoryTool(manager),
            SearchMemoryTool(manager),
        ]

        return llm.APIInstance(
            api=self,
            api_prompt="Use tools to manage memory. Prefer 'private' scope unless asked to share.",
            llm_context=llm_context,
            tools=tools,
        )
