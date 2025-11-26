"""LLM API for Memory Management."""
import logging

import homeassistant.helpers.llm as llm
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
    description = """
    CRITICAL: Use this tool PROACTIVELY. Do not wait for the user to say "save this".
    If the user mentions a personal fact (e.g., "I'm a fan of Fenerbahçe", "My name is Ebru"), a future plan, a preference, or a specific rule for the house, save it IMMEDIATELY.
    - Use 'private' scope (default) for personal details to build a unique bond with the user.
    - Use 'common' scope ONLY if the information is a general fact about the house (e.g., "The garage door is broken") that all assistants must know.
    """

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
        _LOGGER.debug(f"AI Memory (search_memory): {scope} - {content}")

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
    description = """
    Use this tool whenever the user refers to past events, asks a question requiring personal context, or uses vague references like "it", "that thing", or "my team".
    BEFORE answering a personal question (e.g., "What was my plan?", "Do you remember me?"), ALWAYS search memory first.
    This tool searches both your 'private' memories and the 'common' house knowledge base.
    """

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
        _LOGGER.debug(f"AI Memory (search_memory): {query}")

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
