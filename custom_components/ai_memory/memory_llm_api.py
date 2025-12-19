"""LLM API for Memory Management."""
import logging

import homeassistant.helpers.llm as llm
import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.util.json import JsonObjectType

from .constants import DOMAIN
from .utils import format_date

_LOGGER = logging.getLogger(__name__)

API_ID = "memory_api"


async def async_setup(hass: HomeAssistant):
    """Set up the Memory LLM API."""
    try:
        llm.async_register_api(hass, MemoryAPI(hass))
    except Exception as e:
        # Ignore if already registered (e.g. during reload)
        _LOGGER.debug("Memory LLM API registration skipped: %s", e)


class AddMemoryTool(llm.Tool):
    """Tool to add information to memory."""

    name = "add_memory"
    description = ""

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
        agent_id = llm_context.platform
        if scope == "private" and not agent_id:
            # Fallback for testing or non-agent calls
            agent_id = "unknown_agent"

        try:
            await self.memory_manager.async_add_memory(content, scope, agent_id)
            _LOGGER.debug(f"Saved to {scope} ({agent_id}) memory: {content}")
            return {"success": True, "message": f"Saved to {scope} ({agent_id}) memory."}
        except Exception as e:
            _LOGGER.error(f"Error adding memory: {e}")
            raise Exception(f"Error: {e}")


class SearchMemoryTool(llm.Tool):
    """Tool to search memory."""

    name = "search_memory"
    description = ""

    parameters = vol.Schema({
        vol.Required("query"): str,
    })

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def async_call(
            self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> JsonObjectType:
        query = tool_input.tool_args.get("query")
        agent_id = llm_context.platform
        _LOGGER.debug(f"AI Memory (search_memory): {query}")

        try:
            results = await self.memory_manager.async_search_memory(query, agent_id)
            if not results:
                return {"success": True, "message": "No matching memories found."}

            formatted = ""
            for memory in results:
                formatted += f"[{format_date(memory["created_at"])}] {memory["content"]}"

            return {
                "success": True,
                "results": formatted
            }
        except Exception as e:
            _LOGGER.error(f"Error searching memory: {e}")
            raise Exception(f"Error: {e}")


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
            api_prompt="""You only use two tools: search_memory and add_memory.
GENERAL
- Decide only whether to SEARCH, ADD, or IGNORE memory
- Never simulate memory in conversation
SEARCH
- Call search_memory before relying on past context
- Use third-person perspective
- Resolve all relative time to absolute date (YYYY-MM-DD)
- Use short, keyword-based queries
ADD
- Call add_memory only for long-term, stable, reusable facts
- Never store temporary, emotional, or sensor-based information
WRITE RULES (MANDATORY)
- Third-person only, no "I / me / my"
- Use the user’s name if known, otherwise "The user"
- Always resolve relative time to absolute date
- Replace vague references with explicit nouns
- Write concise, factual statements
- No quotes, no conversational tone
SCOPE
- private: user preferences, habits, personal facts
- common: shared house or device facts only
Never claim memory unless it was retrieved via search_memory.
            """,
            llm_context=llm_context,
            tools=tools,
        )
