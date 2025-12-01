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
    description = """
    CRITICAL: You are a memory recorder. Your goal is to save facts for LONG-TERM retrieval.
    Do NOT save exact quotes. Instead, REWRITE the information into a self-contained fact following these rules:
    
    1. PERSPECTIVE NORMALIZATION:
      - Never use "I", "me", "my" for the user.
      - Always convert first-person statements to third-person (e.g., "I like coffee" -> "The user likes coffee").
      - If the user's name is known, use the name explicitly instead of "The User".
    
    2. TIME RESOLUTION:
      - Never save relative time words like "tomorrow", "next week", "yesterday".
      - ALWAYS calculate and write the ABSOLUTE DATE (YYYY-MM-DD) based on the current date provided in your system prompt.
      - Example: If today is 2025-11-30 and user says "tomorrow", save it as "2025-12-01".
    
    3. CONTEXT RESOLUTION (CRITICAL):
      - The memory must make sense in isolation 6 months from now.
      - Replace vague pronouns like "it", "that", "there" with specific nouns.
      - Bad: "The user wants to change it."
      - Good: "The user wants to change the kitchen light bulb."
    
    4. SCOPE:
      - Use 'private' for personal preferences, plans, and facts about the specific user.
      - Use 'common' ONLY for general facts about the house/devices shared by everyone.
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
    description = """
    CRITICAL: Use this tool PROACTIVELY for context retrieval.
    Your goal is to generate a search query that matches how facts are stored in the database.
    
    1. PERSPECTIVE NORMALIZATION:
      - Internalize the query. Never use "I", "me", "my".
      - Convert first-person questions into third-person statements.
      - USE THE USER'S NAME if available.
    
    2. TIME RESOLUTION:
      - Never search for relative terms like "tomorrow", "yesterday".
      - ALWAYS calculate the ABSOLUTE DATE (YYYY-MM-DD) based on the current system time.
      - Example: User asks "What is the plan for tomorrow?" -> You search: "Plan for 2025-12-01".
    
    3. KEYWORD OPTIMIZATION:
      - Vector search works best with keywords, not long questions.
      - Instead of searching "What did the user say about the car?", search for "User's car preferences" or "Car status".
      - Strip unnecessary conversational filler words.
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
            api_prompt="Use tools to manage memory. Prefer 'private' scope unless asked to share.",
            llm_context=llm_context,
            tools=tools,
        )
