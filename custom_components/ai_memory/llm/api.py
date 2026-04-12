"""LLM API registration for AI Memory integration."""
import logging

import homeassistant.helpers.llm as llm
from homeassistant.core import HomeAssistant

from ..constants import DOMAIN
from .prompts import MEMORY_SYSTEM_PROMPT
from .tools import AddMemoryTool, SearchMemoryTool, DeleteMemoryTool

_LOGGER = logging.getLogger(__name__)

API_ID = "memory_api"


async def async_setup(hass: HomeAssistant):
    """Set up the Memory LLM API."""
    try:
        llm.async_register_api(hass, MemoryAPI(hass))
    except Exception as e:
        _LOGGER.debug("Memory LLM API registration skipped: %s", e)


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
            DeleteMemoryTool(manager),
        ]

        return llm.APIInstance(
            api=self,
            api_prompt=MEMORY_SYSTEM_PROMPT,
            llm_context=llm_context,
            tools=tools,
        )
