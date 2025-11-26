"""Debug utility to list all registered LLM APIs and their tools."""
import logging

import homeassistant.helpers.llm as llm
from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


async def async_debug_list_all_apis_and_tools(hass: HomeAssistant):
    """List all registered LLM APIs and their tools for debugging."""
    _LOGGER.info("=" * 80)
    _LOGGER.info("DEBUG: Listing all LLM APIs and Tools")
    _LOGGER.info("=" * 80)

    try:
        apis = llm.async_get_apis(hass)
        _LOGGER.info(f"Total registered APIs: {len(apis)}")

        for api in apis:
            _LOGGER.info(f"\nAPI: {api.id}")
            _LOGGER.info(f"  Name: {api.name}")
            _LOGGER.info(f"  Type: {type(api).__name__}")

            # Try to get an instance to see the tools
            try:
                # Create a dummy context to get the instance
                from homeassistant.helpers.llm import LLMContext
                context = LLMContext(
                    platform="test",
                    context=None,
                    language="en",
                    assistant="test_assistant",
                    device_id=None,
                )

                instance = await api.async_get_api_instance(context)
                if instance and instance.tools:
                    _LOGGER.info(f"  Tools ({len(instance.tools)}):")
                    for tool in instance.tools:
                        _LOGGER.info(f"    - {tool.name}: {tool.description}")
                else:
                    _LOGGER.info("  Tools: None or empty")
            except Exception as e:
                _LOGGER.warning(f"  Could not get API instance for {api.id}: {e}")

    except Exception as e:
        _LOGGER.error(f"Error listing APIs and tools: {e}", exc_info=True)

    _LOGGER.info("=" * 80)
