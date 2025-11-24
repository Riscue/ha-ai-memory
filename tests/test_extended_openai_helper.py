"""Test Extended OpenAI Helper."""
from unittest.mock import MagicMock

from homeassistant.core import HomeAssistant

from custom_components.ai_memory.extended_openai_helper import (
    async_register_with_conversation,
    get_memory_context_for_llm,
)


async def test_register_with_conversation(hass: HomeAssistant, mock_conversation_agent):
    """Test registration with conversation agents."""
    manager = MagicMock()
    manager.memory_name = "Test Memory"
    manager.description = "Test Desc"

    await async_register_with_conversation(hass, manager)

    # Verify registration happened (by checking hass.data)
    from custom_components.ai_memory import DOMAIN
    assert DOMAIN in hass.data
    assert "conversation_contexts" in hass.data[DOMAIN]


def test_get_memory_context_for_llm():
    """Test context generation."""
    manager = MagicMock()
    manager.memory_name = "Test Memory"
    manager.description = "Test Desc"
    manager._memories = [
        {"date": "2023-01-01", "text": "Fact 1"},
        {"date": "2023-01-02", "text": "Fact 2"},
    ]

    context = get_memory_context_for_llm(manager)

    assert "Test Memory" in context
    assert "Test Desc" in context
    assert "Fact 1" in context
    assert "Fact 2" in context
    assert "LONG-TERM MEMORY" in context


async def test_unregister_from_conversation(hass: HomeAssistant):
    """Test unregistering memory."""
    # Setup initial state
    from custom_components.ai_memory import DOMAIN
    hass.data[DOMAIN] = {
        "memory_managers": {
            "test_entry": MagicMock(memory_id="test_mem", memory_name="Test")
        },
        "conversation_contexts": {
            "test_mem": {"manager": MagicMock()}
        }
    }

    from custom_components.ai_memory.extended_openai_helper import async_unregister_from_conversation

    await async_unregister_from_conversation(hass, "test_entry")

    assert "test_mem" not in hass.data[DOMAIN]["conversation_contexts"]


def test_get_all_memory_contexts(hass: HomeAssistant):
    """Test getting all contexts."""
    from custom_components.ai_memory import DOMAIN
    from custom_components.ai_memory.extended_openai_helper import get_all_memory_contexts

    # Mock contexts
    hass.data[DOMAIN] = {
        "conversation_contexts": {
            "mem1": {"get_context": lambda: "Context 1"},
            "mem2": {"get_context": lambda: "Context 2"},
        }
    }

    result = get_all_memory_contexts(hass)
    assert "Context 1" in result
    assert "Context 2" in result
    assert "AVAILABLE LONG-TERM MEMORIES" in result


def test_get_memory_prompt_injection(hass: HomeAssistant):
    """Test prompt injection helper."""
    from custom_components.ai_memory import DOMAIN
    from custom_components.ai_memory.extended_openai_helper import get_memory_prompt_injection

    hass.data[DOMAIN] = {
        "conversation_contexts": {
            "mem1": {"get_context": lambda: "Context 1"},
            "mem2": {"get_context": lambda: "Context 2"},
        }
    }

    # Test all
    result = get_memory_prompt_injection(hass)
    assert "Context 1" in result
    assert "Context 2" in result

    # Test specific
    result_specific = get_memory_prompt_injection(hass, ["mem1"])
    assert "Context 1" in result_specific
    assert "Context 2" not in result_specific


def test_create_template_helper(hass: HomeAssistant):
    """Test template helper creation."""
    from custom_components.ai_memory import DOMAIN
    from custom_components.ai_memory.extended_openai_helper import create_template_helper

    create_template_helper(hass)

    assert "template_helpers" in hass.data[DOMAIN]
    helpers = hass.data[DOMAIN]["template_helpers"]
    assert "get_memory" in helpers
    assert "list_memories" in helpers
