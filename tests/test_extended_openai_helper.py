"""Comprehensive tests for Extended OpenAI Helper."""
from unittest.mock import MagicMock, AsyncMock, patch

from homeassistant.core import HomeAssistant

from custom_components.ai_memory import DOMAIN
from custom_components.ai_memory.extended_openai_helper import (
    async_register_with_conversation,
    _register_conversation_context,
    _register_memory_intents,
    get_memory_context_for_llm,
    get_all_memory_contexts,
    get_memory_prompt_injection,
    async_unregister_from_conversation,
    create_template_helper,
)


async def test_register_with_conversation(hass: HomeAssistant, mock_conversation_agent):
    """Test registration with conversation agents."""
    manager = MagicMock()
    manager.memory_name = "Test Memory"
    manager.description = "Test Desc"

    await async_register_with_conversation(hass, manager)

    # Verify registration happened (by checking hass.data)
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
    hass.data[DOMAIN] = {
        "memory_managers": {
            "test_entry": MagicMock(memory_id="test_mem", memory_name="Test")
        },
        "conversation_contexts": {
            "test_mem": {"manager": MagicMock()}
        }
    }

    await async_unregister_from_conversation(hass, "test_entry")

    assert "test_mem" not in hass.data[DOMAIN]["conversation_contexts"]


def test_get_all_memory_contexts(hass: HomeAssistant):
    """Test getting all contexts."""
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
    create_template_helper(hass)

    assert "template_helpers" in hass.data[DOMAIN]
    helpers = hass.data[DOMAIN]["template_helpers"]
    assert "get_memory" in helpers
    assert "list_memories" in helpers


# Additional tests for improved coverage
async def test_register_with_conversation_no_manager(hass: HomeAssistant):
    """Test registration with no manager provided."""
    await async_register_with_conversation(hass, None)
    # Should not crash and should handle gracefully


async def test_register_with_conversation_context_error(hass: HomeAssistant):
    """Test registration when context registration fails."""
    manager = MagicMock()
    manager.memory_name = "Test Memory"

    with patch('custom_components.ai_memory.extended_openai_helper._register_conversation_context',
               side_effect=Exception("Context error")):
        await async_register_with_conversation(hass, manager)
        # Should handle error gracefully


async def test_register_with_conversation_intents_error(hass: HomeAssistant):
    """Test registration when intents registration fails."""
    manager = MagicMock()
    manager.memory_name = "Test Memory"

    with patch('custom_components.ai_memory.extended_openai_helper._register_memory_intents',
               side_effect=Exception("Intents error")):
        await async_register_with_conversation(hass, manager)
        # Should handle error gracefully


async def test_register_conversation_context(hass: HomeAssistant):
    """Test conversation context registration directly."""
    manager = MagicMock()
    manager.memory_id = "test_mem"

    await _register_conversation_context(hass, manager)

    assert DOMAIN in hass.data
    assert "conversation_contexts" in hass.data[DOMAIN]
    assert "test_mem" in hass.data[DOMAIN]["conversation_contexts"]


async def test_register_conversation_context_with_existing_domain(hass: HomeAssistant):
    """Test conversation context registration when domain already exists."""
    manager = MagicMock()
    manager.memory_id = "test_mem"

    # Pre-populate domain data
    hass.data[DOMAIN] = {"existing": "data"}

    await _register_conversation_context(hass, manager)

    assert DOMAIN in hass.data
    assert "conversation_contexts" in hass.data[DOMAIN]
    assert "test_mem" in hass.data[DOMAIN]["conversation_contexts"]


async def test_register_memory_intents_with_slots(hass: HomeAssistant):
    """Test intent registration with text slots."""
    manager = MagicMock()
    manager.memory_id = "test_memory"
    manager.memory_name = "Test Memory"
    manager.description = "Test Description"
    manager.async_add_memory = AsyncMock()

    await _register_memory_intents(hass, manager)

    # Verify intent was registered by checking it can handle an intent
    intent_type = f"Add{manager.memory_id.title()}Memory"

    # Mock intent with text slot
    mock_intent = MagicMock()
    mock_intent.slots = {"text": {"value": "Remember this"}}
    mock_intent.create_response = MagicMock(return_value={"speech": {"plain": {"speech": "Added"}}})

    # Find and call the intent handler
    with patch('homeassistant.helpers.intent.async_register') as mock_register:
        await _register_memory_intents(hass, manager)
        mock_register.assert_called_once()


def test_get_memory_context_for_llm_empty_memories():
    """Test context generation with empty memories."""
    manager = MagicMock()
    manager.memory_name = "Test Memory"
    manager.description = "Test Desc"
    manager._memories = []

    context = get_memory_context_for_llm(manager)
    assert context is None


def test_get_memory_context_for_llm_no_description():
    """Test context generation without description."""
    manager = MagicMock()
    manager.memory_name = "Test Memory"
    manager.description = None
    manager._memories = [
        {"date": "2023-01-01", "text": "Fact 1"}
    ]

    context = get_memory_context_for_llm(manager)
    assert "Test Memory" in context
    assert "Fact 1" in context
    assert "Description:" not in context


def test_get_memory_context_for_llm_many_memories():
    """Test context generation with more than 20 memories (should limit)."""
    manager = MagicMock()
    manager.memory_name = "Test Memory"
    manager.description = "Test Desc"

    # Create 30 memories
    manager._memories = [
        {"date": f"2023-01-{i:02d}", "text": f"Memory {i}"}
        for i in range(1, 31)
    ]

    context = get_memory_context_for_llm(manager)

    # Should only include last 20 memories
    assert "Memory 30" in context
    assert "Memory 11" in context
    assert "Memory 10" not in context  # Should be outside the limit
    assert "showing the most recent 20" in context.lower()


def test_get_all_memory_contexts_no_contexts(hass: HomeAssistant):
    """Test getting all contexts when no contexts exist."""
    hass.data[DOMAIN] = {"conversation_contexts": {}}

    result = get_all_memory_contexts(hass)
    assert result == ""


def test_get_all_memory_contexts_no_domain(hass: HomeAssistant):
    """Test getting all contexts when domain doesn't exist."""
    result = get_all_memory_contexts(hass)
    assert result == ""


def test_get_all_memory_contexts_empty_contexts(hass: HomeAssistant):
    """Test getting all contexts when all contexts return empty."""
    hass.data[DOMAIN] = {
        "conversation_contexts": {
            "mem1": {"get_context": lambda: None},
            "mem2": {"get_context": lambda: ""},
        }
    }

    result = get_all_memory_contexts(hass)
    assert result == ""


def test_get_all_memory_contexts_with_get_context_none(hass: HomeAssistant):
    """Test get_all_memory_contexts when get_context returns None."""
    hass.data[DOMAIN] = {
        "conversation_contexts": {
            "mem1": {"get_context": MagicMock(return_value=None)},
            "mem2": {"get_context": MagicMock(return_value=None)},
        }
    }

    result = get_all_memory_contexts(hass)
    assert result == ""


def test_get_memory_prompt_injection_no_domain(hass: HomeAssistant):
    """Test prompt injection when domain doesn't exist."""
    result = get_memory_prompt_injection(hass)
    assert result == ""


def test_get_memory_prompt_injection_no_contexts(hass: HomeAssistant):
    """Test prompt injection when no contexts exist."""
    hass.data[DOMAIN] = {"conversation_contexts": {}}

    result = get_memory_prompt_injection(hass)
    assert result == ""


def test_get_memory_prompt_injection_empty_contexts(hass: HomeAssistant):
    """Test prompt injection when contexts return empty."""
    hass.data[DOMAIN] = {
        "conversation_contexts": {
            "mem1": {"get_context": lambda: None},
            "mem2": {"get_context": lambda: ""},
        }
    }

    result = get_memory_prompt_injection(hass)
    assert result == ""


def test_get_memory_prompt_injection_get_context_none(hass: HomeAssistant):
    """Test get_memory_prompt_injection when get_context returns None."""
    hass.data[DOMAIN] = {
        "conversation_contexts": {
            "mem1": {"get_context": MagicMock(return_value=None)},
        }
    }

    result = get_memory_prompt_injection(hass)
    assert result == ""


async def test_unregister_from_conversation_no_manager(hass: HomeAssistant):
    """Test unregistering when manager doesn't exist."""
    hass.data[DOMAIN] = {"memory_managers": {}}

    await async_unregister_from_conversation(hass, "nonexistent")
    # Should not crash


async def test_unregister_from_conversation_no_contexts(hass: HomeAssistant):
    """Test unregistering when no conversation contexts exist."""
    manager = MagicMock()
    manager.memory_id = "test_mem"

    hass.data[DOMAIN] = {
        "memory_managers": {"test_entry": manager},
        "conversation_contexts": {}
    }

    await async_unregister_from_conversation(hass, "test_entry")
    # Should not crash


def test_create_template_helper_existing_helpers(hass: HomeAssistant):
    """Test template helper creation when helpers already exist."""
    hass.data[DOMAIN] = {"template_helpers": {"existing": "helper"}}

    create_template_helper(hass)

    # Should not overwrite existing helpers
    assert hass.data[DOMAIN]["template_helpers"]["existing"] == "helper"


def test_create_template_helper_no_domain(hass: HomeAssistant):
    """Test template helper creation when domain doesn't exist."""
    create_template_helper(hass)

    assert DOMAIN in hass.data
    assert "template_helpers" in hass.data[DOMAIN]


def test_template_helper_get_memory_context_specific(hass: HomeAssistant):
    """Test template helper for specific memory context."""
    # Setup domain data without template_helpers
    hass.data[DOMAIN] = {
        "conversation_contexts": {
            "test_mem": {
                "get_context": lambda: "Test Context"
            }
        }
    }

    create_template_helper(hass)

    helpers = hass.data[DOMAIN]["template_helpers"]
    result = helpers["get_memory"]("test_mem")

    assert result == "Test Context"


def test_template_helper_get_memory_context_all(hass: HomeAssistant):
    """Test template helper for all memory contexts."""
    # Setup domain data
    hass.data[DOMAIN] = {
        "conversation_contexts": {
            "mem1": {"get_context": lambda: "Context 1"},
            "mem2": {"get_context": lambda: "Context 2"}
        }
    }

    create_template_helper(hass)

    helpers = hass.data[DOMAIN]["template_helpers"]
    result = helpers["get_memory"]()

    assert "Context 1" in result
    assert "Context 2" in result


def test_template_helper_get_memory_context_nonexistent(hass: HomeAssistant):
    """Test template helper for nonexistent memory."""
    # Setup domain data
    hass.data[DOMAIN] = {"conversation_contexts": {}}

    create_template_helper(hass)

    helpers = hass.data[DOMAIN]["template_helpers"]
    result = helpers["get_memory"]("nonexistent")

    assert result == ""


def test_template_helper_get_memory_context_no_get_context(hass: HomeAssistant):
    """Test template helper when context has no get_context function."""
    # Setup domain data
    hass.data[DOMAIN] = {
        "conversation_contexts": {
            "test_mem": {"no_get_context": lambda: "Should not be called"}
        }
    }

    create_template_helper(hass)

    helpers = hass.data[DOMAIN]["template_helpers"]
    result = helpers["get_memory"]("test_mem")

    assert result == ""


def test_template_helper_list_available_memories(hass: HomeAssistant):
    """Test template helper for listing available memories."""
    # Setup domain data
    hass.data[DOMAIN] = {
        "conversation_contexts": {
            "mem1": {"get_context": lambda: "Context 1"},
            "mem2": {"get_context": lambda: "Context 2"},
            "mem3": {"get_context": lambda: "Context 3"}
        }
    }

    create_template_helper(hass)

    helpers = hass.data[DOMAIN]["template_helpers"]
    result = helpers["list_memories"]()

    assert set(result) == {"mem1", "mem2", "mem3"}
