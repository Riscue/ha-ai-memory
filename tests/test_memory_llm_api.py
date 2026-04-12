"""Tests for LLM API and Tools."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.ai_memory.constants import DOMAIN

# Mock llm module before importing
mock_llm = MagicMock()


class MockTool:
    def __init__(self, **kwargs): pass


mock_llm.Tool = MockTool


class MockAPI:
    def __init__(self, hass=None, **kwargs):
        self.hass = hass


mock_llm.API = MockAPI


class MockToolInput:
    def __init__(self, tool_args):
        self.tool_args = tool_args


class MockLLMContext:
    def __init__(self, platform):
        self.platform = platform


class MockAPIInstance:
    def __init__(self, api, api_prompt, llm_context, tools):
        self.api = api
        self.api_prompt = api_prompt
        self.llm_context = llm_context
        self.tools = tools


mock_llm.ToolInput = MockToolInput
mock_llm.LLMContext = MockLLMContext
mock_llm.APIInstance = MockAPIInstance
mock_llm.ToolError = Exception

with patch.dict("sys.modules", {
    "homeassistant.components.llm": mock_llm,
    "homeassistant.helpers.llm": mock_llm
}):
    from custom_components.ai_memory.llm import api as llm_api
    from custom_components.ai_memory.llm.tools import AddMemoryTool, SearchMemoryTool, DeleteMemoryTool
    from custom_components.ai_memory import memory_llm_api

    import importlib
    importlib.reload(llm_api)


@pytest.fixture
def mock_manager():
    """Mock MemoryManager."""
    manager = AsyncMock()
    manager.async_add_memory = AsyncMock()
    manager.async_search_memory = AsyncMock(return_value=[])
    manager.async_delete_memory = AsyncMock(return_value=True)
    return manager


async def test_add_memory_tool(mock_manager):
    """Test AddMemoryTool with scope."""
    tool = AddMemoryTool(mock_manager)
    hass = MagicMock(spec=HomeAssistant)

    # Test Private (explicit)
    tool_input = MockToolInput({"content": "Test", "scope": "private"})
    llm_context = MockLLMContext("agent_1")

    await tool.async_call(hass, tool_input, llm_context)
    mock_manager.async_add_memory.assert_called_with(
        "Test", "private", "agent_1",
        summary=None, wing=None, room=None,
    )

    # Test Common
    tool_input = MockToolInput({"content": "Test", "scope": "common"})
    await tool.async_call(hass, tool_input, llm_context)
    mock_manager.async_add_memory.assert_called_with(
        "Test", "common", "agent_1",
        summary=None, wing=None, room=None,
    )


async def test_add_memory_tool_error(mock_manager):
    """Test AddMemoryTool error handling."""
    tool = AddMemoryTool(mock_manager)
    hass = MagicMock(spec=HomeAssistant)
    mock_manager.async_add_memory.side_effect = Exception("Save Error")

    tool_input = MockToolInput({"content": "Test"})
    llm_context = MockLLMContext("agent_1")

    with pytest.raises(Exception, match="Save Error"):
        await tool.async_call(hass, tool_input, llm_context)


async def test_search_memory_tool(mock_manager):
    """Test SearchMemoryTool passes agent_id."""
    tool = SearchMemoryTool(mock_manager)
    hass = MagicMock(spec=HomeAssistant)

    tool_input = MockToolInput({"query": "Test"})
    llm_context = MockLLMContext("agent_1")

    mock_manager.async_search_memory.return_value = [
        {"content": "Result 1", "score": 0.6, "scope": "private", "agent_id": "agent_1",
         "created_at": "2011-08-12T20:17:46.384Z"},
        {"content": "Result 2", "score": 0.3, "scope": "common", "agent_id": "",
         "created_at": "2011-08-12T20:17:46.384Z"},
    ]

    result = await tool.async_call(hass, tool_input, llm_context)
    assert result["success"] is True
    assert "Result 1" in result["results"]


async def test_search_memory_tool_no_results(mock_manager):
    """Test SearchMemoryTool with no results."""
    tool = SearchMemoryTool(mock_manager)
    hass = MagicMock(spec=HomeAssistant)

    tool_input = MockToolInput({"query": "Test"})
    llm_context = MockLLMContext("agent_1")

    mock_manager.async_search_memory.return_value = []

    result = await tool.async_call(hass, tool_input, llm_context)
    assert result["success"] is True
    assert "No matching memories found" in result["message"]


async def test_search_memory_tool_error(mock_manager):
    """Test SearchMemoryTool error handling."""
    tool = SearchMemoryTool(mock_manager)
    hass = MagicMock(spec=HomeAssistant)
    mock_manager.async_search_memory.side_effect = Exception("Search Error")

    tool_input = MockToolInput({"query": "Test"})
    llm_context = MockLLMContext("agent_1")

    with pytest.raises(Exception, match="Search Error"):
        await tool.async_call(hass, tool_input, llm_context)


async def test_delete_memory_tool(mock_manager):
    """Test DeleteMemoryTool."""
    tool = DeleteMemoryTool(mock_manager)
    hass = MagicMock(spec=HomeAssistant)

    tool_input = MockToolInput({"memory_id": "test-id"})
    llm_context = MockLLMContext("agent_1")

    result = await tool.async_call(hass, tool_input, llm_context)
    assert result["success"] is True
    mock_manager.async_delete_memory.assert_called_with("test-id", "agent_1")


async def test_get_api_instance_success(mock_manager):
    """Test getting API instance successfully."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {DOMAIN: {"manager": mock_manager}}

    api = llm_api.MemoryAPI(hass)
    llm_context = MockLLMContext("agent_1")

    instance = await api.async_get_api_instance(llm_context)
    assert hasattr(instance, 'api')
    assert hasattr(instance, 'tools')
    assert len(instance.tools) == 3  # add, search, delete


async def test_get_api_instance_no_manager():
    """Test getting API instance when manager is missing."""
    hass = MagicMock(spec=HomeAssistant)
    hass.data = {DOMAIN: {}}  # No manager

    api = llm_api.MemoryAPI(hass)
    llm_context = MockLLMContext("agent_1")

    instance = await api.async_get_api_instance(llm_context)
    assert hasattr(instance, 'api')
    assert hasattr(instance, 'tools')
    assert len(instance.tools) == 0
    assert "Error: Memory system unavailable" in instance.api_prompt


async def test_async_setup_duplicate_registration():
    """Test that duplicate registration is handled gracefully."""
    hass = MagicMock(spec=HomeAssistant)

    with patch.object(llm_api, "llm") as mock_llm_module:
        mock_llm_module.async_register_api.side_effect = Exception("API already registered")
        await llm_api.async_setup(hass)
        mock_llm_module.async_register_api.assert_called_once()
