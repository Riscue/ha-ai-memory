"""Global fixtures for ai_memory integration."""
from unittest.mock import patch, MagicMock

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

pytest_plugins = "pytest_homeassistant_custom_component"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations defined in the test dir."""
    yield


@pytest.fixture
def mock_config_entry():
    """Mock a config entry."""
    return MockConfigEntry(
        domain="ai_memory",
        data={
            "storage_location": "/tmp/test_ai_memory",
            "max_entries": 500,
        },
        entry_id="test_entry_id",
    )


@pytest.fixture
def mock_setup_entry():
    """Mock setting up a config entry."""
    with patch("custom_components.ai_memory.async_setup_entry", return_value=True) as mock_setup:
        yield mock_setup


@pytest.fixture
def mock_conversation_agent():
    """Mock a conversation agent."""
    agent = MagicMock()
    agent.name = "Test Agent"
    agent.entity_id = "conversation.test_agent"
    return agent


@pytest.fixture
def mock_agent_manager(mock_conversation_agent):
    """Mock the conversation agent manager."""
    with patch("homeassistant.components.conversation.get_agent_manager", create=True) as mock_get_manager:
        manager = MagicMock()
        info = MagicMock()
        info.id = mock_conversation_agent.entity_id
        info.name = mock_conversation_agent.name
        manager.async_get_agent_info.return_value = [info]
        mock_get_manager.return_value = manager
        yield manager
