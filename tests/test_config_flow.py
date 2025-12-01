"""Test the AI Memory config flow."""
from unittest.mock import patch, MagicMock, AsyncMock

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.ai_memory.constants import DOMAIN


async def test_form(hass: HomeAssistant) -> None:
    """Test we get the form."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {}

    with patch(
            "custom_components.ai_memory.async_setup_entry",
            return_value=True,
    ) as mock_setup_entry:
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "max_entries": 500,
                "embedding_engine": "tfidf",
            },
        )
        await hass.async_block_till_done()

    assert result2["type"] == FlowResultType.CREATE_ENTRY
    assert result2["title"] == "AI Memory"
    assert result2["data"]["max_entries"] == 500
    assert result2["data"]["embedding_engine"] == "tfidf"
    assert len(mock_setup_entry.mock_calls) == 1


async def test_singleton(hass: HomeAssistant, mock_config_entry) -> None:
    """Test that we can only create one entry."""
    mock_config_entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "single_instance_allowed"


async def test_options_flow(hass: HomeAssistant, mock_config_entry) -> None:
    """Test options flow."""
    mock_config_entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(
        mock_config_entry.entry_id
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "init"

    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            "max_entries": 2000,
            "embedding_engine": "tfidf",
        },
    )

    assert result2["type"] == FlowResultType.CREATE_ENTRY
    assert mock_config_entry.data["max_entries"] == 2000
    assert mock_config_entry.data["embedding_engine"] == "tfidf"


async def test_form_remote(hass: HomeAssistant) -> None:
    """Test we get the form for remote engine."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["errors"] == {}

    # Select Remote Engine
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "max_entries": 500,
            "embedding_engine": "remote",
        },
    )
    assert result2["type"] == FlowResultType.FORM
    assert result2["step_id"] == "remote_config"

    # Mock network calls for model selection and pull
    with patch("aiohttp.ClientSession.get") as mock_get, \
            patch("aiohttp.ClientSession.post") as mock_post, \
            patch("custom_components.ai_memory.async_setup_entry", return_value=True) as mock_setup_entry:
        # Mock tags response
        mock_get_response = MagicMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value={"models": [{"name": "llama2"}]})
        mock_get.return_value.__aenter__.return_value = mock_get_response

        # Mock pull response
        mock_post_response = MagicMock()
        mock_post_response.status = 200
        mock_post.return_value.__aenter__.return_value = mock_post_response

        # Enter Remote URL
        result3 = await hass.config_entries.flow.async_configure(
            result2["flow_id"],
            {
                "remote_url": "http://localhost:11434",
            },
        )
        assert result3["type"] == FlowResultType.FORM
        assert result3["step_id"] == "model_selection"

        # Select Model
        result4 = await hass.config_entries.flow.async_configure(
            result3["flow_id"],
            {
                "model_name": "llama2",
            },
        )
        await hass.async_block_till_done()

    assert result4["type"] == FlowResultType.CREATE_ENTRY
    assert result4["title"] == "AI Memory"
    assert result4["data"]["max_entries"] == 500
    assert result4["data"]["embedding_engine"] == "remote"
    assert result4["data"]["remote_url"] == "http://localhost:11434"
    assert result4["data"]["model_name"] == "llama2"
    assert len(mock_setup_entry.mock_calls) == 1


async def test_options_flow_remote(hass: HomeAssistant, mock_config_entry) -> None:
    """Test options flow for remote engine."""
    mock_config_entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(
        mock_config_entry.entry_id
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "init"

    # Switch to Remote Engine
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            "max_entries": 2000,
            "embedding_engine": "remote",
        },
    )
    assert result2["type"] == FlowResultType.FORM
    assert result2["step_id"] == "remote_config"

    # Mock network calls for model selection and pull
    with patch("aiohttp.ClientSession.get") as mock_get, \
            patch("aiohttp.ClientSession.post") as mock_post:
        # Mock tags response
        mock_get_response = MagicMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value={"models": [{"name": "remote_model"}]})
        mock_get.return_value.__aenter__.return_value = mock_get_response

        # Mock pull response
        mock_post_response = MagicMock()
        mock_post_response.status = 200
        mock_post.return_value.__aenter__.return_value = mock_post_response

        # Enter Remote URL
        result3 = await hass.config_entries.options.async_configure(
            result2["flow_id"],
            user_input={
                "remote_url": "http://remote:11434",
            },
        )
        assert result3["type"] == FlowResultType.FORM
        assert result3["step_id"] == "model_selection"

        # Select Model
        result4 = await hass.config_entries.options.async_configure(
            result3["flow_id"],
            user_input={
                "model_name": "remote_model",
            },
        )
        await hass.async_block_till_done()

    assert result4["type"] == FlowResultType.CREATE_ENTRY
    assert mock_config_entry.data["max_entries"] == 2000
    assert mock_config_entry.data["embedding_engine"] == "remote"
    assert mock_config_entry.data["remote_url"] == "http://remote:11434"
    assert mock_config_entry.data["model_name"] == "remote_model"
