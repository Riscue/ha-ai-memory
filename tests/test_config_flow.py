"""Test the AI Memory config flow."""
from unittest.mock import patch, MagicMock, AsyncMock

import aiohttp
from voluptuous.validators import In

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.ai_memory.constants import DOMAIN


def _schema_default(result, key):
    """Extract the default value of a field from a shown data schema."""
    marker = next(
        k for k in result["data_schema"].schema if getattr(k, "schema", None) == key
    )
    return marker.default()


def _mock_http(get_json=None, get_status=200, get_error=False, post_status=200):
    """Patch aiohttp get/post like the legacy tests did."""
    get_patch = patch("aiohttp.ClientSession.get")
    post_patch = patch("aiohttp.ClientSession.post")
    mock_get = get_patch.start()
    mock_post = post_patch.start()

    if get_error:
        mock_get.side_effect = aiohttp.ClientConnectionError("Connection refused")
    else:
        mock_get_response = MagicMock()
        mock_get_response.status = get_status
        mock_get_response.json = AsyncMock(return_value=get_json or {})
        mock_get.return_value.__aenter__.return_value = mock_get_response

    mock_post_response = MagicMock()
    mock_post_response.status = post_status
    mock_post.return_value.__aenter__.return_value = mock_post_response

    return get_patch, post_patch, mock_get, mock_post


async def test_form_tfidf(hass: HomeAssistant) -> None:
    """TF-IDF setup completes in a single step."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "user"
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
    assert "identity_text" not in result2["data"]
    assert len(mock_setup_entry.mock_calls) == 1


async def test_singleton(hass: HomeAssistant, mock_config_entry) -> None:
    """Test that we can only create one entry."""
    mock_config_entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "single_instance_allowed"


async def test_form_ollama(hass: HomeAssistant) -> None:
    """Ollama provider: URL + API key, model dropdown, pull gate."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "max_entries": 500,
            "embedding_engine": "ollama",
        },
    )
    assert result2["type"] == FlowResultType.FORM
    assert result2["step_id"] == "remote_config"

    with patch("custom_components.ai_memory.async_setup_entry", return_value=True) as mock_setup_entry:
        get_patch, post_patch, mock_get, mock_post = _mock_http(
            get_json={"models": [{"name": "llama2"}]}
        )
        try:
            result3 = await hass.config_entries.flow.async_configure(
                result2["flow_id"],
                {
                    "remote_url": "http://localhost:11434/",
                    "api_key": "secret",
                },
            )
            assert result3["type"] == FlowResultType.FORM
            assert result3["step_id"] == "model_selection"
            assert result3["errors"] == {}

            result4 = await hass.config_entries.flow.async_configure(
                result3["flow_id"],
                {"model_name": "llama2"},
            )
            await hass.async_block_till_done()
        finally:
            get_patch.stop()
            post_patch.stop()

    assert result4["type"] == FlowResultType.CREATE_ENTRY
    assert result4["data"]["max_entries"] == 500
    assert result4["data"]["embedding_engine"] == "ollama"
    # Trailing slash is stripped so endpoint URLs stay single-slashed
    assert result4["data"]["remote_url"] == "http://localhost:11434"
    assert result4["data"]["model_name"] == "llama2"
    assert result4["data"]["api_key"] == "secret"
    assert len(mock_setup_entry.mock_calls) == 1

    # Pull gate fired exactly once for the selected model
    assert mock_post.call_count == 1
    assert "api/pull" in mock_post.call_args[0][0]


async def test_form_openai(hass: HomeAssistant) -> None:
    """OpenAI-compatible provider: /v1/models list, no pull, direct completion."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "max_entries": 500,
            "embedding_engine": "openai_compatible",
        },
    )
    assert result2["type"] == FlowResultType.FORM
    assert result2["step_id"] == "remote_config"

    with patch("custom_components.ai_memory.async_setup_entry", return_value=True) as mock_setup_entry:
        get_patch, post_patch, mock_get, mock_post = _mock_http(
            get_json={"data": [{"id": "bge-m3"}]}
        )
        try:
            result3 = await hass.config_entries.flow.async_configure(
                result2["flow_id"],
                {"remote_url": "http://localhost:8080"},
            )
            assert result3["type"] == FlowResultType.FORM
            assert result3["step_id"] == "model_selection"
            assert result3["errors"] == {}

            # Model list came from /v1/models
            assert "v1/models" in mock_get.call_args[0][0]

            result4 = await hass.config_entries.flow.async_configure(
                result3["flow_id"],
                {"model_name": "bge-m3"},
            )
            await hass.async_block_till_done()
        finally:
            get_patch.stop()
            post_patch.stop()

    assert result4["type"] == FlowResultType.CREATE_ENTRY
    assert result4["data"]["embedding_engine"] == "openai_compatible"
    assert result4["data"]["remote_url"] == "http://localhost:8080"
    assert result4["data"]["model_name"] == "bge-m3"
    assert "api_key" not in result4["data"]
    assert len(mock_setup_entry.mock_calls) == 1

    # No pull for OpenAI-compatible servers
    mock_post.assert_not_called()


async def test_form_manual_model_entry(hass: HomeAssistant) -> None:
    """Model name can be typed manually when the server is reachable but
    its model-list endpoint is unusable."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "max_entries": 500,
            "embedding_engine": "openai_compatible",
        },
    )

    with patch("custom_components.ai_memory.async_setup_entry", return_value=True) as mock_setup_entry:
        get_patch, post_patch, _, mock_post = _mock_http(get_status=500)
        try:
            result3 = await hass.config_entries.flow.async_configure(
                result2["flow_id"],
                {"remote_url": "http://localhost:8080"},
            )
            # Free-text fallback with an informational error
            assert result3["type"] == FlowResultType.FORM
            assert result3["step_id"] == "model_selection"
            assert result3["errors"] == {"base": "models_fetch_failed"}
            # Free-text field: plain str validator, not a vol.In dropdown
            schema_map = result3["data_schema"].schema
            marker = next(
                k for k in schema_map if getattr(k, "schema", None) == "model_name"
            )
            assert not isinstance(schema_map[marker], In)

            result4 = await hass.config_entries.flow.async_configure(
                result3["flow_id"],
                {"model_name": "my-manual-model"},
            )
            await hass.async_block_till_done()
        finally:
            get_patch.stop()
            post_patch.stop()

    assert result4["type"] == FlowResultType.CREATE_ENTRY
    assert result4["data"]["model_name"] == "my-manual-model"
    assert len(mock_setup_entry.mock_calls) == 1
    mock_post.assert_not_called()


async def test_form_unreachable_server_returns_to_url_step(
        hass: HomeAssistant) -> None:
    """An unreachable server must not ask for a model name — the flow
    bounces back to the connection step until the URL works."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "max_entries": 500,
            "embedding_engine": "openai_compatible",
        },
    )
    assert result2["step_id"] == "remote_config"

    get_patch, post_patch, _, mock_post = _mock_http(get_error=True)
    try:
        result3 = await hass.config_entries.flow.async_configure(
            result2["flow_id"],
            {"remote_url": "http://wrong-host:8080"},
        )
        assert result3["type"] == FlowResultType.FORM
        assert result3["step_id"] == "remote_config"
        assert result3["errors"] == {"base": "cannot_connect"}
        # The attempted URL is prefilled for editing
        marker = next(
            k for k in result3["data_schema"].schema
            if getattr(k, "schema", None) == "remote_url"
        )
        assert marker.default() == "http://wrong-host:8080"
    finally:
        get_patch.stop()
        post_patch.stop()

    # No entry is created and no model step is shown while unreachable
    mock_post.assert_not_called()

    # Once the URL works, the same flow completes normally
    with patch("custom_components.ai_memory.async_setup_entry", return_value=True):
        get_patch, post_patch, _, _ = _mock_http(
            get_json={"data": [{"id": "bge-m3"}]}
        )
        try:
            result4 = await hass.config_entries.flow.async_configure(
                result3["flow_id"],
                {"remote_url": "http://localhost:8080"},
            )
            assert result4["step_id"] == "model_selection"

            result5 = await hass.config_entries.flow.async_configure(
                result4["flow_id"],
                {"model_name": "bge-m3"},
            )
            await hass.async_block_till_done()
        finally:
            get_patch.stop()
            post_patch.stop()

    assert result5["type"] == FlowResultType.CREATE_ENTRY
    assert result5["data"]["remote_url"] == "http://localhost:8080"


async def test_options_flow(hass: HomeAssistant, mock_config_entry) -> None:
    """Options flow: TF-IDF path updates the entry."""
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
    await hass.async_block_till_done()

    assert result2["type"] == FlowResultType.CREATE_ENTRY
    assert mock_config_entry.data["max_entries"] == 2000
    assert mock_config_entry.data["embedding_engine"] == "tfidf"


async def test_options_flow_switch_to_tfidf_cleans_remote_keys(
        hass: HomeAssistant,
) -> None:
    """Switching to TF-IDF drops stale remote keys."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "max_entries": 500,
            "embedding_engine": "ollama",
            "remote_url": "http://remote:11434",
            "model_name": "bge-m3",
            "api_key": "secret",
        },
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            "max_entries": 500,
            "embedding_engine": "tfidf",
        },
    )
    await hass.async_block_till_done()

    assert result2["type"] == FlowResultType.CREATE_ENTRY
    assert entry.data["embedding_engine"] == "tfidf"
    assert "remote_url" not in entry.data
    assert "model_name" not in entry.data
    assert "api_key" not in entry.data


async def test_options_flow_legacy_remote_normalizes(
        hass: HomeAssistant,
) -> None:
    """Legacy entries storing embedding_engine="remote" prefill as Ollama."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "max_entries": 500,
            "embedding_engine": "remote",
            "remote_url": "http://remote:11434",
            "model_name": "bge-m3",
        },
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "init"
    assert _schema_default(result, "embedding_engine") == "ollama"

    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"],
        user_input={
            "max_entries": 2000,
            "embedding_engine": "ollama",
        },
    )
    assert result2["type"] == FlowResultType.FORM
    assert result2["step_id"] == "remote_config"
    assert result2["errors"] == {}

    get_patch, post_patch, _, _ = _mock_http(
        get_json={"models": [{"name": "bge-m3"}]}
    )
    try:
        result3 = await hass.config_entries.options.async_configure(
            result2["flow_id"],
            user_input={"remote_url": "http://remote:11434"},
        )
        assert result3["type"] == FlowResultType.FORM
        assert result3["step_id"] == "model_selection"

        result4 = await hass.config_entries.options.async_configure(
            result3["flow_id"],
            user_input={"model_name": "bge-m3"},
        )
        await hass.async_block_till_done()
    finally:
        get_patch.stop()
        post_patch.stop()

    assert result4["type"] == FlowResultType.CREATE_ENTRY
    assert entry.data["max_entries"] == 2000
    assert entry.data["embedding_engine"] == "ollama"
    assert entry.data["remote_url"] == "http://remote:11434"
    assert entry.data["model_name"] == "bge-m3"
