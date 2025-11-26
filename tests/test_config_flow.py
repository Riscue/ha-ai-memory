"""Test the AI Memory config flow."""
from unittest.mock import patch

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.ai_memory import DOMAIN


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
                "storage_location": "/tmp/test_ai_memory",
                "max_entries": 500,
            },
        )
        await hass.async_block_till_done()

    assert result2["type"] == FlowResultType.CREATE_ENTRY
    assert result2["title"] == "AI Memory"
    assert result2["data"]["storage_location"] == "/tmp/test_ai_memory"
    assert result2["data"]["max_entries"] == 500
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
            "storage_location": "/tmp/new_location",
            "max_entries": 2000,
        },
    )

    assert result2["type"] == FlowResultType.CREATE_ENTRY
    assert mock_config_entry.data["storage_location"] == "/tmp/new_location"
    assert mock_config_entry.data["max_entries"] == 2000


async def test_form_directory_creation_error(hass: HomeAssistant) -> None:
    """Test error handling when directory creation fails."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    
    with patch("os.makedirs", side_effect=OSError("Permission denied")):
        result2 = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "storage_location": "/root/protected",
            },
        )
        
    assert result2["type"] == FlowResultType.FORM
    assert result2["errors"] == {"base": "cannot_create_directory"}
