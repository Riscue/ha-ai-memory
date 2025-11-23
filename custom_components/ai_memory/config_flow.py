"""Config flow for AI Memory integration."""
import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import config_validation as cv

_LOGGER = logging.getLogger(__name__)

DOMAIN = "ai_memory"


class AiMemoryConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for AI Memory."""

    VERSION = 1

    def __init__(self):
        """Initialize config flow."""
        self._storage_location = "/config/ai_memory/"
        self._default_max_entries = 1000

    async def async_step_user(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors = {}

        # Check if already configured
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        if user_input is not None:
            storage_location = user_input.get("storage_location", self._storage_location)
            
            # Create storage directory
            try:
                os.makedirs(storage_location, exist_ok=True)
                _LOGGER.debug(f"Memory directory created/verified: {storage_location}")
            except Exception as e:
                _LOGGER.error(f"Failed to create memory directory: {e}")
                errors["base"] = "cannot_create_directory"

            if not errors:
                return self.async_create_entry(
                    title="AI Memory",
                    data={
                        "storage_location": storage_location,
                        "max_entries": user_input.get("max_entries", self._default_max_entries),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                )

        schema = vol.Schema({
            vol.Required(
                "storage_location",
                default=self._storage_location
            ): cv.string,
            vol.Optional(
                "max_entries",
                default=self._default_max_entries
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10000)),
        })

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(
            config_entry: config_entries.ConfigEntry,
    ) -> "AiMemoryOptionsFlow":
        """Create the options flow."""
        return AiMemoryOptionsFlow(config_entry)


class AiMemoryOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for AI Memory."""

    def __init__(self, config_entry: config_entries.ConfigEntry):
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Manage the options."""
        errors = {}

        if user_input is not None:
            # Update the entry
            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data=user_input
            )
            
            # Reload to apply changes
            await self.hass.config_entries.async_reload(self.config_entry.entry_id)
            return self.async_create_entry(title="", data={})

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required(
                    "storage_location",
                    default=self.config_entry.data.get("storage_location", "/config/ai_memory/")
                ): cv.string,
                vol.Required(
                    "max_entries",
                    default=self.config_entry.data.get("max_entries", 1000)
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10000)),
            }),
            errors=errors
        )
