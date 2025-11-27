"""Config flow for AI Memory integration."""
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import config_validation as cv

from .constants import (
    DOMAIN,
    ENGINE_AUTO,
    ENGINE_FASTEMBED,
    ENGINE_SENTENCE_TRANSFORMER,
    ENGINE_TFIDF,
    ENGINE_NAMES,
)

_LOGGER = logging.getLogger(__name__)

CONF_EMBEDDING_ENGINE = "embedding_engine"


class AiMemoryConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for AI Memory."""

    VERSION = 1

    def __init__(self):
        """Initialize config flow."""
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
            if not errors:
                return self.async_create_entry(
                    title="AI Memory",
                    data={
                        "max_entries": user_input.get("max_entries", self._default_max_entries),
                        "embedding_engine": user_input.get("embedding_engine", ENGINE_AUTO),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                )

        schema = vol.Schema({
            vol.Optional(
                "max_entries",
                default=self._default_max_entries
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10000)),
            vol.Optional(
                "embedding_engine",
                default=ENGINE_AUTO
            ): vol.In(ENGINE_NAMES),
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
                    "max_entries",
                    default=self.config_entry.data.get("max_entries", 1000)
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10000)),
                vol.Optional(
                    "embedding_engine",
                    default=self.config_entry.data.get("embedding_engine", ENGINE_AUTO)
                ): vol.In(ENGINE_NAMES),
            }),
            errors=errors
        )
