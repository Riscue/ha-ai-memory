"""Config flow for AI Memory integration."""
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult

from . import MEMORY_MAX_ENTRIES
from .constants import (
    DOMAIN,
    DEFAULT_MODEL,
    DEFAULT_REMOTE_URL,
    ENGINE_REMOTE,
    ENGINE_TFIDF,
    ENGINE_NAMES,
)

_LOGGER = logging.getLogger(__name__)

CONF_EMBEDDING_ENGINE = "embedding_engine"
CONF_REMOTE_URL = "remote_url"
CONF_MODEL_NAME = "model_name"


class AiMemoryConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for AI Memory."""

    VERSION = 1

    def __init__(self):
        """Initialize config flow."""
        self._default_max_entries = MEMORY_MAX_ENTRIES
        self._user_input = {}

    async def async_step_user(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle the initial step."""
        errors = {}

        # Check if already configured
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        if user_input is not None:
            self._user_input = user_input
            if user_input.get("embedding_engine") == ENGINE_REMOTE:
                return await self.async_step_remote_config()

            if not errors:
                return self.async_create_entry(
                    title="AI Memory",
                    data={
                        "max_entries": user_input.get("max_entries", self._default_max_entries),
                        "embedding_engine": user_input.get("embedding_engine", ENGINE_TFIDF),
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
                default=ENGINE_REMOTE
            ): vol.In(ENGINE_NAMES),
        })

        return self.async_show_form(
            step_id="user",
            data_schema=schema,
            errors=errors
        )

    async def async_step_remote_config(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle remote configuration step (URL)."""
        errors = {}

        if user_input is not None:
            self._user_input[CONF_REMOTE_URL] = user_input[CONF_REMOTE_URL]
            return await self.async_step_model_selection()

        schema = vol.Schema({
            vol.Required(
                CONF_REMOTE_URL,
                default=DEFAULT_REMOTE_URL
            ): str,
        })

        return self.async_show_form(
            step_id="remote_config",
            data_schema=schema,
            errors=errors
        )

    async def async_step_model_selection(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle model selection step."""
        errors = {}
        remote_url = self._user_input.get(CONF_REMOTE_URL)

        if user_input is not None:
            model_name = user_input[CONF_MODEL_NAME]

            # Trigger model download
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            f"{remote_url}/api/pull",
                            json={"name": model_name}
                    ) as response:
                        if response.status != 200:
                            errors["base"] = "pull_failed"
                        else:
                            # Success
                            data = {
                                "max_entries": self._user_input.get("max_entries", self._default_max_entries),
                                "embedding_engine": self._user_input.get("embedding_engine"),
                                "remote_url": remote_url,
                                "model_name": model_name,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            return self.async_create_entry(title="AI Memory", data=data)
            except Exception:
                errors["base"] = "cannot_connect"

        # Fetch models
        models = [DEFAULT_MODEL]
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{remote_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m["name"] for m in data.get("models", [])]
        except Exception:
            errors["base"] = "cannot_connect"

        schema = vol.Schema({
            vol.Required(
                CONF_MODEL_NAME,
                default=models[0] if models else DEFAULT_MODEL
            ): vol.In(models),
        })

        return self.async_show_form(
            step_id="model_selection",
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
        self._user_input = {}

    async def async_step_init(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Manage the options."""
        errors = {}

        if user_input is not None:
            self._user_input = user_input
            if user_input.get("embedding_engine") == ENGINE_REMOTE:
                return await self.async_step_remote_config()

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
                    default=self.config_entry.data.get("max_entries", MEMORY_MAX_ENTRIES)
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10000)),
                vol.Optional(
                    "embedding_engine",
                    default=self.config_entry.data.get("embedding_engine", ENGINE_REMOTE)
                ): vol.In(ENGINE_NAMES),
            }),
            errors=errors
        )

    async def async_step_remote_config(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle remote configuration step (URL)."""
        errors = {}

        if user_input is not None:
            self._user_input[CONF_REMOTE_URL] = user_input[CONF_REMOTE_URL]
            return await self.async_step_model_selection()

        schema = vol.Schema({
            vol.Required(
                CONF_REMOTE_URL,
                default=self.config_entry.data.get("remote_url", DEFAULT_REMOTE_URL)
            ): str,
        })

        return self.async_show_form(
            step_id="remote_config",
            data_schema=schema,
            errors=errors
        )

    async def async_step_model_selection(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle model selection step."""
        errors = {}
        remote_url = self._user_input.get(CONF_REMOTE_URL)

        if user_input is not None:
            model_name = user_input[CONF_MODEL_NAME]

            # Trigger model download
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                            f"{remote_url}/api/pull",
                            json={"name": model_name}
                    ) as response:
                        if response.status != 200:
                            errors["base"] = "pull_failed"
                        else:
                            # Success
                            data = {
                                "max_entries": self._user_input.get("max_entries"),
                                "embedding_engine": self._user_input.get("embedding_engine"),
                                "remote_url": remote_url,
                                "model_name": model_name,
                            }

                            self.hass.config_entries.async_update_entry(
                                self.config_entry,
                                data=data
                            )
                            await self.hass.config_entries.async_reload(self.config_entry.entry_id)
                            return self.async_create_entry(title="", data={})
            except Exception:
                errors["base"] = "cannot_connect"

        # Fetch models
        models = [DEFAULT_MODEL]
        current_model = self.config_entry.data.get("model_name", DEFAULT_MODEL)

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{remote_url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m["name"] for m in data.get("models", [])]
        except Exception:
            errors["base"] = "cannot_connect"

        # Ensure current model is in the list if possible, or default to it
        default_model = current_model if current_model in models else (models[0] if models else DEFAULT_MODEL)

        schema = vol.Schema({
            vol.Required(
                CONF_MODEL_NAME,
                default=default_model
            ): vol.In(models),
        })

        return self.async_show_form(
            step_id="model_selection",
            data_schema=schema,
            errors=errors
        )
