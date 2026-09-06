"""Config flow for AI Memory integration."""
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import aiohttp
import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from . import MEMORY_MAX_ENTRIES
from .constants import (
    DEFAULT_MODEL,
    DEFAULT_REMOTE_URL,
    DOMAIN,
    PROVIDER_NAMES,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_TFIDF,
)

_LOGGER = logging.getLogger(__name__)

CONF_EMBEDDING_ENGINE = "embedding_engine"
CONF_REMOTE_URL = "remote_url"
CONF_MODEL_NAME = "model_name"
CONF_API_KEY = "api_key"

_PROVIDER_SELECTOR = SelectSelector(
    SelectSelectorConfig(
        options=[
            {"value": PROVIDER_OLLAMA, "label": PROVIDER_NAMES[PROVIDER_OLLAMA]},
            {"value": PROVIDER_OPENAI, "label": PROVIDER_NAMES[PROVIDER_OPENAI]},
            {"value": PROVIDER_TFIDF, "label": PROVIDER_NAMES[PROVIDER_TFIDF]},
        ],
        mode=SelectSelectorMode.DROPDOWN,
    )
)

_API_KEY_SELECTOR = TextSelector(
    TextSelectorConfig(type=TextSelectorType.PASSWORD)
)


def _validate_url(url: str) -> bool:
    """Basic URL format validation."""
    return url.startswith("http://") or url.startswith("https://")


def _entry_provider(entry: config_entries.ConfigEntry) -> str:
    """Normalize a stored provider value for UI defaults.

    Entries without a provider but with a remote URL prefill as Ollama.
    """
    value = entry.data.get(CONF_EMBEDDING_ENGINE)
    if value in PROVIDER_NAMES:
        return value
    if entry.data.get(CONF_REMOTE_URL):
        return PROVIDER_OLLAMA
    return PROVIDER_TFIDF


def _auth_headers(api_key: Optional[str]) -> Optional[Dict[str, str]]:
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}


async def _fetch_models(
    remote_url: str, provider: str, api_key: Optional[str]
) -> tuple[Optional[List[str]], Optional[str]]:
    """Fetch available model names.

    Returns (names, error): names is None on failure; error is
    "cannot_connect" when the server is unreachable, "models_fetch_failed"
    when it answered but the list endpoint is unusable, None on success.
    """
    if provider == PROVIDER_OPENAI:
        url = f"{remote_url}/v1/models"
    else:
        url = f"{remote_url}/api/tags"

    kwargs: Dict[str, Any] = {"timeout": aiohttp.ClientTimeout(total=5)}
    headers = _auth_headers(api_key)
    if headers:
        kwargs["headers"] = headers

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, **kwargs) as response:
                if response.status != 200:
                    return None, "models_fetch_failed"
                data = await response.json()
                if provider == PROVIDER_OPENAI:
                    names = [m["id"] for m in data.get("data", []) if m.get("id")]
                else:
                    names = [m["name"] for m in data.get("models", []) if m.get("name")]
                return (names or None), (None if names else "models_fetch_failed")
    except (aiohttp.ClientConnectionError, TimeoutError):
        return None, "cannot_connect"
    except Exception:
        return None, "models_fetch_failed"


async def _pull_model(remote_url: str, model_name: str, api_key: Optional[str]) -> bool:
    """Trigger a model pull on an Ollama server. Returns True on success."""
    kwargs: Dict[str, Any] = {
        "json": {"name": model_name},
        "timeout": aiohttp.ClientTimeout(total=300),
    }
    headers = _auth_headers(api_key)
    if headers:
        kwargs["headers"] = headers

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{remote_url}/api/pull", **kwargs) as response:
                return response.status == 200
    except Exception:
        return False


class AiMemoryConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for AI Memory."""

    # v3: split the legacy "remote" engine into ollama/openai_compatible
    # providers (see async_migrate_entry in __init__.py).
    VERSION = 3

    def __init__(self):
        """Initialize config flow."""
        self._default_max_entries = MEMORY_MAX_ENTRIES
        self._user_input = {}

    async def async_step_user(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle the initial step."""
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        if user_input is not None:
            self._user_input = user_input
            if user_input.get(CONF_EMBEDDING_ENGINE) == PROVIDER_TFIDF:
                return self.async_create_entry(
                    title="AI Memory", data=self._build_entry_data()
                )
            return await self.async_step_remote_config()

        schema = vol.Schema({
            vol.Optional(
                "max_entries",
                default=self._default_max_entries
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10000)),
            vol.Required(
                CONF_EMBEDDING_ENGINE,
                default=PROVIDER_OLLAMA,
            ): _PROVIDER_SELECTOR,
        })

        return self.async_show_form(step_id="user", data_schema=schema, errors={})

    async def async_step_remote_config(
            self, user_input: Optional[Dict[str, Any]] = None,
            connection_error: Optional[str] = None,
    ) -> FlowResult:
        """Handle remote configuration step (URL and optional API key)."""
        errors = {"base": connection_error} if connection_error else {}

        if user_input is not None:
            remote_url = user_input[CONF_REMOTE_URL]
            if not _validate_url(remote_url):
                errors = {"base": "invalid_url"}
            else:
                self._user_input[CONF_REMOTE_URL] = remote_url.rstrip("/")
                if user_input.get(CONF_API_KEY):
                    self._user_input[CONF_API_KEY] = user_input[CONF_API_KEY]
                return await self.async_step_model_selection()

        schema = vol.Schema({
            vol.Required(
                CONF_REMOTE_URL,
                default=self._user_input.get(CONF_REMOTE_URL, DEFAULT_REMOTE_URL)
            ): str,
            vol.Optional(CONF_API_KEY): _API_KEY_SELECTOR,
        })

        return self.async_show_form(
            step_id="remote_config", data_schema=schema, errors=errors
        )

    async def async_step_model_selection(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle model selection step.

        Offers a dropdown when model discovery succeeds; falls back to a
        free-text field when the server is reachable but has no usable
        model-list endpoint. An unreachable server bounces back to the
        connection step.
        """
        errors = {}
        remote_url = self._user_input[CONF_REMOTE_URL]
        provider = self._user_input[CONF_EMBEDDING_ENGINE]
        api_key = self._user_input.get(CONF_API_KEY)

        if user_input is not None:
            model_name = user_input[CONF_MODEL_NAME]

            # Pull gate applies to Ollama only; OpenAI-compatible servers
            # load the model at startup.
            if provider == PROVIDER_OLLAMA:
                if await _pull_model(remote_url, model_name, api_key):
                    self._user_input[CONF_MODEL_NAME] = model_name
                    return self.async_create_entry(
                        title="AI Memory", data=self._build_entry_data()
                    )
                errors["base"] = "pull_failed"
            else:
                self._user_input[CONF_MODEL_NAME] = model_name
                return self.async_create_entry(
                    title="AI Memory", data=self._build_entry_data()
                )

        models, fetch_error = await _fetch_models(remote_url, provider, api_key)
        if fetch_error == "cannot_connect":
            # Asking for a model name is pointless until the URL works.
            return await self.async_step_remote_config(
                connection_error="cannot_connect"
            )

        if models:
            schema = vol.Schema({
                vol.Required(
                    CONF_MODEL_NAME,
                    default=models[0]
                ): vol.In(models),
            })
        else:
            # Server answered but listing failed: accept a manual model name.
            errors["base"] = errors.get("base") or "models_fetch_failed"
            schema = vol.Schema({
                vol.Required(
                    CONF_MODEL_NAME,
                    default=DEFAULT_MODEL
                ): str,
            })

        return self.async_show_form(
            step_id="model_selection", data_schema=schema, errors=errors
        )

    def _build_entry_data(self) -> Dict[str, Any]:
        """Assemble the final entry data."""
        data = {
            "max_entries": self._user_input.get("max_entries", self._default_max_entries),
            "embedding_engine": self._user_input.get(
                CONF_EMBEDDING_ENGINE, PROVIDER_TFIDF
            ),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if CONF_REMOTE_URL in self._user_input:
            data[CONF_REMOTE_URL] = self._user_input[CONF_REMOTE_URL]
            data[CONF_MODEL_NAME] = self._user_input.get(
                CONF_MODEL_NAME, DEFAULT_MODEL
            )
            if self._user_input.get(CONF_API_KEY):
                data[CONF_API_KEY] = self._user_input[CONF_API_KEY]

        return data

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
        self._user_input = {}

    async def async_step_init(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            self._user_input = user_input
            if user_input.get(CONF_EMBEDDING_ENGINE) != PROVIDER_TFIDF:
                return await self.async_step_remote_config()

            data = dict(self.config_entry.data)
            data.update({
                "max_entries": user_input.get("max_entries"),
                CONF_EMBEDDING_ENGINE: PROVIDER_TFIDF,
            })
            for key in (CONF_REMOTE_URL, CONF_MODEL_NAME, CONF_API_KEY):
                data.pop(key, None)

            self.hass.config_entries.async_update_entry(
                self.config_entry,
                data=data,
            )
            await self.hass.config_entries.async_reload(self.config_entry.entry_id)
            return self.async_create_entry(title="", data={})

        current_provider = _entry_provider(self.config_entry)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required(
                    "max_entries",
                    default=self.config_entry.data.get("max_entries", MEMORY_MAX_ENTRIES)
                ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10000)),
                vol.Required(
                    CONF_EMBEDDING_ENGINE,
                    default=current_provider,
                ): _PROVIDER_SELECTOR,
            }),
        )

    async def async_step_remote_config(
            self, user_input: Optional[Dict[str, Any]] = None,
            connection_error: Optional[str] = None,
    ) -> FlowResult:
        """Handle remote configuration step (URL and optional API key)."""
        errors = {"base": connection_error} if connection_error else {}

        if user_input is not None:
            remote_url = user_input[CONF_REMOTE_URL]
            if not _validate_url(remote_url):
                errors = {"base": "invalid_url"}
            else:
                self._user_input[CONF_REMOTE_URL] = remote_url.rstrip("/")
                if user_input.get(CONF_API_KEY):
                    self._user_input[CONF_API_KEY] = user_input[CONF_API_KEY]
                return await self.async_step_model_selection()

        schema = vol.Schema({
            vol.Required(
                CONF_REMOTE_URL,
                default=self.config_entry.data.get(
                    CONF_REMOTE_URL,
                    self._user_input.get(CONF_REMOTE_URL, DEFAULT_REMOTE_URL),
                )
            ): str,
            vol.Optional(
                CONF_API_KEY,
                default=self.config_entry.data.get(CONF_API_KEY, ""),
            ): _API_KEY_SELECTOR,
        })

        return self.async_show_form(
            step_id="remote_config", data_schema=schema, errors=errors
        )

    async def async_step_model_selection(
            self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Handle model selection step."""
        errors = {}
        remote_url = self._user_input[CONF_REMOTE_URL]
        provider = self._user_input[CONF_EMBEDDING_ENGINE]
        api_key = self._user_input.get(CONF_API_KEY)
        current_model = self.config_entry.data.get(CONF_MODEL_NAME, DEFAULT_MODEL)

        if user_input is not None:
            model_name = user_input[CONF_MODEL_NAME]

            if provider == PROVIDER_OLLAMA:
                if not await _pull_model(remote_url, model_name, api_key):
                    errors["base"] = "pull_failed"
                else:
                    return await self._finish(model_name)
            else:
                return await self._finish(model_name)

        models, fetch_error = await _fetch_models(remote_url, provider, api_key)
        if fetch_error == "cannot_connect":
            return await self.async_step_remote_config(
                connection_error="cannot_connect"
            )

        if models:
            default_model = (
                current_model if current_model in models else models[0]
            )
            schema = vol.Schema({
                vol.Required(
                    CONF_MODEL_NAME,
                    default=default_model
                ): vol.In(models),
            })
        else:
            errors["base"] = errors.get("base") or "models_fetch_failed"
            schema = vol.Schema({
                vol.Required(
                    CONF_MODEL_NAME,
                    default=current_model
                ): str,
            })

        return self.async_show_form(
            step_id="model_selection", data_schema=schema, errors=errors
        )

    async def _finish(self, model_name: str) -> FlowResult:
        """Update the entry and reload."""
        data = dict(self.config_entry.data)
        data.update({
            "max_entries": self._user_input.get("max_entries"),
            CONF_EMBEDDING_ENGINE: self._user_input.get(CONF_EMBEDDING_ENGINE),
            CONF_REMOTE_URL: self._user_input[CONF_REMOTE_URL],
            CONF_MODEL_NAME: model_name,
        })
        if self._user_input.get(CONF_API_KEY):
            data[CONF_API_KEY] = self._user_input[CONF_API_KEY]
        else:
            data.pop(CONF_API_KEY, None)

        self.hass.config_entries.async_update_entry(
            self.config_entry,
            data=data,
        )
        await self.hass.config_entries.async_reload(self.config_entry.entry_id)
        return self.async_create_entry(title="", data={})
