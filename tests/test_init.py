"""Tests for AI Memory Init."""
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from homeassistant.core import HomeAssistant

from custom_components.ai_memory import (
    async_setup,
    async_setup_entry,
    async_unload_entry,
    async_reload_entry,
)
from custom_components.ai_memory.constants import DOMAIN


async def test_async_setup(hass: HomeAssistant):
    """Test async_setup."""
    assert await async_setup(hass, {})
    assert DOMAIN in hass.data


async def test_setup_entry_creates_single_manager(hass: HomeAssistant, mock_config_entry):
    """Test that setup creates a single memory manager."""
    mock_config_entry.add_to_hass(hass)

    # Mock dependencies
    mock_memory_api = MagicMock()
    mock_memory_api.async_setup = AsyncMock()

    import custom_components.ai_memory
    with patch.object(custom_components.ai_memory, "MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"), \
            patch.dict("sys.modules", {
                "custom_components.ai_memory.memory_llm_api": mock_memory_api
            }):
        mock_instance = mock_manager_cls.return_value
        mock_instance.async_initialize = AsyncMock()

        assert await async_setup_entry(hass, mock_config_entry)

        # Verify single manager created and stored
        assert "manager" in hass.data[DOMAIN]
        assert hass.data[DOMAIN]["manager"] == mock_instance


async def test_setup_entry_already_initialized(hass: HomeAssistant, mock_config_entry):
    """Test setup when already initialized."""
    mock_config_entry.add_to_hass(hass)
    hass.data[DOMAIN] = {"manager": MagicMock()}

    import custom_components.ai_memory
    with patch.object(custom_components.ai_memory, "MemoryManager") as mock_manager_cls, \
            patch("homeassistant.config_entries.ConfigEntries.async_forward_entry_setups"):
        assert await async_setup_entry(hass, mock_config_entry)

        # Should not create new manager
        mock_manager_cls.assert_not_called()


async def test_unload_entry(hass: HomeAssistant, mock_config_entry):
    """Test unload entry."""
    mock_config_entry.add_to_hass(hass)
    hass.data[DOMAIN] = {"manager": MagicMock()}

    with patch("homeassistant.config_entries.ConfigEntries.async_unload_platforms", return_value=True):
        assert await async_unload_entry(hass, mock_config_entry)

        # Manager should be removed
        assert "manager" not in hass.data[DOMAIN]


async def test_unload_entry_failure(hass: HomeAssistant, mock_config_entry):
    """Test unload entry failure."""
    mock_config_entry.add_to_hass(hass)
    hass.data[DOMAIN] = {"manager": MagicMock()}

    with patch("homeassistant.config_entries.ConfigEntries.async_unload_platforms", return_value=False):
        assert not await async_unload_entry(hass, mock_config_entry)

        # Manager should still exist
        assert "manager" in hass.data[DOMAIN]


async def test_reload_entry(hass: HomeAssistant, mock_config_entry):
    """Test reload entry."""
    with patch("homeassistant.config_entries.ConfigEntries.async_reload", new_callable=AsyncMock) as mock_reload:
        await async_reload_entry(hass, mock_config_entry)
        mock_reload.assert_called_once_with(mock_config_entry.entry_id)


async def _setup_entry(hass: HomeAssistant, data: dict, version: int):
    """Add an entry at the given version and set it up with setup patched."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    entry = MockConfigEntry(domain=DOMAIN, data=data, version=version)
    entry.add_to_hass(hass)

    with patch("custom_components.ai_memory.async_setup_entry", return_value=True):
        result = await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return result, entry


@pytest.mark.parametrize(
    ("entry_version", "stored_engine", "expected_engine"),
    [
        # v2 entries: legacy value, dev-build providers, missing key, garbage
        (2, "remote", "ollama"),
        (2, "tfidf", "tfidf"),
        (2, "openai_compatible", "openai_compatible"),
        (2, "ollama", "ollama"),
        (2, None, "ollama"),
        (2, "bogus", "ollama"),
        # v1 entries from older releases take the same path
        (1, "remote", "ollama"),
        (1, "tfidf", "tfidf"),
        # v3 entries are current — data must pass through untouched
        (3, "ollama", "ollama"),
        (3, "openai_compatible", "openai_compatible"),
        (3, "tfidf", "tfidf"),
    ],
)
async def test_migrate_combinations(
        hass: HomeAssistant, entry_version, stored_engine, expected_engine):
    """Every version x stored-engine combination lands on v3 with a
    canonical provider; non-providers were 'remote' historically."""
    data = {"max_entries": 500}
    if stored_engine is not None:
        data["embedding_engine"] = stored_engine

    result, entry = await _setup_entry(hass, data, entry_version)

    assert result is True
    assert entry.version == 3
    assert entry.data["embedding_engine"] == expected_engine


async def test_migrate_preserves_unrelated_data(hass: HomeAssistant):
    """A realistic 0.2.x entry keeps every key; only embedding_engine is
    rewritten, and re-running setup afterwards changes nothing."""
    legacy_data = {
        "max_entries": 750,
        "embedding_engine": "remote",
        "identity_text": "We never forget the coffee schedule",
        "created_at": "2026-08-01 09:00:00",
        "remote_url": "http://remote:11434",
        "model_name": "bge-m3",
    }
    result, entry = await _setup_entry(hass, dict(legacy_data), version=2)

    assert result is True
    expected = {**legacy_data, "embedding_engine": "ollama"}
    assert entry.data == expected

    # Idempotent: a second migration pass on the migrated entry is a no-op
    from custom_components.ai_memory import async_migrate_entry

    assert await async_migrate_entry(hass, entry) is True
    assert entry.version == 3
    assert entry.data == expected


@pytest.mark.parametrize("future_version", [4, 5])
async def test_migrate_refuses_future_version(
        hass: HomeAssistant, future_version):
    """Entries from a newer version must not be migrated downwards."""
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"embedding_engine": "ollama"},
        version=future_version,
    )
    entry.add_to_hass(hass)

    with patch("custom_components.ai_memory.async_setup_entry", return_value=True):
        result = await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

    assert result is False
    assert entry.version == future_version
    assert entry.data["embedding_engine"] == "ollama"
