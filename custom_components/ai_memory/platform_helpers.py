"""Common platform utilities for AI Memory integration."""
import json
import logging
import os
from typing import List, Callable, TypeVar, Dict

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .constants import DOMAIN
from .memory_manager import MemoryManager

_LOGGER = logging.getLogger(__name__)

T = TypeVar('T')


async def async_setup_platform_entities(
        hass: HomeAssistant,
        entry: ConfigEntry,
        async_add_entities: AddEntitiesCallback,
        entity_factory: Callable[[HomeAssistant, ConfigEntry, MemoryManager], T],
        platform_name: str
) -> None:
    """Generic platform setup function for AI Memory entities."""
    memory_managers = hass.data.get(DOMAIN, {}).get("memory_managers", {})

    if not memory_managers:
        _LOGGER.warning(f"No memory managers found for {platform_name} platform")
        return

    entities = []
    for manager in memory_managers.values():
        entity = entity_factory(hass, entry, manager)
        entities.append(entity)

    _LOGGER.debug(f"Creating {len(entities)} {platform_name} entities")
    async_add_entities(entities, True)


def get_memory_managers(hass: HomeAssistant) -> dict:
    """Get memory managers from hass data."""
    return hass.data.get(DOMAIN, {}).get("memory_managers", {})


def ensure_directory_exists(directory: str) -> bool:
    """Ensure directory exists, return True if successful."""
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        _LOGGER.error(f"Failed to create directory {directory}: {e}")
        return False


def read_json_file(file_path: str) -> List[Dict[str, str]]:
    """Read and parse JSON file with error handling."""
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                return content
            else:
                _LOGGER.warning(f"Invalid format in {file_path}, expected list")
                return []
    except json.JSONDecodeError as e:
        _LOGGER.error(f"JSON decode error in {file_path}: {e}")
        try:
            backup_corrupted_file(file_path)
        except Exception as backup_error:
            _LOGGER.error(f"Failed to backup corrupted file: {backup_error}")
        return []
    except Exception as e:
        _LOGGER.error(f"Error reading {file_path}: {e}")
        return []


def write_json_file(file_path: str, data: List[Dict[str, str]]) -> bool:
    """Write data to JSON file with error handling."""
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        _LOGGER.error(f"Failed to write {file_path}: {e}")
        # Try temp file fallback
        try:
            temp_path = f"{file_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            _LOGGER.warning(f"Saved to temporary file {temp_path}")
            return False
        except Exception as temp_error:
            _LOGGER.error(f"Failed to save temporary file: {temp_error}")
            return False


def backup_corrupted_file(file_path: str) -> None:
    """Backup corrupted file if it exists."""
    try:
        if os.path.exists(file_path):
            backup_path = f"{file_path}.bak"
            os.rename(file_path, backup_path)
            _LOGGER.warning(f"Backed up corrupted file to {backup_path}")
    except Exception as e:
        _LOGGER.error(f"Failed to backup corrupted file: {e}")
