import logging
import os
from datetime import datetime
from typing import List, Dict

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class MemoryManager:
    """Manages a single memory storage."""

    def __init__(
            self,
            hass: HomeAssistant,
            memory_id: str,
            memory_name: str,
            description: str,
            storage_location: str,
            max_entries: int,
            device_info: dict = None
    ):
        self.hass = hass
        self.memory_id = memory_id
        self.memory_name = memory_name
        self.description = description
        self.storage_location = storage_location
        self.max_entries = max_entries
        self.device_info = device_info  # Device info for UI components
        self._memories: List[Dict[str, str]] = []

        # Ensure memory directory exists
        from .platform_helpers import ensure_directory_exists
        ensure_directory_exists(self.storage_location)

    def get_memory_file_path(self) -> str:
        """Get file path for this memory."""
        return os.path.join(self.storage_location, f"{self.memory_id}.json")

    def _read_file(self, file_path: str) -> List[Dict[str, str]]:
        """Read memories from file."""
        from .platform_helpers import read_json_file
        return read_json_file(file_path)

    def _save_to_file(self, data: List[Dict[str, str]], file_path: str):
        """Save memories to file."""
        from .platform_helpers import write_json_file
        success = write_json_file(file_path, data)
        if success:
            _LOGGER.debug(f"Saved {len(data)} memories to {file_path}")

    async def async_load_memories(self):
        """Load memories from file."""
        file_path = self.get_memory_file_path()
        self._memories = await self.hass.async_add_executor_job(self._read_file, file_path)

    async def async_add_memory(self, text: str):
        """Add new memory entry."""
        if not text or not text.strip():
            _LOGGER.warning("Cannot add empty memory")
            return

        # Check memory limit
        if len(self._memories) >= self.max_entries:
            # Remove oldest entry if limit reached
            removed = self._memories.pop(0)
            _LOGGER.warning(
                f"Memory limit ({self.max_entries}) reached for '{self.memory_id}', "
                f"removed oldest entry: {removed['date']}"
            )

        entry = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text.strip()
        }
        self._memories.append(entry)

        file_path = self.get_memory_file_path()
        await self.hass.async_add_executor_job(self._save_to_file, self._memories, file_path)
        _LOGGER.debug(f"Memory added to '{self.memory_id}': {text[:50]}...")

        # Fire event to notify sensors to update
        if hasattr(self.hass, 'bus'):
            self.hass.bus.async_fire("ai_memory_updated", {"memory_id": self.memory_id})

    async def async_clear_memory(self):
        """Clear all memories."""
        count = len(self._memories)
        self._memories = []
        file_path = self.get_memory_file_path()
        await self.hass.async_add_executor_job(self._save_to_file, [], file_path)
        _LOGGER.debug(f"Memory '{self.memory_id}' cleared ({count} entries removed)")

        # Fire event to notify sensors to update
        if hasattr(self.hass, 'bus'):
            self.hass.bus.async_fire("ai_memory_updated", {"memory_id": self.memory_id})
