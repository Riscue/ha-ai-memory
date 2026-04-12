"""Room detection based on content keywords and scope."""
import logging
import re
from typing import Tuple

from .defaults import SCOPE_DEFAULTS
from ..constants import DEFAULT_WING, DEFAULT_ROOM
from ..memory.store import MemoryStore
from .structure import PalaceStructure

_LOGGER = logging.getLogger(__name__)


class RoomDetector:
    """Detects wing/room from content using keyword matching.

    Strategy: LLM-first, keyword-fallback, language-independent.
    1. LLM provides wing/room via tool parameters (handled by caller)
    2. Keyword matching from palace_structure.auto_assign_keywords
    3. Scope-based default (private -> personal, common -> household)
    4. Fallback -> general/general
    """

    def __init__(self, store: MemoryStore):
        """Initialize room detector.

        Args:
            store: MemoryStore instance for database access.
        """
        self._palace = PalaceStructure(store)
        self._keyword_map = None

    def _ensure_keywords_loaded(self):
        """Load keyword map from database if not cached."""
        if self._keyword_map is None:
            self._keyword_map = self._palace.get_all_keywords()

    def refresh_keywords(self):
        """Force reload of keywords from database."""
        self._keyword_map = None

    def detect(self, content: str, scope: str) -> Tuple[str, str]:
        """Detect wing and room from content and scope.

        Args:
            content: The memory content text.
            scope: Memory scope ('private' or 'common').

        Returns:
            Tuple of (wing, room).
        """
        if not content:
            return DEFAULT_WING, DEFAULT_ROOM

        # 1. Keyword matching
        self._ensure_keywords_loaded()
        if self._keyword_map:
            tokens = re.findall(r'\b\w+\b', content.lower())
            match_counts: dict = {}  # "wing/room" -> count

            for token in tokens:
                match = self._keyword_map.get(token)
                if match:
                    key = f"{match['wing']}/{match['room']}"
                    match_counts[key] = match_counts.get(key, 0) + 1

            if match_counts:
                best_match = max(match_counts, key=match_counts.get)
                wing, room = best_match.split("/")
                _LOGGER.debug("Room detected: %s/%s (keyword match)", wing, room)
                return wing, room

        # 2. Scope-based default
        if scope in SCOPE_DEFAULTS:
            wing, room = SCOPE_DEFAULTS[scope]
            _LOGGER.debug("Room detected: %s/%s (scope default)", wing, room)
            return wing, room

        # 3. Fallback
        return DEFAULT_WING, DEFAULT_ROOM
