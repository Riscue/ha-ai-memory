"""Default palace structure definitions."""
from typing import Dict, List

DEFAULT_PALACE: List[Dict] = [
    {"wing": "household", "rooms": ["devices", "maintenance", "events"], "scope": "common"},
    {"wing": "personal", "rooms": ["preferences", "health", "secrets"], "scope": "private"},
    {"wing": "automation", "rooms": ["routines", "schedules"], "scope": "common"},
    {"wing": "general", "rooms": ["general"], "scope": "common"},
]

# English technology terms — LLM determines wing/room language-independently,
# these keywords are only used as fallback when LLM doesn't provide parameters.
ROOM_KEYWORDS: Dict[str, List[str]] = {
    "devices": ["device", "light", "switch", "sensor", "thermostat", "camera", "speaker", "tv", "lock"],
    "maintenance": ["broken", "repair", "fix", "replace", "battery", "filter"],
    "preferences": ["prefer", "like", "dislike", "favorite", "hate", "love"],
    "routines": ["routine", "morning", "evening", "schedule", "alarm"],
    "health": ["medicine", "doctor", "allergy"],
    "events": ["visit", "guest", "party"],
    "secrets": ["password", "code", "pin"],
    "schedules": ["schedule", "calendar", "plan"],
}

# Scope-based defaults when no keyword matches
SCOPE_DEFAULTS = {
    "private": ("personal", "preferences"),
    "common": ("household", "general"),
}
