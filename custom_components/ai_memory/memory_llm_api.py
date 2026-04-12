"""Backward-compatible import shim. Use llm.api instead."""
from .llm.api import async_setup, MemoryAPI, API_ID

__all__ = ["async_setup", "MemoryAPI", "API_ID"]
