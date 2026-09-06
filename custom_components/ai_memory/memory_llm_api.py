"""Backward-compatible import shim. Use llm_api.api instead."""
from .llm_api.api import async_setup, MemoryAPI, API_ID

__all__ = ["async_setup", "MemoryAPI", "API_ID"]
