"""Backward-compatible import shim. Use memory.manager instead."""
from .memory.manager import MemoryManager

__all__ = ["MemoryManager"]
