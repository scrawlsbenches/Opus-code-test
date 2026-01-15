"""
CEL Event Stores.

This module provides different EventStore implementations:
- MemoryEventStore: In-memory store for testing and demos
- StreamingEventStore: File-based store with batching (in performance/)

Usage:
    from cortical.cel.stores import MemoryEventStore

    store = MemoryEventStore()
    store.append(event)
    event = store.get(event_id)
"""

from .memory import MemoryEventStore

__all__ = ['MemoryEventStore']
