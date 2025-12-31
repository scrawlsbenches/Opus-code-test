"""
CDG adapters for backward compatibility with existing graph implementations.

This package provides adapters that make CDG compatible with existing
graph APIs, enabling gradual migration without breaking changes.

Available Adapters:
    - GoTAdapter: Makes CDG look like GoT's VersionedStore

Usage:
    from cortical.cdg.adapters import GoTAdapter
    from cortical.cdg import CDGStore

    # Use CDG through GoT-compatible interface
    adapter = GoTAdapter(CDGStore(Path("./data")))
    adapter.read("entity-id")  # Same API as VersionedStore
"""

# Adapters will be added as they are implemented
__all__ = []
