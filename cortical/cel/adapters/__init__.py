"""
Adapters for integrating the Cognitive Event Lattice with external systems.

Adapters implement the bridge pattern - they wrap external systems
to conform to CEL protocols, enabling gradual migration and
interoperability.

Key Adapters:
    got.py - Bridge to existing GoT (Graph of Thought) system
    memory.py - In-memory adapter for testing
    file.py - File system adapters

Design Pattern:
    Adapters are thin wrappers that:
    1. Translate CEL protocol calls to external system calls
    2. Transform data formats between systems
    3. Handle error mapping and recovery

Migration Strategy:
    Adapters enable gradual migration from GoT to CEL:
    1. Create GoTAdapter that reads existing GoT data
    2. Run both systems in parallel (read GoT, write CEL)
    3. Gradually move reads to CEL
    4. Eventually deprecate GoT adapter
"""

from .got import (
    GoTEventAdapter,
    GoTEntityAdapter,
    GotBridgeEventStore,
)

__all__ = [
    'GoTEventAdapter',
    'GoTEntityAdapter',
    'GotBridgeEventStore',
]
