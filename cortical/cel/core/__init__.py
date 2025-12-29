"""
Core abstractions for the Cognitive Event Lattice.

This module defines the fundamental protocols (interfaces) and data types
that all implementations must adhere to. These are the contracts that
enable dependency injection and future extensibility.

Key Design Decision: Protocol Classes
    We use Python's Protocol (structural subtyping) rather than ABC
    (nominal subtyping) to allow duck typing while maintaining type safety.
    This means any class with the right methods is compatible, even if
    it doesn't explicitly inherit from our protocols.

Contents:
    protocols.py - Interface definitions (EventStore, Materializer, etc.)
    events.py - Event type definitions (Observation, Intention, etc.)
    references.py - Temporal reference types (TemporalReference, etc.)
"""

from .protocols import (
    EventStore,
    Materializer,
    HealthMonitor,
    MigrationEngine,
    SemanticIndex,
    CompactionStrategy,
    EventReducer,
)
from .events import (
    CognitiveEvent,
    EventType,
    Observation,
    Intention,
    Fulfillment,
    Invalidation,
    Compaction,
    MetaCognition,
)
from .references import (
    TemporalReference,
    DeferredReference,
    CausalLink,
    MerkleRoot,
    EventHorizon,
)

__all__ = [
    # Protocols
    'EventStore',
    'Materializer',
    'HealthMonitor',
    'MigrationEngine',
    'SemanticIndex',
    'CompactionStrategy',
    'EventReducer',
    # Events
    'CognitiveEvent',
    'EventType',
    'Observation',
    'Intention',
    'Fulfillment',
    'Invalidation',
    'Compaction',
    'MetaCognition',
    # References
    'TemporalReference',
    'DeferredReference',
    'CausalLink',
    'MerkleRoot',
    'EventHorizon',
]
