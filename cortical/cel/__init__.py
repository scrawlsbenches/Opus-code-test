"""
Cognitive Event Lattice (CEL) - The Double Helix of Wisdom and Sanity.

A self-referential, self-maintaining cognitive substrate for machine reasoning.

Architecture:
    The CEL is built on two intertwined strands:

    WISDOM STRAND (cortical.cel.wisdom)
        Knowledge, memory, relationships - what the system knows.
        - Events: Immutable records of observations and intentions
        - DAG: Merkle-linked directed acyclic graph of events
        - Materialization: Lazy projection of events into entities
        - Semantic: Bloom filters and embeddings for fast access

    SANITY STRAND (cortical.cel.sanity)
        Validation, health, evolution - keeping the system coherent.
        - Health: Self-monitoring and metric collection
        - Migration: Schema evolution without data loss
        - Compaction: Semantic compression preserving meaning
        - Repair: Self-healing capabilities

Design Principles:
    1. EVENTS ARE PRIMARY - Store what happened, derive what is
    2. TEMPORAL ADDRESSING - Everything exists "as of" some event
    3. DEPENDENCY INJECTION - All components accept interfaces, not implementations
    4. INVERSION OF CONTROL - Container manages lifecycle and wiring
    5. SELF-REFERENCE VIA INDIRECTION - Reference hashes, not entities
    6. LAYERED MATERIALIZATION - Different views for different speeds

The Double Helix Metaphor:
    Wisdom without sanity leads to corruption (inconsistent state).
    Sanity without wisdom leads to empty process (no actual knowledge).
    The two strands wrap around each other, each incomplete without the other.

Example:
    >>> from cortical.cel import create_lattice, Container
    >>>
    >>> # Create with default filesystem backend
    >>> container = Container()
    >>> lattice = create_lattice(container, path=".got")
    >>>
    >>> # Create a task (intention) with temporal reference
    >>> task = lattice.intend(
    ...     "Optimize storage format",
    ...     references_system_at=lattice.current_horizon,
    ...     after=["T-20251228-..."]  # Wait for this to complete
    ... )
    >>>
    >>> # Query past state (even after modifications)
    >>> old_state = lattice.materialize("T-xxx", at=task.snapshot_horizon)
"""

from .core.protocols import (
    EventStore,
    Materializer,
    HealthMonitor,
    MigrationEngine,
    SemanticIndex,
)
from .core.events import (
    CognitiveEvent,
    EventType,
    Observation,
    Intention,
    Fulfillment,
    Invalidation,
)
from .core.references import (
    TemporalReference,
    DeferredReference,
    CausalLink,
    MerkleRoot,
)
from .container import Container, Lifecycle, create_lattice

__all__ = [
    # Protocols (interfaces)
    'EventStore',
    'Materializer',
    'HealthMonitor',
    'MigrationEngine',
    'SemanticIndex',
    # Event types
    'CognitiveEvent',
    'EventType',
    'Observation',
    'Intention',
    'Fulfillment',
    'Invalidation',
    # Reference types
    'TemporalReference',
    'DeferredReference',
    'CausalLink',
    'MerkleRoot',
    # Container & DI
    'Container',
    'Lifecycle',
    'create_lattice',
]

__version__ = '0.1.0'
