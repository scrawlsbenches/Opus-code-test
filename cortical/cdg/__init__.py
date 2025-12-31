"""
Cortical Distributed Graph (CDG) - Unified graph storage foundation.

CDG provides a partition-aware, ACID-compliant graph storage layer
that serves as the foundation for all graph implementations in Cortical.

Quick Start:
    from cortical.cdg import CDGStore, Entity, Edge

    store = CDGStore(Path("./data"))

    # Create and store an entity
    entity = Entity(id="E-001", entity_type="document")
    store.write(entity)

    # Read it back
    loaded = store.read("E-001")

Core Types:
    - Entity: Base class for all graph nodes
    - Node: Alias for Entity (CDG terminology)
    - Edge: Relationship between entities with type, weight, confidence

Storage:
    - CDGStore: Partition-aware storage with ACID guarantees
    - Transaction: Snapshot isolation with optimistic locking
    - TransactionState: State machine for transaction lifecycle

See Also:
    - docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md for full specification
"""

from .types import Entity, Node, Edge, VALID_EDGE_TYPES
from .transaction import Transaction, TransactionState, generate_transaction_id
from .errors import (
    CDGError,
    ValidationError,
    CorruptionError,
    TransactionError,
    ConflictError,
)
from .config import CDGConfig, DurabilityMode

__all__ = [
    # Types
    "Entity",
    "Node",
    "Edge",
    "VALID_EDGE_TYPES",
    # Transactions
    "Transaction",
    "TransactionState",
    "generate_transaction_id",
    # Errors
    "CDGError",
    "ValidationError",
    "CorruptionError",
    "TransactionError",
    "ConflictError",
    # Configuration
    "CDGConfig",
    "DurabilityMode",
]
