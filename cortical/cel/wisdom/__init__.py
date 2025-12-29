"""
Wisdom Strand - Knowledge, memory, and relationships.

The Wisdom strand handles WHAT THE SYSTEM KNOWS:
- Events: Immutable records of what happened
- DAG: The causal structure of events
- Materialization: Deriving current state from events
- Semantic: Fast access via probabilistic structures

This strand is paired with the Sanity strand (health, migration,
compaction) to form the complete Double Helix.

Modules:
    dag.py - Merkle DAG implementation for event storage
    materializer.py - Event-to-entity projection
    semantic.py - Bloom filters and embeddings

Design Principle:
    Store EVENTS, derive STATE.
    Events are immutable truth; state is computed projection.
"""

from .dag import MerkleDAG, FileSystemEventStore
from .materializer import CachingMaterializer, EntityReducerRegistry
from .semantic import BloomSemanticIndex, HybridSemanticIndex

__all__ = [
    # DAG
    'MerkleDAG',
    'FileSystemEventStore',
    # Materializer
    'CachingMaterializer',
    'EntityReducerRegistry',
    # Semantic
    'BloomSemanticIndex',
    'HybridSemanticIndex',
]
