"""
CEL Performance Optimizations.

This package provides high-performance components for the Cognitive Event Lattice
designed to scale to millions of events while maintaining sub-millisecond latencies.

Key Components:
    EntityIndex: O(1) entity → events lookup
    OptimizedDAG: Heap-based topological sort
    SnapshotManager: Incremental snapshots for fast recovery
    BatchingWriter: Amortized O(1) writes

Usage:
    from cortical.cel.performance import (
        EntityIndex,
        OptimizedDAG,
        SnapshotManager,
        StreamingEventStore,
    )

    # Create optimized event store
    store = StreamingEventStore(
        base_path=Path(".cel"),
        entity_index=EntityIndex(),
        snapshot_manager=SnapshotManager(interval=1000),
    )
"""

from .entity_index import EntityIndex, ConceptIndex, TemporalIndex
from .optimized_dag import OptimizedDAG, HeapTopologicalSort
from .snapshots import SnapshotManager, Snapshot, SnapshotConfig
from .streaming_store import StreamingEventStore, BatchingWriter

__all__ = [
    # Indexes
    'EntityIndex',
    'ConceptIndex',
    'TemporalIndex',
    # DAG
    'OptimizedDAG',
    'HeapTopologicalSort',
    # Snapshots
    'SnapshotManager',
    'Snapshot',
    'SnapshotConfig',
    # Storage
    'StreamingEventStore',
    'BatchingWriter',
]
