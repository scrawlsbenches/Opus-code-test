"""
GoT (Graph of Thought) - Transactional task and decision tracking system.

This package provides ACID-compliant transaction support for the Graph of Thought
system, enabling reliable concurrent access from multiple agents.

Key components:
- TransactionManager: Main entry point for transactional operations
- Transaction: Transaction object with snapshot isolation
- Entity types: Task, Decision, Edge
- VersionedStore: File-based storage with checksums and versioning
- WALManager: Write-ahead log for crash recovery
"""

from .errors import (
    GoTError,
    TransactionError,
    ConflictError,
    CorruptionError,
    SyncError,
    NotFoundError,
    ValidationError,
)

from .checksums import (
    compute_checksum,
    verify_checksum,
    compute_file_checksum,
    verify_file_checksum,
)

from .types import (
    Entity,
    Task,
    Decision,
    Edge,
)

from .transaction import (
    Transaction,
    TransactionState,
    generate_transaction_id,
)

from .versioned_store import VersionedStore

from .wal import WALManager

from .tx_manager import (
    TransactionManager,
    CommitResult,
    Conflict,
    RecoveryResult,
    ProcessLock,
)

__all__ = [
    # Errors
    'GoTError',
    'TransactionError',
    'ConflictError',
    'CorruptionError',
    'SyncError',
    'NotFoundError',
    'ValidationError',
    # Checksums
    'compute_checksum',
    'verify_checksum',
    'compute_file_checksum',
    'verify_file_checksum',
    # Entity types
    'Entity',
    'Task',
    'Decision',
    'Edge',
    # Transaction
    'Transaction',
    'TransactionState',
    'generate_transaction_id',
    # Storage
    'VersionedStore',
    # WAL
    'WALManager',
    # Transaction Manager
    'TransactionManager',
    'CommitResult',
    'Conflict',
    'RecoveryResult',
    'ProcessLock',
]
