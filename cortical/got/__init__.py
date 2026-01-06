"""
GoT (Graph of Thought) - Transactional task and decision tracking system.

This package provides ACID-compliant transaction support for the Graph of Thought
system, enabling reliable concurrent access from multiple agents.

Key components:
- TransactionManager: Main entry point for transactional operations
- Transaction: Transaction object with snapshot isolation
- Entity types: Task, Decision, Edge, Sprint, Epic, Handoff
- WALManager: Write-ahead log for crash recovery

Storage is delegated to CDG (Cortical Distributed Graph) layer.
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

from cortical.utils.checksums import (
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
    Sprint,
    Epic,
    Handoff,
    # TypedDict schemas for nested structures
    TaskProperties,
    TaskMetadata,
    SprintGoal,
    SprintMetadata,
    EpicPhase,
    EpicMetadata,
    HandoffContext,
    HandoffResult,
    DocumentMetadata,
    # Type aliases
    SprintGoals,
    EpicPhases,
    # Edge type constants
    EdgeTypes,
    VALID_EDGE_TYPES,
)

from .transaction import (
    Transaction,
    TransactionState,
    generate_transaction_id,
)

from .wal import WALManager

from .tx_manager import (
    TransactionManager,
    CommitResult,
    Conflict,
    ProcessLock,
)

from .recovery import (
    RecoveryManager,
    RecoveryResult,
    RepairResult,
)

from .sync import (
    SyncManager,
    SyncResult,
    SyncStatus,
)

from .conflict import (
    ConflictResolver,
    ConflictStrategy,
    SyncConflict,
)

from .config import (
    DurabilityMode,
    GoTConfig,
)

from .api import (
    GoTManager,
    TransactionContext,
    generate_task_id,
    generate_decision_id,
)

from .query_api import QueryAPI

from .protocol import GoTBackend

from .entity_schemas import (
    get_schema_for_entity_type,
    list_entity_types,
    TaskSchema,
    DecisionSchema,
    SprintSchema,
    EpicSchema,
    EdgeSchema,
    HandoffSchema,
    ClaudeMdLayerSchema,
    ClaudeMdVersionSchema,
    TeamSchema,
)

from .orphan import (
    OrphanDetector,
    OrphanReport,
    ConnectionSuggestion,
    SprintSuggestion,
    check_orphan_on_create,
    generate_orphan_report,
)

from .query_builder import (
    Query,
    QueryPlan,
    QueryConfig,  # Per-query configuration
    AggregateFunction,
    Count,
    Collect,
    Sum,
    Avg,
    Min,
    Max,
    # Query logging
    QueryLogLevel,
    set_query_log_level,
    get_query_log_level,
    set_slow_query_threshold,
    get_slow_query_threshold,
    # Query validation
    QueryValidationError,
    enable_syntax_validation,
    disable_syntax_validation,
)

from .indexer import (
    QueryIndexManager,
    IndexEntry,
    IndexStats,
)

from .graph_walker import GraphWalker, WalkerPlan

from .path_finder import PathFinder, PathSearchResult, PathPlan

from .pattern_matcher import (
    Pattern,
    PatternMatcher,
    PatternMatch,
    PatternSearchResult,
    PatternPlan,
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
    'Sprint',
    'Epic',
    'Handoff',
    # TypedDict schemas for nested structures
    'TaskProperties',
    'TaskMetadata',
    'SprintGoal',
    'SprintMetadata',
    'EpicPhase',
    'EpicMetadata',
    'HandoffContext',
    'HandoffResult',
    'DocumentMetadata',
    'SprintGoals',
    'EpicPhases',
    # Edge type constants
    'EdgeTypes',
    'VALID_EDGE_TYPES',
    # Transaction
    'Transaction',
    'TransactionState',
    'generate_transaction_id',
    # WAL
    'WALManager',
    # Transaction Manager
    'TransactionManager',
    'CommitResult',
    'Conflict',
    'ProcessLock',
    # Recovery
    'RecoveryManager',
    'RecoveryResult',
    'RepairResult',
    # Sync
    'SyncManager',
    'SyncResult',
    'SyncStatus',
    # Conflict Resolution
    'ConflictResolver',
    'ConflictStrategy',
    'SyncConflict',
    # Configuration
    'DurabilityMode',
    'GoTConfig',
    # High-level API
    'GoTManager',
    'TransactionContext',
    'generate_task_id',
    'generate_decision_id',
    # Query API
    'QueryAPI',
    # Protocol
    'GoTBackend',
    # Entity Schemas (for querying - schema infrastructure lives in CDG)
    'get_schema_for_entity_type',
    'list_entity_types',
    'TaskSchema',
    'DecisionSchema',
    'SprintSchema',
    'EpicSchema',
    'EdgeSchema',
    'HandoffSchema',
    'ClaudeMdLayerSchema',
    'ClaudeMdVersionSchema',
    'TeamSchema',
    # Orphan Detection
    'OrphanDetector',
    'OrphanReport',
    'ConnectionSuggestion',
    'SprintSuggestion',
    'check_orphan_on_create',
    'generate_orphan_report',
    # Query Builder
    'Query',
    'QueryPlan',
    'AggregateFunction',
    'Count',
    'Collect',
    'Sum',
    'Avg',
    'Min',
    'Max',
    # Query Configuration
    'QueryConfig',
    # Query Logging
    'QueryLogLevel',
    'set_query_log_level',
    'get_query_log_level',
    'set_slow_query_threshold',
    'get_slow_query_threshold',
    # Query Validation
    'QueryValidationError',
    'enable_syntax_validation',
    'disable_syntax_validation',
    # Query Indexing
    'QueryIndexManager',
    'IndexEntry',
    'IndexStats',
    # Graph Walker
    'GraphWalker',
    'WalkerPlan',
    # Path Finder
    'PathFinder',
    'PathSearchResult',
    'PathPlan',
    # Pattern Matcher
    'Pattern',
    'PatternMatcher',
    'PatternMatch',
    'PatternSearchResult',
    'PatternPlan',
]
