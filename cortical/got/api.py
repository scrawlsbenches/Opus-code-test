"""
High-level API for Graph of Thought operations.

Provides convenient context managers and methods for working with the GoT
transactional system. This is the primary user-facing interface.

Example:
    >>> from cortical.core.bootstrap import create_container
    >>> container = create_container(got_dir=Path(".got"))
    >>> manager = container.resolve(GoTManager)
    >>>
    >>> # Single-operation methods
    >>> task = manager.create_task("Implement feature", priority="high")
    >>>
    >>> # Transactional context
    >>> with manager.transaction() as tx:
    ...     task = tx.create_task("Another task", priority="medium")
    ...     tx.update_task(task.id, status="in_progress")
    ...     # Auto-commits on success, rolls back on exception
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from cortical.utils.id_generation import (
    generate_task_id,
    generate_decision_id,
    generate_sprint_id,
    generate_epic_id,
    generate_handoff_id,
    generate_claudemd_layer_id,
    generate_claudemd_version_id,
    generate_document_id,
)
from cortical.cdg.transaction_manager import CDGTransactionManager, CommitResult
from cortical.cdg.recovery import CDGRecoveryManager
from cortical.cdg.config import CDGConfig
from cortical.common.recovery_types import RecoveryResult
from .sync import SyncManager, SyncResult
from .entity_schemas import get_valid_statuses
from .types import Task, Decision, Edge, Entity, Sprint, Epic, Handoff, ClaudeMdLayer, ClaudeMdVersion, Document, EdgeTypes, KnowledgeTransfer
from cortical.cdg.transaction import Transaction
from .errors import TransactionError, CorruptionError
from .config import DurabilityMode
from .query_api import QueryAPI
from cortical.cdg.schema import SchemaRegistry
from .validation import (
    validate_entity_id,
    validate_edge_relationship,
    validate_sprint_id_current_format,
    validate_entity_file,
)

logger = logging.getLogger(__name__)

# ID generation functions are imported from cortical.utils.id_generation
# (canonical source for all ID generation across the codebase)


def _validate_sprint_id_format(sprint_id: str) -> None:
    """
    Validate sprint ID format and log warning if non-standard.

    Supports three formats:
    - Generated (current): S-YYYYMMDD-HHMMSS-hash (e.g., S-20251227-211213-ae934eab)
    - Legacy verbose: S-sprint-NNN-slug (e.g., S-sprint-017-spark-slm)
    - Legacy short: S-NNN (e.g., S-022, S-028)

    Args:
        sprint_id: Sprint identifier to validate

    Note:
        This is a non-breaking validation - logs warning but does not raise errors.
        All formats are supported for backward compatibility.
    """
    import re

    # Check if ID matches any known format
    generated_pattern = r'^S-\d{8}-\d{6}-[a-f0-9]{8}$'
    legacy_verbose_pattern = r'^S-sprint-\d+(-[\w-]+)?$'  # Slug is optional
    legacy_short_pattern = r'^S-\d+$'

    if (re.match(generated_pattern, sprint_id) or
        re.match(legacy_verbose_pattern, sprint_id) or
        re.match(legacy_short_pattern, sprint_id)):
        return  # Valid format

    # Log warning for unrecognized format (but don't break)
    logger.warning(
        f"Sprint ID '{sprint_id}' does not match standard formats. "
        f"Expected: S-YYYYMMDD-HHMMSS-hash (generated), "
        f"S-sprint-NNN-slug (legacy verbose), or S-NNN (legacy short)"
    )


def _require_current_sprint_id_format(entity_id: str) -> None:
    """
    Validate that sprint IDs use the current generated format.

    This is a STRICT validation that REJECTS legacy formats when creating
    new edges to sprints. Legacy sprints can still be read, but new
    relationships should only link to current-format sprint IDs.

    Args:
        entity_id: Entity ID to validate (only checked if starts with 'S-')

    Raises:
        ValueError: If entity_id is a legacy-format sprint ID

    Note:
        This is a compatibility wrapper that delegates to entity_validation module.
        The full validation logic is in cortical/got/entity_validation.py.
    """
    # Delegate to the canonical validation function
    validate_sprint_id_current_format(entity_id)


class GoTManager:
    """
    High-level API for Graph of Thought operations.

    Provides context managers for transactional operations
    and convenient methods for common tasks.

    Example:
        from cortical.core.bootstrap import create_container
        container = create_container(got_dir=Path(".got"))
        manager = container.resolve(GoTManager)

        with manager.transaction() as tx:
            task = tx.create_task("Implement feature", priority="high")
            tx.update_task(task.id, status="in_progress")
        # Auto-commits on success, rolls back on exception

    Caching:
        Entity caching is handled at the CDGStore layer for 10-50x faster
        repeated queries. Cache is automatically invalidated on writes.

        # Get cache statistics
        stats = manager.cache_stats()
        # {'hits': 150, 'misses': 50, 'hit_rate': 0.75, 'size': 80}

        # Clear cache manually
        manager.cache_clear()
    """

    def __init__(
        self,
        durability: DurabilityMode = DurabilityMode.BALANCED,
        cache_enabled: bool = True,  # Deprecated: caching now handled by CDGStore
        *,
        tx_manager: CDGTransactionManager,
        schema_registry: SchemaRegistry,
    ):
        """
        Initialize GoT manager with injected dependencies.

        Args:
            durability: Durability mode controlling fsync behavior (default: BALANCED)
            cache_enabled: DEPRECATED - Caching is now handled by CDGStore.
                          This parameter is ignored but kept for backwards compatibility.
            tx_manager: REQUIRED - Injected CDGTransactionManager instance
            schema_registry: REQUIRED - SchemaRegistry for entity validation (from Container)

        Raises:
            TypeError: If required dependencies are missing or wrong type

        Example:
            # The only supported way to get a GoTManager:
            from cortical.core.bootstrap import create_container

            container = create_container(got_dir=Path(".got"))
            got_manager = container.resolve(GoTManager)
        """
        # Validate required dependencies
        if not isinstance(tx_manager, CDGTransactionManager):
            raise TypeError(
                f"tx_manager is required and must be CDGTransactionManager instance, got {type(tx_manager).__name__}"
            )
        if not isinstance(schema_registry, SchemaRegistry):
            raise TypeError(
                f"schema_registry is required and must be SchemaRegistry instance, got {type(schema_registry).__name__}"
            )

        self.durability = durability
        self.tx_manager = tx_manager
        self._schema_registry = schema_registry
        self._sync_manager = None  # Lazy initialization
        self._recovery_manager = None  # Lazy initialization
        self._query_api = None  # Lazy initialization

        # Cache is now handled by CDGStore at the storage layer
        # GoTManager delegates cache_stats() and cache_clear() to the store

        logger.debug(
            f"GoTManager initialized with durability={durability.value}"
        )

    @property
    def base_dir(self) -> Path:
        """
        Get base directory for GoT storage (e.g., .got/).

        Derived from CDG store's store_dir parent.
        """
        return self.tx_manager.store.store_dir.parent

    @property
    def got_dir(self) -> Path:
        """
        Deprecated: Use base_dir instead.

        This property is retained for backward compatibility with code
        that accessed manager.got_dir. New code should use base_dir.
        """
        return self.base_dir

    @property
    def entities_dir(self) -> Path:
        """
        Get entities directory (e.g., .got/entities/).

        Derived from CDG store's store_dir.
        """
        return self.tx_manager.store.store_dir

    @property
    def sync_manager(self) -> SyncManager:
        """Get sync manager (lazy initialization)."""
        if self._sync_manager is None:
            self._sync_manager = SyncManager(self.base_dir)
        return self._sync_manager

    @property
    def recovery_manager(self) -> CDGRecoveryManager:
        """Get recovery manager (lazy initialization)."""
        if self._recovery_manager is None:
            # CDG handles index management via CDGIndexManager
            # No callbacks needed - indexes are maintained automatically
            config = CDGConfig.for_got()

            self._recovery_manager = CDGRecoveryManager(
                store_dir=self.entities_dir,
                config=config,
                entity_factory=lambda d: d  # GoT uses its own entity factory
            )
        return self._recovery_manager

    @property
    def query_api(self) -> QueryAPI:
        """
        Get query API (lazy initialization).

        The QueryAPI provides read-only query operations for tasks,
        edges, and other entities. Methods on GoTManager delegate
        to this API for query operations.
        """
        if self._query_api is None:
            self._query_api = QueryAPI(self)
        return self._query_api

    def _iter_entities_by_prefix(self, prefix: str):
        """
        Iterate entities by ID prefix using CDGStore.

        Uses the store's iter_entities() method which works with both
        disk-based and in-memory storage through the FileSystem abstraction.

        Args:
            prefix: Entity ID prefix (e.g., "T-", "E-", "D-", "S-", "H-", etc.)

        Yields:
            Entity objects matching the prefix
        """
        yield from self.tx_manager.store.iter_entities(prefix=prefix)

    def _read_entity_file(self, entity_file: Path, prefix: str):
        """Read an entity file based on its prefix type."""
        if prefix == "T-":
            return self._read_task_file(entity_file)
        elif prefix == "E-":
            return self._read_edge_file(entity_file)
        elif prefix == "D-":
            return self._read_decision_file(entity_file)
        elif prefix == "S-":
            return self._read_sprint_file(entity_file)
        elif prefix == "H-":
            return self._read_handoff_file(entity_file)
        elif prefix == "DOC-":
            return self._read_document_file(entity_file)
        elif prefix == "EPIC-":
            return self._read_epic_file(entity_file)
        elif prefix == "KT-":
            return self._read_knowledge_transfer_file(entity_file)
        elif prefix.startswith("CML"):
            return self._read_claudemd_layer_file(entity_file)
        return None

    # ==================== Cache Methods (Delegated to CDGStore) ====================

    def cache_clear(self) -> None:
        """Clear the storage layer cache."""
        store = getattr(self.tx_manager, 'store', None)
        if store is not None and hasattr(store, 'cache_clear'):
            store.cache_clear()

    def cache_configure(self, ttl: Optional[float] = None, max_size: Optional[int] = None) -> None:
        """
        Configure cache behavior.

        Args:
            ttl: Time-to-live in seconds for cached entries. None disables TTL.
            max_size: Maximum number of entries. Oldest entries are evicted when exceeded.
                     None means unlimited.
        """
        store = getattr(self.tx_manager, 'store', None)
        if store is not None and hasattr(store, 'cache_configure'):
            store.cache_configure(ttl=ttl, max_size=max_size)

    def cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics from the storage layer.

        Returns:
            Dictionary with hits, misses, hit_rate, size, enabled, ttl, and max_size
        """
        store = getattr(self.tx_manager, 'store', None)
        if store is not None and hasattr(store, 'cache_stats'):
            return store.cache_stats()
        return {
            'hits': 0,
            'misses': 0,
            'hit_rate': 0.0,
            'size': 0,
            'enabled': False,
            'ttl': None,
            'max_size': None,
        }

    def load_all(self) -> Dict[str, int]:
        """
        Pre-load all entities into memory for sub-millisecond queries.

        This is useful for read-heavy workloads like CLI analyze commands
        where you want to pay the I/O cost upfront and then have fast
        access to all entities.

        Returns:
            Dictionary with counts of each entity type loaded

        Example:
            >>> container = create_container(got_dir=Path(".got"))
            >>> manager = container.resolve(GoTManager)
            >>> counts = manager.load_all()
            >>> print(f"Loaded {counts['tasks']} tasks, {counts['edges']} edges")

            # Now all queries will use cached entities
            >>> Query(manager).tasks().execute()  # Sub-millisecond!
        """
        counts = {
            'tasks': 0,
            'decisions': 0,
            'sprints': 0,
            'epics': 0,
            'edges': 0,
            'handoffs': 0,
        }

        # Load all tasks (iterating populates CDGStore cache)
        for task in self.list_all_tasks():
            counts['tasks'] += 1

        # Load all decisions
        for decision in self.list_decisions():
            counts['decisions'] += 1

        # Load all sprints
        for sprint in self.list_sprints():
            counts['sprints'] += 1

        # Load all epics
        for epic in self.list_epics():
            counts['epics'] += 1

        # Load all edges
        for edge in self.list_edges():
            counts['edges'] += 1

        # Load all handoffs
        for handoff in self.list_handoffs():
            counts['handoffs'] += 1

        return counts

    # ==================== Transaction Methods ====================

    def transaction(self, read_only: bool = False) -> TransactionContext:
        """
        Start a transaction context.

        Args:
            read_only: If True, rollback instead of commit on exit

        Returns:
            TransactionContext for use with 'with' statement
        """
        return TransactionContext(self.tx_manager, read_only=read_only, got_manager=self)

    def create_task(
        self,
        title: str,
        priority: str = "medium",
        status: str = "pending",
        description: str = "",
        **properties
    ) -> Task:
        """
        Create a task in a single-operation transaction.

        Args:
            title: Task title
            priority: Priority level (low, medium, high, critical)
            status: Task status (pending, in_progress, completed, blocked)
            description: Task description
            **properties: Additional task properties

        Returns:
            Created Task object

        Raises:
            TransactionError: If commit fails
        """
        with self.transaction() as tx:
            task = tx.create_task(
                title=title,
                priority=priority,
                status=status,
                description=description,
                **properties
            )
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID (read-only).

        Caching is handled at the CDGStore layer.

        Args:
            task_id: Task identifier

        Returns:
            Task object or None if not found
        """
        with self.transaction(read_only=True) as tx:
            return tx.get_task(task_id)

    def update_task(self, task_id: str, **updates) -> Task:
        """
        Update a task in a single-operation transaction.

        Args:
            task_id: Task identifier
            **updates: Fields to update (status, priority, title, etc.)

        Returns:
            Updated Task object

        Raises:
            TransactionError: If commit fails or task not found
        """
        with self.transaction() as tx:
            task = tx.update_task(task_id, **updates)
        return task

    def create_decision(
        self,
        title: str,
        rationale: str,
        affects: Optional[List[str]] = None,
        **properties
    ) -> Decision:
        """
        Create a decision in a single-operation transaction.

        Args:
            title: Decision title
            rationale: Rationale for the decision
            affects: List of entity IDs affected by this decision
            **properties: Additional decision properties

        Returns:
            Created Decision object

        Raises:
            TransactionError: If commit fails
        """
        with self.transaction() as tx:
            decision = tx.create_decision(
                title=title,
                rationale=rationale,
                affects=affects or [],
                **properties
            )
        return decision

    def log_decision(
        self,
        title: str,
        rationale: str,
        affects: Optional[List[str]] = None,
        **properties
    ) -> Decision:
        """
        Alias for create_decision() - matches CLI 'decision log' command.

        Args:
            title: Decision title
            rationale: Rationale for the decision
            affects: List of entity IDs affected
            **properties: Additional decision properties

        Returns:
            Created Decision object
        """
        return self.create_decision(title, rationale, affects, **properties)

    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """
        Get a decision by ID (read-only).

        Args:
            decision_id: Decision identifier (D-...)

        Returns:
            Decision object or None if not found
        """
        with self.transaction(read_only=True) as tx:
            return tx.get_decision(decision_id)

    def delete_decision(self, decision_id: str, force: bool = False) -> None:
        """
        Delete a decision and all its connected edges atomically.

        Uses transaction to ensure all-or-nothing deletion semantics.
        If any deletion fails, the entire operation is rolled back.

        Args:
            decision_id: Decision identifier to delete (D-...)
            force: If False, raise error if decision has edges

        Raises:
            TransactionError: If decision not found or has edges (and force=False)
        """
        # Delete atomically within a transaction
        # Cache invalidation is handled automatically by CDGStore
        with self.transaction() as tx:
            tx.delete_decision(decision_id, force=force)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        reason: str = "",
        validate_refs: bool = True,
        validate_relationship: bool = True
    ) -> Edge:
        """
        Add an edge between entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            edge_type: Edge type (DEPENDS_ON, BLOCKS, etc.)
            weight: Edge weight (0.0-1.0)
            reason: Why this relationship exists (context capture)
            validate_refs: If True, verify source and target entities exist
                          (default: True for referential integrity)
            validate_relationship: If True, validate entity types can be connected
                                  by this edge type (default: True)

        Returns:
            Created Edge object

        Raises:
            TransactionError: If commit fails
            ValueError: If validate_refs=True and source/target doesn't exist
            ValueError: If entity IDs have invalid format (including legacy sprint IDs)
            ValueError: If validate_relationship=True and relationship is not allowed

        Note:
            Entity validation includes:
            - ID format validation for all entity types
            - Legacy sprint ID rejection (S-NNN, S-sprint-NNN-*)
            - Relationship rules (which entity types can connect)
            - Self-reference prevention
        """
        # Full entity and relationship validation
        # This validates: ID formats, legacy sprint rejection, relationship rules, self-reference
        if validate_relationship:
            validate_edge_relationship(source_id, target_id, edge_type)
        else:
            # Still validate sprint IDs use current format
            _require_current_sprint_id_format(source_id)
            _require_current_sprint_id_format(target_id)

        # Optional FK validation
        if validate_refs:
            if not self.tx_manager.store.exists(source_id):
                raise ValueError(
                    f"Source entity not found: {source_id}. "
                    f"Use validate_refs=False to create edge without validation."
                )
            if not self.tx_manager.store.exists(target_id):
                raise ValueError(
                    f"Target entity not found: {target_id}. "
                    f"Use validate_refs=False to create edge without validation."
                )

        with self.transaction() as tx:
            edge = tx.add_edge(source_id, target_id, edge_type, weight=weight, reason=reason)
        return edge

    def add_dependency(self, task_id: str, depends_on_id: str) -> Edge:
        """
        Add a dependency edge between tasks.

        Args:
            task_id: Task that depends on another
            depends_on_id: Task that is depended on

        Returns:
            Created Edge object

        Raises:
            TransactionError: If commit fails
        """
        return self.add_edge(task_id, depends_on_id, EdgeTypes.DEPENDS_ON)

    def add_blocks(self, blocker_id: str, blocked_id: str) -> Edge:
        """
        Add a blocking edge between tasks.

        Args:
            blocker_id: Task that blocks another
            blocked_id: Task that is blocked

        Returns:
            Created Edge object

        Raises:
            TransactionError: If commit fails
        """
        return self.add_edge(blocker_id, blocked_id, EdgeTypes.BLOCKS)

    def delete_task(self, task_id: str, force: bool = False) -> None:
        """
        Delete a task and all its connected edges atomically.

        Uses transaction to ensure all-or-nothing deletion semantics.
        If any deletion fails, the entire operation is rolled back.

        Args:
            task_id: Task identifier to delete
            force: If False, raise error if task has dependents

        Raises:
            TransactionError: If task has dependents (and force=False) or task not found
        """
        # Get all edges connected to this task for cache invalidation
        outgoing, incoming = self.get_edges_for_task(task_id)
        all_edges = outgoing + incoming
        ids_to_invalidate = [task_id] + [edge.id for edge in all_edges]

        # Delete atomically within a transaction
        # Cache invalidation is handled automatically by CDGStore
        # Index updates happen inside TransactionContext.__exit__
        with self.transaction() as tx:
            tx.delete_task(task_id, force=force)
        # See: docs/design/cdg-transactional-indexing-design.md

    # Sprint management methods
    def create_sprint(
        self,
        title: str,
        number: Optional[int] = None,
        epic_id: str = "",
        **properties
    ) -> Sprint:
        """
        Create a sprint in a single-operation transaction.

        Args:
            title: Sprint title
            number: Optional sprint number (display metadata, not used in ID)
            epic_id: Optional epic ID this sprint belongs to
            **properties: Additional sprint properties

        Returns:
            Created Sprint object with timestamp-based ID (merge-free)

        Raises:
            TransactionError: If commit fails
        """
        with self.transaction() as tx:
            sprint = tx.create_sprint(
                title=title,
                number=number,
                epic_id=epic_id,
                **properties
            )
        return sprint

    def get_sprint(self, sprint_id: str) -> Optional[Sprint]:
        """
        Get a sprint by ID (read-only).

        Args:
            sprint_id: Sprint identifier

        Returns:
            Sprint object or None if not found
        """
        with self.transaction(read_only=True) as tx:
            sprint = tx.get_sprint(sprint_id)
        return sprint

    def update_sprint(self, sprint_id: str, **updates) -> Sprint:
        """
        Update a sprint in a single-operation transaction.

        Args:
            sprint_id: Sprint identifier
            **updates: Fields to update (status, title, goals, etc.)

        Returns:
            Updated Sprint object

        Raises:
            TransactionError: If commit fails or sprint not found
        """
        with self.transaction() as tx:
            sprint = tx.update_sprint(sprint_id, **updates)
        return sprint

    def list_sprints(
        self,
        status: Optional[str] = None,
        epic_id: Optional[str] = None
    ) -> List[Sprint]:
        """
        List sprints, optionally filtered by status or epic.

        Args:
            status: Filter by status ('available', 'in_progress', 'completed', etc.)
            epic_id: Filter by epic ID

        Returns:
            List of matching Sprint objects
        """
        return self.query_api.list_sprints(status=status, epic_id=epic_id)

    def get_current_sprint(self) -> Optional[Sprint]:
        """
        Get the currently active (in_progress) sprint.

        Returns:
            Sprint object or None if no sprint is in progress
        """
        sprints = self.list_sprints(status="in_progress")
        return sprints[0] if sprints else None

    def delete_sprint(self, sprint_id: str, force: bool = False) -> None:
        """
        Delete a sprint and all its connected edges.

        Args:
            sprint_id: Sprint identifier to delete
            force: If False, raise error if sprint has tasks

        Raises:
            TransactionError: If sprint has tasks (and force=False) or sprint not found
        """
        # Delete atomically within a transaction
        # Cache invalidation is handled automatically by CDGStore
        with self.transaction() as tx:
            tx.delete_sprint(sprint_id, force=force)

    def add_task_to_sprint(self, task_id: str, sprint_id: str) -> Edge:
        """
        Add a task to a sprint via CONTAINS edge.

        Args:
            task_id: Task identifier
            sprint_id: Sprint identifier

        Returns:
            Created Edge object

        Raises:
            TransactionError: If commit fails
        """
        return self.add_edge(sprint_id, task_id, EdgeTypes.CONTAINS)

    def get_sprint_tasks(self, sprint_id: str) -> List[Task]:
        """
        Get all tasks in a sprint.

        Args:
            sprint_id: Sprint identifier

        Returns:
            List of Task objects in the sprint
        """
        return self.query_api.get_sprint_tasks(sprint_id)

    def get_sprint_progress(self, sprint_id: str) -> dict:
        """
        Get sprint progress statistics.

        Args:
            sprint_id: Sprint identifier

        Returns:
            Dictionary with progress statistics
        """
        return self.query_api.get_sprint_progress(sprint_id)

    # Epic management methods
    def create_epic(
        self,
        title: str,
        epic_id: Optional[str] = None,
        **properties
    ) -> Epic:
        """
        Create an epic in a single-operation transaction.

        Args:
            title: Epic title
            epic_id: Optional custom epic ID (auto-generated if not provided)
            **properties: Additional epic properties

        Returns:
            Created Epic object

        Raises:
            TransactionError: If commit fails
        """
        with self.transaction() as tx:
            epic = tx.create_epic(
                title=title,
                epic_id=epic_id,
                **properties
            )
        return epic

    def get_epic(self, epic_id: str) -> Optional[Epic]:
        """
        Get an epic by ID (read-only).

        Args:
            epic_id: Epic identifier

        Returns:
            Epic object or None if not found
        """
        with self.transaction(read_only=True) as tx:
            epic = tx.get_epic(epic_id)
        return epic

    def update_epic(self, epic_id: str, **updates) -> Epic:
        """
        Update an epic in a single-operation transaction.

        Args:
            epic_id: Epic identifier
            **updates: Fields to update (status, title, phase, etc.)

        Returns:
            Updated Epic object

        Raises:
            TransactionError: If commit fails or epic not found
        """
        with self.transaction() as tx:
            epic = tx.update_epic(epic_id, **updates)
        return epic

    def list_epics(self, status: Optional[str] = None) -> List[Epic]:
        """
        List epics, optionally filtered by status.

        Args:
            status: Filter by status ('active', 'completed', 'on_hold')

        Returns:
            List of matching Epic objects
        """
        epics = []
        for epic in self._iter_entities_by_prefix("EPIC-"):
            if not isinstance(epic, Epic):
                continue

            # Apply filter
            if status is not None and epic.status != status:
                continue

            epics.append(epic)

        return epics

    def add_sprint_to_epic(self, sprint_id: str, epic_id: str) -> Edge:
        """
        Add a sprint to an epic via CONTAINS edge.

        Args:
            sprint_id: Sprint identifier
            epic_id: Epic identifier

        Returns:
            Created Edge object

        Raises:
            TransactionError: If commit fails
        """
        return self.add_edge(epic_id, sprint_id, EdgeTypes.CONTAINS)

    # Document management methods
    def create_document(
        self,
        path: str,
        title: str = "",
        doc_type: str = "general",
        tags: Optional[List[str]] = None,
        **properties
    ) -> Document:
        """
        Create a document entity in a single-operation transaction.

        Args:
            path: Relative path from repo root (e.g., "docs/architecture.md")
            title: Human-readable title
            doc_type: Document type (architecture, design, memory, etc.)
            tags: List of tags for organization
            **properties: Additional document properties

        Returns:
            Created Document object

        Raises:
            TransactionError: If commit fails
        """
        with self.transaction() as tx:
            doc = tx.create_document(
                path=path,
                title=title,
                doc_type=doc_type,
                tags=tags or [],
                **properties
            )
        return doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID (read-only).

        Args:
            doc_id: Document identifier (e.g., "DOC-docs-architecture-md")

        Returns:
            Document object or None if not found
        """
        with self.transaction(read_only=True) as tx:
            doc = tx.get_document(doc_id)
        return doc

    def get_document_by_path(self, path: str) -> Optional[Document]:
        """
        Get a document by its file path.

        Args:
            path: File path (e.g., "docs/architecture.md")

        Returns:
            Document object or None if not found
        """
        doc_id = generate_document_id(path)
        return self.get_document(doc_id)

    def update_document(self, doc_id: str, **updates) -> Document:
        """
        Update a document in a single-operation transaction.

        Args:
            doc_id: Document identifier
            **updates: Fields to update (title, tags, etc.)

        Returns:
            Updated Document object

        Raises:
            TransactionError: If commit fails or document not found
        """
        with self.transaction() as tx:
            doc = tx.update_document(doc_id, **updates)
        return doc

    def list_documents(
        self,
        doc_type: Optional[str] = None,
        tag: Optional[str] = None,
        is_stale: Optional[bool] = None
    ) -> List[Document]:
        """
        List documents, optionally filtered by type, tag, or staleness.

        Args:
            doc_type: Filter by document type
            tag: Filter by tag (document must have this tag)
            is_stale: Filter by staleness status

        Returns:
            List of matching Document objects
        """
        documents = []
        for doc in self._iter_entities_by_prefix("DOC-"):
            if not isinstance(doc, Document):
                continue

            # Apply filters
            if doc_type is not None and doc.doc_type != doc_type:
                continue
            if tag is not None and tag not in doc.tags:
                continue
            if is_stale is not None and doc.is_stale != is_stale:
                continue

            documents.append(doc)

        return documents

    def link_document_to_task(
        self,
        doc_id: str,
        task_id: str,
        edge_type: str = "DOCUMENTED_BY"
    ) -> Edge:
        """
        Link a document to a task via an edge.

        Edge types:
            - DOCUMENTED_BY: Task is documented by this document
            - PRODUCES: Task produces/creates this document
            - REFERENCES: Task references this document

        Args:
            doc_id: Document identifier
            task_id: Task identifier
            edge_type: Type of relationship (default: DOCUMENTED_BY)

        Returns:
            Created Edge object

        Raises:
            TransactionError: If commit fails
        """
        return self.add_edge(task_id, doc_id, edge_type)

    def get_documents_for_task(self, task_id: str) -> List[Document]:
        """
        Get all documents linked to a task.

        Args:
            task_id: Task identifier

        Returns:
            List of Document objects linked to the task
        """
        return self.query_api.get_documents_for_task(task_id)

    def get_tasks_for_document(self, doc_id: str) -> List[Task]:
        """
        Get all tasks linked to a document.

        Args:
            doc_id: Document identifier

        Returns:
            List of Task objects linked to the document
        """
        return self.query_api.get_tasks_for_document(doc_id)

    def _read_document_file(self, file_path: Path) -> Optional[Document]:
        """Read a document entity from file."""
        # Read from disk
        with open(file_path, "r") as f:
            wrapper = json.load(f)

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid document file {file_path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "document":
            return None

        return Document.from_dict(data)

    def sync(self) -> SyncResult:
        """
        Sync with remote (push/pull).

        Returns:
            SyncResult with sync status and conflicts
        """
        return self.sync_manager.sync()

    def recover(self) -> RecoveryResult:
        """
        Run recovery procedures.

        Returns:
            RecoveryResult with recovery details
        """
        return self.recovery_manager.recover()

    def find_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        title_contains: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Task]:
        """
        Find tasks matching criteria. Scans disk (no in-memory cache).

        Args:
            status: Filter by status ('pending', 'in_progress', 'completed', etc.)
            priority: Filter by priority ('low', 'medium', 'high', 'critical')
            title_contains: Filter by substring in title (case-insensitive)
            category: Filter by category (e.g., 'bugfix', 'feature', 'refactor')

        Returns:
            List of matching Task objects
        """
        return self.query_api.find_tasks(
            status=status,
            priority=priority,
            title_contains=title_contains,
            category=category,
        )

    def get_blockers(self, task_id: str) -> List[Task]:
        """
        Get all tasks that block the given task (have BLOCKS edge pointing to it).

        Args:
            task_id: The task being blocked

        Returns:
            List of blocking Task objects
        """
        return self.query_api.get_blockers(task_id)

    def get_dependents(self, task_id: str) -> List[Task]:
        """
        Get all tasks that depend on the given task (have DEPENDS_ON edge pointing to it).

        Args:
            task_id: The task being depended on

        Returns:
            List of dependent Task objects
        """
        return self.query_api.get_dependents(task_id)

    def list_all_tasks(self) -> List[Task]:
        """
        List all tasks in the store. Use sparingly - scans entire store.

        Returns:
            List of all Task objects
        """
        return self.query_api.list_all_tasks()

    def list_tasks(self, status: Optional[str] = None) -> List[Task]:
        """
        List tasks with optional status filter.

        Args:
            status: Optional status filter

        Returns:
            List of Task objects
        """
        return self.query_api.list_tasks(status=status)

    def list_edges(self) -> List[Edge]:
        """
        List all edges in the store.

        Returns:
            List of all Edge objects
        """
        return self.query_api.list_edges()

    def list_decisions(self) -> List[Decision]:
        """
        List all decisions in the store.

        Returns:
            List of all Decision objects
        """
        return self.query_api.list_decisions()

    def get_edges_for_task(self, task_id: str) -> Tuple[List[Edge], List[Edge]]:
        """
        Get all edges connected to a task.

        Args:
            task_id: Task to query

        Returns:
            Tuple of (outgoing_edges, incoming_edges)
        """
        return self.query_api.get_edges_for_task(task_id)

    def _read_task_file(self, path: Path) -> Optional[Task]:
        """
        Read and parse a task file.

        Args:
            path: Path to task JSON file

        Returns:
            Task object or None if not a task

        Raises:
            CorruptionError: If checksum verification fails
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing
        """
        # Read from disk - handle TOCTOU race where file may be deleted
        # between glob listing and actual read during concurrent operations
        try:
            with open(path, 'r', encoding='utf-8') as f:
                wrapper = json.load(f)
        except FileNotFoundError:
            # File was deleted between listing and read - expected during concurrency
            return None

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid task file {path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "task":
            return None

        return Task.from_dict(data)

    def _read_edge_file(self, path: Path) -> Optional[Edge]:
        """
        Read and parse an edge file.

        Args:
            path: Path to edge JSON file

        Returns:
            Edge object or None if not an edge

        Raises:
            CorruptionError: If checksum verification fails
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing
        """
        # Read from disk - handle TOCTOU race where file may be deleted
        # between glob listing and actual read during concurrent operations
        try:
            with open(path, 'r', encoding='utf-8') as f:
                wrapper = json.load(f)
        except FileNotFoundError:
            # File was deleted between listing and read - expected during concurrency
            return None

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid edge file {path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "edge":
            return None

        return Edge.from_dict(data)

    def _read_decision_file(self, path: Path) -> Optional[Decision]:
        """
        Read and parse a decision file.

        Args:
            path: Path to decision JSON file

        Returns:
            Decision object or None if not a decision

        Raises:
            CorruptionError: If checksum verification fails
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing
        """
        # Read from disk with TOCTOU protection
        try:
            with open(path, 'r', encoding='utf-8') as f:
                wrapper = json.load(f)
        except FileNotFoundError:
            # File was deleted between listing and read - expected during concurrency
            return None

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid decision file {path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "decision":
            return None

        return Decision.from_dict(data)

    def _read_sprint_file(self, path: Path) -> Optional[Sprint]:
        """
        Read and parse a sprint file.

        Args:
            path: Path to sprint JSON file

        Returns:
            Sprint object or None if not a sprint

        Raises:
            CorruptionError: If checksum verification fails
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing
        """
        # Read from disk with TOCTOU protection
        try:
            with open(path, 'r', encoding='utf-8') as f:
                wrapper = json.load(f)
        except FileNotFoundError:
            # File was deleted between listing and read - expected during concurrency
            return None

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid sprint file {path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "sprint":
            return None

        return Sprint.from_dict(data)

    def _read_epic_file(self, path: Path) -> Optional[Epic]:
        """
        Read and parse an epic file.

        Args:
            path: Path to epic JSON file

        Returns:
            Epic object or None if not an epic

        Raises:
            CorruptionError: If checksum verification fails
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing
        """
        # Read from disk
        with open(path, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid epic file {path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "epic":
            return None

        return Epic.from_dict(data)

    def _read_handoff_file(self, path: Path) -> Optional[Handoff]:
        """
        Read and parse a handoff file.

        Args:
            path: Path to handoff JSON file

        Returns:
            Handoff object or None if not a handoff

        Raises:
            CorruptionError: If checksum verification fails
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing
        """
        # Read from disk
        with open(path, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid handoff file {path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "handoff":
            return None

        return Handoff.from_dict(data)

    # Handoff management methods
    def initiate_handoff(
        self,
        source_agent: str,
        target_agent: str,
        task_id: str = "",
        instructions: str = "",
        context: Optional[Dict[str, Any]] = None,
        handoff_id: Optional[str] = None,
    ) -> Handoff:
        """
        Initiate a handoff to another agent.

        Args:
            source_agent: Agent initiating the handoff
            target_agent: Agent receiving the handoff
            task_id: Task being handed off (optional for session handoffs)
            instructions: Instructions for the target agent
            context: Additional context data
            handoff_id: Optional custom handoff ID (auto-generated if not provided)

        Returns:
            Created Handoff object

        Raises:
            TransactionError: If commit fails
        """
        with self.transaction() as tx:
            handoff = tx.initiate_handoff(
                source_agent=source_agent,
                target_agent=target_agent,
                task_id=task_id,
                instructions=instructions,
                context=context or {},
                handoff_id=handoff_id,
            )
        return handoff

    def accept_handoff(
        self,
        handoff_id: str,
        agent: str,
        acknowledgment: str = ""
    ) -> Handoff:
        """
        Accept a handoff.

        Args:
            handoff_id: Handoff identifier
            agent: Agent accepting the handoff
            acknowledgment: Optional acknowledgment message

        Returns:
            Updated Handoff object

        Raises:
            TransactionError: If commit fails or handoff not found
            NotFoundError: If handoff doesn't exist
        """
        with self.transaction() as tx:
            handoff = tx.accept_handoff(handoff_id, agent, acknowledgment)
        return handoff

    def complete_handoff(
        self,
        handoff_id: str,
        agent: str,
        result: Optional[Dict[str, Any]] = None,
        artifacts: Optional[List[str]] = None,
    ) -> Handoff:
        """
        Complete a handoff with results.

        Args:
            handoff_id: Handoff identifier
            agent: Agent completing the handoff
            result: Result data
            artifacts: List of artifact paths/identifiers

        Returns:
            Updated Handoff object

        Raises:
            TransactionError: If commit fails or handoff not found
            NotFoundError: If handoff doesn't exist
        """
        with self.transaction() as tx:
            handoff = tx.complete_handoff(
                handoff_id, agent, result or {}, artifacts or []
            )
        return handoff

    def reject_handoff(
        self,
        handoff_id: str,
        agent: str,
        reason: str = ""
    ) -> Handoff:
        """
        Reject a handoff.

        Args:
            handoff_id: Handoff identifier
            agent: Agent rejecting the handoff
            reason: Rejection reason

        Returns:
            Updated Handoff object

        Raises:
            TransactionError: If commit fails or handoff not found
            NotFoundError: If handoff doesn't exist
        """
        with self.transaction() as tx:
            handoff = tx.reject_handoff(handoff_id, agent, reason)
        return handoff

    def get_handoff(self, handoff_id: str) -> Optional[Handoff]:
        """
        Get a handoff by ID (read-only).

        Args:
            handoff_id: Handoff identifier

        Returns:
            Handoff object or None if not found
        """
        # Use store's read method if available (works with in-memory storage)
        store = getattr(self.tx_manager, 'store', None)
        if store is not None and hasattr(store, 'read'):
            entity = store.read(handoff_id)
            if entity is None:
                return None
            if isinstance(entity, Handoff):
                return entity
            return None

        # Fallback: read from disk
        handoff_file = self.entities_dir / f"{handoff_id}.json"
        if not handoff_file.exists():
            return None

        try:
            return self._read_handoff_file(handoff_file)
        except (CorruptionError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error reading handoff file {handoff_file}: {e}")
            return None

    def list_handoffs(
        self,
        status: Optional[str] = None,
        target_agent: Optional[str] = None,
        source_agent: Optional[str] = None,
    ) -> List[Handoff]:
        """
        List handoffs, optionally filtered.

        Args:
            status: Filter by status ('initiated', 'accepted', 'completed', 'rejected')
            target_agent: Filter by target agent
            source_agent: Filter by source agent

        Returns:
            List of matching Handoff objects
        """
        handoffs = []
        for handoff in self._iter_entities_by_prefix("H-"):
            if not isinstance(handoff, Handoff):
                continue

            # Apply filters
            if status is not None and handoff.status != status:
                continue
            if target_agent is not None and handoff.target_agent != target_agent:
                continue
            if source_agent is not None and handoff.source_agent != source_agent:
                continue

            handoffs.append(handoff)

        return handoffs

    # ==================== KnowledgeTransfer Methods ====================

    def create_knowledge_transfer(
        self,
        title: str,
        summary: str = "",
        status: str = "draft",
        **kwargs
    ) -> KnowledgeTransfer:
        """
        Create a knowledge transfer in a single-operation transaction.

        Args:
            title: KT title (required)
            summary: Executive summary
            status: Initial status (default: draft)
            **kwargs: Additional fields (session_id, sections, tags, code_refs, etc.)

        Returns:
            Created KnowledgeTransfer object

        Raises:
            TransactionError: If commit fails
        """
        from cortical.utils.id_generation import generate_kt_id

        with self.transaction() as tx:
            kt = tx.create_knowledge_transfer(
                title=title,
                summary=summary,
                status=status,
                **kwargs
            )
        return kt

    def get_knowledge_transfer(self, kt_id: str) -> Optional[KnowledgeTransfer]:
        """
        Get a knowledge transfer by ID (read-only).

        Args:
            kt_id: Knowledge transfer identifier

        Returns:
            KnowledgeTransfer object or None if not found
        """
        entity = self.tx_manager.store.read(kt_id)
        if entity is None:
            return None
        if not isinstance(entity, KnowledgeTransfer):
            return None
        return entity

    def update_knowledge_transfer(
        self, kt_id: str, **updates
    ) -> Optional[KnowledgeTransfer]:
        """
        Update a knowledge transfer in a single-operation transaction.

        Args:
            kt_id: Knowledge transfer identifier
            **updates: Fields to update (status, summary, sections, tags, etc.)

        Returns:
            Updated KnowledgeTransfer object or None if not found

        Raises:
            TransactionError: If commit fails
        """
        kt = self.get_knowledge_transfer(kt_id)
        if kt is None:
            return None

        # Apply updates
        for field, value in updates.items():
            if hasattr(kt, field):
                setattr(kt, field, value)

        # Increment version
        kt.version = getattr(kt, 'version', 0) + 1

        # Write back using transaction
        with self.transaction() as tx:
            tx.tx_manager.write(tx.tx, kt)

        return kt

    def append_knowledge_transfer_section(
        self, kt_id: str, section_title: str, content: str
    ) -> Optional[KnowledgeTransfer]:
        """
        Append a section to a knowledge transfer.

        Args:
            kt_id: Knowledge transfer identifier
            section_title: Section heading
            content: Section content

        Returns:
            Updated KnowledgeTransfer object or None if not found
        """
        kt = self.get_knowledge_transfer(kt_id)
        if kt is None:
            return None

        # Get existing sections
        sections = getattr(kt, 'sections', {}) or {}

        # Accumulate content if section already exists
        if section_title in sections:
            sections[section_title] = sections[section_title] + "\n\n" + content
        else:
            sections[section_title] = content

        return self.update_knowledge_transfer(kt_id, sections=sections)

    def list_knowledge_transfers(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[KnowledgeTransfer]:
        """
        List knowledge transfers with optional filtering.

        Args:
            status: Filter by status (draft, published, archived)
            tags: Filter by tags (must have all specified tags)

        Returns:
            List of matching KnowledgeTransfer entities
        """
        transfers = []
        for entity in self._iter_entities_by_prefix("KT-"):
            if not isinstance(entity, KnowledgeTransfer):
                continue

            # Apply filters
            if status is not None and entity.status != status:
                continue
            if tags is not None:
                if not all(tag in entity.tags for tag in tags):
                    continue

            transfers.append(entity)

        return transfers

    # ==================== ClaudeMdLayer Methods ====================

    def create_claudemd_layer(
        self,
        layer_type: str,
        section_id: str,
        title: str,
        content: str,
        layer_number: int = 0,
        inclusion_rule: str = "always",
        freshness_decay_days: int = 0,
        **properties
    ) -> ClaudeMdLayer:
        """
        Create a CLAUDE.md layer in a single-operation transaction.

        Args:
            layer_type: Type of layer (core, operational, contextual, persona, ephemeral)
            section_id: Section identifier (e.g., "architecture", "quick-start")
            title: Human-readable title
            content: Markdown content
            layer_number: Layer number 0-4 (default: 0)
            inclusion_rule: When to include (always, context, user_pref)
            freshness_decay_days: Days before content becomes stale (0 = never)
            **properties: Additional properties

        Returns:
            Created ClaudeMdLayer object

        Raises:
            TransactionError: If commit fails
        """
        with self.transaction() as tx:
            layer = tx.create_claudemd_layer(
                layer_type=layer_type,
                section_id=section_id,
                title=title,
                content=content,
                layer_number=layer_number,
                inclusion_rule=inclusion_rule,
                freshness_decay_days=freshness_decay_days,
                **properties
            )
        return layer

    def get_claudemd_layer(self, layer_id: str) -> Optional[ClaudeMdLayer]:
        """
        Get a CLAUDE.md layer by ID (read-only).

        Args:
            layer_id: Layer identifier

        Returns:
            ClaudeMdLayer object or None if not found
        """
        with self.transaction(read_only=True) as tx:
            layer = tx.get_claudemd_layer(layer_id)
        return layer

    def update_claudemd_layer(self, layer_id: str, **updates) -> ClaudeMdLayer:
        """
        Update a CLAUDE.md layer in a single-operation transaction.

        Args:
            layer_id: Layer identifier
            **updates: Fields to update

        Returns:
            Updated ClaudeMdLayer object

        Raises:
            TransactionError: If commit fails or layer not found
        """
        with self.transaction() as tx:
            layer = tx.update_claudemd_layer(layer_id, **updates)
        return layer

    def list_claudemd_layers(
        self,
        layer_type: Optional[str] = None,
        freshness_status: Optional[str] = None,
        inclusion_rule: Optional[str] = None
    ) -> List[ClaudeMdLayer]:
        """
        List CLAUDE.md layers with optional filters.

        Args:
            layer_type: Filter by layer type
            freshness_status: Filter by freshness (fresh, stale, regenerating)
            inclusion_rule: Filter by inclusion rule

        Returns:
            List of matching ClaudeMdLayer objects
        """
        with self.transaction(read_only=True) as tx:
            layers = tx.list_claudemd_layers(
                layer_type=layer_type,
                freshness_status=freshness_status,
                inclusion_rule=inclusion_rule
            )
        return layers

    def delete_claudemd_layer(self, layer_id: str) -> bool:
        """
        Delete a CLAUDE.md layer.

        Args:
            layer_id: Layer identifier

        Returns:
            True if deleted, False if not found
        """
        with self.transaction() as tx:
            result = tx.delete_claudemd_layer(layer_id)
        return result

    def _read_claudemd_layer_file(self, path: Path) -> Optional[ClaudeMdLayer]:
        """
        Read and parse a CLAUDE.md layer file.

        Args:
            path: Path to layer JSON file

        Returns:
            ClaudeMdLayer object or None if not a layer

        Raises:
            CorruptionError: If checksum verification fails
            json.JSONDecodeError: If file is not valid JSON
            KeyError: If required fields are missing
        """
        # Read from disk
        with open(path, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid claudemd_layer file {path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "claudemd_layer":
            return None

        return ClaudeMdLayer.from_dict(data)


class TransactionContext:
    """
    Context manager for transactional operations.

    Commits on successful exit, rolls back on exception.
    Invalidates cache for written entities after successful commit.
    """

    def __init__(
        self,
        tx_manager: CDGTransactionManager,
        read_only: bool = False,
        got_manager: Optional['GoTManager'] = None
    ):
        """
        Initialize context.

        Args:
            tx_manager: CDG transaction manager
            read_only: If True, rollback instead of commit on exit
            got_manager: Optional GoTManager for cache invalidation
        """
        self.tx_manager = tx_manager
        self.read_only = read_only
        self.tx: Optional[Transaction] = None
        self._got_manager = got_manager

    def __enter__(self) -> TransactionContext:
        """Begin transaction."""
        self.tx = self.tx_manager.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Commit or rollback based on exception.

        Returns:
            False to propagate exceptions (never swallow them)

        Note:
            Index updates are handled automatically by CDGStore via CDGIndexManager.
            Cache invalidation is also handled at the storage layer.
        """
        if self.tx is None:
            return False

        if exc_type is not None:
            # Exception occurred - rollback
            self.tx_manager.rollback(self.tx, reason="exception")
            return False  # Propagate exception

        if self.read_only:
            # Read-only mode - rollback
            self.tx_manager.rollback(self.tx, reason="read_only")
        else:
            # Normal exit - commit
            result = self.tx_manager.commit(self.tx)
            if not result.success:
                raise TransactionError(
                    f"Transaction commit failed: {result.reason}",
                    conflicts=result.conflicts
                )

        return False  # Propagate exceptions

    def create_task(self, title: str, **kwargs) -> Task:
        """
        Create task within transaction.

        Args:
            title: Task title
            **kwargs: Additional task fields (priority, status, description, etc.)

        Returns:
            Created Task object
        """
        task_id = generate_task_id()
        task = Task(
            id=task_id,
            title=title,
            priority=kwargs.get("priority", "medium"),
            status=kwargs.get("status", "pending"),
            description=kwargs.get("description", ""),
            properties=kwargs.get("properties", {}),
        )
        self.tx_manager.write(self.tx, task)
        return task

    def update_task(self, task_id: str, **updates) -> Task:
        """
        Update task within transaction.

        Args:
            task_id: Task identifier
            **updates: Fields to update

        Returns:
            Updated Task object

        Raises:
            TransactionError: If task not found
        """
        task = self.get_task(task_id)
        if task is None:
            raise TransactionError(f"Task not found: {task_id}")

        # Check if task is being marked as completed
        completing_task = updates.get("status") == "completed" and task.status != "completed"

        # Apply updates
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        # Note: Version is bumped automatically by storage layer during commit

        # Write back
        self.tx_manager.write(self.tx, task)

        # Auto-close handoffs when task completes
        if completing_task:
            self._auto_close_handoffs(task_id)

        return task

    def _auto_close_handoffs(self, task_id: str) -> None:
        """
        Auto-close handoffs when a task completes.

        Finds all handoffs referencing the task and marks them as completed
        with a system note indicating auto-closure.

        Args:
            task_id: The task that was completed
        """
        if self._got_manager is None:
            return
        entities_dir = self._got_manager.entities_dir
        if not entities_dir.exists():
            return

        # Find all handoffs for this task
        for handoff_file in entities_dir.glob("H-*.json"):
            try:
                with open(handoff_file, 'r') as f:
                    wrapper = json.load(f)

                data = wrapper.get("data", {})
                if data.get("entity_type") != "handoff":
                    continue

                # Check if this handoff references the completed task
                if data.get("task_id") != task_id:
                    continue

                # Check if handoff is already completed or rejected
                status = data.get("status", "")
                if status in ("completed", "rejected"):
                    continue

                # Load handoff and auto-complete it
                handoff = Handoff.from_dict(data)
                handoff.status = "completed"
                handoff.completed_at = datetime.now(timezone.utc).isoformat()
                handoff.result = {
                    "auto_closed": True,
                    "reason": f"Task {task_id} was marked as completed",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                handoff.artifacts = []
                handoff.bump_version()

                # Write back
                self.tx_manager.write(self.tx, handoff)

                logger.info(f"Auto-closed handoff {handoff.id} for completed task {task_id}")

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupted handoff file {handoff_file}: {e}")
                continue

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task within transaction (sees own writes and deletes).

        Args:
            task_id: Task identifier

        Returns:
            Task object or None if not found or marked for deletion
        """
        # Check if marked for deletion
        if self.tx.has_delete(task_id):
            return None
        entity = self.tx_manager.read(self.tx, task_id)
        if entity is None:
            return None
        if not isinstance(entity, Task):
            return None
        return entity

    def delete_task(self, task_id: str, force: bool = False) -> None:
        """
        Delete task and all connected edges within transaction.

        Args:
            task_id: Task identifier to delete
            force: If False, raise error if task has dependents

        Raises:
            TransactionError: If task has dependents (and force=False) or task not found
        """
        task = self.get_task(task_id)
        if task is None:
            raise TransactionError(f"Task not found: {task_id}")

        # Check for dependents unless force is True
        if not force:
            dependents = self._get_dependents(task_id)
            if dependents:
                dependent_ids = [dep.id for dep in dependents]
                raise TransactionError(
                    f"Cannot delete task {task_id}: has dependents {dependent_ids}. "
                    "Use force=True to override."
                )

        # Get all edges connected to this task
        outgoing, incoming = self._get_edges_for_task(task_id)
        all_edges = outgoing + incoming

        # Mark task for deletion
        self.tx_manager.delete(self.tx, task_id)

        # Mark all connected edges for deletion
        for edge in all_edges:
            self.tx_manager.delete(self.tx, edge.id)

    def delete_decision(self, decision_id: str, force: bool = False) -> None:
        """
        Delete decision and all connected edges within transaction.

        Args:
            decision_id: Decision identifier to delete
            force: If False, raise error if decision has connected edges

        Raises:
            TransactionError: If decision has edges (and force=False) or not found
        """
        decision = self.get_decision(decision_id)
        if decision is None:
            raise TransactionError(f"Decision not found: {decision_id}")

        # Get all edges connected to this decision
        connected_edges = self._get_edges_for_entity(decision_id)

        if not force and connected_edges:
            raise TransactionError(
                f"Cannot delete decision {decision_id}: has {len(connected_edges)} "
                "connected edges. Use force=True to override."
            )

        # Mark decision for deletion
        self.tx_manager.delete(self.tx, decision_id)

        # Mark all connected edges for deletion
        for edge in connected_edges:
            self.tx_manager.delete(self.tx, edge.id)

    def _get_dependents(self, task_id: str) -> list:
        """Get tasks that depend on this task."""
        dependents = []
        # Read all edges to find DEPENDS_ON edges targeting this task
        for entity_id, entity in self.tx.write_set.items():
            if isinstance(entity, Edge):
                if entity.edge_type == EdgeTypes.DEPENDS_ON and entity.target_id == task_id:
                    source = self.get_task(entity.source_id)
                    if source:
                        dependents.append(source)
        # Also check store for edges not in write_set
        if self._got_manager:
            for edge in self._got_manager.list_edges():
                if edge.id not in self.tx.write_set and not self.tx.has_delete(edge.id):
                    if edge.edge_type == EdgeTypes.DEPENDS_ON and edge.target_id == task_id:
                        source = self.get_task(edge.source_id)
                        if source and source.id not in [d.id for d in dependents]:
                            dependents.append(source)
        return dependents

    def _get_edges_for_task(self, task_id: str) -> tuple:
        """Get outgoing and incoming edges for a task."""
        outgoing = []
        incoming = []
        # Check write_set first
        for entity_id, entity in self.tx.write_set.items():
            if isinstance(entity, Edge) and not self.tx.has_delete(entity.id):
                if entity.source_id == task_id:
                    outgoing.append(entity)
                elif entity.target_id == task_id:
                    incoming.append(entity)
        # Also check store
        if self._got_manager:
            for edge in self._got_manager.list_edges():
                if edge.id not in self.tx.write_set and not self.tx.has_delete(edge.id):
                    if edge.source_id == task_id:
                        outgoing.append(edge)
                    elif edge.target_id == task_id:
                        incoming.append(edge)
        return outgoing, incoming

    def _get_edges_for_entity(self, entity_id: str) -> list:
        """Get all edges connected to an entity."""
        edges = []
        # Check write_set first
        for eid, entity in self.tx.write_set.items():
            if isinstance(entity, Edge) and not self.tx.has_delete(entity.id):
                if entity.source_id == entity_id or entity.target_id == entity_id:
                    edges.append(entity)
        # Also check store
        if self._got_manager:
            for edge in self._got_manager.list_edges():
                if edge.id not in self.tx.write_set and not self.tx.has_delete(edge.id):
                    if edge.source_id == entity_id or edge.target_id == entity_id:
                        edges.append(edge)
        return edges

    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """
        Get decision within transaction (sees own writes).

        Args:
            decision_id: Decision identifier

        Returns:
            Decision object or None if not found
        """
        # Check if marked for deletion
        if self.tx.has_delete(decision_id):
            return None
        entity = self.tx_manager.read(self.tx, decision_id)
        if entity is None:
            return None
        if not isinstance(entity, Decision):
            return None
        return entity

    def create_decision(self, title: str, rationale: str, **kwargs) -> Decision:
        """
        Create decision within transaction.

        Args:
            title: Decision title
            rationale: Decision rationale
            **kwargs: Additional decision fields (affects, properties, etc.)

        Returns:
            Created Decision object
        """
        decision_id = generate_decision_id()
        decision = Decision(
            id=decision_id,
            title=title,
            rationale=rationale,
            affects=kwargs.get("affects", []),
            properties=kwargs.get("properties", {}),
        )
        self.tx_manager.write(self.tx, decision)
        return decision

    def log_decision(self, title: str, rationale: str, **kwargs) -> Decision:
        """
        Alias for create_decision() - matches CLI 'decision log' command.

        Args:
            title: Decision title
            rationale: Decision rationale
            **kwargs: Additional decision fields

        Returns:
            Created Decision object
        """
        return self.create_decision(title, rationale, **kwargs)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        validate_relationship: bool = True,
        **kwargs
    ) -> Edge:
        """
        Add edge within transaction.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            edge_type: Edge type
            validate_relationship: If True, validate entity types can be connected
                                  by this edge type (default: True)
            **kwargs: Additional edge fields (weight, confidence, etc.)

        Returns:
            Created Edge object

        Raises:
            ValueError: If entity IDs have invalid format (including legacy sprint IDs)
            ValueError: If validate_relationship=True and relationship is not allowed
            ValueError: If self-reference detected (source_id == target_id)

        Note:
            Entity validation includes:
            - ID format validation for all entity types
            - Legacy sprint ID rejection (S-NNN, S-sprint-NNN-*)
            - Relationship rules (which entity types can connect)
            - Self-reference prevention
        """
        # Full entity and relationship validation
        if validate_relationship:
            validate_edge_relationship(source_id, target_id, edge_type)
        else:
            # Still validate sprint IDs use current format
            _require_current_sprint_id_format(source_id)
            _require_current_sprint_id_format(target_id)

        edge = Edge(
            id="",  # Auto-generated in __post_init__
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=kwargs.get("weight", 1.0),
            confidence=kwargs.get("confidence", 1.0),
            reason=kwargs.get("reason", ""),
        )
        self.tx_manager.write(self.tx, edge)
        return edge

    def create_sprint(self, title: str, **kwargs) -> Sprint:
        """
        Create sprint within transaction.

        Args:
            title: Sprint title
            **kwargs: Additional sprint fields (number, epic_id, status, etc.)

        Returns:
            Created Sprint object
        """
        # Use merge-free timestamp-based ID; number is stored as metadata only
        sprint_id = generate_sprint_id()
        sprint = Sprint(
            id=sprint_id,
            title=title,
            number=kwargs.get("number", 0),
            status=kwargs.get("status", "available"),
            epic_id=kwargs.get("epic_id", ""),
            session_id=kwargs.get("session_id", ""),
            isolation=kwargs.get("isolation", []),
            goals=kwargs.get("goals", []),
            notes=kwargs.get("notes", []),
            properties=kwargs.get("properties", {}),
            metadata=kwargs.get("metadata", {}),
        )
        self.tx_manager.write(self.tx, sprint)
        return sprint

    def update_sprint(self, sprint_id: str, **updates) -> Sprint:
        """
        Update sprint within transaction.

        Args:
            sprint_id: Sprint identifier
            **updates: Fields to update

        Returns:
            Updated Sprint object

        Raises:
            TransactionError: If sprint not found
        """
        # Validate sprint ID format (non-breaking)
        _validate_sprint_id_format(sprint_id)

        sprint = self.get_sprint(sprint_id)
        if sprint is None:
            raise TransactionError(f"Sprint not found: {sprint_id}")

        # Apply updates
        for key, value in updates.items():
            if hasattr(sprint, key):
                setattr(sprint, key, value)

        # Note: Version is bumped automatically by storage layer during commit

        # Write back
        self.tx_manager.write(self.tx, sprint)
        return sprint

    def get_sprint(self, sprint_id: str) -> Optional[Sprint]:
        """
        Get sprint within transaction (sees own writes).

        Args:
            sprint_id: Sprint identifier

        Returns:
            Sprint object or None if not found
        """
        # Validate sprint ID format (non-breaking)
        _validate_sprint_id_format(sprint_id)

        entity = self.tx_manager.read(self.tx, sprint_id)
        if entity is None:
            return None
        if not isinstance(entity, Sprint):
            return None
        return entity

    def get_sprint_tasks(self, sprint_id: str) -> List[Task]:
        """
        Get all tasks in a sprint by finding CONTAINS edges.

        Args:
            sprint_id: Sprint identifier

        Returns:
            List of Task objects in the sprint
        """
        tasks = []
        # Find edges where sprint contains tasks
        edges = self._get_edges_for_entity(sprint_id)
        for edge in edges:
            if edge.edge_type == EdgeTypes.CONTAINS and edge.source_id == sprint_id:
                # Sprint contains this task
                task = self.get_task(edge.target_id)
                if task is not None:
                    tasks.append(task)
        return tasks

    def delete_sprint(self, sprint_id: str, force: bool = False) -> None:
        """
        Delete sprint and all connected edges within transaction.

        Args:
            sprint_id: Sprint identifier to delete
            force: If False, raise error if sprint has tasks

        Raises:
            TransactionError: If sprint has tasks (and force=False) or sprint not found
        """
        sprint = self.get_sprint(sprint_id)
        if sprint is None:
            raise TransactionError(f"Sprint not found: {sprint_id}")

        # Check for contained tasks unless force is True
        if not force:
            tasks = self.get_sprint_tasks(sprint_id)
            if tasks:
                task_ids = [task.id for task in tasks]
                raise TransactionError(
                    f"Cannot delete sprint {sprint_id}: has tasks {task_ids}. "
                    "Use force=True to override."
                )

        # Get all edges connected to this sprint
        connected_edges = self._get_edges_for_entity(sprint_id)

        # Mark sprint for deletion
        self.tx_manager.delete(self.tx, sprint_id)

        # Mark all connected edges for deletion
        for edge in connected_edges:
            self.tx_manager.delete(self.tx, edge.id)

    def create_epic(self, title: str, **kwargs) -> Epic:
        """
        Create epic within transaction.

        Args:
            title: Epic title
            **kwargs: Additional epic fields (epic_id, status, phase, etc.)

        Returns:
            Created Epic object
        """
        epic_id = kwargs.get("epic_id") or generate_epic_id()
        epic = Epic(
            id=epic_id,
            title=title,
            status=kwargs.get("status", "active"),
            phase=kwargs.get("phase", 1),
            phases=kwargs.get("phases", []),
            properties=kwargs.get("properties", {}),
            metadata=kwargs.get("metadata", {}),
        )
        self.tx_manager.write(self.tx, epic)
        return epic

    def update_epic(self, epic_id: str, **updates) -> Epic:
        """
        Update epic within transaction.

        Args:
            epic_id: Epic identifier
            **updates: Fields to update

        Returns:
            Updated Epic object

        Raises:
            TransactionError: If epic not found
        """
        epic = self.get_epic(epic_id)
        if epic is None:
            raise TransactionError(f"Epic not found: {epic_id}")

        # Apply updates
        for key, value in updates.items():
            if hasattr(epic, key):
                setattr(epic, key, value)

        # Note: Version is bumped automatically by storage layer during commit

        # Write back
        self.tx_manager.write(self.tx, epic)
        return epic

    def get_epic(self, epic_id: str) -> Optional[Epic]:
        """
        Get epic within transaction (sees own writes).

        Args:
            epic_id: Epic identifier

        Returns:
            Epic object or None if not found
        """
        entity = self.tx_manager.read(self.tx, epic_id)
        if entity is None:
            return None
        if not isinstance(entity, Epic):
            return None
        return entity

    # Document operations
    def create_document(self, path: str, **kwargs) -> Document:
        """
        Create document within transaction.

        Args:
            path: File path (e.g., "docs/architecture.md")
            **kwargs: Additional document fields (title, doc_type, tags, etc.)

        Returns:
            Created Document object
        """
        doc_id = generate_document_id(path)
        doc = Document(
            id=doc_id,
            path=path,
            title=kwargs.get("title", ""),
            doc_type=kwargs.get("doc_type", "general"),
            tags=kwargs.get("tags", []),
            category=kwargs.get("category", ""),
            properties=kwargs.get("properties", {}),
            metadata=kwargs.get("metadata", {}),
        )
        self.tx_manager.write(self.tx, doc)
        return doc

    def update_document(self, doc_id: str, **updates) -> Document:
        """
        Update document within transaction.

        Args:
            doc_id: Document identifier
            **updates: Fields to update

        Returns:
            Updated Document object

        Raises:
            TransactionError: If document not found
        """
        doc = self.get_document(doc_id)
        if doc is None:
            raise TransactionError(f"Document not found: {doc_id}")

        # Apply updates
        for key, value in updates.items():
            if hasattr(doc, key):
                setattr(doc, key, value)

        # Note: Version is bumped automatically by storage layer during commit

        # Write back
        self.tx_manager.write(self.tx, doc)
        return doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get document within transaction (sees own writes).

        Args:
            doc_id: Document identifier

        Returns:
            Document object or None if not found
        """
        entity = self.tx_manager.read(self.tx, doc_id)
        if entity is None:
            return None
        if not isinstance(entity, Document):
            return None
        return entity

    # Handoff operations
    def initiate_handoff(
        self,
        source_agent: str,
        target_agent: str,
        task_id: str = "",
        instructions: str = "",
        context: Optional[Dict[str, Any]] = None,
        handoff_id: Optional[str] = None,
    ) -> Handoff:
        """
        Initiate a handoff within transaction.

        Args:
            source_agent: Agent initiating the handoff
            target_agent: Agent receiving the handoff
            task_id: Task being handed off (optional for session handoffs)
            instructions: Instructions for the target agent
            context: Additional context data
            handoff_id: Optional custom handoff ID (auto-generated if not provided)

        Returns:
            Created Handoff object

        Raises:
            ValueError: If task_id is provided but does not exist
        """
        # Validate task exists (only if task_id provided)
        if task_id:
            task = self.get_task(task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")

        if handoff_id is None:
            handoff_id = generate_handoff_id()

        handoff = Handoff(
            id=handoff_id,
            source_agent=source_agent,
            target_agent=target_agent,
            task_id=task_id,
            status="initiated",
            instructions=instructions,
            context=context or {},
        )
        self.tx_manager.write(self.tx, handoff)
        return handoff

    def accept_handoff(
        self,
        handoff_id: str,
        agent: str,
        acknowledgment: str = ""
    ) -> Handoff:
        """
        Accept a handoff within transaction.

        Args:
            handoff_id: Handoff identifier
            agent: Agent accepting the handoff
            acknowledgment: Optional acknowledgment message

        Returns:
            Updated Handoff object

        Raises:
            TransactionError: If handoff not found
        """
        handoff = self.get_handoff(handoff_id)
        if handoff is None:
            raise TransactionError(f"Handoff not found: {handoff_id}")

        handoff.status = "accepted"
        handoff.accepted_at = datetime.now(timezone.utc).isoformat()
        if acknowledgment:
            handoff.properties["acknowledgment"] = acknowledgment
        # Note: Version is bumped automatically by storage layer during commit

        self.tx_manager.write(self.tx, handoff)
        return handoff

    def complete_handoff(
        self,
        handoff_id: str,
        agent: str,
        result: Dict[str, Any],
        artifacts: List[str],
    ) -> Handoff:
        """
        Complete a handoff within transaction.

        Args:
            handoff_id: Handoff identifier
            agent: Agent completing the handoff
            result: Result data
            artifacts: List of artifact paths/identifiers

        Returns:
            Updated Handoff object

        Raises:
            TransactionError: If handoff not found
        """
        handoff = self.get_handoff(handoff_id)
        if handoff is None:
            raise TransactionError(f"Handoff not found: {handoff_id}")

        handoff.status = "completed"
        handoff.completed_at = datetime.now(timezone.utc).isoformat()
        handoff.result = result
        handoff.artifacts = artifacts
        # Note: Version is bumped automatically by storage layer during commit

        self.tx_manager.write(self.tx, handoff)
        return handoff

    def reject_handoff(
        self,
        handoff_id: str,
        agent: str,
        reason: str = ""
    ) -> Handoff:
        """
        Reject a handoff within transaction.

        Args:
            handoff_id: Handoff identifier
            agent: Agent rejecting the handoff
            reason: Rejection reason

        Returns:
            Updated Handoff object

        Raises:
            TransactionError: If handoff not found
        """
        handoff = self.get_handoff(handoff_id)
        if handoff is None:
            raise TransactionError(f"Handoff not found: {handoff_id}")

        handoff.status = "rejected"
        handoff.rejected_at = datetime.now(timezone.utc).isoformat()
        handoff.reject_reason = reason
        # Note: Version is bumped automatically by storage layer during commit

        self.tx_manager.write(self.tx, handoff)
        return handoff

    def get_handoff(self, handoff_id: str) -> Optional[Handoff]:
        """
        Get handoff within transaction (sees own writes).

        Args:
            handoff_id: Handoff identifier

        Returns:
            Handoff object or None if not found
        """
        entity = self.tx_manager.read(self.tx, handoff_id)
        if entity is None:
            return None
        if not isinstance(entity, Handoff):
            return None
        return entity

    # ==================== KnowledgeTransfer Methods ====================

    def create_knowledge_transfer(
        self,
        title: str,
        summary: str = "",
        status: str = "draft",
        **kwargs
    ) -> KnowledgeTransfer:
        """
        Create knowledge transfer within transaction.

        Args:
            title: KT title (required)
            summary: Executive summary
            status: Initial status (default: draft)
            **kwargs: Additional fields (session_id, sections, tags, code_refs, etc.)

        Returns:
            Created KnowledgeTransfer object
        """
        from cortical.utils.id_generation import generate_kt_id

        kt_id = generate_kt_id()
        kt = KnowledgeTransfer(
            id=kt_id,
            title=title,
            summary=summary,
            status=status,
            session_id=kwargs.get('session_id', ''),
            session_date=kwargs.get('session_date', ''),
            sections=kwargs.get('sections', {}),
            tags=kwargs.get('tags', []),
            code_refs=kwargs.get('code_refs', []),
            source_file=kwargs.get('source_file'),
            related_tasks=kwargs.get('related_tasks', []),
            related_decisions=kwargs.get('related_decisions', []),
            related_handoffs=kwargs.get('related_handoffs', []),
        )
        self.tx_manager.write(self.tx, kt)
        return kt

    def get_knowledge_transfer(self, kt_id: str) -> Optional[KnowledgeTransfer]:
        """
        Get knowledge transfer within transaction (sees own writes).

        Args:
            kt_id: Knowledge transfer identifier

        Returns:
            KnowledgeTransfer object or None if not found
        """
        entity = self.tx_manager.read(self.tx, kt_id)
        if entity is None:
            return None
        if not isinstance(entity, KnowledgeTransfer):
            return None
        return entity

    # ==================== ClaudeMdLayer Methods ====================

    def create_claudemd_layer(
        self,
        layer_type: str,
        section_id: str,
        title: str,
        content: str,
        **kwargs
    ) -> ClaudeMdLayer:
        """
        Create CLAUDE.md layer within transaction.

        Args:
            layer_type: Type of layer
            section_id: Section identifier
            title: Human-readable title
            content: Markdown content
            **kwargs: Additional fields

        Returns:
            Created ClaudeMdLayer object
        """
        layer_number = kwargs.get("layer_number", 0)
        layer_id = generate_claudemd_layer_id(layer_number, section_id)

        layer = ClaudeMdLayer(
            id=layer_id,
            layer_type=layer_type,
            layer_number=layer_number,
            section_id=section_id,
            title=title,
            content=content,
            freshness_status=kwargs.get("freshness_status", "fresh"),
            freshness_decay_days=kwargs.get("freshness_decay_days", 0),
            inclusion_rule=kwargs.get("inclusion_rule", "always"),
            context_modules=kwargs.get("context_modules", []),
            context_branches=kwargs.get("context_branches", []),
            properties=kwargs.get("properties", {}),
            metadata=kwargs.get("metadata", {}),
        )

        # Compute content hash
        layer.content_hash = layer.compute_content_hash()
        layer.last_regenerated = datetime.now(timezone.utc).isoformat()

        self.tx_manager.write(self.tx, layer)
        return layer

    def get_claudemd_layer(self, layer_id: str) -> Optional[ClaudeMdLayer]:
        """
        Get CLAUDE.md layer within transaction.

        Args:
            layer_id: Layer identifier

        Returns:
            ClaudeMdLayer object or None if not found
        """
        entity = self.tx_manager.read(self.tx, layer_id)
        if entity is None:
            return None
        if not isinstance(entity, ClaudeMdLayer):
            return None
        return entity

    def update_claudemd_layer(self, layer_id: str, **updates) -> ClaudeMdLayer:
        """
        Update CLAUDE.md layer within transaction.

        Args:
            layer_id: Layer identifier
            **updates: Fields to update

        Returns:
            Updated ClaudeMdLayer object

        Raises:
            TransactionError: If layer not found
        """
        layer = self.get_claudemd_layer(layer_id)
        if layer is None:
            raise TransactionError(f"ClaudeMdLayer not found: {layer_id}")

        # Apply updates
        for key, value in updates.items():
            if hasattr(layer, key):
                setattr(layer, key, value)

        # Recompute content hash if content changed
        if "content" in updates:
            layer.content_hash = layer.compute_content_hash()

        # Note: Version is bumped automatically by storage layer during commit
        self.tx_manager.write(self.tx, layer)
        return layer

    def list_claudemd_layers(
        self,
        layer_type: Optional[str] = None,
        freshness_status: Optional[str] = None,
        inclusion_rule: Optional[str] = None
    ) -> List[ClaudeMdLayer]:
        """
        List CLAUDE.md layers within transaction.

        Args:
            layer_type: Filter by layer type
            freshness_status: Filter by freshness status
            inclusion_rule: Filter by inclusion rule

        Returns:
            List of matching ClaudeMdLayer objects
        """
        layers = []

        for entity in self.tx_manager.store.iter_entities(prefix="CML"):
            if not isinstance(entity, ClaudeMdLayer):
                continue

            # Apply filters
            if layer_type and entity.layer_type != layer_type:
                continue
            if freshness_status and entity.freshness_status != freshness_status:
                continue
            if inclusion_rule and entity.inclusion_rule != inclusion_rule:
                continue

            layers.append(entity)

        return layers

    def delete_claudemd_layer(self, layer_id: str) -> bool:
        """
        Delete CLAUDE.md layer within transaction.

        Args:
            layer_id: Layer identifier

        Returns:
            True if deleted, False if not found
        """
        layer = self.get_claudemd_layer(layer_id)
        if layer is None:
            return False

        # Delete through transaction manager (handles cache invalidation)
        self.tx_manager.delete(self.tx, layer_id)
        return True

    def read(self, entity_id: str) -> Optional[Entity]:
        """
        Read any entity by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Entity object or None if not found
        """
        return self.tx_manager.read(self.tx, entity_id)

    def write(self, entity: Entity) -> None:
        """
        Write any entity.

        Args:
            entity: Entity to write
        """
        self.tx_manager.write(self.tx, entity)
