"""
High-level API for Graph of Thought operations.

Provides convenient methods for working with the GoT transactional system.
This is the primary user-facing interface.

Example:
    >>> from cortical.core.bootstrap import create_container
    >>> container = create_container(got_dir=Path(".got"))
    >>> manager = container.resolve(GoTManager)
    >>>
    >>> # Single-operation methods (each is its own transaction)
    >>> task = manager.create_task("Implement feature", priority="high")
    >>> manager.update_task(task.id, status="in_progress")
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator

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

    Provides convenient methods for common tasks. Each method is
    automatically wrapped in a transaction.

    Example:
        from cortical.core.bootstrap import create_container
        container = create_container(got_dir=Path(".got"))
        manager = container.resolve(GoTManager)

        task = manager.create_task("Implement feature", priority="high")
        manager.update_task(task.id, status="in_progress")

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

    @contextmanager
    def transaction(self, read_only: bool = False) -> Generator[Transaction, None, None]:
        """
        Start a transaction context.

        Args:
            read_only: If True, rollback instead of commit on exit

        Yields:
            Transaction object for use with tx_manager methods

        Example:
            with manager.transaction() as tx:
                manager.tx_manager.write(tx, entity)
        """
        tx = self.tx_manager.begin()
        try:
            yield tx
            if read_only:
                self.tx_manager.rollback(tx, reason="read_only")
            else:
                result = self.tx_manager.commit(tx)
                if not result.success:
                    raise TransactionError(
                        f"Transaction commit failed: {result.reason}",
                        conflicts=result.conflicts
                    )
        except Exception:
            self.tx_manager.rollback(tx, reason="exception")
            raise

    def create_task(
        self,
        title: str,
        priority: str = "medium",
        status: str = "pending",
        description: str = "",
        category: str = "feature",
        sprint_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        blocks: Optional[List[str]] = None,
        **properties
    ) -> Task:
        """
        Create a task in a single-operation transaction.

        Args:
            title: Task title
            priority: Priority level (low, medium, high, critical)
            status: Task status (pending, in_progress, completed, blocked)
            description: Task description
            category: Task category (feature, bugfix, refactor, docs, test)
            sprint_id: Optional sprint to add task to
            depends_on: Optional list of task IDs this task depends on
            blocks: Optional list of task IDs this task blocks
            **properties: Additional task properties

        Returns:
            Created Task object

        Raises:
            TransactionError: If commit fails
        """
        # Merge category into properties
        if "category" not in properties:
            properties["category"] = category

        # Add session metadata
        if "metadata" not in properties:
            properties["metadata"] = {
                "session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown"),
                "branch": self._get_current_branch(),
            }

        with self.transaction() as tx:
            task_id = generate_task_id()
            task = Task(
                id=task_id,
                title=title,
                priority=priority,
                status=status,
                description=description,
                properties=properties,
            )
            self.tx_manager.write(tx, task)

        # Add dependencies
        if depends_on:
            for dep_id in depends_on:
                try:
                    self.add_dependency(task.id, dep_id)
                except Exception as e:
                    logger.warning(f"Could not add dependency to {dep_id}: {e}")

        # Add blocks
        if blocks:
            for blocked_id in blocks:
                try:
                    self.add_blocks(task.id, blocked_id)
                except Exception as e:
                    logger.warning(f"Could not add blocks to {blocked_id}: {e}")

        # Add to sprint if specified
        if sprint_id:
            try:
                self.add_edge(sprint_id, task.id, "CONTAINS")
            except Exception as e:
                logger.warning(f"Could not add task to sprint {sprint_id}: {e}")

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
            entity = self.tx_manager.read(tx, task_id)
            if entity is None or not isinstance(entity, Task):
                return None
            return entity

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
            entity = self.tx_manager.read(tx, task_id)
            if entity is None or not isinstance(entity, Task):
                raise TransactionError(f"Task not found: {task_id}")
            task = entity

            # Check if task is being completed
            completing_task = updates.get("status") == "completed" and task.status != "completed"

            # Apply updates
            for key, value in updates.items():
                if hasattr(task, key):
                    setattr(task, key, value)

            self.tx_manager.write(tx, task)

            # Auto-close handoffs when task completes
            if completing_task:
                self._auto_close_handoffs_for_task(tx, task_id)

        return task

    def _auto_close_handoffs_for_task(self, tx: Transaction, task_id: str) -> None:
        """
        Auto-close handoffs when a task completes.

        Finds all handoffs referencing the task and marks them as completed.

        Args:
            tx: Transaction object
            task_id: The task that was completed
        """
        entities_dir = self.entities_dir
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
                self.tx_manager.write(tx, handoff)
                logger.info(f"Auto-closed handoff {handoff.id} for completed task {task_id}")

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupted handoff file {handoff_file}: {e}")
                continue

    def complete_task(self, task_id: str, retrospective: str = "") -> bool:
        """
        Complete a task (set status to completed with timestamp).

        Args:
            task_id: Task identifier
            retrospective: Optional retrospective notes

        Returns:
            True if completed successfully, False if task not found
        """
        task = self.get_task(task_id)
        if not task:
            return False

        # Build update dict
        metadata = dict(task.metadata) if task.metadata else {}
        metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        updates = {"status": "completed", "metadata": metadata}

        if retrospective:
            props = dict(task.properties) if task.properties else {}
            props["retrospective"] = retrospective
            updates["properties"] = props

        self.update_task(task_id, **updates)
        return True

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
            decision_id = generate_decision_id()
            decision = Decision(
                id=decision_id,
                title=title,
                rationale=rationale,
                affects=affects or [],
                properties=properties,
            )
            self.tx_manager.write(tx, decision)
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
        # TODO: Add JUSTIFIES edge creation for each affected entity
        # This was in TransactionalGoTAdapter.log_decision but is part of a separate system
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
            entity = self.tx_manager.read(tx, decision_id)
            if entity is None or not isinstance(entity, Decision):
                return None
            return entity

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
        with self.transaction() as tx:
            entity = self.tx_manager.read(tx, decision_id)
            if entity is None or not isinstance(entity, Decision):
                raise TransactionError(f"Decision not found: {decision_id}")

            # Get all edges connected to this decision
            connected_edges = [e for e in self.list_edges()
                               if e.source_id == decision_id or e.target_id == decision_id]

            if not force and connected_edges:
                raise TransactionError(
                    f"Cannot delete decision {decision_id}: has {len(connected_edges)} "
                    "connected edges. Use force=True to override."
                )

            # Delete decision
            self.tx_manager.delete(tx, decision_id)

            # Delete all connected edges
            for edge in connected_edges:
                self.tx_manager.delete(tx, edge.id)

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
            edge = Edge(
                id="",  # Auto-generated in __post_init__
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                confidence=1.0,
                reason=reason,
            )
            self.tx_manager.write(tx, edge)
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
        with self.transaction() as tx:
            entity = self.tx_manager.read(tx, task_id)
            if entity is None or not isinstance(entity, Task):
                raise TransactionError(f"Task not found: {task_id}")

            # Check for dependents unless force is True
            if not force:
                dependents = self.get_dependents(task_id)
                if dependents:
                    dependent_ids = [dep.id for dep in dependents]
                    raise TransactionError(
                        f"Cannot delete task {task_id}: has dependents {dependent_ids}. "
                        "Use force=True to override."
                    )

            # Get all edges connected to this task
            outgoing, incoming = self.get_edges_for_task(task_id)
            all_edges = outgoing + incoming

            # Delete task
            self.tx_manager.delete(tx, task_id)

            # Delete all connected edges
            for edge in all_edges:
                self.tx_manager.delete(tx, edge.id)

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
            sprint_id = generate_sprint_id()
            sprint = Sprint(
                id=sprint_id,
                title=title,
                number=number or 0,
                status=properties.get("status", "available"),
                epic_id=epic_id,
                session_id=properties.get("session_id", ""),
                isolation=properties.get("isolation", []),
                goals=properties.get("goals", []),
                notes=properties.get("notes", []),
                properties=properties,
                metadata=properties.get("metadata", {}),
            )
            self.tx_manager.write(tx, sprint)
        return sprint

    def get_sprint(self, sprint_id: str) -> Optional[Sprint]:
        """
        Get a sprint by ID (read-only).

        Args:
            sprint_id: Sprint identifier

        Returns:
            Sprint object or None if not found
        """
        _validate_sprint_id_format(sprint_id)
        with self.transaction(read_only=True) as tx:
            entity = self.tx_manager.read(tx, sprint_id)
            if entity is None or not isinstance(entity, Sprint):
                return None
            return entity

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
        _validate_sprint_id_format(sprint_id)
        with self.transaction() as tx:
            entity = self.tx_manager.read(tx, sprint_id)
            if entity is None or not isinstance(entity, Sprint):
                raise TransactionError(f"Sprint not found: {sprint_id}")
            sprint = entity

            # Apply updates
            for key, value in updates.items():
                if hasattr(sprint, key):
                    setattr(sprint, key, value)

            self.tx_manager.write(tx, sprint)
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
        with self.transaction() as tx:
            entity = self.tx_manager.read(tx, sprint_id)
            if entity is None or not isinstance(entity, Sprint):
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
            connected_edges = [e for e in self.list_edges()
                               if e.source_id == sprint_id or e.target_id == sprint_id]

            # Delete sprint
            self.tx_manager.delete(tx, sprint_id)

            # Delete all connected edges
            for edge in connected_edges:
                self.tx_manager.delete(tx, edge.id)

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

    def delete_edge(self, edge_id: str) -> bool:
        """
        Delete an edge by ID.

        Args:
            edge_id: Edge identifier to delete

        Returns:
            True if edge was found and deleted, False if not found
        """
        # Check if edge exists first
        edge = None
        for e in self.list_edges():
            if e.id == edge_id:
                edge = e
                break

        if edge is None:
            return False

        if tx is not None:
            # Use provided transaction
            tx.tx_manager.delete(tx.tx, edge_id)
        else:
            # Create our own transaction
            with self.transaction() as new_tx:
                new_tx.tx_manager.delete(new_tx.tx, edge_id)

        return True

    def unlink_task_from_sprint(self, sprint_id: str, task_id: str) -> bool:
        """
        Unlink a task from a sprint by deleting the CONTAINS edge.

        Args:
            sprint_id: Sprint identifier
            task_id: Task identifier

        Returns:
            True if edge was found and deleted, False if not found
        """
        # Find the CONTAINS edge from sprint to task
        for edge in self.list_edges():
            if (edge.source_id == sprint_id and
                edge.target_id == task_id and
                edge.edge_type == EdgeTypes.CONTAINS):
                with self.transaction() as tx:
                    return self.delete_edge(edge.id, tx=tx)
        return False

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
            eid = epic_id or generate_epic_id()
            epic = Epic(
                id=eid,
                title=title,
                status=properties.get("status", "active"),
                phase=properties.get("phase", 1),
                phases=properties.get("phases", []),
                properties=properties,
                metadata=properties.get("metadata", {}),
            )
            self.tx_manager.write(tx, epic)
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
            entity = self.tx_manager.read(tx, epic_id)
            if entity is None or not isinstance(entity, Epic):
                return None
            return entity

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
            entity = self.tx_manager.read(tx, epic_id)
            if entity is None or not isinstance(entity, Epic):
                raise TransactionError(f"Epic not found: {epic_id}")
            epic = entity

            # Apply updates
            for key, value in updates.items():
                if hasattr(epic, key):
                    setattr(epic, key, value)

            self.tx_manager.write(tx, epic)
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
            doc_id = generate_document_id(path)
            doc = Document(
                id=doc_id,
                path=path,
                title=title,
                doc_type=doc_type,
                tags=tags or [],
                category=properties.get("category", ""),
                properties=properties,
                metadata=properties.get("metadata", {}),
            )
            self.tx_manager.write(tx, doc)
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
            entity = self.tx_manager.read(tx, doc_id)
            if entity is None or not isinstance(entity, Document):
                return None
            return entity

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
            entity = self.tx_manager.read(tx, doc_id)
            if entity is None or not isinstance(entity, Document):
                raise TransactionError(f"Document not found: {doc_id}")
            doc = entity

            # Apply updates
            for key, value in updates.items():
                if hasattr(doc, key):
                    setattr(doc, key, value)

            self.tx_manager.write(tx, doc)
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

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        sprint_id: Optional[str] = None,
        blocked_only: bool = False,
    ) -> List[Task]:
        """
        List tasks with optional filters.

        Args:
            status: Optional status filter
            priority: Optional priority filter
            category: Optional category filter
            sprint_id: Optional sprint ID to filter by
            blocked_only: If True, only return blocked tasks

        Returns:
            List of Task objects matching filters
        """
        # Get tasks from sprint if specified
        if sprint_id:
            sprint_task_ids = set()
            all_edges = self.list_edges()
            for edge in all_edges:
                if (edge.source_id == sprint_id and
                    edge.edge_type == "CONTAINS" and
                    edge.target_id.startswith("T-")):
                    sprint_task_ids.add(edge.target_id)

            if not sprint_task_ids:
                return []

            tasks = [t for t in self.find_tasks(status=status, priority=priority)
                     if t.id in sprint_task_ids]
        else:
            tasks = self.find_tasks(status=status, priority=priority)

        # Filter by category
        if category:
            tasks = [t for t in tasks if t.properties.get("category") == category]

        # Filter blocked only
        if blocked_only:
            tasks = [t for t in tasks if t.status == "blocked"]

        return tasks

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
            # Validate task exists (only if task_id provided)
            if task_id:
                task_entity = self.tx_manager.read(tx, task_id)
                if task_entity is None or not isinstance(task_entity, Task):
                    raise ValueError(f"Task not found: {task_id}")

            hid = handoff_id or generate_handoff_id()
            handoff = Handoff(
                id=hid,
                source_agent=source_agent,
                target_agent=target_agent,
                task_id=task_id,
                status="initiated",
                instructions=instructions,
                context=context or {},
            )
            self.tx_manager.write(tx, handoff)
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
            entity = self.tx_manager.read(tx, handoff_id)
            if entity is None or not isinstance(entity, Handoff):
                raise TransactionError(f"Handoff not found: {handoff_id}")
            handoff = entity

            handoff.status = "accepted"
            handoff.accepted_at = datetime.now(timezone.utc).isoformat()
            if acknowledgment:
                handoff.properties["acknowledgment"] = acknowledgment

            self.tx_manager.write(tx, handoff)
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
            entity = self.tx_manager.read(tx, handoff_id)
            if entity is None or not isinstance(entity, Handoff):
                raise TransactionError(f"Handoff not found: {handoff_id}")
            handoff = entity

            handoff.status = "completed"
            handoff.completed_at = datetime.now(timezone.utc).isoformat()
            handoff.result = result or {}
            handoff.artifacts = artifacts or []

            self.tx_manager.write(tx, handoff)
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
            entity = self.tx_manager.read(tx, handoff_id)
            if entity is None or not isinstance(entity, Handoff):
                raise TransactionError(f"Handoff not found: {handoff_id}")
            handoff = entity

            handoff.status = "rejected"
            handoff.rejected_at = datetime.now(timezone.utc).isoformat()
            handoff.reject_reason = reason

            self.tx_manager.write(tx, handoff)
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
            self.tx_manager.write(tx, kt)
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

    def finalize_knowledge_transfer(
        self,
        kt_id: str,
        handoff_to: Optional[str] = None,
        instructions: str = ""
    ) -> bool:
        """
        Finalize a knowledge transfer (change status to published).

        Args:
            kt_id: Knowledge transfer ID to finalize
            handoff_to: Optional target agent for handoff (currently disabled)
            instructions: Instructions for handoff (currently disabled)

        Returns:
            True if finalized successfully, False if not found
        """
        kt = self.update_knowledge_transfer(kt_id, status="published")
        if kt is None:
            return False

        # TODO(adapter-retirement): Handoff creation under investigation
        # The original adapter code created a handoff when handoff_to was specified.
        # This functionality is disabled pending review of the handoff system.
        # Original code:
        # if handoff_to:
        #     related_tasks = getattr(kt, 'related_tasks', []) or []
        #     task_id = related_tasks[0] if related_tasks else kt_id
        #     self.initiate_handoff(
        #         source_agent="cli",
        #         target_agent=handoff_to,
        #         task_id=task_id,
        #         context={"kt_id": kt_id},
        #         instructions=instructions
        #     )

        return True

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
            lid = generate_claudemd_layer_id(layer_number, section_id)
            layer = ClaudeMdLayer(
                id=lid,
                layer_type=layer_type,
                layer_number=layer_number,
                section_id=section_id,
                title=title,
                content=content,
                freshness_status=properties.get("freshness_status", "fresh"),
                freshness_decay_days=freshness_decay_days,
                inclusion_rule=inclusion_rule,
                context_modules=properties.get("context_modules", []),
                context_branches=properties.get("context_branches", []),
                properties=properties,
                metadata=properties.get("metadata", {}),
            )
            # Compute content hash
            layer.content_hash = layer.compute_content_hash()
            layer.last_regenerated = datetime.now(timezone.utc).isoformat()
            self.tx_manager.write(tx, layer)
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
            entity = self.tx_manager.read(tx, layer_id)
            if entity is None or not isinstance(entity, ClaudeMdLayer):
                return None
            return entity

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
            entity = self.tx_manager.read(tx, layer_id)
            if entity is None or not isinstance(entity, ClaudeMdLayer):
                raise TransactionError(f"ClaudeMdLayer not found: {layer_id}")
            layer = entity

            # Apply updates
            for key, value in updates.items():
                if hasattr(layer, key):
                    setattr(layer, key, value)

            # Recompute content hash if content changed
            if "content" in updates:
                layer.content_hash = layer.compute_content_hash()

            self.tx_manager.write(tx, layer)
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
        Delete a CLAUDE.md layer.

        Args:
            layer_id: Layer identifier

        Returns:
            True if deleted, False if not found
        """
        with self.transaction() as tx:
            entity = self.tx_manager.read(tx, layer_id)
            if entity is None or not isinstance(entity, ClaudeMdLayer):
                return False
            self.tx_manager.delete(tx, layer_id)
        return True

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

    # =========================================================================
    # Methods migrated from TransactionalGoTAdapter
    # TODO: Review and consolidate with existing methods
    # =========================================================================

    def start_task(self, task_id: str) -> bool:
        """Start a task (set status to in_progress)."""
        try:
            task = self.get_task(task_id)
            if not task:
                return False
            task.metadata["started_at"] = datetime.now(timezone.utc).isoformat()
            self.update_task(task_id, status="in_progress", metadata=task.metadata)
            return True
        except Exception as e:
            logger.error(f"Failed to start task {task_id}: {e}")
            return False

    def block_task(self, task_id: str, reason: str = "", blocked_by: Optional[str] = None) -> bool:
        """Block a task."""
        try:
            task = self.get_task(task_id)
            if not task:
                return False
            props = dict(task.properties)
            props["blocked_reason"] = reason or "No reason given"
            self.update_task(task_id, status="blocked", properties=props)
            if blocked_by:
                self.add_blocks(blocked_by, task_id)
            return True
        except Exception as e:
            logger.error(f"Failed to block task {task_id}: {e}")
            return False

    def get_task_sprint(self, task_id: str) -> Optional[Dict[str, str]]:
        """Get the sprint containing this task."""
        _, incoming = self.get_edges_for_task(task_id)
        for edge in incoming:
            if edge.edge_type == "CONTAINS" and edge.source_id.startswith("S-"):
                sprint = self.get_sprint(edge.source_id)
                if sprint:
                    return {'id': sprint.id, 'name': sprint.title}
        return None

    def get_task_dependencies(self, task_id: str) -> List[Task]:
        """Get tasks this task depends on."""
        outgoing, _ = self.get_edges_for_task(task_id)
        deps = []
        for edge in outgoing:
            if edge.edge_type == "DEPENDS_ON":
                task = self.get_task(edge.target_id)
                if task:
                    deps.append(task)
        return deps

    def what_blocks(self, task_id: str) -> List[Task]:
        """Get tasks blocking this task."""
        return self.get_blockers(task_id)

    def what_depends_on(self, task_id: str) -> List[Task]:
        """Get tasks that depend on this task."""
        return self.get_dependents(task_id)

    def get_active_tasks(self) -> List[Task]:
        """Get in-progress tasks."""
        return self.find_tasks(status="in_progress")

    def get_blocked_tasks(self) -> List[Tuple[Task, Optional[str]]]:
        """Get blocked tasks with reasons."""
        tasks = self.find_tasks(status="blocked")
        return [(t, t.properties.get("blocked_reason", "No reason given")) for t in tasks]

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get the next recommended task to work on."""
        # Priority order
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        # Get pending tasks
        pending = self.find_tasks(status="pending")
        if not pending:
            return None

        # Sort by priority
        pending.sort(key=lambda t: (priority_order.get(t.priority, 2), t.created_at))
        task = pending[0]

        return {
            "id": task.id,
            "title": task.title,
            "priority": task.priority,
            "category": task.properties.get("category", ""),
        }

    def claim_sprint(self, sprint_id: str, agent: str) -> Sprint:
        """Claim a sprint for an agent."""
        sprint = self.get_sprint(sprint_id)
        if not sprint:
            raise ValueError(f"Sprint not found: {sprint_id}")
        current_owner = sprint.properties.get("claimed_by")
        if current_owner and current_owner != agent:
            raise ValueError(f"Sprint already claimed by {current_owner}")
        props = dict(sprint.properties)
        props["claimed_by"] = agent
        props["claimed_at"] = datetime.now(timezone.utc).isoformat()
        return self.update_sprint(sprint_id, properties=props)

    def release_sprint(self, sprint_id: str, agent: str) -> Sprint:
        """Release a sprint claim."""
        sprint = self.get_sprint(sprint_id)
        if not sprint:
            raise ValueError(f"Sprint not found: {sprint_id}")
        current_owner = sprint.properties.get("claimed_by")
        if current_owner and current_owner != agent:
            raise ValueError(f"Sprint claimed by {current_owner}, not {agent}")
        props = dict(sprint.properties)
        props.pop("claimed_by", None)
        props.pop("claimed_at", None)
        return self.update_sprint(sprint_id, properties=props)

    def add_sprint_goal(self, sprint_id: str, description: str) -> bool:
        """Add a goal to a sprint."""
        sprint = self.get_sprint(sprint_id)
        if not sprint:
            return False
        goals = list(sprint.goals)
        goals.append({"description": description, "completed": False})
        self.update_sprint(sprint_id, goals=goals)
        return True

    def list_sprint_goals(self, sprint_id: str) -> List[Dict]:
        """List sprint goals."""
        sprint = self.get_sprint(sprint_id)
        return sprint.goals if sprint else []

    def complete_sprint_goal(self, sprint_id: str, goal_index: int) -> bool:
        """Complete a sprint goal."""
        sprint = self.get_sprint(sprint_id)
        if not sprint or goal_index >= len(sprint.goals):
            return False
        goals = list(sprint.goals)
        goals[goal_index]["completed"] = True
        self.update_sprint(sprint_id, goals=goals)
        return True

    def link_task_to_sprint(self, sprint_id: str, task_id: str) -> bool:
        """Link a task to a sprint."""
        try:
            self.add_edge(sprint_id, task_id, "CONTAINS")
            return True
        except Exception:
            return False

    def adapter_create_decision(self, content: str, rationale: str = "",
                       task_id: Optional[str] = None,
                       alternatives: Optional[List[str]] = None) -> str:
        """Create a decision with CLI conveniences."""
        affects = [task_id] if task_id else []
        decision = self.create_decision(
            title=content, rationale=rationale, affects=affects,
            properties={"alternatives": alternatives or []}
        )
        return decision.id

    def why(self, task_id: str) -> List[Dict[str, Any]]:
        """Get decisions affecting a task."""
        decisions = self.list_decisions()
        result = []
        for d in decisions:
            if task_id in d.affects:
                result.append({
                    "decision_id": d.id,
                    "decision": d.title,
                    "rationale": d.rationale,
                    "alternatives": d.properties.get("alternatives", []),
                    "created_at": d.created_at,
                })
        return result

    def adapter_list_handoffs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List handoffs as dicts."""
        handoffs = self.list_handoffs(status=status)
        return [
            {
                "id": h.id,
                "source_agent": h.source_agent,
                "target_agent": h.target_agent,
                "task_id": h.task_id,
                "status": h.status,
                "instructions": h.instructions,
                "context": h.context,
                "result": h.result,
                "artifacts": h.artifacts,
                "created_at": h.created_at,
                "accepted_at": getattr(h, 'accepted_at', ''),
                "completed_at": getattr(h, 'completed_at', ''),
            }
            for h in handoffs
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        all_tasks = self.list_all_tasks()
        by_status = {}
        for task in all_tasks:
            by_status[task.status] = by_status.get(task.status, 0) + 1

        edges = self.list_edges()
        sprints = self.list_sprints()
        epics = self.list_epics()

        return {
            "total_tasks": len(all_tasks),
            "tasks_by_status": by_status,
            "total_edges": len(edges),
            "total_sprints": len(sprints),
            "total_epics": len(epics),
        }

    def validate(self) -> List[str]:
        """Validate GoT state."""
        issues = []
        try:
            tasks = self.list_all_tasks()
            if not tasks:
                issues.append("No tasks found")
        except Exception as e:
            issues.append(f"Validation error: {e}")
        return issues

    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """Natural language query - basic support."""
        results = []
        q = query_str.lower()

        if "blocked" in q:
            for task, reason in self.get_blocked_tasks():
                results.append({"id": task.id, "title": task.title, "reason": reason})
        elif "active" in q or "in_progress" in q:
            for task in self.get_active_tasks():
                results.append({"id": task.id, "title": task.title})
        elif "pending" in q:
            for task in self.find_tasks(status="pending"):
                results.append({"id": task.id, "title": task.title})

        return results

    def infer_edges_from_commit(self, commit_message: str, files_changed: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Infer edges from a commit message."""
        edges_created = []

        # Find all task references
        task_refs = re.findall(r'(?:task:)?(T-[\w-]+)', commit_message, re.IGNORECASE)

        # Find specific relationship patterns
        depends_pattern = re.findall(r'depends on (?:task:)?(T-[\w-]+)', commit_message, re.IGNORECASE)
        blocks_pattern = re.findall(r'blocks (?:task:)?(T-[\w-]+)', commit_message, re.IGNORECASE)
        closes_pattern = re.findall(r'(?:closes?|fixes?|resolves?) (?:task:)?(T-[\w-]+)', commit_message, re.IGNORECASE)

        # Get all known task IDs for matching
        all_tasks = {t.id.upper(): t.id for t in self.list_all_tasks()}

        # Track which tasks were referenced
        referenced_tasks = []
        for ref in task_refs:
            ref_upper = ref.upper()
            if ref_upper in all_tasks:
                referenced_tasks.append(all_tasks[ref_upper])
                edges_created.append({
                    "type": "REFERENCES",
                    "task": all_tasks[ref_upper],
                    "commit_message": commit_message[:50],
                })

        # Handle dependencies
        for dep_ref in depends_pattern:
            dep_upper = dep_ref.upper()
            if dep_upper in all_tasks and referenced_tasks:
                first_task = referenced_tasks[0]
                target_task = all_tasks[dep_upper]
                if first_task != target_task:
                    self.add_dependency(first_task, target_task)
                    edges_created.append({
                        "type": "DEPENDS_ON",
                        "from": first_task,
                        "to": target_task,
                    })

        # Handle blocks
        for block_ref in blocks_pattern:
            block_upper = block_ref.upper()
            if block_upper in all_tasks and referenced_tasks:
                first_task = referenced_tasks[0]
                target_task = all_tasks[block_upper]
                if first_task != target_task:
                    self.add_blocks(first_task, target_task)
                    edges_created.append({
                        "type": "BLOCKS",
                        "from": first_task,
                        "to": target_task,
                    })

        # Handle closes/fixes (mark tasks complete)
        for close_ref in closes_pattern:
            close_upper = close_ref.upper()
            if close_upper in all_tasks:
                task_id = all_tasks[close_upper]
                self.complete_task(task_id, retrospective=f"Closed via commit: {commit_message[:50]}")
                edges_created.append({
                    "type": "CLOSES",
                    "task": task_id,
                })

        return edges_created

    def infer_edges_from_recent_commits(self, count: int = 10, project_root: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Infer edges from recent git commits."""
        cwd = str(project_root) if project_root else str(self.got_dir.parent)
        try:
            result = subprocess.run(
                ["git", "log", f"-{count}", "--pretty=format:%H|%s"],
                capture_output=True, text=True, check=True,
                cwd=cwd
            )
        except Exception as e:
            logger.warning(f"Failed to read git log: {e}")
            return []

        all_edges = []
        for line in result.stdout.strip().split("\n"):
            if "|" in line:
                commit_hash, message = line.split("|", 1)
                edges = self.infer_edges_from_commit(message)
                for edge in edges:
                    edge["commit_hash"] = commit_hash[:8]
                all_edges.extend(edges)

        return all_edges

    def append_to_knowledge_transfer(
        self, kt_id: str, section_title: str, content: str
    ) -> Optional[Any]:
        """Append a section to a knowledge transfer and return the updated entity."""
        result = self.append_knowledge_transfer_section(kt_id, section_title, content)
        if result is not None:
            return self.get_knowledge_transfer(kt_id)
        return None

    def link_knowledge_transfer(
        self, kt_id: str, target_id: str, link_type: str = "DOCUMENTS"
    ) -> bool:
        """Link a knowledge transfer to another entity."""
        try:
            self.add_edge(kt_id, target_id, link_type)
            return True
        except Exception:
            return False

    def _get_current_branch(self) -> str:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"

    @property
    def graph(self):
        """Compatibility property - returns self for methods that access graph."""
        return self

    @property
    def nodes(self):
        """Compatibility property for graph.nodes access."""
        return {t.id: t for t in self.list_all_tasks()}

    @property
    def edges_property(self):
        """Compatibility property for graph.edges access."""
        return self.list_edges()
