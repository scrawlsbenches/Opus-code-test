"""
High-level API for Graph of Thought operations.

Provides convenient context managers and methods for working with the GoT
transactional system. This is the primary user-facing interface.

Example:
    >>> manager = GoTManager("/path/to/.got")
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
from .tx_manager import TransactionManager, CommitResult
from .sync import SyncManager, SyncResult
from .recovery import RecoveryManager, RecoveryResult
from .indexer import QueryIndexManager
from .types import Task, Decision, Edge, Entity, Sprint, Epic, Handoff, ClaudeMdLayer, ClaudeMdVersion, Document
from .transaction import Transaction
from .errors import TransactionError, CorruptionError
from .config import DurabilityMode
from .query_api import QueryAPI
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
        manager = GoTManager("/path/to/.got")

        with manager.transaction() as tx:
            task = tx.create_task("Implement feature", priority="high")
            tx.update_task(task.id, status="in_progress")
        # Auto-commits on success, rolls back on exception

    Caching:
        Entity caching is enabled by default for 10-50x faster repeated queries.
        Cache is automatically invalidated on writes.

        # Disable caching for durability-first use cases
        manager = GoTManager("/path/to/.got", cache_enabled=False)

        # Get cache statistics
        stats = manager.cache_stats()
        # {'hits': 150, 'misses': 50, 'hit_rate': 0.75, 'size': 80}

        # Clear cache manually
        manager.cache_clear()
    """

    def __init__(
        self,
        got_dir: Path,
        durability: DurabilityMode = DurabilityMode.BALANCED,
        cache_enabled: bool = True
    ):
        """
        Initialize GoT manager with directory.

        Args:
            got_dir: Base directory for GoT storage
            durability: Durability mode controlling fsync behavior (default: BALANCED)
            cache_enabled: Enable in-memory entity caching for faster reads (default: True)
        """
        self.got_dir = Path(got_dir)
        self.durability = durability
        self.tx_manager = TransactionManager(self.got_dir, durability=durability)
        self._sync_manager = None  # Lazy initialization
        self._recovery_manager = None  # Lazy initialization
        self._index_manager = None  # Lazy initialization
        self._query_api = None  # Lazy initialization

        # Entity cache for read performance (10-50x faster for repeated queries)
        self._cache_enabled = cache_enabled
        self._entity_cache: Dict[str, Entity] = {}
        self._cache_timestamps: Dict[str, float] = {}  # entity_id -> timestamp
        self._cache_access_order: List[str] = []  # LRU tracking (most recent at end)
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_ttl: Optional[float] = None  # TTL in seconds (None = no expiry)
        self._cache_max_size: Optional[int] = None  # Max entries (None = unlimited)
        self._cache_lock = threading.Lock()  # Thread-safety for cache operations

        # Log initialization at debug level
        cache_status = "enabled" if cache_enabled else "disabled"
        logger.debug(
            f"GoTManager initialized with durability={durability.value}, cache={cache_status}"
        )

    @property
    def sync_manager(self) -> SyncManager:
        """Get sync manager (lazy initialization)."""
        if self._sync_manager is None:
            self._sync_manager = SyncManager(self.got_dir)
        return self._sync_manager

    @property
    def recovery_manager(self) -> RecoveryManager:
        """Get recovery manager (lazy initialization)."""
        if self._recovery_manager is None:
            self._recovery_manager = RecoveryManager(self.got_dir)
        return self._recovery_manager

    @property
    def index_manager(self) -> QueryIndexManager:
        """
        Get index manager (lazy initialization with rebuild).

        The index manager is initialized lazily and rebuilds indexes
        from entities on first access to ensure consistency.
        """
        if self._index_manager is None:
            self._index_manager = QueryIndexManager(self.got_dir)
            # Rebuild indexes from entities to ensure consistency
            self._rebuild_indexes()
        return self._index_manager

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

    def _rebuild_indexes(self) -> None:
        """Rebuild all indexes from current entities."""
        if self._index_manager is None:
            return

        # Get all tasks and edges for index rebuild
        tasks = self.list_all_tasks()
        edges = self.list_edges()

        # Rebuild indexes
        self._index_manager.rebuild_all(tasks, edges)
        logger.debug(f"Rebuilt indexes: {len(tasks)} tasks, {len(edges)} edges")

    def _update_index_for_task(
        self,
        task: Task,
        old_status: Optional[str] = None,
        old_priority: Optional[str] = None,
        is_delete: bool = False
    ) -> None:
        """
        Update index when a task changes.

        Args:
            task: The task that changed
            old_status: Previous status (for update operations)
            old_priority: Previous priority (for update operations)
            is_delete: True if task is being deleted
        """
        if self._index_manager is None:
            return

        if is_delete:
            self._index_manager.remove_task(task.id)
        elif old_status is not None or old_priority is not None:
            # Update operation
            self._index_manager.update_task(
                task.id,
                old_status=old_status,
                new_status=task.status,
                old_priority=old_priority,
                new_priority=task.priority
            )
        else:
            # Create operation
            self._index_manager.index_task(
                task.id,
                status=task.status,
                priority=task.priority
            )

        # Save indexes after each update
        self._index_manager.save()

    # ==================== Cache Methods ====================

    def _cache_get(self, entity_id: str) -> Optional[Entity]:
        """
        Get entity from cache.

        Checks TTL if configured and evicts expired entries.
        Updates LRU access order on hit.

        Args:
            entity_id: Entity identifier

        Returns:
            Cached entity or None if not cached or expired
        """
        if not self._cache_enabled:
            return None

        with self._cache_lock:
            entity = self._entity_cache.get(entity_id)
            if entity is None:
                return None

            # Check TTL expiration
            if self._cache_ttl is not None:
                timestamp = self._cache_timestamps.get(entity_id, 0)
                if time.time() - timestamp > self._cache_ttl:
                    # Entry expired, remove it
                    self._cache_invalidate_locked(entity_id)
                    return None

            # Update LRU order (move to end = most recently used)
            if entity_id in self._cache_access_order:
                self._cache_access_order.remove(entity_id)
            self._cache_access_order.append(entity_id)

            self._cache_hits += 1
            return entity

    def _cache_set(self, entity_id: str, entity: Entity) -> None:
        """
        Add entity to cache.

        Enforces max size via LRU eviction if configured.

        Args:
            entity_id: Entity identifier
            entity: Entity to cache
        """
        if not self._cache_enabled:
            return

        with self._cache_lock:
            # Enforce max size via LRU eviction
            if self._cache_max_size is not None:
                while len(self._entity_cache) >= self._cache_max_size:
                    if not self._cache_access_order:
                        break
                    # Evict least recently used (first in list)
                    lru_id = self._cache_access_order.pop(0)
                    self._entity_cache.pop(lru_id, None)
                    self._cache_timestamps.pop(lru_id, None)

            self._entity_cache[entity_id] = entity
            self._cache_timestamps[entity_id] = time.time()

            # Update LRU order
            if entity_id in self._cache_access_order:
                self._cache_access_order.remove(entity_id)
            self._cache_access_order.append(entity_id)

            self._cache_misses += 1

    def _cache_invalidate_locked(self, entity_id: str) -> None:
        """
        Remove entity from cache (caller holds lock).

        Internal method - use _cache_invalidate() for external calls.
        """
        self._entity_cache.pop(entity_id, None)
        self._cache_timestamps.pop(entity_id, None)
        if entity_id in self._cache_access_order:
            self._cache_access_order.remove(entity_id)

    def _cache_invalidate(self, entity_id: str) -> None:
        """
        Remove entity from cache.

        Args:
            entity_id: Entity identifier to invalidate
        """
        if not self._cache_enabled:
            return

        with self._cache_lock:
            self._cache_invalidate_locked(entity_id)

    def cache_clear(self) -> None:
        """Clear all cached entities and reset statistics."""
        with self._cache_lock:
            self._entity_cache.clear()
            self._cache_timestamps.clear()
            self._cache_access_order.clear()
            self._cache_hits = 0
            self._cache_misses = 0

    def cache_configure(
        self,
        ttl: Optional[float] = None,
        max_size: Optional[int] = None
    ) -> None:
        """
        Configure cache TTL and size limits.

        Args:
            ttl: Time-to-live in seconds for cache entries (None = no expiry)
            max_size: Maximum number of cached entries (None = unlimited).
                     When exceeded, least recently used entries are evicted.

        Example:
            >>> manager.cache_configure(ttl=300, max_size=1000)  # 5 min TTL, 1000 max entries
        """
        self._cache_ttl = ttl
        self._cache_max_size = max_size

    def cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, hit_rate, size, ttl, and max_size
        """
        with self._cache_lock:
            total = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total if total > 0 else 0.0

            return {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'hit_rate': hit_rate,
                'size': len(self._entity_cache),
                'enabled': self._cache_enabled,
                'ttl': self._cache_ttl,
                'max_size': self._cache_max_size,
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
            >>> manager = GoTManager(got_dir)
            >>> counts = manager.load_all()
            >>> print(f"Loaded {counts['tasks']} tasks, {counts['edges']} edges")

            # Now all queries will use cached entities
            >>> Query(manager).tasks().execute()  # Sub-millisecond!
        """
        if not self._cache_enabled:
            # Enable caching temporarily to allow loading
            self._cache_enabled = True
            was_disabled = True
        else:
            was_disabled = False

        counts = {
            'tasks': 0,
            'decisions': 0,
            'sprints': 0,
            'epics': 0,
            'edges': 0,
            'handoffs': 0,
        }

        # Load all tasks
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

        # If caching was disabled, restore that state but keep the loaded data
        if was_disabled:
            # Keep cache enabled so the loaded data is useful
            pass

        return counts

    def _cache_invalidate_many(self, entity_ids: List[str]) -> None:
        """
        Remove multiple entities from cache.

        Args:
            entity_ids: List of entity identifiers to invalidate
        """
        if not self._cache_enabled:
            return

        for entity_id in entity_ids:
            self._entity_cache.pop(entity_id, None)

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

        Uses cache if enabled for faster repeated reads.

        Args:
            task_id: Task identifier

        Returns:
            Task object or None if not found
        """
        # Check cache first
        cached = self._cache_get(task_id)
        if cached is not None and isinstance(cached, Task):
            return cached

        # Cache miss - read from storage
        with self.transaction(read_only=True) as tx:
            task = tx.get_task(task_id)

        # Populate cache on miss
        if task is not None:
            self._cache_set(task_id, task)

        return task

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

        # Invalidate cache to ensure fresh reads
        self._cache_invalidate(task_id)

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
        entities_dir = self.got_dir / "entities"
        decision_file = entities_dir / f"{decision_id}.json"
        if not decision_file.exists():
            return None
        try:
            return self._read_decision_file(decision_file)
        except (CorruptionError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to read decision {decision_id}: {e}")
            return None

    def delete_decision(self, decision_id: str, force: bool = False) -> None:
        """
        Delete a decision and all its connected edges.

        Args:
            decision_id: Decision identifier to delete (D-...)
            force: If False, raise error if decision has edges (default: True behavior)

        Raises:
            TransactionError: If decision not found
        """
        # Check if decision exists
        decision = self.get_decision(decision_id)
        if decision is None:
            raise TransactionError(f"Decision not found: {decision_id}")

        # Find all edges connected to this decision
        all_edges = self.list_edges()
        connected_edges = [
            e for e in all_edges
            if e.source_id == decision_id or e.target_id == decision_id
        ]

        # Check for connected edges unless force is True
        if not force and connected_edges:
            edge_ids = [e.id for e in connected_edges]
            raise TransactionError(
                f"Cannot delete decision {decision_id}: has connected edges {edge_ids}. "
                "Use force=True to override."
            )

        # Collect IDs for cache invalidation
        ids_to_invalidate = [decision_id] + [edge.id for edge in connected_edges]

        # Delete decision and all connected edges
        entities_dir = self.got_dir / "entities"
        decision_file = entities_dir / f"{decision_id}.json"

        # Delete the decision entity file
        if decision_file.exists():
            decision_file.unlink()

        # Delete all connected edge files
        for edge in connected_edges:
            edge_file = entities_dir / f"{edge.id}.json"
            if edge_file.exists():
                edge_file.unlink()

        # Invalidate cache for deleted entities
        self._cache_invalidate_many(ids_to_invalidate)

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
        return self.add_edge(task_id, depends_on_id, "DEPENDS_ON")

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
        return self.add_edge(blocker_id, blocked_id, "BLOCKS")

    def delete_task(self, task_id: str, force: bool = False) -> None:
        """
        Delete a task and all its connected edges.

        Args:
            task_id: Task identifier to delete
            force: If False, raise error if task has dependents

        Raises:
            TransactionError: If task has dependents (and force=False) or task not found
        """
        # Check if task exists
        task = self.get_task(task_id)
        if task is None:
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

        # Collect IDs for cache invalidation
        ids_to_invalidate = [task_id] + [edge.id for edge in all_edges]

        # Delete task and all connected edges in a transaction
        entities_dir = self.got_dir / "entities"
        task_file = entities_dir / f"{task_id}.json"

        # Delete the task entity file
        if task_file.exists():
            task_file.unlink()

        # Delete all connected edge files
        for edge in all_edges:
            edge_file = entities_dir / f"{edge.id}.json"
            if edge_file.exists():
                edge_file.unlink()

        # Invalidate cache for deleted entities
        self._cache_invalidate_many(ids_to_invalidate)

        # Remove task from index
        if self._index_manager is not None:
            self._index_manager.remove_task(task_id)
            self._index_manager.save()

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
        # Check if sprint exists
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
        entities_dir = self.got_dir / "entities"
        connected_edges = []
        for edge_file in entities_dir.glob("E-*.json"):
            try:
                edge = self._read_edge_file(edge_file)
                if edge is None:
                    continue
                if edge.source_id == sprint_id or edge.target_id == sprint_id:
                    connected_edges.append(edge)
            except (CorruptionError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupted edge file {edge_file}: {e}")
                continue

        # Collect IDs for cache invalidation
        ids_to_invalidate = [sprint_id] + [edge.id for edge in connected_edges]

        # Delete sprint entity file
        sprint_file = entities_dir / f"{sprint_id}.json"
        if sprint_file.exists():
            sprint_file.unlink()

        # Delete all connected edge files
        for edge in connected_edges:
            edge_file = entities_dir / f"{edge.id}.json"
            if edge_file.exists():
                edge_file.unlink()

        # Invalidate cache for deleted entities
        self._cache_invalidate_many(ids_to_invalidate)

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
        return self.add_edge(sprint_id, task_id, "CONTAINS")

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
        entities_dir = self.got_dir / "entities"
        if not entities_dir.exists():
            return []

        epics = []
        for entity_file in entities_dir.glob("EPIC-*.json"):
            try:
                # Read epic file
                epic = self._read_epic_file(entity_file)
                if epic is None:
                    continue

                # Apply filter
                if status is not None and epic.status != status:
                    continue

                epics.append(epic)
            except (CorruptionError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupted epic file {entity_file}: {e}")
                continue

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
        return self.add_edge(epic_id, sprint_id, "CONTAINS")

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
        entities_dir = self.got_dir / "entities"
        if not entities_dir.exists():
            return []

        documents = []
        for entity_file in entities_dir.glob("DOC-*.json"):
            try:
                doc = self._read_document_file(entity_file)
                if doc is None:
                    continue

                # Apply filters
                if doc_type is not None and doc.doc_type != doc_type:
                    continue
                if tag is not None and tag not in doc.tags:
                    continue
                if is_stale is not None and doc.is_stale != is_stale:
                    continue

                documents.append(doc)
            except (CorruptionError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupted document file {entity_file}: {e}")
                continue

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
        # Check cache first
        entity_id = file_path.stem
        cached = self._cache_get(entity_id)
        if cached is not None and isinstance(cached, Document):
            return cached

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

        document = Document.from_dict(data)

        # Cache the result
        self._cache_set(entity_id, document)

        return document

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
        # Check cache first
        entity_id = path.stem
        cached = self._cache_get(entity_id)
        if cached is not None and isinstance(cached, Task):
            return cached

        # Read from disk
        with open(path, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid task file {path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "task":
            return None

        task = Task.from_dict(data)

        # Cache the result
        self._cache_set(entity_id, task)

        return task

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
        # Check cache first
        entity_id = path.stem
        cached = self._cache_get(entity_id)
        if cached is not None and isinstance(cached, Edge):
            return cached

        # Read from disk
        with open(path, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid edge file {path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "edge":
            return None

        edge = Edge.from_dict(data)

        # Cache the result
        self._cache_set(entity_id, edge)

        return edge

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
        # Check cache first
        entity_id = path.stem
        cached = self._cache_get(entity_id)
        if cached is not None and isinstance(cached, Decision):
            return cached

        # Read from disk
        with open(path, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid decision file {path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "decision":
            return None

        decision = Decision.from_dict(data)

        # Cache the result
        self._cache_set(entity_id, decision)

        return decision

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
        # Check cache first
        entity_id = path.stem
        cached = self._cache_get(entity_id)
        if cached is not None and isinstance(cached, Sprint):
            return cached

        # Read from disk
        with open(path, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)

        # Validate entity file structure
        is_valid, error = validate_entity_file(wrapper)
        if not is_valid:
            logger.warning(f"Invalid sprint file {path}: {error}")
            return None

        data = wrapper.get("data", {})
        if data.get("entity_type") != "sprint":
            return None

        sprint = Sprint.from_dict(data)

        # Cache the result
        self._cache_set(entity_id, sprint)

        return sprint

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
        # Check cache first
        entity_id = path.stem
        cached = self._cache_get(entity_id)
        if cached is not None and isinstance(cached, Epic):
            return cached

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

        epic = Epic.from_dict(data)

        # Cache the result
        self._cache_set(entity_id, epic)

        return epic

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
        # Check cache first
        entity_id = path.stem
        cached = self._cache_get(entity_id)
        if cached is not None and isinstance(cached, Handoff):
            return cached

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

        handoff = Handoff.from_dict(data)

        # Cache the result
        self._cache_set(entity_id, handoff)

        return handoff

    # Handoff management methods
    def initiate_handoff(
        self,
        source_agent: str,
        target_agent: str,
        task_id: str,
        instructions: str = "",
        context: Optional[Dict[str, Any]] = None,
        handoff_id: Optional[str] = None,
    ) -> Handoff:
        """
        Initiate a handoff to another agent.

        Args:
            source_agent: Agent initiating the handoff
            target_agent: Agent receiving the handoff
            task_id: Task being handed off
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
        entities_dir = self.got_dir / "entities"
        handoff_file = entities_dir / f"{handoff_id}.json"
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
        entities_dir = self.got_dir / "entities"
        if not entities_dir.exists():
            return []

        handoffs = []
        for entity_file in entities_dir.glob("H-*.json"):
            try:
                handoff = self._read_handoff_file(entity_file)
                if handoff is None:
                    continue

                # Apply filters
                if status is not None and handoff.status != status:
                    continue
                if target_agent is not None and handoff.target_agent != target_agent:
                    continue
                if source_agent is not None and handoff.source_agent != source_agent:
                    continue

                handoffs.append(handoff)
            except (CorruptionError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupted handoff file {entity_file}: {e}")
                continue

        return handoffs

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
        # Check cache first
        entity_id = path.stem
        cached = self._cache_get(entity_id)
        if cached is not None and isinstance(cached, ClaudeMdLayer):
            return cached

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

        layer = ClaudeMdLayer.from_dict(data)

        # Cache the result
        self._cache_set(entity_id, layer)

        return layer


class TransactionContext:
    """
    Context manager for transactional operations.

    Commits on successful exit, rolls back on exception.
    Invalidates cache for written entities after successful commit.
    """

    def __init__(
        self,
        tx_manager: TransactionManager,
        read_only: bool = False,
        got_manager: Optional['GoTManager'] = None
    ):
        """
        Initialize context.

        Args:
            tx_manager: Transaction manager
            read_only: If True, rollback instead of commit on exit
            got_manager: Optional GoTManager for cache invalidation
        """
        self.tx_manager = tx_manager
        self.read_only = read_only
        self.tx: Optional[Transaction] = None
        self._got_manager = got_manager
        # Track task state changes for index updates
        # Maps task_id -> {'old_status': str, 'old_priority': str, 'is_create': bool}
        self._task_changes: Dict[str, Dict[str, Any]] = {}

    def __enter__(self) -> TransactionContext:
        """Begin transaction."""
        self.tx = self.tx_manager.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Commit or rollback based on exception.

        Returns:
            False to propagate exceptions (never swallow them)
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

            # Invalidate cache for all written entities
            if self._got_manager is not None and self.tx.write_set:
                written_ids = list(self.tx.write_set.keys())
                self._got_manager._cache_invalidate_many(written_ids)

            # Update indexes for task changes after successful commit
            if self._got_manager is not None and self._task_changes:
                self._apply_index_updates()

        return False  # Propagate exceptions

    def _apply_index_updates(self) -> None:
        """Apply all tracked task changes to the index."""
        if self._got_manager is None or self._got_manager._index_manager is None:
            return

        for task_id, changes in self._task_changes.items():
            task = self.tx.write_set.get(task_id)
            if task is None or not isinstance(task, Task):
                continue

            if changes.get('is_create'):
                # New task - add to index
                self._got_manager._index_manager.index_task(
                    task.id,
                    status=task.status,
                    priority=task.priority
                )
            else:
                # Update task - update index with old/new values
                self._got_manager._index_manager.update_task(
                    task.id,
                    old_status=changes.get('old_status'),
                    new_status=task.status,
                    old_priority=changes.get('old_priority'),
                    new_priority=task.priority
                )

        # Save indexes after all updates
        self._got_manager._index_manager.save()

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

        # Track for index update after commit
        self._task_changes[task.id] = {'is_create': True}

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

        # Track old values for index update (only if not already tracked as create)
        if task_id not in self._task_changes:
            self._task_changes[task_id] = {
                'old_status': task.status,
                'old_priority': task.priority,
                'is_create': False
            }

        # Apply updates
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        # Note: Version is bumped automatically by storage layer during commit

        # Write back
        self.tx_manager.write(self.tx, task)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task within transaction (sees own writes).

        Args:
            task_id: Task identifier

        Returns:
            Task object or None if not found
        """
        entity = self.tx_manager.read(self.tx, task_id)
        if entity is None:
            return None
        if not isinstance(entity, Task):
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
        task_id: str,
        instructions: str = "",
        context: Optional[Dict[str, Any]] = None,
        handoff_id: Optional[str] = None,
    ) -> Handoff:
        """
        Initiate a handoff within transaction.

        Args:
            source_agent: Agent initiating the handoff
            target_agent: Agent receiving the handoff
            task_id: Task being handed off
            instructions: Instructions for the target agent
            context: Additional context data
            handoff_id: Optional custom handoff ID (auto-generated if not provided)

        Returns:
            Created Handoff object
        """
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
        entities_dir = self.tx_manager.got_dir / "entities"
        layers = []

        # Glob for layer files (CML prefix)
        for layer_file in entities_dir.glob("CML*.json"):
            try:
                with open(layer_file, 'r') as f:
                    data = json.load(f)

                entity_data = data.get("data", data)
                if entity_data.get("entity_type") != "claudemd_layer":
                    continue

                layer = ClaudeMdLayer.from_dict(entity_data)

                # Apply filters
                if layer_type and layer.layer_type != layer_type:
                    continue
                if freshness_status and layer.freshness_status != freshness_status:
                    continue
                if inclusion_rule and layer.inclusion_rule != inclusion_rule:
                    continue

                layers.append(layer)

            except (json.JSONDecodeError, KeyError, CorruptionError) as e:
                logger.warning(f"Skipping corrupted layer file {layer_file}: {e}")
                continue

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

        # Delete the layer entity file
        entities_dir = self.tx_manager.got_dir / "entities"
        layer_file = entities_dir / f"{layer_id}.json"
        if layer_file.exists():
            layer_file.unlink()
            return True
        return False

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
