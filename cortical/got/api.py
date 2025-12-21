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
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .tx_manager import TransactionManager, CommitResult
from .sync import SyncManager, SyncResult
from .recovery import RecoveryManager, RecoveryResult
from .types import Task, Decision, Edge, Entity
from .transaction import Transaction
from .errors import TransactionError, CorruptionError

logger = logging.getLogger(__name__)


def generate_task_id() -> str:
    """
    Generate unique task ID.

    Format: T-YYYYMMDD-HHMMSS-XXXX where XXXX is random hex.

    Returns:
        Task ID string
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    random_suffix = secrets.token_hex(2)  # 4 hex chars
    return f"T-{timestamp}-{random_suffix}"


def generate_decision_id() -> str:
    """
    Generate unique decision ID.

    Format: D-YYYYMMDD-HHMMSS-XXXX where XXXX is random hex.

    Returns:
        Decision ID string
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    random_suffix = secrets.token_hex(2)  # 4 hex chars
    return f"D-{timestamp}-{random_suffix}"


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
    """

    def __init__(self, got_dir: Path):
        """
        Initialize GoT manager with directory.

        Args:
            got_dir: Base directory for GoT storage
        """
        self.got_dir = Path(got_dir)
        self.tx_manager = TransactionManager(self.got_dir)
        self._sync_manager = None  # Lazy initialization
        self._recovery_manager = None  # Lazy initialization

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

    def transaction(self, read_only: bool = False) -> TransactionContext:
        """
        Start a transaction context.

        Args:
            read_only: If True, rollback instead of commit on exit

        Returns:
            TransactionContext for use with 'with' statement
        """
        return TransactionContext(self.tx_manager, read_only=read_only)

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

        Args:
            task_id: Task identifier

        Returns:
            Task object or None if not found
        """
        with self.transaction(read_only=True) as tx:
            task = tx.get_task(task_id)
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

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0
    ) -> Edge:
        """
        Add an edge between entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            edge_type: Edge type (DEPENDS_ON, BLOCKS, etc.)
            weight: Edge weight (0.0-1.0)

        Returns:
            Created Edge object

        Raises:
            TransactionError: If commit fails
        """
        with self.transaction() as tx:
            edge = tx.add_edge(source_id, target_id, edge_type, weight=weight)
        return edge

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
        title_contains: Optional[str] = None
    ) -> List[Task]:
        """
        Find tasks matching criteria. Scans disk (no in-memory cache).

        Args:
            status: Filter by status ('pending', 'in_progress', 'completed', etc.)
            priority: Filter by priority ('low', 'medium', 'high', 'critical')
            title_contains: Filter by substring in title (case-insensitive)

        Returns:
            List of matching Task objects
        """
        entities_dir = self.got_dir / "entities"
        if not entities_dir.exists():
            return []

        tasks = []
        for entity_file in entities_dir.glob("T-*.json"):
            try:
                task = self._read_task_file(entity_file)
                if task is None:
                    continue

                # Apply filters
                if status is not None and task.status != status:
                    continue
                if priority is not None and task.priority != priority:
                    continue
                if title_contains is not None and title_contains.lower() not in task.title.lower():
                    continue

                tasks.append(task)
            except (CorruptionError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupted task file {entity_file}: {e}")
                continue

        return tasks

    def get_blockers(self, task_id: str) -> List[Task]:
        """
        Get all tasks that block the given task (have BLOCKS edge pointing to it).

        Args:
            task_id: The task being blocked

        Returns:
            List of blocking Task objects
        """
        entities_dir = self.got_dir / "entities"
        if not entities_dir.exists():
            return []

        # Find all BLOCKS edges pointing to this task
        blocker_ids = []
        for edge_file in entities_dir.glob("E-*.json"):
            try:
                edge = self._read_edge_file(edge_file)
                if edge is None:
                    continue

                if edge.edge_type == "BLOCKS" and edge.target_id == task_id:
                    blocker_ids.append(edge.source_id)
            except (CorruptionError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupted edge file {edge_file}: {e}")
                continue

        # Load the blocker tasks
        blockers = []
        for blocker_id in blocker_ids:
            task = self.get_task(blocker_id)
            if task is not None:
                blockers.append(task)

        return blockers

    def get_dependents(self, task_id: str) -> List[Task]:
        """
        Get all tasks that depend on the given task (have DEPENDS_ON edge pointing to it).

        Args:
            task_id: The task being depended on

        Returns:
            List of dependent Task objects
        """
        entities_dir = self.got_dir / "entities"
        if not entities_dir.exists():
            return []

        # Find all DEPENDS_ON edges pointing to this task
        dependent_ids = []
        for edge_file in entities_dir.glob("E-*.json"):
            try:
                edge = self._read_edge_file(edge_file)
                if edge is None:
                    continue

                if edge.edge_type == "DEPENDS_ON" and edge.target_id == task_id:
                    dependent_ids.append(edge.source_id)
            except (CorruptionError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupted edge file {edge_file}: {e}")
                continue

        # Load the dependent tasks
        dependents = []
        for dependent_id in dependent_ids:
            task = self.get_task(dependent_id)
            if task is not None:
                dependents.append(task)

        return dependents

    def list_all_tasks(self) -> List[Task]:
        """
        List all tasks in the store. Use sparingly - scans entire store.

        Returns:
            List of all Task objects
        """
        return self.find_tasks()

    def get_edges_for_task(self, task_id: str) -> Tuple[List[Edge], List[Edge]]:
        """
        Get all edges connected to a task.

        Args:
            task_id: Task to query

        Returns:
            Tuple of (outgoing_edges, incoming_edges)
        """
        entities_dir = self.got_dir / "entities"
        if not entities_dir.exists():
            return ([], [])

        outgoing = []
        incoming = []

        for edge_file in entities_dir.glob("E-*.json"):
            try:
                edge = self._read_edge_file(edge_file)
                if edge is None:
                    continue

                if edge.source_id == task_id:
                    outgoing.append(edge)
                elif edge.target_id == task_id:
                    incoming.append(edge)
            except (CorruptionError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping corrupted edge file {edge_file}: {e}")
                continue

        return (outgoing, incoming)

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
        with open(path, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)

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
        with open(path, 'r', encoding='utf-8') as f:
            wrapper = json.load(f)

        data = wrapper.get("data", {})
        if data.get("entity_type") != "edge":
            return None

        return Edge.from_dict(data)


class TransactionContext:
    """
    Context manager for transactional operations.

    Commits on successful exit, rolls back on exception.
    """

    def __init__(
        self,
        tx_manager: TransactionManager,
        read_only: bool = False
    ):
        """
        Initialize context.

        Args:
            tx_manager: Transaction manager
            read_only: If True, rollback instead of commit on exit
        """
        self.tx_manager = tx_manager
        self.read_only = read_only
        self.tx: Optional[Transaction] = None

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

        # Apply updates
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        # Bump version
        task.bump_version()

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

    def add_edge(self, source_id: str, target_id: str, edge_type: str, **kwargs) -> Edge:
        """
        Add edge within transaction.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            edge_type: Edge type
            **kwargs: Additional edge fields (weight, confidence, etc.)

        Returns:
            Created Edge object
        """
        edge = Edge(
            id="",  # Auto-generated in __post_init__
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=kwargs.get("weight", 1.0),
            confidence=kwargs.get("confidence", 1.0),
        )
        self.tx_manager.write(self.tx, edge)
        return edge

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
