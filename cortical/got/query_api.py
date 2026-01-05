"""
Query API for Graph of Thought.

Provides read-only query operations for tasks, edges, and other entities.
This module extracts query methods from api.py to improve maintainability.

The QueryAPI class takes a reference to a GoTManager and provides methods for:
- Finding tasks with filters
- Getting blockers and dependents
- Listing entities (tasks, edges, decisions)
- Sprint queries (tasks, progress)
- Document-task relationships

Example:
    >>> from cortical.got import GoTManager
    >>> from cortical.got.query_api import QueryAPI
    >>>
    >>> manager = GoTManager("/path/to/.got")
    >>> query = QueryAPI(manager)
    >>>
    >>> # Find pending high-priority tasks
    >>> tasks = query.find_tasks(status="pending", priority="high")
    >>>
    >>> # Get blockers for a task
    >>> blockers = query.get_blockers("T-20251227-123456-abc")
    >>>
    >>> # Get sprint progress
    >>> progress = query.get_sprint_progress("S-sprint-017")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any

from .types import Task, Decision, Edge, Sprint, Document, EdgeTypes
from .errors import CorruptionError

if TYPE_CHECKING:
    from .api import GoTManager

logger = logging.getLogger(__name__)


class QueryAPI:
    """
    Read-only query operations for the Graph of Thought.

    This class provides all query/read methods extracted from GoTManager
    for better separation of concerns. It uses the manager's internal
    methods for file I/O and caching.

    Attributes:
        manager: Reference to the GoTManager for file I/O operations

    Example:
        >>> query = QueryAPI(manager)
        >>> pending_tasks = query.find_tasks(status="pending")
        >>> blockers = query.get_blockers("T-20251227-123456-abc")
    """

    def __init__(self, manager: GoTManager):
        """
        Initialize QueryAPI with a GoTManager reference.

        Args:
            manager: GoTManager instance for file I/O operations
        """
        self._manager = manager

    @property
    def got_dir(self) -> Path:
        """Get the GoT directory path."""
        return self._manager.got_dir

    def _iter_entities_by_prefix(self, prefix: str):
        """
        Iterate entities by ID prefix using CDGStore.

        This method uses the store's iter_entities() method which works
        with both disk-based and in-memory storage through the FileSystem
        abstraction.

        Args:
            prefix: Entity ID prefix (e.g., "T-", "E-", "D-", "S-")

        Yields:
            Entity objects matching the prefix
        """
        store = getattr(self._manager.tx_manager, 'store', None)
        if store is not None and hasattr(store, 'iter_entities'):
            yield from store.iter_entities(prefix=prefix)
        else:
            # Fallback: scan disk directory (for backwards compatibility)
            entities_dir = self.got_dir / "entities"
            if not entities_dir.exists():
                return

            for entity_file in entities_dir.glob(f"{prefix}*.json"):
                try:
                    if prefix == "T-":
                        entity = self._manager._read_task_file(entity_file)
                    elif prefix == "E-":
                        entity = self._manager._read_edge_file(entity_file)
                    elif prefix == "D-":
                        entity = self._manager._read_decision_file(entity_file)
                    elif prefix == "S-":
                        entity = self._manager._read_sprint_file(entity_file)
                    else:
                        continue
                    if entity is not None:
                        yield entity
                except (CorruptionError, json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping corrupted file {entity_file}: {e}")
                    continue

    def find_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        title_contains: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Task]:
        """
        Find tasks matching criteria.

        Uses the store's iter_entities() method for efficient iteration.

        Args:
            status: Filter by status ('pending', 'in_progress', 'completed', etc.)
            priority: Filter by priority ('low', 'medium', 'high', 'critical')
            title_contains: Filter by substring in title (case-insensitive)
            category: Filter by category (e.g., 'bugfix', 'feature', 'refactor')

        Returns:
            List of matching Task objects
        """
        tasks = []

        for entity in self._iter_entities_by_prefix("T-"):
            if not isinstance(entity, Task):
                continue
            task = entity

            # Apply filters
            if status is not None and task.status != status:
                continue
            if priority is not None and task.priority != priority:
                continue
            if title_contains is not None and title_contains.lower() not in task.title.lower():
                continue
            if category is not None and task.category != category:
                continue

            tasks.append(task)

        return tasks

    def get_blockers(self, task_id: str) -> List[Task]:
        """
        Get all tasks that block the given task (have BLOCKS edge pointing to it).

        Args:
            task_id: The task being blocked

        Returns:
            List of blocking Task objects
        """
        # Find all BLOCKS edges pointing to this task
        blocker_ids = []
        for edge in self._iter_entities_by_prefix("E-"):
            if not isinstance(edge, Edge):
                continue
            if edge.edge_type == EdgeTypes.BLOCKS and edge.target_id == task_id:
                blocker_ids.append(edge.source_id)

        # Load the blocker tasks
        blockers = []
        for blocker_id in blocker_ids:
            task = self._manager.get_task(blocker_id)
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
        # Find all DEPENDS_ON edges pointing to this task
        dependent_ids = []
        for edge in self._iter_entities_by_prefix("E-"):
            if not isinstance(edge, Edge):
                continue
            if edge.edge_type == EdgeTypes.DEPENDS_ON and edge.target_id == task_id:
                dependent_ids.append(edge.source_id)

        # Load the dependent tasks
        dependents = []
        for dependent_id in dependent_ids:
            task = self._manager.get_task(dependent_id)
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

    def list_tasks(self, status: Optional[str] = None) -> List[Task]:
        """
        List tasks with optional status filter.

        This is an alias for find_tasks() with more intuitive naming
        matching list_sprints(), list_decisions().

        Args:
            status: Optional status filter

        Returns:
            List of Task objects
        """
        return self.find_tasks(status=status)

    def list_edges(self) -> List[Edge]:
        """
        List all edges in the store.

        Returns:
            List of all Edge objects
        """
        edges = []
        for edge in self._iter_entities_by_prefix("E-"):
            if isinstance(edge, Edge):
                edges.append(edge)
        return edges

    def list_decisions(self) -> List[Decision]:
        """
        List all decisions in the store.

        Returns:
            List of all Decision objects
        """
        decisions = []
        for decision in self._iter_entities_by_prefix("D-"):
            if isinstance(decision, Decision):
                decisions.append(decision)
        return decisions

    def get_edges_for_task(self, task_id: str) -> Tuple[List[Edge], List[Edge]]:
        """
        Get all edges connected to a task.

        Args:
            task_id: Task to query

        Returns:
            Tuple of (outgoing_edges, incoming_edges)
        """
        outgoing = []
        incoming = []

        for edge in self._iter_entities_by_prefix("E-"):
            if not isinstance(edge, Edge):
                continue
            if edge.source_id == task_id:
                outgoing.append(edge)
            elif edge.target_id == task_id:
                incoming.append(edge)

        return (outgoing, incoming)

    def get_edges_for_entity(self, entity_id: str) -> Tuple[List[Edge], List[Edge]]:
        """
        Get all edges connected to any entity (task, sprint, epic, etc.).

        This is a more general version of get_edges_for_task() that works
        with any entity type.

        Args:
            entity_id: Entity to query

        Returns:
            Tuple of (outgoing_edges, incoming_edges)
        """
        return self.get_edges_for_task(entity_id)

    def get_sprint_tasks(self, sprint_id: str) -> List[Task]:
        """
        Get all tasks in a sprint.

        Args:
            sprint_id: Sprint identifier

        Returns:
            List of Task objects in the sprint
        """
        # Find all CONTAINS edges from sprint to tasks
        task_ids = []
        for edge in self._iter_entities_by_prefix("E-"):
            if not isinstance(edge, Edge):
                continue
            if edge.edge_type == EdgeTypes.CONTAINS and edge.source_id == sprint_id:
                task_ids.append(edge.target_id)

        # Load the tasks
        tasks = []
        for task_id in task_ids:
            task = self._manager.get_task(task_id)
            if task is not None:
                tasks.append(task)

        return tasks

    def get_sprint_progress(self, sprint_id: str) -> Dict[str, Any]:
        """
        Get sprint progress statistics.

        Args:
            sprint_id: Sprint identifier

        Returns:
            Dictionary with progress statistics:
            - total: Total number of tasks
            - completed: Number of completed tasks
            - in_progress: Number of in-progress tasks
            - pending: Number of pending tasks
            - blocked: Number of blocked tasks
            - completion_rate: Fraction of completed tasks (0.0-1.0)
        """
        tasks = self.get_sprint_tasks(sprint_id)

        total = len(tasks)
        if total == 0:
            return {
                "total": 0,
                "completed": 0,
                "in_progress": 0,
                "pending": 0,
                "blocked": 0,
                "completion_rate": 0.0
            }

        status_counts = {
            "completed": sum(1 for t in tasks if t.status == "completed"),
            "in_progress": sum(1 for t in tasks if t.status == "in_progress"),
            "pending": sum(1 for t in tasks if t.status == "pending"),
            "blocked": sum(1 for t in tasks if t.status == "blocked"),
        }

        return {
            "total": total,
            **status_counts,
            "completion_rate": status_counts["completed"] / total if total > 0 else 0.0
        }

    def get_documents_for_task(self, task_id: str) -> List[Document]:
        """
        Get all documents linked to a task.

        Args:
            task_id: Task identifier

        Returns:
            List of Document objects linked to the task
        """
        # Find all edges from task to documents
        doc_ids = []
        for edge in self._iter_entities_by_prefix("E-"):
            if not isinstance(edge, Edge):
                continue
            # Check if edge is from task to a document
            if edge.source_id == task_id and edge.target_id.startswith("DOC-"):
                doc_ids.append(edge.target_id)

        # Load the documents
        documents = []
        for doc_id in doc_ids:
            doc = self._manager.get_document(doc_id)
            if doc is not None:
                documents.append(doc)

        return documents

    def get_tasks_for_document(self, doc_id: str) -> List[Task]:
        """
        Get all tasks linked to a document.

        Args:
            doc_id: Document identifier

        Returns:
            List of Task objects linked to the document
        """
        # Find all edges to this document
        task_ids = []
        for edge in self._iter_entities_by_prefix("E-"):
            if not isinstance(edge, Edge):
                continue
            # Check if edge is to this document from a task
            if edge.target_id == doc_id and edge.source_id.startswith("T-"):
                task_ids.append(edge.source_id)

        # Load the tasks
        tasks = []
        for task_id in task_ids:
            task = self._manager.get_task(task_id)
            if task is not None:
                tasks.append(task)

        return tasks

    def list_sprints(
        self,
        status: Optional[str] = None,
        epic_id: Optional[str] = None
    ) -> List[Sprint]:
        """
        List sprints, optionally filtered by status or epic.

        Args:
            status: Filter by status ('available', 'in_progress', 'completed', etc.)
            epic_id: Filter by epic ID (uses sprint.epic_id attribute)

        Returns:
            List of matching Sprint objects
        """
        sprints = []
        for sprint in self._iter_entities_by_prefix("S-"):
            if not isinstance(sprint, Sprint):
                continue

            # Apply filters
            if status is not None and sprint.status != status:
                continue
            if epic_id is not None and sprint.epic_id != epic_id:
                continue

            sprints.append(sprint)

        return sprints

    def count_tasks_by_status(self) -> Dict[str, int]:
        """
        Count tasks grouped by status.

        Returns:
            Dictionary mapping status to count
        """
        tasks = self.find_tasks()
        counts: Dict[str, int] = {}
        for task in tasks:
            counts[task.status] = counts.get(task.status, 0) + 1
        return counts

    def count_tasks_by_priority(self) -> Dict[str, int]:
        """
        Count tasks grouped by priority.

        Returns:
            Dictionary mapping priority to count
        """
        tasks = self.find_tasks()
        counts: Dict[str, int] = {}
        for task in tasks:
            priority = task.priority or "unset"
            counts[priority] = counts.get(priority, 0) + 1
        return counts

    def get_blocked_tasks(self) -> List[Task]:
        """
        Get all tasks that are blocked.

        A task is considered blocked if:
        1. It has status='blocked' (set via block_task()), OR
        2. There's a BLOCKS edge pointing to it from an incomplete task

        Returns:
            List of blocked Task objects
        """
        blocked_tasks = []
        blocked_ids = set()

        # First: Find tasks with status='blocked'
        for task in self.find_tasks(status="blocked"):
            if task.id not in blocked_ids:
                blocked_tasks.append(task)
                blocked_ids.add(task.id)

        # Second: Find tasks with incoming BLOCKS edges from non-completed tasks
        for edge in self._iter_entities_by_prefix("E-"):
            if not isinstance(edge, Edge):
                continue

            if edge.edge_type == EdgeTypes.BLOCKS:
                # Check if blocker is not completed
                blocker = self._manager.get_task(edge.source_id)
                if blocker is not None and blocker.status != "completed":
                    blocked = self._manager.get_task(edge.target_id)
                    if blocked is not None and blocked.id not in blocked_ids:
                        blocked_tasks.append(blocked)
                        blocked_ids.add(blocked.id)

        return blocked_tasks

    def get_unblocked_pending_tasks(self) -> List[Task]:
        """
        Get pending tasks that are not blocked.

        Returns:
            List of pending Task objects that have no active blockers
        """
        pending = self.find_tasks(status="pending")
        blocked = set(t.id for t in self.get_blocked_tasks())
        return [t for t in pending if t.id not in blocked]
