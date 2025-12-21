"""
Protocol definition for GoT backend implementations.

This module defines the GoTBackend Protocol that all GoT backend implementations
must implement. This ensures API consistency between different backend strategies
(event-sourced, transactional, etc.).
"""

from typing import Protocol, Optional, List, Dict, Any, Tuple
from cortical.reasoning.graph_of_thought import ThoughtNode


class GoTBackend(Protocol):
    """
    Protocol defining the GoT backend interface.

    This protocol establishes the contract that all GoT backend implementations
    must follow. Both GoTProjectManager (event-sourced) and TransactionalGoTAdapter
    (transactional) implement this protocol.

    The protocol ensures:
    - Consistent API across different backend strategies
    - Type safety for backend operations
    - Clear documentation of required methods
    - Easier testing and mocking
    """

    # =========================================================================
    # Task CRUD Operations
    # =========================================================================

    def create_task(
        self,
        title: str,
        priority: str = "medium",
        category: str = "feature",
        description: str = "",
        sprint_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        blocks: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new task.

        Args:
            title: Task title
            priority: Priority level (low, medium, high, critical)
            category: Task category (feature, bugfix, arch, docs, etc.)
            description: Detailed task description
            sprint_id: Optional sprint ID to associate with
            depends_on: List of task IDs this task depends on
            blocks: List of task IDs this task blocks

        Returns:
            str: The created task ID
        """
        ...

    def get_task(self, task_id: str) -> Optional[ThoughtNode]:
        """
        Get a task by ID.

        Args:
            task_id: Task identifier

        Returns:
            ThoughtNode if found, None otherwise
        """
        ...

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        sprint_id: Optional[str] = None,
        blocked_only: bool = False,
    ) -> List[ThoughtNode]:
        """
        List tasks with optional filters.

        Args:
            status: Filter by status (pending, in_progress, completed, blocked)
            priority: Filter by priority level
            category: Filter by category
            sprint_id: Filter by sprint
            blocked_only: Only return blocked tasks

        Returns:
            List of ThoughtNode objects matching the filters
        """
        ...

    def update_task(self, task_id: str, **updates) -> bool:
        """
        Update task fields.

        Args:
            task_id: Task identifier
            **updates: Field updates (status, priority, description, etc.)

        Returns:
            bool: True if update succeeded, False otherwise
        """
        ...

    def delete_task(self, task_id: str, force: bool = False) -> Tuple[bool, str]:
        """
        Delete a task.

        Args:
            task_id: Task identifier
            force: Force deletion even if task has dependents or blocks others

        Returns:
            Tuple of (success: bool, message: str)
        """
        ...

    # =========================================================================
    # Task State Transitions
    # =========================================================================

    def start_task(self, task_id: str) -> bool:
        """
        Start a task (transition to in_progress status).

        Args:
            task_id: Task identifier

        Returns:
            bool: True if transition succeeded
        """
        ...

    def complete_task(self, task_id: str, retrospective: str = "") -> bool:
        """
        Complete a task (transition to completed status).

        Args:
            task_id: Task identifier
            retrospective: Optional retrospective notes

        Returns:
            bool: True if transition succeeded
        """
        ...

    def block_task(self, task_id: str, reason: str = "", blocked_by: Optional[str] = None) -> bool:
        """
        Block a task (transition to blocked status).

        Args:
            task_id: Task identifier
            reason: Reason for blocking
            blocked_by: Optional task ID that is blocking this task

        Returns:
            bool: True if transition succeeded
        """
        ...

    # =========================================================================
    # Relationship Management
    # =========================================================================

    def add_dependency(self, task_id: str, depends_on_id: str) -> bool:
        """
        Add dependency between tasks (task_id DEPENDS_ON depends_on_id).

        Args:
            task_id: Task that depends on another
            depends_on_id: Task that is depended upon

        Returns:
            bool: True if dependency added successfully
        """
        ...

    def add_blocks(self, blocker_id: str, blocked_id: str) -> bool:
        """
        Add blocking relationship (blocker_id BLOCKS blocked_id).

        Args:
            blocker_id: Task that blocks another
            blocked_id: Task that is blocked

        Returns:
            bool: True if relationship added successfully
        """
        ...

    def get_blockers(self, task_id: str) -> List[ThoughtNode]:
        """
        Get tasks blocking this task.

        Args:
            task_id: Task identifier

        Returns:
            List of ThoughtNode objects that block this task
        """
        ...

    def get_dependents(self, task_id: str) -> List[ThoughtNode]:
        """
        Get tasks that depend on this task.

        Args:
            task_id: Task identifier

        Returns:
            List of ThoughtNode objects that depend on this task
        """
        ...

    def get_task_dependencies(self, task_id: str) -> List[ThoughtNode]:
        """
        Get all tasks this task depends on.

        Args:
            task_id: Task identifier

        Returns:
            List of ThoughtNode objects this task depends on
        """
        ...

    # =========================================================================
    # Query and Analytics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get backend statistics.

        Returns:
            Dictionary with stats like task counts, status breakdown, etc.
        """
        ...

    def validate(self) -> List[str]:
        """
        Validate backend integrity.

        Returns:
            List of validation issues (empty if valid)
        """
        ...

    def get_blocked_tasks(self) -> List[Tuple[ThoughtNode, Optional[str]]]:
        """
        Get all blocked tasks with their blocking reasons.

        Returns:
            List of tuples (task, blocking_reason)
        """
        ...

    def get_active_tasks(self) -> List[ThoughtNode]:
        """
        Get all in-progress tasks.

        Returns:
            List of ThoughtNode objects with status='in_progress'
        """
        ...

    def what_blocks(self, task_id: str) -> List[ThoughtNode]:
        """
        Query: What tasks are blocking this task?

        Args:
            task_id: Task identifier

        Returns:
            List of ThoughtNode objects blocking this task
        """
        ...

    def what_depends_on(self, task_id: str) -> List[ThoughtNode]:
        """
        Query: What tasks depend on this task?

        Args:
            task_id: Task identifier

        Returns:
            List of ThoughtNode objects depending on this task
        """
        ...

    def get_all_relationships(self, task_id: str) -> Dict[str, List[ThoughtNode]]:
        """
        Get all relationships for a task.

        Args:
            task_id: Task identifier

        Returns:
            Dictionary mapping relationship types to lists of related tasks
        """
        ...

    # =========================================================================
    # Persistence
    # =========================================================================

    def sync_to_git(self) -> str:
        """
        Sync backend state to git-tracked storage.

        Returns:
            str: Status message or snapshot filename
        """
        ...

    def export_graph(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export graph to JSON format.

        Args:
            output_path: Optional file path to write JSON

        Returns:
            Dictionary representation of the graph
        """
        ...

    # =========================================================================
    # Query Language
    # =========================================================================

    def query(self, query_str: str) -> List[Dict[str, Any]]:
        """
        Execute a query using the simple query language.

        Supported queries:
        - "what blocks TASK_ID"
        - "what depends on TASK_ID"
        - "path from ID1 to ID2"
        - "relationships TASK_ID"
        - "active tasks"
        - "pending tasks"
        - "blocked tasks"

        Args:
            query_str: Query string

        Returns:
            List of query results (format depends on query type)
        """
        ...
