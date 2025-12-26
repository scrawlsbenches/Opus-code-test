"""
Goal Stack for Woven Mind Cortex.

Implements goal tracking with monotonic progress guarantees.
Goals represent desired states that the system works toward,
tracking progress and preventing regression.

Part of Sprint 3: Cortex Abstraction (Woven Mind + PRISM Marriage)

Key concepts:
- Goal: A desired state with progress tracking
- GoalStack: Manages active goals and their progress
- Monotonic progress: Progress can only increase, never decrease

Example:
    >>> from cortical.reasoning.goal_stack import GoalStack, Goal
    >>> stack = GoalStack()
    >>> goal = stack.push_goal("learn_concept", target_nodes={"neural", "network"})
    >>> stack.update_progress(goal.id, 0.3)
    >>> stack.update_progress(goal.id, 0.2)  # Ignored - would be regression
    >>> stack.get_progress(goal.id)  # Still 0.3
    0.3
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any, FrozenSet
import uuid


class GoalStatus(Enum):
    """Status of a goal in the stack."""
    PENDING = "pending"      # Not yet started
    ACTIVE = "active"        # Currently being pursued
    ACHIEVED = "achieved"    # Successfully completed
    ABANDONED = "abandoned"  # Given up on
    BLOCKED = "blocked"      # Waiting on something


class GoalPriority(Enum):
    """Priority levels for goals."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Goal:
    """
    A goal with progress tracking.

    Goals represent desired states that the system works toward.
    Progress is monotonic - it can only increase, never decrease.

    Attributes:
        id: Unique identifier for this goal.
        name: Human-readable name.
        target_nodes: Nodes that should be activated for success.
        target_state: Optional detailed target state.
        priority: How important this goal is.
        status: Current status (pending, active, achieved, etc.).
        progress: Current progress (0.0 to 1.0).
        created_at: When the goal was created.
        completed_at: When the goal was achieved (if applicable).
        parent_id: Parent goal ID (for sub-goals).
        blocking_goals: Goals that must complete first.
        metadata: Additional goal-specific data.
    """
    id: str
    name: str
    target_nodes: FrozenSet[str] = field(default_factory=frozenset)
    target_state: Optional[Dict[str, Any]] = None
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    parent_id: Optional[str] = None
    blocking_goals: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Progress history for debugging
    _progress_history: List[tuple] = field(default_factory=list)

    def update_progress(self, new_progress: float) -> bool:
        """
        Update progress, enforcing monotonicity.

        Args:
            new_progress: New progress value (0.0-1.0).

        Returns:
            True if progress was updated, False if rejected (would regress).
        """
        # Clamp to valid range
        new_progress = max(0.0, min(1.0, new_progress))

        # Enforce monotonicity - progress can only increase
        if new_progress <= self.progress:
            return False

        # Record the change
        self._progress_history.append((datetime.now(), self.progress, new_progress))
        self.progress = new_progress

        # Check for completion
        if self.progress >= 1.0:
            self.status = GoalStatus.ACHIEVED
            self.completed_at = datetime.now()

        return True

    def is_blocked(self) -> bool:
        """Check if this goal is blocked by other goals."""
        return len(self.blocking_goals) > 0 and self.status != GoalStatus.ACHIEVED

    def is_achievable(self) -> bool:
        """Check if this goal can still be achieved."""
        return self.status not in (GoalStatus.ACHIEVED, GoalStatus.ABANDONED)

    def get_progress_velocity(self) -> float:
        """Calculate rate of progress change over recent updates."""
        if len(self._progress_history) < 2:
            return 0.0

        # Look at last 5 updates
        recent = self._progress_history[-5:]
        if len(recent) < 2:
            return 0.0

        # Calculate average change per update
        total_change = recent[-1][2] - recent[0][1]  # new progress - old progress
        return total_change / len(recent)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize goal to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "target_nodes": list(self.target_nodes),
            "target_state": self.target_state,
            "priority": self.priority.value,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "parent_id": self.parent_id,
            "blocking_goals": list(self.blocking_goals),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        """Deserialize goal from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            target_nodes=frozenset(data.get("target_nodes", [])),
            target_state=data.get("target_state"),
            priority=GoalPriority(data.get("priority", 2)),
            status=GoalStatus(data.get("status", "pending")),
            progress=data.get("progress", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            parent_id=data.get("parent_id"),
            blocking_goals=set(data.get("blocking_goals", [])),
            metadata=data.get("metadata", {}),
        )


class GoalStack:
    """
    Manages a stack of goals with progress tracking.

    The GoalStack maintains an ordered collection of goals,
    tracks their progress, and enforces monotonicity.

    Attributes:
        goals: Dictionary of all goals by ID.
        active_goals: Set of currently active goal IDs.
        goal_order: Stack order (most recent first).
    """

    def __init__(self, max_active_goals: int = 10):
        """Initialize the goal stack.

        Args:
            max_active_goals: Maximum concurrent active goals.
        """
        self.max_active_goals = max_active_goals
        self.goals: Dict[str, Goal] = {}
        self.active_goals: Set[str] = set()
        self.goal_order: List[str] = []

        # Statistics
        self._achieved_count = 0
        self._abandoned_count = 0

    def push_goal(
        self,
        name: str,
        target_nodes: Optional[Set[str]] = None,
        target_state: Optional[Dict[str, Any]] = None,
        priority: GoalPriority = GoalPriority.MEDIUM,
        parent_id: Optional[str] = None,
        blocking_goals: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Goal:
        """
        Push a new goal onto the stack.

        Args:
            name: Human-readable goal name.
            target_nodes: Nodes that should activate for success.
            target_state: Detailed target state.
            priority: Goal priority level.
            parent_id: Parent goal (for sub-goals).
            blocking_goals: Goals that must complete first.
            metadata: Additional goal data.

        Returns:
            The created Goal object.
        """
        goal_id = f"G-{uuid.uuid4().hex[:8]}"

        goal = Goal(
            id=goal_id,
            name=name,
            target_nodes=frozenset(target_nodes) if target_nodes else frozenset(),
            target_state=target_state,
            priority=priority,
            parent_id=parent_id,
            blocking_goals=blocking_goals or set(),
            metadata=metadata or {},
        )

        self.goals[goal_id] = goal
        self.goal_order.insert(0, goal_id)  # Push to top

        # Auto-activate if not blocked and room available
        if not goal.is_blocked() and len(self.active_goals) < self.max_active_goals:
            goal.status = GoalStatus.ACTIVE
            self.active_goals.add(goal_id)

        return goal

    def pop_goal(self) -> Optional[Goal]:
        """
        Pop the top goal from the stack.

        Returns:
            The popped Goal, or None if stack is empty.
        """
        if not self.goal_order:
            return None

        goal_id = self.goal_order.pop(0)
        goal = self.goals.pop(goal_id, None)

        if goal_id in self.active_goals:
            self.active_goals.discard(goal_id)

        return goal

    def update_progress(
        self,
        goal_id: str,
        progress: float,
        force: bool = False,
    ) -> bool:
        """
        Update progress for a goal (monotonic).

        Args:
            goal_id: The goal to update.
            progress: New progress value (0.0-1.0).
            force: If True, allow non-monotonic updates (use carefully).

        Returns:
            True if progress was updated, False otherwise.
        """
        if goal_id not in self.goals:
            return False

        goal = self.goals[goal_id]

        if force:
            goal.progress = max(0.0, min(1.0, progress))
            if goal.progress >= 1.0:
                goal.status = GoalStatus.ACHIEVED
                goal.completed_at = datetime.now()
                self._achieved_count += 1
                self.active_goals.discard(goal_id)
                self._unblock_dependents(goal_id)
            return True

        result = goal.update_progress(progress)

        if result and goal.status == GoalStatus.ACHIEVED:
            self._achieved_count += 1
            self.active_goals.discard(goal_id)
            self._unblock_dependents(goal_id)

        return result

    def _unblock_dependents(self, completed_goal_id: str) -> None:
        """Unblock goals that were waiting on a completed goal."""
        for goal in self.goals.values():
            if completed_goal_id in goal.blocking_goals:
                goal.blocking_goals.discard(completed_goal_id)

                # If no longer blocked and not active, activate
                if not goal.is_blocked() and goal.is_achievable():
                    if len(self.active_goals) < self.max_active_goals:
                        goal.status = GoalStatus.ACTIVE
                        self.active_goals.add(goal.id)

    def get_progress(self, goal_id: str) -> float:
        """Get current progress for a goal."""
        if goal_id not in self.goals:
            return 0.0
        return self.goals[goal_id].progress

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self.goals.get(goal_id)

    def get_active_goals(self) -> List[Goal]:
        """Get all currently active goals."""
        return [
            self.goals[gid]
            for gid in self.active_goals
            if gid in self.goals
        ]

    def get_goals_by_status(self, status: GoalStatus) -> List[Goal]:
        """Get all goals with a given status."""
        return [g for g in self.goals.values() if g.status == status]

    def get_goals_by_priority(
        self,
        min_priority: GoalPriority = GoalPriority.LOW,
    ) -> List[Goal]:
        """Get goals at or above a priority level, sorted by priority."""
        goals = [
            g for g in self.goals.values()
            if g.priority.value >= min_priority.value and g.is_achievable()
        ]
        return sorted(goals, key=lambda g: -g.priority.value)

    def abandon_goal(self, goal_id: str, reason: str = "") -> bool:
        """
        Mark a goal as abandoned.

        Args:
            goal_id: The goal to abandon.
            reason: Why the goal was abandoned.

        Returns:
            True if goal was abandoned, False if not found.
        """
        if goal_id not in self.goals:
            return False

        goal = self.goals[goal_id]
        goal.status = GoalStatus.ABANDONED
        goal.metadata["abandon_reason"] = reason
        self.active_goals.discard(goal_id)
        self._abandoned_count += 1

        return True

    def check_achievement(
        self,
        goal_id: str,
        active_nodes: FrozenSet[str],
    ) -> float:
        """
        Check progress toward a goal based on active nodes.

        Args:
            goal_id: The goal to check.
            active_nodes: Currently active nodes.

        Returns:
            Updated progress value.
        """
        if goal_id not in self.goals:
            return 0.0

        goal = self.goals[goal_id]

        if not goal.target_nodes:
            return goal.progress

        # Calculate overlap between active and target nodes
        overlap = len(active_nodes & goal.target_nodes)
        total = len(goal.target_nodes)

        if total == 0:
            return goal.progress

        new_progress = overlap / total

        # Update if better (monotonic)
        goal.update_progress(new_progress)

        return goal.progress

    def get_subgoals(self, parent_id: str) -> List[Goal]:
        """Get all subgoals of a parent goal."""
        return [
            g for g in self.goals.values()
            if g.parent_id == parent_id
        ]

    def get_blocked_goals(self) -> List[Goal]:
        """Get all goals that are currently blocked."""
        return [g for g in self.goals.values() if g.is_blocked()]

    def get_statistics(self) -> Dict[str, Any]:
        """Get goal stack statistics."""
        by_status = {}
        for status in GoalStatus:
            by_status[status.value] = len(self.get_goals_by_status(status))

        return {
            "total_goals": len(self.goals),
            "active_goals": len(self.active_goals),
            "achieved_count": self._achieved_count,
            "abandoned_count": self._abandoned_count,
            "blocked_count": len(self.get_blocked_goals()),
            "by_status": by_status,
            "avg_progress": (
                sum(g.progress for g in self.goals.values()) / len(self.goals)
                if self.goals else 0.0
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize goal stack to dictionary."""
        return {
            "max_active_goals": self.max_active_goals,
            "goals": {gid: g.to_dict() for gid, g in self.goals.items()},
            "active_goals": list(self.active_goals),
            "goal_order": self.goal_order,
            "achieved_count": self._achieved_count,
            "abandoned_count": self._abandoned_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalStack":
        """Deserialize goal stack from dictionary."""
        stack = cls(max_active_goals=data.get("max_active_goals", 10))

        for gid, gdata in data.get("goals", {}).items():
            stack.goals[gid] = Goal.from_dict(gdata)

        stack.active_goals = set(data.get("active_goals", []))
        stack.goal_order = data.get("goal_order", [])
        stack._achieved_count = data.get("achieved_count", 0)
        stack._abandoned_count = data.get("abandoned_count", 0)

        return stack

    def clear(self) -> None:
        """Clear all goals."""
        self.goals.clear()
        self.active_goals.clear()
        self.goal_order.clear()
