"""
Nested Loop Executor: Hierarchical QAPV Cycle Management.

This module provides a complete implementation of nested cognitive loops,
enabling hierarchical goal decomposition with parent-child relationships.

Key Features:
    - Hierarchical goal decomposition into nested loops
    - Depth tracking and limiting to prevent infinite recursion
    - Context propagation from parent to child loops
    - Result aggregation from children back to parents
    - Early termination and loop breaking at any level
    - Integration with existing CognitiveLoop infrastructure

Example:
    >>> executor = NestedLoopExecutor(max_depth=3)
    >>> root_id = executor.start_root("Implement authentication")
    >>> child_id = executor.spawn_child(root_id, "Design database schema")
    >>> executor.advance(child_id)
    >>> executor.record_answer(child_id, "Use PostgreSQL with user table")
    >>> parent_id = executor.complete(child_id, {"schema": "user_table.sql"})
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from .cognitive_loop import (
    CognitiveLoop,
    CognitiveLoopManager,
    LoopPhase,
    LoopStatus,
    TerminationReason,
)


@dataclass
class LoopContext:
    """
    Context for a nested loop instance.

    Tracks the hierarchical position and accumulated state of a loop
    within the nested execution structure.

    Attributes:
        depth: How many levels deep this loop is (0 = root)
        parent_id: ID of parent loop, None for root loops
        goal: What this loop is trying to achieve
        accumulated_answers: Answers collected during this loop's execution
        child_results: Results from completed child loops (child_id -> result)
        metadata: Additional context-specific data
    """
    depth: int
    parent_id: Optional[str]
    goal: str
    accumulated_answers: List[str] = field(default_factory=list)
    child_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_answer(self, answer: str) -> None:
        """Add an answer to this context."""
        self.accumulated_answers.append(answer)

    def add_child_result(self, child_id: str, result: Any) -> None:
        """Record a result from a child loop."""
        self.child_results[child_id] = result

    def get_all_answers(self) -> List[str]:
        """Get all accumulated answers."""
        return self.accumulated_answers.copy()


class NestedLoopExecutor:
    """
    Executor for hierarchical cognitive loops with automatic nesting management.

    Manages a tree of cognitive loops where parent loops can spawn child loops
    for subtasks, and child results are aggregated back to parents.

    Features:
        - Depth limiting to prevent infinite recursion
        - Automatic phase advancement through QAPV cycle
        - Result aggregation from children to parents
        - Early termination with reason tracking
        - Full integration with CognitiveLoopManager

    Example:
        >>> executor = NestedLoopExecutor(max_depth=5)
        >>> root = executor.start_root("Build feature")
        >>> child = executor.spawn_child(root, "Write tests")
        >>> executor.advance(child)  # Move to next phase
        >>> executor.record_answer(child, "Used pytest framework")
        >>> parent = executor.complete(child, {"tests_written": 10})
    """

    def __init__(self, max_depth: int = 5):
        """
        Initialize the nested loop executor.

        Args:
            max_depth: Maximum nesting depth allowed (default: 5)

        Raises:
            ValueError: If max_depth < 1
        """
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")

        self._max_depth = max_depth
        self._loops: Dict[str, CognitiveLoop] = {}
        self._contexts: Dict[str, LoopContext] = {}
        self._manager = CognitiveLoopManager()

        # Track phase order for advancement
        self._phase_order = [
            LoopPhase.QUESTION,
            LoopPhase.ANSWER,
            LoopPhase.PRODUCE,
            LoopPhase.VERIFY,
        ]

    def start_root(self, goal: str) -> str:
        """
        Start a root-level loop.

        Creates a new top-level cognitive loop and initializes its context.

        Args:
            goal: What this root loop should achieve

        Returns:
            Loop ID for the newly created root loop

        Example:
            >>> executor = NestedLoopExecutor()
            >>> loop_id = executor.start_root("Implement user auth")
        """
        loop = self._manager.create_loop(goal=goal, parent_id=None)
        loop.start(initial_phase=LoopPhase.QUESTION)

        context = LoopContext(
            depth=0,
            parent_id=None,
            goal=goal,
        )

        self._loops[loop.id] = loop
        self._contexts[loop.id] = context

        return loop.id

    def spawn_child(self, parent_id: str, subgoal: str) -> str:
        """
        Spawn a child loop under the specified parent.

        Creates a nested loop for a subtask, maintaining the hierarchical
        relationship and enforcing depth limits.

        Args:
            parent_id: ID of the parent loop
            subgoal: What the child loop should achieve

        Returns:
            Loop ID for the newly created child loop

        Raises:
            KeyError: If parent_id is not found
            RecursionError: If max_depth would be exceeded
            ValueError: If parent loop is not active

        Example:
            >>> parent = executor.start_root("Build API")
            >>> child = executor.spawn_child(parent, "Design endpoints")
        """
        if parent_id not in self._loops:
            raise KeyError(f"Parent loop {parent_id} not found")

        parent_loop = self._loops[parent_id]
        parent_context = self._contexts[parent_id]

        if parent_loop.status != LoopStatus.ACTIVE:
            raise ValueError(f"Parent loop {parent_id} is not active (status: {parent_loop.status})")

        # Check depth limit
        child_depth = parent_context.depth + 1
        if child_depth >= self._max_depth:
            raise RecursionError(
                f"Maximum nesting depth {self._max_depth} would be exceeded "
                f"(current depth: {parent_context.depth})"
            )

        # Create child loop using parent's spawn method
        child_loop = parent_loop.spawn_child(goal=subgoal)
        child_loop.start(initial_phase=LoopPhase.QUESTION)

        # Create child context
        child_context = LoopContext(
            depth=child_depth,
            parent_id=parent_id,
            goal=subgoal,
        )

        self._loops[child_loop.id] = child_loop
        self._contexts[child_loop.id] = child_context

        # Pause parent while child is active
        parent_loop.pause(reason=f"Spawned child loop {child_loop.id} for: {subgoal}")

        return child_loop.id

    def advance(self, loop_id: str) -> LoopPhase:
        """
        Advance a loop to the next phase in the QAPV cycle.

        Transitions through: QUESTION → ANSWER → PRODUCE → VERIFY → QUESTION (repeat)

        Args:
            loop_id: ID of the loop to advance

        Returns:
            The new current phase after advancement

        Raises:
            KeyError: If loop_id is not found
            ValueError: If loop is not active

        Example:
            >>> loop_id = executor.start_root("Test feature")
            >>> # Starts in QUESTION phase
            >>> executor.advance(loop_id)  # Now in ANSWER
            >>> executor.advance(loop_id)  # Now in PRODUCE
        """
        if loop_id not in self._loops:
            raise KeyError(f"Loop {loop_id} not found")

        loop = self._loops[loop_id]

        if loop.status != LoopStatus.ACTIVE:
            raise ValueError(f"Loop {loop_id} is not active (status: {loop.status})")

        current_phase = loop.current_phase
        if current_phase is None:
            raise ValueError(f"Loop {loop_id} has no current phase")

        # Find next phase in cycle
        current_idx = self._phase_order.index(current_phase)
        next_idx = (current_idx + 1) % len(self._phase_order)
        next_phase = self._phase_order[next_idx]

        # Transition to next phase
        loop.transition(
            to_phase=next_phase,
            reason=f"Advanced from {current_phase.value} to {next_phase.value}"
        )

        return next_phase

    def record_answer(self, loop_id: str, answer: str) -> None:
        """
        Record an answer in the current loop.

        Stores the answer in both the loop's phase context and the
        nested loop context for aggregation.

        Args:
            loop_id: ID of the loop
            answer: Answer text to record

        Raises:
            KeyError: If loop_id is not found

        Example:
            >>> loop_id = executor.start_root("Research database")
            >>> executor.record_answer(loop_id, "PostgreSQL is best fit")
        """
        if loop_id not in self._loops:
            raise KeyError(f"Loop {loop_id} not found")

        loop = self._loops[loop_id]
        context = self._contexts[loop_id]

        # Add to loop's current phase context
        loop.add_note(f"ANSWER: {answer}")

        # Add to nested context for aggregation
        context.add_answer(answer)

    def complete(self, loop_id: str, result: Any) -> Optional[str]:
        """
        Complete a loop with the given result.

        Marks the loop as completed and propagates the result to the parent
        if this is a child loop. Resumes the parent loop if applicable.

        Args:
            loop_id: ID of the loop to complete
            result: Result value to return from this loop

        Returns:
            Parent loop ID if this is a child, None if this is a root loop

        Raises:
            KeyError: If loop_id is not found

        Example:
            >>> child = executor.spawn_child(parent, "Write docs")
            >>> parent_id = executor.complete(child, {"pages": 5})
            >>> # parent_id is now returned and parent is resumed
        """
        if loop_id not in self._loops:
            raise KeyError(f"Loop {loop_id} not found")

        loop = self._loops[loop_id]
        context = self._contexts[loop_id]

        # Complete the loop
        loop.complete(reason=TerminationReason.SUCCESS)
        loop.add_note(f"COMPLETED with result: {result}")

        # If this is a child loop, propagate result to parent
        parent_id = context.parent_id
        if parent_id is not None:
            parent_loop = self._loops[parent_id]
            parent_context = self._contexts[parent_id]

            # Add result to parent's context
            parent_context.add_child_result(loop_id, result)
            parent_loop.add_note(f"Child loop {loop_id} completed with result: {result}")

            # Resume parent
            if parent_loop.status == LoopStatus.PAUSED:
                parent_loop.resume()

        return parent_id

    def get_context(self, loop_id: str) -> LoopContext:
        """
        Get the current context for a loop.

        Args:
            loop_id: ID of the loop

        Returns:
            LoopContext for the specified loop

        Raises:
            KeyError: If loop_id is not found

        Example:
            >>> ctx = executor.get_context(loop_id)
            >>> print(f"Depth: {ctx.depth}, Answers: {len(ctx.accumulated_answers)}")
        """
        if loop_id not in self._contexts:
            raise KeyError(f"Loop {loop_id} not found")

        return self._contexts[loop_id]

    def break_loop(self, loop_id: str, reason: str) -> None:
        """
        Break out of a loop early with a reason.

        Abandons the loop without completing it normally, recording
        the reason for the early termination.

        Args:
            loop_id: ID of the loop to break
            reason: Explanation for why the loop is being broken

        Raises:
            KeyError: If loop_id is not found

        Example:
            >>> executor.break_loop(loop_id, "Requirements changed")
        """
        if loop_id not in self._loops:
            raise KeyError(f"Loop {loop_id} not found")

        loop = self._loops[loop_id]
        context = self._contexts[loop_id]

        # Abandon the loop with reason
        loop.abandon(reason=reason)

        # If this is a child, resume parent
        if context.parent_id is not None:
            parent_loop = self._loops[context.parent_id]
            parent_loop.add_note(f"Child loop {loop_id} was broken: {reason}")

            if parent_loop.status == LoopStatus.PAUSED:
                parent_loop.resume()

    def get_loop(self, loop_id: str) -> CognitiveLoop:
        """
        Get the underlying CognitiveLoop instance.

        Args:
            loop_id: ID of the loop

        Returns:
            CognitiveLoop instance

        Raises:
            KeyError: If loop_id is not found
        """
        if loop_id not in self._loops:
            raise KeyError(f"Loop {loop_id} not found")

        return self._loops[loop_id]

    def get_all_loops(self) -> Dict[str, CognitiveLoop]:
        """
        Get all loops managed by this executor.

        Returns:
            Dictionary mapping loop IDs to CognitiveLoop instances
        """
        return self._loops.copy()

    def get_active_loops(self) -> List[str]:
        """
        Get IDs of all currently active loops.

        Returns:
            List of loop IDs that are currently active
        """
        return [
            loop_id for loop_id, loop in self._loops.items()
            if loop.status == LoopStatus.ACTIVE
        ]

    def get_loop_hierarchy(self, loop_id: str) -> List[str]:
        """
        Get the full hierarchy path from root to the specified loop.

        Args:
            loop_id: ID of the loop

        Returns:
            List of loop IDs from root to the specified loop (inclusive)

        Raises:
            KeyError: If loop_id is not found

        Example:
            >>> hierarchy = executor.get_loop_hierarchy(child_id)
            >>> # Returns: [root_id, parent_id, child_id]
        """
        if loop_id not in self._contexts:
            raise KeyError(f"Loop {loop_id} not found")

        path = []
        current_id = loop_id

        while current_id is not None:
            path.insert(0, current_id)
            current_id = self._contexts[current_id].parent_id

        return path

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the executor's current state.

        Returns:
            Dictionary with statistics about loops and nesting
        """
        total_loops = len(self._loops)
        active_loops = len(self.get_active_loops())

        # Calculate depth statistics
        depths = [ctx.depth for ctx in self._contexts.values()]
        max_depth_reached = max(depths) if depths else 0

        # Count by status
        status_counts = {}
        for loop in self._loops.values():
            status_name = loop.status.name
            status_counts[status_name] = status_counts.get(status_name, 0) + 1

        return {
            'total_loops': total_loops,
            'active_loops': active_loops,
            'max_depth_limit': self._max_depth,
            'max_depth_reached': max_depth_reached,
            'status_counts': status_counts,
        }
