"""
Kanban Orchestration Layer

This module implements the top-level orchestration using kanban principles:
- Continuous flow of incoming goals
- WIP limits for system stability
- Pull-based work assignment
- Bottleneck detection and relief
- Flow metrics and visualization

The orchestrator sits above Directors and manages the flow of goals
through the system.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Literal

from .types import (
    Constraint,
    Event,
    EventBus,
    Goal,
    Result,
    TaskStatus,
)
from .agents import Director, DirectorContext, HybridDirector
from .evolution import StrategyGenome, StrategyPool


# =============================================================================
# KANBAN COLUMN
# =============================================================================


@dataclass
class KanbanColumn:
    """A column on the kanban board."""

    name: str
    wip_limit: int | None = None
    entry_criteria: list[str] | None = None

    # Items in this column
    items: list[Goal] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of items in column."""
        return len(self.items)

    def can_accept(self) -> bool:
        """Check if column can accept more items."""
        if self.wip_limit is None:
            return True
        return self.count < self.wip_limit


@dataclass
class WIPViolation:
    """A WIP limit violation."""

    column: str
    current: int
    limit: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Bottleneck:
    """A detected bottleneck in the flow."""

    location: str
    queue_depth: int
    blocked_items: list[str]
    recommendation: str
    detected_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# FLOW METRICS
# =============================================================================


@dataclass
class FlowMetrics:
    """Kanban flow health indicators."""

    # Throughput
    completed_per_day: float = 0.0
    avg_cycle_time: timedelta = field(default_factory=lambda: timedelta())

    # Flow efficiency
    active_time_ratio: float = 0.0
    blocked_time_ratio: float = 0.0

    # WIP health
    wip_by_column: dict[str, int] = field(default_factory=dict)
    wip_violations: list[WIPViolation] = field(default_factory=list)

    # Bottlenecks
    bottleneck_column: str | None = None
    queue_depths: dict[str, int] = field(default_factory=dict)


# =============================================================================
# ORCHESTRATION BOARD
# =============================================================================


@dataclass
class OrchestrationBoard:
    """Kanban board for goal flow management."""

    columns: list[KanbanColumn] = field(default_factory=lambda: [
        KanbanColumn(
            name="backlog",
            wip_limit=None,
            entry_criteria=None,
        ),
        KanbanColumn(
            name="ready",
            wip_limit=10,
            entry_criteria=["goal_is_clear", "resources_available"],
        ),
        KanbanColumn(
            name="in_progress",
            wip_limit=3,
            entry_criteria=["director_assigned", "strategy_selected"],
        ),
        KanbanColumn(
            name="review",
            wip_limit=5,
            entry_criteria=["execution_complete", "outputs_ready"],
        ),
        KanbanColumn(
            name="done",
            wip_limit=None,
            entry_criteria=["user_accepted", "feedback_collected"],
        ),
    ])

    # Metrics
    metrics: FlowMetrics = field(default_factory=FlowMetrics)

    # History
    completed_goals: list[Goal] = field(default_factory=list)

    def get_column(self, name: str) -> KanbanColumn | None:
        """Get a column by name."""
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def get_wip(self, column_name: str) -> int:
        """Get WIP count for a column."""
        column = self.get_column(column_name)
        return column.count if column else 0

    def get_limit(self, column_name: str) -> int | None:
        """Get WIP limit for a column."""
        column = self.get_column(column_name)
        return column.wip_limit if column else None

    def add_to_column(self, goal: Goal, column_name: str) -> bool:
        """Add a goal to a column."""
        column = self.get_column(column_name)
        if not column:
            return False

        if not column.can_accept():
            return False

        column.items.append(goal)
        self._update_metrics()
        return True

    def move(
        self,
        goal: Goal,
        from_column: str,
        to_column: str,
    ) -> bool:
        """Move a goal between columns."""
        source = self.get_column(from_column)
        target = self.get_column(to_column)

        if not source or not target:
            return False

        if goal not in source.items:
            return False

        if not target.can_accept():
            return False

        source.items.remove(goal)
        target.items.append(goal)
        self._update_metrics()
        return True

    def _update_metrics(self) -> None:
        """Update flow metrics."""
        self.metrics.wip_by_column = {
            column.name: column.count
            for column in self.columns
        }

        # Detect bottlenecks
        for i, column in enumerate(self.columns[:-1]):
            next_column = self.columns[i + 1]
            if next_column.wip_limit and next_column.count >= next_column.wip_limit:
                if column.count > 0:
                    self.metrics.bottleneck_column = next_column.name

    def visualize(self) -> str:
        """Render board state for visibility."""
        lines = ["┌" + "─" * 70 + "┐"]
        lines.append("│" + " ORCHESTRATION KANBAN BOARD ".center(70) + "│")
        lines.append("├" + "─" * 70 + "┤")

        for column in self.columns:
            wip_status = ""
            if column.wip_limit:
                ratio = column.count / column.wip_limit
                if ratio > 1.0:
                    wip_status = " ⚠️  OVER"
                elif ratio > 0.8:
                    wip_status = " ⚡ near"
                else:
                    wip_status = " ✓"

            limit_str = str(column.wip_limit) if column.wip_limit else "∞"
            line = f"│ {column.name:15} │ {column.count:3}/{limit_str:>3}{wip_status:10} │"
            lines.append(line.ljust(71) + "│")

        lines.append("└" + "─" * 70 + "┘")
        return "\n".join(lines)


# =============================================================================
# KANBAN POLICIES
# =============================================================================


@dataclass
class KanbanPolicies:
    """Explicit policies for kanban flow management."""

    # Class of service
    expedite_allowed: bool = True
    expedite_wip_limit: int = 1

    # Blocking
    max_blocked_time: timedelta = field(
        default_factory=lambda: timedelta(hours=2)
    )
    escalate_after_blocked: timedelta = field(
        default_factory=lambda: timedelta(minutes=30)
    )

    # WIP
    wip_violation_action: Literal["block_new_pulls", "alert_only"] = (
        "block_new_pulls"
    )

    # Aging
    aging_threshold_days: int = 3
    aging_action: Literal["priority_boost", "escalate"] = "priority_boost"

    def handle_wip_violation(
        self,
        violation: WIPViolation,
    ) -> dict[str, Any]:
        """Handle a WIP limit violation."""
        if self.wip_violation_action == "block_new_pulls":
            return {
                "action": "block",
                "message": f"WIP limit exceeded in {violation.column}",
            }
        else:
            return {
                "action": "alert",
                "message": f"WIP at {violation.current}/{violation.limit} in {violation.column}",
            }


# =============================================================================
# KANBAN ORCHESTRATOR
# =============================================================================


class KanbanOrchestrator:
    """
    Pull-based orchestration using kanban principles.

    Key responsibilities:
    - Manage goal flow through the system
    - Enforce WIP limits
    - Detect and address bottlenecks
    - Provide visibility into system state
    - Coordinate with evolutionary layer
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        strategy_pool: StrategyPool | None = None,
    ):
        self.board = OrchestrationBoard()
        self.policies = KanbanPolicies()
        self.event_bus = event_bus or EventBus()
        self.strategy_pool = strategy_pool
        self.exploration_rate = 0.1

        # Active directors
        self.directors: dict[str, Director] = {}

        # Tracking
        self.goals_processed = 0
        self.total_cycle_time = timedelta()

    # =========================================================================
    # GOAL SUBMISSION
    # =========================================================================

    async def submit_goal(self, goal: Goal) -> bool:
        """Submit a new goal to the backlog."""
        goal.created_at = datetime.now()
        goal.status = TaskStatus.PENDING

        success = self.board.add_to_column(goal, "backlog")

        if success:
            await self.event_bus.publish(Event(
                type="orchestrator.goal_submitted",
                payload={"goal_id": goal.id, "description": goal.description},
            ))

            # Try to advance to ready
            await self._try_advance_to_ready(goal)

        return success

    async def _try_advance_to_ready(self, goal: Goal) -> bool:
        """Try to move a goal from backlog to ready."""
        # Check entry criteria
        if not self._meets_ready_criteria(goal):
            return False

        success = self.board.move(goal, "backlog", "ready")
        if success:
            goal.entered_ready_at = datetime.now()
            goal.status = TaskStatus.READY

        return success

    def _meets_ready_criteria(self, goal: Goal) -> bool:
        """Check if a goal meets ready criteria."""
        # Criteria: goal is clear, resources available
        return len(goal.description) > 0

    # =========================================================================
    # PULL-BASED WORK ASSIGNMENT
    # =========================================================================

    async def pull_next_goal(self) -> Goal | None:
        """
        Directors pull work when they have capacity.

        Not push-based assignment—respects WIP limits.
        """
        # Check WIP limit
        in_progress = self.board.get_column("in_progress")
        if in_progress and not in_progress.can_accept():
            return None

        # Pull highest priority from ready
        ready = self.board.get_column("ready")
        if not ready or not ready.items:
            return None

        prioritized = self._prioritize(ready.items)
        goal = prioritized[0]

        # Move to in_progress
        success = self.board.move(goal, "ready", "in_progress")
        if success:
            goal.status = TaskStatus.IN_PROGRESS
            await self.event_bus.publish(Event(
                type="orchestrator.goal_pulled",
                payload={"goal_id": goal.id},
            ))

        return goal if success else None

    def _prioritize(self, goals: list[Goal]) -> list[Goal]:
        """Prioritize goals considering age, urgency, value."""
        return sorted(goals, key=lambda g: (
            -g.urgency,
            -self._age_factor(g),
            -g.value,
            g.cost,
        ))

    def _age_factor(self, goal: Goal) -> float:
        """Aging: older items get priority boost."""
        if not goal.entered_ready_at:
            return 0.0

        age = datetime.now() - goal.entered_ready_at
        age_days = age.total_seconds() / 86400
        return min(age_days * 0.1, 1.0)

    # =========================================================================
    # DIRECTOR MANAGEMENT
    # =========================================================================

    async def assign_director(self, goal: Goal) -> Director:
        """Create and assign a director for a goal."""
        director_id = f"director-{goal.id}"

        # Select strategy
        genome = self._select_strategy(goal)

        # Create director context
        context = DirectorContext(
            role="goal_director",
            goal=goal.description,
            scope=self._create_scope(goal),
            can_spawn=["worker"],
            tools_available=["search", "read", "write"],
            event_bus=self.event_bus,
        )

        # Create director
        director = HybridDirector(director_id, context)
        self.directors[director_id] = director

        return director

    def _select_strategy(self, goal: Goal) -> StrategyGenome | None:
        """Select strategy with exploration/exploitation tradeoff."""
        if not self.strategy_pool:
            return None

        # Epsilon-greedy
        if random.random() < self.exploration_rate:
            # Explore
            return self.strategy_pool.get_random()
        else:
            # Exploit
            return self.strategy_pool.get_best_for(goal.description)

    def _create_scope(self, goal: Goal) -> Any:
        """Create scope for a goal."""
        from .types import Scope
        return Scope()

    # =========================================================================
    # WIP LIMIT ENFORCEMENT
    # =========================================================================

    def enforce_wip_limits(self) -> list[dict[str, Any]]:
        """Check and react to WIP violations."""
        actions = []

        for column in self.board.columns:
            if column.wip_limit and column.count > column.wip_limit:
                violation = WIPViolation(
                    column=column.name,
                    current=column.count,
                    limit=column.wip_limit,
                )
                self.board.metrics.wip_violations.append(violation)

                action = self.policies.handle_wip_violation(violation)
                actions.append(action)

        return actions

    # =========================================================================
    # BOTTLENECK DETECTION
    # =========================================================================

    def detect_bottlenecks(self) -> list[Bottleneck]:
        """Find where work is piling up."""
        bottlenecks = []

        for i, column in enumerate(self.board.columns[:-1]):
            next_column = self.board.columns[i + 1]

            # Bottleneck: queue building before a full column
            if next_column.wip_limit:
                if (column.count > 0 and
                    next_column.count >= next_column.wip_limit):

                    bottleneck = Bottleneck(
                        location=next_column.name,
                        queue_depth=column.count,
                        blocked_items=[g.id for g in column.items],
                        recommendation=self._recommend_bottleneck_action(
                            next_column
                        ),
                    )
                    bottlenecks.append(bottleneck)

        return bottlenecks

    def _recommend_bottleneck_action(self, column: KanbanColumn) -> str:
        """Suggest how to relieve bottleneck."""
        recommendations = {
            "in_progress": "Consider: swarm on blocked items, or temporarily increase WIP",
            "review": "Consider: expedite reviews, or batch similar items",
        }
        return recommendations.get(
            column.name,
            "Investigate column-specific constraints"
        )

    # =========================================================================
    # GOAL COMPLETION
    # =========================================================================

    async def complete_goal(
        self,
        goal: Goal,
        result: Result,
    ) -> None:
        """Mark a goal as complete."""
        # Move through review to done
        self.board.move(goal, "in_progress", "review")

        # For now, auto-advance to done
        self.board.move(goal, "review", "done")

        goal.status = TaskStatus.COMPLETED
        self.board.completed_goals.append(goal)

        # Update metrics
        self.goals_processed += 1
        if goal.entered_ready_at:
            cycle_time = datetime.now() - goal.entered_ready_at
            self.total_cycle_time += cycle_time

        await self.event_bus.publish(Event(
            type="orchestrator.goal_completed",
            payload={
                "goal_id": goal.id,
                "success": result.success,
            },
        ))

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def run(self) -> None:
        """Main orchestration loop."""
        while True:
            # 1. Try to pull and assign work
            goal = await self.pull_next_goal()
            if goal:
                director = await self.assign_director(goal)
                # Run director in background
                asyncio.create_task(self._run_director(goal, director))

            # 2. Check WIP limits
            self.enforce_wip_limits()

            # 3. Detect bottlenecks
            bottlenecks = self.detect_bottlenecks()
            for bottleneck in bottlenecks:
                await self.event_bus.publish(Event(
                    type="orchestrator.bottleneck_detected",
                    payload={
                        "location": bottleneck.location,
                        "queue_depth": bottleneck.queue_depth,
                    },
                ))

            # 4. Advance backlog items to ready
            backlog = self.board.get_column("backlog")
            if backlog:
                for goal in list(backlog.items):
                    await self._try_advance_to_ready(goal)

            await asyncio.sleep(0.5)

    async def _run_director(self, goal: Goal, director: Director) -> None:
        """Run a director and handle completion."""
        try:
            result = await director.run()
            await self.complete_goal(goal, result)
        except Exception as e:
            await self.complete_goal(
                goal,
                Result(success=False, error=str(e))
            )

    # =========================================================================
    # METRICS AND VISUALIZATION
    # =========================================================================

    def get_flow_metrics(self) -> FlowMetrics:
        """Get current flow metrics."""
        metrics = self.board.metrics

        # Calculate throughput
        if self.goals_processed > 0:
            # Simplified: just use count
            metrics.completed_per_day = self.goals_processed

            # Average cycle time
            if self.goals_processed > 0:
                avg_seconds = (
                    self.total_cycle_time.total_seconds() /
                    self.goals_processed
                )
                metrics.avg_cycle_time = timedelta(seconds=avg_seconds)

        return metrics

    def visualize(self) -> str:
        """Get visualization of current board state."""
        return self.board.visualize()


# =============================================================================
# EVOLUTIONARY ORCHESTRATOR
# =============================================================================


class EvolutionaryOrchestrator(KanbanOrchestrator):
    """
    Orchestrator that evolves strategies over time.

    Extends KanbanOrchestrator with:
    - Strategy selection
    - Trace collection for evolution
    - Fitness evaluation integration
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        strategy_pool: StrategyPool | None = None,
        exploration_rate: float = 0.1,
    ):
        super().__init__(event_bus, strategy_pool)
        self.exploration_rate = exploration_rate
        self.trace_buffer: list[Any] = []

    async def _run_director(self, goal: Goal, director: Director) -> None:
        """Run director with trace collection."""
        from .evolution import ExecutionTrace, ExecutionMetrics

        trace = ExecutionTrace(
            trace_id=f"trace-{goal.id}",
            goal=goal.description,
            strategy_genome_id="default",  # Would come from selection
        )

        start_time = datetime.now()

        try:
            result = await director.run()

            # Collect metrics
            trace.metrics = ExecutionMetrics(
                total_duration_ms=(
                    datetime.now() - start_time
                ).total_seconds() * 1000,
                goal_achieved=result.success,
            )

            trace.result = result
            self.trace_buffer.append(trace)

            await self.complete_goal(goal, result)

            # Maybe trigger evolution
            await self._maybe_evolve()

        except Exception as e:
            trace.result = Result(success=False, error=str(e))
            self.trace_buffer.append(trace)
            await self.complete_goal(
                goal,
                Result(success=False, error=str(e))
            )

    async def _maybe_evolve(self) -> None:
        """Decide whether to trigger evolution."""
        should_evolve = any([
            len(self.trace_buffer) >= 100,
            # Other triggers...
        ])

        if should_evolve and self.strategy_pool:
            from .evolution import StrategyEvolver

            evolver = StrategyEvolver(self.strategy_pool)
            # Would trigger evolution
            self.trace_buffer.clear()

            await self.event_bus.publish(Event(
                type="evolution.triggered",
                payload={"trace_count": len(self.trace_buffer)},
            ))
