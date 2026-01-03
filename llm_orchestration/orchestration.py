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
    type: str  # "wip_violation", "queue_buildup", "slow_stage", "blocked_work"
    severity: float  # 0.0-1.0
    queue_depth: int
    blocked_items: list[str]
    recommendation: str
    detected_at: datetime = field(default_factory=datetime.now)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class Optimization:
    """Suggested optimization for flow improvement."""

    type: str  # "rebalance", "scale", "escalate", "throttle"
    target: str  # Column or worker affected
    action: str  # Specific action to take
    priority: int  # 1-5, higher = more urgent
    estimated_impact: float  # Expected improvement %
    rationale: str = ""
    prerequisites: list[str] = field(default_factory=list)


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

    # Historical data for trend analysis
    queue_history: dict[str, list[int]] = field(default_factory=dict)
    cycle_time_by_stage: dict[str, list[timedelta]] = field(default_factory=dict)
    blocked_items: dict[str, datetime] = field(default_factory=dict)


# =============================================================================
# BOTTLENECK DETECTOR
# =============================================================================


class BottleneckDetector:
    """Detects bottlenecks in agent workflow using multiple algorithms."""

    def __init__(
        self,
        wip_threshold: float = 0.9,
        queue_growth_threshold: float = 0.2,
        slow_stage_threshold: float = 0.5,
        blocked_time_threshold: timedelta = timedelta(minutes=30),
    ):
        """
        Initialize detector with thresholds.

        Args:
            wip_threshold: Fraction of WIP limit to trigger warning (0.9 = 90%)
            queue_growth_threshold: Growth rate to trigger queue buildup alert
            slow_stage_threshold: Slowdown ratio to trigger slow stage alert
            blocked_time_threshold: Time before item is considered blocked
        """
        self.wip_threshold = wip_threshold
        self.queue_growth_threshold = queue_growth_threshold
        self.slow_stage_threshold = slow_stage_threshold
        self.blocked_time_threshold = blocked_time_threshold

    def detect(
        self,
        flow_metrics: FlowMetrics,
        board: OrchestrationBoard,
    ) -> list[Bottleneck]:
        """
        Identify bottlenecks in the current flow.

        Uses multiple detection algorithms:
        - WIP violations
        - Queue buildup
        - Slow stages
        - Blocked work
        """
        bottlenecks = []
        bottlenecks.extend(self._detect_wip_violations(flow_metrics, board))
        bottlenecks.extend(self._detect_queue_buildup(flow_metrics, board))
        bottlenecks.extend(self._detect_slow_stages(flow_metrics, board))
        bottlenecks.extend(self._detect_blocked_work(flow_metrics, board))
        return bottlenecks

    def _detect_wip_violations(
        self,
        flow_metrics: FlowMetrics,
        board: OrchestrationBoard,
    ) -> list[Bottleneck]:
        """Detect columns exceeding or near WIP limits."""
        bottlenecks = []

        for column in board.columns:
            if column.wip_limit is None:
                continue

            ratio = column.count / column.wip_limit if column.wip_limit > 0 else 0

            # Exceeding limit
            if ratio > 1.0:
                severity = min((ratio - 1.0) / 0.5, 1.0)  # 0-1 scale
                bottlenecks.append(Bottleneck(
                    location=column.name,
                    type="wip_violation",
                    severity=severity,
                    queue_depth=column.count,
                    blocked_items=[g.id for g in column.items],
                    recommendation=f"WIP limit exceeded: {column.count}/{column.wip_limit}. "
                                   f"Consider: reduce intake, increase capacity, or temporarily raise limit.",
                    metrics={
                        "wip_ratio": ratio,
                        "excess_count": column.count - column.wip_limit,
                    }
                ))

            # Near limit (warning)
            elif ratio >= self.wip_threshold:
                severity = (ratio - self.wip_threshold) / (1.0 - self.wip_threshold) * 0.5
                bottlenecks.append(Bottleneck(
                    location=column.name,
                    type="wip_violation",
                    severity=severity,
                    queue_depth=column.count,
                    blocked_items=[],
                    recommendation=f"WIP approaching limit: {column.count}/{column.wip_limit}. "
                                   f"Monitor closely and prepare to reduce intake.",
                    metrics={
                        "wip_ratio": ratio,
                        "remaining_capacity": column.wip_limit - column.count,
                    }
                ))

        return bottlenecks

    def _detect_queue_buildup(
        self,
        flow_metrics: FlowMetrics,
        board: OrchestrationBoard,
    ) -> list[Bottleneck]:
        """Detect growing queues indicating flow issues."""
        bottlenecks = []

        for column in board.columns:
            # Need historical data to detect growth
            if column.name not in flow_metrics.queue_history:
                flow_metrics.queue_history[column.name] = []

            history = flow_metrics.queue_history[column.name]
            current_depth = column.count

            # Add current reading
            history.append(current_depth)
            if len(history) > 10:  # Keep last 10 readings
                history.pop(0)

            # Need at least 3 readings to detect trend
            if len(history) < 3:
                continue

            # Calculate growth rate
            recent_avg = sum(history[-3:]) / 3
            older_avg = sum(history[:-3]) / max(len(history) - 3, 1)

            if older_avg == 0:
                continue

            growth_rate = (recent_avg - older_avg) / older_avg

            # Growing queue detected
            if growth_rate > self.queue_growth_threshold:
                severity = min(growth_rate / 0.5, 1.0)  # Scale to 0-1

                # Diagnose root cause
                root_cause = self._diagnose_queue_cause(column, board)

                bottlenecks.append(Bottleneck(
                    location=column.name,
                    type="queue_buildup",
                    severity=severity,
                    queue_depth=current_depth,
                    blocked_items=[g.id for g in column.items[:min(5, len(column.items))]],
                    recommendation=f"Queue growing at {growth_rate:.1%} rate. "
                                   f"Root cause: {root_cause}. "
                                   f"Consider: {self._queue_remediation(root_cause)}",
                    metrics={
                        "growth_rate": growth_rate,
                        "recent_avg": recent_avg,
                        "older_avg": older_avg,
                        "root_cause": root_cause,
                    }
                ))

        return bottlenecks

    def _diagnose_queue_cause(
        self,
        column: KanbanColumn,
        board: OrchestrationBoard,
    ) -> str:
        """Determine why queue is building up."""
        # Find next column
        idx = board.columns.index(column)
        if idx >= len(board.columns) - 1:
            return "end_of_flow"

        next_column = board.columns[idx + 1]

        # Next column is full (slow processing)
        if next_column.wip_limit and next_column.count >= next_column.wip_limit:
            return "downstream_bottleneck"

        # High intake rate
        if column.name == "backlog" or column.name == "ready":
            return "high_intake_rate"

        return "slow_processing"

    def _queue_remediation(self, root_cause: str) -> str:
        """Suggest remediation based on root cause."""
        remediation = {
            "downstream_bottleneck": "Address bottleneck in downstream stage",
            "high_intake_rate": "Throttle intake or increase ready capacity",
            "slow_processing": "Add parallel workers or optimize processing",
            "end_of_flow": "Review completion criteria",
        }
        return remediation.get(root_cause, "Investigate further")

    def _detect_slow_stages(
        self,
        flow_metrics: FlowMetrics,
        board: OrchestrationBoard,
    ) -> list[Bottleneck]:
        """Detect stages with abnormally long cycle times."""
        bottlenecks = []

        for column in board.columns:
            if column.name not in flow_metrics.cycle_time_by_stage:
                flow_metrics.cycle_time_by_stage[column.name] = []

            history = flow_metrics.cycle_time_by_stage[column.name]

            # Need baseline to compare
            if len(history) < 5:
                continue

            # Calculate baseline (median of historical data)
            sorted_history = sorted(history)
            baseline = sorted_history[len(sorted_history) // 2]

            # Calculate current average (last 3)
            if len(history) < 3:
                continue

            current_avg = sum(history[-3:], timedelta()) / 3

            # Check if current is significantly slower
            if baseline.total_seconds() == 0:
                continue

            slowdown_ratio = current_avg.total_seconds() / baseline.total_seconds()

            if slowdown_ratio > (1.0 + self.slow_stage_threshold):
                severity = min((slowdown_ratio - 1.0) / 1.0, 1.0)

                bottlenecks.append(Bottleneck(
                    location=column.name,
                    type="slow_stage",
                    severity=severity,
                    queue_depth=column.count,
                    blocked_items=[],
                    recommendation=f"Stage is {slowdown_ratio:.1%} slower than baseline. "
                                   f"Baseline: {baseline}, Current: {current_avg}. "
                                   f"Consider: investigate delays, parallelize work, or optimize process.",
                    metrics={
                        "slowdown_ratio": slowdown_ratio,
                        "baseline_seconds": baseline.total_seconds(),
                        "current_seconds": current_avg.total_seconds(),
                    }
                ))

        return bottlenecks

    def _detect_blocked_work(
        self,
        flow_metrics: FlowMetrics,
        board: OrchestrationBoard,
    ) -> list[Bottleneck]:
        """Detect items blocked for too long."""
        bottlenecks = []
        now = datetime.now()

        for column in board.columns:
            blocked_in_column = []

            for goal in column.items:
                # Check if item is in blocked tracking
                if goal.id in flow_metrics.blocked_items:
                    blocked_since = flow_metrics.blocked_items[goal.id]
                    blocked_duration = now - blocked_since

                    if blocked_duration > self.blocked_time_threshold:
                        blocked_in_column.append(goal.id)
                else:
                    # Check if item should be tracked as blocked
                    # For simplicity, consider items blocked if they've been
                    # in a column for too long without progress
                    if hasattr(goal, 'entered_ready_at') and goal.entered_ready_at:
                        time_in_stage = now - goal.entered_ready_at
                        if time_in_stage > self.blocked_time_threshold * 2:
                            flow_metrics.blocked_items[goal.id] = goal.entered_ready_at
                            blocked_in_column.append(goal.id)

            # Create bottleneck if blocked items found
            if blocked_in_column:
                blocked_ratio = len(blocked_in_column) / max(column.count, 1)
                severity = min(blocked_ratio * 2, 1.0)

                bottlenecks.append(Bottleneck(
                    location=column.name,
                    type="blocked_work",
                    severity=severity,
                    queue_depth=column.count,
                    blocked_items=blocked_in_column,
                    recommendation=f"{len(blocked_in_column)} items blocked for >{self.blocked_time_threshold}. "
                                   f"Consider: escalate, reassign, or remove blockers.",
                    metrics={
                        "blocked_count": len(blocked_in_column),
                        "blocked_ratio": blocked_ratio,
                        "threshold_minutes": self.blocked_time_threshold.total_seconds() / 60,
                    }
                ))

        return bottlenecks


# =============================================================================
# FLOW OPTIMIZER
# =============================================================================


class FlowOptimizer:
    """Suggests optimizations based on detected bottlenecks."""

    def __init__(self):
        """Initialize flow optimizer."""
        pass

    def suggest(self, bottlenecks: list[Bottleneck]) -> list[Optimization]:
        """
        Generate optimization suggestions.

        Analyzes bottlenecks and suggests concrete actions to improve flow.
        """
        optimizations = []

        for bottleneck in bottlenecks:
            if bottleneck.type == "wip_violation":
                optimizations.extend(self._optimize_wip_violation(bottleneck))
            elif bottleneck.type == "queue_buildup":
                optimizations.extend(self._optimize_queue_buildup(bottleneck))
            elif bottleneck.type == "slow_stage":
                optimizations.extend(self._optimize_slow_stage(bottleneck))
            elif bottleneck.type == "blocked_work":
                optimizations.extend(self._optimize_blocked_work(bottleneck))

        # Sort by priority and impact
        optimizations.sort(key=lambda o: (-o.priority, -o.estimated_impact))

        return optimizations

    def _optimize_wip_violation(self, bottleneck: Bottleneck) -> list[Optimization]:
        """Suggest fixes for WIP violations."""
        opts = []

        wip_ratio = bottleneck.metrics.get("wip_ratio", 0)

        if wip_ratio > 1.0:
            # Exceeded limit - urgent
            opts.append(Optimization(
                type="throttle",
                target=bottleneck.location,
                action=f"Block new work from entering {bottleneck.location} until WIP drops below limit",
                priority=5,
                estimated_impact=0.8,
                rationale="Prevent system overload by enforcing WIP limit",
                prerequisites=["downstream_capacity_available"],
            ))

            opts.append(Optimization(
                type="rebalance",
                target=bottleneck.location,
                action=f"Swarm {bottleneck.location}: redirect available workers to clear backlog",
                priority=4,
                estimated_impact=0.6,
                rationale="Increase throughput by adding temporary capacity",
                prerequisites=["workers_available"],
            ))
        else:
            # Near limit - preventive
            opts.append(Optimization(
                type="throttle",
                target=bottleneck.location,
                action=f"Reduce intake rate to {bottleneck.location} by 20%",
                priority=3,
                estimated_impact=0.4,
                rationale="Prevent WIP violation by reducing intake",
            ))

        return opts

    def _optimize_queue_buildup(self, bottleneck: Bottleneck) -> list[Optimization]:
        """Suggest fixes for queue buildup."""
        opts = []

        root_cause = bottleneck.metrics.get("root_cause", "unknown")
        growth_rate = bottleneck.metrics.get("growth_rate", 0)

        if root_cause == "downstream_bottleneck":
            opts.append(Optimization(
                type="rebalance",
                target=bottleneck.location,
                action=f"Address downstream bottleneck before adding to {bottleneck.location}",
                priority=4,
                estimated_impact=0.7,
                rationale="Queue buildup caused by downstream constraint",
                prerequisites=["downstream_bottleneck_resolved"],
            ))

        elif root_cause == "high_intake_rate":
            opts.append(Optimization(
                type="throttle",
                target=bottleneck.location,
                action=f"Throttle intake to {bottleneck.location} by {min(growth_rate * 100, 50):.0f}%",
                priority=4,
                estimated_impact=0.6,
                rationale="Match intake rate to processing capacity",
            ))

        elif root_cause == "slow_processing":
            opts.append(Optimization(
                type="scale",
                target=bottleneck.location,
                action=f"Add parallel workers to {bottleneck.location} to increase throughput",
                priority=3,
                estimated_impact=0.5,
                rationale="Increase processing capacity to clear queue",
                prerequisites=["workers_available"],
            ))

        return opts

    def _optimize_slow_stage(self, bottleneck: Bottleneck) -> list[Optimization]:
        """Suggest fixes for slow stages."""
        opts = []

        slowdown_ratio = bottleneck.metrics.get("slowdown_ratio", 1.0)

        opts.append(Optimization(
            type="escalate",
            target=bottleneck.location,
            action=f"Investigate why {bottleneck.location} is {slowdown_ratio:.1%} slower than baseline",
            priority=3,
            estimated_impact=0.6,
            rationale="Identify root cause of slowdown",
        ))

        if slowdown_ratio > 2.0:
            # Severe slowdown
            opts.append(Optimization(
                type="scale",
                target=bottleneck.location,
                action=f"Parallelize work in {bottleneck.location} to compensate for slowdown",
                priority=4,
                estimated_impact=0.7,
                rationale="Mitigate severe slowdown with parallel processing",
                prerequisites=["work_is_parallelizable"],
            ))

        opts.append(Optimization(
            type="rebalance",
            target=bottleneck.location,
            action=f"Review and optimize process for {bottleneck.location}",
            priority=2,
            estimated_impact=0.5,
            rationale="Long-term improvement to stage efficiency",
        ))

        return opts

    def _optimize_blocked_work(self, bottleneck: Bottleneck) -> list[Optimization]:
        """Suggest fixes for blocked work."""
        opts = []

        blocked_count = bottleneck.metrics.get("blocked_count", 0)
        blocked_ratio = bottleneck.metrics.get("blocked_ratio", 0)

        if blocked_ratio > 0.5:
            # Majority blocked - urgent
            opts.append(Optimization(
                type="escalate",
                target=bottleneck.location,
                action=f"Escalate {blocked_count} blocked items in {bottleneck.location} immediately",
                priority=5,
                estimated_impact=0.8,
                rationale="Majority of work is blocked - urgent intervention needed",
            ))
        else:
            opts.append(Optimization(
                type="escalate",
                target=bottleneck.location,
                action=f"Review and unblock {blocked_count} items in {bottleneck.location}",
                priority=4,
                estimated_impact=0.6,
                rationale="Clear blocked work to restore flow",
            ))

        opts.append(Optimization(
            type="rebalance",
            target=bottleneck.location,
            action=f"Reassign blocked items in {bottleneck.location} to available workers",
            priority=3,
            estimated_impact=0.5,
            rationale="Route around blockers when possible",
            prerequisites=["blockers_identified", "workers_available"],
        ))

        return opts


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

        # Bottleneck detection and optimization
        self.bottleneck_detector = BottleneckDetector()
        self.flow_optimizer = FlowOptimizer()

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
        """
        Find where work is piling up using comprehensive detection.

        Uses BottleneckDetector to identify:
        - WIP violations
        - Queue buildup
        - Slow stages
        - Blocked work
        """
        metrics = self.get_flow_metrics()
        return self.bottleneck_detector.detect(metrics, self.board)

    def get_optimizations(self, bottlenecks: list[Bottleneck] | None = None) -> list[Optimization]:
        """
        Get optimization suggestions based on bottlenecks.

        Args:
            bottlenecks: Optional list of bottlenecks. If None, will detect them.

        Returns:
            List of suggested optimizations, sorted by priority and impact.
        """
        if bottlenecks is None:
            bottlenecks = self.detect_bottlenecks()

        return self.flow_optimizer.suggest(bottlenecks)

    def apply_optimization(self, optimization: Optimization) -> bool:
        """
        Apply an optimization to the system.

        Args:
            optimization: The optimization to apply

        Returns:
            True if successfully applied, False otherwise
        """
        # Check prerequisites
        for prereq in optimization.prerequisites:
            if not self._check_prerequisite(prereq):
                return False

        # Apply based on type
        if optimization.type == "throttle":
            return self._apply_throttle(optimization)
        elif optimization.type == "rebalance":
            return self._apply_rebalance(optimization)
        elif optimization.type == "scale":
            return self._apply_scale(optimization)
        elif optimization.type == "escalate":
            return self._apply_escalate(optimization)

        return False

    def _check_prerequisite(self, prereq: str) -> bool:
        """Check if a prerequisite is met."""
        # Simplified: always return True
        # In real system, would check actual conditions
        return True

    def _apply_throttle(self, optimization: Optimization) -> bool:
        """Apply throttling optimization."""
        # Simplified: just log the action
        # In real system, would actually throttle intake
        return True

    def _apply_rebalance(self, optimization: Optimization) -> bool:
        """Apply rebalancing optimization."""
        # Simplified: just log the action
        # In real system, would actually rebalance workers
        return True

    def _apply_scale(self, optimization: Optimization) -> bool:
        """Apply scaling optimization."""
        # Simplified: just log the action
        # In real system, would actually add workers
        return True

    def _apply_escalate(self, optimization: Optimization) -> bool:
        """Apply escalation optimization."""
        # Simplified: just log the action
        # In real system, would actually escalate items
        return True

    def _recommend_bottleneck_action(self, column: KanbanColumn) -> str:
        """Suggest how to relieve bottleneck (legacy method)."""
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
        """Main orchestration loop with auto-optimization."""
        while True:
            # 1. Try to pull and assign work
            goal = await self.pull_next_goal()
            if goal:
                director = await self.assign_director(goal)
                # Run director in background
                asyncio.create_task(self._run_director(goal, director))

            # 2. Check WIP limits
            self.enforce_wip_limits()

            # 3. Detect bottlenecks and get optimizations
            bottlenecks = self.detect_bottlenecks()
            if bottlenecks:
                for bottleneck in bottlenecks:
                    await self.event_bus.publish(Event(
                        type="orchestrator.bottleneck_detected",
                        payload={
                            "location": bottleneck.location,
                            "type": bottleneck.type,
                            "severity": bottleneck.severity,
                            "queue_depth": bottleneck.queue_depth,
                            "recommendation": bottleneck.recommendation,
                        },
                    ))

                # Get optimization suggestions
                optimizations = self.get_optimizations(bottlenecks)
                if optimizations:
                    # Publish top optimization
                    top_opt = optimizations[0]
                    await self.event_bus.publish(Event(
                        type="orchestrator.optimization_suggested",
                        payload={
                            "type": top_opt.type,
                            "target": top_opt.target,
                            "action": top_opt.action,
                            "priority": top_opt.priority,
                            "estimated_impact": top_opt.estimated_impact,
                        },
                    ))

                    # Auto-apply high-priority optimizations (priority >= 4)
                    if top_opt.priority >= 4:
                        success = self.apply_optimization(top_opt)
                        if success:
                            await self.event_bus.publish(Event(
                                type="orchestrator.optimization_applied",
                                payload={
                                    "type": top_opt.type,
                                    "target": top_opt.target,
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
