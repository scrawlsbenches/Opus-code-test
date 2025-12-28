"""
Agile Practices for Worker-Level Execution

This module implements agile/scrum practices at the worker level:
- Time-boxed sprints
- Velocity tracking
- Estimation and planning
- Retrospectives for learning
- Increment delivery

Workers use these practices to deliver predictable, high-quality work
that feeds into the evolutionary learning system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from .types import (
    Impediment,
    Increment,
    Retrospective,
    SprintTask,
    TaskStatus,
)


# =============================================================================
# SPRINT CONFIGURATION
# =============================================================================


@dataclass
class SprintConfig:
    """Configuration for a sprint."""

    # Timing
    default_timebox: timedelta = field(
        default_factory=lambda: timedelta(minutes=15)
    )
    min_timebox: timedelta = field(
        default_factory=lambda: timedelta(minutes=5)
    )
    max_timebox: timedelta = field(
        default_factory=lambda: timedelta(minutes=60)
    )

    # Velocity
    default_velocity: float = 5.0
    velocity_window: int = 5  # Number of sprints to average

    # Points
    points_per_minute: float = 0.2  # 1 point ≈ 5 minutes
    max_points_per_task: int = 8  # Encourage breaking down large tasks

    # Acceptance
    min_acceptance_criteria: int = 1
    require_verification: bool = True


# =============================================================================
# SPRINT
# =============================================================================


@dataclass
class WorkerSprint:
    """A time-boxed sprint for focused work."""

    sprint_id: str
    goal: str
    timebox: timedelta

    # Tasks
    tasks: list[SprintTask] = field(default_factory=list)

    # Tracking
    estimated_points: int = 0
    completed_points: int = 0

    # Issues
    impediments: list[Impediment] = field(default_factory=list)

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Outcomes
    increment: Increment | None = None
    retrospective: Retrospective | None = None

    @property
    def velocity(self) -> float:
        """Actual velocity achieved."""
        if not self.started_at or not self.completed_at:
            return 0.0
        duration = (self.completed_at - self.started_at).total_seconds() / 60
        if duration == 0:
            return 0.0
        return self.completed_points / duration * 15  # Normalize to 15-min sprint

    @property
    def completion_rate(self) -> float:
        """Percentage of estimated points completed."""
        if self.estimated_points == 0:
            return 0.0
        return self.completed_points / self.estimated_points

    @property
    def is_complete(self) -> bool:
        """Check if sprint is complete."""
        return all(
            t.status in {TaskStatus.COMPLETED, TaskStatus.BLOCKED}
            for t in self.tasks
        )


# =============================================================================
# SPRINT PLANNER
# =============================================================================


class SprintPlanner:
    """Plans sprints based on velocity and capacity."""

    def __init__(self, config: SprintConfig | None = None):
        self.config = config or SprintConfig()

    def plan_sprint(
        self,
        goal: str,
        tasks: list[SprintTask],
        velocity: float,
        timebox: timedelta | None = None,
    ) -> WorkerSprint:
        """Plan a sprint with appropriate capacity."""
        if timebox is None:
            timebox = self.config.default_timebox

        # Calculate capacity in points
        capacity = self._calculate_capacity(timebox, velocity)

        # Select tasks that fit
        selected_tasks = []
        total_points = 0

        for task in self._prioritize_tasks(tasks):
            if total_points + task.estimate_points <= capacity:
                selected_tasks.append(task)
                total_points += task.estimate_points

        return WorkerSprint(
            sprint_id=f"sprint-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            goal=goal,
            timebox=timebox,
            tasks=selected_tasks,
            estimated_points=total_points,
        )

    def _calculate_capacity(
        self,
        timebox: timedelta,
        velocity: float,
    ) -> int:
        """Calculate capacity in points for a timebox."""
        minutes = timebox.total_seconds() / 60
        # Normalize velocity to the timebox
        normalized_velocity = velocity * (minutes / 15)  # Base is 15-min sprint
        return int(normalized_velocity)

    def _prioritize_tasks(
        self,
        tasks: list[SprintTask],
    ) -> list[SprintTask]:
        """Prioritize tasks for sprint selection."""
        # Simple prioritization: smaller tasks first (more likely to complete)
        return sorted(tasks, key=lambda t: t.estimate_points)

    def estimate_task(self, task: SprintTask) -> int:
        """Estimate a task in points."""
        # Placeholder - would use historical data, complexity analysis
        base_estimate = 1

        # Adjust based on description length (proxy for complexity)
        if len(task.description) > 100:
            base_estimate += 1
        if len(task.description) > 200:
            base_estimate += 2

        # Cap at max
        return min(base_estimate, self.config.max_points_per_task)


# =============================================================================
# VELOCITY TRACKER
# =============================================================================


class VelocityTracker:
    """Tracks velocity across sprints."""

    def __init__(self, config: SprintConfig | None = None):
        self.config = config or SprintConfig()
        self.history: list[float] = []

    def record_sprint(self, sprint: WorkerSprint) -> None:
        """Record a completed sprint's velocity."""
        if sprint.velocity > 0:
            self.history.append(sprint.velocity)

    def get_velocity(self) -> float:
        """Get current velocity (rolling average)."""
        if not self.history:
            return self.config.default_velocity

        window = self.history[-self.config.velocity_window:]
        return sum(window) / len(window)

    def get_velocity_trend(self) -> Literal["improving", "stable", "declining"]:
        """Determine velocity trend."""
        if len(self.history) < 3:
            return "stable"

        recent = self.history[-3:]
        older = self.history[-6:-3] if len(self.history) >= 6 else self.history[:-3]

        if not older:
            return "stable"

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if recent_avg > older_avg * 1.1:
            return "improving"
        elif recent_avg < older_avg * 0.9:
            return "declining"
        return "stable"

    def predict_completion(
        self,
        remaining_points: int,
    ) -> timedelta:
        """Predict time to complete remaining points."""
        velocity = self.get_velocity()
        if velocity == 0:
            return timedelta(hours=999)  # Unknown

        sprints_needed = remaining_points / velocity
        # Assume 15-minute sprints
        minutes = sprints_needed * 15
        return timedelta(minutes=minutes)


# =============================================================================
# ESTIMATION
# =============================================================================


@dataclass
class EstimationHistory:
    """Historical estimation data for a task type."""

    task_type: str
    estimates: list[int] = field(default_factory=list)
    actuals: list[int] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Estimation accuracy ratio."""
        if not self.estimates or not self.actuals:
            return 1.0
        total_estimated = sum(self.estimates)
        total_actual = sum(self.actuals)
        if total_estimated == 0:
            return 1.0
        return total_actual / total_estimated

    @property
    def average_estimate(self) -> float:
        """Average estimate for this task type."""
        if not self.estimates:
            return 1.0
        return sum(self.estimates) / len(self.estimates)


class Estimator:
    """Estimates task effort based on historical data."""

    def __init__(self):
        self.history: dict[str, EstimationHistory] = {}

    def estimate(
        self,
        task: SprintTask,
        task_type: str = "default",
    ) -> int:
        """Estimate a task using historical data."""
        history = self.history.get(task_type)

        if not history:
            # No history - use heuristic
            return self._heuristic_estimate(task)

        # Adjust based on historical accuracy
        base = self._heuristic_estimate(task)
        adjusted = int(base * history.accuracy)
        return max(1, adjusted)

    def record(
        self,
        task: SprintTask,
        task_type: str,
        actual_points: int,
    ) -> None:
        """Record actual effort for learning."""
        if task_type not in self.history:
            self.history[task_type] = EstimationHistory(task_type=task_type)

        self.history[task_type].estimates.append(task.estimate_points)
        self.history[task_type].actuals.append(actual_points)

    def _heuristic_estimate(self, task: SprintTask) -> int:
        """Heuristic estimation based on task properties."""
        # Base estimate
        estimate = 1

        # Complexity indicators
        description = task.description.lower()

        if any(w in description for w in ["refactor", "redesign", "migrate"]):
            estimate += 2

        if any(w in description for w in ["simple", "small", "minor"]):
            estimate -= 0  # Stay at base

        if any(w in description for w in ["complex", "large", "major"]):
            estimate += 3

        # Uncertainty indicators
        if any(w in description for w in ["investigate", "explore", "research"]):
            estimate += 1

        return max(1, min(estimate, 8))


# =============================================================================
# RETROSPECTIVE ENGINE
# =============================================================================


class RetrospectiveEngine:
    """Generates retrospectives from sprint data."""

    def generate(self, sprint: WorkerSprint) -> Retrospective:
        """Generate a retrospective from sprint data."""
        went_well = []
        improvements = []
        action_items = []

        # Analyze completion
        if sprint.completion_rate >= 0.9:
            went_well.append("Completed most planned work")
        elif sprint.completion_rate < 0.5:
            improvements.append("Completed less than half of planned work")
            action_items.append("Consider smaller task decomposition")

        # Analyze velocity
        if sprint.velocity > 0:
            went_well.append(f"Achieved velocity of {sprint.velocity:.1f}")

        # Analyze impediments
        if sprint.impediments:
            improvements.append(
                f"Encountered {len(sprint.impediments)} impediment(s)"
            )
            for imp in sprint.impediments:
                action_items.append(f"Address: {imp.description}")
        else:
            went_well.append("No impediments encountered")

        # Analyze estimation
        estimation_accuracy = sprint.completion_rate
        if 0.8 <= estimation_accuracy <= 1.2:
            went_well.append("Estimation was accurate")
        elif estimation_accuracy < 0.8:
            improvements.append("Overestimated capacity")
            action_items.append("Reduce sprint commitment next time")
        else:
            improvements.append("Underestimated capacity")
            action_items.append("Could commit to more work")

        return Retrospective(
            sprint_id=sprint.sprint_id,
            went_well=went_well,
            improvements=improvements,
            action_items=action_items,
            velocity_actual=sprint.completed_points,
            velocity_planned=sprint.estimated_points,
            estimation_accuracy=estimation_accuracy,
            impediment_count=len(sprint.impediments),
        )


# =============================================================================
# INCREMENT BUILDER
# =============================================================================


class IncrementBuilder:
    """Builds deliverable increments from sprint outputs."""

    def build(
        self,
        sprint: WorkerSprint,
        outputs: dict[str, Any],
    ) -> Increment:
        """Build an increment from sprint work."""
        # Calculate acceptance
        completed_tasks = [
            t for t in sprint.tasks
            if t.status == TaskStatus.COMPLETED
        ]

        acceptance_met = len(completed_tasks) > 0 and all(
            self._verify_acceptance(t) for t in completed_tasks
        )

        return Increment(
            sprint_id=sprint.sprint_id,
            goal=sprint.goal,
            outputs=outputs,
            acceptance_met=acceptance_met,
            metrics={
                "planned_points": sprint.estimated_points,
                "completed_points": sprint.completed_points,
                "completion_rate": sprint.completion_rate,
                "velocity": sprint.velocity,
                "tasks_completed": len(completed_tasks),
                "tasks_total": len(sprint.tasks),
            },
        )

    def _verify_acceptance(self, task: SprintTask) -> bool:
        """Verify task meets acceptance criteria."""
        # Placeholder - would verify each criterion
        return task.status == TaskStatus.COMPLETED


# =============================================================================
# SPRINT METRICS
# =============================================================================


@dataclass
class SprintMetrics:
    """Aggregated metrics across sprints."""

    # Velocity
    avg_velocity: float = 0.0
    velocity_trend: Literal["improving", "stable", "declining"] = "stable"

    # Completion
    avg_completion_rate: float = 0.0
    total_points_completed: int = 0
    total_points_planned: int = 0

    # Estimation
    estimation_accuracy: float = 1.0

    # Health
    impediment_rate: float = 0.0  # Impediments per sprint
    sprint_count: int = 0


class SprintMetricsCollector:
    """Collects and aggregates sprint metrics."""

    def __init__(self):
        self.sprints: list[WorkerSprint] = []

    def record(self, sprint: WorkerSprint) -> None:
        """Record a completed sprint."""
        self.sprints.append(sprint)

    def get_metrics(self) -> SprintMetrics:
        """Get aggregated metrics."""
        if not self.sprints:
            return SprintMetrics()

        velocities = [s.velocity for s in self.sprints if s.velocity > 0]
        completion_rates = [s.completion_rate for s in self.sprints]
        impediment_counts = [len(s.impediments) for s in self.sprints]

        total_estimated = sum(s.estimated_points for s in self.sprints)
        total_completed = sum(s.completed_points for s in self.sprints)

        # Velocity trend
        trend: Literal["improving", "stable", "declining"] = "stable"
        if len(velocities) >= 3:
            recent = velocities[-3:]
            older = velocities[:-3] if len(velocities) > 3 else velocities[:1]
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older) if older else recent_avg

            if recent_avg > older_avg * 1.1:
                trend = "improving"
            elif recent_avg < older_avg * 0.9:
                trend = "declining"

        return SprintMetrics(
            avg_velocity=sum(velocities) / len(velocities) if velocities else 0,
            velocity_trend=trend,
            avg_completion_rate=(
                sum(completion_rates) / len(completion_rates)
                if completion_rates else 0
            ),
            total_points_completed=total_completed,
            total_points_planned=total_estimated,
            estimation_accuracy=(
                total_completed / total_estimated if total_estimated > 0 else 1.0
            ),
            impediment_rate=(
                sum(impediment_counts) / len(self.sprints) if self.sprints else 0
            ),
            sprint_count=len(self.sprints),
        )
