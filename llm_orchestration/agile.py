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

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple

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
    """Tracks velocity across sprints with basic averaging and trend detection."""

    def __init__(self, config: SprintConfig | None = None):
        self.config = config or SprintConfig()
        self.history: list[float] = []
        # Optional: Use advanced predictor for better predictions
        self._predictor: VelocityPredictor | None = None

    def enable_advanced_prediction(self) -> None:
        """Enable advanced prediction using VelocityPredictor."""
        self._predictor = VelocityPredictor(
            history_window=self.config.velocity_window
        )
        # Migrate existing history
        for velocity in self.history:
            self._predictor.record_velocity(velocity)

    def record_sprint(
        self,
        sprint: WorkerSprint,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a completed sprint's velocity.

        Args:
            sprint: Completed sprint
            context: Optional sprint context for advanced prediction
        """
        if sprint.velocity > 0:
            self.history.append(sprint.velocity)
            if self._predictor:
                # Build context from sprint if not provided
                if context is None:
                    context = {
                        "impediments": len(sprint.impediments),
                        "completion_rate": sprint.completion_rate,
                    }
                self._predictor.record_velocity(sprint.velocity, context)

    def get_velocity(self) -> float:
        """Get current velocity (rolling average)."""
        if self._predictor:
            # Use advanced prediction if enabled
            prediction = self._predictor.predict_next()
            return prediction.predicted_velocity

        if not self.history:
            return self.config.default_velocity

        window = self.history[-self.config.velocity_window:]
        return sum(window) / len(window)

    def get_velocity_prediction(self) -> VelocityPrediction | None:
        """
        Get advanced velocity prediction with confidence metrics.

        Returns:
            VelocityPrediction if advanced prediction is enabled, None otherwise
        """
        if not self._predictor:
            return None
        return self._predictor.predict_next()

    def get_velocity_trend(self) -> Literal["improving", "stable", "declining"]:
        """Determine velocity trend."""
        if self._predictor:
            # Use advanced trend detection if enabled
            trend = self._predictor.get_trend()
            # Map to expected return types
            if trend == "increasing":
                return "improving"
            elif trend == "decreasing":
                return "declining"
            return "stable"

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

    def get_anomalies(self) -> List[VelocityAnomaly]:
        """
        Get detected velocity anomalies.

        Returns:
            List of anomalies if advanced prediction is enabled, empty list otherwise
        """
        if not self._predictor:
            return []
        return self._predictor.detect_anomalies()


# =============================================================================
# VELOCITY PREDICTION WITH HISTORICAL ANALYSIS
# =============================================================================


@dataclass
class VelocityPrediction:
    """Prediction for next sprint velocity with confidence metrics."""

    predicted_velocity: float
    confidence_interval: Tuple[float, float]  # (low, high)
    confidence: float  # 0-1 scale
    trend: str  # "increasing", "stable", "decreasing"
    factors: List[str]  # What influenced the prediction


@dataclass
class VelocityAnomaly:
    """Detected anomaly in velocity data."""

    sprint_index: int
    velocity: float
    expected_velocity: float
    deviation: float  # Standard deviations from expected
    potential_causes: List[str]


class VelocityPredictor:
    """
    Advanced velocity prediction using historical data analysis.

    Features:
    - Exponential moving average for predictions
    - Linear regression for trend detection
    - Confidence intervals based on variance
    - Anomaly detection for unusual velocities
    - Pattern recognition for seasonal/cyclical changes
    """

    def __init__(self, history_window: int = 10):
        """
        Initialize predictor.

        Args:
            history_window: Number of sprints to consider for predictions
        """
        self._history: List[float] = []
        self._contexts: List[Dict[str, Any]] = []  # Sprint context metadata
        self._window = history_window
        self._alpha = 0.3  # EMA smoothing factor (0.3 = 30% weight to new data)

    def record_velocity(
        self,
        velocity: float,
        sprint_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record actual velocity from completed sprint.

        Args:
            velocity: Completed sprint velocity
            sprint_context: Optional metadata (team_size, complexity, etc.)
        """
        if velocity < 0:
            raise ValueError("Velocity cannot be negative")

        self._history.append(velocity)
        self._contexts.append(sprint_context or {})

        # Keep only the window size
        if len(self._history) > self._window * 2:  # Keep 2x for better trends
            self._history = self._history[-self._window * 2:]
            self._contexts = self._contexts[-self._window * 2:]

    def predict_next(self) -> VelocityPrediction:
        """
        Predict velocity for next sprint with confidence metrics.

        Returns:
            VelocityPrediction with predicted value, confidence interval, and factors
        """
        if not self._history:
            # No history - return default with low confidence
            return VelocityPrediction(
                predicted_velocity=5.0,
                confidence_interval=(3.0, 7.0),
                confidence=0.0,
                trend="stable",
                factors=["No historical data - using default"],
            )

        if len(self._history) == 1:
            # Only one data point
            v = self._history[0]
            return VelocityPrediction(
                predicted_velocity=v,
                confidence_interval=(v * 0.7, v * 1.3),
                confidence=0.3,
                trend="stable",
                factors=["Only one sprint completed"],
            )

        # Calculate prediction using exponential moving average
        predicted = self._exponential_moving_average()

        # Calculate confidence interval based on variance
        std_dev = self._standard_deviation()
        confidence_interval = (
            max(0, predicted - 1.96 * std_dev),  # 95% confidence
            predicted + 1.96 * std_dev,
        )

        # Calculate confidence (inverse of coefficient of variation)
        mean = self._simple_moving_average()
        cv = std_dev / mean if mean > 0 else 1.0
        confidence = max(0.0, min(1.0, 1.0 - cv))

        # Detect trend
        trend = self.get_trend()

        # Analyze factors
        factors = self._analyze_prediction_factors()

        return VelocityPrediction(
            predicted_velocity=predicted,
            confidence_interval=confidence_interval,
            confidence=confidence,
            trend=trend,
            factors=factors,
        )

    def get_trend(self) -> str:
        """
        Analyze velocity trend using linear regression.

        Returns:
            "increasing", "stable", or "decreasing"
        """
        if len(self._history) < 3:
            return "stable"

        # Use recent history for trend
        recent = self._history[-min(len(self._history), self._window):]

        # Linear regression: y = mx + b
        slope, _ = self._linear_regression(recent)

        # Determine trend based on slope
        # Slope threshold is relative to mean velocity
        mean_velocity = sum(recent) / len(recent)
        threshold = mean_velocity * 0.1  # 10% change is significant

        if slope > threshold / len(recent):
            return "increasing"
        elif slope < -threshold / len(recent):
            return "decreasing"
        else:
            return "stable"

    def detect_anomalies(self) -> List[VelocityAnomaly]:
        """
        Detect unusual velocity values using statistical analysis.

        Returns:
            List of detected anomalies with potential causes
        """
        if len(self._history) < 5:  # Need enough data
            return []

        anomalies = []
        mean = self._simple_moving_average()
        std_dev = self._standard_deviation()

        if std_dev == 0:  # No variance
            return []

        # Check each velocity for anomalies
        for i, velocity in enumerate(self._history):
            deviation = abs(velocity - mean) / std_dev

            if deviation > 2.0:  # More than 2 standard deviations
                # Identify potential causes
                causes = self._identify_anomaly_causes(i, velocity, mean)

                anomalies.append(VelocityAnomaly(
                    sprint_index=i,
                    velocity=velocity,
                    expected_velocity=mean,
                    deviation=deviation,
                    potential_causes=causes,
                ))

        return anomalies

    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistical summary of velocity history.

        Returns:
            Dictionary with mean, median, std_dev, min, max, trend_slope
        """
        if not self._history:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "trend_slope": 0.0,
                "count": 0,
            }

        sorted_history = sorted(self._history)
        n = len(sorted_history)
        median = (
            sorted_history[n // 2]
            if n % 2 == 1
            else (sorted_history[n // 2 - 1] + sorted_history[n // 2]) / 2
        )

        slope, _ = self._linear_regression(self._history)

        return {
            "mean": self._simple_moving_average(),
            "median": median,
            "std_dev": self._standard_deviation(),
            "min": min(self._history),
            "max": max(self._history),
            "trend_slope": slope,
            "count": len(self._history),
        }

    # =========================================================================
    # STATISTICAL METHODS
    # =========================================================================

    def _simple_moving_average(self) -> float:
        """Calculate simple moving average of recent history."""
        if not self._history:
            return 0.0

        window = self._history[-self._window:]
        return sum(window) / len(window)

    def _exponential_moving_average(self) -> float:
        """
        Calculate exponential moving average.

        Gives more weight to recent data while considering all history.
        """
        if not self._history:
            return 0.0

        ema = self._history[0]
        for velocity in self._history[1:]:
            ema = self._alpha * velocity + (1 - self._alpha) * ema

        return ema

    def _standard_deviation(self) -> float:
        """Calculate standard deviation of recent history."""
        if len(self._history) < 2:
            return 0.0

        window = self._history[-self._window:]
        mean = sum(window) / len(window)
        variance = sum((v - mean) ** 2 for v in window) / len(window)
        return math.sqrt(variance)

    def _linear_regression(
        self,
        data: List[float],
    ) -> Tuple[float, float]:
        """
        Calculate linear regression: y = mx + b

        Args:
            data: List of velocity values

        Returns:
            (slope, intercept)
        """
        if len(data) < 2:
            return (0.0, data[0] if data else 0.0)

        n = len(data)
        x = list(range(n))  # Time indices
        y = data

        # Calculate means
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        # Calculate slope
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0.0
        intercept = y_mean - slope * x_mean

        return (slope, intercept)

    # =========================================================================
    # ANALYSIS HELPERS
    # =========================================================================

    def _analyze_prediction_factors(self) -> List[str]:
        """Analyze what factors are influencing the prediction."""
        factors = []

        # Data quality
        if len(self._history) < 5:
            factors.append("Limited historical data")
        elif len(self._history) >= self._window:
            factors.append("Sufficient historical data")

        # Variance
        std_dev = self._standard_deviation()
        mean = self._simple_moving_average()
        if mean > 0:
            cv = std_dev / mean
            if cv > 0.3:
                factors.append("High variance in velocity")
            elif cv < 0.1:
                factors.append("Consistent velocity")

        # Trend
        trend = self.get_trend()
        if trend != "stable":
            factors.append(f"Velocity is {trend}")

        # Recent performance
        if len(self._history) >= 2:
            recent_change = (
                (self._history[-1] - self._history[-2]) / self._history[-2]
                if self._history[-2] != 0 else 0
            )
            if abs(recent_change) > 0.2:
                direction = "increased" if recent_change > 0 else "decreased"
                factors.append(f"Recent velocity {direction} significantly")

        # Anomalies
        anomalies = self.detect_anomalies()
        if anomalies:
            factors.append(f"{len(anomalies)} anomalies detected in history")

        return factors

    def _identify_anomaly_causes(
        self,
        index: int,
        velocity: float,
        expected: float,
    ) -> List[str]:
        """Identify potential causes of velocity anomaly."""
        causes = []

        # Check if unusually high or low
        if velocity > expected:
            causes.append("Velocity unusually high")

            # Check context if available
            if index < len(self._contexts):
                context = self._contexts[index]
                if context.get("team_size", 1) > 1:
                    causes.append("Possible team size increase")
                if context.get("complexity") == "low":
                    causes.append("Lower complexity tasks")
        else:
            causes.append("Velocity unusually low")

            # Check context if available
            if index < len(self._contexts):
                context = self._contexts[index]
                if context.get("impediments", 0) > 0:
                    causes.append("Impediments encountered")
                if context.get("complexity") == "high":
                    causes.append("Higher complexity tasks")
                if context.get("team_changes", False):
                    causes.append("Team composition changed")

        return causes


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
