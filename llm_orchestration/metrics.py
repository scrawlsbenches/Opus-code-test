"""
Unified Metrics for the LLM Orchestration Framework

This module provides comprehensive metrics combining:
- Kanban flow metrics (throughput, cycle time, WIP)
- Agile sprint metrics (velocity, estimation accuracy)
- Evolution fitness metrics (success, efficiency, quality)

Metrics feed into the evolutionary system for fitness evaluation
and strategy improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

from .agile import SprintMetrics
from .orchestration import FlowMetrics
from .evolution import FitnessScore


# =============================================================================
# HYBRID METRICS
# =============================================================================


@dataclass
class HybridMetrics:
    """Combined kanban + agile + evolution metrics."""

    # Kanban (flow) metrics
    throughput: float = 0.0              # Goals completed per day
    lead_time: timedelta = field(default_factory=timedelta)
    cycle_time: timedelta = field(default_factory=timedelta)
    flow_efficiency: float = 0.0         # Active / (Active + Wait)
    wip_stability: float = 0.0           # How often WIP limits respected

    # Agile (sprint) metrics
    velocity_trend: list[float] = field(default_factory=list)
    velocity_stability: float = 0.0      # Variance in velocity
    estimation_accuracy: float = 0.0     # Actual vs estimated
    sprint_completion_rate: float = 0.0  # % of sprint goals met
    impediment_resolution_time: timedelta = field(default_factory=timedelta)

    # Combined health
    predictability: float = 0.0          # Can we forecast delivery?
    responsiveness: float = 0.0          # Time to start new work
    quality: float = 0.0                 # Defect rate, rework rate

    def to_fitness_score(self) -> FitnessScore:
        """Convert to fitness score for evolution."""
        return FitnessScore(
            success=self.sprint_completion_rate,
            efficiency=self.flow_efficiency,
            quality=self.quality,
            stability=self.wip_stability,
            elegance=self.predictability,
            user_satisfaction=0.5,  # Would come from feedback
        )


# =============================================================================
# METRICS COLLECTOR
# =============================================================================


@dataclass
class MetricDataPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates metrics from all layers."""

    def __init__(self):
        self.data_points: list[MetricDataPoint] = []
        self.flow_snapshots: list[FlowMetrics] = []
        self.sprint_snapshots: list[SprintMetrics] = []

        # Aggregates
        self._goal_times: list[tuple[datetime, datetime]] = []
        self._wip_violations: int = 0
        self._total_goals: int = 0

    def record(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a metric data point."""
        self.data_points.append(MetricDataPoint(
            name=name,
            value=value,
            labels=labels or {},
        ))

    def record_flow_snapshot(self, metrics: FlowMetrics) -> None:
        """Record a flow metrics snapshot."""
        self.flow_snapshots.append(metrics)

        # Track WIP violations
        self._wip_violations += len(metrics.wip_violations)

    def record_sprint_snapshot(self, metrics: SprintMetrics) -> None:
        """Record a sprint metrics snapshot."""
        self.sprint_snapshots.append(metrics)

    def record_goal_completion(
        self,
        started_at: datetime,
        completed_at: datetime,
    ) -> None:
        """Record a goal completion for cycle time."""
        self._goal_times.append((started_at, completed_at))
        self._total_goals += 1

    def get_hybrid_metrics(self) -> HybridMetrics:
        """Compute hybrid metrics from collected data."""
        metrics = HybridMetrics()

        # Flow metrics
        if self._goal_times:
            # Throughput
            if len(self._goal_times) >= 2:
                first = min(t[0] for t in self._goal_times)
                last = max(t[1] for t in self._goal_times)
                days = (last - first).total_seconds() / 86400
                if days > 0:
                    metrics.throughput = len(self._goal_times) / days

            # Cycle time
            cycle_times = [
                (t[1] - t[0]).total_seconds()
                for t in self._goal_times
            ]
            avg_cycle = sum(cycle_times) / len(cycle_times)
            metrics.cycle_time = timedelta(seconds=avg_cycle)

        # WIP stability
        if self._total_goals > 0:
            metrics.wip_stability = 1.0 - (
                self._wip_violations / self._total_goals
            )

        # Sprint metrics
        if self.sprint_snapshots:
            latest = self.sprint_snapshots[-1]
            metrics.velocity_trend = [
                s.avg_velocity for s in self.sprint_snapshots
            ]
            metrics.estimation_accuracy = latest.estimation_accuracy
            metrics.sprint_completion_rate = latest.avg_completion_rate

            # Velocity stability (lower variance = higher stability)
            if len(metrics.velocity_trend) > 1:
                mean = sum(metrics.velocity_trend) / len(metrics.velocity_trend)
                variance = sum(
                    (v - mean) ** 2 for v in metrics.velocity_trend
                ) / len(metrics.velocity_trend)
                metrics.velocity_stability = 1.0 / (1.0 + variance)

        # Derived metrics
        if metrics.cycle_time.total_seconds() > 0:
            # Responsiveness: inverse of cycle time (normalized)
            max_acceptable = timedelta(hours=24).total_seconds()
            metrics.responsiveness = 1.0 - min(
                metrics.cycle_time.total_seconds() / max_acceptable,
                1.0
            )

        # Predictability: combination of estimation accuracy and velocity stability
        metrics.predictability = (
            metrics.estimation_accuracy * 0.5 +
            metrics.velocity_stability * 0.5
        )

        return metrics

    def get_time_series(
        self,
        metric_name: str,
        since: datetime | None = None,
    ) -> list[tuple[datetime, float]]:
        """Get time series for a metric."""
        points = [
            (p.timestamp, p.value)
            for p in self.data_points
            if p.name == metric_name
        ]

        if since:
            points = [(t, v) for t, v in points if t >= since]

        return sorted(points, key=lambda x: x[0])

    def get_summary(self) -> dict[str, Any]:
        """Get summary of collected metrics."""
        hybrid = self.get_hybrid_metrics()

        return {
            "total_data_points": len(self.data_points),
            "flow_snapshots": len(self.flow_snapshots),
            "sprint_snapshots": len(self.sprint_snapshots),
            "goals_completed": len(self._goal_times),
            "throughput": hybrid.throughput,
            "avg_cycle_time_minutes": (
                hybrid.cycle_time.total_seconds() / 60
            ),
            "sprint_completion_rate": hybrid.sprint_completion_rate,
            "predictability": hybrid.predictability,
        }


# =============================================================================
# EVOLUTION TARGET IDENTIFICATION
# =============================================================================


@dataclass
class EvolutionTarget:
    """A target for evolutionary improvement."""

    gene: str
    reason: str
    direction: str
    priority: Literal["high", "medium", "low"] = "medium"


class EvolutionTargetIdentifier:
    """Identifies what evolution should focus on."""

    def identify(self, metrics: HybridMetrics) -> list[EvolutionTarget]:
        """Identify evolution targets from metrics."""
        targets = []

        # Low flow efficiency → evolve coordination
        if metrics.flow_efficiency < 0.5:
            targets.append(EvolutionTarget(
                gene="coordination_protocols",
                reason="Low flow efficiency indicates waiting/blocking",
                direction="reduce_handoff_overhead",
                priority="high",
            ))

        # Poor estimation → evolve decomposition
        if metrics.estimation_accuracy < 0.7:
            targets.append(EvolutionTarget(
                gene="decomposition_patterns",
                reason="Estimation accuracy low",
                direction="smaller_more_predictable_tasks",
                priority="medium",
            ))

        # High impediment rate → evolve failure strategies
        if metrics.impediment_resolution_time > timedelta(minutes=10):
            targets.append(EvolutionTarget(
                gene="failure_strategies",
                reason="Impediments taking too long to resolve",
                direction="faster_escalation_or_swarming",
                priority="high",
            ))

        # Low WIP stability → evolve delegation
        if metrics.wip_stability < 0.8:
            targets.append(EvolutionTarget(
                gene="delegation_strategies",
                reason="WIP limits frequently violated",
                direction="more_conservative_parallelism",
                priority="medium",
            ))

        # Low predictability → evolve multiple areas
        if metrics.predictability < 0.6:
            targets.append(EvolutionTarget(
                gene="context_compression_methods",
                reason="Low predictability, possibly context loss",
                direction="preserve_more_decision_context",
                priority="low",
            ))

        return sorted(targets, key=lambda t: {
            "high": 0, "medium": 1, "low": 2
        }[t.priority])


# =============================================================================
# FITNESS INTEGRATION
# =============================================================================


class FitnessCalculator:
    """Calculates fitness scores from hybrid metrics."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {
            # Flow health
            "throughput": 0.15,
            "flow_efficiency": 0.10,
            "wip_discipline": 0.10,

            # Sprint health
            "velocity_stability": 0.15,
            "estimation_accuracy": 0.10,
            "completion_rate": 0.15,

            # Combined
            "predictability": 0.15,
            "responsiveness": 0.10,
        }

    def calculate(self, metrics: HybridMetrics) -> FitnessScore:
        """Calculate fitness score from hybrid metrics."""
        # Normalize throughput (assume 5 goals/day is excellent)
        throughput_score = min(metrics.throughput / 5.0, 1.0)

        # Compute weighted scores
        scores = {
            "throughput": throughput_score,
            "flow_efficiency": metrics.flow_efficiency,
            "wip_discipline": metrics.wip_stability,
            "velocity_stability": metrics.velocity_stability,
            "estimation_accuracy": metrics.estimation_accuracy,
            "completion_rate": metrics.sprint_completion_rate,
            "predictability": metrics.predictability,
            "responsiveness": metrics.responsiveness,
        }

        # Weighted average for efficiency
        weighted_sum = sum(
            scores.get(k, 0) * v
            for k, v in self.weights.items()
        )

        return FitnessScore(
            success=metrics.sprint_completion_rate,
            efficiency=weighted_sum,
            quality=metrics.quality,
            stability=metrics.wip_stability,
            elegance=metrics.predictability,
            user_satisfaction=0.5,  # Placeholder
        )


# =============================================================================
# DASHBOARD
# =============================================================================


class MetricsDashboard:
    """Generates dashboard views of metrics."""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def render_text(self) -> str:
        """Render a text dashboard."""
        metrics = self.collector.get_hybrid_metrics()
        summary = self.collector.get_summary()

        lines = [
            "┌" + "─" * 60 + "┐",
            "│" + " LLM ORCHESTRATION METRICS DASHBOARD ".center(60) + "│",
            "├" + "─" * 60 + "┤",
            "│" + " FLOW METRICS ".ljust(60) + "│",
            "├" + "─" * 60 + "┤",
            f"│  Throughput:        {metrics.throughput:.2f} goals/day".ljust(61) + "│",
            f"│  Cycle Time:        {metrics.cycle_time.total_seconds() / 60:.1f} minutes".ljust(61) + "│",
            f"│  Flow Efficiency:   {metrics.flow_efficiency:.1%}".ljust(61) + "│",
            f"│  WIP Stability:     {metrics.wip_stability:.1%}".ljust(61) + "│",
            "├" + "─" * 60 + "┤",
            "│" + " SPRINT METRICS ".ljust(60) + "│",
            "├" + "─" * 60 + "┤",
            f"│  Velocity Trend:    {self._trend_indicator(metrics.velocity_trend)}".ljust(61) + "│",
            f"│  Estimation Acc:    {metrics.estimation_accuracy:.1%}".ljust(61) + "│",
            f"│  Completion Rate:   {metrics.sprint_completion_rate:.1%}".ljust(61) + "│",
            "├" + "─" * 60 + "┤",
            "│" + " HEALTH INDICATORS ".ljust(60) + "│",
            "├" + "─" * 60 + "┤",
            f"│  Predictability:    {self._health_bar(metrics.predictability)}".ljust(61) + "│",
            f"│  Responsiveness:    {self._health_bar(metrics.responsiveness)}".ljust(61) + "│",
            f"│  Quality:           {self._health_bar(metrics.quality)}".ljust(61) + "│",
            "├" + "─" * 60 + "┤",
            "│" + " SUMMARY ".ljust(60) + "│",
            "├" + "─" * 60 + "┤",
            f"│  Goals Completed:   {summary['goals_completed']}".ljust(61) + "│",
            f"│  Data Points:       {summary['total_data_points']}".ljust(61) + "│",
            "└" + "─" * 60 + "┘",
        ]

        return "\n".join(lines)

    def _trend_indicator(self, trend: list[float]) -> str:
        """Generate trend indicator."""
        if len(trend) < 2:
            return "─ (insufficient data)"

        recent = trend[-3:] if len(trend) >= 3 else trend
        avg = sum(recent) / len(recent)

        older = trend[:-3] if len(trend) > 3 else trend[:1]
        old_avg = sum(older) / len(older) if older else avg

        if avg > old_avg * 1.1:
            return f"↑ {avg:.1f} (improving)"
        elif avg < old_avg * 0.9:
            return f"↓ {avg:.1f} (declining)"
        return f"→ {avg:.1f} (stable)"

    def _health_bar(self, value: float) -> str:
        """Generate a health bar."""
        filled = int(value * 10)
        empty = 10 - filled

        bar = "█" * filled + "░" * empty

        if value >= 0.8:
            status = "✓"
        elif value >= 0.5:
            status = "⚠"
        else:
            status = "✗"

        return f"[{bar}] {value:.0%} {status}"

    def get_evolution_recommendations(self) -> list[str]:
        """Get evolution recommendations based on metrics."""
        metrics = self.collector.get_hybrid_metrics()
        identifier = EvolutionTargetIdentifier()
        targets = identifier.identify(metrics)

        recommendations = []
        for target in targets[:3]:  # Top 3
            recommendations.append(
                f"[{target.priority.upper()}] {target.gene}: {target.reason} → {target.direction}"
            )

        return recommendations
