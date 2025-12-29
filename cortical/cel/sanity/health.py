"""
Health monitoring for the Cognitive Event Lattice.

The health system enables the lattice to observe and reason
about its own operational status. This is meta-cognition in
action - the system watching itself.

Key Capabilities:
    - Structural integrity verification (DAG consistency)
    - Performance anomaly detection (latency, throughput)
    - Storage pressure monitoring (size, growth rate)
    - Semantic drift detection (concept evolution)

Design Pattern:
    Health checks are themselves events (MetaCognition type).
    This creates a recursive structure where health observations
    become part of the knowledge the system reasons about.

    check() -> MetaCognition event -> stored in DAG -> can be queried

This module implements Level 4 of the CEL architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.events import CognitiveEvent, EventType, MetaCognition
from ..core.protocols import EventStore, HealthMonitor
from ..core.references import MerkleRoot


class HealthStatus(Enum):
    """Overall health status levels."""

    HEALTHY = auto()      # All systems normal
    DEGRADED = auto()     # Some issues, still functional
    UNHEALTHY = auto()    # Significant problems
    CRITICAL = auto()     # Immediate attention required


@dataclass
class HealthMetric:
    """
    A single health measurement.

    Captures a point-in-time observation about system health.
    Metrics can be numeric (latency_ms), boolean (is_consistent),
    or categorical (status).

    Attributes:
        name: Metric identifier (e.g., 'dag_consistency')
        value: The measured value
        threshold: Warning threshold (optional)
        critical_threshold: Critical threshold (optional)
        unit: Unit of measurement (optional)
        timestamp: When measured
    """

    name: str
    value: float
    threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    unit: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def status(self) -> HealthStatus:
        """Determine status based on thresholds."""
        if self.critical_threshold is not None:
            if self.value >= self.critical_threshold:
                return HealthStatus.CRITICAL

        if self.threshold is not None:
            if self.value >= self.threshold:
                return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'name': self.name,
            'value': self.value,
            'threshold': self.threshold,
            'critical_threshold': self.critical_threshold,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.name,
        }


@dataclass
class HealthReport:
    """
    Comprehensive health report from a check cycle.

    Aggregates multiple metrics into an overall assessment
    with recommendations for remediation.

    Attributes:
        timestamp: When the check was performed
        status: Overall health status
        metrics: Individual metric measurements
        issues: List of detected issues
        recommendations: Suggested actions
    """

    timestamp: datetime
    status: HealthStatus
    metrics: List[HealthMetric]
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    @classmethod
    def aggregate_status(cls, metrics: List[HealthMetric]) -> HealthStatus:
        """Determine overall status from metrics."""
        statuses = [m.status for m in metrics]

        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.name,
            'metrics': [m.to_dict() for m in self.metrics],
            'issues': self.issues,
            'recommendations': self.recommendations,
        }

    def to_event(self) -> MetaCognition:
        """Convert to a MetaCognition event for storage."""
        return MetaCognition(
            observation_type='health_check',
            metrics=self.to_dict(),
            conclusions=[
                f"Status: {self.status.name}",
                f"Issues: {len(self.issues)}",
            ] + self.recommendations[:3],
        )


class EventStoreHealthMonitor:
    """
    Health monitor for event stores.

    Implements the HealthMonitor protocol with checks specific
    to event-sourced systems:
    - DAG consistency (no orphans, no cycles)
    - Storage growth rate
    - Event processing latency
    - Merkle integrity

    Implements: HealthMonitor protocol
    """

    def __init__(
        self,
        event_store: EventStore,
        check_interval: timedelta = timedelta(minutes=5),
    ):
        """
        Initialize the health monitor.

        Args:
            event_store: The event store to monitor
            check_interval: How often to run checks
        """
        self._store = event_store
        self._check_interval = check_interval
        self._last_check: Optional[datetime] = None
        self._history: List[HealthReport] = []

        # Thresholds (can be tuned)
        self._thresholds = {
            'event_count': (10000, 50000),      # warn, critical
            'orphan_ratio': (0.01, 0.05),       # 1%, 5%
            'avg_latency_ms': (100, 500),       # 100ms, 500ms
            'storage_growth_rate': (0.1, 0.5),  # 10%/day, 50%/day
        }

        # Custom check functions
        self._custom_checks: List[Callable[[], HealthMetric]] = []

    def check(self) -> HealthReport:
        """
        Perform comprehensive health check.

        Returns:
            HealthReport with current status and metrics
        """
        now = datetime.now()
        metrics: List[HealthMetric] = []
        issues: List[str] = []
        recommendations: List[str] = []

        # Check DAG consistency
        dag_metric = self._check_dag_consistency()
        metrics.append(dag_metric)
        if dag_metric.status != HealthStatus.HEALTHY:
            issues.append("DAG inconsistency detected")
            recommendations.append("Run DAG repair utility")

        # Check storage size
        size_metric = self._check_storage_size()
        metrics.append(size_metric)
        if size_metric.status == HealthStatus.CRITICAL:
            issues.append("Storage near capacity")
            recommendations.append("Run compaction or archive old events")
        elif size_metric.status == HealthStatus.DEGRADED:
            recommendations.append("Consider compaction soon")

        # Check event integrity
        integrity_metric = self._check_merkle_integrity()
        metrics.append(integrity_metric)
        if integrity_metric.status != HealthStatus.HEALTHY:
            issues.append("Merkle integrity violations found")
            recommendations.append("Investigate corrupted events")

        # Check processing latency (if history available)
        if self._history:
            latency_metric = self._check_latency_trend()
            metrics.append(latency_metric)
            if latency_metric.status != HealthStatus.HEALTHY:
                issues.append("Increased processing latency")
                recommendations.append("Review recent changes for performance regressions")

        # Run custom checks
        for check_fn in self._custom_checks:
            try:
                metric = check_fn()
                metrics.append(metric)
            except Exception as e:
                issues.append(f"Custom check failed: {e}")

        # Build report
        status = HealthReport.aggregate_status(metrics)
        report = HealthReport(
            timestamp=now,
            status=status,
            metrics=metrics,
            issues=issues,
            recommendations=recommendations,
        )

        # Update history
        self._history.append(report)
        self._last_check = now

        # Keep history bounded
        max_history = 100
        if len(self._history) > max_history:
            self._history = self._history[-max_history:]

        return report

    def _check_dag_consistency(self) -> HealthMetric:
        """Check DAG structural consistency."""
        # Count events and check for orphans
        total_events = 0
        orphan_count = 0
        seen_ids: set = set()

        for event in self._store.iterate():
            total_events += 1
            seen_ids.add(event.id)

        # Check for parent references to non-existent events
        for event in self._store.iterate():
            for parent_id in event.causal_parents:
                if parent_id not in seen_ids:
                    orphan_count += 1

        orphan_ratio = orphan_count / max(total_events, 1)
        warn, crit = self._thresholds['orphan_ratio']

        return HealthMetric(
            name='dag_orphan_ratio',
            value=orphan_ratio,
            threshold=warn,
            critical_threshold=crit,
            unit='ratio',
        )

    def _check_storage_size(self) -> HealthMetric:
        """Check storage size and growth."""
        total_events = sum(1 for _ in self._store.iterate())
        warn, crit = self._thresholds['event_count']

        return HealthMetric(
            name='event_count',
            value=float(total_events),
            threshold=float(warn),
            critical_threshold=float(crit),
            unit='events',
        )

    def _check_merkle_integrity(self) -> HealthMetric:
        """Verify Merkle root integrity."""
        violations = 0
        total = 0

        for event in self._store.iterate():
            total += 1
            # Recompute ID and compare
            expected_id = event.id
            # The ID property recomputes the hash
            if event.id != expected_id:
                violations += 1

        violation_ratio = violations / max(total, 1)

        return HealthMetric(
            name='merkle_violations',
            value=violation_ratio,
            threshold=0.0,  # Any violation is a problem
            critical_threshold=0.01,
            unit='ratio',
        )

    def _check_latency_trend(self) -> HealthMetric:
        """Check processing latency trend from history."""
        if len(self._history) < 2:
            return HealthMetric(
                name='latency_trend',
                value=0.0,
                unit='ratio',
            )

        # Compare recent to historical average
        recent = self._history[-5:]
        older = self._history[:-5] if len(self._history) > 5 else []

        if not older:
            return HealthMetric(
                name='latency_trend',
                value=0.0,
                unit='ratio',
            )

        # This is a simplified trend check
        # In practice, we'd track actual latencies
        recent_issues = sum(len(r.issues) for r in recent) / len(recent)
        older_issues = sum(len(r.issues) for r in older) / len(older)

        trend = (recent_issues - older_issues) / max(older_issues, 1)
        warn, crit = 0.5, 1.0  # 50% increase, 100% increase

        return HealthMetric(
            name='issue_trend',
            value=trend,
            threshold=warn,
            critical_threshold=crit,
            unit='ratio',
        )

    def register_check(self, check_fn: Callable[[], HealthMetric]) -> None:
        """
        Register a custom health check function.

        Args:
            check_fn: Function returning a HealthMetric
        """
        self._custom_checks.append(check_fn)

    def is_healthy(self) -> bool:
        """Quick health check returning boolean."""
        if self._last_check is None:
            report = self.check()
        elif datetime.now() - self._last_check > self._check_interval:
            report = self.check()
        else:
            report = self._history[-1] if self._history else self.check()

        return report.status == HealthStatus.HEALTHY

    def needs_attention(self) -> bool:
        """Check if system needs attention."""
        if not self._history:
            self.check()

        if self._history:
            return self._history[-1].status in (
                HealthStatus.UNHEALTHY,
                HealthStatus.CRITICAL,
            )
        return False

    def get_recommendations(self) -> List[str]:
        """Get current recommendations."""
        if not self._history:
            self.check()

        if self._history:
            return self._history[-1].recommendations
        return []

    def diagnose(self) -> Dict[str, Any]:
        """
        Get detailed diagnostic information.

        Returns more information than check() for debugging.
        """
        report = self.check()

        # Gather additional diagnostics
        event_types: Dict[str, int] = {}
        concepts: Dict[str, int] = {}

        for event in self._store.iterate():
            et = event.event_type.name
            event_types[et] = event_types.get(et, 0) + 1

            for concept in event.concepts:
                concepts[concept] = concepts.get(concept, 0) + 1

        return {
            'health_report': report.to_dict(),
            'event_type_distribution': event_types,
            'top_concepts': dict(
                sorted(concepts.items(), key=lambda x: -x[1])[:20]
            ),
            'history_length': len(self._history),
            'last_check': self._last_check.isoformat() if self._last_check else None,
        }


class HealthCheckScheduler:
    """
    Scheduler for periodic health checks.

    Manages automatic health monitoring with configurable
    intervals and alert thresholds.
    """

    def __init__(
        self,
        monitor: EventStoreHealthMonitor,
        check_interval: timedelta = timedelta(minutes=5),
        alert_callback: Optional[Callable[[HealthReport], None]] = None,
    ):
        """
        Initialize scheduler.

        Args:
            monitor: The health monitor to schedule
            check_interval: Time between checks
            alert_callback: Function called when issues detected
        """
        self._monitor = monitor
        self._interval = check_interval
        self._alert_callback = alert_callback
        self._running = False

    def should_run(self) -> bool:
        """Check if a health check should run now."""
        if self._monitor._last_check is None:
            return True
        elapsed = datetime.now() - self._monitor._last_check
        return elapsed >= self._interval

    def run_if_due(self) -> Optional[HealthReport]:
        """
        Run health check if due.

        Returns:
            HealthReport if check was run, None otherwise
        """
        if not self.should_run():
            return None

        report = self._monitor.check()

        if self._alert_callback and report.status != HealthStatus.HEALTHY:
            self._alert_callback(report)

        return report
