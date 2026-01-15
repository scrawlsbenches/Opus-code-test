"""
Contract Materializer - Query contract state from CEL events.

This module provides high-level APIs for querying contract state:
- Current statistics (violations, check counts)
- Temporal queries (violations in time range)
- Contract health reports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Import CEL types gracefully
try:
    from cortical.cel.core.events import CognitiveEvent, EventType
    from cortical.cel.core.protocols import EventStore
    from cortical.cel.wisdom.materializer import CachingMaterializer, EntityReducerRegistry
    CEL_AVAILABLE = True
except ImportError:
    CEL_AVAILABLE = False


@dataclass
class ContractState:
    """
    Materialized contract state.

    This represents the current state of all contracts,
    derived from CEL events.
    """

    total_checks: int = 0
    total_violations: int = 0
    violations_by_method: Dict[str, int] = field(default_factory=dict)
    violations_by_type: Dict[str, int] = field(default_factory=dict)
    last_check: Optional[datetime] = None
    last_violation: Optional[datetime] = None
    first_check: Optional[datetime] = None

    @property
    def violation_rate(self) -> float:
        """Calculate violation rate as percentage."""
        if self.total_checks == 0:
            return 0.0
        return (self.total_violations / self.total_checks) * 100

    @property
    def is_healthy(self) -> bool:
        """Contract system is healthy if violation rate < 1%."""
        return self.violation_rate < 1.0

    @property
    def most_violated_method(self) -> Optional[str]:
        """Get the method with most violations."""
        if not self.violations_by_method:
            return None
        return max(self.violations_by_method, key=self.violations_by_method.get)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'total_checks': self.total_checks,
            'total_violations': self.total_violations,
            'violation_rate': self.violation_rate,
            'is_healthy': self.is_healthy,
            'violations_by_method': self.violations_by_method,
            'violations_by_type': self.violations_by_type,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'last_violation': self.last_violation.isoformat() if self.last_violation else None,
            'first_check': self.first_check.isoformat() if self.first_check else None,
            'most_violated_method': self.most_violated_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContractState:
        """Deserialize from dictionary."""
        return cls(
            total_checks=data.get('total_checks', 0),
            total_violations=data.get('total_violations', 0),
            violations_by_method=data.get('violations_by_method', {}),
            violations_by_type=data.get('violations_by_type', {}),
            last_check=datetime.fromisoformat(data['last_check']) if data.get('last_check') else None,
            last_violation=datetime.fromisoformat(data['last_violation']) if data.get('last_violation') else None,
            first_check=datetime.fromisoformat(data['first_check']) if data.get('first_check') else None,
        )

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "HEALTHY" if self.is_healthy else "UNHEALTHY"
        return (
            f"ContractState({status}): "
            f"{self.total_checks} checks, {self.total_violations} violations "
            f"({self.violation_rate:.2f}%)"
        )


@dataclass
class ViolationRecord:
    """Record of a single contract violation."""

    timestamp: datetime
    contract_type: str
    description: str
    method: str
    source_file: Optional[str] = None
    source_line: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'contract_type': self.contract_type,
            'description': self.description,
            'method': self.method,
            'source_file': self.source_file,
            'source_line': self.source_line,
            'context': self.context,
        }


class ContractMaterializer:
    """
    High-level API for querying contract state from CEL.

    This provides a convenient interface for:
    - Getting current contract state
    - Querying violations in time ranges
    - Generating health reports
    - Temporal queries ("What was the state yesterday?")

    Usage:
        from cortical.contracts import ContractMaterializer

        # With CEL store
        materializer = ContractMaterializer(event_store)

        # Get current state
        state = materializer.current_state()
        print(f"Violation rate: {state.violation_rate}%")

        # Query violations
        recent = materializer.violations_since(hours=24)
        for v in recent:
            print(f"{v.timestamp}: {v.method} - {v.description}")

        # Health report
        report = materializer.health_report()
    """

    def __init__(
        self,
        event_store: Optional[Any] = None,
        emitter: Optional[Any] = None,
    ):
        """
        Initialize the materializer.

        Args:
            event_store: CEL EventStore for reading events
            emitter: ContractEventEmitter for reading buffered events
        """
        self._store = event_store
        self._emitter = emitter

    def current_state(self) -> ContractState:
        """
        Materialize current contract state from all events.

        This scans all contract-related events and computes
        the current state.
        """
        state = ContractState()

        # Use CEL store if available
        if CEL_AVAILABLE and self._store is not None:
            for event in self._store.iterate():
                self._process_event(state, event)
        # Fall back to emitter buffer
        elif self._emitter is not None:
            for event_data in self._emitter.buffered_events:
                self._process_event_data(state, event_data)

        return state

    def _process_event(self, state: ContractState, event: Any) -> None:
        """Process a CEL event into contract state."""
        if not hasattr(event, 'content'):
            return

        content = event.content
        timestamp = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))

        # Contract check observation
        if content.get('type') == 'contract_check':
            state.total_checks += 1
            state.last_check = timestamp
            if state.first_check is None:
                state.first_check = timestamp

        # Contract violation (MetaCognition)
        if hasattr(event, 'observation_type') and event.observation_type == 'contract_violation':
            state.total_violations += 1
            state.last_violation = timestamp

            metrics = getattr(event, 'metrics', {})
            method = metrics.get('method', 'unknown')
            if method not in state.violations_by_method:
                state.violations_by_method[method] = 0
            state.violations_by_method[method] += 1

    def _process_event_data(self, state: ContractState, event_data: Dict[str, Any]) -> None:
        """Process buffered event data into contract state."""
        content = event_data.get('content', {})
        timestamp_str = event_data.get('timestamp')
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()

        if content.get('type') == 'contract_check':
            state.total_checks += 1
            state.last_check = timestamp
            if state.first_check is None:
                state.first_check = timestamp

        if content.get('observation_type') == 'contract_violation':
            state.total_violations += 1
            state.last_violation = timestamp

            violation_detail = content.get('violation_detail', {})
            method = violation_detail.get('method', 'unknown')
            if method not in state.violations_by_method:
                state.violations_by_method[method] = 0
            state.violations_by_method[method] += 1

    def violations_since(
        self,
        hours: Optional[int] = None,
        since: Optional[datetime] = None,
    ) -> List[ViolationRecord]:
        """
        Get violations since a specific time.

        Args:
            hours: Number of hours to look back
            since: Specific datetime to start from

        Returns:
            List of ViolationRecord objects
        """
        if hours is not None:
            since = datetime.now(timezone.utc) - timedelta(hours=hours)
        elif since is None:
            since = datetime.min.replace(tzinfo=timezone.utc)

        violations = []

        # Scan events for violations
        if CEL_AVAILABLE and self._store is not None:
            for event in self._store.iterate():
                if hasattr(event, 'observation_type') and event.observation_type == 'contract_violation':
                    event_time = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
                    if event_time >= since:
                        violations.append(self._event_to_violation(event))
        elif self._emitter is not None:
            for event_data in self._emitter.buffered_events:
                content = event_data.get('content', {})
                if content.get('observation_type') == 'contract_violation':
                    timestamp_str = event_data['timestamp']
                    event_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    # Ensure timezone aware
                    if event_time.tzinfo is None:
                        event_time = event_time.replace(tzinfo=timezone.utc)
                    if event_time >= since:
                        violations.append(self._event_data_to_violation(event_data))

        return violations

    def _event_to_violation(self, event: Any) -> ViolationRecord:
        """Convert CEL event to ViolationRecord."""
        metrics = getattr(event, 'metrics', {})
        conclusions = getattr(event, 'conclusions', [])

        return ViolationRecord(
            timestamp=datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')),
            contract_type='unknown',  # Would need to store this in event
            description=conclusions[0] if conclusions else 'Contract violated',
            method=metrics.get('method', 'unknown'),
        )

    def _event_data_to_violation(self, event_data: Dict[str, Any]) -> ViolationRecord:
        """Convert buffered event data to ViolationRecord."""
        content = event_data.get('content', {})
        detail = content.get('violation_detail', {})

        return ViolationRecord(
            timestamp=datetime.fromisoformat(event_data['timestamp']),
            contract_type=detail.get('contract_type', 'unknown'),
            description=detail.get('description', 'Contract violated'),
            method=detail.get('method', 'unknown'),
            source_file=detail.get('source_file'),
            source_line=detail.get('source_line'),
            context=detail.get('context', {}),
        )

    def health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive health report.

        Returns:
            Dictionary with health metrics and recommendations
        """
        state = self.current_state()
        recent_violations = self.violations_since(hours=24)

        recommendations = []
        if state.violation_rate > 5:
            recommendations.append("CRITICAL: Violation rate > 5%. Review contract definitions.")
        elif state.violation_rate > 1:
            recommendations.append("WARNING: Violation rate > 1%. Investigate recent violations.")

        if state.most_violated_method:
            recommendations.append(
                f"Focus attention on: {state.most_violated_method} "
                f"({state.violations_by_method[state.most_violated_method]} violations)"
            )

        if len(recent_violations) > 10:
            recommendations.append(
                f"High recent activity: {len(recent_violations)} violations in last 24h"
            )

        return {
            'status': 'HEALTHY' if state.is_healthy else 'UNHEALTHY',
            'state': state.to_dict(),
            'recent_violations_24h': len(recent_violations),
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat(),
        }

    def state_at(self, horizon: str) -> ContractState:
        """
        Get contract state at a specific event horizon.

        This enables temporal queries:
        "What was the contract state yesterday?"
        "How many violations had we seen before commit X?"

        Args:
            horizon: Event ID to materialize up to

        Returns:
            ContractState at that point in time
        """
        state = ContractState()

        if CEL_AVAILABLE and self._store is not None:
            for event in self._store.iterate(to_event=horizon):
                self._process_event(state, event)

        return state
