"""
CEL Integration for Behavioral Contracts.

This module connects contracts to the Cognitive Event Lattice:
1. ContractEventEmitter: Emits CEL events when contracts are checked
2. contract_reducer: Materializes contract state from events

Event Flow:
    @requires check → Observation event (contract_check)
    @ensures check → Observation event (contract_check)
    @invariant check → Observation event (contract_check)
    Violation → MetaCognition event (contract_violation)

The event stream enables:
    - Temporal queries: "What contracts were violated in the last hour?"
    - Audit trails: "Show me all contract checks for TransactionManager"
    - Compaction: Summarize contract history into statistics
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .decorators import ContractSpec
    from .registry import ContractRegistry

# Import CEL types - gracefully handle if CEL not fully available
try:
    from cortical.cel.core.events import (
        CognitiveEvent,
        EventType,
        Observation,
        MetaCognition,
    )
    from cortical.cel.core.protocols import EventStore
    CEL_AVAILABLE = True
except ImportError:
    CEL_AVAILABLE = False
    CognitiveEvent = None  # type: ignore
    EventStore = None  # type: ignore


class ContractEventEmitter:
    """
    Emits CEL events for contract checks and violations.

    Each contract check creates an Observation event.
    Each violation creates a MetaCognition event (system self-awareness).

    Usage:
        from cortical.contracts import ContractRegistry, ContractEventEmitter

        # Create with CEL event store
        emitter = ContractEventEmitter(event_store)
        registry = ContractRegistry(emitter=emitter)

        # Or standalone (for testing)
        emitter = ContractEventEmitter()  # In-memory events
    """

    def __init__(
        self,
        event_store: Optional[Any] = None,
        registry: Optional[ContractRegistry] = None,
        emit_all_checks: bool = False,
    ):
        """
        Initialize the emitter.

        Args:
            event_store: CEL EventStore for persistence (optional)
            registry: ContractRegistry for stats updates (optional)
            emit_all_checks: If True, emit events for ALL checks.
                            If False (default), only emit violations.
                            This reduces event volume in production.
        """
        self._store = event_store
        self._registry = registry
        self._emit_all_checks = emit_all_checks

        # In-memory event buffer (used if no store provided)
        self._event_buffer: List[Dict[str, Any]] = []
        self._max_buffer = 1000

        # Statistics
        self._checks_emitted = 0
        self._violations_emitted = 0
        self._last_violation: Optional[datetime] = None

    def emit_check(
        self,
        spec: ContractSpec,
        passed: bool,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Emit a CEL event for a contract check.

        Args:
            spec: The contract specification
            passed: Whether the check passed
            context: Execution context (method, args, etc.)

        Returns:
            Event ID if emitted, None otherwise
        """
        now = datetime.now(timezone.utc)

        # Update registry stats
        if self._registry is not None:
            contract_id = f"{spec.class_name or 'global'}.{spec.method_name}.{spec.contract_type}.0"
            self._registry.update_stats(contract_id, passed, now)

        # Decide whether to emit
        if not self._emit_all_checks and passed:
            return None  # Skip passed checks in production mode

        event_id = None

        if passed:
            # Observation: Contract check passed
            event_id = self._emit_observation(spec, context, now)
            self._checks_emitted += 1
        else:
            # MetaCognition: Contract violation detected
            event_id = self._emit_violation(spec, context, now)
            self._violations_emitted += 1
            self._last_violation = now

        return event_id

    def _emit_observation(
        self,
        spec: ContractSpec,
        context: Dict[str, Any],
        timestamp: datetime,
    ) -> Optional[str]:
        """Emit an Observation event for a passed contract check."""
        content = {
            'type': 'contract_check',
            'contract_type': spec.contract_type,
            'description': spec.description,
            'method': f"{spec.class_name}.{spec.method_name}",
            'passed': True,
            'context': context,
        }

        concepts = (
            'contract',
            spec.contract_type,
            spec.method_name or 'unknown',
        )

        if CEL_AVAILABLE and self._store is not None:
            event = Observation(
                content=content,
                concepts=concepts,
                timestamp=timestamp.isoformat(),
            )
            self._store.append(event)
            return event.id
        else:
            # Buffer in memory
            event_data = {
                'timestamp': timestamp.isoformat(),
                'event_type': 'OBSERVATION',
                'content': content,
                'concepts': concepts,
            }
            self._buffer_event(event_data)
            return None

    def _emit_violation(
        self,
        spec: ContractSpec,
        context: Dict[str, Any],
        timestamp: datetime,
    ) -> Optional[str]:
        """Emit a MetaCognition event for a contract violation."""
        content = {
            'type': 'contract_violation',
            'contract_type': spec.contract_type,
            'description': spec.description,
            'method': f"{spec.class_name}.{spec.method_name}",
            'source_file': spec.source_file,
            'source_line': spec.source_line,
            'context': context,
        }

        # MetaCognition captures system self-awareness
        observation_type = 'contract_violation'
        metrics = {
            'total_violations': self._violations_emitted + 1,
            'method': f"{spec.class_name}.{spec.method_name}",
        }
        conclusions = [
            f"Contract violated: {spec.description}",
            f"Method: {spec.class_name}.{spec.method_name}",
        ]
        actions_triggered = ['log_violation', 'update_stats']

        if CEL_AVAILABLE and self._store is not None:
            event = MetaCognition(
                observation_type=observation_type,
                metrics=metrics,
                conclusions=conclusions,
                actions_triggered=actions_triggered,
                timestamp=timestamp.isoformat(),
            )
            # Store the violation content in metadata
            # (MetaCognition has specific structure, so we add extra info here)
            self._store.append(event)
            return event.id
        else:
            event_data = {
                'timestamp': timestamp.isoformat(),
                'event_type': 'METACOGNITION',
                'content': {
                    'observation_type': observation_type,
                    'metrics': metrics,
                    'conclusions': conclusions,
                    'actions_triggered': actions_triggered,
                    'violation_detail': content,
                },
            }
            self._buffer_event(event_data)
            return None

    def _buffer_event(self, event_data: Dict[str, Any]) -> None:
        """Buffer event in memory (when no store available)."""
        self._event_buffer.append(event_data)
        # Circular buffer
        if len(self._event_buffer) > self._max_buffer:
            self._event_buffer.pop(0)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get emitter statistics."""
        return {
            'checks_emitted': self._checks_emitted,
            'violations_emitted': self._violations_emitted,
            'last_violation': self._last_violation.isoformat() if self._last_violation else None,
            'emit_all_checks': self._emit_all_checks,
            'has_event_store': self._store is not None,
            'buffer_size': len(self._event_buffer),
        }

    @property
    def buffered_events(self) -> List[Dict[str, Any]]:
        """Get buffered events (for testing or when no store)."""
        return list(self._event_buffer)

    def clear_buffer(self) -> int:
        """Clear the event buffer, return count of cleared events."""
        count = len(self._event_buffer)
        self._event_buffer.clear()
        return count


# =============================================================================
# CONTRACT REDUCER - Materialize contract state from events
# =============================================================================


def contract_reducer(
    state: Optional[Dict[str, Any]],
    event: Any,
) -> Optional[Dict[str, Any]]:
    """
    Reducer for materializing contract state from CEL events.

    This reducer folds contract-related events into a summary:
    - Total checks
    - Total violations
    - Violation rate
    - Last violation time
    - Violations by method
    - Violations by contract type

    Usage with CEL Materializer:
        from cortical.cel.wisdom.materializer import EntityReducerRegistry
        from cortical.contracts import contract_reducer

        registry = EntityReducerRegistry()
        registry.add(_ContractReducerWrapper())

        materializer = CachingMaterializer(store, registry)
        contract_state = materializer.materialize('CONTRACT-SUMMARY')

    The state is keyed by 'CONTRACT-SUMMARY' (singleton).
    """
    if not CEL_AVAILABLE:
        return state

    # Initialize state
    if state is None:
        state = {
            'entity_type': 'contract_summary',
            'total_checks': 0,
            'total_violations': 0,
            'violations_by_method': {},
            'violations_by_type': {},
            'last_check': None,
            'last_violation': None,
            'first_check': None,
        }

    # Handle contract check observations
    if hasattr(event, 'event_type') and hasattr(event, 'content'):
        content = event.content

        if content.get('type') == 'contract_check':
            state['total_checks'] += 1
            state['last_check'] = event.timestamp
            if state['first_check'] is None:
                state['first_check'] = event.timestamp

        # Handle violation (MetaCognition with contract_violation)
        if hasattr(event, 'observation_type'):
            if event.observation_type == 'contract_violation':
                state['total_violations'] += 1
                state['last_violation'] = event.timestamp

                # Track by method
                metrics = getattr(event, 'metrics', {})
                method = metrics.get('method', 'unknown')
                if method not in state['violations_by_method']:
                    state['violations_by_method'][method] = 0
                state['violations_by_method'][method] += 1

    return state


class _ContractReducerWrapper:
    """Wrapper to make contract_reducer implement EventReducer protocol."""

    @property
    def entity_type(self) -> str:
        return 'contract_summary'

    def __call__(self, state: Optional[Dict], event: Any) -> Optional[Dict]:
        return contract_reducer(state, event)


def create_contract_reducer():
    """Factory for creating a contract reducer instance."""
    return _ContractReducerWrapper()
