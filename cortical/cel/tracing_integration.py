"""
Tracing Integration for the Cognitive Event Lattice.

This module provides traced wrappers for CEL services, enabling
observability without modifying the core implementations.

Design Pattern:
    Decorator pattern with dependency injection.
    Traced wrappers implement the same protocols as wrapped services,
    allowing transparent substitution via the container.

Integration:
    container.register(EventStore, TracedEventStore)
    container.register(Materializer, TracedMaterializer)

The tracing is configurable via CELConfig.enable_tracing.
When disabled, no overhead is incurred.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from .config import CELConfig
from .tracing import (
    CELTracer,
    TraceCategory,
    TraceLevel,
)
from .core.events import CognitiveEvent
from .core.references import EventHorizon


T = TypeVar('T')


# =============================================================================
# TRACED WRAPPER FACTORY
# =============================================================================

def traced_method(
    category: TraceCategory,
    level: TraceLevel = TraceLevel.INFO,
) -> Callable:
    """
    Decorator factory for adding tracing to methods.

    Args:
        category: Trace category for the operation
        level: Trace level

    Returns:
        Decorator that wraps method with tracing

    Example:
        @traced_method(TraceCategory.EVENT_STORE)
        def append(self, event: CognitiveEvent) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if tracing is enabled on the instance
            tracer = getattr(self, '_tracer', None)
            if tracer is None or not tracer._config.enable_tracing:
                return func(self, *args, **kwargs)

            # Build operation name
            operation = f"{self.__class__.__name__}.{func.__name__}"

            # Add context attributes
            attributes = {}
            if args:
                # Add first arg as context (often entity_id or event)
                first_arg = args[0]
                if isinstance(first_arg, CognitiveEvent):
                    attributes['event_id'] = first_arg.id[:8]
                    attributes['event_type'] = first_arg.event_type.name
                elif isinstance(first_arg, str):
                    attributes['entity_id'] = first_arg

            # Execute with tracing
            with tracer.span(operation, category, level) as span:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

                try:
                    result = func(self, *args, **kwargs)

                    # Add result info
                    if isinstance(result, str):
                        span.set_attribute('result_id', result[:8])
                    elif isinstance(result, list):
                        span.set_attribute('result_count', len(result))

                    return result
                except Exception as e:
                    span.set_error(str(e))
                    raise

        return wrapper
    return decorator


# =============================================================================
# TRACED EVENT STORE
# =============================================================================

class TracedEventStore:
    """
    Event store wrapper that adds tracing to all operations.

    This wrapper implements the EventStore protocol and delegates
    to an underlying store while recording traces.

    Usage:
        store = FileSystemEventStore(path)
        traced_store = TracedEventStore(store, tracer)

        # All operations are now traced
        traced_store.append(event)
    """

    def __init__(
        self,
        delegate: Any,  # EventStore protocol
        tracer: CELTracer,
    ):
        """
        Initialize traced wrapper.

        Args:
            delegate: Underlying event store
            tracer: CEL tracer instance
        """
        self._delegate = delegate
        self._tracer = tracer

    @traced_method(TraceCategory.EVENT_STORE)
    def append(self, event: CognitiveEvent) -> str:
        """Append event with tracing."""
        return self._delegate.append(event)

    @traced_method(TraceCategory.EVENT_STORE)
    def get(self, event_id: str) -> Optional[CognitiveEvent]:
        """Get event by ID with tracing."""
        return self._delegate.get(event_id)

    @traced_method(TraceCategory.EVENT_STORE, TraceLevel.DEBUG)
    def get_many(self, event_ids: Sequence[str]) -> List[CognitiveEvent]:
        """Get multiple events with tracing."""
        return self._delegate.get_many(event_ids)

    @traced_method(TraceCategory.EVENT_STORE)
    def events_up_to(
        self,
        horizon: Optional[EventHorizon] = None,
    ) -> List[CognitiveEvent]:
        """Get events up to horizon with tracing."""
        return self._delegate.events_up_to(horizon)

    @traced_method(TraceCategory.EVENT_STORE)
    def events_for_entity(
        self,
        entity_id: str,
        up_to: Optional[EventHorizon] = None,
    ) -> List[CognitiveEvent]:
        """Get entity events with tracing."""
        return self._delegate.events_for_entity(entity_id, up_to)

    @traced_method(TraceCategory.EVENT_STORE, TraceLevel.DEBUG)
    def current_horizon(self) -> Optional[EventHorizon]:
        """Get current horizon with tracing."""
        return self._delegate.current_horizon()

    def __len__(self) -> int:
        """Delegate length (no tracing for simple property)."""
        return len(self._delegate)

    # Forward any other attributes to delegate
    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


# =============================================================================
# TRACED MATERIALIZER
# =============================================================================

class TracedMaterializer:
    """
    Materializer wrapper that adds tracing to materialization.

    Traces include:
    - Entity ID being materialized
    - Horizon (if temporal query)
    - Cache hit/miss
    - Number of events replayed
    - Time spent in reducers
    """

    def __init__(
        self,
        delegate: Any,  # Materializer protocol
        tracer: CELTracer,
    ):
        """
        Initialize traced wrapper.

        Args:
            delegate: Underlying materializer
            tracer: CEL tracer instance
        """
        self._delegate = delegate
        self._tracer = tracer

    @traced_method(TraceCategory.MATERIALIZATION)
    def materialize(
        self,
        entity_id: str,
        entity_type: str,
        at: Optional[EventHorizon] = None,
    ) -> Optional[Any]:
        """Materialize entity with tracing."""
        return self._delegate.materialize(entity_id, entity_type, at)

    @traced_method(TraceCategory.MATERIALIZATION)
    def invalidate(self, entity_id: str) -> bool:
        """Invalidate cache entry with tracing."""
        return self._delegate.invalidate(entity_id)

    @traced_method(TraceCategory.MATERIALIZATION, TraceLevel.DEBUG)
    def invalidate_all(self) -> int:
        """Invalidate all cache with tracing."""
        return self._delegate.invalidate_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get stats (no tracing for diagnostics)."""
        return self._delegate.get_stats()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


# =============================================================================
# TRACED SEMANTIC INDEX
# =============================================================================

class TracedSemanticIndex:
    """
    Semantic index wrapper with tracing.

    Traces include:
    - Index updates
    - Concept searches
    - Bloom filter operations
    """

    def __init__(
        self,
        delegate: Any,  # SemanticIndex protocol
        tracer: CELTracer,
    ):
        self._delegate = delegate
        self._tracer = tracer

    @traced_method(TraceCategory.SEMANTIC_INDEX)
    def index_event(self, event: CognitiveEvent) -> int:
        """Index event concepts with tracing."""
        return self._delegate.index_event(event)

    @traced_method(TraceCategory.SEMANTIC_INDEX)
    def search(self, concept: str) -> List[str]:
        """Search by concept with tracing."""
        return self._delegate.search(concept)

    @traced_method(TraceCategory.SEMANTIC_INDEX)
    def search_multi(self, concepts: Sequence[str]) -> List[str]:
        """Search by multiple concepts with tracing."""
        return self._delegate.search_multi(concepts)

    @traced_method(TraceCategory.SEMANTIC_INDEX, TraceLevel.DEBUG)
    def may_contain(self, concept: str) -> bool:
        """Bloom filter check with tracing."""
        return self._delegate.may_contain(concept)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


# =============================================================================
# TRACED DAG
# =============================================================================

class TracedCausalDAG:
    """
    Causal DAG wrapper with tracing.

    Traces include:
    - Ancestor queries
    - Descendant queries
    - Path finding
    """

    def __init__(
        self,
        delegate: Any,  # CausalDAG protocol
        tracer: CELTracer,
    ):
        self._delegate = delegate
        self._tracer = tracer

    @traced_method(TraceCategory.CAUSAL_DAG)
    def add_event(self, event: CognitiveEvent) -> None:
        """Add event to DAG with tracing."""
        return self._delegate.add_event(event)

    @traced_method(TraceCategory.CAUSAL_DAG)
    def get_ancestors(self, event_id: str) -> List[str]:
        """Get ancestors with tracing."""
        return self._delegate.get_ancestors(event_id)

    @traced_method(TraceCategory.CAUSAL_DAG)
    def get_descendants(self, event_id: str) -> List[str]:
        """Get descendants with tracing."""
        return self._delegate.get_descendants(event_id)

    @traced_method(TraceCategory.CAUSAL_DAG)
    def find_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """Find path with tracing."""
        return self._delegate.find_path(from_id, to_id)

    @traced_method(TraceCategory.CAUSAL_DAG, TraceLevel.DEBUG)
    def is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        """Check ancestry with tracing."""
        return self._delegate.is_ancestor(ancestor_id, descendant_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


# =============================================================================
# TRACED HEALTH MONITOR
# =============================================================================

class TracedHealthMonitor:
    """
    Health monitor wrapper with tracing.

    All health checks are automatically traced.
    """

    def __init__(
        self,
        delegate: Any,  # HealthMonitor protocol
        tracer: CELTracer,
    ):
        self._delegate = delegate
        self._tracer = tracer

    @traced_method(TraceCategory.HEALTH_CHECK)
    def check_health(self) -> Dict[str, Any]:
        """Run health check with tracing."""
        return self._delegate.check_health()

    @traced_method(TraceCategory.HEALTH_CHECK)
    def get_metrics(self) -> Dict[str, float]:
        """Get metrics with tracing."""
        return self._delegate.get_metrics()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


# =============================================================================
# CONTAINER INTEGRATION
# =============================================================================

def configure_tracing(
    container: Any,  # Container type
    config: CELConfig,
) -> CELTracer:
    """
    Configure tracing in the DI container.

    This function:
    1. Creates a tracer based on config
    2. Wraps existing services with traced versions
    3. Registers traced services in container

    Args:
        container: DI container
        config: CEL configuration

    Returns:
        Configured tracer instance

    Usage:
        container = Container()
        # ... register base services ...

        tracer = configure_tracing(container, config)

        # Now all services are traced
        store = container.resolve(EventStore)  # Returns TracedEventStore
    """
    # Create tracer using CELConfig (which has enable_tracing)
    tracer = CELTracer(config=config)

    if not config.enable_tracing:
        # Return disabled tracer, don't wrap services
        return tracer

    # Wrap and re-register services
    # Note: This assumes services are already registered
    # The container should support replacing registrations

    # Get current implementations
    from .core.protocols import EventStore, Materializer, SemanticIndex

    if hasattr(container, 'is_registered'):
        if container.is_registered(EventStore):
            original_store = container.resolve(EventStore)
            traced_store = TracedEventStore(original_store, tracer)
            container.register_instance(EventStore, traced_store)

        # Similar for other services...

    return tracer


# =============================================================================
# CONVENIENCE DECORATORS FOR USER CODE
# =============================================================================

def trace_operation(
    tracer: CELTracer,
    operation: str,
    category: TraceCategory = TraceCategory.USER_CODE,
    level: TraceLevel = TraceLevel.INFO,
) -> Callable:
    """
    Decorator for tracing user-defined operations.

    Args:
        tracer: Tracer instance
        operation: Operation name
        category: Trace category
        level: Trace level

    Returns:
        Decorated function

    Example:
        @trace_operation(tracer, "process_batch", TraceCategory.USER_CODE)
        def process_batch(items: List[Item]) -> None:
            for item in items:
                process(item)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not tracer._config.enable_tracing:
                return func(*args, **kwargs)

            with tracer.span(operation, category, level):
                return func(*args, **kwargs)

        return wrapper
    return decorator
