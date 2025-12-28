"""
Tracing and Debugging Infrastructure for CEL.

This module provides comprehensive observability for the Cognitive Event Lattice,
enabling deep debugging of complex cognitive operations.

Design Philosophy:
    "A system you can't observe is a system you can't fix."

    Every operation in CEL can be traced, from event creation through
    materialization to health checks. Traces themselves become events
    (meta-cognition), creating a self-documenting audit trail.

Key Concepts:
    - Trace: A single operation with timing and context
    - Span: A causal chain of related traces
    - TraceContext: Propagated context for distributed tracing

Debugging Strategies:
    1. TIME TRAVEL: Replay traces to understand what happened
    2. CAUSAL ANALYSIS: Follow parent links to find root cause
    3. CONCEPT SEARCH: Find all traces involving a concept
    4. ANOMALY DETECTION: Identify unusual patterns

Usage:
    tracer = CELTracer(config)

    with tracer.span("materialize_task") as span:
        span.set_attribute("task_id", task_id)
        span.set_attribute("horizon", horizon.event_id)

        result = do_materialization()

        span.set_attribute("version", result.version)

    # Traces are automatically recorded as events if configured
"""

from __future__ import annotations

import functools
import json
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

from .config import CELConfig, TraceConfig, utc_now_iso


# =============================================================================
# TRACE TYPES
# =============================================================================

class TraceLevel(Enum):
    """Trace severity/importance levels."""
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    CRITICAL = 50


class TraceCategory(Enum):
    """Categories of traceable operations."""
    EVENT_STORE = "event_store"
    MATERIALIZATION = "materialization"
    SEMANTIC_INDEX = "semantic_index"
    HEALTH_CHECK = "health_check"
    COMPACTION = "compaction"
    MIGRATION = "migration"
    QUERY = "query"
    META = "meta"


@dataclass
class TraceAttribute:
    """A single attribute attached to a trace."""
    key: str
    value: Any
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self._serialize_value(self.value),
            "timestamp": self.timestamp,
        }

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize value for JSON storage."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [TraceAttribute._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: TraceAttribute._serialize_value(v) for k, v in value.items()}
        if hasattr(value, 'to_dict'):
            return value.to_dict()
        return str(value)


@dataclass
class Trace:
    """
    A single trace record.

    Captures one operation with its timing, attributes, and relationships.
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    category: TraceCategory
    level: TraceLevel
    start_time: str
    end_time: Optional[str] = None
    duration_ms: Optional[float] = None
    attributes: List[TraceAttribute] = field(default_factory=list)
    error: Optional[str] = None
    stack_trace: Optional[str] = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Add an attribute to this trace."""
        self.attributes.append(TraceAttribute(key=key, value=value))

    def set_error(self, error: Exception, include_stack: bool = True) -> None:
        """Record an error on this trace."""
        self.error = str(error)
        self.level = TraceLevel.ERROR
        if include_stack:
            self.stack_trace = traceback.format_exc()

    def finish(self) -> None:
        """Mark trace as complete."""
        self.end_time = utc_now_iso()
        if self.start_time and self.end_time:
            start = datetime.fromisoformat(self.start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(self.end_time.replace('Z', '+00:00'))
            self.duration_ms = (end - start).total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation": self.operation,
            "category": self.category.value,
            "level": self.level.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": [a.to_dict() for a in self.attributes],
            "error": self.error,
            "stack_trace": self.stack_trace,
        }

    def summary(self) -> str:
        """Human-readable one-line summary."""
        status = "ERR" if self.error else "OK "
        duration = f"{self.duration_ms:.1f}ms" if self.duration_ms else "..."
        attrs = ", ".join(f"{a.key}={a.value}" for a in self.attributes[:3])
        if len(self.attributes) > 3:
            attrs += f" (+{len(self.attributes) - 3})"
        return f"[{status}] {self.operation}: {duration} | {attrs}"


# =============================================================================
# TRACE CONTEXT (For Distributed Tracing)
# =============================================================================

@dataclass
class TraceContext:
    """
    Propagated context for distributed tracing.

    When operations span multiple nodes or async boundaries,
    TraceContext maintains the causal chain.
    """
    trace_id: str
    span_id: str
    baggage: Dict[str, str] = field(default_factory=dict)

    def child_context(self, new_span_id: str) -> 'TraceContext':
        """Create a child context for a new span."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=new_span_id,
            baggage=self.baggage.copy(),
        )

    def to_headers(self) -> Dict[str, str]:
        """Serialize to HTTP headers for propagation."""
        headers = {
            "X-Trace-ID": self.trace_id,
            "X-Span-ID": self.span_id,
        }
        for key, value in self.baggage.items():
            headers[f"X-Baggage-{key}"] = value
        return headers

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional['TraceContext']:
        """Deserialize from HTTP headers."""
        trace_id = headers.get("X-Trace-ID")
        span_id = headers.get("X-Span-ID")
        if not trace_id or not span_id:
            return None

        baggage = {}
        for key, value in headers.items():
            if key.startswith("X-Baggage-"):
                baggage_key = key[10:]  # Remove prefix
                baggage[baggage_key] = value

        return cls(trace_id=trace_id, span_id=span_id, baggage=baggage)


# =============================================================================
# TRACER IMPLEMENTATION
# =============================================================================

# Thread-local storage for current span
_current_context: threading.local = threading.local()


def _generate_id() -> str:
    """Generate a unique trace/span ID."""
    import hashlib
    import random
    data = f"{time.time_ns()}-{random.random()}-{threading.current_thread().ident}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class SpanContext:
    """
    Context manager for a tracing span.

    Automatically handles timing and parent relationships.
    """

    def __init__(
        self,
        tracer: 'CELTracer',
        operation: str,
        category: TraceCategory,
        level: TraceLevel = TraceLevel.INFO,
    ):
        self._tracer = tracer
        self._operation = operation
        self._category = category
        self._level = level
        self._trace: Optional[Trace] = None
        self._parent_context: Optional[TraceContext] = None

    def __enter__(self) -> Trace:
        # Get parent context
        self._parent_context = getattr(_current_context, 'context', None)

        # Generate IDs
        span_id = _generate_id()
        trace_id = self._parent_context.trace_id if self._parent_context else _generate_id()
        parent_span_id = self._parent_context.span_id if self._parent_context else None

        # Create trace
        self._trace = Trace(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=self._operation,
            category=self._category,
            level=self._level,
            start_time=utc_now_iso(),
        )

        # Set current context
        _current_context.context = TraceContext(trace_id=trace_id, span_id=span_id)

        return self._trace

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._trace:
            if exc_val:
                self._trace.set_error(exc_val)

            self._trace.finish()
            self._tracer._record(self._trace)

        # Restore parent context
        _current_context.context = self._parent_context


class CELTracer:
    """
    Main tracer for the Cognitive Event Lattice.

    Provides structured tracing with optional output to:
    - stderr (for development)
    - file (for post-mortem analysis)
    - events (for self-documenting audit trail)
    """

    def __init__(
        self,
        config: CELConfig = None,
        trace_config: TraceConfig = None,
    ):
        self._config = config or CELConfig()
        self._trace_config = trace_config or TraceConfig()
        self._traces: List[Trace] = []
        self._lock = threading.Lock()

        # Output handlers
        self._file_handle = None
        if self._trace_config.trace_to_file:
            self._file_handle = open(self._trace_config.trace_to_file, 'a')

    def span(
        self,
        operation: str,
        category: TraceCategory = TraceCategory.META,
        level: TraceLevel = TraceLevel.INFO,
    ) -> SpanContext:
        """
        Create a new tracing span.

        Usage:
            with tracer.span("operation_name") as span:
                span.set_attribute("key", value)
                do_work()
        """
        if not self._config.enable_tracing:
            return _NoOpSpanContext()

        return SpanContext(self, operation, category, level)

    def trace_event(
        self,
        operation: str,
        category: TraceCategory,
        attributes: Dict[str, Any] = None,
        level: TraceLevel = TraceLevel.INFO,
    ) -> None:
        """Record a point-in-time trace (no duration)."""
        if not self._config.enable_tracing:
            return

        trace = Trace(
            trace_id=_generate_id(),
            span_id=_generate_id(),
            parent_span_id=None,
            operation=operation,
            category=category,
            level=level,
            start_time=utc_now_iso(),
            end_time=utc_now_iso(),
            duration_ms=0.0,
        )

        if attributes:
            for key, value in attributes.items():
                trace.set_attribute(key, value)

        self._record(trace)

    def _record(self, trace: Trace) -> None:
        """Record a completed trace."""
        # Sampling
        if self._config.trace_sample_rate < 1.0:
            import random
            if random.random() > self._config.trace_sample_rate:
                return

        # Duration filtering
        if trace.duration_ms and trace.duration_ms < self._trace_config.min_duration_ms:
            return

        # Concept filtering
        if self._trace_config.concept_filter:
            trace_concepts = set()
            for attr in trace.attributes:
                if attr.key == 'concepts':
                    trace_concepts.update(attr.value if isinstance(attr.value, list) else [attr.value])
            if not trace_concepts & self._trace_config.concept_filter:
                return

        with self._lock:
            self._traces.append(trace)

        # Output
        if self._trace_config.trace_to_stderr:
            self._output_stderr(trace)

        if self._file_handle:
            self._output_file(trace)

    def _output_stderr(self, trace: Trace) -> None:
        """Output trace to stderr."""
        if self._trace_config.output_format == "json":
            print(json.dumps(trace.to_dict()), file=sys.stderr)
        else:
            print(trace.summary(), file=sys.stderr)

    def _output_file(self, trace: Trace) -> None:
        """Output trace to file."""
        self._file_handle.write(json.dumps(trace.to_dict()) + "\n")
        self._file_handle.flush()

    def get_traces(
        self,
        category: TraceCategory = None,
        operation: str = None,
        min_duration_ms: float = None,
        has_error: bool = None,
    ) -> List[Trace]:
        """Query recorded traces with optional filters."""
        with self._lock:
            traces = self._traces.copy()

        if category:
            traces = [t for t in traces if t.category == category]
        if operation:
            traces = [t for t in traces if t.operation == operation]
        if min_duration_ms is not None:
            traces = [t for t in traces if (t.duration_ms or 0) >= min_duration_ms]
        if has_error is not None:
            traces = [t for t in traces if (t.error is not None) == has_error]

        return traces

    def get_trace_tree(self, trace_id: str) -> Dict[str, Any]:
        """
        Reconstruct the causal tree for a trace.

        Returns a nested structure showing parent-child relationships.
        """
        with self._lock:
            traces = [t for t in self._traces if t.trace_id == trace_id]

        if not traces:
            return {}

        # Build parent -> children map
        children: Dict[Optional[str], List[Trace]] = {}
        for trace in traces:
            parent = trace.parent_span_id
            if parent not in children:
                children[parent] = []
            children[parent].append(trace)

        def build_tree(parent_id: Optional[str]) -> List[Dict[str, Any]]:
            result = []
            for trace in children.get(parent_id, []):
                node = trace.to_dict()
                node['children'] = build_tree(trace.span_id)
                result.append(node)
            return result

        # Find root (no parent)
        roots = children.get(None, [])
        if roots:
            tree = roots[0].to_dict()
            tree['children'] = build_tree(roots[0].span_id)
            return tree

        return {}

    def summary(self) -> Dict[str, Any]:
        """Get summary statistics of recorded traces."""
        with self._lock:
            traces = self._traces.copy()

        if not traces:
            return {"total": 0}

        by_category: Dict[str, int] = {}
        by_operation: Dict[str, List[float]] = {}
        error_count = 0
        total_duration = 0.0

        for trace in traces:
            # By category
            cat = trace.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            # By operation (with durations)
            op = trace.operation
            if op not in by_operation:
                by_operation[op] = []
            if trace.duration_ms:
                by_operation[op].append(trace.duration_ms)
                total_duration += trace.duration_ms

            # Errors
            if trace.error:
                error_count += 1

        # Calculate operation stats
        operation_stats = {}
        for op, durations in by_operation.items():
            if durations:
                operation_stats[op] = {
                    "count": len(durations),
                    "avg_ms": sum(durations) / len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                }
            else:
                operation_stats[op] = {"count": 1, "avg_ms": 0}

        return {
            "total": len(traces),
            "error_count": error_count,
            "total_duration_ms": total_duration,
            "by_category": by_category,
            "operations": operation_stats,
        }

    def clear(self) -> None:
        """Clear recorded traces."""
        with self._lock:
            self._traces.clear()

    def close(self) -> None:
        """Close any open file handles."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None


class _NoOpSpanContext:
    """No-op span context when tracing is disabled."""

    def __enter__(self) -> '_NoOpTrace':
        return _NoOpTrace()

    def __exit__(self, *args) -> None:
        pass


class _NoOpTrace:
    """No-op trace when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_error(self, error: Exception, include_stack: bool = True) -> None:
        pass


# =============================================================================
# DECORATOR FOR AUTOMATIC TRACING
# =============================================================================

F = TypeVar('F', bound=Callable[..., Any])


def traced(
    category: TraceCategory = TraceCategory.META,
    level: TraceLevel = TraceLevel.INFO,
    include_args: bool = False,
    include_result: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to automatically trace a function.

    Usage:
        @traced(category=TraceCategory.MATERIALIZATION)
        def materialize_task(task_id: str) -> Task:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get tracer from first arg if it's an object with _tracer
            tracer = None
            if args and hasattr(args[0], '_tracer'):
                tracer = args[0]._tracer
            elif args and hasattr(args[0], 'tracer'):
                tracer = args[0].tracer

            if tracer is None:
                # No tracer available, just call function
                return func(*args, **kwargs)

            operation = f"{func.__module__}.{func.__qualname__}"

            with tracer.span(operation, category, level) as span:
                if include_args:
                    span.set_attribute("args_count", len(args))
                    span.set_attribute("kwargs_keys", list(kwargs.keys()))

                result = func(*args, **kwargs)

                if include_result and result is not None:
                    span.set_attribute("result_type", type(result).__name__)

                return result

        return wrapper
    return decorator


# =============================================================================
# TRACE ANALYSIS UTILITIES
# =============================================================================

def find_slow_operations(
    tracer: CELTracer,
    percentile: float = 95.0,
) -> List[Trace]:
    """Find operations above the specified duration percentile."""
    traces = tracer.get_traces()
    durations = sorted([t.duration_ms for t in traces if t.duration_ms])

    if not durations:
        return []

    idx = int(len(durations) * percentile / 100)
    threshold = durations[min(idx, len(durations) - 1)]

    return [t for t in traces if t.duration_ms and t.duration_ms >= threshold]


def find_error_chains(tracer: CELTracer) -> List[Dict[str, Any]]:
    """Find all error traces with their causal chains."""
    error_traces = tracer.get_traces(has_error=True)

    chains = []
    for trace in error_traces:
        tree = tracer.get_trace_tree(trace.trace_id)
        if tree:
            chains.append({
                "error_trace": trace.to_dict(),
                "causal_chain": tree,
            })

    return chains


def generate_trace_report(tracer: CELTracer) -> str:
    """Generate a human-readable trace report."""
    summary = tracer.summary()

    lines = [
        "=" * 60,
        "CEL TRACE REPORT",
        "=" * 60,
        f"\nTotal Traces: {summary['total']}",
        f"Errors: {summary['error_count']}",
        f"Total Duration: {summary['total_duration_ms']:.1f}ms",
        "\nBy Category:",
    ]

    for cat, count in summary.get('by_category', {}).items():
        lines.append(f"  {cat}: {count}")

    lines.append("\nOperation Statistics:")
    for op, stats in summary.get('operations', {}).items():
        lines.append(f"  {op}:")
        lines.append(f"    count: {stats['count']}, avg: {stats['avg_ms']:.2f}ms")

    # Slow operations
    slow = find_slow_operations(tracer, 95.0)
    if slow:
        lines.append(f"\nSlowest Operations (95th percentile):")
        for trace in slow[:5]:
            lines.append(f"  {trace.summary()}")

    # Error chains
    errors = find_error_chains(tracer)
    if errors:
        lines.append(f"\nError Chains ({len(errors)} found):")
        for chain in errors[:3]:
            et = chain['error_trace']
            lines.append(f"  {et['operation']}: {et['error']}")

    lines.append("=" * 60)
    return "\n".join(lines)
