"""
Observability Module
====================

Provides timing hooks, metrics collection, and trace context for monitoring
the Cortical Text Processor's performance and operations.

This module follows the "Native Over External" principle - no external
dependencies required. Uses only Python's standard library.

Example:
    from cortical import CorticalTextProcessor

    # Enable metrics collection
    processor = CorticalTextProcessor(enable_metrics=True)
    processor.process_document("doc1", "Neural networks process data.")
    processor.compute_all()

    # Get metrics
    metrics = processor.get_metrics()
    print(f"compute_all took {metrics['compute_all']['avg_ms']:.2f}ms")

    # Get metrics summary
    print(processor.get_metrics_summary())

Logging Configuration:
    The observability module uses Python's standard logging. Configure levels as needed:

    # For production (minimal output):
    logging.getLogger('cortical.observability').setLevel(logging.WARNING)

    # For development (verbose):
    logging.getLogger('cortical.observability').setLevel(logging.DEBUG)

    # To see timing logs:
    logging.basicConfig(level=logging.DEBUG)
"""

import time
import functools
import logging
from typing import Dict, Any, Optional, Callable, List, Deque
from collections import defaultdict, deque
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and aggregates timing and count metrics for operations.

    Note: This class is NOT thread-safe. For multi-threaded applications,
    wrap method calls with appropriate locking (e.g., threading.Lock).
    Consider using thread-local MetricsCollector instances for concurrent access.

    Attributes:
        enabled: Whether metrics collection is active
        operations: Dict mapping operation names to timing/count data
        traces: Dict mapping trace IDs to operation logs
        max_timing_history: Maximum timing entries to keep per operation
    """

    def __init__(self, enabled: bool = True, max_timing_history: int = 1000):
        """
        Initialize metrics collector.

        Args:
            enabled: Start with metrics collection enabled
            max_timing_history: Maximum number of timing measurements to keep per operation.
                               Set to 0 to disable timing history entirely (saves memory).
                               Default 1000 keeps last 1000 timings for percentile analysis.
        """
        self.enabled = enabled
        self.max_timing_history = max_timing_history
        # operation_name -> {'count': int, 'total_ms': float, 'min_ms': float, 'max_ms': float, 'timings': deque}
        # Need to capture max_timing_history in closure
        max_history = max_timing_history
        self.operations: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_ms': 0.0,
            'min_ms': float('inf'),
            'max_ms': 0.0,
            'timings': deque(maxlen=max_history if max_history > 0 else 0)
        })
        # trace_id -> list of (operation, duration_ms, context) tuples
        self.traces: Dict[str, List[tuple]] = defaultdict(list)
        self._current_trace_id: Optional[str] = None

    def record_timing(
        self,
        operation: str,
        duration_ms: float,
        trace_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a timing measurement for an operation.

        Args:
            operation: Name of the operation (e.g., "compute_all")
            duration_ms: Duration in milliseconds
            trace_id: Optional trace ID for request tracing
            context: Optional context dict (doc_id, query, etc.)
        """
        if not self.enabled:
            return

        op_data = self.operations[operation]
        op_data['count'] += 1
        op_data['total_ms'] += duration_ms
        op_data['min_ms'] = min(op_data['min_ms'], duration_ms)
        op_data['max_ms'] = max(op_data['max_ms'], duration_ms)
        op_data['timings'].append(duration_ms)

        # Record trace if trace_id is provided or active
        effective_trace_id = trace_id or self._current_trace_id
        if effective_trace_id:
            self.traces[effective_trace_id].append((operation, duration_ms, context or {}))

    def record_count(self, metric_name: str, count: int = 1) -> None:
        """
        Record a simple count metric.

        Args:
            metric_name: Name of the metric (e.g., "cache_hits")
            count: Count to add (default 1)
        """
        if not self.enabled:
            return

        # Store counts separately from timings
        if metric_name not in self.operations:
            self.operations[metric_name] = {'count': 0}

        if 'count' in self.operations[metric_name]:
            self.operations[metric_name]['count'] += count
        else:
            self.operations[metric_name]['count'] = count

    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """
        Get statistics for a specific operation.

        Args:
            operation: Operation name

        Returns:
            Dict with count, total_ms, avg_ms, min_ms, max_ms
        """
        if operation not in self.operations:
            return {}

        op_data = self.operations[operation]
        stats = {
            'count': op_data['count']
        }

        # Only add timing stats if they exist
        if 'total_ms' in op_data:
            stats['total_ms'] = op_data['total_ms']
            stats['avg_ms'] = op_data['total_ms'] / op_data['count'] if op_data['count'] > 0 else 0.0
            stats['min_ms'] = op_data['min_ms'] if op_data['min_ms'] != float('inf') else 0.0
            stats['max_ms'] = op_data['max_ms']

        return stats

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all operations.

        Returns:
            Dict mapping operation names to their stats
        """
        return {op: self.get_operation_stats(op) for op in self.operations}

    def get_trace(self, trace_id: str) -> List[tuple]:
        """
        Get all operations recorded for a trace ID.

        Args:
            trace_id: Trace identifier

        Returns:
            List of (operation, duration_ms, context) tuples
        """
        return self.traces.get(trace_id, [])

    def reset(self) -> None:
        """Clear all collected metrics."""
        self.operations.clear()
        self.traces.clear()
        self._current_trace_id = None

    def enable(self) -> None:
        """Enable metrics collection."""
        self.enabled = True

    def disable(self) -> None:
        """Disable metrics collection."""
        self.enabled = False

    @contextmanager
    def trace_context(self, trace_id: str):
        """
        Context manager for tracing a block of operations.

        Args:
            trace_id: Unique trace identifier

        Example:
            >>> with metrics.trace_context("request-123"):
            ...     processor.find_documents_for_query("neural")
            >>> print(metrics.get_trace("request-123"))
        """
        previous_trace = self._current_trace_id
        self._current_trace_id = trace_id
        try:
            yield
        finally:
            self._current_trace_id = previous_trace

    def get_summary(self) -> str:
        """
        Get a human-readable summary of all metrics.

        Returns:
            Formatted string with metrics table
        """
        if not self.operations:
            return "No metrics collected."

        lines = ["Metrics Summary", "=" * 80]

        # Separate timing operations from counts
        timing_ops = []
        count_ops = []

        for op_name in sorted(self.operations.keys()):
            stats = self.get_operation_stats(op_name)
            if 'total_ms' in stats:
                timing_ops.append((op_name, stats))
            else:
                count_ops.append((op_name, stats))

        # Display timing operations
        if timing_ops:
            lines.append("\nTiming Operations:")
            lines.append(f"{'Operation':<30} {'Count':>8} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Total(ms)':>12}")
            lines.append("-" * 80)

            for op_name, stats in timing_ops:
                lines.append(
                    f"{op_name:<30} {stats['count']:>8} "
                    f"{stats['avg_ms']:>10.2f} {stats['min_ms']:>10.2f} "
                    f"{stats['max_ms']:>10.2f} {stats['total_ms']:>12.2f}"
                )

        # Display count operations
        if count_ops:
            lines.append("\nCount Metrics:")
            lines.append(f"{'Metric':<40} {'Count':>10}")
            lines.append("-" * 50)

            for op_name, stats in count_ops:
                lines.append(f"{op_name:<40} {stats['count']:>10}")

        # Display trace summary
        if self.traces:
            lines.append(f"\nActive Traces: {len(self.traces)}")

        return "\n".join(lines)


class TraceContext:
    """
    Context for request tracing across operations.

    Stores trace ID and optional metadata for correlating operations.
    """

    def __init__(self, trace_id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize trace context.

        Args:
            trace_id: Unique identifier for this trace
            metadata: Optional metadata (user_id, session_id, etc.)
        """
        self.trace_id = trace_id
        self.metadata = metadata or {}
        self.start_time = time.time()

    def elapsed_ms(self) -> float:
        """Get elapsed time since trace started in milliseconds."""
        return (time.time() - self.start_time) * 1000.0


def timed(operation_name: Optional[str] = None, include_args: bool = False):
    """
    Decorator for timing method calls and recording to metrics.

    Args:
        operation_name: Custom name for the operation (defaults to function name)
        include_args: Include function arguments in trace context

    Example:
        >>> class MyClass:
        ...     @timed("my_operation")
        ...     def my_method(self):
        ...         time.sleep(0.1)
        ...         return "done"
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if instance has metrics enabled
            metrics = getattr(self, '_metrics', None)

            if not metrics or not metrics.enabled:
                # No metrics collection, just run the function
                return func(self, *args, **kwargs)

            # Prepare context
            context = {}
            if include_args and args:
                # Include first few args in context (avoid huge dumps)
                context['args'] = str(args[:3])
            if include_args and kwargs:
                # Include important kwargs
                for key in ['doc_id', 'query', 'query_text', 'top_n']:
                    if key in kwargs:
                        context[key] = kwargs[key]

            # Time the operation
            start = time.perf_counter()
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                metrics.record_timing(op_name, duration_ms, context=context)

        return wrapper
    return decorator


def measure_time(func: Callable) -> Callable:
    """
    Simple timing decorator that logs execution time.

    Useful for debugging without full metrics collection.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs its execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.debug(f"{func.__name__} took {duration_ms:.2f}ms")
        return result
    return wrapper


# Convenience functions for standalone usage

_global_metrics = MetricsCollector(enabled=False)


def get_global_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _global_metrics


def enable_global_metrics() -> None:
    """Enable global metrics collection."""
    _global_metrics.enable()


def disable_global_metrics() -> None:
    """Disable global metrics collection."""
    _global_metrics.disable()


def reset_global_metrics() -> None:
    """Reset global metrics."""
    _global_metrics.reset()
