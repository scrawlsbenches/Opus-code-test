"""
Unit tests for CEL tracing infrastructure.

These tests exercise the actual cortical/cel/tracing.py and
cortical/cel/tracing_integration.py modules to ensure coverage.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from cortical.cel.config import CELConfig, TraceConfig, utc_now_iso, parse_iso_timestamp
from cortical.cel.tracing import (
    CELTracer,
    Trace,
    TraceAttribute,
    TraceCategory,
    TraceContext,
    TraceLevel,
    SpanContext,
    _NoOpSpanContext,
    find_slow_operations,
    find_error_chains,
    generate_trace_report,
)
from cortical.cel.tracing_integration import (
    TracedEventStore,
    TracedMaterializer,
    TracedSemanticIndex,
    TracedCausalDAG,
    TracedHealthMonitor,
    traced_method,
    trace_operation,
    configure_tracing,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def enabled_config():
    """CEL config with tracing enabled."""
    return CELConfig(enable_tracing=True, trace_sample_rate=1.0)


@pytest.fixture
def disabled_config():
    """CEL config with tracing disabled."""
    return CELConfig(enable_tracing=False)


@pytest.fixture
def tracer(enabled_config):
    """Create a tracer with tracing enabled."""
    return CELTracer(config=enabled_config)


@pytest.fixture
def disabled_tracer(disabled_config):
    """Create a tracer with tracing disabled."""
    return CELTracer(config=disabled_config)


# =============================================================================
# TEST: TRACE TYPES
# =============================================================================

class TestTraceTypes:
    """Tests for trace type enums and dataclasses."""

    def test_trace_level_ordering(self):
        """TraceLevel values are ordered by severity."""
        assert TraceLevel.DEBUG.value < TraceLevel.INFO.value
        assert TraceLevel.INFO.value < TraceLevel.WARN.value
        assert TraceLevel.WARN.value < TraceLevel.ERROR.value
        assert TraceLevel.ERROR.value < TraceLevel.CRITICAL.value

    def test_trace_category_values(self):
        """All trace categories have string values."""
        for category in TraceCategory:
            assert isinstance(category.value, str)
            assert len(category.value) > 0

    def test_trace_category_new_additions(self):
        """New trace categories are present."""
        assert TraceCategory.CAUSAL_DAG.value == "causal_dag"
        assert TraceCategory.USER_CODE.value == "user_code"

    def test_trace_attribute_serialization(self):
        """TraceAttribute serializes correctly."""
        attr = TraceAttribute(key="test_key", value="test_value")
        d = attr.to_dict()

        assert d["key"] == "test_key"
        assert d["value"] == "test_value"
        assert "timestamp" in d

    def test_trace_attribute_complex_values(self):
        """TraceAttribute handles complex values."""
        attr = TraceAttribute(key="complex", value={"nested": [1, 2, 3]})
        d = attr.to_dict()
        assert d["value"] == {"nested": [1, 2, 3]}


# =============================================================================
# TEST: CEL TRACER
# =============================================================================

class TestCELTracer:
    """Tests for the main CELTracer class."""

    def test_tracer_creation_enabled(self, enabled_config):
        """Tracer can be created with enabled config."""
        tracer = CELTracer(config=enabled_config)
        assert tracer._config.enable_tracing is True

    def test_tracer_creation_disabled(self, disabled_config):
        """Tracer can be created with disabled config."""
        tracer = CELTracer(config=disabled_config)
        assert tracer._config.enable_tracing is False

    def test_tracer_span_creates_trace(self, tracer):
        """Span creates a trace when completed."""
        with tracer.span("test_op", TraceCategory.EVENT_STORE) as span:
            span.set_attribute("key", "value")

        assert len(tracer._traces) == 1
        trace = tracer._traces[0]
        assert trace.operation == "test_op"
        assert trace.category == TraceCategory.EVENT_STORE

    def test_tracer_disabled_returns_noop(self, disabled_tracer):
        """Disabled tracer returns no-op span."""
        span = disabled_tracer.span("test_op", TraceCategory.EVENT_STORE)
        assert isinstance(span, _NoOpSpanContext)

    def test_tracer_span_captures_duration(self, tracer):
        """Span captures duration in milliseconds."""
        with tracer.span("slow_op", TraceCategory.MATERIALIZATION) as span:
            time.sleep(0.01)  # 10ms

        trace = tracer._traces[0]
        assert trace.duration_ms >= 10  # At least 10ms

    def test_tracer_span_captures_error(self, tracer):
        """Span captures exceptions."""
        with pytest.raises(ValueError):
            with tracer.span("error_op", TraceCategory.EVENT_STORE) as span:
                raise ValueError("test error")

        trace = tracer._traces[0]
        assert trace.error is not None
        assert "test error" in trace.error

    def test_tracer_nested_spans(self, tracer):
        """Tracer handles nested spans correctly."""
        with tracer.span("outer", TraceCategory.MATERIALIZATION) as outer:
            outer.set_attribute("level", "outer")
            with tracer.span("inner", TraceCategory.EVENT_STORE) as inner:
                inner.set_attribute("level", "inner")

        assert len(tracer._traces) == 2

    def test_tracer_attributes(self, tracer):
        """Span attributes are recorded."""
        with tracer.span("attr_op", TraceCategory.SEMANTIC_INDEX) as span:
            span.set_attribute("string", "value")
            span.set_attribute("number", 42)
            span.set_attribute("boolean", True)
            span.set_attribute("list", [1, 2, 3])

        trace = tracer._traces[0]
        assert len(trace.attributes) == 4


# =============================================================================
# TEST: TRACE CONTEXT
# =============================================================================

class TestTraceContext:
    """Tests for distributed tracing context."""

    def test_trace_context_creation(self):
        """TraceContext can be created."""
        ctx = TraceContext(trace_id="abc123", span_id="def456")
        assert ctx.trace_id == "abc123"
        assert ctx.span_id == "def456"

    def test_trace_context_to_headers(self):
        """TraceContext generates propagation headers."""
        ctx = TraceContext(trace_id="abc123", span_id="def456")
        headers = ctx.to_headers()
        assert headers["X-Trace-ID"] == "abc123"
        assert headers["X-Span-ID"] == "def456"

    def test_trace_context_child_context(self):
        """TraceContext can create child contexts."""
        parent = TraceContext(trace_id="abc123", span_id="def456")
        child = parent.child_context("child789")
        assert child.trace_id == "abc123"  # Same trace
        assert child.span_id == "child789"  # New span


# =============================================================================
# TEST: TRACING INTEGRATION
# =============================================================================

class TestTracedMethod:
    """Tests for the traced_method decorator."""

    def test_traced_method_records_trace(self, tracer):
        """traced_method decorator records traces."""

        class MockService:
            def __init__(self, tracer):
                self._tracer = tracer

            @traced_method(TraceCategory.EVENT_STORE)
            def do_work(self, data: str) -> str:
                return f"processed: {data}"

        service = MockService(tracer)
        result = service.do_work("test")

        assert result == "processed: test"
        assert len(tracer._traces) == 1
        assert "MockService.do_work" in tracer._traces[0].operation

    def test_traced_method_skips_when_no_tracer(self):
        """traced_method works when _tracer is None."""

        class MockService:
            @traced_method(TraceCategory.EVENT_STORE)
            def do_work(self, data: str) -> str:
                return f"processed: {data}"

        service = MockService()
        result = service.do_work("test")
        assert result == "processed: test"

    def test_traced_method_skips_when_disabled(self, disabled_tracer):
        """traced_method skips tracing when disabled."""

        class MockService:
            def __init__(self, tracer):
                self._tracer = tracer

            @traced_method(TraceCategory.EVENT_STORE)
            def do_work(self, data: str) -> str:
                return f"processed: {data}"

        service = MockService(disabled_tracer)
        result = service.do_work("test")

        assert result == "processed: test"
        assert len(disabled_tracer._traces) == 0


class TestTraceOperation:
    """Tests for the trace_operation decorator."""

    def test_trace_operation_records_trace(self, tracer):
        """trace_operation decorator records traces."""

        @trace_operation(tracer, "my_operation", TraceCategory.USER_CODE)
        def my_func(x: int) -> int:
            return x * 2

        result = my_func(21)
        assert result == 42
        assert len(tracer._traces) == 1
        assert tracer._traces[0].operation == "my_operation"

    def test_trace_operation_skips_when_disabled(self, disabled_tracer):
        """trace_operation skips when tracing disabled."""

        @trace_operation(disabled_tracer, "my_op", TraceCategory.USER_CODE)
        def my_func() -> int:
            return 42

        result = my_func()
        assert result == 42
        assert len(disabled_tracer._traces) == 0


# =============================================================================
# TEST: TRACED WRAPPERS
# =============================================================================

class TestTracedEventStore:
    """Tests for TracedEventStore wrapper."""

    def test_traced_event_store_delegates(self, tracer):
        """TracedEventStore delegates to underlying store."""
        mock_store = MagicMock()
        mock_store.append.return_value = "event123"

        traced = TracedEventStore(mock_store, tracer)
        result = traced.append(MagicMock())

        assert result == "event123"
        mock_store.append.assert_called_once()

    def test_traced_event_store_traces_operations(self, tracer):
        """TracedEventStore records traces."""
        mock_store = MagicMock()
        mock_store.get.return_value = None

        traced = TracedEventStore(mock_store, tracer)
        traced.get("event123")

        assert len(tracer._traces) == 1

    def test_traced_event_store_forwards_attributes(self, tracer):
        """TracedEventStore forwards unknown attributes."""
        mock_store = MagicMock()
        mock_store.custom_attr = "custom_value"

        traced = TracedEventStore(mock_store, tracer)
        assert traced.custom_attr == "custom_value"


class TestTracedMaterializer:
    """Tests for TracedMaterializer wrapper."""

    def test_traced_materializer_delegates(self, tracer):
        """TracedMaterializer delegates to underlying materializer."""
        mock_mat = MagicMock()
        mock_mat.materialize.return_value = {"id": "task1", "status": "done"}

        traced = TracedMaterializer(mock_mat, tracer)
        result = traced.materialize("task1", "task")

        assert result["status"] == "done"
        mock_mat.materialize.assert_called_once()


class TestTracedSemanticIndex:
    """Tests for TracedSemanticIndex wrapper."""

    def test_traced_semantic_index_delegates(self, tracer):
        """TracedSemanticIndex delegates search."""
        mock_index = MagicMock()
        mock_index.search.return_value = ["event1", "event2"]

        traced = TracedSemanticIndex(mock_index, tracer)
        result = traced.search("concept")

        assert result == ["event1", "event2"]
        assert len(tracer._traces) == 1


class TestTracedCausalDAG:
    """Tests for TracedCausalDAG wrapper."""

    def test_traced_dag_delegates(self, tracer):
        """TracedCausalDAG delegates traversal."""
        mock_dag = MagicMock()
        mock_dag.get_ancestors.return_value = ["parent1", "parent2"]

        traced = TracedCausalDAG(mock_dag, tracer)
        result = traced.get_ancestors("event1")

        assert result == ["parent1", "parent2"]
        assert len(tracer._traces) == 1


class TestTracedHealthMonitor:
    """Tests for TracedHealthMonitor wrapper."""

    def test_traced_health_monitor_delegates(self, tracer):
        """TracedHealthMonitor delegates health check."""
        mock_monitor = MagicMock()
        mock_monitor.check_health.return_value = {"status": "healthy"}

        traced = TracedHealthMonitor(mock_monitor, tracer)
        result = traced.check_health()

        assert result["status"] == "healthy"
        assert len(tracer._traces) == 1


# =============================================================================
# TEST: ANALYSIS UTILITIES
# =============================================================================

class TestTraceAnalysis:
    """Tests for trace analysis utilities."""

    def test_find_slow_operations(self, tracer):
        """find_slow_operations finds traces above percentile."""
        # Create some traces with varying durations
        for i in range(10):
            with tracer.span(f"op_{i}", TraceCategory.EVENT_STORE) as span:
                if i >= 8:
                    time.sleep(0.02)  # Make top 20% slow

        # Find top 10% slowest (should include slow ones)
        slow = find_slow_operations(tracer, percentile=90.0)
        assert len(slow) >= 1

    def test_find_error_chains(self, tracer):
        """find_error_chains finds traces with errors."""
        # Create some traces, one with error
        with tracer.span("success_op", TraceCategory.EVENT_STORE):
            pass

        try:
            with tracer.span("error_op", TraceCategory.MATERIALIZATION):
                raise ValueError("test error")
        except ValueError:
            pass

        errors = find_error_chains(tracer)
        assert len(errors) >= 1

    def test_generate_trace_report(self, tracer):
        """generate_trace_report produces readable output."""
        with tracer.span("test_op", TraceCategory.EVENT_STORE):
            pass

        report = generate_trace_report(tracer)
        assert isinstance(report, str)
        assert len(report) > 0


# =============================================================================
# TEST: CONFIG UTILITIES
# =============================================================================

class TestConfigUtilities:
    """Tests for configuration utilities."""

    def test_utc_now_iso_has_timezone(self):
        """utc_now_iso includes timezone info."""
        ts = utc_now_iso()
        assert "+00:00" in ts or "Z" in ts

    def test_parse_iso_timestamp_utc(self):
        """parse_iso_timestamp parses UTC timestamps."""
        ts = "2025-01-01T12:00:00+00:00"
        dt = parse_iso_timestamp(ts)
        assert dt.tzinfo is not None

    def test_parse_iso_timestamp_zulu(self):
        """parse_iso_timestamp handles Z suffix."""
        ts = "2025-01-01T12:00:00Z"
        dt = parse_iso_timestamp(ts)
        assert dt.tzinfo is not None

    def test_cel_config_defaults(self):
        """CELConfig has sensible defaults."""
        config = CELConfig()
        assert config.max_events_before_compaction > 0
        assert config.bloom_filter_size > 0
        assert 0 < config.bloom_false_positive_rate < 1

    def test_cel_config_validation(self):
        """CELConfig validates parameters."""
        config = CELConfig(bloom_false_positive_rate=0.5)
        config.validate()  # Should pass

        with pytest.raises(ValueError):
            invalid = CELConfig(bloom_false_positive_rate=2.0)
            invalid.validate()


# =============================================================================
# TEST: THREAD SAFETY
# =============================================================================

class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_tracer_thread_safe(self, enabled_config):
        """Tracer handles concurrent spans safely."""
        tracer = CELTracer(config=enabled_config)
        errors = []

        def worker(worker_id: int):
            try:
                for i in range(10):
                    with tracer.span(f"worker_{worker_id}_op_{i}", TraceCategory.EVENT_STORE):
                        time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(tracer._traces) == 40  # 4 workers x 10 ops
