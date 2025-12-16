---
title: "Observability"
generated: "2025-12-16T17:02:01.491559Z"
generator: "architecture"
source_files:
  - "/home/user/Opus-code-test/cortical/results.py"
  - "/home/user/Opus-code-test/cortical/observability.py"
  - "/home/user/Opus-code-test/cortical/progress.py"
tags:
  - architecture
  - modules
  - observability
---

# Observability

Metrics collection and progress tracking.

## Modules

- **observability.py**: Observability Module
- **progress.py**: Progress reporting infrastructure for long-running operations.
- **results.py**: Result Dataclasses for Cortical Text Processor


## observability.py

Observability Module
====================

Provides timing hooks, metrics collection, and trace context for monitoring
the Cortical Text Processor's performance and operations.

This module follows th...


### Classes

#### MetricsCollector

Collects and aggregates timing and count metrics for operations.

**Methods:**

- `record_timing`
- `record_count`
- `get_operation_stats`
- `get_all_stats`
- `get_trace`
- `reset`
- `enable`
- `disable`
- `trace_context`
- `get_summary`

#### TraceContext

Context for request tracing across operations.

**Methods:**

- `elapsed_ms`

### Functions

#### timed

```python
timed(operation_name: Optional[str] = None, include_args: bool = False)
```

Decorator for timing method calls and recording to metrics.

#### measure_time

```python
measure_time(func: Callable) -> Callable
```

Simple timing decorator that logs execution time.

#### get_global_metrics

```python
get_global_metrics() -> MetricsCollector
```

Get the global metrics collector instance.

#### enable_global_metrics

```python
enable_global_metrics() -> None
```

Enable global metrics collection.

#### disable_global_metrics

```python
disable_global_metrics() -> None
```

Disable global metrics collection.

#### reset_global_metrics

```python
reset_global_metrics() -> None
```

Reset global metrics.

#### MetricsCollector.record_timing

```python
MetricsCollector.record_timing(self, operation: str, duration_ms: float, trace_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> None
```

Record a timing measurement for an operation.

#### MetricsCollector.record_count

```python
MetricsCollector.record_count(self, metric_name: str, count: int = 1) -> None
```

Record a simple count metric.

#### MetricsCollector.get_operation_stats

```python
MetricsCollector.get_operation_stats(self, operation: str) -> Dict[str, Any]
```

Get statistics for a specific operation.

#### MetricsCollector.get_all_stats

```python
MetricsCollector.get_all_stats(self) -> Dict[str, Dict[str, Any]]
```

Get statistics for all operations.

#### MetricsCollector.get_trace

```python
MetricsCollector.get_trace(self, trace_id: str) -> List[tuple]
```

Get all operations recorded for a trace ID.

#### MetricsCollector.reset

```python
MetricsCollector.reset(self) -> None
```

Clear all collected metrics.

#### MetricsCollector.enable

```python
MetricsCollector.enable(self) -> None
```

Enable metrics collection.

#### MetricsCollector.disable

```python
MetricsCollector.disable(self) -> None
```

Disable metrics collection.

#### MetricsCollector.trace_context

```python
MetricsCollector.trace_context(self, trace_id: str)
```

Context manager for tracing a block of operations.

#### MetricsCollector.get_summary

```python
MetricsCollector.get_summary(self) -> str
```

Get a human-readable summary of all metrics.

#### TraceContext.elapsed_ms

```python
TraceContext.elapsed_ms(self) -> float
```

Get elapsed time since trace started in milliseconds.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `contextlib.contextmanager`
- `functools`
- `logging`
- `time`
- ... and 5 more



## progress.py

Progress reporting infrastructure for long-running operations.

This module provides a flexible progress reporting system that supports:
- Console output with nice formatting
- Custom callbacks for in...


### Classes

#### ProgressReporter

Protocol for progress reporters.

**Methods:**

- `update`
- `complete`

#### ConsoleProgressReporter

Console-based progress reporter with nice formatting.

**Methods:**

- `update`
- `complete`

#### CallbackProgressReporter

Progress reporter that calls a custom callback function.

**Methods:**

- `update`
- `complete`

#### SilentProgressReporter

No-op progress reporter for silent operation.

**Methods:**

- `update`
- `complete`

#### MultiPhaseProgress

Helper for tracking progress across multiple sequential phases.

**Methods:**

- `start_phase`
- `update`
- `complete_phase`
- `overall_progress`

### Functions

#### ProgressReporter.update

```python
ProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Update progress for a specific phase.

#### ProgressReporter.complete

```python
ProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Mark a phase as complete.

#### ConsoleProgressReporter.update

```python
ConsoleProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Update progress display.

#### ConsoleProgressReporter.complete

```python
ConsoleProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Mark phase as complete and move to new line.

#### CallbackProgressReporter.update

```python
CallbackProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Call callback with progress update.

#### CallbackProgressReporter.complete

```python
CallbackProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Call callback with completion notification.

#### SilentProgressReporter.update

```python
SilentProgressReporter.update(self, phase: str, percent: float, message: Optional[str] = None) -> None
```

Do nothing.

#### SilentProgressReporter.complete

```python
SilentProgressReporter.complete(self, phase: str, message: Optional[str] = None) -> None
```

Do nothing.

#### MultiPhaseProgress.start_phase

```python
MultiPhaseProgress.start_phase(self, phase: str) -> None
```

Start a new phase.

#### MultiPhaseProgress.update

```python
MultiPhaseProgress.update(self, percent: float, message: Optional[str] = None) -> None
```

Update progress within current phase.

#### MultiPhaseProgress.complete_phase

```python
MultiPhaseProgress.complete_phase(self, message: Optional[str] = None) -> None
```

Mark current phase as complete.

#### MultiPhaseProgress.overall_progress

```python
MultiPhaseProgress.overall_progress(self) -> float
```

Get overall progress across all phases (0-100).

### Dependencies

**Standard Library:**

- `abc.ABC`
- `abc.abstractmethod`
- `sys`
- `time`
- `typing.Any`
- ... and 4 more



## results.py

Result Dataclasses for Cortical Text Processor
===============================================

Strongly-typed result containers for query operations that provide
IDE autocomplete and type checking su...


### Classes

#### DocumentMatch

A document search result with relevance score.

**Methods:**

- `to_dict`
- `to_tuple`
- `from_tuple`
- `from_dict`

#### PassageMatch

A passage retrieval result with text, location, and relevance score.

**Methods:**

- `to_dict`
- `to_tuple`
- `location`
- `length`
- `from_tuple`
- `from_dict`

#### QueryResult

Complete query result with matches and metadata.

**Methods:**

- `to_dict`
- `top_match`
- `match_count`
- `average_score`
- `from_dict`

### Functions

#### convert_document_matches

```python
convert_document_matches(results: List[tuple], metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> List[DocumentMatch]
```

Convert list of (doc_id, score) tuples to DocumentMatch objects.

#### convert_passage_matches

```python
convert_passage_matches(results: List[tuple], metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> List[PassageMatch]
```

Convert list of (doc_id, text, start, end, score) tuples to PassageMatch objects.

#### DocumentMatch.to_dict

```python
DocumentMatch.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

#### DocumentMatch.to_tuple

```python
DocumentMatch.to_tuple(self) -> tuple
```

Convert to tuple format (doc_id, score).

#### DocumentMatch.from_tuple

```python
DocumentMatch.from_tuple(cls, doc_id: str, score: float, metadata: Optional[Dict[str, Any]] = None) -> 'DocumentMatch'
```

Create from tuple format (doc_id, score).

#### DocumentMatch.from_dict

```python
DocumentMatch.from_dict(cls, data: Dict[str, Any]) -> 'DocumentMatch'
```

Create from dictionary.

#### PassageMatch.to_dict

```python
PassageMatch.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

#### PassageMatch.to_tuple

```python
PassageMatch.to_tuple(self) -> tuple
```

Convert to tuple format (doc_id, text, start, end, score).

#### PassageMatch.location

```python
PassageMatch.location(self) -> str
```

Get citation-style location string.

#### PassageMatch.length

```python
PassageMatch.length(self) -> int
```

Get passage length in characters.

#### PassageMatch.from_tuple

```python
PassageMatch.from_tuple(cls, doc_id: str, text: str, start: int, end: int, score: float, metadata: Optional[Dict[str, Any]] = None) -> 'PassageMatch'
```

Create from tuple format (doc_id, text, start, end, score).

#### PassageMatch.from_dict

```python
PassageMatch.from_dict(cls, data: Dict[str, Any]) -> 'PassageMatch'
```

Create from dictionary.

#### QueryResult.to_dict

```python
QueryResult.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary with nested match dicts.

#### QueryResult.top_match

```python
QueryResult.top_match(self) -> Union[DocumentMatch, PassageMatch, None]
```

Get the highest-scoring match.

#### QueryResult.match_count

```python
QueryResult.match_count(self) -> int
```

Get number of matches.

#### QueryResult.average_score

```python
QueryResult.average_score(self) -> float
```

Get average relevance score across all matches.

#### QueryResult.from_dict

```python
QueryResult.from_dict(cls, data: Dict[str, Any]) -> 'QueryResult'
```

Create from dictionary.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `typing.Any`
- `typing.Dict`
- ... and 3 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*
