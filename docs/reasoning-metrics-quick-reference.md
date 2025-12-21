# Reasoning Metrics - Quick Reference

## Installation

The metrics module is already integrated into `cortical.reasoning`:

```python
from cortical.reasoning import (
    ReasoningMetrics,
    PhaseMetrics,
    MetricsContextManager,
    create_loop_metrics_handler,
)
```

## Quick Start

### 1. Basic Usage

```python
from cortical.reasoning import ReasoningMetrics, LoopPhase

metrics = ReasoningMetrics()

# Time a phase
with metrics.phase_timer(LoopPhase.QUESTION):
    # Do work here
    pass

# Record events
metrics.record_decision("architecture")
metrics.record_question("technical")
metrics.record_verification(passed=True)

# Get summary
print(metrics.get_summary())
```

### 2. Automatic Loop Integration

```python
from cortical.reasoning import (
    CognitiveLoopManager,
    ReasoningMetrics,
    create_loop_metrics_handler,
)

metrics = ReasoningMetrics()
manager = CognitiveLoopManager()

# Register handler for automatic tracking
handler = create_loop_metrics_handler(metrics)
manager.register_transition_handler(handler)

# Now all loops are automatically tracked
loop = manager.create_loop("Implement feature")
loop.start()  # Automatically recorded
loop.transition(LoopPhase.ANSWER, reason="Done")  # Automatically timed
```

## API Reference

### ReasoningMetrics

| Method | Description |
|--------|-------------|
| `record_phase_transition(from_phase, to_phase, duration_ms)` | Track phase changes |
| `record_decision(decision_type)` | Log decision by type |
| `record_question(category)` | Log question by category |
| `record_production(artifact_type)` | Track artifact creation |
| `record_verification(passed, level)` | Track verification result |
| `record_crisis(recovered, level)` | Track crisis event |
| `record_loop_start()` | Mark loop started |
| `record_loop_complete(success)` | Mark loop completed |
| `get_verification_pass_rate()` | Get pass rate % |
| `get_crisis_recovery_rate()` | Get recovery rate % |
| `get_loop_completion_rate()` | Get completion rate % |
| `get_summary()` | Human-readable report |
| `get_metrics_dict()` | Export to dict |
| `reset()` | Clear all metrics |
| `enable()` / `disable()` | Toggle collection |
| `phase_timer(phase)` | Context manager for timing |

### PhaseMetrics

| Attribute | Description |
|-----------|-------------|
| `phase_name` | Name of the phase |
| `entry_count` | Times entered |
| `total_duration_ms` | Total time spent |
| `min_duration_ms` | Shortest duration |
| `max_duration_ms` | Longest duration |

| Method | Description |
|--------|-------------|
| `record_entry(duration_ms)` | Add entry |
| `get_average_ms()` | Get average duration |
| `to_dict()` | Export to dict |

## Common Patterns

### Pattern 1: Manual Phase Timing

```python
metrics = ReasoningMetrics()

# Using context manager
with metrics.phase_timer(LoopPhase.PRODUCE):
    write_code()
    write_tests()
```

### Pattern 2: Automatic Loop Tracking

```python
# Setup once
metrics = ReasoningMetrics()
manager = CognitiveLoopManager()
manager.register_transition_handler(
    create_loop_metrics_handler(metrics)
)

# Use normally - metrics collected automatically
loop = manager.create_loop("Build feature")
loop.start()
# ... work ...
loop.transition(LoopPhase.VERIFY, reason="Code complete")
```

### Pattern 3: Integration with Verification

```python
metrics = ReasoningMetrics()
vm = VerificationManager()

suite = vm.create_suite("unit_tests")
for check in suite.checks:
    run_check(check)
    passed = (check.status == VerificationStatus.PASSED)
    metrics.record_verification(
        passed=passed,
        level=check.level.name.lower()
    )

print(f"Pass rate: {metrics.get_verification_pass_rate():.1f}%")
```

### Pattern 4: Integration with Crisis Manager

```python
metrics = ReasoningMetrics()
cm = CrisisManager()

crisis = cm.record_crisis(
    CrisisLevel.OBSTACLE,
    "Tests failing repeatedly",
    context={'attempts': 3}
)

# Attempt recovery
recovered = attempt_recovery(crisis)
crisis.resolve(
    RecoveryAction.ADAPT if recovered else RecoveryAction.ESCALATE,
    "Changed approach"
)

metrics.record_crisis(recovered=recovered, level="obstacle")
print(f"Recovery rate: {metrics.get_crisis_recovery_rate():.1f}%")
```

### Pattern 5: Export to Observability Format

```python
# Get metrics in observability.py-compatible format
metrics_dict = metrics.get_metrics_dict()

# Can be merged with processor metrics
from cortical.observability import MetricsCollector
processor_metrics = MetricsCollector()
# ... use processor ...

# Combine metrics for unified reporting
all_metrics = {**processor_metrics.get_all_stats(), **metrics_dict}
```

## Metrics Output Format

### Summary Format

```
Reasoning Metrics Summary
================================================================================

Phase Transitions:
Phase              Count    Avg(ms)    Min(ms)    Max(ms)    Total(ms)
--------------------------------------------------------------------------------
question               3     150.25     120.10     180.50       450.75
answer                 3     200.50     180.25     220.75       601.50
produce                2     500.25     450.10     550.40      1000.50
verify                 2     100.50      95.25     105.75       201.00

Production Metrics:
  Decisions made: 12
    - architecture: 5
    - implementation: 4
    - design: 3
  Questions asked: 8
  Artifacts produced: 15

Verification Metrics:
  Passed: 45
  Failed: 5
  Total: 50
  Pass rate: 90.0%

Crisis Metrics:
  Detected: 3
  Recovered: 2
  Recovery rate: 66.7%

Loop Lifecycle:
  Started: 5
  Completed: 4
  Aborted: 1
  Completion rate: 80.0%
```

### Dictionary Format

```python
{
    'phase_question': {
        'count': 3,
        'total_ms': 450.75,
        'avg_ms': 150.25,
        'min_ms': 120.10,
        'max_ms': 180.50
    },
    'decisions_made': {'count': 12},
    'questions_asked': {'count': 8},
    'productions_created': {'count': 15},
    'verifications_passed': {'count': 45},
    'verifications_failed': {'count': 5},
    'verification_pass_rate': {'value': 90.0},
    'crises_detected': {'count': 3},
    'crises_recovered': {'count': 2},
    'crisis_recovery_rate': {'value': 66.7},
    'loops_started': {'count': 5},
    'loops_completed': {'count': 4},
    'loops_aborted': {'count': 1},
    'loop_completion_rate': {'value': 80.0}
}
```

## Best Practices

1. **Use automatic integration** - Register handler once, track automatically
2. **Disable when not needed** - `metrics.disable()` for zero overhead
3. **Reset between sessions** - `metrics.reset()` for fresh metrics
4. **Check rates regularly** - Monitor pass/recovery/completion rates
5. **Export for analysis** - Use `get_metrics_dict()` for historical tracking

## Examples

See `/home/user/Opus-code-test/examples/reasoning_metrics_demo.py` for comprehensive demonstrations.

## Testing

Run tests:
```bash
python -m pytest tests/unit/test_reasoning_metrics.py -v
```

Coverage:
```bash
python -m coverage run --source=cortical/reasoning -m pytest tests/unit/test_reasoning_metrics.py
python -m coverage report --include="cortical/reasoning/metrics.py"
```

## Performance

- Minimal overhead when enabled (~1-2% for typical workloads)
- Zero overhead when disabled
- No external dependencies
- Memory-efficient (no history retention by default)
- Thread-safe within single collector instance
