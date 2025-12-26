# Woven Mind User Guide

A dual-process cognitive architecture for intelligent text processing.

## Overview

Woven Mind is a biologically-inspired cognitive architecture that implements dual-process theory:

- **System 1 (Hive)**: Fast, automatic pattern matching
- **System 2 (Cortex)**: Slow, deliberate reasoning
- **The Loom**: Orchestrates mode switching based on surprise detection
- **Consolidation**: "Sleep-like" cycles for learning transfer

## Quick Start

```python
from cortical.reasoning.woven_mind import WovenMind

# Create a WovenMind instance
mind = WovenMind()

# Train on text
mind.train("neural networks process information")
mind.train("deep learning uses neural networks")

# Process input and get result
result = mind.process(["neural", "networks"])

print(f"Mode: {result.mode.name}")  # FAST or SLOW
print(f"Activations: {result.activations}")
print(f"Source: {result.source}")  # 'hive' or 'cortex'
```

## Core Concepts

### Dual-Process Architecture

```
┌─────────────────────────────────────────┐
│              CORTEX (System 2)          │
│         Slow, deliberate reasoning      │
│     - Explicit abstraction formation    │
│     - Goal-directed planning            │
│     - Pattern analysis                  │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│              THE LOOM                    │
│         Mode switching & routing         │
│     - Surprise detection                 │
│     - Attention routing                  │
│     - Mode transitions                   │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────┴───────────────────────┐
│              HIVE (System 1)            │
│         Fast, automatic processing      │
│     - Pattern matching                  │
│     - Prediction generation             │
│     - Spreading activation              │
└─────────────────────────────────────────┘
```

### Mode Switching

The Loom monitors **surprise** - the difference between predictions and actual input:

- **Low surprise**: Stay in FAST mode (Hive handles it)
- **High surprise**: Switch to SLOW mode (engage Cortex)

```python
from cortical.reasoning.loom import ThinkingMode

# Process with automatic mode selection
result = mind.process(["input", "tokens"])
print(f"Auto-selected mode: {result.mode.name}")

# Force a specific mode
result = mind.process(["input", "tokens"], mode=ThinkingMode.SLOW)
print(f"Forced SLOW mode: {result.mode.name}")

# Get current mode
current = mind.get_current_mode()
```

### Training

Training builds the Hive's prediction capabilities:

```python
# Train on individual texts
mind.train("machine learning algorithms")
mind.train("deep learning processes data")

# Observe patterns for abstraction
mind.observe_pattern(["neural", "network"])
```

## Configuration

### WovenMindConfig

```python
from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig

config = WovenMindConfig(
    # Surprise threshold for mode switching (0.0-1.0)
    surprise_threshold=0.3,

    # K-winners-take-all sparsity
    k_winners=5,

    # Minimum frequency for abstraction
    min_frequency=3,

    # Consolidation settings
    consolidation_threshold=3,
    consolidation_decay_factor=0.9,

    # Enable auto-consolidation during processing
    enable_auto_consolidation=True,
)

mind = WovenMind(config=config)
```

## Consolidation (Learning Transfer)

Consolidation implements "sleep-like" cycles that:
1. Transfer frequent Hive patterns to Cortex abstractions
2. Mine latent structure in patterns
3. Apply decay to unused connections

```python
from cortical.reasoning.consolidation import ConsolidationResult

# Run a consolidation cycle
result = mind.consolidate()

print(f"Patterns transferred: {result.patterns_transferred}")
print(f"Abstractions formed: {result.abstractions_formed}")
print(f"Connections decayed: {result.connections_decayed}")
print(f"Duration: {result.cycle_duration_ms}ms")
```

### Scheduled Consolidation

For long-running applications:

```python
# Start periodic consolidation (every 300 seconds)
mind.consolidation.start_scheduler(interval_seconds=300)

# Stop when done
mind.consolidation.stop_scheduler()
```

### Recording Patterns

Patterns are automatically recorded during processing if `enable_auto_consolidation` is True. You can also record manually:

```python
# Manual pattern recording
mind.consolidation.record_pattern({"term1", "term2", "term3"})

# Get frequent patterns
patterns = mind.consolidation.get_frequent_patterns(min_frequency=3)
```

## Introspection

### Statistics

```python
# Full system statistics
stats = mind.get_stats()
print(stats)
# {
#     'mode': 'FAST',
#     'hive': {...},
#     'cortex': {...},
#     'loom': {...},
#     'consolidation': {...}
# }

# Consolidation-specific stats
consolidation_stats = mind.get_consolidation_stats()
print(f"Total cycles: {consolidation_stats['total_cycles']}")
print(f"Patterns transferred: {consolidation_stats['total_patterns_transferred']}")
```

### Surprise Baseline

```python
# Get current surprise baseline
baseline = mind.get_surprise_baseline()
print(f"Surprise baseline: {baseline}")  # 0.0 to 1.0
```

### Transition History

```python
# Get mode transition history
history = mind.get_transition_history()
for transition in history:
    print(f"{transition['from']} -> {transition['to']}: {transition['reason']}")
```

## Serialization

Save and restore WovenMind state:

```python
# Save to dictionary
data = mind.to_dict()

# Restore from dictionary
restored = WovenMind.from_dict(data)

# Save to JSON file
import json
with open("woven_mind_state.json", "w") as f:
    json.dump(mind.to_dict(), f)

# Load from JSON file
with open("woven_mind_state.json", "r") as f:
    data = json.load(f)
    mind = WovenMind.from_dict(data)
```

## Components

### Direct Component Access

```python
# Access Hive (System 1)
predictions = mind.hive.generate_predictions(["context"])
activations = mind.hive.activate(["tokens"])

# Access Cortex (System 2)
abstractions = mind.cortex.get_abstractions()

# Access Loom
current_mode = mind.loom.get_current_mode()

# Access Router
routing = mind.router.route(["context"], mode=None)
```

### Providing Custom Components

```python
from cortical.reasoning.loom import Loom
from cortical.reasoning.loom_hive import LoomHiveConnector
from cortical.reasoning.loom_cortex import LoomCortexConnector

# Create custom components
custom_hive = LoomHiveConnector()
custom_cortex = LoomCortexConnector()
custom_loom = Loom()

# Use custom components
mind = WovenMind(
    loom=custom_loom,
    hive=custom_hive,
    cortex=custom_cortex,
)
```

## WovenMindResult

Processing returns a `WovenMindResult` dataclass:

```python
@dataclass
class WovenMindResult:
    mode: ThinkingMode       # FAST or SLOW
    activations: Set[str]    # Activated terms
    surprise: Optional[float]  # Surprise level (0.0-1.0)
    predictions: Dict[str, float]  # Prediction probabilities
    source: str              # 'hive' or 'cortex'
    metadata: Dict[str, Any]  # Additional metadata
```

## Best Practices

### 1. Train Before Processing

Always train on representative data before expecting meaningful processing:

```python
# Good: Train first
mind = WovenMind()
mind.train("your domain-specific content here")
result = mind.process(["domain", "query"])

# Bad: Processing without training
mind = WovenMind()
result = mind.process(["domain", "query"])  # Limited predictions
```

### 2. Periodic Consolidation

For best learning retention, consolidate periodically:

```python
# After training session
for doc in training_docs:
    mind.train(doc)
    mind.process(doc.split()[:5])

# Consolidate to transfer patterns
mind.consolidate()
```

### 3. Use Explicit Mode for Critical Tasks

When you need specific reasoning behavior:

```python
# Force SLOW mode for important decisions
from cortical.reasoning.loom import ThinkingMode
result = mind.process(["critical", "decision"], mode=ThinkingMode.SLOW)
```

### 4. Monitor Surprise Baseline

Track surprise baseline for system health:

```python
baseline = mind.get_surprise_baseline()
if baseline < 0.1:
    print("System may be over-fitted to training data")
elif baseline > 0.7:
    print("System encountering many novel inputs")
```

### 5. Save State Regularly

Persist state to avoid data loss:

```python
import json

# Save after significant training
with open("checkpoint.json", "w") as f:
    json.dump(mind.to_dict(), f)
```

## Performance Tips

1. **Batch training**: Train on multiple documents before processing
2. **Limit consolidation frequency**: Every 5-10 minutes, not every second
3. **Use appropriate k_winners**: Lower values = faster but less accurate
4. **Monitor memory**: Large corpora increase memory usage

## Error Handling

```python
try:
    result = mind.process(["input"])
except Exception as e:
    print(f"Processing error: {e}")
    # Consider resetting or reloading state
    mind.reset()
```

## API Reference

### WovenMind

| Method | Description |
|--------|-------------|
| `train(text)` | Train on text |
| `process(context, mode=None)` | Process input |
| `observe_pattern(pattern)` | Record pattern for abstraction |
| `consolidate()` | Run consolidation cycle |
| `get_current_mode()` | Get current thinking mode |
| `force_mode(mode, reason)` | Force specific mode |
| `get_stats()` | Get full statistics |
| `get_surprise_baseline()` | Get surprise baseline |
| `get_transition_history()` | Get mode transitions |
| `get_consolidation_stats()` | Get consolidation stats |
| `reset()` | Clear all state |
| `to_dict()` | Serialize to dict |
| `from_dict(data)` | Deserialize from dict |

### ConsolidationEngine

| Method | Description |
|--------|-------------|
| `consolidate()` | Run full cycle |
| `pattern_transfer()` | Transfer Hive patterns |
| `abstraction_mining()` | Mine abstractions |
| `decay_cycle()` | Apply decay |
| `record_pattern(pattern)` | Record pattern |
| `get_frequent_patterns(min_freq)` | Get frequent patterns |
| `start_scheduler(interval)` | Start periodic consolidation |
| `stop_scheduler()` | Stop scheduler |
| `get_stats()` | Get statistics |
| `get_history(limit)` | Get cycle history |

## See Also

- [Roadmap: Woven Mind + PRISM Marriage](roadmap-woven-prism-marriage.md)
- [Architecture Documentation](architecture.md)
- [Graph of Thought](graph-of-thought.md)
