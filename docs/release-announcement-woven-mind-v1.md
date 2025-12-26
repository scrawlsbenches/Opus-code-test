# Woven Mind v1.0: A Dual-Process Cognitive Architecture

**Release Date:** December 2025
**Version:** 1.0.0
**Status:** Production Ready

---

## Announcing Woven Mind

We are excited to announce the release of **Woven Mind**, a biologically-inspired dual-process cognitive architecture for intelligent text processing. Woven Mind implements dual-process theory from cognitive science, enabling systems that combine fast intuitive processing with deliberate analytical reasoning.

## What is Woven Mind?

Woven Mind is a cognitive architecture that mimics how the human brain processes information using two complementary systems:

```
┌─────────────────────────────────────────────────────────────┐
│                     WOVEN MIND                               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              CORTEX (System 2)                       │   │
│   │         Slow, deliberate reasoning                   │   │
│   │     • Explicit abstraction formation                 │   │
│   │     • Goal-directed planning                         │   │
│   │     • Pattern analysis                               │   │
│   └─────────────────────┬───────────────────────────────┘   │
│                         │                                     │
│   ┌─────────────────────┴───────────────────────────────┐   │
│   │              THE LOOM                                │   │
│   │         Mode switching & routing                     │   │
│   │     • Surprise detection                             │   │
│   │     • Attention routing                              │   │
│   │     • Adaptive transitions                           │   │
│   └─────────────────────┬───────────────────────────────┘   │
│                         │                                     │
│   ┌─────────────────────┴───────────────────────────────┐   │
│   │              HIVE (System 1)                         │   │
│   │         Fast, automatic processing                   │   │
│   │     • Pattern matching                               │   │
│   │     • Prediction generation                          │   │
│   │     • Spreading activation                           │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │         CONSOLIDATION ENGINE                         │   │
│   │     • Sleep-like memory transfer                     │   │
│   │     • Abstraction mining                             │   │
│   │     • Adaptive decay                                 │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

### Dual-Process Architecture
- **System 1 (Hive)**: Fast, automatic pattern matching for familiar inputs
- **System 2 (Cortex)**: Slow, deliberate reasoning for novel or complex situations
- **The Loom**: Intelligent orchestrator that routes between systems based on surprise detection

### Surprise-Based Mode Switching
The Loom monitors the difference between predictions and actual input:
- **Low surprise** → Stay in FAST mode (Hive handles it)
- **High surprise** → Switch to SLOW mode (engage Cortex)

### Consolidation Engine (Sleep-Like Learning)
Periodic consolidation cycles that:
- Transfer frequent patterns from Hive to Cortex abstractions
- Mine latent structure across patterns
- Apply decay to unused connections
- Support both manual and scheduled consolidation

### Homeostatic Regulation
- Maintains stable activity levels across components
- Prevents runaway activation or collapse
- Adaptive gain control based on activity history

### Full State Persistence
- Serialize/deserialize complete system state
- Resume learning from checkpoints
- JSON-based portable format

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Training throughput | 100+ documents/second |
| Processing latency | < 5ms per query |
| Consolidation cycle | < 100ms |
| Memory efficiency | < 2x growth after heavy use |
| Serialization | < 500ms for large states |

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

print(f"Mode: {result.mode.name}")       # FAST or SLOW
print(f"Activations: {result.activations}")
print(f"Source: {result.source}")        # 'hive' or 'cortex'

# Run consolidation cycle (like sleep)
consolidation_result = mind.consolidate()
print(f"Patterns transferred: {consolidation_result.patterns_transferred}")
```

## Configuration

```python
from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig

config = WovenMindConfig(
    # Surprise threshold for mode switching (0.0-1.0)
    surprise_threshold=0.3,

    # K-winners-take-all sparsity
    k_winners=5,

    # Consolidation settings
    consolidation_threshold=3,
    consolidation_decay_factor=0.9,

    # Enable auto-consolidation during processing
    enable_auto_consolidation=True,
)

mind = WovenMind(config=config)
```

## Development History

Woven Mind was developed over 6 sprints as part of the Woven Mind + PRISM Marriage project:

| Sprint | Focus | Status |
|--------|-------|--------|
| Sprint 1 | Foundation & Loom | Complete |
| Sprint 2 | Hive (System 1) | Complete |
| Sprint 3 | Cortex (System 2) | Complete |
| Sprint 4 | Router & Integration | Complete |
| Sprint 5 | Consolidation Engine | Complete |
| Sprint 6 | Integration & Polish | Complete |

### Sprint Highlights

**Sprint 1: Foundation**
- Core Loom orchestrator with mode switching
- ThinkingMode enum (FAST/SLOW)
- Transition tracking and history

**Sprint 2: Hive (System 1)**
- Sparse Distributed Memory for fast pattern matching
- Prediction generation from learned patterns
- Lateral activation spreading

**Sprint 3: Cortex (System 2)**
- AbstractionManager for concept formation
- Planning and goal-directed processing
- Explicit reasoning capabilities

**Sprint 4: Router & Integration**
- ModeRouter for intelligent mode selection
- Surprise-based adaptive routing
- Full WovenMind facade integration

**Sprint 5: Consolidation Engine**
- ConsolidationEngine for memory transfer
- Pattern frequency tracking
- Abstraction mining from frequent patterns
- Decay cycles for unused connections
- Scheduled consolidation support

**Sprint 6: Integration & Polish**
- Comprehensive E2E test suite (25 tests)
- Performance benchmarks (19 tests)
- User documentation and tutorials
- Interactive demo application

## Test Coverage

- **230+ tests** covering all components
- **92-99% coverage** on core modules
- **Performance benchmarks** with timing assertions
- **Edge case handling** for empty inputs, unicode, long texts

## Documentation

| Resource | Description |
|----------|-------------|
| [User Guide](woven-mind-user-guide.md) | Complete API documentation |
| [Tutorial Notebook](../examples/woven_mind_tutorial.ipynb) | Interactive learning |
| [Demo Script](../examples/woven_mind_demo.py) | Runnable demonstrations |
| [Roadmap](roadmap-woven-prism-marriage.md) | Development plan details |

## Use Cases

### Adaptive Text Classification
Use Woven Mind to classify text with automatic complexity adaptation:
- Familiar patterns processed quickly by Hive
- Novel or ambiguous inputs routed to Cortex for deeper analysis

### Intelligent Chatbots
Build chatbots that know when to give quick responses vs. when to think carefully:
- Common queries handled instantly
- Complex questions trigger deliberate reasoning

### Knowledge Extraction
Extract knowledge from text with progressive learning:
- Initial patterns captured by Hive
- Consolidated into abstractions during periodic "sleep" cycles

### Anomaly Detection
Detect unusual patterns through surprise monitoring:
- High surprise indicates novelty
- Track surprise baseline for adaptive thresholds

## Requirements

- Python 3.9+
- Zero external dependencies (pure Python)
- Works on Linux and macOS

## Getting Started

1. **Import the module:**
   ```python
   from cortical.reasoning.woven_mind import WovenMind
   ```

2. **Create an instance:**
   ```python
   mind = WovenMind()
   ```

3. **Train on your data:**
   ```python
   for document in your_corpus:
       mind.train(document)
   ```

4. **Process queries:**
   ```python
   result = mind.process(["your", "query", "tokens"])
   ```

5. **Consolidate periodically:**
   ```python
   mind.consolidate()  # Transfer learning
   ```

## Future Directions

The Woven Mind architecture provides a foundation for several future enhancements:

- **Multi-modal processing**: Extend beyond text to images and audio
- **Hierarchical consolidation**: Multiple levels of abstraction
- **Distributed processing**: Scale across multiple nodes
- **Real-time adaptation**: Online learning without explicit consolidation

## Acknowledgments

Woven Mind draws inspiration from:
- Dual-Process Theory (Kahneman)
- Sparse Distributed Memory (Kanerva)
- Predictive Processing Framework
- Sleep and Memory Consolidation Research

---

**Ready to get started?** Check out the [User Guide](woven-mind-user-guide.md) or run the [interactive demo](../examples/woven_mind_demo.py).

```bash
python examples/woven_mind_demo.py --section all
```
