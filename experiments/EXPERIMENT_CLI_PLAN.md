# Experiment CLI and Management System Plan

**Status**: Implemented
**Created**: 2026-01-14
**Updated**: 2026-01-14
**Branch**: claude/review-attention-graph-icHzv

## Overview

A command-line interface and experiment management system for running, tracking, and comparing AttentionGraph training experiments with configurable hyperparameters.

## Motivation

As we scale up experiments (more dimensions, attention heads, different architectures), we need:
1. Reproducible experiments via CLI parameters
2. Persistent logging of results for comparison
3. Easy parameter sweeps without code changes
4. Recovery of promising configurations

## MVP Scope

### Phase 1: CLI Runner (Priority 1)

**File**: `cortical/experiments/cli.py`

```python
# Core CLI with argparse
python -m cortical.experiments.cli run \
  --embedding-dim 32 \
  --num-heads 4 \
  --num-layers 2 \
  --epochs 500 \
  --lr 0.03 \
  --input samples/unix_evolution.txt \
  --name "my-experiment"
```

**Required parameters:**
- `--input`: Input text file path
- `--name`: Experiment name (used in output directory)

**Hyperparameters with defaults:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embedding-dim` | 16 | Embedding dimension |
| `--num-heads` | 1 | Number of attention heads |
| `--num-layers` | 2 | Number of attention layers |
| `--epochs` | 500 | Training epochs |
| `--lr` | 0.03 | Learning rate |
| `--clip-grad` | 1.0 | Gradient clipping max norm |
| `--max-tokens` | 50 | Maximum tokens to use |
| `--seed` | 42 | Random seed |

### Phase 2: Experiment Config (Priority 1)

**File**: `cortical/experiments/config.py`

```python
@dataclass
class ExperimentConfig:
    # Required
    name: str
    input_path: str

    # Model architecture
    embedding_dim: int = 16
    num_heads: int = 1
    num_layers: int = 2

    # Training
    epochs: int = 500
    lr: float = 0.03
    clip_grad: float = 1.0
    max_tokens: int = 50
    seed: int = 42

    # Optional/Future
    dropout: float = 0.0
    use_bias: bool = False
    loss_fn: str = "mse"  # "mse" or "cross_entropy"
    position_encoding: str = "none"  # "none", "learned", "sinusoidal"

    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig": ...

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExperimentConfig": ...
```

### Phase 3: Experiment Logging (Priority 1)

**File**: `cortical/experiments/logging.py`

**Output structure:**
```
experiments/
  runs/
    {date}_{name}/
      config.json      # Full configuration
      metrics.json     # Training metrics
      summary.txt      # Human-readable summary
```

**metrics.json structure:**
```json
{
  "train_losses": [1.0, 0.9, ...],
  "accuracies": [0.55, 0.60, ...],
  "gradient_norms": [...],
  "final_accuracy": 0.959,
  "final_loss": 1.80,
  "training_time_seconds": 12.5,
  "git_commit": "abc123",
  "timestamp": "2026-01-14T10:30:00"
}
```

### Phase 4: Compare Command (Priority 2)

```bash
python -m cortical.experiments.cli compare \
  experiments/runs/2026-01-14_baseline \
  experiments/runs/2026-01-14_larger
```

Output:
```
| Experiment | Embed | Heads | Accuracy | Loss  |
|------------|-------|-------|----------|-------|
| baseline   | 16    | 2     | 95.9%    | 1.80  |
| larger     | 32    | 4     | 98.0%    | 1.20  |
```

## Deferred Features (Wishlist)

These are NOT part of MVP but documented for future reference:

1. **Checkpointing** - Save/load model weights
2. **Resume training** - Continue from checkpoint
3. **Parameter sweeps** - Grid/random search over hyperparameters
4. **TensorBoard integration** - Visual training curves
5. **Auto-naming** - Generate experiment names from config hash
6. **Validation split** - Train/val metrics during training
7. **Early stopping** - Stop when validation loss plateaus
8. **Learning rate scheduling** - Decay, warmup, cosine annealing

## Architecture Decisions

### Decision 1: JSON for logging (not database)
**WHY**: Simple, human-readable, git-friendly, no dependencies.
**TRADEOFF**: Harder to query across many experiments.

### Decision 2: Flat config (not nested)
**WHY**: Easier CLI mapping, simpler serialization.
**TRADEOFF**: May need restructuring for complex architectures.

### Decision 3: Directory per experiment (not single file)
**WHY**: Allows adding artifacts (checkpoints, plots) later.
**TRADEOFF**: More filesystem overhead.

## Escape Clauses for Agents

**IMPORTANT**: These escape clauses allow agents to defer complex implementations rather than create weak or incomplete code.

### When to Use Escape Clauses

Use a TODO comment instead of implementing when:
1. Implementation would take >50% of remaining context
2. Feature requires research or external dependencies
3. Implementation would be incomplete or hacky
4. Feature is marked as "Wishlist" above

### Escape Clause Patterns

**Pattern 1: Deferred Implementation**
```python
def load_checkpoint(self, path: str) -> None:
    # TODO(agent): Implement checkpoint loading
    # SESSION_HANDOFF: Requires numpy save/load for all parameters
    # CONTEXT: This is wishlist item, not MVP
    raise NotImplementedError("Checkpoint loading not yet implemented")
```

**Pattern 2: Stub with Warning**
```python
def compare_experiments(paths: List[str]) -> str:
    # TODO(agent): Implement rich comparison table
    # MINIMAL_IMPL: Basic version below, enhance later
    if len(paths) < 2:
        return "Need at least 2 experiments to compare"
    # Basic implementation that works but isn't pretty
    ...
```

**Pattern 3: Config Placeholder**
```python
@dataclass
class ExperimentConfig:
    # MVP parameters
    embedding_dim: int = 16
    ...

    # TODO(agent): Add when implementing respective features
    # position_encoding: str = "none"  # Requires PositionEncoder class
    # lr_schedule: str = "constant"    # Requires LRScheduler class
```

**Pattern 4: CLI Argument Placeholder**
```python
# MVP arguments
parser.add_argument("--epochs", type=int, default=500)
...

# TODO(agent): Add these when features are implemented
# parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
# parser.add_argument("--early-stop", type=int, help="Early stopping patience")
```

### Required TODO Format

All escape clause TODOs MUST include:
1. `TODO(agent):` prefix for searchability
2. Brief description of what's needed
3. One of: `SESSION_HANDOFF:`, `MINIMAL_IMPL:`, `CONTEXT:`, `BLOCKED_BY:`

Example:
```python
# TODO(agent): Implement sinusoidal position encoding
# SESSION_HANDOFF: See "Attention Is All You Need" paper, Section 3.5
# BLOCKED_BY: Need to decide if positions are absolute or relative
```

## Implementation Order

1. `config.py` - ExperimentConfig dataclass
2. `logging.py` - ExperimentLog class with JSON save
3. `cli.py` - Main CLI with `run` command
4. `cli.py` - Add `compare` command (can be stub initially)

## Success Criteria

MVP is complete when:
- [x] `python -m cortical.experiments.cli run --input samples/unix_evolution.txt --name test` works
- [x] Config is saved to `experiments/runs/{date}_test/config.json`
- [x] Metrics are saved to `experiments/runs/{date}_test/metrics.json`
- [x] All hyperparameters can be overridden via CLI
- [x] Experiments are reproducible with same seed

Additional features implemented:
- [x] Cross-entropy loss with vocabulary projection (`--loss-fn cross_entropy`)
- [x] Checkpoint saving/loading using pickle format
- [x] Compare command for experiment comparison
- [x] List command to show all experiments

## Testing Strategy

1. Unit tests for ExperimentConfig serialization
2. Unit tests for ExperimentLog file operations
3. Integration test: run CLI, verify output files exist
4. Reproducibility test: same seed produces same results

## Notes

- Integrates with existing `ExperimentKernel` from `cortical/experiments/kernel.py`
- Uses existing tokenizer from `cortical/experiments/tokenizer.py`
- Uses existing profiler from `cortical/experiments/profiler.py`
- New `cortical/experiments/projection.py` provides `VocabProjection` and `CrossEntropyWithLogits` for language modeling
- Checkpoint saving uses pickle format (see `cortical/experiments/logging.py`)
