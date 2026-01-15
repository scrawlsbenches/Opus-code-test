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

1. ~~**Checkpointing**~~ - ✅ Implemented (pickle format)
2. **Resume training** - Continue from checkpoint (see implementation plan below)
3. **Parameter sweeps** - Grid/random search over hyperparameters
4. **TensorBoard integration** - Visual training curves
5. **Auto-naming** - Generate experiment names from config hash
6. ~~**Validation split**~~ - ✅ Implemented (`--val-split` flag)
7. **Early stopping** - Stop when validation loss plateaus (see implementation plan below)
8. **Learning rate scheduling** - Decay, warmup, cosine annealing (see implementation plan below)

---

## Implementation Plans for Remaining Features

### Feature 1: Resume Training

**Priority**: High (infrastructure exists, just needs wiring)

**Current State**:
- `save_checkpoint()` saves parameters to `checkpoint.pkl`
- `load_checkpoint()` and `restore_parameters()` exist in `logging.py`
- Optimizers have `state_dict()` and `load_state_dict()` methods (trainable.py:450-466)
- Adam saves: `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `amsgrad`, `t`, `m`, `v`, `v_max`

**What's Missing**:
1. Checkpoint doesn't save optimizer state or epoch number
2. No `--resume` CLI argument
3. No logic to load and continue training

**Implementation Steps**:

1. **Update `save_checkpoint()` in `logging.py`** to include:
   ```python
   checkpoint_data = {
       "parameters": [...],  # existing
       "config": self.config.to_dict(),  # existing
       "timestamp": datetime.now().isoformat(),  # existing
       "optimizer_state": optimizer.state_dict(),  # NEW
       "epoch": epoch,  # NEW
       "train_losses": train_losses,  # NEW (for continuity)
       "val_losses": val_losses,  # NEW
   }
   ```

2. **Add CLI argument in `cli.py`**:
   ```python
   run_parser.add_argument(
       "--resume",
       type=str,
       default=None,
       help="Path to checkpoint.pkl to resume training from",
   )
   ```

3. **Add resume logic in `run_experiment()`** (before training loop):
   ```python
   start_epoch = 0
   if args.resume:
       checkpoint = ExperimentLog.load_checkpoint(args.resume)
       restored = ExperimentLog.restore_parameters(all_params, checkpoint)
       print(f"Restored {restored} parameters from checkpoint")

       if "optimizer_state" in checkpoint:
           optimizer.load_state_dict(checkpoint["optimizer_state"])

       start_epoch = checkpoint.get("epoch", 0) + 1
       train_losses = checkpoint.get("train_losses", [])
       val_losses = checkpoint.get("val_losses", [])
       print(f"Resuming from epoch {start_epoch}")

   for epoch in range(start_epoch, config.epochs):
       # ... existing training loop
   ```

**Files to Modify**:
- `cortical/experiments/logging.py` - update `save_checkpoint()` signature and data
- `cortical/experiments/cli.py` - add `--resume` arg and resume logic

**Estimated Complexity**: Low (2-3 hours)

---

### Feature 2: Early Stopping

**Priority**: Medium (requires validation split to be useful)

**Current State**:
- Validation loss is computed each epoch when `val_split > 0`
- `val_losses` list tracks history
- Loop runs for fixed `config.epochs`

**What's Missing**:
1. No patience counter
2. No best model tracking
3. No early exit from loop

**Implementation Steps**:

1. **Add CLI arguments**:
   ```python
   run_parser.add_argument(
       "--early-stop",
       type=int,
       default=0,
       help="Early stopping patience (0 = disabled). Requires --val-split > 0",
   )
   run_parser.add_argument(
       "--early-stop-min-delta",
       type=float,
       default=1e-4,
       help="Minimum improvement to reset patience (default: 1e-4)",
   )
   ```

2. **Add to ExperimentConfig**:
   ```python
   early_stop_patience: int = 0
   early_stop_min_delta: float = 1e-4
   ```

3. **Add early stopping logic in training loop**:
   ```python
   best_val_loss = float('inf')
   best_epoch = 0
   patience_counter = 0
   best_params = None  # Store best parameter values

   for epoch in range(start_epoch, config.epochs):
       # ... training step ...

       # Early stopping check
       if config.early_stop_patience > 0 and val_targets:
           if val_loss < best_val_loss - config.early_stop_min_delta:
               best_val_loss = val_loss
               best_epoch = epoch
               patience_counter = 0
               # Save best parameters (deep copy)
               best_params = [p.data.copy() for p in all_params]
           else:
               patience_counter += 1
               if patience_counter >= config.early_stop_patience:
                   print(f"Early stopping at epoch {epoch + 1} (best: {best_epoch + 1})")
                   # Restore best parameters
                   if best_params:
                       for p, best_data in zip(all_params, best_params):
                           p.data[:] = best_data
                   break
   ```

4. **Validation**: Require `--val-split > 0` when `--early-stop > 0`

**Files to Modify**:
- `cortical/experiments/cli.py` - add args and early stopping logic
- `cortical/experiments/config.py` - add config fields

**Estimated Complexity**: Medium (3-4 hours)

---

### Feature 3: Learning Rate Scheduling

**Priority**: Medium

**Current State**:
- Optimizer has `self.lr` that can be modified at any time
- `optimizer.load_state_dict()` can update `lr`
- Training loop has access to epoch number

**Design Decision**: Create scheduler as separate classes (like PyTorch) vs inline logic?
- **Recommendation**: Start with inline logic for simplicity, extract to classes later if needed

**Common Schedulers to Implement**:

1. **StepLR** - Decay by factor every N epochs
2. **CosineAnnealing** - Smooth decay following cosine curve
3. **ReduceLROnPlateau** - Decay when validation loss plateaus

**Implementation Steps**:

1. **Add CLI arguments**:
   ```python
   run_parser.add_argument(
       "--lr-schedule",
       type=str,
       choices=["constant", "step", "cosine", "plateau"],
       default="constant",
       help="Learning rate schedule (default: constant)",
   )
   run_parser.add_argument(
       "--lr-step-size",
       type=int,
       default=100,
       help="Epochs between LR decay for 'step' schedule (default: 100)",
   )
   run_parser.add_argument(
       "--lr-gamma",
       type=float,
       default=0.1,
       help="LR decay factor for 'step' and 'plateau' schedules (default: 0.1)",
   )
   run_parser.add_argument(
       "--lr-min",
       type=float,
       default=1e-6,
       help="Minimum learning rate (default: 1e-6)",
   )
   ```

2. **Create scheduler module** `cortical/experiments/scheduler.py`:
   ```python
   import math
   from typing import Optional

   class LRScheduler:
       """Base class for learning rate schedulers."""
       def __init__(self, optimizer, lr_min: float = 1e-6):
           self.optimizer = optimizer
           self.initial_lr = optimizer.lr
           self.lr_min = lr_min

       def step(self, epoch: int, val_loss: Optional[float] = None) -> float:
           """Update learning rate. Returns new LR."""
           raise NotImplementedError

   class StepLR(LRScheduler):
       def __init__(self, optimizer, step_size: int, gamma: float = 0.1, **kwargs):
           super().__init__(optimizer, **kwargs)
           self.step_size = step_size
           self.gamma = gamma

       def step(self, epoch: int, val_loss: Optional[float] = None) -> float:
           new_lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))
           new_lr = max(new_lr, self.lr_min)
           self.optimizer.lr = new_lr
           return new_lr

   class CosineAnnealingLR(LRScheduler):
       def __init__(self, optimizer, T_max: int, **kwargs):
           super().__init__(optimizer, **kwargs)
           self.T_max = T_max

       def step(self, epoch: int, val_loss: Optional[float] = None) -> float:
           new_lr = self.lr_min + (self.initial_lr - self.lr_min) * \
                    (1 + math.cos(math.pi * epoch / self.T_max)) / 2
           self.optimizer.lr = new_lr
           return new_lr

   class ReduceLROnPlateau(LRScheduler):
       def __init__(self, optimizer, patience: int = 10, gamma: float = 0.1, **kwargs):
           super().__init__(optimizer, **kwargs)
           self.patience = patience
           self.gamma = gamma
           self.best_loss = float('inf')
           self.counter = 0

       def step(self, epoch: int, val_loss: Optional[float] = None) -> float:
           if val_loss is None:
               return self.optimizer.lr

           if val_loss < self.best_loss:
               self.best_loss = val_loss
               self.counter = 0
           else:
               self.counter += 1
               if self.counter >= self.patience:
                   new_lr = max(self.optimizer.lr * self.gamma, self.lr_min)
                   self.optimizer.lr = new_lr
                   self.counter = 0
                   print(f"Reducing LR to {new_lr:.2e}")

           return self.optimizer.lr

   def create_scheduler(schedule_type: str, optimizer, config) -> Optional[LRScheduler]:
       if schedule_type == "constant":
           return None
       elif schedule_type == "step":
           return StepLR(optimizer, config.lr_step_size, config.lr_gamma,
                        lr_min=config.lr_min)
       elif schedule_type == "cosine":
           return CosineAnnealingLR(optimizer, config.epochs, lr_min=config.lr_min)
       elif schedule_type == "plateau":
           return ReduceLROnPlateau(optimizer, patience=config.lr_step_size,
                                    gamma=config.lr_gamma, lr_min=config.lr_min)
       else:
           raise ValueError(f"Unknown schedule: {schedule_type}")
   ```

3. **Integrate in training loop**:
   ```python
   scheduler = create_scheduler(config.lr_schedule, optimizer, config)

   for epoch in range(config.epochs):
       # ... training step ...

       # Update learning rate
       if scheduler:
           current_lr = scheduler.step(epoch, val_loss)
           if args.verbose and epoch % 50 == 0:
               print(f"  LR: {current_lr:.2e}")
   ```

**Files to Create/Modify**:
- `cortical/experiments/scheduler.py` - NEW file with scheduler classes
- `cortical/experiments/cli.py` - add args and scheduler integration
- `cortical/experiments/config.py` - add schedule config fields

**Estimated Complexity**: Medium-High (4-6 hours)

---

## Implementation Priority Order

1. **Resume Training** - Low effort, high value (enables long training runs)
2. **Early Stopping** - Medium effort, prevents overfitting
3. **LR Scheduling** - Medium effort, improves convergence

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
- [x] Dropout rate CLI argument (`--dropout`)
- [x] Attention bias CLI argument (`--use-bias`)

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
