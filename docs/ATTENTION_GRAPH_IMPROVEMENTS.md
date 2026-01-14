# AttentionGraph Improvements Plan

**Status**: Complete
**Created**: 2026-01-14
**Branch**: claude/review-attention-graph-9dbtC
**Last Review**: Deep code review of AttentionGraph and supporting code
**Current Session**: Implemented 5 priority features for small graph experimentation

## Overview

This document captures bugs, missing implementations, and improvements identified during a comprehensive code review of the AttentionGraph system and supporting experiment infrastructure.

## Completed (This Session)

- [x] Add `--dropout` CLI argument (`cli.py`)
- [x] Add `--use-bias` CLI argument (`cli.py`)
- [x] Pass dropout/use_bias to `create_causal_attention_graph()` (`cli.py:196-203`)
- [x] Clear VocabProjection caches in `zero_grad()` (`projection.py:193-201`)
- [x] Fix TrainableGraph multi-layer backward pass (`trainable.py`)
- [x] Fix softmax numerical underflow edge case (`attention.py`)

All 235 tests pass (166 attention + 69 trainable graph tests).

---

## Completed (Current Session - 2026-01-14)

Implemented priority features for small graph experimentation:

- [x] **Refactor 2**: Fix edge weight key parsing bug (`trainable.py:1409,1433`)
  - Changed separator from `"_"` to `"::"` for safe node ID handling
  - Added backward compatibility for legacy checkpoints
- [x] **Feature 2**: Residual connections in AttentionLayer (`attention.py`)
  - Added `use_residual` parameter to AttentionLayer and AttentionGraph
  - Updated forward pass: `output = attention_output + input`
  - Updated backward pass: gradient flows through both paths
  - Added `--residual` CLI flag
  - Added TODO comments for LayerNorm integration
- [x] **Feature 1**: Sinusoidal position encoding (`position.py`)
  - Implemented full `SinusoidalPositionEncoding` class
  - Pre-computed encoding matrix for efficiency
  - Added to CLI choices: `--position-encoding sinusoidal`
- [x] **Feature 5**: Weight decay CLI argument (`cli.py`, `config.py`)
  - Added `--weight-decay` CLI argument
  - Passes to Adam optimizer
- [x] **Feature 4**: Train/validation split (`cli.py`, `config.py`, `logging.py`)
  - Added `--val-split` CLI argument (0.0-0.5)
  - Split prediction positions (not tokens) for validation
  - Compute and log validation loss each epoch
  - Track val_losses, val_accuracies in ExperimentMetrics
  - Report final val accuracy

**Deferred** (can add later if needed):
- Feature 3: Layer Normalization - TODO comments added in code
- Feature 6: Feed-Forward Network (FFN) - TODO comments added in code

---

## Priority 1: Bugs (All Fixed)

### ~~Bug 2: TrainableGraph backward uses wrong layer input~~ FIXED
**Location**: `cortical/graph/trainable.py:1283-1286`
**Severity**: Medium
**Effort**: Medium
**Status**: FIXED - Added layer_inputs field and proper storage/retrieval

**Problem**:
```python
if layer == 0:
    layer_input = node.embedding.data
else:
    layer_input = node.output if node.output is not None else node.embedding.data
```

For layers > 0, `node.output` stores the output from the **last** layer, not the input to the current layer being processed. This causes incorrect gradient computation for the transform matrix.

**Fix**: Store layer inputs during forward pass (similar to how AttentionGraph does with `node.layer_inputs`).

**Impact**: Affects multi-layer TrainableGraph training. AttentionGraph is not affected.

---

### ~~Bug 3: Softmax numerical instability edge case~~ FIXED
**Location**: `cortical/graph/attention.py:492-494`
**Severity**: Low
**Effort**: Low
**Status**: FIXED - Explicit check for underflow with uniform attention fallback

**Problem**:
```python
scores_stable = scores - np.max(scores)
exp_scores = np.exp(scores_stable)
attention_weights = exp_scores / (np.sum(exp_scores) + 1e-10)
```

When all scores are extremely negative after subtracting max, `exp_scores` can underflow to zeros. The `1e-10` prevents division by zero but results in near-uniform attention.

**Fix**: Add check for sum near zero and handle gracefully (e.g., fall back to uniform attention with warning, or use log-sum-exp trick).

---

## Priority 2: Missing Core Features (High Impact)

### ~~Feature 1: Sinusoidal Position Encoding~~ IMPLEMENTED
**Location**: `cortical/experiments/position.py:171-343`
**Effort**: Low (2-3 hours)
**Impact**: High - completes documented feature
**Status**: IMPLEMENTED

**Implementation**:
```python
# From "Attention Is All You Need" paper, Section 3.5
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Benefits**:
- No trainable parameters (faster training)
- Generalizes to unseen sequence lengths
- Classic baseline for comparison with learned encodings

**Tasks**:
- [x] Implement `SinusoidalPositionEncoding` class
- [x] Add "sinusoidal" to config validation (`config.py:72`)
- [x] Update CLI choices for `--position-encoding`
- [ ] Add unit tests (TODO)

---

### ~~Feature 2: Residual Connections~~ IMPLEMENTED
**Location**: `cortical/graph/attention.py` (AttentionLayer and AttentionGraph)
**Effort**: Medium (4-6 hours)
**Impact**: High - critical for training stability
**Status**: IMPLEMENTED

**Implementation** (Pre-LN style without LayerNorm):
```python
# Simple residual: output = attention_output + input
# LayerNorm can be added later for full transformer block
```

**Design Decision**: Implemented simple residual without LayerNorm.
- For small graphs (1-3 layers), LayerNorm is optional
- TODO comments added for adding LayerNorm when needed

**Tasks**:
- [x] Add residual connection to AttentionLayer.forward()
- [x] Update AttentionLayer.backward() for residual gradient flow
- [x] Add `--residual` CLI flag (default False for backward compatibility)
- [ ] Add unit tests for residual gradient flow (TODO)

---

### Feature 3: Layer Normalization
**Location**: New file `cortical/graph/normalization.py` or in `attention.py`
**Effort**: Medium (4-6 hours)
**Impact**: High - critical for training stability

**Implementation**:
```python
class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = Parameter(np.ones(dim), name="layernorm_gamma")
        self.beta = Parameter(np.zeros(dim), name="layernorm_beta")
        self.eps = eps

    def forward(self, x: Array) -> Array:
        mean = x.mean()
        var = x.var()
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma.data * x_norm + self.beta.data

    def backward(self, grad_output: Array) -> Array:
        # Gradient through normalization
        ...
```

**Tasks**:
- [ ] Implement LayerNorm class with forward/backward
- [ ] Integrate into AttentionLayer (pre-LN or post-LN option)
- [ ] Add `--layer-norm` CLI flag
- [ ] Add unit tests with numerical gradient checks

---

## Priority 3: Training Infrastructure

### ~~Feature 4: Train/Validation Split~~ IMPLEMENTED
**Location**: `cortical/experiments/cli.py`, `cortical/experiments/logging.py`, `cortical/experiments/config.py`
**Effort**: Low (2-3 hours)
**Impact**: Medium - important for detecting overfitting
**Status**: IMPLEMENTED

**Implementation**:
1. Added `--val-split` CLI argument (default 0.0, range 0.0-0.5)
2. Split prediction positions (not tokens) - uses last N% for validation
3. Compute validation loss each epoch
4. Log to `metrics.json` with `val_losses` and `val_accuracies` arrays
5. Report final validation loss and accuracy

**Tasks**:
- [x] Add `--val-split` argument to CLI
- [x] Implement position split logic in `run_experiment()`
- [x] Add compute_loss() and compute_accuracy() helpers
- [x] Update ExperimentMetrics/ExperimentLog for val metrics
- [ ] Add `--early-stopping` patience option (TODO - can add later)

---

### ~~Feature 5: Weight Decay CLI Option~~ IMPLEMENTED
**Location**: `cortical/experiments/cli.py`, `cortical/experiments/config.py`
**Effort**: Low (1 hour)
**Impact**: Low-Medium - regularization helps generalization
**Status**: IMPLEMENTED

**Tasks**:
- [x] Add `--weight-decay` CLI argument (default 0.0)
- [x] Add `weight_decay` to ExperimentConfig
- [x] Pass to Adam optimizer in `run_experiment()`

---

## Priority 4: Model Improvements

### Feature 6: Feed-Forward Network (FFN)
**Location**: `cortical/graph/attention.py`
**Effort**: Medium (4-6 hours)
**Impact**: Medium - increases model expressivity

Standard transformer has FFN after attention:
```python
FFN(x) = GELU(xW1 + b1)W2 + b2
# Typically: hidden_dim = 4 * embedding_dim
```

**Tasks**:
- [ ] Implement FeedForwardLayer class
- [ ] Add to AttentionGraph as option
- [ ] Add `--ffn-dim` CLI argument (0 = disabled)

---

## Priority 5: Code Quality

### Refactor 1: Extract Parameter Class
**Location**: `cortical/graph/trainable.py:220-250`, `cortical/graph/attention.py:236-280`
**Effort**: Low (1-2 hours)

**Current State**: Parameter class is duplicated in both files.

**Tasks**:
- [ ] Create `cortical/graph/parameters.py`
- [ ] Move Parameter class there
- [ ] Update imports in trainable.py and attention.py
- [ ] Verify all tests pass

---

### ~~Refactor 2: Fix Edge Weight Key Parsing~~ FIXED
**Location**: `cortical/graph/trainable.py:1407-1447`
**Effort**: Low (1 hour)
**Status**: FIXED

**Problem**: Key parsing with `split("_")` breaks for node IDs containing underscores (e.g., "pos_0").

**Fix**: Changed to use `"::"` separator with backward compatibility for legacy checkpoints.
- Save: `f"{edge.source_id}::{edge.target_id}"`
- Load: Check for `"::"` first, fall back to `"_"` for legacy

---

### Refactor 3: Use Local Random State
**Location**: Multiple files
**Effort**: Low (2 hours)

**Problem**: `np.random.seed(seed)` modifies global state.

**Fix**: Use `np.random.Generator` for local random state:
```python
self._rng = np.random.default_rng(seed)
# Then use self._rng.random(), self._rng.randn(), etc.
```

---

## Architecture Considerations (Future)

These are larger changes to consider for scaling:

1. **Batch Processing**: Current implementation processes one sequence at a time
2. **Gradient Checkpointing**: Memory grows linearly with sequence length
3. **Mixed Precision**: Could reduce memory and speed up training

---

## Testing Strategy

For each feature implemented:
1. Unit tests for the component in isolation
2. Numerical gradient checks where applicable
3. Integration test with CLI
4. Verify existing tests still pass

---

## Dependencies

```
Feature 2 (Residual) + Feature 3 (LayerNorm) = Full Transformer Block
Feature 6 (FFN) requires Feature 2 and 3 for proper training
```

Recommended implementation order:
1. Sinusoidal Position Encoding (standalone)
2. LayerNorm (needed by others)
3. Residual Connections (needs LayerNorm for best results)
4. FFN (needs residual + LayerNorm)
5. Train/Val Split (standalone)

---

## Notes

- AttentionGraph is the primary focus; TrainableGraph bugs are lower priority
- All changes should maintain backward compatibility with existing experiments
- Position encoding shows mixed results (see `position.py` TODOs) - may need tuning
