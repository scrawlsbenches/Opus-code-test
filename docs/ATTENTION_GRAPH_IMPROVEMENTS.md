# AttentionGraph Improvements Plan

**Status**: Planning
**Created**: 2026-01-14
**Branch**: claude/review-implementation-gaps-lZMG7
**Last Review**: Deep code review of AttentionGraph and supporting code

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

### Feature 1: Sinusoidal Position Encoding
**Location**: `cortical/experiments/position.py:171-189`
**Effort**: Low (2-3 hours)
**Impact**: High - completes documented feature

**Current State**:
```python
class SinusoidalPositionEncoding:
    def __init__(self, max_len: int, embedding_dim: int):
        raise NotImplementedError(...)
```

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
- [ ] Implement `SinusoidalPositionEncoding` class
- [ ] Add "sinusoidal" to config validation (`config.py:72`)
- [ ] Update CLI choices for `--position-encoding`
- [ ] Add unit tests

---

### Feature 2: Residual Connections
**Location**: `cortical/graph/attention.py` (AttentionLayer and AttentionGraph)
**Effort**: Medium (4-6 hours)
**Impact**: High - critical for training stability

**Current State**: No residual connections. Output is directly from attention.

**Standard Transformer Pattern**:
```python
# Pre-LN variant (more stable)
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

# Post-LN variant (original paper)
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

**Implementation Plan**:
1. Add `residual: bool = True` parameter to AttentionLayer
2. In forward: `output = input + attention_output` when residual=True
3. In backward: gradient flows through both paths
4. Add to AttentionGraph constructor as option

**Tasks**:
- [ ] Add residual connection to AttentionLayer.forward()
- [ ] Update AttentionLayer.backward() for residual gradient flow
- [ ] Add `--residual` CLI flag (default True)
- [ ] Add unit tests for residual gradient flow

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

### Feature 4: Train/Validation Split
**Location**: `cortical/experiments/cli.py`, `cortical/experiments/logging.py`
**Effort**: Low (2-3 hours)
**Impact**: Medium - important for detecting overfitting

**Current State**: All data used for training, no validation.

**Implementation**:
1. Add `--val-split` CLI argument (default 0.1)
2. Split tokens into train/val sets
3. Compute validation loss each epoch
4. Log to `metrics.json` with `val_losses` array
5. Optional: early stopping based on val loss

**Tasks**:
- [ ] Add `--val-split` argument to CLI
- [ ] Implement token split logic in `run_experiment()`
- [ ] Update ExperimentKernel or add eval loop
- [ ] Update ExperimentMetrics/ExperimentLog for val metrics
- [ ] Add `--early-stopping` patience option

---

### Feature 5: Weight Decay CLI Option
**Location**: `cortical/experiments/cli.py`
**Effort**: Low (1 hour)
**Impact**: Low-Medium - regularization helps generalization

**Current State**: Adam optimizer supports weight_decay but not exposed in CLI.

**Tasks**:
- [ ] Add `--weight-decay` CLI argument (default 0.0)
- [ ] Pass to optimizer in `run_experiment()`

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

### Refactor 2: Fix Edge Weight Key Parsing
**Location**: `cortical/graph/trainable.py:1424-1430`
**Effort**: Low (1 hour)

**Problem**: Key parsing with `split("_")` breaks for node IDs containing underscores.

**Fix**: Use a safer separator like `"::"` or store as JSON tuple.

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
