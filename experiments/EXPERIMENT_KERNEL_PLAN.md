# ExperimentKernel Implementation Plan

**Date:** 2026-01-14
**Status:** Planning
**Branch:** claude/review-attention-graph-icHzv

---

## 1. Overview

### Purpose
Implement a minimal `ExperimentKernel` to enable training and evaluation of `AttentionGraph` on real data. This unblocks validation of the attention implementation through overfitting tests.

### Success Criteria for MVP
- [ ] Train AttentionGraph on `samples/unix_evolution.txt`
- [ ] Loss decreases monotonically toward near-zero (overfitting)
- [ ] Profiling data captured: timing, memory, gradient norms
- [ ] Attention weights become interpretable (not uniform/degenerate)
- [ ] All code has tests or is tested via the overfitting script

### Non-Goals for MVP
- Batch training (single sequence only)
- Multi-head attention
- Distributed training
- Checkpoint management beyond save/load_state
- GPU acceleration

---

## 2. MVP Scope

### 2.1 ExperimentKernel Class
**Location:** `cortical/experiments/kernel.py`

```python
class ExperimentKernel:
    """Training harness for TrainableGraphProtocol implementations."""

    def __init__(self, graph, optimizer, loss_fn, profiling=True)
    def train_step(self, targets, num_layers) -> StepMetrics
    def fit(self, targets, epochs, num_layers, ...) -> TrainingHistory
    def profile_report(self) -> ProfilingReport
```

**Key Features:**
- Works with any `TrainableGraphProtocol` implementation
- Built-in profiling (opt-out via flag)
- Gradient clipping as utility function (not graph method)
- Rich metrics per step

### 2.2 Profiling Capabilities
**Metrics to capture:**

| Metric | Per-Step | Aggregated |
|--------|----------|------------|
| Forward time (ms) | Yes | mean, std, max |
| Backward time (ms) | Yes | mean, std, max |
| Update time (ms) | Yes | mean, std, max |
| Total step time (ms) | Yes | mean, std, max |
| Memory delta (bytes) | Yes | peak, trend |
| Gradient norm | Yes | mean, max, min |
| Loss value | Yes | curve |

### 2.3 Tokenization
**Location:** `cortical/experiments/tokenizer.py`

Simple word-level tokenizer for MVP:
```python
def tokenize(text: str) -> List[str]
def build_vocab(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]
def tokens_to_ids(tokens: List[str], vocab: Dict[str, int]) -> List[int]
```

### 2.4 Overfitting Test Script
**Location:** `experiments/overfit_attention.py`

```python
# Load document
# Build vocabulary and token IDs
# Create causal AttentionGraph (seq_len = num_tokens)
# Initialize embeddings from vocabulary
# Train with ExperimentKernel
# Report: loss curve, attention visualization, profiling summary
```

---

## 3. Architecture

### Component Diagram
```
┌─────────────────────────────────────────────────────────┐
│                    ExperimentKernel                      │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Profiler  │  │  Optimizer  │  │   LossFunction  │ │
│  │  (timing,   │  │  (SGD/Adam  │  │  (MSE/CrossEnt) │ │
│  │   memory)   │  │  from       │  │  from           │ │
│  │             │  │  trainable) │  │  trainable)     │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  TrainableGraphProtocol │
              │  (AttentionGraph)       │
              ├─────────────────────────┤
              │  forward()              │
              │  backward()             │
              │  parameters()           │
              │  zero_grad()            │
              │  save_state()           │
              │  load_state()           │
              └─────────────────────────┘
```

### Data Flow (Single Training Step)
```
1. inputs = token_embeddings[:-1]  (all but last)
2. targets = token_embeddings[1:]  (all but first)
3. outputs = graph.forward(num_layers)
4. loss, grads = loss_fn(outputs, targets)
5. graph.backward(grads, num_layers)
6. clip_gradients(graph.parameters(), max_norm)
7. optimizer.step()
8. optimizer.zero_grad()
```

---

## 4. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `cortical/experiments/` package with `__init__.py`
- [ ] Implement `Profiler` class (timing context managers, memory tracking)
- [ ] Implement `clip_gradients(params, max_norm)` utility function
- [ ] Implement `ExperimentKernel` class

### Phase 2: Text Processing
- [ ] Implement `tokenizer.py` with word-level tokenization
- [ ] Implement vocabulary building
- [ ] Implement token-to-embedding conversion

### Phase 3: Overfitting Test
- [ ] Create `experiments/overfit_attention.py` script
- [ ] Load and tokenize `unix_evolution.txt`
- [ ] Set up causal AttentionGraph
- [ ] Train and capture metrics
- [ ] Generate report with visualizations

### Phase 4: Validation
- [ ] Verify loss approaches zero
- [ ] Verify attention patterns are meaningful
- [ ] Review profiling data for anomalies
- [ ] Document findings

---

## 5. Escape Clauses for Agents

> **IMPORTANT:** When implementing this plan, agents may encounter situations
> where they cannot complete a component in a single response. Follow these
> guidelines:

### 5.1 Incomplete Implementation
If you cannot finish a function/class in one shot:
```python
def incomplete_function():
    # TODO(agent): Continue implementation
    # NEXT: [describe what needs to be done next]
    # CONTEXT: [any important context for continuation]
    raise NotImplementedError("Implementation incomplete - see TODO above")
```

### 5.2 Incomplete File
If you cannot finish a file:
```python
# === FILE INCOMPLETE ===
# TODO(agent): Continue from Section X
# COMPLETED: [list what's done]
# REMAINING: [list what's left]
# CONTEXT: [important state/decisions made]
```

### 5.3 Blocked by External Factor
If blocked by missing dependency, unclear requirement, or needed decision:
```python
# TODO(agent): BLOCKED - [reason]
# NEEDS: [what's needed to unblock]
# WORKAROUND: [temporary solution if any]
```

### 5.4 Test Coverage Gap
If tests aren't complete:
```python
# TODO(agent): Add tests for:
# - [ ] test case 1 description
# - [ ] test case 2 description
# PRIORITY: [high/medium/low]
```

### 5.5 Handoff Protocol
When ending a session with incomplete work:
1. Commit all working code (even if partial)
2. Update this plan document with progress
3. Add `# SESSION_HANDOFF:` comment at continuation point
4. Create/update `KNOWLEDGE_TRANSFER.md` if significant decisions were made

---

## 6. Wishlist (Future Enhancements)

### High Priority (Post-MVP)
- [ ] **Batch training**: Process multiple sequences per step
- [ ] **Validation split**: Proper train/val/test separation
- [ ] **Learning rate scheduling**: ReduceLROnPlateau, cosine annealing
- [ ] **Early stopping**: Prevent overfitting on real tasks
- [ ] **Checkpointing**: Save/restore training state periodically

### Medium Priority
- [ ] **Multi-head attention**: Enable num_heads > 1 in AttentionGraph
- [ ] **Subword tokenization**: BPE or SentencePiece for better vocab
- [ ] **Gradient accumulation**: Simulate larger batches
- [ ] **Mixed precision**: Float16 for memory efficiency
- [ ] **TensorBoard integration**: Visual training monitoring

### Low Priority (Nice to Have)
- [ ] **Distributed training**: Multi-GPU/multi-node
- [ ] **Hyperparameter search**: Grid/random/Bayesian optimization
- [ ] **Model surgery**: Freeze layers, transfer learning
- [ ] **Attention visualization**: Interactive HTML reports
- [ ] **ONNX export**: Model portability

### Research Ideas
- [ ] **Hybrid graphs**: Combine attention + message passing
- [ ] **Sparse attention**: For longer sequences
- [ ] **Relative position encoding**: Beyond absolute positions
- [ ] **Memory-augmented attention**: External memory banks

---

## 7. Open Questions

### Resolved
- ~~Q: Reuse `train_step`/`fit` from trainable.py?~~
  A: No, create new implementation for protocol compatibility and profiling

- ~~Q: Character-level or word-level tokenization?~~
  A: Word-level for MVP (simpler, faster convergence)

### Deferred
- Q: Should ExperimentKernel live in `cortical/experiments/` or `cortical/graph/`?
  Decision: `cortical/experiments/` - separate concern from graph implementation

- Q: How to handle OOV (out-of-vocabulary) tokens?
  Decision: Use `<UNK>` token for MVP; revisit with subword tokenization

### Needs Input
- Q: Target sequence length for overfitting test?
  Proposal: Use full document (~300 tokens) if memory allows; truncate to 128 if needed

---

## 8. Progress Log

| Date | Status | Notes |
|------|--------|-------|
| 2026-01-14 | Planning | Initial plan created |
| | | |

---

## References

- `cortical/graph/attention.py` - AttentionGraph implementation (approved for training)
- `cortical/graph/trainable.py` - Existing Optimizer, LossFunction, TrainingHistory
- `samples/unix_evolution.txt` - Target document for overfitting test
- `samples/attention_graph_implementation_review.txt` - Review findings
