# SparkSLM Research Roadmap: Self-Adjusting Plan

**A Fail-Safe, System-Constrained Research Agenda**

*Built by the system, for the system, to use as desired*

---

## Philosophy: The Self-Adjusting Plan

This roadmap is designed to:

1. **Constrain** goals to what the system can actually build
2. **Verify** progress through automated checkpoints
3. **Adjust** direction based on measured outcomes
4. **Fail gracefully** when hypotheses prove wrong
5. **Learn** from each iteration to improve future plans

The plan is not a rigid schedule—it's a living document that evolves as we learn.

---

## Verification Protocol

Every milestone includes:

```python
class Milestone:
    hypothesis: str      # What we believe to be true
    experiment: str      # How we test it
    success_criteria: str  # Measurable threshold
    fail_safe: str       # What to do if hypothesis is wrong
    next_if_pass: str    # Next step on success
    next_if_fail: str    # Alternative path on failure
```

---

## Phase 1: Foundation Verification (COMPLETED ✓)

### Milestone 1.1: N-gram Model Works ✓
```yaml
hypothesis: Trigram model can learn corpus patterns in <1s
experiment: Train on 147 documents, measure time and perplexity
success_criteria:
  - Training time < 1s
  - Perplexity < 500 (vs ~4800 random baseline)
fail_safe: If too slow, reduce n-gram order to bigrams
result: PASSED
  - Training: 0.34s
  - Perplexity: 376.9
```

### Milestone 1.2: Alignment Index Works ✓
```yaml
hypothesis: Structured memory can store/retrieve user knowledge
experiment: Load samples/alignment/, query terms
success_criteria:
  - Load 20+ entries from markdown
  - Retrieve correct definitions by key
fail_safe: Simplify format if parsing fails
result: PASSED
  - Loaded: 27 entries
  - Retrieval: 100% accuracy on known keys
```

### Milestone 1.3: Processor Integration Works ✓
```yaml
hypothesis: SparkMixin integrates without breaking existing functionality
experiment: Run full test suite with spark enabled
success_criteria:
  - All existing tests pass
  - New spark tests pass (target: 50+)
fail_safe: Isolate spark in separate class if mixin conflicts
result: PASSED
  - Existing tests: 100% pass
  - New spark tests: 72 tests passing
```

---

## Phase 2: Quality Measurement (CURRENT)

### Milestone 2.1: Prediction Quality Baseline
```yaml
hypothesis: Predictions provide useful signal despite low accuracy
experiment: Measure accuracy@1, accuracy@5, perplexity on held-out data
success_criteria:
  - Accuracy@5 > 30% (better than random top-5 from 4823 vocab)
  - Perplexity stable across runs
fail_safe: If predictions useless, pivot to pure alignment-based priming
next_if_pass: Proceed to query expansion integration
next_if_fail: Research alternative statistical methods (skip-grams, character n-grams)

verification_command: |
  python -c "
  from cortical import CorticalTextProcessor
  p = CorticalTextProcessor(spark=True)
  # Load corpus...
  p.train_spark()
  # Run held-out evaluation...
  "
```

### Milestone 2.2: Query Expansion Impact
```yaml
hypothesis: Spark-enhanced expansion improves search relevance
experiment: Compare search results with/without spark on 20 test queries
success_criteria:
  - Precision@5 improvement > 10%
  - No regression in recall
fail_safe: Reduce spark_boost if predictions hurt relevance
next_if_pass: Document findings, proceed to anomaly detection
next_if_fail: Investigate which query types benefit, add selective application

verification_command: |
  python scripts/evaluate_spark_expansion.py --queries test_queries.txt
```

### Milestone 2.3: Alignment Acceleration
```yaml
hypothesis: Alignment context reduces disambiguation rounds
experiment: Simulate user sessions, count clarification requests
success_criteria:
  - 30%+ reduction in disambiguation rounds
fail_safe: Expand alignment corpus if context insufficient
next_if_pass: Proceed to Phase 3
next_if_fail: Research better alignment retrieval (semantic search vs keyword)

verification_command: |
  python scripts/simulate_alignment_sessions.py --sessions 100
```

---

## Phase 3: Anomaly Detection (PLANNED)

### Milestone 3.1: Perplexity-Based Anomaly Detection
```yaml
hypothesis: Unusual queries have higher perplexity scores
experiment: Inject adversarial/malformed queries, measure perplexity
success_criteria:
  - Anomalous queries: perplexity > 2x normal mean
  - False positive rate < 5%
fail_safe: Use ensemble of multiple n-gram orders
next_if_pass: Add to query pipeline as safety filter
next_if_fail: Research other anomaly signals (alignment mismatch, topic drift)

implementation_sketch: |
  class AnomalyDetector:
      def __init__(self, ngram_model):
          self.model = ngram_model
          self.baseline_perplexity = None

      def calibrate(self, normal_queries):
          perplexities = [self.model.perplexity(q) for q in normal_queries]
          self.baseline_perplexity = statistics.mean(perplexities)
          self.threshold = self.baseline_perplexity * 2

      def is_anomalous(self, query):
          return self.model.perplexity(query) > self.threshold
```

### Milestone 3.2: Prompt Injection Detection
```yaml
hypothesis: Injection attempts differ statistically from normal queries
experiment: Test against known injection patterns
success_criteria:
  - Detect 80%+ of known injection patterns
  - False positive rate < 10%
fail_safe: Add pattern-based rules as fallback
next_if_pass: Integrate as pre-processing filter
next_if_fail: This may not be solvable with statistical methods alone
```

---

## Phase 4: Sample Generation (PLANNED)

### Milestone 4.1: Pattern-Based Generation
```yaml
hypothesis: N-gram model can generate plausible text samples
experiment: Generate 100 samples, evaluate coherence
success_criteria:
  - 50%+ samples are syntactically valid
  - Style matches training corpus
fail_safe: Use templates with slot-filling instead of free generation
next_if_pass: Use for test case generation
next_if_fail: Abandon generative use case, focus on discriminative applications

implementation_sketch: |
  def generate_sample(self, seed: str, length: int = 20) -> str:
      tokens = seed.split()
      for _ in range(length):
          predictions = self.ngram.predict(tokens[-2:])
          if not predictions:
              break
          next_token = self._sample_from_distribution(predictions)
          tokens.append(next_token)
      return ' '.join(tokens)
```

### Milestone 4.2: Alignment Corpus Expansion
```yaml
hypothesis: Can infer new alignment entries from usage patterns
experiment: Analyze query patterns, suggest definitions
success_criteria:
  - 70%+ of suggestions are useful (human evaluation)
fail_safe: Require human approval for all new entries
next_if_pass: Semi-automated alignment growth
next_if_fail: Keep alignment fully manual
```

---

## Phase 5: Cross-Project Transfer (RESEARCH)

### Milestone 5.1: Shared Vocabulary Extraction
```yaml
hypothesis: Programming concepts transfer across codebases
experiment: Train on project A, test on project B
success_criteria:
  - 20%+ of predictions valid on new project
fail_safe: Maintain separate models per project
next_if_pass: Build shared "programming language model"
next_if_fail: Accept project-specific models as sufficient
```

### Milestone 5.2: Domain Adaptation
```yaml
hypothesis: Fine-tuning from base improves faster than training from scratch
experiment: Compare adaptation time/quality vs fresh training
success_criteria:
  - Adaptation reaches 90% of full training quality in 50% time
fail_safe: Just use fresh training (it's fast enough)
next_if_pass: Implement adapter layers
next_if_fail: Proceed without transfer learning
```

---

## Self-Adjustment Mechanisms

### Automatic Checkpoint Verification

```python
# scripts/verify_milestone.py
def verify_milestone(milestone_id: str) -> MilestoneResult:
    """
    Run verification for a milestone, return structured result.

    Automatically adjusts roadmap based on outcome.
    """
    milestone = load_milestone(milestone_id)

    # Run experiment
    result = run_experiment(milestone.experiment)

    # Check criteria
    passed = evaluate_criteria(result, milestone.success_criteria)

    if passed:
        log_success(milestone_id, result)
        queue_next(milestone.next_if_pass)
    else:
        log_failure(milestone_id, result)
        execute_failsafe(milestone.fail_safe)
        queue_next(milestone.next_if_fail)

    return MilestoneResult(
        milestone_id=milestone_id,
        passed=passed,
        metrics=result,
        next_action=milestone.next_if_pass if passed else milestone.next_if_fail
    )
```

### Roadmap State Tracking

```python
# State persisted in tasks/roadmap_state.json
{
    "current_phase": 2,
    "current_milestone": "2.1",
    "completed_milestones": ["1.1", "1.2", "1.3"],
    "failed_milestones": [],
    "failsafe_triggered": [],
    "metrics_history": {
        "1.1": {"training_time": 0.34, "perplexity": 376.9},
        ...
    },
    "adjustments": [
        {"date": "2025-12-19", "reason": "...", "action": "..."}
    ]
}
```

### Adjustment Triggers

The roadmap automatically adjusts when:

1. **Milestone fails**: Execute fail-safe, log adjustment
2. **Metrics degrade**: Alert and suggest rollback
3. **New capability discovered**: Add research spike
4. **User feedback received**: Incorporate into alignment corpus
5. **Resource constraint hit**: Scale back scope

---

## Realistic Constraints

### What We Can Build

| Capability | Feasibility | Confidence |
|------------|-------------|------------|
| N-gram training <1s | ✓ Proven | 100% |
| Alignment storage/retrieval | ✓ Proven | 100% |
| Processor integration | ✓ Proven | 100% |
| Perplexity-based anomaly detection | Likely | 80% |
| Pattern-based generation | Possible | 60% |
| Cross-project transfer | Uncertain | 40% |
| Real-time adaptation | Needs research | 30% |

### What We Cannot Build (Without New Dependencies)

- Neural embeddings (requires ML framework)
- Transformer-based prediction (requires GPU)
- Semantic similarity (requires embeddings)
- Online learning (requires streaming infrastructure)

### Resource Constraints

| Resource | Available | Limit |
|----------|-----------|-------|
| Compute | CPU only | No GPU |
| Memory | ~2GB | Fits in memory |
| Dependencies | Zero | No pip install |
| Training time | Seconds | Not hours |
| Model size | Megabytes | Not gigabytes |

---

## Success Metrics Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│              SparkSLM Research Dashboard                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Phase 1: Foundation          [████████████████████] 100%   │
│  Phase 2: Quality             [████████░░░░░░░░░░░░]  40%   │
│  Phase 3: Anomaly Detection   [░░░░░░░░░░░░░░░░░░░░]   0%   │
│  Phase 4: Sample Generation   [░░░░░░░░░░░░░░░░░░░░]   0%   │
│  Phase 5: Transfer Learning   [░░░░░░░░░░░░░░░░░░░░]   0%   │
│                                                              │
│  Overall Progress: 28%                                       │
│                                                              │
│  Key Metrics:                                                │
│  ├─ Perplexity: 376.9 (target: <500) ✓                      │
│  ├─ Training Time: 0.34s (target: <1s) ✓                    │
│  ├─ Test Coverage: 72 tests passing ✓                       │
│  ├─ Query Expansion Boost: TBD                              │
│  └─ Alignment Acceleration: TBD                              │
│                                                              │
│  Next Milestone: 2.1 - Prediction Quality Baseline          │
│  Estimated Effort: 1-2 sessions                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## How to Use This Plan

### For the System (Claude/AI)

1. **Check current milestone** before starting work
2. **Run verification command** after completing work
3. **Update roadmap state** based on results
4. **Follow fail-safe** if milestone fails
5. **Log adjustments** when deviating from plan

### For Humans

1. **Review progress** via dashboard metrics
2. **Provide feedback** to adjust priorities
3. **Add alignment entries** for new vocabulary
4. **Approve fail-safe triggers** when judgment needed
5. **Celebrate milestones** when they pass ✓

### For the Codebase

1. **Tests enforce** milestone success criteria
2. **Scripts verify** metrics automatically
3. **State persists** across sessions
4. **History enables** learning from past adjustments

---

## Next Immediate Actions

```markdown
## Sprint 17: SparkSLM Core (In Progress)

### Completed ✓
- [x] T-SPARK-001: Implement NGramModel
- [x] T-SPARK-002: Implement AlignmentIndex
- [x] T-SPARK-003: Implement SparkPredictor facade
- [x] T-SPARK-004: Integrate with CorticalTextProcessor
- [x] T-SPARK-005: Write unit tests (72 tests)

### Remaining
- [ ] T-SPARK-006: Implement AnomalyDetector (Milestone 3.1)
- [ ] T-SPARK-007: Integrate with query expansion pipeline (Milestone 2.2)
- [ ] T-SPARK-008: Add training script for CLI usage
- [ ] T-SPARK-010: Documentation and examples

### Verification Checkpoint
After completing remaining tasks:
1. Run: `python scripts/verify_milestone.py 2.1`
2. If pass: Proceed to Sprint 18
3. If fail: Execute fail-safe, adjust plan
```

---

## Appendix: Adjustment History

| Date | Adjustment | Reason | Outcome |
|------|------------|--------|---------|
| 2025-12-19 | Created roadmap | Initial planning | - |
| - | - | - | - |

*This log will grow as the system learns and adapts.*

---

*Plan generated by the system, for the system*
*Last updated: 2025-12-19*
*Next review: After Milestone 2.1 completion*
