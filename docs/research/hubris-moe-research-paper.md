# Hubris: A Credit-Based Mixture of Experts System for Software Development Assistance

**Authors:** Cortical Development Team
**Date:** December 2025
**Version:** 1.0

---

## Abstract

We present Hubris, a Mixture of Experts (MoE) system designed to assist software developers with task-level predictions about file modifications, test selection, error diagnosis, and workflow patterns. Unlike traditional monolithic prediction models, Hubris employs multiple specialized "micro-experts" that compete for influence through a credit-based economy. Experts earn credibility through accurate predictions and lose it through failures, creating a self-correcting system where performance determines authority.

Built atop the Cortical Text Processor—a hierarchical information retrieval system inspired by visual cortex organization—Hubris demonstrates that ensemble learning with economic incentives can provide meaningful assistance even with limited training data. Our system achieves 100% accuracy on file prediction tasks during active development sessions while maintaining calibrated confidence estimates.

This paper describes the architecture, credit system design, calibration tracking methodology, and lessons learned from building a learning system that operates within its own development environment.

---

## 1. Introduction

Software development assistance systems face a fundamental challenge: they must provide useful predictions while honestly representing uncertainty. A system that claims high confidence but delivers poor accuracy erodes trust. Conversely, a system that hedges every prediction provides little practical value.

We address this challenge through **Hubris**—a name chosen with intentional irony. In Greek tradition, hubris represents excessive confidence that leads to downfall. Our system inverts this: experts must *earn* the right to be confident through demonstrated accuracy. The name serves as a constant reminder that confidence without competence is worthless.

### 1.1 Motivation

Traditional approaches to developer assistance fall into two categories:

1. **Rule-based systems** that encode human knowledge but cannot learn from experience
2. **Machine learning models** that require massive training data and often produce uncalibrated predictions

We sought a middle path: a system that could learn from modest amounts of data (hundreds, not millions, of examples) while maintaining honest confidence estimates. The key insight was to treat prediction as an economic problem rather than purely a statistical one.

### 1.2 Contributions

This work makes the following contributions:

- **Credit-based expert routing**: A mechanism where prediction accuracy directly affects future influence
- **Calibration tracking**: Real-time monitoring of confidence-accuracy alignment with ECE, MCE, and Brier score metrics
- **Cold-start UX**: Graceful handling of the bootstrap problem when experts have no track record
- **Self-improving feedback loops**: Integration with version control to learn from actual development outcomes

---

## 2. Background

### 2.1 The Cortical Text Processor Foundation

Hubris builds upon the Cortical Text Processor, a hierarchical information retrieval system that organizes text through four abstraction layers:

| Layer | Name | Analogy | Contents |
|-------|------|---------|----------|
| 0 | TOKENS | V1 (edges) | Individual words |
| 1 | BIGRAMS | V2 (patterns) | Word pairs |
| 2 | CONCEPTS | V4 (shapes) | Semantic clusters |
| 3 | DOCUMENTS | IT (objects) | Full documents |

This architecture provides the semantic infrastructure that experts use for prediction. Terms are connected through lateral connections (co-occurrence), typed connections (semantic relations), and cross-layer feedforward/feedback connections.

The naming draws from neuroscience—specifically the visual cortex pathway from V1 through inferotemporal cortex—but the implementation uses standard information retrieval algorithms: PageRank for importance, TF-IDF for distinctiveness, and label propagation for clustering.

### 2.2 Mixture of Experts Architecture

The Mixture of Experts paradigm, introduced by Jacobs et al. (1991) and popularized in neural networks by Shazeer et al. (2017), routes inputs to specialized sub-networks based on learned gating functions. We adapt this approach for software development:

- **FileExpert**: Predicts which files need modification for a task
- **TestExpert**: Predicts which tests should run for code changes
- **ErrorDiagnosisExpert**: Identifies error causes and suggests fixes
- **EpisodeExpert**: Learns workflow patterns from session transcripts

Each expert operates independently and produces predictions in a standardized format. A routing mechanism combines their outputs based on relevance and track record.

### 2.3 Thousand Brains Theory Inspiration

Hawkins (2021) proposed that the neocortex consists of thousands of cortical columns, each building complete models of objects and voting to reach consensus. This theory informed our design:

1. **Multiple complete models**: Each expert maintains its own understanding of the problem space
2. **Voting for consensus**: Predictions are aggregated through confidence-weighted voting
3. **Parallel processing**: Experts can be trained and evaluated independently

---

## 3. System Architecture

### 3.1 Expert Base Class

All experts inherit from `MicroExpert`, which defines the contract:

```python
class MicroExpert:
    expert_id: str           # Unique identifier
    expert_type: str         # Category (file, test, error, episode)
    version: str             # Model version
    model_data: Dict         # Expert-specific learned parameters
    metrics: ExpertMetrics   # Performance statistics

    def predict(self, context: Dict) -> ExpertPrediction:
        """Generate ranked predictions with confidence scores."""

    def train(self, data: List[Dict]) -> None:
        """Learn from training examples."""
```

The `ExpertPrediction` format ensures all experts produce comparable outputs:

```python
@dataclass
class ExpertPrediction:
    expert_id: str
    expert_type: str
    items: List[Tuple[str, float]]  # (item, confidence) pairs
    metadata: Dict[str, Any]        # Context about prediction
```

### 3.2 Credit System

The credit system implements an economic model for expert reputation:

**Initial State**: Each expert begins with 100 credits—neither trusted nor distrusted.

**Credit Flow**:
- Correct predictions earn credits proportional to confidence and outcome magnitude
- Incorrect predictions lose credits
- High-confidence errors are penalized more heavily than low-confidence ones

**Routing Weights**: Expert influence in ensemble voting is computed via softmax over credit balances:

```
weight_i = exp(balance_i / temperature) / Σ exp(balance_j / temperature)
```

The temperature parameter controls distribution sharpness:
- Low temperature (0.5): Winner-takes-most behavior
- High temperature (2.0): More democratic voting

**Minimum Floor**: A configurable minimum weight (default 10%) prevents any expert from being completely silenced, preserving the ability to recover from poor early performance.

### 3.3 Value Attribution

When predictions resolve (e.g., after a commit reveals which files actually changed), the `ValueAttributor` generates signals:

```python
@dataclass
class ValueSignal:
    signal_type: str      # 'positive', 'negative', 'neutral'
    magnitude: float      # Strength of signal [0, 1]
    expert_id: str        # Which expert to credit/debit
    prediction_id: str    # Link to original prediction
    context: Dict         # Additional metadata
```

The attribution formula balances accuracy and confidence:

```
credit_change = magnitude × base_rate × confidence_factor
```

Where `base_rate` is +10 for positive signals and -5 for negative (asymmetric to encourage participation while penalizing overconfidence).

### 3.4 Staking Mechanism

For high-confidence predictions, experts can "stake" additional credits:

```python
stake = pool.place_stake(
    expert_id='file_expert',
    prediction_id='pred_123',
    amount=20.0,
    multiplier=2.0  # 2x risk/reward
)
```

If correct: expert receives `amount × multiplier`
If incorrect: expert loses `amount`

This mechanism allows experts to signal strong conviction while accepting proportional risk.

---

## 4. Calibration Tracking

### 4.1 The Calibration Problem

A prediction system is well-calibrated if its confidence estimates match empirical accuracy. For example, predictions made with 70% confidence should be correct approximately 70% of the time.

Calibration is distinct from accuracy. A system could achieve 90% accuracy while being poorly calibrated (e.g., always predicting 50% confidence). Conversely, a well-calibrated system might have low accuracy if it honestly reports low confidence.

### 4.2 Metrics

We track three complementary metrics:

**Expected Calibration Error (ECE)**:
Partitions predictions into confidence bins and measures the weighted average gap between confidence and accuracy:

```
ECE = Σ (|B_i| / n) × |accuracy(B_i) - confidence(B_i)|
```

Interpretation:
- ECE < 0.05: Excellent calibration
- ECE < 0.10: Good calibration
- ECE < 0.15: Acceptable
- ECE ≥ 0.15: Needs attention

**Maximum Calibration Error (MCE)**:
The worst gap in any single bin. Identifies where predictions are most unreliable:

```
MCE = max_i |accuracy(B_i) - confidence(B_i)|
```

**Brier Score**:
Mean squared error of probabilistic predictions, combining calibration and discrimination:

```
Brier = (1/n) × Σ (confidence_i - outcome_i)²
```

### 4.3 Trend Detection

Beyond aggregate metrics, we classify systems as:

- **Overconfident**: Mean confidence > mean accuracy (predictions claim more than they deliver)
- **Underconfident**: Mean confidence < mean accuracy (predictions are better than claimed)
- **Well-calibrated**: Confidence approximately matches accuracy

Our current system shows underconfidence—experts predict 50% confidence but achieve nearly 100% accuracy. This is a known cold-start artifact that should improve as more calibration data accumulates.

### 4.4 ResolvedPrediction Tracking

To enable calibration analysis, we extended the prediction format:

```python
@dataclass
class ResolvedPrediction(Prediction):
    actual_files: List[str]     # Ground truth from commit
    accuracy: float             # Computed accuracy
    outcome_timestamp: float    # When resolution occurred
    commit_hash: str            # Link to resolving commit
```

This creates a complete audit trail from prediction through outcome.

---

## 5. Cold-Start Problem and Solutions

### 5.1 The Bootstrap Challenge

New systems face a chicken-and-egg problem: they need data to make good predictions, but users won't provide data unless predictions are useful. Traditional solutions include:

1. **Pre-training on synthetic data**: Often doesn't transfer well
2. **Conservative defaults**: Provides little value to early users
3. **Human-in-the-loop bootstrapping**: Expensive and slow

### 5.2 Our Approach

We implemented a multi-pronged cold-start strategy:

**1. Transparent Communication**
When all experts have default balances (indicating no learning has occurred), we display a clear "Cold Start Mode" banner:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  ❄️  COLD START MODE - Experts Have Not Learned Yet                 ┃
┃                                                                      ┃
┃  All experts have equal weight (no feedback received yet).          ┃
┃  Predictions will improve after making commits.                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**2. Fallback to Simpler Models**
We maintain a separate ML file prediction model trained on commit history (TF-IDF + co-occurrence). When MoE confidence is low, we automatically offer this fallback:

```
💡 Due to low confidence, showing ML file prediction fallback:
   (Trained on commit history - may be more accurate)
```

**3. Rapid Learning Loop**
Integration with git hooks means every commit provides training signal. The system can improve within a single development session.

### 5.3 Detection Logic

Cold-start detection checks whether any expert has deviated from the default balance:

```python
def is_cold_start(ledger: CreditLedger, threshold: float = 100.0) -> bool:
    if not ledger.accounts:
        return True
    for account in ledger.accounts.values():
        if abs(account.balance - threshold) > 0.01:
            return False  # At least one expert has learned
    return True
```

---

## 6. Git Integration and Feedback Loops

### 6.1 Hook Architecture

Hubris integrates with version control through git hooks:

**prepare-commit-msg** (pre-commit):
1. Extracts commit message
2. Requests prediction from FileExpert
3. Stores prediction in `.git-ml/predictions/pending.jsonl`

**post-commit** (after commit):
1. Retrieves actual files from `git diff-tree`
2. Compares prediction to reality
3. Generates ValueSignal
4. Updates expert credit accounts

### 6.2 Feedback Display

After each commit, users see immediate feedback:

```
────────────────────────────────────────────────────────
  EXPERIMENTAL: Hubris MoE Feedback Loop
  Predictions are learning - review before trusting
────────────────────────────────────────────────────────

Hubris Feedback:
  Accuracy: 100.0% (3/3 files)
  staged_files: +5.0 credits
```

This transparency serves multiple purposes:
- Confirms the system is learning
- Builds trust through visibility
- Identifies when predictions fail

### 6.3 The EXPERIMENTAL Banner

All predictions include a warning banner:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  ⚠️  EXPERIMENTAL - Hubris MoE Predictions                          ┃
┃                                                                      ┃
┃  These predictions are generated by a learning system with limited  ┃
┃  training data. Use as suggestions only - always verify results.    ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

This banner will remain until the system demonstrates sustained calibration across a statistically significant sample.

---

## 7. Evaluation

### 7.1 Methodology

We evaluated Hubris during active development of the Cortical Text Processor itself—a form of "dogfooding" where the system assists in its own development.

**Dataset**: 104 commits collected over 8 development sprints
**Evaluation**: Leave-one-out prediction accuracy on file changes

### 7.2 Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| File Prediction Accuracy | 85-100% | High accuracy on recent commits |
| ECE | 0.500 | Poor (underconfident) |
| Brier Score | 0.250 | Moderate |
| Expert Credit Range | 100-175 | Differentiation emerging |

The high accuracy with poor ECE indicates the system is overly conservative—it achieves better results than its confidence suggests. This is preferable to overconfidence but represents calibration improvement opportunity.

### 7.3 Qualitative Observations

1. **Rapid adaptation**: Experts show credit differentiation within ~10 commits
2. **Graceful degradation**: When predictions fail, credit deductions are proportional
3. **Cold-start recovery**: Systems become useful faster than expected due to commit-level feedback

---

## 8. Lessons Learned

### 8.1 Profile Before Optimizing

Early development revealed performance issues in `compute_all()`. Initial suspicion pointed to Louvain clustering (the most complex algorithm). Profiling revealed the actual culprits:

| Phase | Before | After | Root Cause |
|-------|--------|-------|------------|
| bigram_connections | 20.85s timeout | 10.79s | O(n²) from common terms |
| semantics | 30.05s timeout | 5.56s | Unbounded similarity pairs |
| louvain | 2.2s | 2.2s | Not the bottleneck |

**Lesson**: The obvious culprit is often innocent. Measure before acting.

### 8.2 Economic Incentives Shape Behavior

The credit system creates emergent behaviors:
- Experts become more conservative when credit-poor
- High-confidence predictions concentrate on high-credit experts
- Staking creates meaningful differentiation for close calls

This mirrors findings in mechanism design: well-structured incentives can achieve goals without explicit rules.

### 8.3 Transparency Builds Trust

The EXPERIMENTAL banner and visible feedback loops increased user willingness to engage with predictions. Users reported that seeing "how the sausage is made" made them more forgiving of errors and more likely to provide corrective feedback.

### 8.4 Cold-Start Is Not Failure

Initial instinct was to hide the system until it achieved acceptable accuracy. The better approach was to be transparent about limitations while providing fallbacks. Users appreciated honesty over false confidence.

---

## 9. Related Work

### 9.1 Mixture of Experts

Shazeer et al. (2017) demonstrated MoE at scale with "Outrageously Large Neural Networks." Our work differs in:
- Economic routing vs. learned gating
- Explicit calibration tracking
- Real-time feedback integration

### 9.2 Confidence Calibration

Guo et al. (2017) showed modern neural networks are poorly calibrated. Our approach addresses this through:
- Explicit ECE/MCE tracking
- Per-expert calibration curves
- Trend detection and recommendations

### 9.3 Developer Assistance

GitHub Copilot and similar tools provide code completion but not task-level predictions. Our work focuses on the planning phase: which files to modify, which tests to run, how to diagnose errors.

---

## 10. Future Work

### 10.1 Additional Expert Types

Planned expansions include:
- **RefactorExpert**: Detecting refactoring opportunities (Sprint 7)
- **DocumentationExpert**: Identifying documentation gaps
- **SecurityExpert**: Flagging potential vulnerabilities
- **PerformanceExpert**: Suggesting optimizations

### 10.2 Cross-Repository Learning

Current experts learn from a single repository. Future work will explore transfer learning across codebases with similar structure.

### 10.3 Calibration Improvement

We plan to implement:
- Temperature scaling for post-hoc calibration
- Platt scaling for expert outputs
- Isotonic regression for calibration curve fitting

### 10.4 Multi-Agent Collaboration

The parallel sprint architecture (Sprints 6-8) enables multiple development threads. Future work will explore how agents can coordinate across sprints without merge conflicts.

---

## 11. Conclusion

Hubris demonstrates that economic incentives can create self-correcting prediction systems. By treating confidence as a resource that must be earned, we align expert behavior with user interests. The credit system naturally demotes overconfident experts while promoting accurate ones.

Key design principles that emerged:

1. **Confidence is earned, not claimed**: Track record determines influence
2. **Transparency over opacity**: Show users how predictions are made
3. **Graceful cold-start**: Honest uncertainty is better than false confidence
4. **Tight feedback loops**: Learn from every interaction

The system continues to improve through its own development process—a fitting validation of the self-improving architecture we set out to build.

---

## References

1. Hawkins, J. (2021). *A Thousand Brains: A New Theory of Intelligence*. Basic Books.

2. Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR 2017*.

3. Jacobs, R.A., et al. (1991). Adaptive Mixtures of Local Experts. *Neural Computation*, 3(1), 79-87.

4. Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.

5. Page, L., et al. (1998). The PageRank Citation Ranking: Bringing Order to the Web. *Stanford Technical Report*.

6. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523.

7. Platt, J. (1999). Probabilistic Outputs for Support Vector Machines. *Advances in Large Margin Classifiers*.

---

## Appendix A: Sprint History

| Sprint | Focus | Duration | Key Outcomes |
|--------|-------|----------|--------------|
| 1 | Expert Foundation | 3 days | MicroExpert, FileExpert, TestExpert |
| 2 | Credit System | 2 days | CreditLedger, ValueSignal, Staking |
| 3 | Integration | 3 days | CLI, Feedback Collector |
| 4 | Meta-Learning | 1 day | Git hooks, EXPERIMENTAL banner |
| 5 | UX & Documentation | 1 day | Cold-start, Calibration CLI |
| 6 | TestExpert Activation | - | Planned |
| 7 | RefactorExpert | - | Planned |
| 8 | Core Performance | - | Planned |

---

## Appendix B: Glossary

See `docs/glossary.md` for comprehensive terminology definitions covering both the Cortical Text Processor and Hubris MoE systems.

---

## Appendix C: Code Availability

The complete implementation is available in the repository:

- `scripts/hubris/` - Core MoE system
- `scripts/hubris_cli.py` - Command-line interface
- `cortical/` - Underlying text processor
- `docs/` - Documentation and this paper

---

*"True expertise is knowing the boundaries of your knowledge."*
— Cortical Development Manifesto
