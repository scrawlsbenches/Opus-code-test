# Hubris - Mixture of Experts (MoE) System

**Micro-model system for specialized coding task predictions**

Hubris is a Mixture of Experts system designed to assist Claude Code with predictions about file changes, test selection, error diagnosis, and workflow patterns. The name "Hubris" reflects the system's goal: **experts that learn to be confident in their predictions** through a credit-based economy where successful predictions earn rewards and poor predictions incur costs.

## Table of Contents

- [Overview](#overview)
- [Why "Hubris"?](#why-hubris)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Credit System](#credit-system)
- [Training](#training)
- [CLI Usage](#cli-usage)
- [Extending](#extending)
- [Expert Details](#expert-details)

---

## Overview

Hubris implements a Mixture of Experts (MoE) architecture inspired by **Thousand Brains Theory**, where multiple specialized cortical columns vote to reach consensus. Instead of a single monolithic model, Hubris uses:

- **4 specialized micro-experts**, each trained on specific aspects of coding tasks
- **Confidence-weighted voting** to aggregate predictions
- **Credit-based routing** where experts with better track records get more influence
- **Staking mechanism** for experts to bet on high-confidence predictions

### Key Goals

| Goal | Expert | Description |
|------|--------|-------------|
| **File Prediction** | FileExpert | Which files need modification for a task? |
| **Test Prediction** | TestExpert | Which tests should run for code changes? |
| **Error Diagnosis** | ErrorDiagnosisExpert | What's causing this error and how to fix it? |
| **Workflow Learning** | EpisodeExpert | What action should come next? |

---

## Why "Hubris"?

The name "Hubris" (Greek: excessive confidence) is ironic and intentional. In this system:

1. **Experts start with equal confidence** - All experts begin with 100 credits
2. **Confidence is earned through accuracy** - Successful predictions increase credit balance
3. **Overconfidence is punished** - Staking high on wrong predictions loses credits
4. **Humility emerges naturally** - Low-performing experts get less routing weight

The credit system creates a **meritocracy** where experts must **earn the right to be confident** through demonstrated value.

---

## Architecture

```
hubris/
├── Base Classes
│   ├── micro_expert.py          # MicroExpert, ExpertPrediction, ExpertMetrics
│   ├── voting_aggregator.py     # Confidence-weighted voting
│   └── expert_router.py          # Intent-based expert selection
│
├── Experts (Specialized Models)
│   ├── experts/file_expert.py   # File prediction (TF-IDF + co-occurrence)
│   ├── experts/test_expert.py   # Test selection (naming + history)
│   ├── experts/error_expert.py  # Error diagnosis (patterns + stack traces)
│   └── experts/episode_expert.py # Workflow learning (action sequences)
│
├── Orchestration
│   └── expert_consolidator.py   # Unified training/prediction hub
│
└── Credit System (Value Economy)
    ├── credit_account.py         # CreditLedger, CreditAccount
    ├── value_signal.py           # ValueSignal, ValueAttributor
    ├── credit_router.py          # Credit-weighted routing
    └── staking.py                # Confidence staking mechanism
```

### Component Interaction

```
User Query
    ↓
ExpertRouter
(classify intent: fix_bug, add_feature, debug_error, etc.)
    ↓
ExpertConsolidator
(load relevant experts)
    ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ FileExpert  │ TestExpert  │ ErrorExpert │EpisodeExpert│
│  (credit:   │  (credit:   │  (credit:   │  (credit:   │
│   120.5)    │   98.2)     │   110.0)    │   105.3)    │
└─────────────┴─────────────┴─────────────┴─────────────┘
    ↓             ↓             ↓             ↓
    Prediction   Prediction   Prediction   Prediction
    ↓─────────────────────────────────────────↓
VotingAggregator / CreditRouter
(weight by confidence × credit balance)
    ↓
AggregatedPrediction
(ranked items with consensus metrics)
    ↓
User receives prediction
    ↓
Outcome observed (test pass/fail, files modified, etc.)
    ↓
ValueSignal generated
    ↓
ValueAttributor applies credit/debit to expert accounts
    ↓
Expert credit balances updated → routing weights adjusted
```

---

## Quick Start

### Installation

The Hubris system is already integrated into this project. No additional dependencies required.

### Basic Prediction

```python
from pathlib import Path
from scripts.hubris.expert_consolidator import ExpertConsolidator

# Initialize consolidator and load trained experts
consolidator = ExpertConsolidator()
consolidator.load_all_experts(Path('.git-ml/models'))

# Get ensemble prediction for a task
prediction = consolidator.get_ensemble_prediction({
    'query': 'Add authentication feature',
    'recent_files': ['auth.py', 'login.py']
})

# View results
print("Top file predictions:")
for file, confidence in prediction.items[:5]:
    print(f"  {file}: {confidence:.3f}")

print(f"\nContributing experts: {prediction.contributing_experts}")
print(f"Disagreement score: {prediction.disagreement_score:.3f}")
print(f"Overall confidence: {prediction.confidence:.3f}")
```

### Using Specific Experts

```python
from scripts.hubris.experts.file_expert import FileExpert
from scripts.hubris.experts.test_expert import TestExpert

# File prediction
file_expert = FileExpert.load(Path('.git-ml/models/file_expert.json'))
file_pred = file_expert.predict({
    'query': 'Fix search bug',
    'seed_files': ['cortical/query/search.py'],
    'top_n': 10
})

# Test prediction
test_expert = TestExpert.load(Path('.git-ml/models/test_expert.json'))
test_pred = test_expert.predict({
    'changed_files': ['cortical/query/search.py', 'cortical/analysis.py'],
    'query': 'search improvements'
})

# Error diagnosis
from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

error_expert = ErrorDiagnosisExpert()
diagnosis = error_expert.diagnose(
    error_message="TypeError: 'NoneType' object is not subscriptable",
    stack_trace="..."
)
print(f"Error type: {diagnosis['error_type']}")
print(f"Likely causes: {diagnosis['likely_causes']}")
print(f"Suggested fixes: {diagnosis['suggested_fixes']}")
```

---

## Credit System

The credit system creates a **value-based economy** where experts earn influence through accurate predictions.

### How It Works

1. **Initial Balance**: Each expert starts with **100 credits**
2. **Earning Credits**: Successful predictions earn credits based on value signals
3. **Losing Credits**: Failed predictions lose credits
4. **Routing Weight**: Credit balance determines influence in ensemble voting

### Value Signals

Value signals provide feedback about prediction quality:

| Signal Type | Source | Credit Awarded |
|-------------|--------|----------------|
| **Positive** | Test passed, files predicted correctly | `magnitude × 10.0 × confidence` |
| **Negative** | Test failed, wrong file predictions | `magnitude × -5.0 × confidence` |
| **Neutral** | Inconclusive outcome | `0.0` |

### Credit Attribution

```python
from scripts.hubris.credit_account import CreditLedger
from scripts.hubris.value_signal import ValueAttributor, ValueSignal

# Initialize
ledger = CreditLedger()
attributor = ValueAttributor()

# Create a value signal
signal = ValueSignal(
    signal_type='positive',
    magnitude=0.8,
    timestamp=time.time(),
    source='test_result',
    expert_id='file_expert',
    prediction_id='pred_123',
    context={'confidence': 0.9, 'test_passed': True}
)

# Apply to ledger (credits expert account)
amount = attributor.process_signal(signal, ledger)
print(f"Expert credited: {amount:.2f} credits")

# Check balance
account = ledger.get_or_create_account('file_expert')
print(f"New balance: {account.balance:.2f}")
```

### Credit-Weighted Routing

```python
from scripts.hubris.credit_router import CreditRouter

# Initialize with ledger
router = CreditRouter(ledger, min_weight=0.1, temperature=1.0)

# Compute weights based on credit balances
weights = router.compute_weights(['file_expert', 'test_expert', 'error_expert'])

for expert_id, weight_info in weights.items():
    print(f"{expert_id}:")
    print(f"  Raw weight: {weight_info.raw_weight:.3f}")
    print(f"  Normalized: {weight_info.normalized_weight:.3f}")
    print(f"  Confidence boost: {weight_info.confidence_boost:.3f}")

# Aggregate predictions with credit weighting
predictions = {
    'file_expert': file_pred,
    'test_expert': test_pred
}
aggregated = router.aggregate_predictions(predictions)
```

**Weight calculation:**
1. Apply softmax to credit balances with temperature control
2. Normalize to sum to 1.0
3. Apply minimum weight floor (default: 0.1)
4. Renormalize after floor
5. Boost high-credit experts (>150 credits get up to +10% confidence)

### Staking Mechanism

Experts can **stake credits** on high-confidence predictions to earn multiplied rewards:

```python
from scripts.hubris.staking import StakePool, StakeStrategy

# Initialize stake pool
pool = StakePool(ledger, max_stake_ratio=0.5, min_stake=5.0)

# Place a stake
stake = pool.place_stake(
    expert_id='file_expert',
    prediction_id='pred_456',
    amount=20.0,
    multiplier=2.0  # 2x risk/reward
)

# Later, resolve based on outcome
if prediction_was_correct:
    net_gain = pool.resolve_stake(stake.stake_id, success=True)
    # Returns +20.0 (20 × 2.0 = 40 payout, minus 20 original = 20 profit)
else:
    net_loss = pool.resolve_stake(stake.stake_id, success=False)
    # Returns -20.0 (stake forfeited)
```

**Staking strategies:**
- `CONSERVATIVE`: 1.0x multiplier (no risk)
- `MODERATE`: 1.5x multiplier
- `AGGRESSIVE`: 2.0x multiplier
- `YOLO`: 3.0x multiplier (maximum risk)

**Auto-staking** based on confidence:
```python
from scripts.hubris.staking import AutoStaker

auto_staker = AutoStaker(pool)

# Decide whether to stake based on prediction confidence
decision = auto_staker.decide_stake(
    expert_id='file_expert',
    prediction=file_pred,
    strategy=StakeStrategy.AGGRESSIVE
)

if decision:
    amount, multiplier = decision
    stake = pool.place_stake('file_expert', 'pred_789', amount, multiplier)
```

---

## Training

Each expert trains on different data sources:

| Expert | Data Source | What It Learns |
|--------|-------------|----------------|
| **FileExpert** | Commit history | File co-occurrence, commit types, keyword associations |
| **TestExpert** | Commit history | Source-to-test mappings, failure patterns, naming conventions |
| **ErrorDiagnosisExpert** | Error records | Error-to-file mappings, stack trace patterns, common causes |
| **EpisodeExpert** | Session transcripts | Action sequences, context-to-action mappings, success patterns |

### Consolidated Training

```python
from scripts.hubris.expert_consolidator import ExpertConsolidator, train_all_experts

# Option 1: Train all experts with data router
consolidator = train_all_experts(
    commits=commit_history,
    transcripts=session_transcripts,
    errors=error_records,
    model_dir=Path('.git-ml/models')
)

# Option 2: Manual training per expert
consolidator = ExpertConsolidator()
consolidator.create_all_experts()

results = consolidator.consolidate_training(
    commits=commit_history,
    transcripts=session_transcripts,
    errors=error_records
)

print(f"Training results: {results}")

# Save all experts atomically
consolidator.save_all_experts(Path('.git-ml/models'))
```

### Individual Expert Training

```python
# FileExpert (uses existing ml_file_prediction model)
from scripts.ml_file_prediction import FilePredictionModel
from scripts.hubris.experts.file_expert import FileExpert

v1_model = FilePredictionModel.load('.git-ml/models/file_prediction.json')
file_expert = FileExpert.from_v1_model(v1_model)

# TestExpert
from scripts.hubris.experts.test_expert import TestExpert

test_expert = TestExpert()
test_expert.train(commits)  # Learns source-to-test mappings
test_expert.save(Path('.git-ml/models/test_expert.json'))

# ErrorDiagnosisExpert
from scripts.hubris.experts.error_expert import ErrorDiagnosisExpert

error_expert = ErrorDiagnosisExpert()
error_records = [
    {
        'error_type': 'TypeError',
        'error_message': 'unsupported operand type',
        'files_modified': ['cortical/analysis.py'],
        'resolution': 'Added type check'
    }
]
error_expert.train(error_records)

# EpisodeExpert
from scripts.hubris.experts.episode_expert import EpisodeExpert

episode_expert = EpisodeExpert()
episodes = EpisodeExpert.extract_episodes(transcript_exchanges)
episode_expert.train(episodes)
```

### Evaluation Metrics

```python
# Check expert performance
for expert_type, expert in consolidator.experts.items():
    if expert.metrics:
        print(f"\n{expert_type} Expert Metrics:")
        print(f"  MRR: {expert.metrics.mrr:.3f}")
        print(f"  Recall@5: {expert.metrics.recall_at_k.get(5, 0):.3f}")
        print(f"  Recall@10: {expert.metrics.recall_at_k.get(10, 0):.3f}")
        print(f"  Precision@1: {expert.metrics.precision_at_k.get(1, 0):.3f}")
        print(f"  Calibration error: {expert.metrics.calibration_error:.3f}")
        print(f"  Test examples: {expert.metrics.test_examples}")
```

---

## CLI Usage

**Note:** The `hubris_cli.py` is being created by a parallel agent. Reference implementation:

```bash
# Train all experts on latest data
python scripts/hubris_cli.py train

# Get prediction for a task
python scripts/hubris_cli.py predict "Add authentication feature"

# Get prediction with seed files
python scripts/hubris_cli.py predict "Fix search bug" --seed cortical/query/search.py

# View expert statistics
python scripts/hubris_cli.py stats

# View credit leaderboard
python scripts/hubris_cli.py leaderboard

# Diagnose an error
python scripts/hubris_cli.py diagnose --error "TypeError: 'NoneType' object is not subscriptable"

# Suggest tests for changed files
python scripts/hubris_cli.py suggest-tests --files cortical/query/search.py cortical/analysis.py

# View routing decisions
python scripts/hubris_cli.py route "Debug authentication error"
```

---

## Extending

### Adding a New Expert

1. **Create expert class** inheriting from `MicroExpert`:

```python
from scripts.hubris.micro_expert import MicroExpert, ExpertPrediction

class MyExpert(MicroExpert):
    def __init__(self, expert_id="my_expert", version="1.0.0", **kwargs):
        kwargs.pop('expert_type', None)
        super().__init__(
            expert_id=expert_id,
            expert_type="my_type",
            version=version,
            **kwargs
        )

        if not self.model_data:
            self.model_data = {
                'pattern_data': {},
                'total_examples': 0
            }

    def predict(self, context: Dict[str, Any]) -> ExpertPrediction:
        query = context.get('query', '')

        # Your prediction logic here
        items = self._score_items(query)

        return ExpertPrediction(
            expert_id=self.expert_id,
            expert_type=self.expert_type,
            items=items,
            metadata={'query': query}
        )

    def _score_items(self, query: str) -> List[Tuple[str, float]]:
        # Implement your scoring logic
        return [("item1", 0.9), ("item2", 0.7)]

    def train(self, training_data: List[Dict[str, Any]]) -> None:
        # Implement training logic
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MyExpert':
        metrics = ExpertMetrics.from_dict(data['metrics']) if data.get('metrics') else None
        return cls(
            expert_id=data.get('expert_id', 'my_expert'),
            version=data['version'],
            created_at=data['created_at'],
            trained_on_commits=data['trained_on_commits'],
            trained_on_sessions=data['trained_on_sessions'],
            git_hash=data['git_hash'],
            model_data=data['model_data'],
            metrics=metrics,
            calibration_curve=data.get('calibration_curve')
        )
```

2. **Register in ExpertConsolidator**:

```python
# In expert_consolidator.py
from experts.my_expert import MyExpert

class ExpertConsolidator:
    EXPERT_CLASSES = {
        'file': FileExpert,
        'test': TestExpert,
        'error': ErrorDiagnosisExpert,
        'episode': EpisodeExpert,
        'my_type': MyExpert,  # Add your expert
    }
```

3. **Add routing patterns** (optional):

```python
# In expert_router.py
INTENT_TO_EXPERTS = {
    'my_intent': ['my_type', 'file'],
    # ...
}

INTENT_PATTERNS = {
    'my_intent': [
        r'\bmy_keyword\b',
        r'\bmy_pattern\b',
    ],
    # ...
}
```

### ExpertPrediction Format

All experts must return `ExpertPrediction`:

```python
@dataclass
class ExpertPrediction:
    expert_id: str              # "file_expert"
    expert_type: str            # "file"
    items: List[Tuple[str, float]]  # [("file.py", 0.95), ...]
    metadata: Dict[str, Any]    # Additional context
```

**Best practices:**
- **Items**: Sorted by confidence descending
- **Confidence**: Calibrated 0-1 scores (use `get_confidence_calibration()`)
- **Metadata**: Include query, keywords, signals used, etc.
- **Limit**: Return top 10-20 items to avoid noise

### Integration Points

```python
# Prediction flow
query = "Fix authentication bug"

# 1. Route to experts
router = ExpertRouter()
decision = router.route(query)
expert_types = decision.expert_types  # ['file', 'error', 'test']

# 2. Get predictions from each expert
predictions = []
for expert_type in expert_types:
    expert = consolidator.get_expert(expert_type)
    pred = expert.predict({'query': query})
    predictions.append(pred)

# 3. Aggregate
aggregator = VotingAggregator()
result = aggregator.aggregate(predictions)

# 4. Generate value signal from outcome
signal = attributor.attribute_from_commit_result(
    expert_id='file_expert',
    prediction_id='pred_123',
    files_correct=['auth.py'],
    files_total=['auth.py', 'login.py'],
    confidence=0.8
)

# 5. Apply to ledger
attributor.process_signal(signal, ledger)
```

---

## Expert Details

### FileExpert

**Purpose**: Predict which files need modification for a given task

**Training Data**: Commit history with file changes and messages

**Model Components**:
- File co-occurrence matrix (which files change together)
- Commit type patterns (feat:, fix:, docs:)
- Keyword to file associations
- File change frequency

**Prediction Signals**:
1. Commit type match (2.0x weight)
2. Keyword match (1.5x weight)
3. Co-occurrence with seed files (3.0x weight)
4. Semantic similarity (0.5x weight, optional)

**Example**:
```python
prediction = file_expert.predict({
    'query': 'feat: Add graph boosted search',
    'seed_files': ['cortical/query/search.py'],
    'use_semantic': True,
    'top_n': 10
})
```

---

### TestExpert

**Purpose**: Predict which tests to run for code changes

**Training Data**: Commit history with source and test file co-changes

**Model Components**:
- Source-to-test mappings (historical)
- Naming convention patterns
- Module-level test associations
- Failure patterns (tests that often fail with certain files)

**Prediction Signals**:
1. Naming convention (3.0x weight)
2. Historical source-to-test mapping (2.0x weight)
3. Failure patterns (2.5x weight)
4. Module-level mapping (1.5x weight)
5. Query keyword match (1.0x weight)

**Example**:
```python
prediction = test_expert.predict({
    'changed_files': ['cortical/query/search.py', 'cortical/analysis.py'],
    'query': 'search improvements',
    'top_n': 10
})
```

---

### ErrorDiagnosisExpert

**Purpose**: Diagnose errors and suggest fixes

**Training Data**: Error records with resolutions

**Model Components**:
- Error type to file mappings
- Stack trace patterns
- Keyword to fix associations
- Common causes per error type
- Resolution history

**Prediction Signals**:
1. Error type to files (2.0x weight)
2. Stack trace analysis (2.5x weight)
3. Keyword-based fixes (1.5x weight)
4. Historical resolutions (1.8x weight)

**Example**:
```python
diagnosis = error_expert.diagnose(
    error_message="TypeError: 'NoneType' object is not subscriptable",
    stack_trace="Traceback (most recent call last):\n  File \"cortical/query/search.py\", line 42..."
)
```

**Diagnosis Output**:
```python
{
    'error_type': 'TypeError',
    'category': 'type',
    'likely_causes': [
        ('Object is None', 0.95),
        ('Missing None check', 0.80)
    ],
    'suggested_fixes': [
        ('Check for None before subscripting', 0.90),
        ('Add type validation', 0.75)
    ],
    'files_to_check': [
        ('cortical/query/search.py', 0.85)
    ]
}
```

---

### EpisodeExpert

**Purpose**: Learn workflow patterns from session transcripts

**Training Data**: Session transcript exchanges (query, tools used, outcomes)

**Model Components**:
- Action sequence patterns (Read → Edit → Bash)
- Context to action mappings
- Success patterns
- Failure patterns (actions to avoid)

**Prediction Signals**:
1. Action sequence continuation (2.5x weight)
2. Context keyword matching (2.0x weight)
3. File type patterns (1.5x weight)
4. Success pattern matching (1.8x weight)
5. Failure avoidance (negative weight)

**Example**:
```python
prediction = episode_expert.predict({
    'query': 'Fix authentication bug',
    'last_actions': ['Read', 'Grep'],
    'files_touched': ['auth.py'],
    'top_n': 5
})
```

**Action Categories**:
- `read`: Read, Grep, Glob
- `write`: Write, Edit, MultiEdit, NotebookEdit
- `execute`: Bash, BashOutput
- `search`: Grep, Glob, WebSearch, WebFetch
- `organize`: TodoWrite, Task, SlashCommand, Skill

---

## Performance Tuning

### Temperature Control

Credit routing uses **softmax with temperature** to control weight distribution:

```python
# Lower temperature (0.5) = sharper distinctions (winner-takes-most)
router = CreditRouter(ledger, temperature=0.5)

# Higher temperature (2.0) = smoother distribution (more democratic)
router = CreditRouter(ledger, temperature=2.0)
```

### Minimum Weight Floor

Prevent experts from being completely ignored:

```python
# Allow 10% minimum influence even for low-credit experts
router = CreditRouter(ledger, min_weight=0.1)

# Stricter meritocracy (5% minimum)
router = CreditRouter(ledger, min_weight=0.05)
```

### Disagreement Penalty

Reduce confidence when experts disagree:

```python
config = AggregationConfig(
    disagreement_penalty=0.3  # Reduce score by 30% of disagreement
)
aggregated = aggregator.aggregate(predictions, config)
```

### Confidence Calibration

Apply calibration curves to raw predictions:

```python
# Calibration curve: [(predicted, actual)]
calibration_curve = [
    (0.0, 0.0),
    (0.5, 0.3),
    (0.7, 0.5),
    (0.9, 0.8),
    (1.0, 1.0)
]

expert.calibration_curve = calibration_curve
expert.save(path)

# Predictions automatically use calibrated confidence
pred = expert.predict(context)
```

---

## Design Principles

1. **Modularity**: Each expert is independent and can be trained/evaluated separately
2. **Composability**: Experts combine through voting, not monolithic integration
3. **Accountability**: Credit system tracks which experts add value
4. **Adaptability**: Routing weights adjust based on performance
5. **Interpretability**: Predictions include metadata about signals used
6. **Incrementality**: Experts can be added/removed without retraining others

---

## Future Extensions

Potential new experts:

- **RefactorExpert**: Suggest refactoring opportunities
- **DocumentationExpert**: Predict documentation needs
- **ReviewExpert**: Code review suggestions
- **PerformanceExpert**: Performance optimization suggestions
- **SecurityExpert**: Security vulnerability detection
- **DependencyExpert**: Dependency update recommendations

---

## References

- **Thousand Brains Theory**: Jeff Hawkins, "A Thousand Brains" (2021)
- **Mixture of Experts**: Shazeer et al., "Outrageously Large Neural Networks" (2017)
- **TF-IDF**: Salton & Buckley, "Term-weighting approaches in automatic text retrieval" (1988)
- **PageRank**: Page et al., "The PageRank Citation Ranking" (1998)

---

## License

Part of the Cortical Text Processor project. See main repository for license details.
