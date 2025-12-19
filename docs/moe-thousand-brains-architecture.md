# Micro-Model Mixture of Experts Architecture

## Inspired by Thousand Brains Theory

This document outlines an architecture for training specialized micro-models per conversation thread, then combining them using a Mixture of Experts (MoE) approach inspired by Jeff Hawkins' Thousand Brains Theory.

### Core Insight

In Thousand Brains Theory, the neocortex consists of thousands of "cortical columns," each building a complete model of an object. These columns vote to reach consensus. Similarly, we can train many small, specialized experts that each model different aspects of coding tasks, then aggregate their predictions through voting.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Micro-Model MoE System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │ Session 1    │   │ Session 2    │   │ Session N    │        │
│  │ Transcript   │   │ Transcript   │   │ Transcript   │   ...  │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘        │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              Episode Expert Trainer                   │      │
│  │  (Extracts patterns from each session)                │      │
│  └──────────────────────────┬───────────────────────────┘      │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ FileExpert  │    │ TestExpert  │    │ ErrorExpert │   ...  │
│  │ (prediction)│    │ (selection) │    │ (diagnosis) │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                │
│                            ▼                                    │
│               ┌─────────────────────────┐                      │
│               │     Expert Router       │                      │
│               │  (intent classification)│                      │
│               └────────────┬────────────┘                      │
│                            ▼                                    │
│               ┌─────────────────────────┐                      │
│               │   Voting Aggregator     │                      │
│               │ (confidence-weighted)   │                      │
│               └────────────┬────────────┘                      │
│                            ▼                                    │
│                    Final Prediction                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Expert Types

### 1. FileExpert (Existing - to be migrated)
**Purpose:** Predict which files to modify for a given task
**Input:** Task description, optional seed files
**Output:** Ranked list of (file_path, confidence) tuples
**Training Data:** Commit history with file changes

### 2. TestExpert (New)
**Purpose:** Predict which tests to run for a code change
**Input:** Modified files, change description
**Output:** Ranked list of (test_file, confidence) tuples
**Training Data:**
- Git commits with their CI test results
- File-to-test co-occurrence patterns
- Failed test → fixed file associations

### 3. ErrorDiagnosisExpert (New)
**Purpose:** Predict source of errors from error messages
**Input:** Error message, stack trace
**Output:** Ranked list of (file_path, line_range, confidence) tuples
**Training Data:**
- Chat sessions with error debugging
- Stack traces → fix commit associations
- Error pattern → file mappings

### 4. DocumentationExpert (New)
**Purpose:** Predict which docs need updating after code changes
**Input:** Modified code files, change description
**Output:** Ranked list of (doc_file, confidence) tuples
**Training Data:**
- Code change → doc update co-occurrence
- API changes → README/docstring updates
- Feature commits → changelog entries

### 5. RefactorExpert (New)
**Purpose:** Predict cascade effects of refactoring
**Input:** Files being refactored, refactor type
**Output:** Ranked list of (affected_file, impact_type, confidence) tuples
**Training Data:**
- Large refactor commits (many files changed together)
- Import graph relationships
- Test file associations

### 6. ReviewExpert (Future)
**Purpose:** Predict likely review comments/issues
**Input:** Diff, file context
**Output:** Ranked list of (issue_type, location, confidence) tuples
**Training Data:**
- PR comments and requested changes
- Common code review patterns
- Style guide violations

---

## Data Classes

```python
@dataclass
class MicroExpert:
    """Base class for all micro-experts."""
    expert_id: str              # Unique identifier
    expert_type: str            # 'file', 'test', 'error', 'doc', 'refactor', 'review'
    version: str                # Semantic version
    created_at: str             # ISO timestamp
    trained_on_commits: int     # Number of training examples
    trained_on_sessions: int    # Number of sessions contributing
    git_hash: str               # Git commit at training time

    # Expert-specific model data (varies by type)
    model_data: Dict[str, Any]

    # Performance metrics
    metrics: Optional[ExpertMetrics] = None

    # Confidence calibration
    calibration_curve: Optional[List[Tuple[float, float]]] = None


@dataclass
class ExpertMetrics:
    """Performance metrics for an expert."""
    mrr: float = 0.0                    # Mean Reciprocal Rank
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    calibration_error: float = 0.0      # Expected calibration error
    test_examples: int = 0


@dataclass
class ExpertPrediction:
    """A single prediction from an expert."""
    expert_id: str
    expert_type: str
    items: List[Tuple[str, float]]      # (item, confidence) pairs
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedPrediction:
    """Final aggregated prediction from multiple experts."""
    items: List[Tuple[str, float]]      # Final ranked (item, confidence)
    contributing_experts: List[str]     # Expert IDs that contributed
    disagreement_score: float           # How much experts disagreed
    confidence: float                   # Overall confidence
```

---

## Expert Router

The router determines which experts to consult for a given query.

### Intent Classification

```python
class ExpertRouter:
    """Routes queries to appropriate experts based on intent."""

    # Intent → Expert type mapping
    INTENT_TO_EXPERTS = {
        'implement_feature': ['file', 'test', 'doc'],
        'fix_bug': ['file', 'error', 'test'],
        'debug_error': ['error', 'file'],
        'add_tests': ['test', 'file'],
        'refactor': ['file', 'refactor', 'test'],
        'update_docs': ['doc', 'file'],
        'code_review': ['review', 'test'],
    }

    def classify_intent(self, query: str) -> str:
        """Classify query intent using keyword/pattern matching."""
        # Uses patterns similar to existing extract_commit_type()
        ...

    def get_experts(self, query: str, context: Dict = None) -> List[str]:
        """Get list of expert types to consult."""
        intent = self.classify_intent(query)
        experts = self.INTENT_TO_EXPERTS.get(intent, ['file'])

        # Context can override (e.g., if user explicitly asks about tests)
        if context and context.get('explicit_experts'):
            experts = context['explicit_experts']

        return experts
```

### Dynamic Routing (Future)

Learn routing weights from feedback:
- Track which experts contributed to successful predictions
- Increase routing weight for experts with good track record
- Decrease weight for experts with poor performance in specific contexts

---

## Voting Aggregator

Combines predictions from multiple experts using confidence-weighted voting.

### Basic Algorithm

```python
def aggregate_predictions(
    predictions: List[ExpertPrediction],
    expert_weights: Dict[str, float] = None,
    top_n: int = 10
) -> AggregatedPrediction:
    """
    Aggregate predictions from multiple experts.

    Uses confidence-weighted voting where each expert's vote is
    scaled by their confidence and their historical accuracy.
    """
    # Initialize scores
    item_scores: Dict[str, float] = defaultdict(float)
    item_voters: Dict[str, List[str]] = defaultdict(list)

    for pred in predictions:
        expert_weight = expert_weights.get(pred.expert_id, 1.0) if expert_weights else 1.0

        for item, confidence in pred.items:
            # Weighted vote
            vote = confidence * expert_weight
            item_scores[item] += vote
            item_voters[item].append(pred.expert_id)

    # Sort by aggregated score
    sorted_items = sorted(item_scores.items(), key=lambda x: -x[1])

    # Calculate disagreement (entropy of expert votes)
    disagreement = calculate_disagreement(predictions)

    # Overall confidence based on consensus
    max_score = sorted_items[0][1] if sorted_items else 0
    overall_confidence = max_score / len(predictions) if predictions else 0

    return AggregatedPrediction(
        items=sorted_items[:top_n],
        contributing_experts=list(set(p.expert_id for p in predictions)),
        disagreement_score=disagreement,
        confidence=overall_confidence
    )
```

### Disagreement Handling

When experts strongly disagree:
1. **Flag for human review** - High disagreement suggests uncertainty
2. **Request more context** - Ask user clarifying questions
3. **Provide multiple options** - Show predictions from each expert separately
4. **Log for analysis** - Track disagreements for model improvement

---

## Per-Session Training Pipeline

Each session contributes to expert training through an "episode expert" pattern.

### Session → Episode Expert Flow

```
Session End
    │
    ▼
┌────────────────────────────┐
│  Extract Training Signals  │
├────────────────────────────┤
│  - Files referenced        │
│  - Files modified          │
│  - Tools used              │
│  - Errors encountered      │
│  - Tests mentioned         │
│  - CI results              │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Create Episode Expert     │
├────────────────────────────┤
│  - Session-specific model  │
│  - Captures local patterns │
│  - Short-term memory       │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│  Consolidate into Domain   │
│  Experts (periodically)    │
├────────────────────────────┤
│  - Merge similar episodes  │
│  - Update domain weights   │
│  - Prune stale patterns    │
└────────────────────────────┘
```

### Episode Expert Training

```python
def train_episode_expert(session_data: Dict) -> MicroExpert:
    """
    Train a micro-expert from a single session's data.

    This captures the patterns specific to this development episode,
    like a "short-term memory" that can be consolidated later.
    """
    expert = MicroExpert(
        expert_id=f"episode_{session_data['id']}",
        expert_type='episode',
        version='1.0.0',
        created_at=datetime.now().isoformat(),
        trained_on_commits=0,  # Will be updated
        trained_on_sessions=1,
        git_hash=get_current_git_hash(),
        model_data={}
    )

    # Extract file co-occurrence from session
    files_modified = set()
    files_referenced = set()
    for chat in session_data.get('chats', []):
        files_modified.update(chat.get('files_modified', []))
        files_referenced.update(chat.get('files_referenced', []))

    # Build session-specific co-occurrence
    expert.model_data['file_cooccurrence'] = build_cooccurrence(
        files_modified | files_referenced
    )

    # Extract query → file patterns
    expert.model_data['query_patterns'] = extract_query_patterns(
        session_data.get('chats', [])
    )

    # Extract error → fix patterns
    expert.model_data['error_patterns'] = extract_error_patterns(
        session_data.get('chats', [])
    )

    return expert
```

### Consolidation Strategy

Periodically merge episode experts into domain experts:

```python
def consolidate_experts(
    episode_experts: List[MicroExpert],
    target_expert: MicroExpert,
    strategy: str = 'weighted_merge'
) -> MicroExpert:
    """
    Consolidate multiple episode experts into a domain expert.

    Strategies:
    - 'weighted_merge': Average weights by recency
    - 'union': Keep all patterns (larger model)
    - 'intersection': Keep only common patterns (more confident)
    """
    if strategy == 'weighted_merge':
        # More recent episodes get higher weight
        weights = compute_recency_weights(episode_experts)
        merged_data = weighted_merge(
            [e.model_data for e in episode_experts],
            weights
        )
    elif strategy == 'union':
        merged_data = union_merge([e.model_data for e in episode_experts])
    else:
        merged_data = intersection_merge([e.model_data for e in episode_experts])

    target_expert.model_data = merged_data
    target_expert.trained_on_sessions += len(episode_experts)

    return target_expert
```

---

## Storage Structure

```
.git-ml/
├── experts/
│   ├── domain/                      # Long-term domain experts
│   │   ├── file_expert.json         # Main file prediction expert
│   │   ├── test_expert.json         # Test selection expert
│   │   ├── error_expert.json        # Error diagnosis expert
│   │   ├── doc_expert.json          # Documentation expert
│   │   └── refactor_expert.json     # Refactoring expert
│   │
│   ├── episodes/                    # Per-session episode experts
│   │   ├── episode_abc123.json      # Episode from session abc123
│   │   ├── episode_def456.json      # Episode from session def456
│   │   └── ...
│   │
│   ├── history/                     # Version history for domain experts
│   │   ├── file_expert_20251217_120000.json
│   │   └── ...
│   │
│   └── routing/                     # Router configuration and learned weights
│       ├── router_config.json       # Static routing rules
│       └── learned_weights.json     # Dynamically learned expert weights
│
├── models/                          # Existing model storage
│   └── file_prediction.json         # Current file prediction (to migrate)
│
└── tracked/                         # Git-tracked training data
    ├── commits.jsonl
    └── sessions.jsonl
```

---

## Implementation Phases

### Phase 1: Foundation (Tasks 1-3)
- [ ] Create `MicroExpert` base class with serialization
- [ ] Create `ExpertRouter` with intent classification
- [ ] Create `VotingAggregator` with basic weighted voting
- [ ] Design storage schema for experts

### Phase 2: Migration (Task 4)
- [ ] Migrate existing `FilePredictionModel` to `MicroExpert` format
- [ ] Ensure backwards compatibility
- [ ] Update CLI to use new expert format

### Phase 3: New Experts (Tasks 5-6)
- [ ] Implement `TestExpert` for test selection
- [ ] Implement `ErrorDiagnosisExpert` for error debugging
- [ ] Add training pipelines for each expert type

### Phase 4: Session Integration (Tasks 7-8)
- [ ] Create episode expert training from session transcripts
- [ ] Implement consolidation pipeline
- [ ] Add hooks for automatic training on session end

### Phase 5: Learning & Evolution (Future)
- [ ] Implement dynamic routing weight learning
- [ ] Add feedback integration for expert improvement
- [ ] Create expert performance dashboard
- [ ] Implement expert pruning for stale patterns

---

## Milestones

Building on existing milestones in `ml_collector/config.py`:

```python
MOE_MILESTONES = {
    # When to train each expert type
    "file_expert": {"commits": 500, "sessions": 100},      # Existing milestone
    "test_expert": {"commits": 300, "sessions": 75, "ci_results": 100},
    "error_expert": {"sessions": 150, "error_chats": 50},
    "doc_expert": {"commits": 400, "doc_commits": 50},
    "refactor_expert": {"commits": 500, "large_commits": 100},  # >5 files

    # When routing becomes viable
    "moe_routing": {"experts": 3, "total_predictions": 500},
}
```

---

## Benefits

1. **Specialization**: Each expert optimizes for its domain
2. **Parallelism**: Experts can be trained independently
3. **Incremental Learning**: New sessions add to experts without full retraining
4. **Interpretability**: Can explain which expert contributed what
5. **Resilience**: If one expert fails, others provide fallback
6. **Natural Growth**: New expert types can be added modularly

---

## Thousand Brains Alignment

| Thousand Brains Concept | MoE Implementation |
|------------------------|-------------------|
| Cortical columns | Individual micro-experts |
| Reference frames | Query context / intent |
| Voting mechanism | Confidence-weighted aggregation |
| Hierarchy | Episode → Domain expert consolidation |
| Lateral connections | Expert co-occurrence patterns |
| Predictions | Each expert's ranked output |
| Learning | Session-based episode training |

---

## Next Steps

1. **Review this architecture** with stakeholders
2. **Start with Phase 1** - Foundation classes
3. **Migrate file prediction** to validate the format
4. **Add TestExpert** as first new expert type
5. **Integrate with session end hook** for automatic training

---

*Document created: 2025-12-17*
*Status: Draft - Ready for implementation*
