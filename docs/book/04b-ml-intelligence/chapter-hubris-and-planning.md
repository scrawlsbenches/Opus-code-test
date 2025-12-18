# Hubris: A Mixture of Experts for AI-Assisted Development

**Tags:** `moe`, `hubris`, `planning`, `meta-learning`, `orchestration`
**Generated:** 2025-12-18

---

## Introduction

As the Cortical Text Processor project evolved, a meta-problem emerged: **how do we make AI assistants better at predicting what needs to be done?** When Claude Code suggests files to modify, which files are actually correct? When it recommends tests to run, which ones matter? When debugging an error, which files should we examine first?

Enter **Hubris** - a Mixture of Experts (MoE) system where specialized micro-models learn from development history and compete to provide the best predictions. The name is intentionally ironic: these experts must **earn confidence through demonstrated accuracy**, not assume it through design.

### What Problem Does Hubris Solve?

Traditional AI assistants face several challenges:
- **Monolithic models** try to be good at everything, master of none
- **No feedback loop** - predictions aren't scored against outcomes
- **No specialization** - file prediction uses the same model as error diagnosis
- **No accountability** - can't trace which component made which prediction

Hubris addresses these through:
1. **Specialized experts** - FileExpert, TestExpert, ErrorDiagnosisExpert, EpisodeExpert
2. **Credit-based economy** - experts earn influence through accurate predictions
3. **Confidence staking** - experts can bet credits on high-confidence predictions
4. **Intent routing** - queries route to relevant experts based on classified intent
5. **Voting aggregation** - multiple experts contribute, weighted by track record

### The Name: Hubris

In Greek tragedy, *hubris* (ὕβρις) means excessive confidence leading to downfall. Our Hubris system inverts this:

```
Traditional AI:  High confidence by default → no consequences for errors
Hubris MoE:      Earn confidence through accuracy → lose credits for mistakes
```

Experts start with **100 credits** and equal influence. Successful predictions increase credit balance and routing weight. Failed predictions decrease credits. Over time, the **most accurate experts gain the most influence** - a meritocracy where hubris must be justified.

---

## The Journey: From Ad-Hoc to Structured Planning

Before Hubris existed, the project needed **a system to track work across multiple parallel agents**. This journey mirrors the evolution of many software projects: from informal to formal, from chaotic to structured.

### Early Days: Single Tasks, Manual Tracking

**Problem:** When you're working alone with one Claude session, you can keep tasks in your head or in a simple `TODO.md`:

```markdown
## Tasks
- [ ] Add PageRank algorithm
- [ ] Implement TF-IDF scoring
- [ ] Add Louvain clustering
```

**Limitations:**
- No task IDs (how do you reference a task in commits?)
- No prioritization or dependencies
- No tracking across sessions
- Manual markdown editing
- **Merge conflicts** when multiple agents work simultaneously

### Evolution: Task System with Merge-Friendly IDs

As the project grew, we needed:
1. **Unique task identifiers** that don't collide when multiple agents create tasks
2. **Structured task metadata** (priority, status, dependencies, effort estimates)
3. **Git-friendly storage** that doesn't create merge conflicts
4. **Automatic tracking** via git hooks and session integration

**Solution:** Timestamp-based, collision-resistant task IDs:

```
T-20251213-143052-a1b2
│ │        │      │
│ │        │      └── 4-char session suffix (unique per agent)
│ │        └── Time created (HHMMSS)
│ └── Date created (YYYYMMDD)
└── Task prefix
```

Each agent session writes to its **own task file**:

```
tasks/
├── 2025-12-13_14-30-52_a1b2.json    # Agent A's session
├── 2025-12-13_14-31-05_c3d4.json    # Agent B's session
└── 2025-12-13_15-00-00_e5f6.json    # Agent C's session
```

**No merge conflicts.** Tasks are periodically consolidated, like `git gc`.

This pattern is documented in `docs/merge-friendly-tasks.md` and implemented in `scripts/task_utils.py`.

### Current: Epics → Sprints → Tasks Hierarchy

As parallel agent orchestration became common, we needed **higher-level organization**:

```
Epic: MoE Foundation
├── Sprint 1: Foundation Classes
│   ├── Task: Design MicroExpert base class
│   ├── Task: Implement ExpertRouter
│   └── Task: Create VotingAggregator
├── Sprint 2: New Experts
│   ├── Task: Migrate FilePredictionModel
│   ├── Task: Implement TestExpert
│   └── Task: Implement ErrorDiagnosisExpert
└── Sprint 3: Integration
    ├── Task: CLI interface
    ├── Task: Session-end training
    └── Task: Credit system integration
```

**Benefits:**
- **Epics** capture large initiatives (weeks to months)
- **Sprints** group related work (days to weeks)
- **Tasks** are atomic units (hours to days)
- **Dependencies** are explicit (blocking relationships)
- **Parallel execution** - different agents take non-overlapping sprints

This hierarchy enabled the **Batch Orchestration Pattern** documented in `docs/BATCH2-ORCHESTRATION-PLAN.md`, where multiple agents work on independent file claims simultaneously.

---

## The Meta-Learning Insight

The leap from "task tracking system" to "Hubris MoE" came from a realization:

> **We're building a system to help AI assistants get better at development. Why not use that system to make the AI assistants *learn* to get better?**

This is **meta-learning** - using the development process itself as training data for models that predict development patterns.

### The Feedback Loop

Traditional workflow:
```
1. AI suggests files to modify
2. Human accepts or rejects
3. Files are modified
4. (No feedback to AI about correctness)
```

Hubris workflow:
```
1. FileExpert predicts files to modify (with confidence scores)
2. Human modifies some files (actual outcome)
3. Commit recorded with file changes
4. ValueSignal generated: correct files earn +credits, wrong files lose -credits
5. Expert's credit balance updated
6. Future routing weights adjusted
7. Next prediction is more accurate
```

### Systems That Improve Themselves

Hubris embodies several self-improvement patterns:

**1. Predictions → Outcomes → Credit Updates**
- Expert makes prediction with confidence score
- Actual outcome observed (commit, test result, etc.)
- Credit awarded/deducted based on accuracy
- Routing weights recalculated with softmax

**2. Staking as Calibration**
- Expert can stake extra credits on high-confidence predictions
- Correct prediction: earn `stake × multiplier`
- Wrong prediction: lose entire stake
- Forces experts to **calibrate confidence realistically**

**3. Episode Learning**
- Each development session becomes training data
- EpisodeExpert extracts patterns: "When fixing auth bugs, these files changed together"
- Episode experts consolidate into domain experts
- Recent patterns weighted higher than old patterns

**4. Continuous Retraining**
- Session ends → transcript captured → experts retrained
- Commit pushed → ML data collector records → file patterns updated
- CI passes/fails → test-to-file associations strengthened
- Error fixed → error-to-fix patterns added to ErrorDiagnosisExpert

---

## Architecture Overview

Hubris implements a **Thousand Brains Theory**-inspired architecture, where multiple specialized "cortical columns" (experts) vote to reach consensus.

### Expert Types

| Expert | Purpose | Training Data | Prediction Output |
|--------|---------|---------------|-------------------|
| **FileExpert** | Which files to modify for a task? | Commit history with file changes | `[(file_path, confidence), ...]` |
| **TestExpert** | Which tests to run for changes? | Commit-to-test co-occurrence | `[(test_file, confidence), ...]` |
| **ErrorDiagnosisExpert** | What's causing this error? | Error records with resolutions | `[(file, line_range, confidence), ...]` |
| **EpisodeExpert** | What action comes next? | Session transcript tool sequences | `[(action, confidence), ...]` |

Each expert is a **specialized micro-model** (~1000-10000 lines of training data, lightweight, fast inference).

### The Credit Economy

Credits create a **value-based economy** where influence is earned, not assumed.

#### Initial State
```python
FileExpert:     100 credits
TestExpert:     100 credits
ErrorExpert:    100 credits
EpisodeExpert:  100 credits
```

#### After 50 Predictions
```python
FileExpert:     156.3 credits  ← Most accurate
TestExpert:     134.2 credits  ← Good accuracy
ErrorExpert:    109.5 credits  ← Moderate accuracy
EpisodeExpert:   87.3 credits  ← Needs improvement
```

#### Credit-Weighted Routing

When a query comes in, routing weights are computed using **softmax with temperature**:

```python
# Credit balances → routing weights
weights = softmax([156.3, 134.2, 109.5, 87.3], temperature=1.0)
# Result: FileExpert gets 42% influence, TestExpert 31%, etc.

# High-credit experts get confidence boost
if credit_balance > 150:
    confidence_multiplier = 1.05  # 5% boost
```

This creates **natural selection** - experts with poor track records fade, experts with good track records dominate.

### Staking: Confidence as Currency

Experts can **stake credits on predictions** to earn multiplied rewards:

```python
# FileExpert is 95% confident about this prediction
prediction = file_expert.predict({
    'query': 'Add authentication feature',
    'confidence': 0.95
})

# AutoStaker decides to stake based on confidence
if confidence > 0.9:
    stake = pool.place_stake(
        expert_id='file_expert',
        amount=20.0,  # 20 credits at risk
        multiplier=2.0  # 2x risk/reward
    )

# Outcome: Prediction was correct
pool.resolve_stake(stake.id, success=True)
# → Expert gains +20.0 (40 payout - 20 original = 20 profit)

# Outcome: Prediction was wrong
pool.resolve_stake(stake.id, success=False)
# → Expert loses -20.0 (stake forfeited)
```

**Staking strategies:**
- `CONSERVATIVE`: 1.0x multiplier (no extra risk)
- `MODERATE`: 1.5x multiplier
- `AGGRESSIVE`: 2.0x multiplier
- `YOLO`: 3.0x multiplier (high risk, high reward)

Staking forces experts to **avoid overconfidence** - if you stake high on wrong predictions, you go bankrupt.

### Component Interaction

```
User Query: "Fix authentication bug"
    ↓
ExpertRouter.route()
  → Intent: 'fix_bug'
  → Experts: [file, error, test]
    ↓
ExpertConsolidator.get_ensemble_prediction()
    ↓
┌─────────────┬─────────────┬─────────────┐
│ FileExpert  │ ErrorExpert │ TestExpert  │
│ (156.3cr)   │ (109.5cr)   │ (134.2cr)   │
└──────┬──────┴──────┬──────┴──────┬──────┘
       │             │             │
   Prediction    Prediction    Prediction
       │             │             │
       └─────────────┼─────────────┘
                     ↓
           CreditRouter.aggregate()
         (weight by credit balance)
                     ↓
          AggregatedPrediction
          [auth.py: 0.823, tests/test_auth.py: 0.756, ...]
                     ↓
              User receives prediction
                     ↓
          Files modified (actual outcome)
                     ↓
             ValueSignal generated
                     ↓
          Experts' credits updated
```

---

## Multi-Agent Orchestration

As Hubris evolved, so did our ability to **coordinate multiple Claude agents working simultaneously**.

### The Problem: Parallel Agent Conflicts

When running 3 agents in parallel:
- Agent A modifies `cortical/query/search.py`
- Agent B modifies `cortical/query/search.py`
- **Merge conflict** guaranteed

**Solution:** File claims and non-overlapping domains.

### Parallel Sub-Agents Pattern

Batch orchestration plan defines **exclusive file claims**:

```python
# docs/BATCH2-ORCHESTRATION-PLAN.md
Agent ε (MoE Foundation):
  Files: scripts/hubris/**/*.py         (exclusive write)
  Tasks: T-efba-001, T-efba-002, T-efba-003

Agent ζ (Epic 2 Completion):
  Files: scripts/session_memory_*.py    (exclusive write)
  Tasks: Sprint-2.3, Sprint-2.4

File Claim Matrix:
                     ε(MoE)    ζ(Epic2)
scripts/hubris/**    ██        ░░
scripts/session_*    ░░        ██

██ = Exclusive (can modify)
░░ = Read-only
```

**Benefits:**
- No merge conflicts (disjoint file sets)
- Parallel execution (agents work simultaneously)
- Clear ownership (each agent knows its domain)
- Faster completion (parallelism reduces wall-clock time)

### The Director Pattern

For complex multi-step tasks, the **Director Agent** orchestrates sub-agents:

```
Director
├── Phase 1: Foundation (Agent α)
│   └── Create base classes
├── Phase 2: Experts (Agent β)
│   └── Implement specialized experts
├── Phase 3: Integration (Agent γ)
│   └── Wire up credit system
└── Phase 4: Testing (Agent δ)
    └── Comprehensive test coverage
```

**Director responsibilities:**
1. **Task decomposition** - break epic into phases
2. **Dependency management** - ensure Phase N+1 waits for Phase N
3. **Batch verification** - check all agents completed successfully
4. **Error recovery** - retry failed phases, rollback if needed
5. **Progress reporting** - aggregate status from all agents

Tools for orchestration:
- `scripts/orchestration_utils.py` - Track batches and metrics
- `scripts/verify_batch.py` - Automated verification
- `.claude/commands/director.md` - Director slash command

### Coordination Rules

**1. Merge Order**
- Foundation layers merge first
- Dependent layers merge after dependencies
- Example: MoE base classes → Experts → Integration → CLI

**2. Completion Report Format**
```json
{
  "agent": "epsilon",
  "status": "complete",
  "files_created": ["scripts/hubris/micro_expert.py", ...],
  "files_modified": ["scripts/ml_file_prediction.py"],
  "tests_added": 42,
  "tests_passed": true,
  "blockers": []
}
```

**3. Conflict Resolution**
- If file claims overlap, **higher-priority agent wins**
- If both equal priority, **first to commit wins**
- Blocked agent rebases and resolves conflicts

---

## Lessons Learned

### 1. Wire the Feedback Loop Early

**Lesson:** Prediction without feedback is guessing. Track outcomes and update models.

We initially had file prediction but no mechanism to record "was the prediction correct?" Now:
- Commits link to predictions
- Credit system provides quantitative feedback
- Routing weights adjust automatically

**Implementation:**
- `ValueSignal` class captures outcome quality
- `ValueAttributor` translates outcomes to credit deltas
- `CreditRouter` updates routing weights

### 2. Log Everything for Future Training

**Lesson:** You can't train on data you didn't collect.

Early sessions weren't captured - lost training opportunities. Now:
- Session transcripts saved automatically
- Commits tracked with ML collector
- CI results recorded with commits
- All data flows into `.git-ml/` for future training

**Implementation:**
- `.claude/settings.local.json` - SessionStart/Stop hooks
- `scripts/ml-session-capture-hook.sh` - Transcript capture
- `scripts/ml_data_collector.py` - Automated data collection
- `.github/workflows/ci.yml` - CI result capture

### 3. Start Simple, Add Complexity as Needed

**Lesson:** Don't build MoE with 10 experts on day 1. Start with 1 expert, validate the pattern, then add more.

**Evolution:**
1. **FilePredictionModel** (simple TF-IDF + co-occurrence)
2. **Migrate to MicroExpert format** (standardize interface)
3. **Add credit system** (track performance)
4. **Add TestExpert** (second expert, validates voting works)
5. **Add staking** (confidence calibration)
6. **Add episode learning** (session-based patterns)

Each step validated before adding the next.

### 4. Use the System to Improve Itself

**Lesson:** Dog-food your own tools. Hubris uses Cortical Text Processor for semantic search. Task system uses its own collision-resistant IDs. ML collector trains models that improve the ML collector.

**Examples:**
- Hubris CLI uses FileExpert to predict what to modify when debugging Hubris
- Task consolidation uses task system's own IDs
- Session memory generator uses ML data it collects
- Book generator indexes its own chapters for search

**Benefits:**
- Real usage reveals bugs and UX issues
- Continuous improvement loop
- Validates that abstractions are general-purpose

### 5. Parallel Agents Need Structure

**Lesson:** You can't just say "3 agents, go!" - they'll step on each other. Need explicit coordination.

**Requirements for successful parallelism:**
- **Disjoint file claims** (no overlapping writes)
- **Clear task boundaries** (agent knows what's in scope)
- **Merge order** (dependencies merge before dependents)
- **Batch verification** (automated checks before merge)
- **Rollback plan** (what if one agent fails?)

Without these, merge conflicts and wasted work are inevitable.

---

## Future Directions

### SprintPlanningExpert (After More Data)

Once we have 50+ completed sprints, train an expert to predict:
- **Sprint duration** from task descriptions
- **Task dependencies** from semantic similarity
- **Effort estimates** from historical patterns
- **Risk factors** (e.g., "authentication tasks often take 2x longer")

**Training data:** Completed sprint metadata with actual outcomes.

### User Preference Learning

Track which predictions users accept/reject to learn **personal preferences**:
- User prefers test-first development → boost TestExpert
- User rarely modifies docs → reduce DocumentationExpert weight
- User always adds type hints → predict typing files

**Implementation:** Per-user credit ledgers, personalized routing weights.

### Confidence Calibration Tracking

Monitor calibration error over time:
```
Calibration Curve:
  Expert predicts 0.9 confidence → actual accuracy 0.85
  Expert predicts 0.7 confidence → actual accuracy 0.70
  Expert predicts 0.5 confidence → actual accuracy 0.45

Calibration Error: 3.2% (excellent)
```

Use calibration curves to adjust raw confidences before aggregation.

### Cross-Repository Learning

Train experts across multiple repositories:
- "When fixing auth bugs in repo A, these patterns apply"
- "When repo B adds a feature, these files usually change together"
- Aggregate patterns across projects for general developer intelligence

**Challenge:** Privacy, data isolation, different tech stacks.

### Explainable Predictions

Show users **why** an expert predicted something:
```
FileExpert predicts cortical/auth.py (confidence: 0.92)
  Reason 1: Commit type 'fix' → auth.py (12 historical examples)
  Reason 2: Keyword 'authentication' → auth.py (TF-IDF: 0.85)
  Reason 3: Co-occurs with login.py in 8 recent commits
```

**Benefits:** Trust, debugging, learning.

---

## Summary

**Hubris** represents a shift from static AI assistance to **adaptive, self-improving AI**:

- **Multiple specialized experts** instead of one generalist
- **Credit-based meritocracy** instead of equal weighting
- **Continuous learning** from development outcomes
- **Confidence staking** to enforce calibration
- **Intent-based routing** to activate relevant experts

**Planning evolution** mirrors software maturity:
- **Ad-hoc tasks** → **Structured tasks** → **Epics/Sprints** → **Parallel orchestration**
- Collision-resistant IDs enable parallel agents
- File claims prevent merge conflicts
- Director pattern orchestrates complex multi-phase work

**Meta-learning insight:** Use the development process to train models that improve the development process. Close the loop from **predictions → outcomes → credit updates → better predictions**.

The name "Hubris" reminds us: **confidence must be earned through accuracy**, not assumed through design.

---

## References

- **Hubris Implementation:** `scripts/hubris/` - MoE system with credit economy
- **Task System:** `docs/merge-friendly-tasks.md` - Collision-resistant task IDs
- **Orchestration:** `docs/BATCH2-ORCHESTRATION-PLAN.md` - Parallel agent coordination
- **Architecture:** `docs/moe-thousand-brains-architecture.md` - Thousand Brains-inspired design
- **Thousand Brains Theory:** Jeff Hawkins, "A Thousand Brains" (2021)

---

*This chapter is part of the Cortical Chronicles, documenting the evolution of AI-assisted development through meta-learning and self-improvement.*
