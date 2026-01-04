# Knowledge Transfer: Learning System Redesign

**Date:** 2026-01-04
**Session:** claude/senior-engineer-consultation-6zjbT
**Author:** Claude (Opus 4.5)
**Reviewer:** Senior Principal Computer Scientist

---

## Executive Summary

This session diagnosed "Integration Theater" in the learning system—components initialized but not influencing behavior—and implemented semantic intent matching and file risk tracking. We discovered PRISM-SLM is trained on this codebase (15,814 terms, 37,318 documents) and can provide codebase-specific guidance that generic LLMs cannot.

**Key Deliverables:**
- Semantic intent matching (find similar tasks by meaning)
- File outcome tracking (identify risky files)
- Real GoT data demo proving the system works
- Architecture plan for PRISM integration

---

## What We Discovered

### 1. Integration Theater

The learning system was "wired up but not wired IN":

| Component | Initialized | Records Data | Uses Data |
|-----------|-------------|--------------|-----------|
| GoT Learning | ✅ | ✅ `capture_task_completion()` | ❌ **Never queried** |
| PRISM-GoT | ✅ | ✅ `record_activation()` | ❌ **Never predicts** |
| WovenMind | ✅ | ✅ `force_mode()` | ❌ **Never processes** |

**Location of the problem:** `llm_orchestration/agents.py:1614-1680`
- `got_lessons` was retrieved but discarded
- `self._synaptic_graph.record_activation()` was write-only

### 2. Learning Pipeline Mechanics

The Experience → Pattern → Lesson pipeline works, but has specific requirements:

```
Experience → Pattern → Lesson → Guidance
     ↓           ↓          ↓
  7+ needed   confidence  >= 0.4 threshold
              formula:
              log(occurrences + 1) / 5
```

**Key insight:** Need 7+ similar experiences before a lesson is generated.

**Location:** `llm_orchestration/learning.py:868` (LessonDistiller.distill_from_pattern)

### 3. PRISM-SLM Is Trained on This Codebase

```
PRISM-SLM Model Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Vocabulary:      15,814 unique terms
N-gram contexts: 144,199 patterns
Total tokens:    649,107
Documents:       37,318
N-gram size:     3 (trigram)

Codebase terms it knows:
  cortical, got, prism, synaptic, transaction, wal,
  hebbian, processor, minicolumn, pagerank, tokenizer,
  indexer, tfidf, bigram, thought

Sample knowledge:
  "is cortical" → layers, got, moe
  "got indexer" → py  ← Knows the file extension!
  "hebbian learning" → is, a, mean
```

**Location:** `benchmarks/codebase_slm/models/prism_augmented.json` (13MB)

**Implication:** PRISM-SLM can suggest codebase-specific paths that a generic LLM cannot.

---

## What We Built

### 1. Semantic Intent Matching

**Files:**
- `llm_orchestration/learning.py:1532-1739` (implementation)
- `tests/behavioral/test_semantic_intent_matching.py` (5 tests)

**New Methods on LearningCycle:**

```python
# Extract keywords from intent
keywords = cycle.extract_keywords("Implement JWT authentication")
# Returns: {'implement', 'jwt', 'authentication'}

# Calculate similarity between intents
similarity = cycle.intent_similarity(
    "Implement JWT authentication",
    "Add JWT token verification"
)
# Returns: 0.167 (share 'jwt')

# Find experiences by semantic similarity
experiences = cycle.find_by_intent(
    intent="Add JWT token authentication",
    min_similarity=0.15,
    limit=5
)

# Combined context + intent search
results = cycle.find_by_context_and_intent(
    context=Context(goal_type="feature", domain="api"),
    intent="Add JWT authentication",
    context_weight=0.3,
    intent_weight=0.7,
)
```

**Integration:** Wired into `cortical/got/learning_integration.py:539-588`
- `get_guidance_for_task()` now uses semantic matching
- Merges semantic results with context-based results
- Adds recommendations from similar successful tasks
- Adds warnings from similar failed tasks

### 2. File Outcome Tracking

**Files:**
- `llm_orchestration/learning.py:1741-1966` (implementation)
- `tests/behavioral/test_file_outcome_tracking.py` (5 tests)

**New Methods on LearningCycle:**

```python
# Get files touched in an experience
files = cycle.get_files_from_experience(exp_id)
# Returns: ['src/auth/jwt.py', 'tests/test_jwt.py']

# Get history for a specific file
history = cycle.get_file_history("src/auth/jwt.py")
# Returns: {
#   'total_experiences': 5,
#   'success_count': 2,
#   'failure_count': 3,
#   'success_rate': 0.4,
#   'error_patterns': {'ImportError': 2, 'TypeError': 1},
#   'recent_experiences': [...]
# }

# Find risky files
risky = cycle.get_risky_files(min_experiences=3, max_success_rate=0.5)
# Returns files with high failure rates

# Get guidance with file risk assessment
guidance = cycle.get_guidance_for_files(
    intent="Add OAuth to authentication",
    files_to_modify=["src/auth/jwt.py", "src/auth/oauth.py"],
)
# Returns: lessons, recommendations, warnings, file_risks
```

**Integration:** Wired into `cortical/got/learning_integration.py:590-650`
- `get_guidance_for_task()` assesses file risk when `files_to_modify` provided
- Adds warnings for risky files
- Includes error pattern warnings (e.g., "Common error: ImportError")

### 3. Real Data Demos

**Files:**
- `demo_learning_system.py` - Hardcoded example (for understanding)
- `demo_real_got_data.py` - Uses actual `.got/` data

**Demo output from real data:**
```
📊 GoT Statistics:
   Tasks: 23
   Experiences: 27
   Edges: 3

📈 Experience Outcomes:
   Successes: 26
   Failures: 1
   Success Rate: 96.3%

🔍 Query: "Fix index save failure"
   [✓] CRITICAL: Fix index-store inconsistency on index save f
       Similarity: 57.1%
```

---

## Architecture: Current State

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CURRENT ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LAYER 1: Learning System (✅ IMPLEMENTED)                              │
│  ├── Semantic intent matching via keywords                              │
│  ├── File outcome tracking with error patterns                          │
│  ├── Experience → Pattern → Lesson pipeline                             │
│  └── Wired into get_guidance_for_task()                                 │
│                                                                          │
│  LAYER 2: PRISM-GoT (⚠️ PARTIAL - records only)                         │
│  ├── ✅ SynapticMemoryGraph initialized in Worker                       │
│  ├── ✅ record_activation() called for QAPV phases                      │
│  ├── ❌ predict_next_thoughts() NEVER called                            │
│  ├── ❌ apply_hebbian_learning() NEVER called                           │
│  └── ❌ apply_reward() NEVER called                                     │
│                                                                          │
│  LAYER 3: PRISM-SLM (⚠️ EXISTS but not integrated)                      │
│  ├── ✅ 13MB trained model exists                                       │
│  ├── ✅ Knows 15,814 codebase terms                                     │
│  └── ❌ Not queried during task execution                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps (Agreed Plan)

### Step 2: Wire PRISM-GoT Confusion Detection

**Goal:** Use PRISM-GoT's prediction capability to detect cognitive confusion.

**What "confusion" means:**
- Repeated thoughts (looping)
- Low prediction confidence (unusual pattern)
- Unexpected thought transitions

**Where to wire it:** `llm_orchestration/agents.py`
- After each QAPV phase, query `predict_next_thoughts()`
- If confidence < threshold, trigger confusion signal
- If actual next thought differs from prediction, update learning

**Confirmation point:** Demo showing confusion detection during task execution.

### Step 3: Wire PRISM-SLM Codebase Knowledge

**Goal:** Use PRISM-SLM's codebase knowledge to suggest relevant files/patterns.

**Interface concept:**
```python
# Given a task context, suggest relevant code locations
suggestions = prism_slm.suggest_for_context("working on indexer")
# Returns: ["got/indexer.py", "cortical/processor"]
```

**Why this matters:** Generic LLM doesn't know `got indexer → py`. PRISM-SLM does.

**Confirmation point:** Demo showing PRISM-SLM suggesting codebase-specific paths.

### Step 1+: Enrich with Git History

**Goal:** Use actual git history to improve file risk assessment.

**Data sources:**
- `git log --format='%H %s' -- <file>` - Commit history per file
- `git blame <file>` - Who changed what
- Commit messages for intent extraction

**Confirmation point:** Demo showing file risk from actual git data.

---

## Key Files Reference

### Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `llm_orchestration/learning.py` | 1532-1966 | Semantic matching + file tracking |
| `cortical/got/learning_integration.py` | 471-650 | GoT ↔ Learning bridge |
| `llm_orchestration/agents.py` | 1614-1680 | Worker guidance integration |
| `cortical/reasoning/prism_got.py` | 741-783 | PRISM prediction methods |
| `benchmarks/codebase_slm/models/prism_augmented.json` | - | Trained PRISM-SLM model |

### Test Files

| File | Tests | Purpose |
|------|-------|---------|
| `tests/behavioral/test_learning_pipeline_hypothesis.py` | 4 | Prove pipeline works |
| `tests/behavioral/test_semantic_intent_matching.py` | 5 | Prove semantic matching |
| `tests/behavioral/test_file_outcome_tracking.py` | 5 | Prove file tracking |
| `tests/behavioral/test_cognitive_team_e2e.py` | 17 | End-to-end cognitive tests |

### Demo Files

| File | Purpose |
|------|---------|
| `demo_learning_system.py` | Hardcoded example for understanding |
| `demo_real_got_data.py` | Real GoT data demonstration |

---

## How to Continue This Work

### Running Tests

```bash
# All learning-related tests (14 tests)
python3 -m pytest tests/behavioral/test_learning_pipeline_hypothesis.py \
                  tests/behavioral/test_semantic_intent_matching.py \
                  tests/behavioral/test_file_outcome_tracking.py -v

# Full cognitive suite (31 tests)
python3 -m pytest tests/behavioral/test_cognitive_team_e2e.py -v --tb=short
```

### Running Demos

```bash
# Demo with real GoT data
python3 demo_real_got_data.py

# Analyze PRISM-SLM model
python3 -c "
import json
with open('benchmarks/codebase_slm/models/prism_augmented.json') as f:
    m = json.load(f)
print(f'Vocab: {len(m[\"vocab\"])}')
print(f'Contexts: {len(m[\"counts\"])}')
"
```

### Key Commands for Development

```bash
# Check GoT data
ls -la .got/
cat .got/T-*.json | python3 -m json.tool | head -50

# Check learning experiences
ls .got/learning/experiences/
cat .got/learning/experiences/exp_*.json | python3 -m json.tool

# Validate GoT integrity
python scripts/got_utils.py validate
```

---

## Commits from This Session

```
2e3498fd - feat(learning): Add semantic intent matching for experience retrieval
76fd3b20 - feat(learning): Wire semantic matching into GoT learning bridge
efb4a9ff - feat(learning): Add file outcome tracking for risk assessment
03e77c28 - feat(learning): Wire file risk tracking into GoT learning bridge
82072395 - demo: Add learning system demo with real data examples
86152e4e - demo: Add real GoT data demo for learning system
```

---

## Open Questions

1. **Confusion thresholds:** What prediction confidence triggers recovery? (Proposed: < 0.3)
2. **PRISM-SLM query interface:** How do we query the n-gram model for suggestions?
3. **Git enrichment scope:** How far back in git history? Only files we touch?
4. **Feedback loop:** How do we know if guidance was helpful?

---

## Principles Applied

1. **TDD:** Wrote behavioral tests before implementation
2. **No sub-agents:** Worked through issues together directly
3. **Commit immediately:** Pushed after every save
4. **Real data:** Proved system works on actual GoT data
5. **Explain reasoning:** Documented why, not just what

---

*This knowledge transfer captures the state as of session end. Continue with Step 2 (PRISM-GoT confusion detection) with confirmation demos between each step.*
