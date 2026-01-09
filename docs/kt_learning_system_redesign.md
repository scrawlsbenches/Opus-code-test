# Knowledge Transfer: Learning System Redesign

**Date:** 2026-01-04
**Session:** claude/senior-engineer-consultation-6zjbT
**Author:** Claude (Opus 4.5)
**Reviewer:** Senior Principal Computer Scientist
**Status:** HANDOFF READY

---

## Executive Summary

This session diagnosed "Integration Theater" in the learning system—components initialized but not influencing behavior—and implemented semantic intent matching and file risk tracking. We discovered PRISM-SLM is trained on this codebase (15,814 terms, 37,318 documents) and can provide codebase-specific guidance that generic LLMs cannot.

**CRITICAL FINDING (End of Session):** We discovered a **duplicate learning system architecture** that must be consolidated BEFORE adding PRISM integration. Two separate learning systems are running in parallel, one orphaned.

**Key Deliverables:**
- Semantic intent matching (find similar tasks by meaning)
- File outcome tracking (identify risky files)
- Real GoT data demo proving the system works
- Architecture plan for PRISM integration
- **NEW:** Complete architectural analysis of duplicate learning systems

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

### 4. CRITICAL: Duplicate Learning Systems (Session End Discovery)

We traced the complete execution flow in `agents.py` and discovered TWO parallel learning systems:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DUPLICATE LEARNING SYSTEMS ANALYSIS                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SYSTEM 1: LOCAL USER CYCLE                                                 │
│  ────────────────────────────                                               │
│  Path:      ~/.llm_orchestration/learning/                                  │
│  Scope:     User-level (persists across all projects)                       │
│  Variable:  self._learning_cycle (Worker.__init__ line 887)                 │
│                                                                              │
│  Usage:                                                                      │
│  • Line 1611: lessons = self._get_lessons_for_task(task_context)            │
│  • Line 1717: learning_cycle = LearningCycle(storage_dir)                   │
│  • Line 1726: experience = learning_cycle.start_experience(...)             │
│  • Line 1958: learning_cycle.complete_experience(experience, outcome)       │
│                                                                              │
│  Feedback:   ❌ NONE - Lessons retrieved but never validated                │
│  Status:     ORPHANED - Writes data that is never read by GoT              │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SYSTEM 2: GOT PROJECT CYCLE (OUR WORK)                                     │
│  ───────────────────────────────────────                                    │
│  Path:      .got/learning/                                                  │
│  Scope:     Project-level (persists within this codebase)                   │
│  Variable:  self._got_learning_bridge (Worker.__init__ line 899)            │
│                                                                              │
│  Usage:                                                                      │
│  • Line 1627: got_guidance = self._got_learning_bridge.get_guidance_for_task│
│  • Line 1907: self._got_learning_bridge.cycle.validate_lesson(was_helpful)  │
│  • Line 1920: self._got_learning_bridge.capture_task_completion(...)        │
│  • Line 1981: self._got_learning_bridge.capture_task_failure(...)           │
│                                                                              │
│  Feedback:   ✅ COMPLETE - Lessons validated on success/failure             │
│  Status:     ACTIVE - Full semantic matching + file tracking                │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DATA FLOW (CURRENT):                                                       │
│                                                                              │
│       execute_task()                                                        │
│            │                                                                 │
│            ├──► Line 1611: OLD lessons retrieved (no feedback) ──► metrics │
│            │                                                                 │
│            ├──► Line 1627: GOT guidance retrieved ──┬──► QAPV context      │
│            │                                        └──► has feedback loop  │
│            │                                                                 │
│            ├──► Line 1726: LOCAL experience started                         │
│            │                                                                 │
│            ├──► [task executes]                                              │
│            │                                                                 │
│            ├──► Line 1958: LOCAL experience completed ──► ~/.llm_orch/     │
│            │                                                                 │
│            └──► Line 1920: GOT experience captured ──► .got/learning/       │
│                                                                              │
│  RESULT: Same task recorded in TWO places, only ONE has feedback           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why this matters:** If we wire PRISM-GoT confusion detection, which learning system's data should PRISM learn from? Duplicate experiences create confused training data.

**Confirmed via grep:**
```bash
# Three instantiation points for LearningCycle in agents.py:
llm_orchestration/agents.py:887:  self._learning_cycle = LearningCycle(storage_dir)
llm_orchestration/agents.py:1717: learning_cycle = LearningCycle(storage_dir)
llm_orchestration/agents.py:2511: learning_cycle = LearningCycle(storage_dir)

# GoTLearningBridge uses .got/learning/:
cortical/got/learning_integration.py:206: self.learning_dir = self.got_dir / "learning"
cortical/got/learning_integration.py:220: self.cycle = LearningCycle(self.learning_dir)
```

**DESIGN.md confirms:** "Cross-project learning" is listed as an OPEN QUESTION (line 478), not an implemented feature. The two storage locations are accidental complexity.

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

## Next Steps (Revised Plan)

### ⚠️ STEP 0: Consolidate Learning Systems (BLOCKING)

**MUST DO FIRST.** Before wiring PRISM, consolidate to a single learning system.

**Proposed Resolution:**

| Current State | Proposed Change |
|---------------|-----------------|
| `~/.llm_orchestration/learning/` (user home) | **DELETE** - Not project-aware, no feedback loop |
| `.got/learning/` (project root) | **KEEP** - Our semantic + file tracking lives here |
| `self._learning_cycle` (line 887) | Remove or rewire to GoT |
| `learning_cycle` (line 1717) | Remove - duplicates GoT capture |
| Lines 1611 (old guidance) | Remove - GoT has better guidance |
| Lines 1627 (new guidance) | Keep - This is our semantic matching |

**Approach:**
1. Write behavioral test proving "single source of truth for experiences"
2. Remove local learning cycle instantiations
3. Rewire metrics at line 1840 to use `got_guidance` instead of `lessons`
4. Verify tests pass

**Confirmation point:** Tests pass, only one learning system remains.

### Step 1: Wire PRISM-GoT Confusion Detection

**Goal:** Use PRISM-GoT's prediction capability to detect cognitive confusion.

**What "confusion" means:**
- Repeated thoughts (looping)
- Low prediction confidence (unusual pattern)
- Unexpected thought transitions

**Where to wire it:** `llm_orchestration/agents.py`
- After each QAPV phase, query `predict_next_thoughts()`
- If confidence < threshold, trigger confusion signal
- If actual next thought differs from prediction, update learning

**Current state:** `SynapticConfusionDetector` (recovery.py:397-540) does pattern matching but does NOT call `predict_next_thoughts()`.

**Confirmation point:** Demo showing confusion detection during task execution.

### Step 2: Wire PRISM-SLM Codebase Knowledge

**Goal:** Use PRISM-SLM's codebase knowledge to suggest relevant files/patterns.

**Interface concept:**
```python
# Given a task context, suggest relevant code locations
suggestions = prism_slm.suggest_for_context("working on indexer")
# Returns: ["got/indexer.py", "cortical/processor"]
```

**Why this matters:** Generic LLM doesn't know `got indexer → py`. PRISM-SLM does.

**Confirmation point:** Demo showing PRISM-SLM suggesting codebase-specific paths.

### Step 3: Enrich with Git History

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

### Critical Lines for Consolidation (agents.py)

| Line | Code | Issue |
|------|------|-------|
| 887 | `self._learning_cycle = LearningCycle(...)` | Creates LOCAL cycle in `__init__` |
| 899 | `self._got_learning_bridge = GoTLearningBridge(...)` | Creates GOT bridge (KEEP) |
| 1611 | `lessons = self._get_lessons_for_task(...)` | Uses LOCAL cycle, no feedback |
| 1627 | `got_guidance = self._got_learning_bridge.get_guidance_for_task(...)` | Uses GOT bridge (KEEP) |
| 1717 | `learning_cycle = LearningCycle(storage_dir)` | Creates ANOTHER local cycle |
| 1726 | `experience = learning_cycle.start_experience(...)` | Records to LOCAL |
| 1840 | `"lessons_retrieved": len(lessons)` | Uses LOCAL lessons for metrics |
| 1907 | `self._got_learning_bridge.cycle.validate_lesson(...)` | GOT feedback (KEEP) |
| 1920 | `self._got_learning_bridge.capture_task_completion(...)` | GOT capture (KEEP) |
| 1958 | `learning_cycle.complete_experience(...)` | Records to LOCAL (duplicate!) |
| 1981 | `self._got_learning_bridge.capture_task_failure(...)` | GOT capture (KEEP) |

### SynapticConfusionDetector Analysis

**Location:** `llm_orchestration/recovery.py:397-540`

The current confusion detector does pattern matching but does NOT use PRISM predictions:

```python
def detect(self, context: Optional[Dict[str, Any]] = None) -> List[ConfusionSignal]:
    """Detect confusion from synaptic patterns."""
    signals = []
    loop_signal = self._detect_activation_loop()           # Pattern matching
    contradiction_signals = self._detect_contradictory_activations()  # Pattern matching
    stagnation_signal = self._detect_stagnation()          # Pattern matching
    oscillation_signal = self._detect_oscillation()        # Pattern matching
    # NOTE: Does NOT call predict_next_thoughts()!
    return signals
```

**For PRISM-GoT integration:** This is where `predict_next_thoughts()` should be wired.

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
python -m cortical.got validate
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
6. **Understand before building:** Traced full execution flow before adding complexity

---

## Handoff Instructions

### For the Next Session

1. **Read this document first** - especially the "Duplicate Learning Systems" section

2. **The blocking task is STEP 0** - Consolidate learning systems before PRISM work

3. **Start with:**
   ```bash
   # Verify system state
   python -m cortical.got validate
   python -m pytest tests/smoke/ -v

   # Run the demos to understand current state
   python3 demo_real_got_data.py
   python3 demo_learning_system.py
   ```

4. **Key decision needed:** Confirm the consolidation approach (remove local, keep GoT)

5. **Working constraints from this session:**
   - No sub-agents - work through issues together directly
   - Commit and push immediately after saves
   - TDD approach - write behavioral tests first
   - Confirmation demos between each step

### User Context

The reviewer is a **Senior Principal Computer Scientist** who:
- Caught my dismissal of PRISM-SLM (I was wrong - it knows the codebase)
- Asked to "take things slow" and "understand before building"
- Prefers Option B (understand full execution flow) over Option A (just run tests)
- Values confirmation examples between each step

### Current Todo List

1. **[in_progress]** Consolidate duplicate learning systems before adding PRISM
2. **[pending]** Wire PRISM-GoT confusion detection into Worker
3. **[pending]** Enrich Layer 1 with git history data
4. **[pending]** Wire PRISM-SLM codebase knowledge into guidance

---

*This knowledge transfer captures the state as of session end. Continue with STEP 0 (consolidate learning systems) before any PRISM work.*
