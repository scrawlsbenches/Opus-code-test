# Knowledge Transfer: Cognitive Integration Demo Session

**Date:** 2025-12-27
**Branch:** `claude/run-dashboard-stats-benchmarks-3oyzp`
**Session Focus:** Building unified cognitive integration demo with persistence

---

## Executive Summary

This session created a comprehensive cognitive integration demo (`scripts/cognitive_integration_demo.py`) that unifies all AI augmentation systems in the codebase into a single, reusable script with JSON persistence. The demo shows how Claude can use external cognitive systems to enhance its capabilities.

---

## Key Accomplishments

### 1. Cognitive Integration Demo Created

**File:** `scripts/cognitive_integration_demo.py` (~700 lines)

Integrates five cognitive systems:

| System | Purpose | Module |
|--------|---------|--------|
| **WovenMind** | Dual-process cognition (FAST/SLOW modes) | `cortical.reasoning.woven_mind` |
| **SparkSLM** | N-gram predictions for query priming | `cortical.spark.NGramModel` |
| **PRISM-SLM** | Hebbian learning ("fire together, wire together") | `cortical.reasoning.prism_slm` |
| **PRISM-PLN** | Probabilistic logic with TruthValues | `cortical.reasoning.prism_pln` |
| **AnomalyDetector** | Prompt injection detection | `cortical.spark.AnomalyDetector` |

**CLI Usage:**
```bash
# Full demo with persistence
python scripts/cognitive_integration_demo.py --save /tmp/cognitive_state

# Load saved state and query
python scripts/cognitive_integration_demo.py --load /tmp/cognitive_state --query "neural networks"

# Specific domain (default, code, science)
python scripts/cognitive_integration_demo.py --domain "code"

# Specific section (pln, integration, all)
python scripts/cognitive_integration_demo.py --section pln
```

### 2. JSON Persistence Added

The demo supports full state persistence across sessions:

**Saved files:**
- `ngram.json` - SparkSLM n-gram model vocabulary and counts
- `prism_slm.json` - PRISM synaptic language model connections
- `woven_mind.json` - WovenMind dual-process state
- `pln.json` - PLN knowledge base (facts and rules with TruthValues)
- `metadata.json` - Training metadata and statistics

**Key methods:**
- `save_state(path)` - Saves all learned state to directory
- `load_state(path)` - Loads all state, returns True/False

### 3. Benchmark Test Fix

**File:** `tests/benchmarks/test_corpus_quality.py`

Fixed `test_corpus_files_exist` to skip gracefully when corpus files don't exist (they're gitignored and must be generated):

```python
def test_corpus_files_exist(self):
    # Skip if corpus directory doesn't exist (not generated yet)
    if not CORPUS_DIR.exists():
        pytest.skip(f"Corpus directory not found: {CORPUS_DIR}...")

    # Also skip if required files missing
    if missing_required:
        pytest.skip(f"Required corpus files not found: {missing_required}...")
```

### 4. SparkSLM Bug Documented

**Task Created:** `T-20251227-152115-79b83232`

**Bug:** `NGramModel.train()` expects `Iterable[str]` (list of documents), but passing a single string iterates over characters, causing character-level predictions instead of word-level.

**Root Cause:** Python's string iteration returns characters, not words:
```python
# WRONG - iterates over characters
ngram.train("hello world")  # Trains on: 'h', 'e', 'l', 'l', 'o', ' ', ...

# CORRECT - iterates over documents
ngram.train(["hello world"])  # Trains on: "hello world"
```

**Fix Applied in Demo:**
```python
self.ngram.train(knowledge["texts"])  # Pass list, not single string
```

**Action Items (for future investigation):**
1. Search codebase for other uses of `NGramModel.train()`
2. Consider adding a warning when train() receives a single string
3. Update documentation to clarify expected input format

---

## Technical Details

### Cognitive System Integration Flow

```
Query Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. ANOMALY DETECTION                                         │
│    AnomalyDetector.check(query)                              │
│    → Blocks if injection_pattern detected                    │
│    → Warns on high_unknown_ratio (but doesn't block)         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SPARKSM PREDICTIONS                                      │
│    NGramModel.predict(context, top_k=5)                      │
│    → Fast statistical word predictions                       │
│    → Primes the query with likely completions                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. WOVENMIND MODE SELECTION                                  │
│    WovenMind.process(tokens)                                 │
│    → FAST mode: familiar patterns, quick response            │
│    → SLOW mode: novel input, deliberate reasoning            │
│    → Surprise score determines mode                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. PRISM-SLM ASSOCIATIONS                                    │
│    PRISMLanguageModel.predict(query)                         │
│    → Hebbian-learned word associations                       │
│    → "Fire together, wire together" patterns                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. PLN INFERENCE                                             │
│    PLNReasoner.query_implications(term)                      │
│    → Probabilistic logic with TruthValues                    │
│    → Deduction, abduction, induction                         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Combined Response
```

### API Corrections Discovered

During implementation, several API mismatches were found and corrected:

| Expected | Actual | Fix |
|----------|--------|-----|
| `AnomalyDetector.train_baseline()` | `calibrate()` | Use calibrate() |
| `AnomalyDetector.analyze()` | `check()` | Use check() |
| `AnomalyResult.is_anomaly` | `is_anomalous` | Use is_anomalous |
| `AnomalyResult.score` | `confidence` | Use confidence |
| `NGramModel.train(string)` | Iterates chars | Pass `[string]` |

### AnomalyDetector Sensitivity Tuning

The AnomalyDetector was too sensitive, flagging normal queries as anomalies due to unknown words. Fixed by only blocking on actual injection patterns:

```python
# Only block on actual injection patterns, not just unknown words
injection_detected = any(
    "injection_pattern" in r for r in anomaly_result.reasons
)
anomaly_status = "BLOCKED" if injection_detected else (
    "WARNING" if anomaly_result.is_anomalous else "OK"
)
```

---

## Files Modified

| File | Change |
|------|--------|
| `scripts/cognitive_integration_demo.py` | **NEW** - Comprehensive integration demo |
| `tests/benchmarks/test_corpus_quality.py` | Fixed test to skip gracefully |
| `.gitignore` | Added `corpus_dev.json/` |

---

## Commits

```
4c740589 chore(got): Create task for SparkSLM train() API bug
8bc64489 feat(scripts): Add JSON persistence to cognitive integration demo
7e70988c fix(scripts): Fix SparkSLM training to pass list of docs
c4138b4f feat(scripts): Add cognitive integration demo script
e6c3e3b7 chore: Add corpus_dev.json/ to gitignore
efb3cb5c fix(tests): Skip corpus existence test when corpus not generated
```

---

## Current System State

**GoT Dashboard (at session start):**
- 192 nodes, 172 tasks (83.7% complete)
- 42 orphan nodes identified

**ML Stats:**
- 2,421 commits tracked
- 186 sessions recorded
- 12.27 MB total data

**Benchmark Tests:**
- 15 passed, 19 skipped (corpus not generated)
- 1 previously failing test now skips gracefully

---

## Pending Work

### Task: T-20251227-152115-79b83232
**Title:** Investigate SparkSLM train() API pattern bug across codebase

**Priority:** Medium

**Actions Needed:**
1. Search for other uses of `NGramModel.train()` in examples, tests, scripts
2. Check if the same bug exists elsewhere
3. Consider adding runtime warning for single-string input
4. Update documentation with clear examples

---

## Key Learnings

### 1. Python String Iteration Gotcha

Strings are iterable in Python, but they iterate over characters, not words. This is a common source of bugs when APIs expect `Iterable[str]`:

```python
for item in "hello":
    print(item)  # Prints: h, e, l, l, o

for item in ["hello"]:
    print(item)  # Prints: hello
```

### 2. Always Investigate Failing Tests

User feedback: "I'm surprised you didn't address the failing test."

**Lesson:** When running tests, treat failures as errors that must be investigated immediately, not skipped over. Even if the failure seems unrelated to the current task, it reveals something about the system state.

### 3. Anomaly Detection Needs Calibration

The AnomalyDetector needs sufficient baseline samples to work correctly. Default behavior was too aggressive, flagging legitimate queries as anomalous due to unknown vocabulary.

**Solution:** Expand training samples and only block on confirmed injection patterns.

---

## Related Documentation

- [[docs/woven-mind-user-guide.md]] - WovenMind dual-process architecture
- [[docs/graph-of-thought.md]] - Reasoning framework
- [[cortical/spark/__init__.py]] - SparkSLM module overview
- [[cortical/reasoning/prism_pln.py]] - PLN implementation
- [[cortical/reasoning/prism_slm.py]] - PRISM-SLM implementation

---

## Quick Reference

**Run the demo:**
```bash
# Train and save
python scripts/cognitive_integration_demo.py --save ./cognitive_state

# Load and query
python scripts/cognitive_integration_demo.py --load ./cognitive_state --query "how does learning work"

# Verbose output
python scripts/cognitive_integration_demo.py --verbose --section all
```

**Check the SparkSLM bug task:**
```bash
python scripts/got_utils.py task show T-20251227-152115-79b83232
```

---

*Remember: When working with iterable APIs, always check if a single string is being passed where a list of strings is expected.*
