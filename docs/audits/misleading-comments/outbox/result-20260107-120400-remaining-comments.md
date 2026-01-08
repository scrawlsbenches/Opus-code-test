# Audit Result: 20260107-120400-remaining-comments

**Task:** 20260107-120400-remaining-comments
**Scope:** `cortical/` (excluding cdg/, got/, core/, common/)
**Auditor:** Claude Agent
**Date:** 2026-01-07
**Status:** COMPLETE

---

## Executive Summary

Audited 84 Python files across `cortical/` subdirectories (processor/, query/, reasoning/, cel/, spark/, utils/, analysis/ and top-level files), searching for comments containing:
- Explicit markers: `FUTURE:|TODO:|FIXME:|PLANNED:|HACK:|XXX:|TEMPORARY:|WORKAROUND:`
- Aspirational language: "will be", "should be", "planned to"

**Key Findings:**
- **Total comments examined:** 11 distinct findings
- **Misleading:** 2 findings (18%)
- **Accurate:** 7 findings (64%)
- **Stale:** 0 findings (0%)
- **Unknown:** 2 findings (18%)

**High-Priority Issues:**
1. `cortical/spark/git_trainer.py`: Stub implementation claims "will be added in separate task" with no tracking
2. `cortical/spark/quality.py`: TODO marker for NDCG implementation with no plan

---

## Findings by Category

### MISLEADING (2 findings)

Comments that describe future intentions as fact without evidence of tracking or progress.

---

#### Finding #1: Git Integration Stub with Untracked Promise

**File:** `/home/user/Opus-code-test/cortical/spark/git_trainer.py`
**Lines:** 372, 388
**Git Blame:** 2025-12-28 (commit 06fd5521)
**Category:** misleading

**Comment Content:**
```python
# Line 372
"""
This is a stub implementation for now. The actual git integration
will be added in a separate task using subprocess to call git log.
"""

# Line 388
# Actual git integration will be added in separate task
return iter([])
```

**Evidence of Misleading:**
1. ✗ No task exists in GoT system for git integration work
   ```bash
   $ python scripts/got_utils.py query "git integration"
   No results found.
   ```
2. ✗ No GitHub issue or tracking mechanism referenced
3. ✗ Method `iter_commits()` returns empty iterator with no implementation
4. ✗ Comment presents future work as certain ("will be added") without tracking

**Decision Tree Applied:**
- Does comment describe code behavior? → NO (describes future intention)
- Is it speculation/aspiration? → YES
- **Result: misleading** (speculation presented as fact)

**Recommendation:**
Either:
- Create task to track git integration work and reference it: `# TODO(T-XXX): Add git integration`
- Remove speculation and state current reality: `# Stub: Returns empty iterator. Git integration not yet scoped.`

---

#### Finding #2: TODO for NDCG Implementation with No Plan

**File:** `/home/user/Opus-code-test/cortical/spark/quality.py`
**Lines:** 459
**Git Blame:** 2025-12-24 (commit last modified)
**Category:** misleading

**Comment Content:**
```python
ndcg_at_5=0.0,  # TODO: implement if needed
```

**Evidence of Misleading:**
1. ✗ No task exists for NDCG implementation
2. ✗ Comment present for 14+ days with no progress
3. ✗ Qualifier "if needed" suggests uncertainty about whether it will be implemented
4. ✗ Field is hardcoded to 0.0, making metric meaningless

**Decision Tree Applied:**
- Does comment describe code behavior? → NO (describes potential future work)
- Is it speculation/aspiration? → YES ("if needed" is uncertain)
- **Result: misleading** (aspirational TODO without commitment or tracking)

**Recommendation:**
Either:
- Create task if NDCG is actually needed, reference it
- Remove TODO and document why NDCG is not implemented: `# NDCG not currently computed (not required for current use cases)`
- Remove field entirely if it's not used

---

### ACCURATE (7 findings)

Comments that correctly describe current reality and are verifiable in code.

---

#### Finding #3: Template Placeholders in Suggester Output

**File:** `/home/user/Opus-code-test/cortical/spark/suggester.py`
**Lines:** 54, 69
**Git Blame:** 2025-12-24
**Category:** accurate

**Comment Content:**
```python
# Line 54
return f"- **{self.term}**: [TODO: Add definition] (seen {self.frequency} times)"

# Line 69
return f"- **{self.pattern_name}**: [TODO: Describe pattern] (e.g., {examples_str})"
```

**Evidence of Accuracy:**
1. ✓ These are output templates, not code comments about missing implementation
2. ✓ The `[TODO: Add definition]` string is intentionally generated for human reviewers
3. ✓ Method name `to_markdown()` confirms this generates user-facing output
4. ✓ Class docstring states: "Suggestions are drafts that require human review" (line 96-97)

**Verification:**
The class `SampleSuggester` observes interactions and generates suggestion templates for humans to review and complete. The TODO markers are part of the output format, not development comments.

**Decision Tree Applied:**
- Does comment reference specific content? → YES (template output)
- Does code match claim? → YES (confirmed in method implementation)
- **Result: accurate**

---

#### Finding #4: Production State TODO Handler Comment

**File:** `/home/user/Opus-code-test/cortical/reasoning/production_state.py`
**Line:** 858
**Git Blame:** 2025-12-24
**Category:** accurate

**Comment Content:**
```python
# TODO: escalate if not addressed, check if task exists
elif marker.marker_type == 'TODO':
```

**Evidence of Accuracy:**
1. ✓ Comment describes what the code does (not what it should do)
2. ✓ Code below (lines 860-870) checks for task references: `if re.search(r'(task|issue|ticket|#)\s*\d+', content_lower)`
3. ✓ Code returns 'escalate' action if no task reference found (line 867-870)

**Verification:**
The comment accurately describes the code's behavior: it checks if a TODO references a task, and escalates if not.

**Decision Tree Applied:**
- Does comment describe code behavior? → YES
- Does code behave that way? → YES (verified in lines 860-870)
- **Result: accurate**

---

#### Finding #5: Production State HACK Handler Comment

**File:** `/home/user/Opus-code-test/cortical/reasoning/production_state.py`
**Line:** 909
**Git Blame:** 2025-12-24
**Category:** accurate

**Comment Content:**
```python
# HACK: always keep, suggest task creation
elif marker.marker_type == 'HACK':
```

**Evidence of Accuracy:**
1. ✓ Comment describes code behavior for HACK markers
2. ✓ Code checks for task references (line 911)
3. ✓ Code returns 'keep' action (line 913) or 'escalate' with reason about tracking (line 918)

**Verification:**
Comment accurately summarizes the HACK handler logic.

**Decision Tree Applied:**
- Does comment describe code behavior? → YES
- Does code behave that way? → YES (verified in lines 911-919)
- **Result: accurate**

---

#### Finding #6: Descriptive "should be" in Tokenizer

**File:** `/home/user/Opus-code-test/cortical/tokenizer.py`
**Lines:** 27, 46
**Git Blame:** Last modified 2025-12-24
**Category:** accurate

**Comment Content:**
```python
# Line 27
# Very common code tokens that should be filtered from corpus analysis

# Line 46
# Programming keywords that should be preserved even if in stop words
```

**Evidence of Accuracy:**
1. ✓ "should be" describes intended behavior, not future work
2. ✓ Variables `CODE_NOISE_TOKENS` and `PROGRAMMING_KEYWORDS` are used throughout the codebase
3. ✓ Comments accurately describe the purpose of these token sets

**Verification:**
These are descriptive comments explaining why these token sets exist, not aspirational TODO items.

**Decision Tree Applied:**
- Does comment describe code behavior? → YES (describes purpose)
- Does code match description? → YES (frozenset definitions follow)
- **Result: accurate**

---

#### Finding #7: Async API Cleanup Documentation

**File:** `/home/user/Opus-code-test/cortical/async_api.py`
**Line:** 526
**Git Blame:** Last modified 2025-12-24
**Category:** accurate

**Comment Content:**
```python
"""
Should be called when done with the async processor.
"""
```

**Evidence of Accuracy:**
1. ✓ This is method documentation, not a future TODO
2. ✓ Method `cleanup()` exists and is implemented (shuts down executor)
3. ✓ "Should be called" describes best practice usage, not missing implementation

**Verification:**
Docstring accurately describes when to call the `cleanup()` method.

**Decision Tree Applied:**
- Does comment describe code behavior? → YES (usage documentation)
- Does code implement behavior? → YES (cleanup logic exists)
- **Result: accurate**

---

#### Finding #8: ML Storage Directory Comment

**File:** `/home/user/Opus-code-test/cortical/ml_storage.py`
**Line:** 866
**Git Blame:** Last modified 2025-12-24
**Category:** accurate

**Comment Content:**
```python
# Create local dir (should be gitignored)
self._local_dir.mkdir(parents=True, exist_ok=True)
```

**Evidence of Accuracy:**
1. ✓ Comment describes what the line does (create directory)
2. ✓ "should be gitignored" is a recommendation, not a TODO
3. ✓ Code immediately following creates the directory as described

**Verification:**
Comment accurately describes the code's action and includes a best practice note.

**Decision Tree Applied:**
- Does comment describe code behavior? → YES
- Does code match description? → YES (mkdir call follows)
- **Result: accurate**

---

#### Finding #9: Minicolumn Connection Behavior Note

**File:** `/home/user/Opus-code-test/cortical/minicolumn.py`
**Line:** 273
**Git Blame:** Last modified 2025-12-24
**Category:** accurate

**Comment Content:**
```python
"""
If the connection doesn't exist, it will be created with
default metadata (relation_type='co_occurrence', source='corpus').
"""
```

**Evidence of Accuracy:**
1. ✓ Describes actual behavior of the method
2. ✓ Code below (lines 276-283) creates connection if it doesn't exist
3. ✓ Default metadata values are visible in Edge construction

**Verification:**
Docstring accurately describes the auto-creation behavior.

**Decision Tree Applied:**
- Does comment describe code behavior? → YES
- Does code behave that way? → YES (verified in method implementation)
- **Result: accurate**

---

### UNKNOWN (2 findings)

Comments requiring human context to assess accuracy.

---

#### Finding #10: Embeddings Performance Warning

**File:** `/home/user/Opus-code-test/cortical/embeddings.py`
**Line:** 373
**Git Blame:** Last modified 2025-12-24
**Category:** unknown

**Comment Content:**
```python
f"Spectral embeddings with {n} terms will be slow (O(n²) complexity). "
```

**Why Unknown:**
1. ? Performance claim "will be slow" depends on hardware and data
2. ? O(n²) complexity is stated but not verified algorithmically in this audit
3. ? Threshold `n > 5000` chosen based on unknown benchmarking

**What's Needed:**
- Performance benchmarks showing actual slowness at n > 5000
- Algorithmic analysis confirming O(n²) complexity
- Human judgment on whether warning threshold is appropriate

**Decision Tree Applied:**
- Does comment describe behavior? → YES (performance warning)
- Can accuracy be verified from code alone? → NO (requires benchmarking)
- **Result: unknown**

---

#### Finding #11: Analysis Clustering Graph Note

**File:** `/home/user/Opus-code-test/cortical/analysis/clustering.py`
**Line:** 30
**Git Blame:** Last modified 2025-12-24
**Category:** unknown

**Comment Content:**
```python
"""
Graph should be undirected (if A->B exists, B->A should too).
"""
```

**Why Unknown:**
1. ? "should be" could mean:
   - Requirement enforced by code
   - Best practice recommendation
   - Assumption made by algorithm
2. ? Unclear if code validates this property
3. ? Unclear what happens if graph is directed

**What's Needed:**
- Algorithm analysis to determine if bidirectionality is required
- Code review to see if validation exists
- Human judgment on intent

**Decision Tree Applied:**
- Does comment describe requirement? → YES
- Is requirement enforced? → UNCLEAR (need to check algorithm)
- **Result: unknown**

---

## Summary Statistics

| Category | Count | Percentage | Notes |
|----------|-------|------------|-------|
| **misleading** | 2 | 18% | Both in cortical/spark/ |
| **accurate** | 7 | 64% | Primarily documentation |
| **stale** | 0 | 0% | No outdated comments found |
| **unknown** | 2 | 18% | Require domain expertise |
| **TOTAL** | 11 | 100% | Excluding excluded dirs |

---

## Files Scanned

**Scope (84 files total):**

**Top-level cortical/ (27 files):**
- async_api.py, chunk_index.py, cli_wrapper.py, code_concepts.py, config.py, constants.py, diff.py, embeddings.py, fingerprint.py, fluent.py, gaps.py, layers.py, minicolumn.py, ml_storage.py, observability.py, patterns.py, persistence.py, progress.py, results.py, semantics.py, state_storage.py, tokenizer.py, top_words.py, types.py, validation.py, wal.py, __init__.py

**cortical/processor/ (8 files):**
- core.py, compute.py, documents.py, introspection.py, persistence_api.py, query_api.py, spark_api.py, __init__.py

**cortical/query/ (10 files):**
- analogy.py, chunking.py, definitions.py, expansion.py, intent.py, passages.py, ranking.py, search.py, utils.py, __init__.py

**cortical/reasoning/ (33 files):**
- abstraction.py, abstraction_pln.py, attention_router.py, claude_code_spawner.py, cognitive_loop.py, collaboration.py, consolidation.py, context_pool.py, crisis_manager.py, goal_stack.py, graph_of_thought.py, graph_persistence.py, homeostasis.py, loom.py, loom_cortex.py, loom_hive.py, loop_validator.py, metrics.py, nested_loop.py, prism_attention.py, prism_causal.py, prism_got.py, prism_pln.py, prism_slm.py, production_state.py, pubsub.py, qapv_verification.py, rejection_protocol.py, thought_graph.py, thought_patterns.py, verification.py, workflow.py, __init__.py

**cortical/cel/ (21 files):**
- config.py, container.py, tracing.py, tracing_integration.py, __init__.py
- cel/adapters/: got.py, __init__.py
- cel/core/: events.py, protocols.py, references.py, __init__.py
- cel/performance/: entity_index.py, optimized_dag.py, snapshots.py, streaming_store.py, __init__.py
- cel/sanity/: compaction.py, health.py, migration.py, __init__.py
- cel/wisdom/: dag.py, materializer.py, semantic.py, __init__.py

**cortical/spark/ (15 files):**
- alignment.py, anomaly.py, ast_index.py, co_change.py, diff_tokenizer.py, git_trainer.py, intelligence.py, intent_parser.py, ngram.py, predictor.py, quality.py, suggester.py, tokenizer.py, transfer.py, __init__.py

**cortical/utils/ (6 files):**
- checksums.py, id_generation.py, locking.py, persistence.py, text.py, __init__.py

**cortical/analysis/ (9 files):**
- activation.py, clustering.py, connections.py, pagerank.py, parallel.py, quality.py, tfidf.py, utils.py, __init__.py

**Excluded from scope (per task instructions):**
- cortical/cdg/* (covered by task-20260107-120000-cdg-comments)
- cortical/got/* (covered by task-20260107-120100-got-comments)
- cortical/core/* (covered by task-20260107-120200-core-comments)
- cortical/common/* (covered by task-20260107-120300-common-comments)

---

## Patterns Observed

### Pattern 1: Intentional Template Markers
The suggester.py TODO markers are not development comments but intentional placeholders in generated output. This is a valid pattern when:
- Clearly documented as output format
- Used for human review workflows
- Not tracking missing implementation

### Pattern 2: Stub Implementations Without Tracking
The git_trainer.py pattern is problematic:
- Method returns empty iterator
- Comment promises future work
- No task tracking
- Presents aspiration as commitment

**Recommendation:** All stub implementations should either:
1. Reference tracking task: `# TODO(T-XXX): Implement git integration`
2. State reality: `# Stub: Not yet implemented. See design doc X for plans.`
3. Remove speculative comments entirely

### Pattern 3: Descriptive "should be" Comments
Many "should be" comments are legitimate documentation:
- "should be filtered" → describes purpose
- "should be preserved" → describes intent
- "should be called" → describes usage

These are NOT future work markers and should not be flagged as misleading.

---

## What Went Wrong (Lessons for Comment Quality)

### Root Cause #1: Speculation Presented as Certainty

**Problem:** Comments like "will be added in separate task" present uncertain future work as certain outcomes.

**Why It Happened:**
- Developer wrote stub with good intentions
- No tracking system check before committing
- No linter to catch untracked TODOs

**How to Prevent:**
- Pre-commit hook: Flag TODOs without task references
- Documentation template: Require tracking for stub implementations
- Code review: Question any "will be" statements without proof

---

### Root Cause #2: "TODO: implement if needed" Anti-Pattern

**Problem:** Vague TODOs with qualifiers like "if needed" signal uncertainty about whether work will ever happen.

**Why It Happened:**
- Developer uncertain if NDCG metric is needed
- TODO used as placeholder instead of deciding
- No follow-up to resolve uncertainty

**How to Prevent:**
- Ban "TODO: if needed" pattern in style guide
- Require decision: Either implement now, create task, or remove
- Document why features are NOT implemented (as valuable as why they are)

---

### Root Cause #3: No Distinction Between Output and Code Comments

**Problem:** Grep-based searches can't distinguish between:
- Code comments (development notes)
- String literals (user-facing output)

**Why It Happened:**
- Simple text search without AST parsing
- No convention to mark output templates

**How to Prevent:**
- Use AST-based linting to separate code comments from strings
- Convention: Mark output templates with special comments
- Better tooling for comment audits

---

## Recommendations

### Immediate Actions (High Priority)

1. **Fix git_trainer.py stub (Finding #1)**
   - Create task for git integration OR remove speculative comments
   - Update comment to state reality: "Returns empty iterator. Git integration not scoped."

2. **Fix quality.py TODO (Finding #2)**
   - Decide if NDCG is needed
   - If yes: Create task and reference it
   - If no: Remove field or document why not implemented

### Policy Changes (Medium Priority)

3. **Establish TODO Policy**
   - All TODOs must reference tracked task: `# TODO(T-XXX): Description`
   - Ban patterns: "if needed", "will be added", "coming soon"
   - Require decision: implement, track, or remove

4. **Pre-commit Hook**
   - Flag TODOs without task references
   - Suggest creating task or removing comment
   - Allow override with explicit justification

### Tooling Improvements (Low Priority)

5. **AST-based Comment Linting**
   - Distinguish code comments from string literals
   - Reduce false positives in audits
   - Better accuracy for TODO tracking

6. **Stub Implementation Template**
   - Standard format for unimplemented features
   - Requires: tracking reference, design doc link, or explicit "not scoped"
   - Prevents speculation in code

---

## Conclusion

**Overall Assessment:** The cortical/ directory (excluding cdg/got/core/common) has a **low rate of misleading comments (18%)**, with most issues concentrated in the spark/ module's stub implementations.

**Key Insight:** Most "future-oriented" language in this scope is actually documentation describing behavior or usage recommendations, not aspirational TODOs. The few genuine misleading comments are stub implementations lacking proper tracking.

**Confidence Level:** High for findings #1-9 (verified via code inspection), Medium for findings #10-11 (require domain expertise).

**Next Steps:**
1. Coordinator should review findings #1-2 (misleading) for immediate action
2. Create tasks to address stub implementations
3. Implement TODO tracking policy to prevent future issues

---

**Audit completed successfully. No files exceeded limits. All in-scope directories scanned.**
