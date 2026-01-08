# Audit Result: 20260107-120100-got-comments

**Audit:** misleading-comments-2026-01-07
**Scope:** `cortical/got/`
**Patterns:** `FUTURE:|TODO:|FIXME:|PLANNED:|HACK:|XXX:|TEMPORARY:|WORKAROUND:` + "will be", "should be", "planned to"
**Date:** 2026-01-07
**Status:** COMPLETE

---

## Executive Summary

Searched 56 Python files in `cortical/got/` for comment patterns indicating future work, TODOs, or workarounds.

**Total Findings:** 5
**Breakdown:**
- **misleading:** 3 (60%)
- **accurate:** 2 (40%)
- **stale:** 0 (0%)
- **unknown:** 0 (0%)

**Key Issues:**
1. Two FUTURE comments reference a design document that does not exist
2. One workaround comment presents speculation as fact

---

## Findings

### FINDING 1: Non-existent Design Document Reference

**File:** `/home/user/Opus-code-test/cortical/got/indexer.py`
**Line:** 478-480
**Git Blame:** d7a3e8415 (2026-01-05)

**Comment:**
```python
FUTURE: When CDG index is implemented, this will be handled at the
storage layer with WAL-based recovery. See:
docs/design/cdg-transactional-indexing-design.md
```

**Category:** **misleading**

**Evidence:**
```bash
$ ls -la docs/design/cdg-transactional-indexing-design.md
ls: cannot access 'docs/design/cdg-transactional-indexing-design.md': No such file or directory
```

The referenced design document does not exist. Comment written 2026-01-05, verified 2026-01-07.

**Decision Tree Path:**
- Does the comment reference a specific file or document? **YES**
- Does that file exist? **NO**
- **Category: misleading** (reference doesn't exist)

---

### FINDING 2: Non-existent Design Document Reference (Duplicate)

**File:** `/home/user/Opus-code-test/cortical/got/indexer.py`
**Line:** 508-510
**Git Blame:** d7a3e8415 (2026-01-05)

**Comment:**
```python
FUTURE: When CDG index is implemented, this will be replaced by
CDG's recovery manager which rebuilds indexes as part of WAL replay.
See: docs/design/cdg-transactional-indexing-design.md
```

**Category:** **misleading**

**Evidence:**
Same as Finding 1 - references the same non-existent design document.

**Decision Tree Path:**
- Does the comment reference a specific file or document? **YES**
- Does that file exist? **NO**
- **Category: misleading** (reference doesn't exist)

---

### FINDING 3: Unimplemented Decision Tracking

**File:** `/home/user/Opus-code-test/cortical/got/orphan.py`
**Line:** 195
**Git Blame:** ^25d80384 (2025-12-24)

**Comment:**
```python
orphan_decisions=[],  # TODO: Add decision tracking
```

**Category:** **accurate**

**Evidence:**
The `OrphanReport` dataclass defines `orphan_decisions: List[str]` field (line 47), but it is always returned as an empty list. The field exists in the data structure but the tracking functionality has not been implemented.

```python
# From orphan.py lines 42-50
@dataclass
class OrphanReport:
    """Report of orphan entities in the GoT graph."""

    orphan_tasks: List[str] = field(default_factory=list)
    orphan_decisions: List[str] = field(default_factory=list)  # Field exists
    total_tasks: int = 0
    total_decisions: int = 0
    orphan_rate: float = 0.0
```

Current implementation (line 195) returns empty list with TODO noting the missing functionality.

**Decision Tree Path:**
- Does the comment reference a specific file or document? **NO**
- Does the comment describe code behavior? **YES**
- Does code actually behave that way? **NO** (feature not implemented)
- Is it speculation/aspiration? **NO** (TODO marker indicates known missing feature)
- **Category: accurate** (correctly identifies unimplemented feature)

---

### FINDING 4: Unimplemented Learning Integration

**File:** `/home/user/Opus-code-test/cortical/got/cli/failure.py`
**Line:** 420-422
**Git Blame:** ^9de46f31 (2026-01-03)

**Comment:**
```python
# TODO: Feed into LearningCycle if available
# This would require checking for the learning cycle module
# and calling an appropriate method to record the lesson
```

**Category:** **accurate**

**Evidence:**
`LearningCycle` exists in the codebase at `/home/user/Opus-code-test/llm_orchestration/learning.py` and is imported by `cortical/got/learning_integration.py`. However, the failure recording code in `cli/failure.py` does not integrate with it yet.

```bash
$ find /home/user/Opus-code-test -name "learning.py"
/home/user/Opus-code-test/llm_orchestration/learning.py
/home/user/Opus-code-test/benchmarks/prism_slm/learning.py

$ grep -n "class LearningCycle" cortical/got/learning_integration.py
41:    class LearningCycle:
```

The TODO correctly identifies a missing integration point where failures could be fed into the learning system.

**Decision Tree Path:**
- Does the comment reference a specific file or document? **NO**
- Does the comment describe code behavior? **YES**
- Does code actually behave that way? **NO** (integration not implemented)
- Is it speculation/aspiration? **NO** (TODO marker, target system exists)
- **Category: accurate** (correctly identifies missing integration)

---

### FINDING 5: Workaround Speculation as Fact

**File:** `/home/user/Opus-code-test/cortical/got/expression/executor.py`
**Line:** 401
**Git Blame:** 07b7a1692 (2026-01-05)

**Comment:**
```python
# NOTE: This directly modifies the query builder's internal state.
# Ideally, Query builder would expose .where_op(field, operator, value)
# This workaround will be replaced when that API is added.
```

**Category:** **misleading**

**Evidence:**
The comment states "will be replaced when that API is added" but there is no evidence that a `where_op` API is planned:

1. **Query builder does not have where_op method:**
```bash
$ grep -n "def where_op" cortical/got/query_builder.py
(no results)
```

2. **No documentation of planned API:**
```bash
$ grep -r "where_op" docs/
(no results)
```

3. **No commit history or TODOs:**
```bash
$ git log --all --oneline --grep="where_op"
(no relevant results)

$ grep -r "TODO.*where_op\|FUTURE.*where_op" cortical/got/
(no results)
```

The current code uses the workaround (accessing `_where_clauses` directly) with no concrete plan to change it. The phrase "will be replaced when that API is added" presents speculation as fact.

**Decision Tree Path:**
- Does the comment reference a specific file or document? **NO**
- Does the comment describe code behavior? **YES**
- Does code actually behave that way? **YES** (workaround exists and is used)
- Is it speculation/aspiration? **YES** (states future plan with no evidence)
- **Category: misleading** (speculation presented as fact)

---

## Findings Excluded (Not in Scope)

The following matches were found but **excluded** because they use "will be"/"should be" **descriptively** (describing current behavior) rather than **aspirationally** (describing future plans):

| File | Line | Phrase | Reason for Exclusion |
|------|------|--------|---------------------|
| validation.py | 308 | "should be prevented" | Design principle comment, not future plan |
| validation.py | 567 | "should be hex string" | Documentation of validation rule |
| validation.py | 769 | "should be used when creating" | Usage guidance, not TODO |
| tx_manager.py | 453 | "will be created" | Describes current behavior (future tense grammar) |
| tx_manager.py | 454 | "will be appended" | Describes current behavior (future tense grammar) |
| recovery.py | 142 | "should be performed" | Function return value documentation |
| graph_walker.py | 514 | "should be visited" | Method docstring describing behavior |
| cli/query.py | 537 | "will be executed" | Describes what explain command shows |
| cli/query.py | 673 | "will be parsed and executed" | Help text describing command behavior |
| cli/failure.py | 72 | "XXXXXXXX: 8 hex characters" | ID format documentation (not XXX marker) |
| types.py | 1285 | "should be included" | Method docstring |
| types.py | 1291 | "should be included" | Return value documentation |

---

## What Went Wrong

### Root Cause Analysis

**Why did misleading comments get written?**

1. **Finding 1 & 2 (indexer.py FUTURE comments):**
   - Design document was referenced before it was created
   - Comment written 2026-01-05, document never materialized
   - Likely cause: Developer intended to write design doc but didn't follow through

2. **Finding 5 (executor.py workaround comment):**
   - Developer expressed personal preference ("Ideally...") as future plan
   - No verification that team agreed to add the API
   - Conflation of "would be nice" with "will happen"

### Pattern: Premature Reference

All three misleading comments share a pattern:
- Reference something that doesn't exist (design doc, API)
- Use definitive language ("will be", "See:")
- No follow-up to verify the reference is valid

### Preventive Measures

**For future work:**

1. **Design documents first:** Don't reference design docs that don't exist
2. **Conditional language:** Use "could be" instead of "will be" for uncertain plans
3. **TODO audit cycle:** Periodically check that referenced artifacts exist

---

## Recommendations

### Immediate Actions

1. **Update indexer.py comments (lines 478-480, 508-510):**
   - Remove reference to non-existent design document
   - Either create the document or change to: "FUTURE: Consider WAL-based recovery at CDG layer"

2. **Update executor.py comment (line 401):**
   - Change "will be replaced" to "could be improved"
   - OR: Create a task to add where_op API if the team agrees it's needed
   - Suggested rewording:
   ```python
   # NOTE: This directly modifies the query builder's internal state.
   # Could be improved with a dedicated .where_op(field, operator, value) API
   ```

3. **Keep orphan.py and failure.py TODOs:**
   - These are accurate markers of unimplemented features
   - Both reference systems that exist (OrphanReport fields, LearningCycle)
   - No action needed

### Long-term

- Establish convention: Don't reference design docs until they exist
- Consider pre-commit hook to check file references in comments
- Periodic TODO/FUTURE audit (quarterly?)

---

## Metadata

**Search Commands Used:**
```bash
grep -n -r -E "(FUTURE:|TODO:|FIXME:|PLANNED:|HACK:|XXX:|TEMPORARY:|WORKAROUND:|will be|should be|planned to)" \
  /home/user/Opus-code-test/cortical/got/ --include="*.py"

git blame -L <line>,<line> <file> --date=short

ls -la docs/design/cdg-transactional-indexing-design.md
grep -n "def where_op" cortical/got/query_builder.py
find /home/user/Opus-code-test -name "learning.py"
```

**Files Scanned:** 56 Python files in `cortical/got/`
**Total Lines Scanned:** Approximately 15,000
**Duration:** ~15 minutes
**Constraints Respected:**
- ✅ Stayed within scope (cortical/got/ only)
- ✅ Under 50 findings limit (5 findings)
- ✅ Under 2 hour limit (~15 minutes)

---

## Certification

I certify that:
- ✅ All three pre-flight questions answered YES
- ✅ All four category definitions were applied using the decision tree
- ✅ Evidence is cited with specific file paths, line numbers, and command outputs
- ✅ No category was assigned without definitive evidence
- ✅ The decision tree path is documented for each finding

**Auditor:** Claude (Sonnet 4.5)
**Date:** 2026-01-07
**Task ID:** 20260107-120100-got-comments
