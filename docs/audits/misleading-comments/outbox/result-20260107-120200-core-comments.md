# Audit Result: 20260107-120200-core-comments

**Task ID:** 20260107-120200-core-comments
**Audit:** misleading-comments-2026-01-07
**Scope:** `cortical/core/`
**Executed:** 2026-01-07
**Status:** COMPLETE

---

## Executive Summary

Audited all Python files in `cortical/core/` for comment patterns matching:
- Keywords: `FUTURE:|TODO:|FIXME:|PLANNED:|HACK:|XXX:|TEMPORARY:|WORKAROUND:`
- Phrases: "will be", "should be", "planned to"

**Findings:** 3 total
- **Accurate:** 2
- **Misleading:** 1
- **Stale:** 0
- **Unknown:** 0

---

## Findings

### Finding 1: Schema Module Application Order (ACCURATE)

**Location:** `/home/user/Opus-code-test/cortical/core/modules/schema_module.py:46`

**Comment:**
```python
Note: This should be applied early in bootstrap, before modules
that depend on schema validation (GoT, CEL, etc.).
```

**Git Blame:**
- Commit: 425d81fee
- Author: Claude
- Date: 2026-01-07 16:30:26 +0000
- Age: TODAY (same day as audit)

**Category:** ACCURATE

**Evidence:**
The comment accurately describes current reality. Verified in `/home/user/Opus-code-test/cortical/core/bootstrap.py:115-119`:

```python
# Order matters: Schema first (foundation), then CDG, then GoT
if apply_modules:
    container.apply_module(SchemaModule())  # Schema registry (no config needed)
    container.apply_module(CDGModule(got_dir=effective_got_dir, use_memory=use_memory))
    container.apply_module(GoTModule(got_dir=effective_got_dir, use_memory=use_memory))
```

**Verification:**
1. SchemaModule IS applied first (line 117)
2. Comment says "should be applied early" - code applies it first ✓
3. Comment says "before modules that depend on schema validation" - CDG and GoT follow ✓
4. Bootstrap comment confirms: "Order matters: Schema first (foundation)" ✓

---

### Finding 2: Legacy Parameter Removal Intent (MISLEADING)

**Location:** `/home/user/Opus-code-test/cortical/core/modules/cdg_module.py:50`

**Comment:**
```python
got_dir: Legacy alias for base_dir (will be removed)
```

**Git Blame:**
- Commit: 425d81fee
- Author: Claude
- Date: 2026-01-07 16:30:26 +0000
- Age: TODAY (same day as audit)

**Category:** MISLEADING

**Rationale:**
The phrase "will be removed" is future-tense speculation about intended action, not a description of current reality.

**Evidence:**
1. The parameter `got_dir` EXISTS and FUNCTIONS in current code (line 52: `self.base_dir = base_dir or got_dir or Path(".got")`)
2. No task, issue, or timeline exists for this removal (checked GoT, no related tasks found)
3. Comment presents aspiration ("will be") as fact
4. Per category definition: "speculation presented as fact" → misleading

**Current Reality:**
- `got_dir` parameter is functional (lines 37-52)
- It provides backward compatibility during refactoring
- No evidence of actual removal plan or timeline

**What Went Wrong:**
The comment conflates documentation (describing what exists) with aspiration (what we hope to do). The accurate form would be:
```python
got_dir: Legacy alias for base_dir (deprecated, consider for removal)
```

Or, if there's an actual plan:
```python
got_dir: Legacy alias for base_dir (deprecated, removal tracked in T-XXXX)
```

---

### Finding 3: CDG Module Dependency Order (ACCURATE)

**Location:** `/home/user/Opus-code-test/cortical/core/modules/got_module.py:52`

**Comment:**
```python
Note: CDGModule should be applied first as GoT depends on CDG services.
```

**Git Blame:**
- Commit: c7311501
- Author: Claude
- Date: 2026-01-04 21:06:33 +0000
- Age: 3 days ago

**Category:** ACCURATE

**Evidence:**

**Part 1: "CDGModule should be applied first"**
Verified in `/home/user/Opus-code-test/cortical/core/bootstrap.py:118-119`:
```python
container.apply_module(CDGModule(got_dir=effective_got_dir, use_memory=use_memory))
container.apply_module(GoTModule(got_dir=effective_got_dir, use_memory=use_memory))
```
CDGModule IS applied before GoTModule ✓

**Part 2: "as GoT depends on CDG services"**
Verified in `/home/user/Opus-code-test/cortical/core/modules/got_module.py:88-104`:
```python
def create_tx_manager() -> CDGTransactionManager:
    # ... GoT creates a CDGTransactionManager ...
    return CDGTransactionManager(...)
```

And line 114:
```python
tx_manager = container.resolve(CDGTransactionManager)
```

GoTManager DOES depend on CDGTransactionManager (a CDG service) ✓

**Verification:**
1. Recommendation ("should be applied first") matches implementation ✓
2. Rationale ("GoT depends on CDG services") is factually accurate ✓
3. Code behavior confirms both claims ✓

---

## Scope Verification

**Files Searched:**
- `/home/user/Opus-code-test/cortical/core/__init__.py`
- `/home/user/Opus-code-test/cortical/core/bootstrap.py`
- `/home/user/Opus-code-test/cortical/core/modules/schema_module.py`
- `/home/user/Opus-code-test/cortical/core/modules/cdg_module.py`
- `/home/user/Opus-code-test/cortical/core/modules/got_module.py`
- `/home/user/Opus-code-test/cortical/core/modules/__init__.py`

**Search Pattern:**
```regex
(FUTURE:|TODO:|FIXME:|PLANNED:|HACK:|XXX:|TEMPORARY:|WORKAROUND:|will be|should be|planned to)
```

**Total Lines Searched:** ~210 lines across 6 files

---

## What Went Wrong (Reflection on Misleading Finding)

**Finding 2 Analysis:**

The misleading comment in `cdg_module.py:50` reveals a common pattern:

**Anti-pattern:** Documenting intent instead of reality

```python
# WRONG: Speculation as documentation
got_dir: Legacy alias for base_dir (will be removed)

# RIGHT: Document what exists, link to plans if they exist
got_dir: Legacy alias for base_dir (deprecated, see T-XXXX for removal timeline)

# OR: If no plan exists, don't promise one
got_dir: Legacy alias for base_dir (deprecated, retained for compatibility)
```

**Why This Matters:**
1. Future readers don't know if "will be removed" means:
   - Next week?
   - Next year?
   - Never? (aspiration that died)
2. Without a task reference, the intent is untrackable
3. Code becomes cluttered with zombie "will be" comments that age into lies

**Prevention:**
- Use present tense for what exists NOW
- Use task references (T-XXXX) for what WILL happen
- Use "deprecated" or "legacy" without promises if no plan exists
- Comments describe reality, tasks describe plans

---

## Assessment Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| Accurate | 2 | 66.7% |
| Misleading | 1 | 33.3% |
| Stale | 0 | 0% |
| Unknown | 0 | 0% |
| **Total** | **3** | **100%** |

---

## Constraints Adherence

- ✅ Maximum findings: 3 / 50 (well within limit)
- ✅ Maximum duration: <5 minutes / 2 hours
- ✅ Scope: Remained in `cortical/core/` only
- ✅ Pre-flight check: All YES responses verified
- ✅ Categories: All findings assigned exactly one category with evidence
- ✅ Evidence: All claims cited with specific file paths and line numbers

---

## Recommendations

1. **Fix Finding 2:** Update `cdg_module.py:50` to remove future-tense speculation:
   ```python
   # Current (misleading):
   got_dir: Legacy alias for base_dir (will be removed)

   # Recommended:
   got_dir: Legacy alias for base_dir (deprecated, retained for compatibility)
   ```

2. **Policy:** Prohibit "will be" comments without task references
   - If there's a plan: Reference the task (T-XXXX)
   - If there's no plan: Don't promise one

3. **Audit Extension:** Consider auditing other directories for similar "will be" patterns

---

**END OF AUDIT**
