# GoT API Cleanup - Remaining Work

**Date:** 2026-01-09
**Status:** In Progress
**Branch:** `claude/handoff-commands-pV8E0`

---

## Overview

This document tracks remaining cleanup work for the GoT API after the TransactionalGoTAdapter retirement and CDG Query Language implementation.

---

## Completed Work

- [x] Migrated from TransactionalGoTAdapter to GoTManager
- [x] Implemented CDG Query Language at `cortical/cdg/query/`
- [x] Added `list_entities(entity_type)` to GoTManager for schema-driven queries
- [x] Updated all graph/filter functions to accept `entity_type` parameter
- [x] Deleted stale test files (`test_query_language.py`, `test_tx_adapter.py`)

---

## Remaining Work

### 1. CLI Type Hint Updates (Low Effort)

**Problem:** 11 CLI files have TYPE_CHECKING imports referencing the deleted `TransactionalGoTAdapter`.

**Files affected:**
- `cortical/got/cli/analyze.py`
- `cortical/got/cli/backlog.py`
- `cortical/got/cli/backup.py`
- `cortical/got/cli/batch.py`
- `cortical/got/cli/decision.py`
- `cortical/got/cli/edge.py`
- `cortical/got/cli/failure.py`
- `cortical/got/cli/handoff.py`
- `cortical/got/cli/knowledge_transfer.py`
- `cortical/got/cli/orphan.py`
- `cortical/got/cli/sprint.py`
- `cortical/got/cli/task.py`

**Fix:** Find-and-replace across all files:
```python
# FROM:
from cortical.got.adapter import TransactionalGoTAdapter
def cmd_foo(args, manager: "TransactionalGoTAdapter") -> int:

# TO:
from cortical.got.api import GoTManager
def cmd_foo(args, manager: "GoTManager") -> int:
```

**Impact:** Type checking (mypy) fails without this fix. Runtime is unaffected.

---

### 2. protocol.py Docstring Update (Trivial)

**File:** `cortical/got/protocol.py`

**Problem:** The `GoTBackend` Protocol docstring references the deleted adapter:
```python
"""Both GoTProjectManager (event-sourced) and TransactionalGoTAdapter
(transactional) implement this protocol."""
```

**Fix:** Update docstring to:
```python
"""Protocol defining the GoT backend interface.

GoTManager implements this protocol, providing a consistent API for
all GoT operations.
"""
```

**Note:** The Protocol itself is valid and should be kept - it defines the interface contract.

---

### 3. Expression System Cleanup (Medium Effort)

**Directories:**
- `cortical/got/expression/` (OLD - 88KB total)
- `cortical/cdg/query/` (NEW - 83KB total)

**Analysis:**

| Component | got/expression | cdg/query | Status |
|-----------|----------------|-----------|--------|
| lexer.py | 9KB | 9KB | Superseded |
| parser.py | 16KB | 18KB | Superseded |
| executor.py | 19KB | 17KB | Superseded |
| ast.py | 2KB | 3KB | Superseded |
| errors.py | 6KB | 9KB | Superseded |
| registry.py | 2KB | 7KB | Superseded |
| optimizer.py | 14KB | - | See planner.py |
| planner.py | - | 14KB | Replacement |
| functions/graph.py | 32KB | - | See got.py |
| functions/filters.py | 14KB | - | See got.py |
| functions/got.py | - | 32KB | Replacement |

**Unique to got/expression (evaluate before deleting):**

| File | Size | Decision |
|------|------|----------|
| `validator.py` | 4KB | **MIGRATE** - `COMMON_FIELDS` used by CLI (`query.py:540`) |
| `translator.py` | 6KB | **EVALUATE** - NL→DSL translation, nice UX feature |
| `grammar.py` | 5KB | **DELETE** - Documentation only, no runtime use |
| `aggregate_functions.py` | 3KB | **CHECK** - Verify cdg/query has equivalents |

**Action Plan:**
1. Migrate `COMMON_FIELDS` from `validator.py` to `cdg/query/`
2. Evaluate if `translator.py` adds value (NL queries like "blocked tasks" → `blocked()`)
3. Delete superseded files after migration

---

### 4. Command Unification (Low Effort)

**Current state:**
- `got query` → Uses CDGQueryEngine ✅
- `got expr` → Uses CDGQueryEngine ✅
- `got infer` → Separate command (should be unified)

**Goal:** `got infer --commits 10` becomes `got query "infer(commits=10)"`

**Implementation:**
1. Register `infer` function in `cortical/cdg/query/functions/got.py`
2. Function should call existing `manager.infer_edges_from_recent_commits()`
3. Deprecate standalone `got infer` command

**Function signature:**
```python
@FunctionRegistry.register('infer')
class InferFunction(QueryFunction):
    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name='infer',
            description='Infer edges from git commit history',
            required_args=[],
            optional_args={'commits': 10, 'message': None},
            returns='List of inferred edges',
            category='git'
        )
```

---

## Priority Order

1. **CLI Type Hints** - Quick win, fixes type checking
2. **protocol.py** - Trivial docstring update
3. **Expression System** - Medium effort, biggest cleanup
4. **Command Unification** - Nice-to-have, improves consistency

---

## Files to Delete (After Migration)

```
cortical/got/expression/
├── __init__.py          # DELETE after migration
├── ast.py               # DELETE (superseded by cdg/query/ast.py)
├── errors.py            # DELETE (superseded by cdg/query/errors.py)
├── executor.py          # DELETE (superseded by cdg/query/executor.py)
├── grammar.py           # DELETE (documentation only)
├── lexer.py             # DELETE (superseded by cdg/query/lexer.py)
├── optimizer.py         # DELETE (superseded by cdg/query/planner.py)
├── parser.py            # DELETE (superseded by cdg/query/parser.py)
├── registry.py          # DELETE (superseded by cdg/query/registry.py)
├── translator.py        # EVALUATE then DELETE or MIGRATE
├── validator.py         # MIGRATE COMMON_FIELDS then DELETE
└── functions/
    ├── __init__.py      # DELETE
    ├── aggregate_functions.py  # CHECK then DELETE
    ├── filters.py       # DELETE (superseded by cdg/query/functions/got.py)
    └── graph.py         # DELETE (superseded by cdg/query/functions/got.py)
```

**Estimated deletion:** ~88KB of superseded code

---

## Success Criteria

- [ ] All CLI files use `GoTManager` type hints
- [ ] `protocol.py` docstring updated
- [ ] `COMMON_FIELDS` migrated to cdg/query
- [ ] `got/expression/` deleted (except migrated code)
- [ ] `infer` function registered in cdg/query
- [ ] All tests pass
- [ ] No imports of `cortical.got.adapter` remain in codebase

---

*This document will be updated as cleanup progresses.*
