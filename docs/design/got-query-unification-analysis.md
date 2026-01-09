# GoT Query System Unification Analysis

**Date:** 2026-01-09
**Status:** Draft for evaluation
**Author:** Claude (session claude/handoff-commands-pV8E0)

---

## Background

The GoT query system has two components that were built separately due to a communication breakdown:

1. **Expression Query System** (`cortical/got/expression/`) - SQL-like queries
2. **Infer System** (`cmd_infer`) - Git commit analysis

Both should have been built on unified infrastructure, but diverged. This document captures the current state and options for moving forward.

---

## Current State (What Works)

### Expression System
- Parses SQL-like queries: `status = 'pending' AND priority = 'high'`
- Functions registered via `FunctionRegistry`
- Returns `Any` - duck typing handles display
- **Works for current needs**

### Display Layer
- Uses `hasattr(item, '__dict__')` to detect primitives vs entities
- Uses `callable()` to skip method attributes
- Handles strings, bools, dicts, and entity objects
- **Works after recent fix (665a3ec4)**

### Existing Infrastructure (Underutilized)
- `cortical/results.py` has `QueryResult`, `DocumentMatch`, `PassageMatch` - designed for text processor
- `cortical/cdg/schema` has `FieldType` enum, `SchemaRegistry` with type metadata
- `SchemaRegistry.get_entity_type_by_prefix()` - exists but `type_of` function hard-codes prefixes

---

## The Gap

### Two Type Systems That Don't Communicate

| System | Location | Purpose | Machine-Readable? |
|--------|----------|---------|-------------------|
| Python annotations | `execute() -> Optional[str]` | Static typing | Yes (via `inspect`) |
| Signature.returns | `"Boolean: True if..."` | Documentation | No (free-form string) |

Neither is used at runtime to inform the display layer.

### Parallel Infrastructure

```
Text Processor Path:
  cortical/results.py → QueryResult → typed matches

GoT Query Path:
  cortical/got/expression/ → Any → duck typing
```

These should have shared a common result abstraction.

---

## Options

### Option A: Leave As-Is (Recommended for Now)

**Rationale:**
- Current system works for our needs
- Duck typing in display layer handles heterogeneous results
- No breaking changes

**When to revisit:**
- If we add more scalar-returning functions
- If we need result type introspection for other consumers
- If the expression system expands significantly

### Option B: Minimal Bridge (Low Risk)

Add optional `result_category` to `FunctionSignature`:

```python
class ResultCategory(Enum):
    ENTITY_LIST = "entity_list"
    SCALAR = "scalar"
    MAPPING = "mapping"

@dataclass
class FunctionSignature:
    # ... existing fields ...
    result_category: Optional[ResultCategory] = None  # Optional, backward compatible
```

**Pros:** Non-breaking, gradual adoption
**Cons:** Another field to maintain, may never be used

### Option C: Full Unification (Future)

Adopt `QueryResult` pattern from `cortical/results.py` for GoT queries:

```python
@dataclass
class GoTQueryResult:
    query: str
    result_type: ResultCategory
    data: Any
    timing_ms: Optional[float] = None
```

**Pros:** Consistent with text processor, rich metadata
**Cons:** Breaking change, significant refactor

---

## Immediate Fixes (Non-Breaking)

These can be done now without architectural changes:

### 1. type_of Function - Use SchemaRegistry

**Current** (hard-coded in `graph.py:918-924`):
```python
prefix_map = {
    'T-': 'task',
    'S-': 'sprint',
    # ... hard-coded ...
}
```

**Should be:**
```python
from cortical.cdg.schema import SchemaRegistry
registry = container.resolve(SchemaRegistry)
entity_type = registry.get_entity_type_by_prefix(prefix)
```

**Risk:** Low - internal implementation change
**Benefit:** Single source of truth for entity types

### 2. Document the Duck Typing Contract

Add docstring to `_print_result_item` explaining the expected contract:
- Objects with `__dict__` are treated as entities
- Objects without `__dict__` are printed directly
- Callable attributes are skipped

---

## Decision Needed

1. **Do we proceed with Option A (leave as-is) for the architecture?**
2. **Do we fix `type_of` to use SchemaRegistry?**
3. **Do we have bandwidth to address behavioral test issues from earlier audit?**

---

## Related Work (Parked)

From earlier in this session, we identified 25 behavioral tests with issues across 4 categories:
- T-20260109-135135-*: time.sleep violations (12 files)
- T-20260109-135142-*: always-passing assertions (8 files)
- T-20260109-135148-*: timing-based flaky assertions (4 files)
- T-20260109-135154-*: weak/incorrect assertions (3 files)

These are tracked in GoT but not yet addressed.

---

## Summary

The query system works. The divergence is technical debt, not a bug. I recommend:

1. **Now:** Fix `type_of` to use SchemaRegistry (5 min, no risk)
2. **Now:** Document the duck typing contract
3. **Later:** Revisit unification if/when the expression system expands
4. **Parallel:** Address behavioral test issues per existing tasks

What are your thoughts on this prioritization?
