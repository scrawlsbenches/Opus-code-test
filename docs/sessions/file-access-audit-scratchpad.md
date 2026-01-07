# CDG Layer Migration - Working Scratchpad

*Session: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*
*Previous branch: claude/refactor-cortical-codebase-8ij55*

---

## ⛔ SESSION OVERRIDES

- **DO NOT RUN FULL TEST SUITE** - some tests have broken imports from refactoring
- **DO RUN `got_utils.py validate`** - safe and helpful
- **DO RUN `pytest tests/smoke/`** - smoke tests pass

---

## REFACTORING RULES

1. **NO DEPRECATION** - fix now or note for later, no deprecation comments
2. **NO BACKWARD COMPATIBILITY** - breaking changes are intentional
3. **WE CONTROL THE DATA** - no defensive normalization needed
4. **GENERIC COMMENTS** - no external product references

---

## COMPLETED PHASES

- **Phase 1:** Deleted `got/wal.py`, `got/tx_manager.py`, `got/versioned_store.py`
- **Phase 2:** Deleted `got/recovery.py`, created `common/recovery_types.py`
- **Phase 3:** Deleted `got/indexer.py`, CDGIndexManager now owns indexing

---

## ACTIVE TASKS

### 1. FIX TEST IMPORTS [BLOCKING]

4 test files have broken imports:
- `tests/unit/got/test_config.py` - unknown import error
- `tests/unit/test_query_builder_entities.py` - `ensure_schemas_registered` doesn't exist
- `tests/unit/test_schema.py` - imports from `cortical.got.schema` (moved to `cdg/schema`)
- `tests/unit/test_generate_book.py` - missing `yaml` module (low priority)

### 2. VALIDATION RULE EXTRACTION [TECHNICAL DEBT]

9 if/elif blocks in `_validate_entity_specific()` should use schema validators.

### 3. GRAPH TRAVERSAL CONSOLIDATION [DEFERRED]

3 overlapping utilities. Lower priority.

---

## ARCHITECTURE RULES

**CDG owns:**
- Storage, transactions, WAL, recovery, indexes

**GoT is thin domain layer:**
- Query/facade over CDG
- Entity types and business logic
- NO file I/O for infrastructure

**Right pattern:**
```python
Field("status", FieldType.STRING, indexed=True)  # Schema declares index
# CDGIndexManager updates automatically on write
```

---

## KEY INSIGHTS

- **Durability:** Configured in CDG, not GoT
- **got/expression/*:** Half-baked but good ideas, needs general schema support
- **got/cli/*:** Container as member variable, don't import bootstrap in functions

---

## DESIGN DECISIONS

1. Container-first (DI for all dependencies)
2. CDG is foundation, GoT is thin domain layer
3. Move GoT functionality to CDG when possible
