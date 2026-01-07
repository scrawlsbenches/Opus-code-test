# CDG Layer Migration - Working Scratchpad

*Session: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*

---

## SESSION OVERRIDES

- **DO NOT RUN FULL TEST SUITE** - some tests have broken imports
- **DO RUN `pytest tests/smoke/`** - smoke tests pass
- **DO RUN `got_utils.py validate`** - safe and helpful

---

## ACTIVE TASKS

### 1. FIX TEST IMPORTS [BLOCKING]

4 test files have broken imports:
- `tests/unit/got/test_config.py`
- `tests/unit/test_query_builder_entities.py` - `ensure_schemas_registered` doesn't exist
- `tests/unit/test_schema.py` - imports from `cortical.got.schema` (moved to `cdg/schema`)
- `tests/unit/test_generate_book.py` - missing `yaml` module (low priority)

### 2. VALIDATION RULE EXTRACTION [TECHNICAL DEBT]

### 3. GRAPH TRAVERSAL CONSOLIDATION [DEFERRED]

---

## REFACTORING RULES

1. No deprecation - fix now or note for later
2. No backward compatibility - breaking changes intentional
3. No defensive normalization - we control the data
4. Generic comments - no external product references

---

## ARCHITECTURE

- **CDG owns:** storage, transactions, WAL, recovery, indexes
- **GoT is:** thin domain layer, query/facade, NO file I/O for infrastructure
