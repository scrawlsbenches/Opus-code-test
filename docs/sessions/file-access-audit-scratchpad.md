# CDG Layer Migration - Working Scratchpad

*Session: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*

---

## SESSION OVERRIDES

- **DO NOT RUN FULL TEST SUITE** - some tests have broken imports
- **DO RUN `pytest tests/smoke/`** - 33 pass, 1 unrelated failure
- **DO RUN `got_utils.py validate`** - safe and helpful

---

## CURRENT FOCUS: Fix test import errors

4 test files have broken imports from refactoring:
- `tests/unit/got/test_config.py` - unknown error
- `tests/unit/test_query_builder_entities.py` - `ensure_schemas_registered` missing
- `tests/unit/test_schema.py` - imports from `cortical.got.schema` (moved to `cdg/schema`)
- `tests/unit/test_generate_book.py` - missing `yaml` module (low priority)

---

## QUEUED

1. test_field_validator.py - all tests missing registry argument
2. Smoke test using real .got/ instead of temp dir

---

## ARCHITECTURE

- **CDG owns:** storage, transactions, WAL, recovery, indexes
- **GoT is:** thin domain layer, NO file I/O
