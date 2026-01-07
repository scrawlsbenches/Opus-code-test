# CDG Layer Migration - Working Scratchpad

*Session: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*

---

## SESSION OVERRIDES

- **DO NOT RUN FULL TEST SUITE** - some tests have broken imports
- **DO RUN `pytest tests/smoke/`** - 33 pass, 1 unrelated failure
- **DO RUN `got_utils.py validate`** - safe and helpful

---

## QUEUED

1. Test import errors (4 files)
2. test_field_validator.py missing registry
3. Smoke test using real .got/

---

## ARCHITECTURE

- **CDG owns:** storage, transactions, WAL, recovery, indexes
- **GoT is:** thin domain layer, NO file I/O
