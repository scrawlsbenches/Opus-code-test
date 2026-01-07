# CDG Layer Migration - Working Scratchpad

*Session: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*

---

## SESSION OVERRIDES

- **DO RUN `pytest tests/smoke/`** - 34 tests passing
- **DO RUN `got_utils.py validate`** - safe and helpful

---

## COMPLETED THIS SESSION

1. Fixed `use_memory=True` test isolation bug
   - Root cause: CDGTransactionManager created CDGStore without filesystem parameter
   - Fix: Added filesystem parameter, GoTModule now resolves from container

2. Extracted CDG durability tests from GoT test file
   - Moved TestParanoidMode, TestBalancedMode, TestRelaxedMode to tests/unit/cdg/test_cdg_durability.py
   - GoT tests now only test GoT config, not CDG internals

---

## REMAINING CONSIDERATIONS

1. **InMemoryFileSystem adoption** - Could speed up unit tests further
2. **3 failing durability tests** - Pre-existing, durability behavior not fully implemented
3. **CDG behavioral test gaps** - Schema validation, caching, checksum under failure

---

## ARCHITECTURE

- **CDG owns:** storage, transactions, WAL, recovery, indexes
- **GoT is:** thin domain layer, NO file I/O
- **FileSystem abstraction:** InMemoryFileSystem for tests, RealFileSystem for production
