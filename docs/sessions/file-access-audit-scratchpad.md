# CDG Layer Migration - Working Scratchpad

*Session: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*

---

## SESSION OVERRIDES

- **DO NOT RUN FULL TEST SUITE** - some tests have broken imports
- **DO RUN `pytest tests/smoke/`** - 33 pass, 1 unrelated failure
- **DO RUN `got_utils.py validate`** - safe and helpful

---

## CURRENT FOCUS: Task delete investigation

**Problem**: `got_utils.py task delete` doesn't remove file from disk

**Code path**:
- cli/task.py:408 → api.py:635 → transaction_manager.py:344 → storage.py:719
- storage.py:719 calls `self._fs.unlink(path)` - should delete

**Symptoms**: Deleted task file reappears in `.got/entities/`

**To investigate**:
- Is transaction committing?
- Is unlink actually called?
- Is something recreating the file?

---

## QUEUED

1. Test import errors (4 files)
2. test_field_validator.py missing registry
3. Smoke test using real .got/

---

## ARCHITECTURE

- **CDG owns:** storage, transactions, WAL, recovery, indexes
- **GoT is:** thin domain layer, NO file I/O
