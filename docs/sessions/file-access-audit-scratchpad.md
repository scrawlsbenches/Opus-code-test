# CDG Layer Migration - Working Scratchpad

*Session: 2026-01-07*
*Branch: claude/refactor-cortical-codebase-OZ8em*

---

## SESSION OVERRIDES

- **DO NOT RUN FULL TEST SUITE** - some tests have broken imports
- **DO RUN `pytest tests/smoke/`** - smoke tests pass (1 unrelated failure)
- **DO RUN `got_utils.py validate`** - safe and helpful

---

## COMPLETED THIS SESSION

1. **Fixed `expr` command bug** - commit dc89109b
   - Added SchemaRegistry resolution from container in `cortical/got/expression/__init__.py`
   - `python scripts/got_utils.py expr "status = 'completed'"` now works

---

## NEXT: Task delete investigation

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

## QUEUED ISSUES

1. **Test import errors** (4 files) - from refactoring
   - test_config.py, test_query_builder_entities.py, test_schema.py, test_generate_book.py

2. **test_field_validator.py** - all tests missing registry argument (related to expr fix)

3. **Smoke test using real .got/** - test_query_tasks expects 2 tasks, gets 405

---

## ARCHITECTURE

- **CDG owns:** storage, transactions, WAL, recovery, indexes
- **GoT is:** thin domain layer, query/facade, NO file I/O for infrastructure
