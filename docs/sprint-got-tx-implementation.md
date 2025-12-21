# Sprint: GoT Transactional Architecture Implementation

**Sprint ID:** SPRINT-GOT-TX-2025-W52
**Start Date:** 2025-12-21
**Goal:** Implement ACID-compliant transaction layer for Graph of Thought

---

## Overview

This sprint implements the design in `docs/got-transactional-architecture.md`.
All progress is tracked in this file (not in GoT, since we're replacing it).

---

## Phase 1: Storage Layer

### Tests First (TDD)
- [ ] `tests/unit/got/test_checksums.py` - Checksum verification
- [ ] `tests/unit/got/test_versioned_store.py` - VersionedStore operations
  - [ ] test_write_creates_file_with_checksum
  - [ ] test_read_verifies_checksum
  - [ ] test_corrupted_checksum_raises_error
  - [ ] test_atomic_write_survives_crash
  - [ ] test_version_increments_on_write
  - [ ] test_read_at_version_returns_historical
  - [ ] test_fsync_called_on_write

### Implementation
- [ ] `cortical/got/errors.py` - Exception classes
- [ ] `cortical/got/checksums.py` - Checksum utilities
- [ ] `cortical/got/types.py` - Entity base class with to_json/from_json
- [ ] `cortical/got/versioned_store.py` - VersionedStore

---

## Phase 2: WAL Layer

### Tests First (TDD)
- [ ] `tests/unit/got/test_wal.py` - WAL operations
  - [ ] test_log_appends_with_checksum
  - [ ] test_fsync_called_on_every_log
  - [ ] test_corrupted_entry_detected
  - [ ] test_incomplete_tx_detected
  - [ ] test_truncate_archives_old_wal
  - [ ] test_recovery_finds_incomplete_transactions

### Implementation
- [ ] `cortical/got/wal.py` - WALManager

---

## Phase 3: Transaction Layer

### Tests First (TDD)
- [ ] `tests/unit/got/test_transaction.py` - Transaction object
  - [ ] test_transaction_state_machine
  - [ ] test_write_buffered_until_commit
  - [ ] test_read_sees_own_writes
  - [ ] test_read_sees_snapshot_version

- [ ] `tests/unit/got/test_tx_manager.py` - TransactionManager
  - [ ] test_begin_creates_transaction
  - [ ] test_commit_applies_writes
  - [ ] test_rollback_discards_writes
  - [ ] test_conflict_detected_on_version_mismatch
  - [ ] test_crash_recovery_rolls_back_incomplete
  - [ ] test_lock_acquired_during_commit

### Implementation
- [ ] `cortical/got/transaction.py` - Transaction, TransactionState
- [ ] `cortical/got/tx_manager.py` - TransactionManager

---

## Phase 4: High-Level API

### Tests First (TDD)
- [ ] `tests/unit/got/test_api.py` - High-level API
  - [ ] test_context_manager_commits_on_success
  - [ ] test_context_manager_rolls_back_on_exception
  - [ ] test_create_task_in_transaction
  - [ ] test_update_task_in_transaction
  - [ ] test_read_only_context

### Implementation
- [ ] `cortical/got/__init__.py` - Public API exports
- [ ] `cortical/got/api.py` - GoTTransactionalManager, TransactionContext

---

## Phase 5: Sync Layer

### Tests First (TDD)
- [ ] `tests/unit/got/test_sync.py` - Sync operations
  - [ ] test_push_fails_with_active_transaction
  - [ ] test_pull_fails_with_active_transaction
  - [ ] test_push_rejected_returns_pull_first
  - [ ] test_conflict_detected_on_version_mismatch

- [ ] `tests/unit/got/test_conflict.py` - Conflict resolution
  - [ ] test_ours_strategy_keeps_local
  - [ ] test_theirs_strategy_takes_remote
  - [ ] test_merge_strategy_combines_non_overlapping
  - [ ] test_merge_fails_on_same_field_conflict

### Implementation
- [ ] `cortical/got/sync.py` - SyncManager
- [ ] `cortical/got/conflict.py` - ConflictResolver

---

## Phase 6: Recovery & Edge Cases

### Tests First (TDD)
- [ ] `tests/unit/got/test_recovery.py` - Recovery procedures
  - [ ] test_startup_recovery_rolls_back_incomplete
  - [ ] test_corrupted_entity_detected
  - [ ] test_corrupted_wal_detected
  - [ ] test_recovery_from_git_history

- [ ] `tests/unit/got/test_edge_cases.py` - Edge cases from design
  - [ ] test_power_loss_during_wal_write
  - [ ] test_power_loss_during_commit
  - [ ] test_stale_lock_recovered
  - [ ] test_concurrent_creates_different_ids
  - [ ] test_large_transaction_memory_limit

### Implementation
- [ ] `cortical/got/recovery.py` - Recovery procedures

---

## Phase 7: Migration

- [ ] Migrate existing .got/ data to new format
- [ ] Verification tests that old data preserved
- [ ] Update CLAUDE.md with new GoT commands

---

## Progress Log

| Date | What was done |
|------|---------------|
| 2025-12-21 | Design document created and approved |
| | |

---

## Notes

- All tests written BEFORE implementation (TDD)
- No file exceeds 250 lines
- All storage is JSON text (no binary)
- Run `python -m pytest tests/unit/got/ -v` to verify progress
