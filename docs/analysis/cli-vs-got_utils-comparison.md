# CLI vs got_utils.py Cross-Reference Analysis

**Date:** 2026-01-08
**Purpose:** Determine if CLI modules have all functionality from got_utils.py inline versions

---

## Executive Summary

**Verdict: CLI modules are MORE COMPLETE than got_utils.py inline versions.**

The inline `cmd_*` functions in `scripts/got_utils.py` (lines 3166-3704) are:
- Incomplete stubs missing major features
- Dead code (never called - `main()` uses imported handlers)
- Safe to delete

---

## Comparison Tables

### Handoff (handoff.py vs got_utils.py:3386-3485)

| Function | CLI | Inline | Winner | Notes |
|----------|-----|--------|--------|-------|
| cmd_handoff_initiate | ✓ | ✓ | CLI | CLI has stdin support, truncation |
| cmd_handoff_accept | ✓ | ✓ | Equal | |
| cmd_handoff_complete | ✓ | ✓ | Equal | |
| cmd_handoff_reject | ✓ | ✗ | CLI | **Missing from inline** |
| cmd_handoff_show | ✓ | ✗ | CLI | **Missing from inline** |
| cmd_handoff_list | ✓ | ✓ | CLI | CLI has status alias, limit |

**CLI Issues:** Duplicate `cmd_handoff_reject` (lines 105-125 AND 238-253)

---

### Query (query.py vs got_utils.py:3551-3704)

| Function | CLI | Inline | Winner | Notes |
|----------|-----|--------|--------|-------|
| cmd_query | ✓ | ✓ | Equal | |
| cmd_validate | ✓ | ✓ | CLI | CLI has entity discovery, --check-refs |
| cmd_infer | ✓ | ✓ | Equal | |
| cmd_blocked | ✓ | ✗ | CLI | **Missing from inline** |
| cmd_active | ✓ | ✗ | CLI | **Missing from inline** |
| cmd_stats | ✓ | ✗ | CLI | **Missing from inline** |
| cmd_dashboard | ✓ | ✗ | CLI | **Missing from inline** |
| cmd_expr | ✓ | ✗ | CLI | **Missing from inline** (expression queries) |

**CLI Issues:** Duplicate orphan calculation code (lines 153-212 vs 214-224)

---

### Backup (backup.py vs got_utils.py:3166-3385)

| Function | CLI | Inline | Winner | Notes |
|----------|-----|--------|--------|-------|
| cmd_backup_create | ✓ | ✗ | CLI | **Missing from inline** |
| cmd_backup_list | ✓ | ✓ | Equal | |
| cmd_backup_verify | ✓ | ✓ | Equal | |
| cmd_backup_restore | ✓ | ✓ | Equal | |
| cmd_sync | ✓ | ✗ | CLI | **Missing from inline** |

---

### Decision (decision.py vs got_utils.py:3487-3549)

| Function | CLI | Inline | Winner | Notes |
|----------|-----|--------|--------|-------|
| cmd_decision_log | ✓ | ✓ | CLI | CLI has task linkage prompts |
| cmd_decision_list | ✓ | ✓ | CLI | CLI checks multiple methods |
| cmd_decision_why | ✓ | ✓ | Equal | |
| cmd_decision_show | ✓ | ✗ | CLI | **Missing from inline** |
| cmd_decision_trace | ✓ | ✗ | CLI | **Missing from inline** |
| cmd_decision_delete | ✓ | ✗ | CLI | **Missing from inline** |

---

## Recommended Actions

### 1. Fix CLI Bugs (Priority: High)
- [ ] `handoff.py:238-253` - Remove duplicate `cmd_handoff_reject`
- [ ] `query.py:214-224` - Remove duplicate orphan calculation

### 2. Delete Dead Code in got_utils.py (Priority: Medium)
Lines 3166-3704 contain ~540 lines of dead inline functions that are:
- Never called (main() uses imported handlers)
- Less complete than CLI versions

### 3. Keep CLI as Single Source of Truth
The modular CLI in `cortical/got/cli/` should be the authoritative implementation.

---

## Line Counts

| File | Lines | Notes |
|------|-------|-------|
| got_utils.py | 3902 | ~540 lines dead code |
| cortical/got/cli/handoff.py | 374 | Has bugs |
| cortical/got/cli/query.py | 757 | Has bugs |
| cortical/got/cli/backup.py | ~400 | Clean |
| cortical/got/cli/decision.py | ~500 | Clean |
