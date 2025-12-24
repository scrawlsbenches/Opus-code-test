# Knowledge Transfer: Migration Gap Audit (2025-12-24)

**Session ID:** claude/got-handoff-command-0tOum
**Key Finding:** Previous agents' claims must be verified empirically before acting on them

---

## Executive Summary

This session discovered that the claim "TX backend replaces all functionality" was **FALSE**. The `infer` CLI command was broken in production because `infer_edges_from_commit()` was never migrated to `TransactionalGoTAdapter`.

---

## What Happened

### The Chain of Trust Failure

```
1. Decision D-20251222-144524-2dedd20f claimed:
   "TX backend replaces all functionality"
   
2. Task T-20251222-145525-445df343 was created:
   "Remove deprecated EventLog and GoTProjectManager (~2200 lines)"
   
3. GoTBackendFactory was changed to ONLY return TransactionalGoTAdapter
   
4. BUT: infer_edges_from_commit() was never migrated
   
5. RESULT: `python scripts/got_utils.py infer --message "test"` threw:
   AttributeError: 'TransactionalGoTAdapter' object has no attribute 'infer_edges_from_commit'
```

### How We Found It

1. User asked us to audit before proceeding with "Option B" (continue refactoring)
2. We questioned whether deletion was safe
3. Deep analysis revealed the gap
4. **We tested the command** - it failed
5. This proved the previous agents were wrong

---

## What We Fixed

### Code Change
- **File:** `scripts/got_utils.py`
- **Commit:** `4dcaa50a`
- **Change:** Added `infer_edges_from_commit()` and `infer_edges_from_recent_commits()` to `TransactionalGoTAdapter`

### GoT Artifacts Created
- **Decision:** `D-20251224-052658-40924db3` - Documents the migration gap
- **Task:** `T-20251224-052817-827bc246` - Tracked the fix (completed)
- **Blocked:** `T-20251222-145525-445df343` - Prevented premature deletion

---

## Lessons Learned

### 1. Trust But Verify
Previous agents' decisions and tasks are **input**, not **truth**. They can be:
- Based on incomplete analysis
- Made under time pressure
- Missing edge cases they didn't test

### 2. Test Before Trusting Claims About Equivalence
When someone claims "X replaces Y", verify by:
- Listing all methods in Y
- Confirming each exists in X
- **Actually running the commands**

### 3. Destructive Tasks Need Extra Scrutiny
Tasks that say "delete", "remove", or "deprecate" should be treated as high-risk:
- Check for dependencies
- Verify the replacement is complete
- Test all affected functionality

### 4. The Audit Pattern
When inheriting work from previous agents:
```
1. Read the task/decision
2. Identify the claims being made
3. Find evidence to verify/refute
4. Test empirically if possible
5. Only proceed if verified
```

---

## Commands That Work (Verified 2025-12-24)

| Command | Status | Notes |
|---------|--------|-------|
| `got_utils.py infer` | ✅ Fixed | Was broken, now works |
| `got_utils.py compact` | ✅ Works | Tested with --dry-run |
| `got_utils.py validate` | ✅ Works | Shows stats correctly |
| `got_utils.py handoff list` | ✅ Works | Lists handoffs |
| `got_utils.py export` | ✅ Works | Exports JSON |
| `got_utils.py stats` | ✅ Works | Shows statistics |
| `got_utils.py dashboard` | ✅ Works | Full dashboard |

---

## Remaining Work

Task `T-20251222-145525-445df343` (Remove deprecated code) is now **BLOCKED** pending:
1. Full audit of ALL methods in GoTProjectManager vs TransactionalGoTAdapter
2. Integration tests for every CLI command
3. Verification no other gaps exist

---

## Key Quotes

> "Assume nothing until we've confirmed it ourselves." - Session principle

> "Previous agents' decisions must be verified empirically, not trusted blindly." - Decision D-20251224-052658-40924db3

---

## Files Modified This Session

- `scripts/got_utils.py` - Added missing methods (+117 lines)
- `.got/entities/decisions/` - New decision logged
- `.got/entities/tasks/` - New task created and completed
- `.git-ml/` - 2,526 files untracked (separate fix from handoff)

---

**Next Session Should:**
1. Review this document
2. Consider whether CLAUDE.md needs audit guidance
3. Continue with verified, tested changes only
