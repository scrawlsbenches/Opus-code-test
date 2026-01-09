# TransactionalGoTAdapter Retirement Progress

**Plan:** `docs/design/transactional-adapter-retirement-plan.md`
**Branch:** claude/handoff-commands-pV8E0

---

## Phase Checklist

- [ ] Phase 1: Move status transitions (start_task, complete_task, block_task)
- [ ] Phase 2: Move query helpers (get_active_tasks, get_blocked_tasks, what_blocks, etc.)
- [ ] Phase 3: Move sprint helpers (claim_sprint, goals, link/unlink)
- [ ] Phase 4: Move KT/Handoff methods
- [ ] Phase 5: Move introspection (get_stats, validate, graph/nodes/edges)
- [ ] Phase 6: Create git_inference.py module
- [ ] Phase 7: Handle query() method
- [ ] Phase 8: Update 27 consumer files
- [ ] Phase 9: Delete adapter.py
- [ ] Phase 10: Cleanup

---

## Session Log

### 2026-01-09 - Plan Created
**Agent:** claude/handoff-commands-pV8E0
**Status:** Plan created, ready to begin Phase 1
**Files created:**
- `docs/design/transactional-adapter-retirement-plan.md`
- `docs/sessions/adapter-retirement-progress.md` (this file)

### 2026-01-09 - TODO Comments Added
**Agent:** claude/handoff-commands-pV8E0
**Status:** Added evaluation TODO comments to adapter.py
**Files modified:**
- `cortical/got/adapter.py` - Added TODO comments to all sections

**Summary of method disposition:**
- **PURE DELEGATION (remove):** ~25 methods that just call manager
- **MOVE TO GoTManager:** ~15 methods with actual logic
- **MOVE TO git_inference.py:** 3 git-related functions
- **DELETE (incomplete):** query() method

**Next step:** Review annotations with user, then begin Phase 1

---

<!-- Add new session updates above this line -->
