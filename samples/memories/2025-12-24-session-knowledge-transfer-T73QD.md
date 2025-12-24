# Knowledge Transfer: Session T73QD (2025-12-24)

**Session ID:** T73QD  
**Date:** 2025-12-24  
**Branch:** `claude/engineering-session-T73QD`  
**Agent:** Opus 4.5

---

## Executive Summary

This session performed a comprehensive repository audit following a "catastrophic event" and fixed multiple systemic issues. Key accomplishments:

1. **Orphan Rate: 71.6% → 8.3%** - Auto-sprint-linking + backlog sprint for remaining orphans
2. **ML Data Preservation** - Fixed CALI gitignore to preserve irreplaceable data
3. **Handoff System Fixes** - Added validation, auto-close, and reject CLI command
4. **Hook Improvements** - Session start/end now show sprint context and handoff prompts
5. **Test Fixes** - Fixed 6 failing orphan tests caused by sub-agent modifications
6. **Validation Bug Fix** - Corrected orphan counting to use set difference (was inflating connected count)

---

## Problems Found & Fixed

### 1. Orphan Task Problem (71.6% → 4.8%)

**Root Cause:** Tasks created via `got task create` were not automatically linked to sprints.

**Analysis:**
- 74 total tasks, 53 were orphans
- Most orphans created on 2025-12-22 (36 tasks)
- Edges existed but connected to decisions/documents, not sprints
- Sprints existed but `task_ids` property was empty

**Solutions Implemented:**
1. **Auto-link at creation** (`cortical/got/cli/task.py`)
   - New tasks automatically linked to current sprint
   - Added `--no-sprint` flag to opt out
   - Uses `get_current_sprint()` method

2. **Sprint context in hooks** (`scripts/ml-session-start-hook.sh`, `scripts/ml-session-capture-hook.sh`)
   - Session start shows current sprint and progress
   - Session end shows sprint status summary

3. **Bulk orphan linking**
   - Linked 37 existing orphan tasks to S-018
   - Sprint S-018 now at 54/55 tasks (98.2% complete)

**Commit:** `0ce54272` - feat(got): Auto-link tasks to sprints

---

### 2. ML Data Loss (CALI Gitignore)

**Root Cause:** `.gitignore` had `.git-ml/cali/` which ignored ALL CALI data, including irreplaceable logs and objects.

**Analysis:**
```
.git-ml/cali/
├── local/     # 35KB - Regeneratable indices (SHOULD be ignored)
├── logs/      # 21KB - NOT regeneratable (WAS being lost!)
├── objects/   # 150KB - NOT regeneratable (WAS being lost!)
└── manifest.json
```

**Solution:**
Changed `.gitignore`:
```diff
- .git-ml/cali/
+ .git-ml/cali/local/
+ !.git-ml/cali/logs/
+ !.git-ml/cali/objects/
```

**Result:** 171KB of irreplaceable ML data now preserved. 53 CALI files now tracked.

**Commit:** `4f7c21a8` - fix(got,ml): Parallel fixes for handoff validation and CALI data preservation

---

### 3. Handoff System Bugs

**Problems Found:**
1. No validation that task exists when handoff created via non-CLI paths
2. Handoffs not auto-closed when associated task completes
3. No orphan detection for stale handoffs
4. No `reject` CLI command

**Evidence:**
- `H-20251224-092345-a404eaeb` - Referenced non-existent task
- `H-20251224-112407-f49a64c1` - Task completed but handoff still open

**Solutions:**
1. Added task validation in `TransactionContext.initiate_handoff()` (api.py:1907-1910)
2. Added `_auto_close_handoffs()` when task completes (api.py:1584-1656)
3. Added `find_orphan_handoffs()` to `OrphanDetector` (orphan.py:177-207)
4. Added `got handoff reject` CLI command (handoff.py:125-140, 182-186)

**Commits:** 
- `59f8443a` - feat(got): Add handoff reject CLI command and audit fixes
- `4f7c21a8` - fix(got,ml): Parallel fixes for handoff validation and CALI data preservation

---

### 4. _version.json Merge Conflicts

**Root Cause:** Task `T-20251223-005914-f298d276` was marked complete but never actually implemented.

**Evidence:** The `.gitignore` entry was only a COMMENT:
```
#   - .got/entities/_version.json - Version counter (self-healing, see .gitattributes)
```

**Solution:**
1. Added actual `.gitignore` entry: `.got/entities/_version.json`
2. Removed from tracking: `git rm --cached .got/entities/_version.json`
3. File remains locally (self-healing on load)

**Decision logged:** `D-20251224-115112-4272c7d2`
**Commit:** `9ccb852b` - fix(got): Actually remove _version.json from git tracking

---

### 5. Session Hooks Improvements

**Added to Session Start Hook:**
- Session type detection (CONTINUATION/RESUMPTION/FRESH)
- Pending handoff detection with acceptance prompts
- Sprint context display
- Task details for pending handoffs

**Added to Session End Hook:**
- Session summary with recent commits
- Modified files list
- Sprint status
- In-progress tasks with handoff creation instructions

**Commit:** `2cd3a1f5` - feat(hooks): Improve session hooks with handoff detection and prompts

---

### 6. Test Fixes

**Problem:** Sub-agent modified `OrphanDetector` to add `find_orphan_handoffs()` but didn't update test mocks.

**Error:** `TypeError: 'Mock' object is not iterable` in 6 tests

**Solution:** Added `manager.list_handoffs.return_value = []` to three fixtures:
- `TestOrphanDetector.mock_manager`
- `TestOrphanCLI.mock_got_manager`
- `TestConvenienceFunctions.test_generate_orphan_report_convenience`

**Commit:** `e249056d` - fix(tests): Add list_handoffs mock to orphan tests

---

### 7. Validation Orphan Counting Bug (Continuation Session)

**Problem:** `got validate` reported 4 orphans (4.8%) when actual count was 23 (27.4%).

**Root Cause:** The orphan calculation used subtraction (`total_nodes - len(nodes_with_edges)`) but `nodes_with_edges` contained references to non-existent nodes (old sprint IDs like `S-sprint-001`, `S-sprint-017-spark-slm`). This inflated the "connected" count.

**Fix:** Changed calculation in both `cortical/got/cli/query.py` and `scripts/got_utils.py`:
```python
# Before (wrong)
orphan_count = total_nodes - len(nodes_with_edges)

# After (correct)
all_node_ids = set(manager.graph.nodes.keys())
nodes_with_edges = set()
for edge in manager.graph.edges:
    if edge.source_id in all_node_ids:
        nodes_with_edges.add(edge.source_id)
    if edge.target_id in all_node_ids:
        nodes_with_edges.add(edge.target_id)
orphan_count = len(all_node_ids - nodes_with_edges)
```

**Commit:** `e2e58a6c` - fix(got): Correct orphan count calculation in validate command

---

### 8. Backlog Sprint for Orphan Tasks (Continuation Session)

**Problem:** 16 orphan tasks existed with no sprint assignment.

**Solution:** Created `S-000 "Backlog"` sprint and linked all 16 orphan tasks to it.

**Result:** Orphan rate dropped from 27.4% to 8.3%. Remaining 7 orphans are decisions (acceptable).

**Commit:** `5066d806` - chore(got): Add backlog sprint and link orphan tasks

---

## Key Decisions Made

| Decision ID | Decision | Rationale |
|-------------|----------|-----------|
| `D-20251224-115112-4272c7d2` | Remove _version.json from git | Previous fix incomplete, file caused merge conflicts |
| `D-20251224-052658-40924db3` | (Prior session) TX backend verification | Verified infer command was broken before deletion |

---

## Current State

### Code Coverage
- **87%** (down from 89% baseline due to new code)
- All tests passing (**8864 passed**)

### GoT Health
- Tasks: 74
- Orphan rate: **8.3%** (was 71.6% → 27.4% → 8.3%)
- Edges: 94 (was 42 → 78 → 94)
- Sprints: S-018 (active), S-000 (backlog with 16 tasks)

### Remaining Orphans (7 nodes)
All 7 are **decisions** without explicit task links. This is acceptable - decisions can exist standalone or be linked later. All 16 orphan tasks are now in backlog S-000.

---

## Files Modified (Key Changes)

| File | Change |
|------|--------|
| `cortical/got/api.py` | Added handoff validation and auto-close logic |
| `cortical/got/orphan.py` | Added find_orphan_handoffs(), updated OrphanReport |
| `cortical/got/cli/handoff.py` | Added reject command |
| `cortical/got/cli/task.py` | Added auto-sprint-linking at creation |
| `cortical/got/cli/query.py` | **Fixed orphan counting bug** - use set difference |
| `scripts/got_utils.py` | **Fixed orphan counting bug** (same fix as query.py) |
| `scripts/ml-session-start-hook.sh` | Session type detection, sprint context, handoff prompts |
| `scripts/ml-session-capture-hook.sh` | Session summary, sprint status, handoff prompts |
| `.gitignore` | Fixed CALI and _version.json patterns |
| `CLAUDE.md` | Updated ML data documentation |
| `tests/unit/test_orphan.py` | Fixed mock fixtures |

---

## Parallel Sub-Agent Usage

This session successfully used parallel sub-agents for:

1. **Handoff validation fixes** - Modified api.py and orphan.py
2. **CALI data preservation** - Modified .gitignore and CLAUDE.md
3. **Auto-sprint-linking** - Modified task.py CLI
4. **Sprint context in hooks** - Modified both hook scripts
5. **Orphan bulk linking** - Linked 37 tasks to S-018

**Key insight:** Sub-agents work well for mechanical tasks but may miss test updates. Always verify changes after sub-agent completion.

---

## Pending Tasks (For Next Session)

### High Priority
- None - all critical issues fixed

### Medium Priority (16 orphan design tasks)
| Category | Tasks |
|----------|-------|
| Testing/Quality | 6 tasks (test refactoring, paradigm standardization) |
| Design/Architecture | 5 tasks (layer system, versioning, context detection) |
| ML/Data Collection | 3 tasks (SparkSLM training, full ML collection, GoT integration) |
| Validation | 1 task (fault tolerance and recovery) |

**Recommendation:** Create focused sprints:
- Sprint 023: "Quality & Testing Improvements"
- Sprint 024: "Architecture & Design"
- Link ML tasks to Sprint S-017 "SparkSLM"

---

## Commands Reference

```bash
# Create task (auto-links to current sprint)
python scripts/got_utils.py task create "Title" --priority high

# Opt out of auto-sprint-linking
python scripts/got_utils.py task create "Title" --no-sprint

# Reject a handoff
python scripts/got_utils.py handoff reject HANDOFF_ID --agent NAME --reason "..."

# Check orphan rate
python scripts/got_utils.py validate | grep -i orphan

# Sprint status
python scripts/got_utils.py sprint status
```

---

## Lessons Learned

1. **Verify "completed" tasks** - Task T-20251223-005914-f298d276 was marked complete but never implemented
2. **Sub-agents need test verification** - They may modify code without updating all tests
3. **Parallel sub-agents are effective** - 3 agents completed faster than sequential work
4. **ML collection hooks now work** - CALI data is being preserved (53 files tracked)
5. **Auto-linking prevents orphans** - Future tasks will automatically link to sprints
6. **Validate command can have bugs** - Always verify metrics with direct analysis when something seems off
7. **Duplicate code paths exist** - The validate function existed in both `query.py` and `got_utils.py` - both needed fixing
8. **Backlog sprint is useful** - S-000 "Backlog" provides a place for unassigned tasks without leaving them orphaned

---

## Verification Checklist (For Next Session)

Before proceeding with new work, verify these are working:

```bash
# 1. Tests pass
python3 -m pytest tests/ -x --tb=short -q

# 2. Orphan rate is reasonable (<10%)
python3 scripts/got_utils.py validate | grep -i orphan

# 3. Backlog sprint exists
python3 scripts/got_utils.py sprint list | grep "S-000"

# 4. No pending handoffs (unless intentional)
python3 scripts/got_utils.py handoff list --status initiated
```

---

**End of Knowledge Transfer**
