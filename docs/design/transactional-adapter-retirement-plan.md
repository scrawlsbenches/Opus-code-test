# TransactionalGoTAdapter Retirement Plan

**Created:** 2026-01-09
**Status:** Active
**Branch:** claude/handoff-commands-pV8E0

---

## Goal

Retire `cortical/got/adapter.py` (TransactionalGoTAdapter) by:
1. Moving useful methods into GoTManager
2. Updating all consumers to use GoTManager directly
3. Deleting the adapter file

**Architecture After:**
```
CLI Handlers → GoTManager → CDG
```

**What stays where:**
- **CDG:** General graph storage (no domain logic)
- **GoTManager:** Domain logic for tasks, sprints, decisions, edges
- **CLI Handlers:** Command parsing, output formatting

---

## Escape Clause: When Confused

If you lose context or get stuck:

1. **Read this file first:** `docs/design/transactional-adapter-retirement-plan.md`
2. **Check current phase:** Look for `# PHASE X COMPLETE` comments in adapter.py
3. **Run validation:** `python -m pytest tests/smoke/ -v --tb=short`
4. **If tests fail after your changes:** `git stash` and reassess
5. **If totally lost:** Create a handoff with what you learned:
   ```bash
   python -m cortical.got handoff session \
       --target "next-agent" \
       --summary "Adapter retirement phase X - stuck on Y" \
       --blockers "describe the blocker"
   ```

**TODO Comment Template** (use when skipping something):
```python
# TODO(adapter-retirement): [DESCRIPTION]
# Reason: [WHY SKIPPED]
# To fix: [WHAT NEEDS TO HAPPEN]
# See: docs/design/transactional-adapter-retirement-plan.md Phase X
```

---

## Pre-Flight Checklist

Before starting ANY phase:
```bash
python -m pytest tests/smoke/ -v --tb=short  # Must pass
python -m cortical.got validate              # Must be healthy
git status                                    # Must be clean
```

---

## Phase 1: Move Status Transition Methods to GoTManager

**Methods to move:**
| Adapter Method | Target Location | Notes |
|----------------|-----------------|-------|
| `start_task(task_id)` | GoTManager | Set status="in_progress" |
| `complete_task(task_id, retrospective)` | GoTManager | Set status="completed" + retrospective |
| `block_task(task_id, reason, blocked_by)` | GoTManager | Set status="blocked" + create BLOCKS edge |

**Steps:**
1. Read current implementation in adapter.py
2. Add method to GoTManager (api.py) with same signature
3. Update adapter method to delegate: `return self._manager.start_task(task_id)`
4. Run smoke tests
5. Commit: `refactor(got): Move start_task to GoTManager`
6. Repeat for each method

**Validation:**
```bash
python -c "from cortical.got import GoTManager; print(hasattr(GoTManager, 'start_task'))"
python -m pytest tests/smoke/ -v --tb=short
```

**Exit Criteria:** All 3 methods exist in GoTManager, adapter delegates to them, tests pass.

---

## Phase 2: Move Query Helper Methods to GoTManager

**Methods to move:**
| Adapter Method | Target Location | Notes |
|----------------|-----------------|-------|
| `get_active_tasks()` | GoTManager | Find tasks with status="in_progress" |
| `get_blocked_tasks()` | GoTManager | Find tasks with status="blocked" + reason |
| `get_next_task()` | GoTManager | Priority-based task selection |
| `what_blocks(task_id)` | GoTManager | Find BLOCKS edges targeting task |
| `what_depends_on(task_id)` | GoTManager | Find DEPENDS_ON edges from task |
| `get_task_dependencies(task_id)` | GoTManager | Alias for dependencies |
| `get_blockers(task_id)` | GoTManager | Alias for what_blocks |
| `get_dependents(task_id)` | GoTManager | Find tasks depending on this one |

**Steps:** Same as Phase 1 - move, delegate, test, commit.

**Validation:**
```bash
python -c "from cortical.got import GoTManager; m = GoTManager.__dict__; print([k for k in m if 'block' in k.lower()])"
python -m pytest tests/smoke/ tests/unit/got/ -v --tb=short
```

**Exit Criteria:** All 8 methods exist in GoTManager, tests pass.

---

## Phase 3: Move Sprint Helper Methods to GoTManager

**Methods to move:**
| Adapter Method | Target Location | Notes |
|----------------|-----------------|-------|
| `claim_sprint(sprint_id, agent)` | GoTManager | Set claimed_by field |
| `release_sprint(sprint_id, agent)` | GoTManager | Clear claimed_by field |
| `add_sprint_goal(sprint_id, desc)` | GoTManager | Append to goals list |
| `list_sprint_goals(sprint_id)` | GoTManager | Return goals list |
| `complete_sprint_goal(sprint_id, idx)` | GoTManager | Mark goal complete |
| `link_task_to_sprint(sprint_id, task_id)` | GoTManager | Use existing add_task_to_sprint |
| `unlink_task_from_sprint(sprint_id, task_id)` | GoTManager | Remove CONTAINS edge |
| `get_task_sprint(task_id)` | GoTManager | Find sprint containing task |

**Note:** `link_task_to_sprint` may be redundant with `add_task_to_sprint`. Check and consolidate.

**Exit Criteria:** All sprint methods in GoTManager, tests pass.

---

## Phase 4: Move KT and Handoff Methods to GoTManager

**Methods to move:**
| Adapter Method | Target Location | Notes |
|----------------|-----------------|-------|
| `append_kt_section(kt_id, title, content)` | GoTManager | Append section to KT |
| `append_to_knowledge_transfer(kt_id, content)` | GoTManager | Legacy append |
| `finalize_knowledge_transfer(kt_id)` | GoTManager | Set status=finalized |
| `link_knowledge_transfer(kt_id, entity_id)` | GoTManager | Create REFERENCES edge |
| Handoff methods (if any unique) | GoTManager | Check what's missing |

**Exit Criteria:** All KT/Handoff methods in GoTManager, tests pass.

---

## Phase 5: Move Introspection Methods to GoTManager

**Methods to move:**
| Adapter Method | Target Location | Notes |
|----------------|-----------------|-------|
| `get_stats()` | GoTManager | Return counts/metrics |
| `validate()` | GoTManager | May already exist, check |
| `graph` (property) | GoTManager | Graph accessor |
| `nodes` (property) | GoTManager | Node accessor |
| `edges` (property) | GoTManager | Edge accessor |

**Exit Criteria:** Introspection methods in GoTManager, tests pass.

---

## Phase 6: Move Git Integration to Standalone Module

**Methods to relocate:**
| Adapter Method | Target Location | Notes |
|----------------|-----------------|-------|
| `infer_edges_from_commit(msg)` | `cortical/got/git_inference.py` | New module |
| `infer_edges_from_recent_commits(n)` | `cortical/got/git_inference.py` | New module |
| `_get_current_branch()` | `cortical/got/git_inference.py` | Helper |

**Steps:**
1. Create `cortical/got/git_inference.py`
2. Move functions (they don't need self, just manager)
3. Update adapter to import and delegate
4. Update CLI handlers that use these

**Exit Criteria:** Git functions in new module, CLI still works.

---

## Phase 7: Retire query() Method

The `query()` method is incomplete (only handles 3 query types). Options:

**Option A: Delete and let tests fail** (mark as known failures)
**Option B: Implement missing query types** (scope creep)
**Option C: Redirect to expression system**

**Recommended:** Option A for now - the expression system (`got expr`) is the future.

**Steps:**
1. Add `# TODO(adapter-retirement): query() method incomplete, use `got expr` instead`
2. Leave method but don't move to GoTManager
3. Update tests to use expression system or mark as expected failures

---

## Phase 8: Update All Consumers

**Files importing TransactionalGoTAdapter (27 total):**

### CLI Modules (14 files):
```
cortical/got/cli/task.py
cortical/got/cli/sprint.py
cortical/got/cli/edge.py
cortical/got/cli/decision.py
cortical/got/cli/query.py
cortical/got/cli/handoff.py
cortical/got/cli/knowledge_transfer.py
cortical/got/cli/backup.py
cortical/got/cli/backlog.py
cortical/got/cli/batch.py
cortical/got/cli/analyze.py
cortical/got/cli/failure.py
cortical/got/cli/orphan.py
cortical/got/factory.py
```

### Tests (4 files):
```
tests/unit/got/test_query_language.py
tests/unit/got/test_tx_adapter.py
tests/unit/test_got_cli.py
tests/behavioral/knowledge_transfer_stories.py
```

### Scripts (1 file):
```
scripts/got_dashboard.py
```

**Update Pattern:**
```python
# Before:
from cortical.got.adapter import TransactionalGoTAdapter
manager = TransactionalGoTAdapter(got_dir)

# After:
from cortical.core.bootstrap import create_container
from cortical.got import GoTManager
container = create_container(got_dir=got_dir)
manager = container.resolve(GoTManager)
```

**Steps per file:**
1. Update import
2. Update instantiation
3. Verify method calls still work (they should - we moved methods)
4. Run tests
5. Commit

---

## Phase 9: Delete Adapter

**Pre-deletion checklist:**
- [ ] All methods moved to GoTManager or new modules
- [ ] All consumers updated
- [ ] Smoke tests pass
- [ ] Unit tests pass (or failures are known/expected)
- [ ] No remaining imports of TransactionalGoTAdapter

**Steps:**
1. `git rm cortical/got/adapter.py`
2. Update `cortical/got/__init__.py` if it exports adapter
3. Run full test suite
4. Commit: `refactor(got): Remove TransactionalGoTAdapter`

---

## Phase 10: Cleanup

1. Remove `GoTBackendFactory` if only purpose was adapter creation
2. Update documentation references
3. Update CLAUDE.md if it mentions adapter
4. Create KT documenting what was done

---

## Validation Commands

```bash
# Quick sanity
python -m pytest tests/smoke/ -v --tb=short

# GoT-specific tests
python -m pytest tests/unit/got/ -v --tb=short

# Full test suite
python -m pytest tests/ -v --tb=short

# Check for remaining adapter imports
grep -r "TransactionalGoTAdapter" --include="*.py" cortical/ tests/ scripts/

# Verify GoTManager has needed methods
python -c "
from cortical.got import GoTManager
needed = ['start_task', 'complete_task', 'block_task', 'get_active_tasks', 'get_blocked_tasks']
for m in needed:
    print(f'{m}: {hasattr(GoTManager, m)}')
"
```

---

## Progress Tracking

Mark phases complete by adding comments to this file:

```
Phase 1: [ ] Not started / [~] In progress / [x] Complete
Phase 2: [ ]
Phase 3: [ ]
Phase 4: [ ]
Phase 5: [ ]
Phase 6: [ ]
Phase 7: [ ]
Phase 8: [ ]
Phase 9: [ ]
Phase 10: [ ]
```

**Current Status:** Phase 1 - Not started

---

## Recovery Information

**Branch:** claude/handoff-commands-pV8E0
**Key files:**
- This plan: `docs/design/transactional-adapter-retirement-plan.md`
- Adapter being retired: `cortical/got/adapter.py`
- Target for methods: `cortical/got/api.py` (GoTManager)
- Git inference new home: `cortical/got/git_inference.py` (to create)

**If resuming after context loss:**
1. Read this file
2. Check "Progress Tracking" section above
3. Run validation commands
4. Continue from incomplete phase
