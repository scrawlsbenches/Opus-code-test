# Director Execution Plan: Parallel Task Completion

**Date:** 2025-12-16
**Session:** Parallel Task Execution with Recovery
**Branch:** claude/code-review-task-batching-tOB7H

---

## Task Analysis Summary

### Tasks to Execute (Prioritized)

| ID | Task | Priority | Dependencies | Risk |
|----|------|----------|--------------|------|
| T-f0ff-007 | Mark TASK_LIST.md removal complete | HIGH | None (already done) | None |
| T-f0ff-001 | Generate .ai_meta files | MEDIUM | None | Low |
| T-f0ff-008 | Update CI for task validation | MEDIUM | Legacy removal done | Low |
| T-f0ff-005 | Add deployment guide docs | MEDIUM | None | Low |
| T-f0ff-006 | Add fluent.py API docs | LOW | None | Low |
| T-f0ff-002 | Integrate ML with .ai_meta | MEDIUM | T-f0ff-001 | Medium |
| T-f0ff-009 | Add test coverage to .ai_meta | LOW | T-f0ff-001 | Medium |
| T-f0ff-003 | Add tool output collection | LOW | None | Medium |
| T-f0ff-004 | Add pre-commit hook for ML | LOW | T-f0ff-002 | Medium |

### Duplicate/Completed Tasks (Skip)

- T-20251215-203333-4e1b-001: Already completed (TASK_LIST.md removed)
- T-20251216-090253-f0ff-007: Duplicate of above, just needs marking
- T-20251215-203333-4e1b-002: Same as T-f0ff-008

### Deferred Tasks (Out of Scope)

Legacy tasks requiring significant design work:
- LEGACY-078, LEGACY-080, LEGACY-100, etc.
- T-20251215-145621-16f3-*: ML ephemeral environment (research needed)
- T-20251215-213424-8400-*: Large file refactoring (separate session)

---

## Execution Strategy

### Batch 0: Housekeeping (Sequential, 2 min)
**Goal:** Clean up completed work, mark duplicates

| Agent | Task | Scope |
|-------|------|-------|
| Director | Mark T-f0ff-007 complete | Update task file |
| Director | Mark T-4e1b-001 complete | Update task file |

**Verification:** `python scripts/task_utils.py list --status completed | grep -i legacy`
**Recovery:** Manual task file edit if script fails

---

### Batch 1: Foundation (Parallel, 5-10 min)
**Goal:** Generate .ai_meta + Update CI + Start Docs

| Agent | Task | Files | Isolation |
|-------|------|-------|-----------|
| Agent 1 | Generate .ai_meta files (T-f0ff-001) | cortical/*.ai_meta | Independent |
| Agent 2 | Update CI workflow (T-f0ff-008) | .github/workflows/ci.yml | Independent |
| Agent 3 | Start deployment guide (T-f0ff-005) | docs/deployment.md | Independent |

**Parallel Safety:** No file overlap between agents
**Verification:**
```bash
# Agent 1: Check metadata generated
ls cortical/*.ai_meta cortical/**/*.ai_meta | wc -l

# Agent 2: Validate CI syntax
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"

# Agent 3: Check doc exists
test -f docs/deployment.md && echo "OK"
```

**Recovery Strategies:**
- Agent 1 fails: Run `python scripts/generate_ai_metadata.py` manually
- Agent 2 fails: Revert CI changes, create minimal validation job
- Agent 3 fails: Continue without docs (non-blocking)

---

### Batch 2: Documentation (Parallel, 10-15 min)
**Goal:** Complete documentation tasks

| Agent | Task | Files | Isolation |
|-------|------|-------|-----------|
| Agent 1 | Complete deployment guide | docs/deployment.md | Continue from B1 |
| Agent 2 | Write fluent.py API docs (T-f0ff-006) | docs/fluent-api.md | Independent |

**Parallel Safety:** Different documentation files
**Verification:**
```bash
# Check both docs exist and have content
wc -l docs/deployment.md docs/fluent-api.md
```

**Recovery Strategies:**
- Either fails: Document what was completed, create stub with TODOs
- Both fail: Skip to Batch 3 (docs are non-blocking)

---

### Batch 3: ML Integration (Sequential, 15-20 min)
**Goal:** Integrate ML with metadata system

**Depends on:** Batch 1 Agent 1 (ai_meta files must exist)

| Step | Agent | Task | Files |
|------|-------|------|-------|
| 3.1 | Agent 1 | Integrate ML prediction with .ai_meta (T-f0ff-002) | scripts/ml_file_prediction.py |
| 3.2 | Verify | Test ML predictions use metadata | Run prediction tests |
| 3.3 | Agent 2 | Add test coverage mapping (T-f0ff-009) | scripts/generate_ai_metadata.py |

**Verification:**
```bash
# Test ML integration
python scripts/ml_file_prediction.py predict "Add new feature" --top 5

# Check test coverage in metadata
grep -l "test_coverage" cortical/*.ai_meta
```

**Recovery Strategies:**
- Step 3.1 fails: ML works without metadata, skip integration
- Step 3.3 fails: Coverage mapping is optional, continue

---

### Batch 4: Advanced Features (Parallel, 10-15 min)
**Goal:** Add tool output collection and pre-commit hook

**Depends on:** Batch 3 Step 3.1 (ML integration)

| Agent | Task | Files | Isolation |
|-------|------|-------|-----------|
| Agent 1 | Add tool output collection (T-f0ff-003) | scripts/ml_data_collector.py | Independent |
| Agent 2 | Add pre-commit hook (T-f0ff-004) | scripts/ml-precommit-hook.sh, .githooks/ | Independent |

**Parallel Safety:** Different script domains
**Verification:**
```bash
# Agent 1: Check tool output capture
grep -n "output" scripts/ml_data_collector.py | head -5

# Agent 2: Check hook exists
test -f scripts/ml-precommit-hook.sh && echo "OK"
```

**Recovery Strategies:**
- Either fails: These are enhancements, document partial progress
- Both fail: Create tasks for next session

---

## Rollback Procedures

### Per-Batch Rollback
```bash
# Save checkpoint before each batch
git stash push -m "checkpoint-before-batch-N"

# If batch fails catastrophically
git stash pop
```

### Full Session Rollback
```bash
# Return to session start
git reset --hard HEAD~N  # Where N = commits made

# Or use reflog
git reflog
git reset --hard HEAD@{N}
```

### Partial Recovery
If individual agents fail but others succeed:
1. Commit successful work immediately
2. Create follow-up task for failed work
3. Continue with next batch (if no dependencies)

---

## Success Criteria

| Batch | Must Have | Nice to Have |
|-------|-----------|--------------|
| 0 | Tasks marked complete | - |
| 1 | .ai_meta files exist | CI validates tasks |
| 2 | At least one doc complete | Both docs complete |
| 3 | ML integration tested | Coverage in metadata |
| 4 | Hook script exists | Full tool output capture |

### Minimum Viable Completion
- Batches 0-1 complete = **60% success**
- Batches 0-2 complete = **75% success**
- Batches 0-3 complete = **90% success**
- All batches complete = **100% success**

---

## Execution Commands

### Start Director Session
```bash
# Use /director command with this plan
/director

# Or invoke via Task tool with director orchestration
```

### Quick Verification Between Batches
```bash
# Run after each batch
python scripts/verify_batch.py --quick 2>/dev/null || echo "Manual verify needed"
git status
python -m pytest tests/smoke/ -q 2>/dev/null || python -c "from cortical import CorticalTextProcessor; print('Core OK')"
```

### Commit Pattern
```bash
# After each successful batch
git add -A
git commit -m "batch-N: [description of changes]"
git push -u origin claude/code-review-task-batching-tOB7H
```

---

## Time Budget

| Batch | Estimated | Max Allowed | Timeout Action |
|-------|-----------|-------------|----------------|
| 0 | 2 min | 5 min | Force complete |
| 1 | 10 min | 20 min | Commit partial, continue |
| 2 | 15 min | 25 min | Skip to Batch 3 |
| 3 | 20 min | 30 min | Commit partial, continue |
| 4 | 15 min | 25 min | Create follow-up tasks |

**Total:** 62-105 minutes

---

*Plan generated: 2025-12-16*
