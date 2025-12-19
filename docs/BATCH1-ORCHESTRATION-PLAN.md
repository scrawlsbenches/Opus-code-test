# Batch 1 Orchestration Plan

**Created:** 2025-12-17
**Status:** Active
**Session:** b41bc44b

## Overview

Batch 1 focuses on **Foundation** tasks - quality improvements across 4 non-overlapping domains that can execute in parallel without file conflicts.

## Agent Assignments

### Agent α: Session Lifecycle
**Priority:** HIGH
**Estimated Time:** 2-3 hours
**Branch:** `claude/batch1-alpha-{session_id}`

**File Claims (Exclusive):**
- `scripts/*hook*.sh`
- `scripts/branch_manifest.py`

**Tasks:**
| ID | Description | Priority |
|----|-------------|----------|
| T-20251217-224801-dbf8-002 | Implement test suite at session start/end hooks | HIGH |
| T-20251217-224812-dbf8-003 | Implement checkpoint commit system for crash protection | HIGH |
| T-20251215-145626-16f3-002 | Re-enable and fix git hooks for ML data collection | MEDIUM |

**Success Criteria:**
- [ ] SessionStart hook runs tests and reports pass/fail
- [ ] SessionEnd hook runs tests before allowing commit
- [ ] Checkpoint commits every 15 minutes with squash on end
- [ ] All existing tests pass
- [ ] No coverage regression on modified files

---

### Agent β: Core Quality
**Priority:** MEDIUM
**Estimated Time:** 2-3 hours
**Branch:** `claude/batch1-beta-{session_id}`

**File Claims (Exclusive):**
- `cortical/processor/*.py`
- `cortical/config.py`
- `cortical/observability.py`

**Tasks:**
| ID | Description | Priority |
|----|-------------|----------|
| T-20251217-025138-6b01-002 | Fix MetricsCollector thread-safety documentation | MEDIUM |
| T-20251217-025201-6b01-005 | Bound MetricsCollector timing history to prevent memory growth | MEDIUM |
| T-20251217-025247-6b01-011 | Add type annotations for return types in to_dict methods | LOW |
| T-20251217-025300-6b01-012 | Add docstrings to private methods in processor module | LOW |

**Success Criteria:**
- [ ] MetricsCollector has thread-safety docs and bounded history
- [ ] All to_dict methods have return type annotations
- [ ] Private methods have docstrings
- [ ] All existing tests pass
- [ ] No coverage regression on modified files

---

### Agent γ: Analysis/Query
**Priority:** MEDIUM
**Estimated Time:** 2-3 hours
**Branch:** `claude/batch1-gamma-{session_id}`

**File Claims (Exclusive):**
- `cortical/query/*.py`
- `cortical/analysis.py`

**Tasks:**
| ID | Description | Priority |
|----|-------------|----------|
| T-20251217-025153-6b01-003 | Implement LRU eviction for query expansion cache | MEDIUM |
| T-20251217-025157-6b01-004 | Add input validation for graph_boosted_search weight parameters | MEDIUM |
| T-20251217-025215-6b01-006 | Extract common document name boost logic from search functions | LOW |
| T-20251217-025219-6b01-007 | Factor out common PageRank iteration logic | LOW |
| T-20251217-025321-6b01-015 | Add validation for division by zero in scoring calculations | MEDIUM |

**Success Criteria:**
- [ ] Query cache has LRU eviction with configurable max size
- [ ] Weight parameters validated in graph_boosted_search
- [ ] Document name boost logic DRY
- [ ] PageRank iteration logic factored out
- [ ] Division by zero prevented in scoring
- [ ] All existing tests pass
- [ ] No coverage regression on modified files

---

### Agent δ: Test Quality
**Priority:** MEDIUM
**Estimated Time:** 3-4 hours
**Branch:** `claude/batch1-delta-{session_id}`

**File Claims (Exclusive):**
- `tests/**/*.py`

**Tasks:**
| ID | Description | Priority |
|----|-------------|----------|
| T-20251215-213424-8400-004 | Proactively refactor tests/unit/test_analysis.py before it exceeds token limit | HIGH |
| T-20251217-025242-6b01-010 | Add integration tests for checkpoint resume functionality | MEDIUM |
| T-20251217-111334-6b01-018 | Add regression tests for edge cases identified in code review | MEDIUM |
| T-20251217-025325-6b01-016 | Add performance benchmarks for common operations | LOW |

**Success Criteria:**
- [ ] test_analysis.py refactored into smaller focused files
- [ ] Checkpoint resume has integration tests
- [ ] Edge case regression tests added
- [ ] Performance benchmarks documented
- [ ] All existing tests pass
- [ ] Coverage improved or maintained

---

## Coordination Rules

### Branch Naming
```
claude/batch1-{agent_letter}-{session_id}
```

### Merge Order
1. Agent α (Session) - merged first, foundational
2. Agent β (Processor) - second, core changes
3. Agent γ (Query) - third, depends on stable core
4. Agent δ (Tests) - last, validates all changes

### Conflict Resolution
- If an agent needs a file outside their claim: **STOP and document**
- If tests fail: **Fix before completing**
- If blocked: **Report blocker and continue with remaining tasks**

### Completion Report Format
```json
{
  "agent": "alpha|beta|gamma|delta",
  "status": "complete|partial|blocked",
  "files_modified": ["list", "of", "files"],
  "tests_run": true,
  "tests_passed": true,
  "coverage_delta": "+0.5%|-0.2%",
  "issues_found": ["optional list of issues"],
  "blockers": ["optional list of blockers"]
}
```

---

## File Claim Matrix

```
                        α(Session) β(Processor) γ(Query) δ(Tests)
scripts/*hook*.sh           ██         ░░         ░░        ░░
scripts/branch_manifest.py  ██         ░░         ░░        ░░
cortical/processor/*.py     ░░         ██         ░░        ░░
cortical/config.py          ░░         ██         ░░        ░░
cortical/observability.py   ░░         ██         ░░        ░░
cortical/query/*.py         ░░         ░░         ██        ░░
cortical/analysis.py        ░░         ░░         ██        ░░
tests/**/*.py               ░░         ░░         ░░        ██

██ = Exclusive (can modify)
░░ = Read-only (can read, cannot modify)
```

---

## Related Documents

- `docs/CONTINUOUS-CONSCIOUSNESS-ROADMAP.md` - Overall roadmap
- `docs/parallel-agent-orchestration.md` - Orchestration patterns
- `docs/merge-friendly-tasks.md` - Task system
