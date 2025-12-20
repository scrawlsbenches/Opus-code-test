# Parallel Sprint Tracker

> **Multi-Thread Mode Enabled**
>
> Sprints 6, 7, and 8 are designed for parallel execution across different threads.
> Each sprint works on independent areas to minimize merge conflicts.

---

# Active Sprints

## Sprint 6: TestExpert Activation ✅
**Sprint ID:** sprint-006-test-expert
**Epic:** Hubris MoE System (efba)
**Status:** Complete ✅
**Session:** 0c9WR
**Isolation:** `scripts/hubris/experts/test_expert.py`, `tests/hubris/`

### Goals
- [x] Wire TestExpert to actual test outcomes (pass/fail signals)
- [x] Add `suggest-tests` command to CLI
- [x] Integrate with CI results from `.git-ml/commits/` (test_passed field)
- [x] Create test selection accuracy metrics
- [x] Add TestExpert to feedback loop (post-test-run hook)

### Key Files
- `scripts/hubris/experts/test_expert.py` - Core expert logic
- `scripts/hubris_cli.py` - Add suggest-tests command
- `scripts/hubris/feedback_collector.py` - Add test outcome signals

### Success Criteria
- TestExpert predicts relevant tests for code changes
- Accuracy tracked via calibration system
- CLI command shows test suggestions with confidence

### Notes
- TestExpert exists but isn't wired to real outcomes
- CI already captures test results in commit data
- Low risk of conflicts with Sprint 7 or 8

---

## Sprint 7: RefactorExpert (New Expert Type)
**Sprint ID:** sprint-007-refactor-expert
**Epic:** Hubris MoE System (efba)
**Status:** Complete ✅
**Isolation:** `scripts/hubris/experts/refactor_expert.py` (new file)
**Session:** 0u0mC

### Goals
- [x] Create RefactorExpert class inheriting from MicroExpert
- [x] Define refactoring signal types (extract, inline, rename, move, dedupe, simplify)
- [x] Train on commit history patterns (commits with "refactor:" prefix)
- [x] Add refactoring detection heuristics (file size, function count, nesting depth, imports)
- [x] Register in ExpertConsolidator
- [x] Add `suggest-refactor` CLI command
- [x] Write unit tests (29 tests passing)

### Key Files (New)
- `scripts/hubris/experts/refactor_expert.py` - New expert (implemented)
- `tests/hubris/test_refactor_expert.py` - Unit tests (29 tests)

### Key Files (Modify)
- `scripts/hubris/expert_consolidator.py` - Register new expert
- `scripts/hubris_cli.py` - Add suggest-refactor command

### Success Criteria
- RefactorExpert can predict files needing refactoring
- Integrates with existing credit system
- Documented in README

### Notes
- Self-contained new expert - minimal conflicts
- Can reuse FileExpert's TF-IDF infrastructure
- Training data: commits with "refactor:" in message

---

## Sprint 8: Core Library Performance
**Sprint ID:** sprint-008-core-performance
**Epic:** Cortical Text Processor (core)
**Status:** Available 🟢
**Isolation:** `cortical/` directory

### Goals
- [ ] Profile `compute_all()` phases with real corpus
- [ ] Optimize slowest phase identified by profiling
- [ ] Address coverage debt: `cortical/query/analogy.py` (3%)
- [ ] Address coverage debt: `cortical/gaps.py` (9%)
- [ ] Add performance regression tests
- [ ] Update performance benchmarks in docs

### Key Files
- `cortical/processor/compute.py` - Optimization target
- `cortical/query/analogy.py` - Coverage improvement
- `cortical/gaps.py` - Coverage improvement
- `tests/performance/` - Add benchmarks
- `scripts/profile_full_analysis.py` - Profiling tool

### Success Criteria
- No performance regression (verify via benchmarks)
- Coverage improved on target modules
- Profiling data documented

### Notes
- Independent of Hubris work (different directory)
- Profile before optimizing - follow CLAUDE.md principles
- Small, focused improvements over large rewrites

---

## Sprint 9: Projects Architecture ✅
**Sprint ID:** sprint-009-projects-arch
**Epic:** Cortical Text Processor (core)
**Status:** Complete ✅
**Isolation:** `cortical/projects/` (new), `tests/`, `.github/workflows/`
**Session:** ac6d

### Goals
- [x] Create `cortical/projects/` directory structure
- [x] Move MCP server to `cortical/projects/mcp/`
- [x] Move proto to `cortical/projects/proto/` (already removed, placeholder added)
- [x] Update pyproject.toml with per-project dependencies
- [x] Update CI to isolate project tests
- [x] Document Projects architecture pattern
- [x] Verify all tests pass (4,769 core + 38 MCP = 4,807 total)
- [x] Create knowledge transfer document

### Task Dependency Graph (for Sub-Agents)

```
                    ┌─────────────────────┐
                    │ T-001: Create       │
                    │ directory structure │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ T-002: Move MCP         │     │ T-003: Move proto       │
│ (can run in parallel)   │     │ (can run in parallel)   │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│ T-004: Update           │ │ T-006: Document         │
│ pyproject.toml          │ │ architecture            │
└───────────┬─────────────┘ └─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│ T-005: Update CI        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ T-007: Verify tests     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ T-008: Knowledge        │
│ transfer document       │
└─────────────────────────┘
```

### Parallelization Opportunities

| Phase | Tasks | Sub-Agents |
|-------|-------|------------|
| 1. Setup | T-001 | 1 agent |
| 2. Move | T-002, T-003 | 2 parallel agents |
| 3. Config | T-004, T-006 | 2 parallel agents |
| 4. CI | T-005 | 1 agent |
| 5. Verify | T-007 | 1 agent |
| 6. Document | T-008 | 1 agent |

### Key Files

**New Files:**
- `cortical/projects/__init__.py`
- `cortical/projects/mcp/__init__.py`
- `cortical/projects/mcp/server.py`
- `cortical/projects/mcp/tests/`
- `cortical/projects/proto/__init__.py`
- `docs/projects-architecture.md`

**Modified Files:**
- `pyproject.toml` - Add project dependency groups
- `.github/workflows/ci.yml` - Isolate project tests
- `CLAUDE.md` - Add Projects section

**Removed Files:**
- `cortical/mcp_server.py` (moved)
- `tests/test_mcp_server.py` (moved)

### Success Criteria
- Core tests run independently of project tests
- MCP tests only run when `[mcp]` dependencies installed
- CI shows separate status for core vs projects
- Zero-dependency core library maintained
- All existing tests pass

### Notes
- This sprint addresses the recurring MCP test failures
- Creates pattern for future optional features
- Maintains backward compatibility via `cortical.projects.mcp`
- Sub-agents should coordinate via task file updates

---

# Parallel Work Guidelines

## Thread Assignment

| Thread | Sprint | Primary Focus |
|--------|--------|---------------|
| Thread A | Sprint 6 | TestExpert wiring |
| Thread B | Sprint 7 | RefactorExpert creation |
| Thread C | Sprint 8 | Core library performance |

## Conflict Avoidance

Each sprint is designed with isolated file sets:

```
Sprint 6 (TestExpert):
  └── scripts/hubris/experts/test_expert.py
  └── Tests for test selection

Sprint 7 (RefactorExpert):
  └── scripts/hubris/experts/refactor_expert.py (NEW)
  └── scripts/hubris/refactor_patterns.py (NEW)

Sprint 8 (Core Performance):
  └── cortical/**
  └── tests/performance/
```

## Shared Files (Coordinate Changes)

These files may be touched by multiple sprints - coordinate:

- `scripts/hubris_cli.py` - Sprint 6 and 7 both add commands
- `scripts/hubris/expert_consolidator.py` - Sprint 7 registers new expert
- `scripts/hubris/README.md` - All sprints update docs

**Strategy:** Add new functions/commands at end of file to minimize merge conflicts.

## Branch Naming

```
Sprint 6: claude/sprint-006-test-expert-{session-id}
Sprint 7: claude/sprint-007-refactor-expert-{session-id}
Sprint 8: claude/sprint-008-core-performance-{session-id}
```

## Documentation Maintenance

Each sprint should update:
1. **README** for the component being modified
2. **CURRENT_SPRINT.md** - Mark goals complete
3. **Memory document** if significant learnings

---

# Previous Sprints

## Sprint 5: UX & Documentation (Complete ✅)
**Dates:** 2025-12-18
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ Cold-start UX fix with ML fallback (ce81d6d)
- ✅ Calibration command for CLI (8ac951c)
- ✅ Hubris documentation update (e2fc096)

## Sprint 4: Meta-Learning (Complete ✅)
**Dates:** 2025-12-18
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ Wire feedback_collector to live git hooks
- ✅ Add EXPERIMENTAL banner to predictions
- ✅ Create sprint tracking persistence
- ✅ Add git lock detection
- ✅ Exception handling audit (9 fixes)
- ✅ Calibration tracking Phase 1+2

## Sprint 3: MoE Integration (Complete ✅)
**Dates:** 2025-12-15 to 2025-12-17
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ CLI interface (hubris_cli.py)
- ✅ Feedback collector
- ✅ README documentation
- ✅ Integration tests

## Sprint 2: Credit System (Complete ✅)
**Dates:** 2025-12-13 to 2025-12-14
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ CreditAccount, CreditLedger
- ✅ ValueSignal, ValueAttributor
- ✅ CreditRouter, Staking

## Sprint 1: Expert Foundation (Complete ✅)
**Dates:** 2025-12-10 to 2025-12-12
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ MicroExpert base class
- ✅ FileExpert, TestExpert, ErrorDiagnosisExpert, EpisodeExpert
- ✅ ExpertConsolidator and routing

---

# Epics

## Active: Hubris MoE (efba)
**Started:** 2025-12-10
**Status:** Phase 5 - Expert Expansion

### Phases:
- **Phase 1:** Expert foundation ✅ (Sprint 1)
- **Phase 2:** Credit system ✅ (Sprint 2)
- **Phase 3:** Integration ✅ (Sprint 3)
- **Phase 4:** Meta-learning ✅ (Sprint 4-5)
- **Phase 5:** Expert expansion 🔄 (Sprint 6-7)
  - TestExpert activation
  - RefactorExpert creation
  - Additional expert types

## Active: Cortical Core (core)
**Status:** Maintenance

### Focus Areas:
- Performance optimization (Sprint 8)
- Coverage improvement
- Search quality

---

# Sprint Selection Guide

**For New Thread:**

1. Read this file to see available sprints
2. Pick an **Available 🟢** sprint
3. Create branch: `claude/{sprint-id}-{session-id}`
4. Update sprint status to **In Progress 🟡**
5. Work on goals
6. Mark complete, update status to **Complete ✅**

**Sprint Status Key:**
- 🟢 Available - Ready to start
- 🟡 In Progress - Being worked on
- ✅ Complete - All goals done
- 🔴 Blocked - Waiting on dependency

---

# Future Sprints (Backlog)

## Sprint 10: DocumentationExpert
Create expert for documentation suggestions.

## Sprint 11: SecurityExpert
Create expert for security vulnerability detection.

## Sprint 12: PerformanceExpert
Create expert for performance optimization suggestions.

## Sprint 13: DependencyExpert
Create expert for dependency update recommendations.

## Sprint 14: CLI Project Migration
Move CLI wrapper to projects/cli if it becomes problematic.

---

# Stats

| Sprint | Duration | Status | Epic |
|--------|----------|--------|------|
| Sprint 1 | 3 days | ✅ | Hubris MoE |
| Sprint 2 | 2 days | ✅ | Hubris MoE |
| Sprint 3 | 3 days | ✅ | Hubris MoE |
| Sprint 4 | 1 day | ✅ | Hubris MoE |
| Sprint 5 | 1 day | ✅ | Hubris MoE |
| Sprint 6 | 1 day | ✅ | Hubris MoE |
| Sprint 7 | 1 day | ✅ | Hubris MoE |
| Sprint 8 | - | 🟢 | Core |
| Sprint 9 | 1 day | ✅ | Core (Projects) |

---

# Blocked/Deferred Backlog

Items that are intentionally deferred or blocked on external dependencies.

## Deferred Tasks

| Task ID | Description | Reason | Unblock Condition |
|---------|-------------|--------|-------------------|
| T-d0b7-004 | Chunk storage compression | Low priority, git already compresses | Performance profiling shows need |
| LEGACY-190 | REST API wrapper | Scope creep risk | Clear use case emerges |
| LEGACY-191 | Interactive REPL mode | Nice-to-have | User demand |

## Resolved Blockers

| Task ID | Was Blocked On | Resolution |
|---------|----------------|------------|
| LEGACY-133 | WAL implementation | Implemented in graph_persistence.py (2025-12-20) |

## Notes

- Use `deferred` status in task JSON files for backlog items
- Review deferred items monthly for relevance
- Move to active sprint when unblock condition is met
