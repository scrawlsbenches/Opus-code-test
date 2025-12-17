# Session Knowledge Transfer: Coverage Analysis & Workflow Improvements

**Date:** 2025-12-17
**Session:** Git forensic analysis, coverage audit, workflow design
**Branch:** `claude/analyze-git-merge-issues-t8olE`

## Summary

Comprehensive session covering git merge forensics, code coverage audit, task hygiene, and design of a "Continuous Consciousness" framework for sustainable development practices.

## Key Findings

### 1. Git Merge Health: CLEAN
- No remaining merge issues from last 2 days
- 40+ merge commits examined (PRs #94-#114)
- All conflicts properly resolved
- See: `samples/memories/2025-12-17-git-merge-forensic-analysis.md`

### 2. Code Coverage Reality Check

| Metric | Value |
|--------|-------|
| **Actual Coverage** | 61% |
| **CLAUDE.md Target** | >89% |
| **Gap** | 28 percentage points |
| **Files in cortical/** | 57 |
| **Files with tests** | 55 (96.5%) |
| **Files with 0% coverage** | 2 |

**Files with NO coverage:**
- `cortical/cli_wrapper.py` (1,164 lines) - CLI entry point
- `cortical/types.py` (161 lines) - Type aliases only

**Lowest coverage files (<30%):**
| File | Coverage | Notes |
|------|----------|-------|
| `cortical/query/analogy.py` | 3% | Analogy completion |
| `cortical/mcp_server.py` | 4% | MCP server |
| `cortical/gaps.py` | 9% | Gap detection |
| `cortical/analysis/quality.py` | 17% | Clustering metrics |
| `cortical/ml_experiments/metrics.py` | 24% | ML metrics |
| `cortical/fluent.py` | 25% | Fluent API |
| `cortical/query/ranking.py` | 25% | Multi-stage ranking |

**Highest coverage files (>90%):**
- `cortical/analysis/__init__.py` - 100%
- `cortical/processor/core.py` - 100%
- `cortical/chunk_index.py` - 98%
- `cortical/validation.py` - 96%
- `cortical/analysis/tfidf.py` - 95%
- `cortical/observability.py` - 93%

### 3. Task Hygiene Issues Found

**Tasks incorrectly marked pending (now fixed):**
- T-20251217-025222-6b01-008: JSON default format (completed by commit 42106eb)
- T-20251217-103523-6b01-017: Pickle phase-out (completed - removed entirely)

**Root cause:** Task completion happens in code but task status updates are forgotten.

## The Continuous Consciousness Framework (Proposal)

### Problem Statement
Claude sessions are ephemeral but the codebase must be continuous. Knowledge, coverage baselines, and task states get lost between sessions.

### Proposed Solutions

#### A. Coverage Covenant
- Per-file coverage baselines stored in `.coverage-baseline/`
- Pre-commit check: warn (not block) if coverage drops
- Explicit debt acknowledgment for known-low files
- No silent regressions

#### B. Branch Awareness Protocol
- Branch manifests in `.branch-state/active/`
- Track which files each branch is touching
- Early warning for potential conflicts
- Periodic main sync reminders

#### C. Task Auto-Lifecycle
- Session start: show pending tasks for current branch
- Post-commit: auto-detect task references in commit messages
- Pre-commit: suggest task links for significant changes
- Staleness detection: warn for tasks pending >7 days

#### D. Knowledge Capture Triggers
- New module created → prompt for purpose doc
- >200 lines changed → prompt for rationale
- Session >2 hours → auto-generate handoff draft
- Test failure fixed → prompt for learning

#### E. Safety Net (Not Blocking)
- Everything allowed with `--force` or `--acknowledge`
- All overrides logged to `.dev-safety-net/`
- Debt accumulates visibly
- Weekly debt summary auto-generated

#### F. Book Generation as Living Documentation Hub

The existing book generation system (`scripts/generate_book.py`) is the **natural consolidation point** for all framework outputs. Currently has 16 generators - extend with:

**New Generators to Add:**

| Generator | Source | Output |
|-----------|--------|--------|
| `CoverageChapterGenerator` | `.coverage-baseline/`, coverage runs | Chapter showing coverage trends, debt, improvements |
| `DebtRegisterGenerator` | `.dev-safety-net/`, task debt | Chapter tracking technical debt with burndown |
| `SessionJournalGenerator` | `samples/memories/`, session handoffs | Auto-compiled session history with learnings |
| `BranchHistoryGenerator` | `.branch-state/merged/` | Record of parallel work and conflict resolutions |
| `TaskTimelineGenerator` | `tasks/*.json` | Visual timeline of task creation → completion |

**Integration Points:**

```
SessionEnd hook
    ↓
Auto-generate session memory draft
    ↓
Book generation picks up memories
    ↓
BOOK.md includes session learnings
    ↓
Future sessions can search the book
```

**Why Book Generation is Central:**
1. Already indexes and searches its own content (dog-fooding)
2. Generates human-readable output from machine data
3. Creates permanent, searchable institutional memory
4. The "living book" becomes the system's long-term memory

**Existing generators to leverage:**
- `CommitNarrativeGenerator` - already tells the story of changes
- `DecisionStoryGenerator` - captures ADRs
- `CaseStudyGenerator` - synthesizes debug stories from ML data
- `ConceptEvolutionGenerator` - tracks concept cluster changes

**Book generation schedule:**
- On-demand: `python scripts/generate_book.py`
- CI: Generate after successful merge to main
- Weekly: Full regeneration with coverage/debt chapters

### Implementation Phases

**Phase 1 (Immediate Value):**
- [ ] SessionStart hook shows health dashboard
- [ ] Create coverage baseline from current state
- [ ] Task-consciousness in post-commit
- [ ] Add `SessionJournalGenerator` to book generation (leverages existing memories)

**Phase 2 (Medium Effort):**
- [ ] Branch manifest system
- [ ] Pre-commit coverage warnings
- [ ] Auto memory draft generation on SessionEnd
- [ ] Add `CoverageChapterGenerator` to book generation
- [ ] Add `TaskTimelineGenerator` to book generation

**Phase 3 (Higher Effort):**
- [ ] Bidirectional task-code linking
- [ ] Stale task detection
- [ ] Weekly debt summaries
- [ ] Add `DebtRegisterGenerator` to book generation
- [ ] CI integration: regenerate book on merge to main
- [ ] Add `BranchHistoryGenerator` for parallel work documentation

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Don't block on coverage | Blocking is annoying and gets bypassed; visibility is better |
| Track debt explicitly | Known debt is fine; hidden debt is dangerous |
| 61% is current baseline | Acknowledge reality, improve incrementally |
| Auto-capture over discipline | Systems beat willpower |

## Context for Next Session

### If continuing this work:
1. Review the framework proposal above
2. Decide which Phase 1 items to implement first
3. Start with SessionStart health dashboard enhancement

### Files to review:
- `.claude/settings.local.json` - current hooks
- `scripts/ml-session-start-hook.sh` - where to add health check
- `scripts/generate_book.py` - book generation system (4,970 lines, 16 generators)
- `docs/REFACTOR-BOOK-GENERATION.md` - planned refactoring into package
- `CLAUDE.md` - update coverage target to be realistic

### Open questions:
1. Should we lower the CLAUDE.md coverage target from 89% to something achievable (70%)?
2. Or keep it aspirational and track debt explicitly?
3. Which Phase 1 item gives most value for least effort?

## Connections

- Related: [[2025-12-17-git-merge-forensic-analysis.md]]
- Related: [[2025-12-17-session-refactor-book-generation-plan.md]]
- Task file: `tasks/2025-12-17_20-49-13_dbf8.json`
- Book generation task: T-20251217-111356-6b01-019
- Coverage docs: Run `python -m coverage report --include="cortical/*"`

## Tags

`coverage`, `workflow`, `technical-debt`, `continuous-consciousness`, `knowledge-transfer`, `task-hygiene`, `book-generation`, `living-documentation`
