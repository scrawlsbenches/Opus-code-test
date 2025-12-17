# Continuous Development Consciousness Roadmap

**Created:** 2025-12-17
**Status:** Proposal
**Related:** `samples/memories/2025-12-17-session-coverage-and-workflow-analysis.md`

## Vision

A unified development system where Claude sessions are ephemeral but knowledge is continuous. The system automatically captures, consolidates, and surfaces institutional knowledge so each session starts with full context.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE DEVELOPMENT LIFECYCLE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   SESSION START                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ 1. Run full test suite → validate baseline                  │   │
│   │    • If tests FAIL → fix first OR postpone new work        │   │
│   │    • If tests PASS → proceed with confidence               │   │
│   │ 2. Health Dashboard:                                        │   │
│   │    • Coverage: 61% (3 files regressed since yesterday)     │   │
│   │    • Tasks: 5 pending on this branch, 2 stale (>7 days)    │   │
│   │    • Branch: 6 hours since main sync, no conflicts         │   │
│   │    • Model: File prediction ready (523 commits trained)    │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   WORK PHASE                                                         │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Commits → ML Data Captured (batched, not per-commit)       │   │
│   │  File touched → Task auto-linked                             │   │
│   │  Sub-agent spawned → Branch manifest updated                 │   │
│   │  Coverage changed → Delta tracked                            │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   SESSION END                                                        │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  1. Run full test suite → validate no regressions           │   │
│   │     • If tests FAIL → fix before proceeding                 │   │
│   │     • If tests PASS → continue with commit flow             │   │
│   │  2. Commit all changes                                       │   │
│   │  3. Pull latest from origin                                  │   │
│   │  4. Merge origin into feature (preserve origin's changes)   │   │
│   │  5. Push to remote                                           │   │
│   │  6. Batch commit ML data (single commit, not recursive)     │   │
│   │  7. Auto-generate session memory draft                       │   │
│   │  8. Update task statuses from commits                        │   │
│   │  9. Archive branch manifest                                  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│   MERGE TO MAIN                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  1. CI runs full test suite                                  │   │
│   │  2. Coverage report generated                                │   │
│   │  3. Book regenerated with new content                        │   │
│   │  4. Model retrained if threshold reached                     │   │
│   │  5. Debt register updated                                    │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Epic Structure

### Epic 1: Foundation Stability 🏗️
**Goal:** Stop the bleeding - fix what's broken

| Sprint | Tasks | Status |
|--------|-------|--------|
| 1.1 | Fix ML recursive commit loop | ✅ Done |
| 1.2 | Update CLAUDE.md coverage policy | ✅ Done |
| 1.3 | Establish coverage baseline | ✅ Done (61%) |
| 1.4 | Clean up stale/incorrect tasks | ✅ Done |
| 1.5 | Fix branch state tracking | ✅ Done |

### Epic 2: Automatic Knowledge Capture 📚
**Goal:** Never lose knowledge again

| Sprint | Tasks | Status |
|--------|-------|--------|
| 2.1 | SessionEnd auto-memory generation | Pending |
| 2.2 | Post-commit task auto-linking | Pending |
| 2.3 | Add SessionJournalGenerator to book | Pending |
| 2.4 | Branch manifest for parallel work | Pending |

### Epic 3: Intelligent Assistance 🧠
**Goal:** Claude helps Claude

| Sprint | Tasks | Status |
|--------|-------|--------|
| 3.1 | SessionStart health dashboard | Pending |
| 3.2 | Pre-commit coverage warnings | Pending |
| 3.3 | ML model reaches 500 commits | In Progress |
| 3.4 | File prediction in workflow | Pending |

### Epic 4: Self-Healing System 🔄
**Goal:** Problems fix themselves

| Sprint | Tasks | Status |
|--------|-------|--------|
| 4.1 | Stale task detection + escalation | Pending |
| 4.2 | Conflict early warning system | Pending |
| 4.3 | Coverage debt burndown tracking | Pending |
| 4.4 | Automatic weekly debt summary | Pending |

### Epic 5: Living Documentation 📖
**Goal:** The book becomes the brain

| Sprint | Tasks | Status |
|--------|-------|--------|
| 5.1 | Refactor generate_book.py into package | Planned |
| 5.2 | Add CoverageChapterGenerator | Pending |
| 5.3 | Add TaskTimelineGenerator | Pending |
| 5.4 | Add DebtRegisterGenerator | Pending |
| 5.5 | CI: Regenerate book on merge | Pending |

---

## Problem Solutions

### Problem 1: Merge Issues 🔀

**Current:** Parallel branches diverge, conflicts surprise us at merge time.

**Solution: Branch Awareness Protocol**

```
.branch-state/
├── active/
│   ├── claude-feature-abc.json
│   │   {
│   │     "branch": "claude/feature-abc",
│   │     "started": "2025-12-17T10:00:00",
│   │     "files_claimed": ["cortical/query/search.py"],
│   │     "files_touched": ["cortical/query/search.py", "tests/test_search.py"],
│   │     "last_main_sync": "2025-12-17T09:00:00",
│   │     "sub_agents": ["agent-1", "agent-2"]
│   │   }
│   └── claude-bugfix-def.json
├── merged/                    # Historical record
└── conflicts.json             # Auto-detected overlaps
```

**Director Integration:**
```python
# When Director spawns sub-agents:
for agent in sub_agents:
    agent.claim_files(["file1.py", "file2.py"])  # Recorded in manifest

# Before sub-agent commits:
conflicts = check_manifest_conflicts(my_files)
if conflicts:
    warn_director(conflicts)  # Director resolves before merge
```

---

### Problem 2: Data Collection Issues 📊

**Current:** Post-commit hooks create recursive loops, every commit triggers another.

**Solution: Batched Collection with Deferred Commits**

```python
# scripts/ml-session-capture.py (new approach)

class MLSessionCollector:
    """Collects ML data in memory, commits once at session end."""

    def __init__(self):
        self.pending_data = {
            'commits': [],
            'chats': [],
            'actions': []
        }

    def capture_commit(self, commit_data):
        """Add to pending, don't write yet."""
        self.pending_data['commits'].append(commit_data)

    def flush(self):
        """Called once at session end - single atomic commit."""
        if not any(self.pending_data.values()):
            return

        # Write all pending data
        for data_type, items in self.pending_data.items():
            write_batch(data_type, items)

        # Single commit for all ML data
        git_commit("chore: ML data sync (batched)")

        self.pending_data = {'commits': [], 'chats': [], 'actions': []}
```

**Hook Changes:**
- `post-commit`: Capture to memory, don't commit
- `SessionEnd`: Flush all captured data in single commit
- Remove recursive loop entirely

---

### Problem 3: Model Creation 🤖

**Current:** ~400 commits, need 500 for reliable predictions.

**Solution: Accelerated Training + Better Features**

```python
# Immediate actions:
1. Backfill remaining historical commits
2. Weight recent commits higher (last 100 = 2x)
3. Add semantic features:
   - Commit type (feat/fix/refactor/docs)
   - File path patterns (cortical/ vs tests/ vs scripts/)
   - Co-modified file clusters
4. Cross-reference with task completion for better labels

# Threshold triggers:
- At 500 commits: Auto-retrain, notify
- At 1000 commits: Add more complex features
- At 2000 commits: Consider neural approaches
```

**Model Integration Points:**
```
Pre-commit: "Based on 'feat: Add authentication', you might want to also modify: tests/test_auth.py, docs/api.md"

Task creation: "Similar tasks in the past modified: cortical/processor/, tests/unit/"

Code review: "Files with high churn that weren't touched: config.py (usually modified with auth changes)"
```

---

### Problem 4: Director/Sub-Agent Coordination 🎭

**Current:** Sub-agents work in isolation, may create conflicts.

**Solution: Orchestration Protocol**

```
┌─────────────────────────────────────────────────────────────┐
│                    DIRECTOR WORKFLOW                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. PLANNING PHASE                                          │
│     ┌─────────────────────────────────────────────────┐     │
│     │ Director analyzes task                           │     │
│     │ Identifies parallelizable work                   │     │
│     │ Creates batch plan with file assignments         │     │
│     │ Checks branch manifest for conflicts             │     │
│     └─────────────────────────────────────────────────┘     │
│                          │                                   │
│                          ▼                                   │
│  2. EXECUTION PHASE                                         │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│     │ Agent 1  │  │ Agent 2  │  │ Agent 3  │               │
│     │ files:   │  │ files:   │  │ files:   │               │
│     │ a.py     │  │ b.py     │  │ c.py     │               │
│     │ a_test.py│  │ b_test.py│  │ c_test.py│               │
│     └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│          │             │             │                      │
│          ▼             ▼             ▼                      │
│  3. AGGREGATION PHASE                                       │
│     ┌─────────────────────────────────────────────────┐     │
│     │ Director collects results                        │     │
│     │ Checks for unexpected file overlaps              │     │
│     │ Resolves conflicts if any                        │     │
│     │ Creates single coordinated commit                │     │
│     └─────────────────────────────────────────────────┘     │
│                          │                                   │
│                          ▼                                   │
│  4. VERIFICATION PHASE                                      │
│     ┌─────────────────────────────────────────────────┐     │
│     │ Run tests                                        │     │
│     │ Check coverage delta                             │     │
│     │ Update tasks                                     │     │
│     │ Generate batch summary                           │     │
│     └─────────────────────────────────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### Problem 5: Book Generation Enhancement 📖

**Current:** 16 generators, 4,970 lines, monolithic.

**Solution: Package Structure + New Generators**

```
scripts/book_generation/
├── __init__.py
├── base.py                    # BaseGenerator class
├── loaders.py                 # Shared data loading (git, ML, tasks)
├── formatters.py              # Markdown/frontmatter helpers
├── generators/
│   ├── __init__.py
│   ├── algorithm.py           # Existing: AlgorithmChapterGenerator
│   ├── module_doc.py          # Existing: ModuleDocGenerator
│   ├── commit_narrative.py    # Existing: CommitNarrativeGenerator
│   ├── coverage.py            # NEW: CoverageChapterGenerator
│   ├── session_journal.py     # NEW: SessionJournalGenerator
│   ├── task_timeline.py       # NEW: TaskTimelineGenerator
│   ├── debt_register.py       # NEW: DebtRegisterGenerator
│   └── metrics.py             # NEW: MetricsChapterGenerator
├── cli.py                     # Command-line interface
└── scheduler.py               # CI/weekly regeneration logic
```

**New Chapter Structure:**
```
book/
├── 01-foundations/            # Algorithms (existing)
├── 02-architecture/           # Modules (existing)
├── 03-decisions/              # ADRs (existing)
├── 04-evolution/              # Commits (existing)
├── 05-case-studies/           # Debug stories (existing)
├── 06-coverage/               # NEW: Coverage trends
├── 07-sessions/               # NEW: Session journal
├── 08-tasks/                  # NEW: Task timeline
├── 09-debt/                   # NEW: Technical debt
└── 10-metrics/                # NEW: System health
```

---

## Implementation Sprints

### Sprint 1: Foundation (This Week)
```
[x] Fix CLAUDE.md coverage policy
[x] Create coverage baseline (61%)
[x] Document the vision
[x] Fix ML data recursive commit issue
    - Added skip patterns for "ml:", "data: ML", "chore: ML" commits
    - Updated post-commit hook and hooks.py installation script
    - Prevents recursive capture when session capture commits ML data
[x] Create simple branch manifest
    - scripts/branch_manifest.py created with init/touch/status/conflicts/archive commands
    - Auto-initialized on session start via ml-session-start-hook.sh
    - Archived on session end via ml-session-capture-hook.sh
    - Conflict detection with check_conflicts()
```

### Sprint 2: Capture (Next Week)
```
[ ] Test suite at session start/end (Safety Sandwich)
    - SessionStart: Run full tests, decide fix-first vs proceed
    - SessionEnd: Run full tests, block commit if failing
    - Integrate with stop hook workflow
[ ] Checkpoint commit system (Crash Protection)
    - Auto-commit to WIP branch every 15 minutes
    - "wip: Checkpoint [timestamp]" message pattern
    - Squash on session end if work continues
    - Recoverable if session terminates unexpectedly
[ ] SessionEnd auto-memory generation
    - Parse commits in session
    - Extract significant changes
    - Generate draft memory
    - Save to samples/memories/[DRAFT]-...
[ ] Post-commit task linking
    - Regex for T-XXXXX in commit messages
    - Auto-update task status
    - Add commit to task context
[ ] Add SessionJournalGenerator
    - Read samples/memories/*.md
    - Compile into chapter
    - Add to book generation
```

### Sprint 3: Intelligence (Week 3)
```
[ ] SessionStart health dashboard
    - Show coverage delta
    - Show stale tasks
    - Show branch age
    - Show model readiness
[ ] Pre-commit coverage check
    - Calculate coverage on modified files
    - Warn if regression (don't block)
    - Track in .coverage-baseline/
[ ] Reach 500 commit threshold
    - Backfill historical commits
    - Monitor progress
    - Auto-retrain when reached
```

### Sprint 4: Self-Healing (Week 4)
```
[ ] Stale task detection
    - Scan tasks/*.json daily
    - Flag pending > 7 days
    - Auto-create reminder task
[ ] Conflict early warning
    - Check branch manifests on session start
    - Warn if overlap detected
    - Suggest coordination
[ ] Coverage debt burndown
    - Track coverage by file over time
    - Generate burndown chart
    - Include in book
```

---

## Open Questions

These need answers before implementation:

1. **ML Recursive Commit Fix Priority:**
   Should this be Sprint 1 Priority 1? It's annoying but not breaking anything critical.

2. **Director Sub-Agent Coordination:**
   How often is Director mode used? Should branch manifests track sub-agents, or is that overkill?

3. **Book Regeneration Trigger:**
   - On every merge to main (more current, more CI time)
   - Weekly scheduled (less current, less overhead)
   - Manual only (full control)

4. **Health Dashboard Verbosity:**
   - Minimal (3 numbers: coverage, tasks, branch age)
   - Standard (add model status, recent commits)
   - Verbose (full breakdown, only with --verbose)

5. **Memory Auto-Generation:**
   - Always generate draft (may create noise)
   - Only on significant sessions (>5 commits, >2 hours)
   - Never auto-generate (manual only)

---

## Success Metrics

How we know this is working:

| Metric | Current | Target |
|--------|---------|--------|
| Coverage | 61% | 70% (don't regress) |
| Tasks forgotten | Unknown | 0 per week |
| Merge conflicts | Occasional | Detected early |
| ML model commits | ~400 | 500+ |
| Session handoffs | Manual | Auto-generated |
| Book chapters | 5 | 10 |

---

## Related Documents

### Core Process
- `docs/merge-friendly-tasks.md` - Task system with collision-free IDs
- `docs/definition-of-done.md` - When is a task truly complete?
- `docs/dogfooding-checklist.md` - Testing with real usage
- `docs/text-as-memories.md` - Knowledge management guide

### ML Training
- `docs/ml-milestone-thresholds.md` - Why 500/2000/5000 commits for training
- `docs/ml-training-best-practices.md` - Training workflow and guidelines
- `docs/ml-data-collection-knowledge-transfer.md` - Data collection architecture
- `docs/ml-precommit-suggestions.md` - Pre-commit file prediction hook

### Orchestration
- `docs/parallel-agent-orchestration.md` - Director/sub-agent patterns
- `docs/director-orchestration-implementation-plan.md` - Implementation details
- `docs/director-continuation-prompt.md` - Resuming orchestration

### Book Generation
- `docs/REFACTOR-BOOK-GENERATION.md` - Book generation refactoring
- `docs/BOOK-GENERATION-VISION.md` - Long-term vision

### Session Knowledge
- `samples/memories/2025-12-17-session-coverage-and-workflow-analysis.md`
- `samples/memories/2025-12-17-git-merge-forensic-analysis.md`

## Tags

`roadmap`, `continuous-consciousness`, `epics`, `sprints`, `workflow`, `book-generation`, `ml-training`
