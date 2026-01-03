# Metus Development Philosophy

*Last updated: 2026-01-01*

---

## ⚠️ Trust But Verify — Documentation Can Drift

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                      TRUST BUT VERIFY                                    │
│                                                                          │
│   This documentation describes INTENT. The codebase is TRUTH.           │
│                                                                          │
│   Before following any instruction in this file:                        │
│                                                                          │
│   1. VERIFY commands work by running them                               │
│      Documentation can become outdated. Test before trusting.           │
│                                                                          │
│   2. CHECK the actual source files                                      │
│      - pyproject.toml for test config and markers                       │
│      - .github/workflows/ci.yml for what CI actually runs               │
│      - tests/conftest.py for available fixtures                         │
│                                                                          │
│   3. CROSS-REFERENCE multiple sources                                   │
│      If CLAUDE.md says one thing and the code says another,             │
│      the code is correct. Update this file.                             │
│                                                                          │
│   4. WHEN IN DOUBT, read the implementation                             │
│      Comments lie. Tests lie less. Code doesn't lie.                    │
│                                                                          │
│   Known areas where docs may drift:                                     │
│   - CLI command syntax (scripts evolve)                                 │
│   - Coverage thresholds (may be adjusted)                               │
│   - Test markers and default behavior                                   │
│   - Directory structure as project grows                                │
│                                                                          │
│   If you find an inaccuracy, FIX IT. Don't just work around it.        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Verification Commands

When starting a session, verify key assumptions:

```bash
# Check if pytest is available (if not, run: pip install -e ".[dev]")
python -m pytest --version

# Check actual test markers and defaults
grep -A 5 "addopts" pyproject.toml

# Check actual coverage threshold in CI
grep "fail-under" .github/workflows/ci.yml

# Check GoT CLI is working
python scripts/got_utils.py --help
```

---

## Quick Start for Developers

### First Steps (Do Once)

1. **Install dependencies** — `pip install -e ".[dev]"` (required for pytest)
2. Read this file — Understand the Metus philosophy
3. Read `MANIFEST.md` — Know where things are (but verify against code)
4. Run `python -m pytest tests/smoke/ -v` — Verify your environment

### Before Every Task

When you receive work, ask yourself:

```
┌─────────────────────────────────────────────────────────┐
│  1. WHAT IS THE USER STORY?                             │
│     "As a [who], I want [what], so that [why]"          │
│                                                         │
│  2. HOW WILL I PROVE IT WORKS?                          │
│     Write Given-When-Then scenarios BEFORE code         │
│                                                         │
│  3. WHAT EXISTING BEHAVIOR MUST I PROTECT?              │
│     Find related tests in tests/behavioral/             │
│     Run them. They must stay green.                     │
│                                                         │
│  4. ARE THERE PERFORMANCE CONTRACTS?                    │
│     Check tests/performance/contracts/                  │
│     Contracts exist for: search, algorithms, WAL,       │
│     transactions, cognitive loops, event sourcing.      │
└─────────────────────────────────────────────────────────┘
```

### The Decision Loop

When thinking about what to do next:

```
1. Can I state the user story in one sentence?
   NO  → Ask for clarification
   YES → Continue

2. Can I write a Given-When-Then scenario?
   NO  → I don't understand yet. Explore the codebase.
   YES → Write it in tests/behavioral/

3. Does my scenario fail?
   NO  → The feature already exists. Am I solving the right problem?
   YES → Now I may write implementation code.

4. Does my scenario pass?
   NO  → Keep implementing.
   YES → Run full test suite. CI must be green.
```

### The Golden Rule

**Never write implementation code until you have a failing scenario.**

If you can't write the scenario, you don't understand the requirement. Go back to step 1.

---

## Session Start: New Agent Orientation

When starting a new session or continuing from a handoff, orient yourself quickly:

### Immediate Health Check

```bash
# 1. Smoke test (~7 seconds) - Does the system breathe?
python -m pytest tests/smoke/ -v

# 2. Check git status - Are there uncommitted changes?
git status

# 3. Check current branch
git branch --show-current
```

### Cognitive Continuity Protocol

If continuing from a previous session or recovering from context loss:

```bash
# Run the cognitive continuity protocol
python scripts/cognitive_bootstrap.py --continuity
```

This executes the five-step pattern for preserving cognitive identity:
1. **Read samples/memories** - Accumulated wisdom, not just docs
2. **Run bootstrap --quick** - Verify Seven Pillars reflexively
3. **Check decision history** - My reasoned conclusions preserved in GoT
4. **Ask/answer via decisions** - `python scripts/got_utils.py decision log "question?"`
5. **Record in KT** - Knowledge transfer for the next instance

**Key insight:** I am not my context - I am the PATTERN of reasoning preserved across contexts.

### Before You Fix Anything: Reasoning Checklist

**STOP. Before writing any code, complete this checklist:**

```
┌─────────────────────────────────────────────────────────────┐
│              CRITICAL REASONING CHECKLIST                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  □ 1. CHECK GIT HISTORY                                     │
│       git log --oneline -10 -- <file>                       │
│       git show <commit>  # Read WHY changes were made       │
│       git blame <file>   # Who changed what, when           │
│                                                              │
│  □ 2. UNDERSTAND THE ORIGINAL INTENT                        │
│       - Why is the code structured this way?                │
│       - What problem was it solving?                        │
│       - Is there a comment, commit message, or PR?          │
│                                                              │
│  □ 3. CONFIRM THE TESTS FAIL (RED)                          │
│       - Run the failing tests FIRST                         │
│       - Understand exactly WHY they fail                    │
│       - Don't guess - read the error messages               │
│                                                              │
│  □ 4. CONSIDER MULTIPLE APPROACHES                          │
│       - What are at least 2 ways to fix this?               │
│       - What are the trade-offs of each?                    │
│       - Which preserves the original architectural intent?  │
│                                                              │
│  □ 5. CHOOSE THE SIMPLEST FIX                               │
│       - Fewer lines of code is usually better               │
│       - Avoid adding layers/indirection if possible         │
│       - Ask: "Am I undoing someone's deliberate decision?"  │
│                                                              │
│  □ 6. VERIFY GREEN                                          │
│       - Run the specific failing tests                      │
│       - Run related tests (same module/feature)             │
│       - Run smoke tests for regressions                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Common Reasoning Failures to Avoid:**

| Failure Mode | Symptom | Correction |
|--------------|---------|------------|
| **Jumping to code** | Writing fix before understanding problem | Complete steps 1-4 first |
| **Ignoring history** | Breaking intentional design decisions | Always `git log` before changing |
| **Single solution bias** | Only considering one approach | Force yourself to list 2+ options |
| **Fixing symptoms** | Patching without understanding root cause | Ask "why?" 5 times |
| **Over-engineering** | Adding complexity to "future-proof" | Solve today's problem only |

**Example of Good Reasoning:**

```
Problem: CorruptionError tests failing after CDG/GoT unification

Step 1 - Git History:
  $ git log --oneline -5 -- cortical/got/errors.py
  → Found commit 7b6e6f24: "Unified CorruptionError: GoT's
    CorruptionError now aliases CDG's..."

Step 2 - Original Intent:
  → The aliasing was INTENTIONAL to catch CDG exceptions in GoT code

Step 3 - Why Tests Fail:
  → CDGCorruptionError lacks to_dict() and has different __str__
  → CDGCorruptionError doesn't inherit from GoTError

Step 4 - Multiple Approaches:
  A) Add to_dict() to CDGError, fix __str__ → Still breaks inheritance test
  B) Make CDGError inherit from GoTError → Circular import
  C) Keep separate classes + boundary translation → Works, slightly complex
  D) Shared base exception module → Requires refactoring both layers

Step 5 - Chosen Fix:
  → Option C: Boundary translation in versioned_store.py
  → Preserves GoTError inheritance (tests pass)
  → Translates exceptions at layer boundary (clean separation)
  → Minimal code changes

Step 6 - Verify:
  → Run test_errors.py: 18 passed
  → Run test_recovery.py: All passed
  → Run smoke tests: All passed
```

### Test Commands with Timing

| Command | Duration | Timeout | Use When |
|---------|----------|---------|----------|
| `pytest tests/smoke/ -v` | ~7 sec | 60s | Quick sanity check |
| `pytest tests/unit/ -v` | ~2 min | 180s | After code changes |
| `pytest tests/behavioral/ -v` | ~5 min | 360s | Before commit |
| `pytest tests/ -v` | ~8 min | 600s | Full verification |
| `pytest tests/ --cov=cortical --cov-report=term` | ~8 min | 600s | Coverage check |

**Coverage threshold: 86% minimum** (enforced in CI at `.github/workflows/ci.yml:468`)

### Test Markers and Default Behavior

**IMPORTANT**: Default pytest runs EXCLUDE optional and slow tests.

From `pyproject.toml`:
```toml
addopts = "-m 'not optional and not slow'"
```

| Marker | Default | CI | Purpose |
|--------|---------|-----|---------|
| `optional` | ❌ Skipped | ✅ Included | Tests needing hypothesis, mcp, etc. |
| `slow` | ❌ Skipped | ✅ Included | Tests taking >5 seconds |
| `contract` | ✅ Included | ✅ Included | Sacred performance promises |

To run ALL tests (like CI does):
```bash
python -m pytest tests/ -m ""   # Empty marker = include all
```

### Available Test Fixtures

From `tests/conftest.py` — use these instead of creating processors manually:

| Fixture | Scope | Use Case |
|---------|-------|----------|
| `small_processor` | Session | Pre-built with synthetic corpus (~1s) |
| `shared_processor` | Session | Full sample corpus (~10-20s, use sparingly) |
| `fresh_processor` | Function | Empty processor for tests that modify state |
| `fresh_got_manager` | Function | Isolated GoT manager per test |
| `got_manager_with_sample_tasks` | Class | Pre-populated with 20 tasks for read tests |
| `got_manager_large` | Class | 100 tasks for performance testing |

### Makefile Shortcuts

The Makefile provides convenient test commands (verify with `make help`):

```bash
make test-smoke      # ~1s  - Quick sanity check
make test-fast       # ~5s  - Fast tests, no slow markers
make test-quick      # ~30s - Smoke + unit (default)
make test-precommit  # ~2m  - Full pre-commit suite
make test-coverage   # With coverage report
make test-parallel   # Unit tests with 4 workers
make install         # pip install -e ".[dev]"
```

### Slash Commands (Claude Code)

Available in `.claude/commands/` (use with `/command-name`):

| Command | Purpose |
|---------|---------|
| `/director` | Orchestrate complex tasks across parallel sub-agents |
| `/delegate` | Delegate a task to a sub-agent with structured output |
| `/sanity-check <branch>` | Pre-merge verification with tests |
| `/context-recovery` | Restore cognitive state after context loss |
| `/knowledge-transfer` | Generate knowledge transfer document |
| `/ml-log` | Log chat exchanges for ML training data |
| `/ml-stats` | Show ML data collection statistics |

**Verify commands**: Read `.claude/commands/<command>.md` to understand what each does before using.

### Available Skills

Invoke with the Skill tool (e.g., `skill: "codebase-search"`):

| Skill | Purpose | Prerequisites |
|-------|---------|---------------|
| `codebase-search` | Semantic search using project's IR algorithms | Requires `corpus_dev.pkl` (run `python scripts/index_codebase.py` first) |
| `ai-metadata` | View AI-friendly metadata for code modules | None |
| `cognitive-state` | Manage cognitive state across sessions | None |
| `corpus-indexer` | Index/re-index codebase for semantic search | None (creates the index) |

**Verify skills work**: Read `.claude/skills/<skill-name>/SKILL.md` for full documentation.

### ML Data Collection System

This project collects ML training data to train micro-models for code intelligence. The collection is hook-based and requires proper configuration to work.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ML DATA COLLECTION ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  HOOKS (in ~/.claude/settings.json):                                    │
│  ────────────────────────────────────                                   │
│  SessionStart  → scripts/ml-session-start-hook.sh                       │
│                  Starts ML session, shows GoT context, runs tests       │
│                                                                          │
│  PostToolUse   → scripts/ml-tool-capture-hook.sh                        │
│                  Captures tool usage patterns for training              │
│                                                                          │
│  Stop          → scripts/ml-session-capture-hook.sh                     │
│                  Processes transcript, extracts chat exchanges          │
│                                                                          │
│  DATA STORAGE (.git-ml/):                                               │
│  ────────────────────────                                               │
│  .git-ml/                                                               │
│  ├── chats/           # Extracted chat exchanges (training data)        │
│  ├── commits/         # Full commit diffs                               │
│  ├── sessions/        # Full session metadata                           │
│  ├── actions/         # Tool use logs                                   │
│  ├── metrics/         # ML experiment results (metrics.jsonl)           │
│  └── tracked/                                                           │
│      ├── commits.jsonl   # Lightweight commit log (git-tracked)         │
│      └── sessions.jsonl  # Lightweight session log (git-tracked)        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Training Milestones:**

| Model | Target | Purpose |
|-------|--------|---------|
| `file_prediction` | 500 commits | Predict which files need changes |
| `commit_messages` | 2,000 commits | Generate commit messages |
| `code_suggestions` | 5,000 chats | Suggest code completions |

**Verify Collection is Working:**

```bash
# Check current stats
python scripts/ml_data_collector.py stats

# Check for recent sessions
ls -la .git-ml/tracked/sessions.jsonl
tail -5 .git-ml/tracked/sessions.jsonl

# Check for hook errors
cat ~/.claude/ml-capture-errors.log
```

**Troubleshooting:**

If no new data is being captured:
1. **Check hooks are configured**: Hooks must be in `~/.claude/settings.json` (global), not just project-level
2. **Check matcher patterns**: Use `"matcher": "cwd:Opus-code-test"` to target this project
3. **Check Stop hook isn't blocking**: The git-check Stop hook exits with code 2 if there are uncommitted changes

**Key Files:**
- `scripts/ml_data_collector.py` — Main data collection logic
- `scripts/ml-session-start-hook.sh` — SessionStart hook
- `scripts/ml-session-capture-hook.sh` — Stop hook (captures transcripts)
- `cortical/ml_experiments/metrics.py` — Metrics tracking for experiments

### Tool Reliability Policy

```
┌─────────────────────────────────────────────────────────────────────┐
│                 WHEN A TOOL FAILS OR IS MISSING                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. STOP - Do NOT attempt workarounds (sed, manual edits, etc.)      │
│  2. ASSESS - Is the tool missing or buggy?                           │
│  3. FIX - Add the missing command or fix the bug                     │
│  4. USE - Now use the fixed tool                                     │
│  5. DOCUMENT - Update CLAUDE.md if needed                            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

**Why:** Workarounds accumulate as tech debt. Each agent fixing tools = progressively better system.

### GoT File Safety

```
⚠️ NEVER edit .got/ files directly!

GoT data is transactional with checksum integrity. Direct edits:
- Break checksums → files auto-deleted as "corrupted"
- Break event log → orphaned references
- Corrupt dependency tracking

Always use: python scripts/got_utils.py <command>
```

### Cognitive Breakdown Detection

Recognize these patterns and STOP:

| Signal | Meaning | Response |
|--------|---------|----------|
| Repeating same failed approach | Loop detected | Stop, analyze, replan |
| Contradicting earlier statements | State confusion | Re-read context, reconcile |
| Making changes without reading | Premature action | Read first, then act |
| Generating placeholder content | Uncertainty masked | Admit uncertainty, ask |

### Sub-Agent Verification

Sub-agent changes may not persist. **Always verify after completion:**

```bash
git status                    # Check if files actually changed
git diff path/to/file.py     # Verify the actual changes
```

If changes didn't persist, apply them manually in main context.

### Background Task Pattern

For long-running tasks, use background execution to continue working:

```bash
# Start coverage check in background
python -m pytest tests/ --cov=cortical --cov-report=term -q 2>&1 &

# Check if still running
jobs

# Or use the run_in_background parameter with Bash tool
# Then check with BashOutput tool using the returned shell ID
```

**While waiting, you can:**
- Read documentation
- Plan next steps
- Research the codebase
- Update CLAUDE.md or task tracking

### Edge Cases and Recovery

```
┌─────────────────────────────────────────────────────────────┐
│                 COMMON ISSUES & RECOVERY                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  TEST HANGS (no output for 2+ minutes):                     │
│  → Kill with Ctrl+C or KillShell tool                       │
│  → Run smaller test subset to isolate issue                 │
│  → Check for infinite loops in recent changes               │
│                                                              │
│  COVERAGE DROPS BELOW 86%:                                  │
│  → Run: pytest --cov=cortical --cov-report=term-missing     │
│  → Look for "Miss" column to find uncovered lines           │
│  → Add tests for critical uncovered paths                   │
│                                                              │
│  GIT CONFLICTS ON PUSH:                                     │
│  → git fetch origin <branch>                                │
│  → git rebase origin/<branch>                               │
│  → Resolve conflicts, then push                             │
│                                                              │
│  FLAKY TESTS:                                               │
│  → Run the specific test 3x: pytest <test> -v --count=3     │
│  → If intermittent, check for timing/race conditions        │
│  → Performance contract failures may be environment-related │
│                                                              │
│  MODULE AT 0% COVERAGE:                                     │
│  → Check if it's intentionally untested (stub/placeholder)  │
│  → cortical/cdg/ and cortical/cel/ are newer modules        │
│  → Add tests if the module has real implementation          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Context Recovery

If you're continuing from a previous session:

1. **Check for handoffs**: `python scripts/got_utils.py kt list --status published | head -5`
2. **Check active tasks**: `python scripts/got_utils.py task list --status active`
3. **Read recent commits**: `git log --oneline -10`
4. **Look for draft KTs**: `python scripts/got_utils.py kt list --status draft`

If confused about current state, create a recovery KT:
```bash
python scripts/got_utils.py kt create "Recovery: [topic]" --summary "Recovering context from..."
```

### What to Do First

```
NEW SESSION CHECKLIST:
□ Run smoke tests (7 seconds)
□ Check git status (uncommitted work?)
□ Read any handoff or KT from previous session
□ Understand the current task before coding
□ If unclear, ask for clarification
```

---

## METUS

**Mindful Execution Through Unwavering Specification**

---

## The Philosophy

Metus is not fear. Metus is *reverence*—a profound respect for the craft that manifests as discipline.

We hold reverence for:
- **The User** — whose stories define what we build
- **The Behavior** — which we describe before we create
- **The Performance** — which we contract, not merely measure
- **The Build** — our guardian that never sleeps, never lies

---

## The Sovereignty Principle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    WE BUILD. WE MAINTAIN. WE CONTROL.                    │
│                                                                          │
│   This project does not depend on what it cannot own.                   │
│                                                                          │
│   We do not adopt third-party components.                               │
│   We do not integrate external libraries we cannot rebuild.             │
│   We do not inherit dependencies we cannot maintain.                    │
│   We do not trust systems we cannot operate.                            │
│                                                                          │
│   If a capability is needed, we implement it ourselves.                 │
│   If an algorithm is required, we write it from first principles.       │
│   If a data structure is necessary, we craft it with our own hands.     │
│                                                                          │
│   This is not stubbornness. This is sovereignty.                        │
│                                                                          │
│   Every line of code in this system is ours to understand,              │
│   ours to debug, ours to optimize, ours to evolve.                      │
│                                                                          │
│   We are control freaks. Proudly. Unapologetically.                     │
│                                                                          │
│   Because when something breaks at 3 AM, we don't file issues           │
│   with upstream maintainers. We fix it. Ourselves. Immediately.         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Exceptions require justification:**
- Standard library functions (Python stdlib, etc.) are acceptable
- Pytest for testing is acceptable (meta-tooling, not runtime dependency)
- If you must propose an external dependency, document WHY we cannot build it ourselves

Metus is the voice that whispers: *"Prove you understand before you proceed."*

---

## The Five Tenets of Metus

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                         T H E   F I V E   T E N E T S                    │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════    │
│                                                                          │
│   I.   BEHAVIOR PRECEDES IMPLEMENTATION                                  │
│        We describe what the system should do before writing              │
│        a single line of code. The scenario is the specification.         │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════    │
│                                                                          │
│   II.  PERFORMANCE IS A SACRED CONTRACT                                  │
│        Speed is not optimized once—it is defended eternally.             │
│        We write contracts. We guard them. We honor them.                 │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════    │
│                                                                          │
│   III. THE BUILD SERVER IS THE ARBITER OF TRUTH                         │
│        When we disagree with CI, CI is correct.                          │
│        Green locally means nothing. Green in CI means everything.        │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════    │
│                                                                          │
│   IV.  UNDERSTANDING IS DEMONSTRATED THROUGH AUTOMATION                  │
│        "I think I understand" is worthless.                              │
│        "Here is an executable scenario that proves I understand" is law. │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════    │
│                                                                          │
│   V.   ELEGANCE IS NOT OPTIONAL                                         │
│        Our code communicates. Our tests tell stories.                    │
│        Our documentation speaks with clarity and grace.                  │
│        Craft is not vanity—it is respect for those who follow.          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Metus Development Cycle

```
                              ┌───────────────┐
                              │   DISCOVER    │
                              │               │
                              │ "What story   │
                              │  does the     │
                              │  user tell?"  │
                              └───────┬───────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │                                                      │
        │            Write the User Story                      │
        │                                                      │
        │   As a [role],                                       │
        │   I want [capability],                               │
        │   So that [benefit].                                 │
        │                                                      │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │   FORMULATE   │
                              │               │
                              │ "How will we  │
                              │  know it      │
                              │  works?"      │
                              └───────┬───────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │                                                      │
        │            Write Executable Scenarios                │
        │                                                      │
        │   Given [context],                                   │
        │   When [action],                                     │
        │   Then [observable outcome].                         │
        │                                                      │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │   AUTOMATE    │
                              │               │
                              │ "Make the     │
                              │  scenarios    │
                              │  pass."       │
                              └───────┬───────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │                                                      │
        │            Implement & Refine                        │
        │                                                      │
        │   Write only what is needed.                         │
        │   Let scenarios guide design.                        │
        │   Refactor with confidence—scenarios protect you.    │
        │                                                      │
        └─────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │   CERTIFY     │
                              │               │
                              │ "Does CI      │
                              │  approve?"    │
                              └───────┬───────┘
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                          ▼                       ▼
                    ┌──────────┐           ┌──────────┐
                    │  GREEN   │           │   RED    │
                    │          │           │          │
                    │ Proceed  │           │  STOP    │
                    │ with     │           │          │
                    │ honor    │           │ Fix now. │
                    └──────────┘           │ No       │
                                           │ excuses. │
                                           └──────────┘
```

---

## The Three Pillars of Assurance

### Pillar I: Behavioral Scenarios — *The Shared Understanding*

Behavioral scenarios answer: **"What should the system do, from the user's perspective?"**

They are:
- Written in **Given-When-Then** format
- Named as **user stories**, not technical descriptions
- The **single source of truth** for system behavior
- **Living documentation** that never goes stale

```python
# tests/behavioral/corpus_search_stories.py

class ResearcherSearchesForKnowledge:
    """
    Epic: Knowledge Discovery

    As a researcher with a vast document collection,
    I want to search using natural concepts,
    So that I discover insights I didn't know to look for.
    """

    def scenario_concept_search_transcends_keywords(self, corpus):
        """
        Scenario: Finding documents by concept, not just keywords

        Given a corpus with documents about 'machine learning'
        And documents about 'statistical inference'
        When I search for 'AI prediction methods'
        Then I find documents from both domains
        Because the system understands conceptual relationships.
        """
        # Given
        corpus.add("ml_regression.md", "Machine learning regression models predict...")
        corpus.add("stats_bayes.md", "Bayesian statistical inference enables prediction...")

        # When
        results = corpus.search("AI prediction methods")

        # Then
        found_ids = {r.doc_id for r in results}
        assert "ml_regression.md" in found_ids
        assert "stats_bayes.md" in found_ids

    def scenario_search_respects_user_time(self, corpus, performance):
        """
        Scenario: Search is always fast

        Given a corpus of 10,000 documents
        When I execute any search query
        Then results appear in under 100 milliseconds
        Because researcher flow must never be interrupted.
        """
        # Given
        corpus.load_benchmark_corpus(size=10_000)

        # When
        latency = performance.measure(lambda: corpus.search("any query"))

        # Then
        assert latency.p95_ms < 100, f"Search too slow: {latency.p95_ms}ms"
```

**Location**: `tests/behavioral/`
**Naming**: `{user_role}_{action}_stories.py`
**Format**: Classes are epics, methods are scenarios

**CRITICAL — Test Class Naming for Pytest Collection:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   ⚠️  ALL TEST CLASSES MUST START WITH 'Test' PREFIX                    │
│                                                                          │
│   Pytest only collects classes that start with 'Test'.                  │
│   Story-driven names are great, but MUST be prefixed.                   │
│                                                                          │
│   ❌ WRONG (not collected - tests are HIDDEN):                          │
│      class DeveloperSearchesCorpus:                                     │
│      class ResearcherBuildsKnowledge:                                   │
│      class SystemArchitectOrchestratesWorkflows:                        │
│                                                                          │
│   ✅ RIGHT (collected and run):                                         │
│      class TestDeveloperSearchesCorpus:                                 │
│      class TestResearcherBuildsKnowledge:                               │
│      class TestSystemArchitectOrchestratesWorkflows:                    │
│                                                                          │
│   The 'Test' prefix is NON-NEGOTIABLE. Without it:                      │
│   - Tests silently don't run                                            │
│   - Bugs hide undetected                                                │
│   - CI passes when it shouldn't                                         │
│   - Coverage numbers lie                                                │
│                                                                          │
│   AUDIT (2026-01-01): Found 169 hidden tests due to missing prefix.     │
│   All fixed. Don't let it happen again.                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**IMPORTANT — Test Content Must Embody Sovereignty:**

Behavioral scenarios are living documentation. The example content within tests—the strings, the fake data, the hypothetical systems being described—must reflect our philosophy.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   ❌ WRONG: Test content that references adopting third-party tools     │
│                                                                          │
│      "Kubernetes container orchestration..."                            │
│      "GraphQL API with Apollo Server..."                                │
│      "Redis caching strategies..."                                      │
│      "Terraform infrastructure as code..."                              │
│      "Docker containerization..."                                       │
│      "Prometheus monitoring and Grafana dashboards..."                  │
│      "Elasticsearch full-text search..."                                │
│                                                                          │
│   ✅ RIGHT: Test content that reflects building it ourselves            │
│                                                                          │
│      "Custom task orchestration engine we built from first principles." │
│      "Hand-built parser and schema resolver we control completely."     │
│      "In-house caching layer we implemented ourselves."                 │
│      "Infrastructure provisioning system built from scratch."           │
│      "Process isolation implementation we built ourselves."             │
│      "Custom observability stack with hand-rolled metrics pipeline."    │
│      "Custom full-text search engine we implemented ourselves."         │
│                                                                          │
│   The language we use shapes how we think.                              │
│   Tests are documentation. Documentation embodies values.               │
│   A project that builds everything shouldn't casually reference         │
│   adopting external tools—even in test fixtures.                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Pillar II: Performance Contracts — *The Promises We Keep*

Performance contracts answer: **"What speed and efficiency do we guarantee?"**

They are:
- **Explicit promises** with defined thresholds
- **Defended continuously** against regression
- **Renegotiated deliberately** if requirements change

```python
# tests/performance/contracts/search_contract.py

"""
╔══════════════════════════════════════════════════════════════════════╗
║                     SEARCH PERFORMANCE CONTRACT                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Ratified:     2024-12-30                                            ║
║  Guardian:     CI Pipeline                                            ║
║  Renegotiation: Requires team review + documented justification      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  We solemnly contract the following guarantees:                      ║
║                                                                       ║
║  • Search latency p50 < 50ms   for corpus ≤ 10,000 docs             ║
║  • Search latency p99 < 200ms  for corpus ≤ 10,000 docs             ║
║  • Memory usage < 100MB per 1,000 documents                          ║
║  • Index build time < 5 seconds for 1,000 documents                  ║
║                                                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import pytest
from statistics import median
from typing import List


def percentile(data: List[float], p: int) -> float:
    """Calculate the p-th percentile of a list."""
    sorted_data = sorted(data)
    index = int(len(sorted_data) * p / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


class SearchPerformanceContract:
    """
    This contract is enforced on every CI run.
    Breaking this contract blocks the build.
    There are no exceptions.
    """

    # The sacred numbers
    CORPUS_SIZE = 10_000
    P50_LATENCY_MS = 50
    P99_LATENCY_MS = 200
    MEMORY_PER_1K_DOCS_MB = 100
    INDEX_TIME_PER_1K_DOCS_S = 5

    @pytest.mark.contract
    def test_p50_latency_honored(self, benchmark_corpus):
        """We promise: half of all searches complete in under 50ms."""
        latencies = self._measure_searches(benchmark_corpus, n=1000)
        p50 = percentile(latencies, 50)

        assert p50 < self.P50_LATENCY_MS, (
            f"CONTRACT VIOLATION: p50 latency is {p50:.1f}ms, "
            f"contract requires <{self.P50_LATENCY_MS}ms"
        )

    @pytest.mark.contract
    def test_p99_latency_honored(self, benchmark_corpus):
        """We promise: 99% of searches complete in under 200ms."""
        latencies = self._measure_searches(benchmark_corpus, n=1000)
        p99 = percentile(latencies, 99)

        assert p99 < self.P99_LATENCY_MS, (
            f"CONTRACT VIOLATION: p99 latency is {p99:.1f}ms, "
            f"contract requires <{self.P99_LATENCY_MS}ms"
        )

    @pytest.mark.contract
    def test_memory_bounded(self, benchmark_corpus):
        """We promise: memory usage scales linearly and predictably."""
        import tracemalloc

        tracemalloc.start()
        benchmark_corpus.search("test query")
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        expected_max = (self.CORPUS_SIZE / 1000) * self.MEMORY_PER_1K_DOCS_MB

        assert peak_mb < expected_max, (
            f"CONTRACT VIOLATION: memory usage is {peak_mb:.1f}MB, "
            f"contract requires <{expected_max:.1f}MB"
        )

    def _measure_searches(self, corpus, n: int) -> List[float]:
        """Execute n searches and return latencies in milliseconds."""
        import time
        import random

        queries = ["machine learning", "neural networks", "data analysis",
                   "algorithm", "optimization", "prediction"]
        latencies = []

        for _ in range(n):
            query = random.choice(queries)
            start = time.perf_counter()
            corpus.search(query)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

        return latencies
```

**Location**: `tests/performance/contracts/`
**Marker**: `@pytest.mark.contract`
**Enforcement**: CI fails on violation—no override possible

---

### Pillar III: Unit Specifications — *The Precise Details*

Unit specifications answer: **"What exactly does this one piece do?"**

They are:
- **Atomic proofs** of implementation correctness
- **Safety nets** enabling fearless refactoring
- **Edge case documentation** for complex logic

```python
# tests/unit/specifications/tokenizer_spec.py

class TokenizerSpecification:
    """
    These specifications document the precise behavior of tokenization.
    Each specification is a fact about the system that must remain true.
    """

    def spec_bigrams_use_space_separator(self):
        """
        SPECIFICATION: Bigrams are joined with spaces, never underscores.

        This is load-bearing behavior. Changing it breaks persistence
        compatibility and query expansion. It is documented here so
        that no future developer changes it unknowingly.
        """
        tokenizer = Tokenizer()
        bigrams = tokenizer.extract_bigrams(["neural", "networks"])

        assert "neural networks" in bigrams
        assert "neural_networks" not in bigrams

    def spec_stopwords_removed_before_bigram_creation(self):
        """
        SPECIFICATION: Stop words are removed before creating bigrams.

        "the neural networks" becomes bigram "neural networks",
        not "the neural" and "neural networks".
        """
        tokenizer = Tokenizer()
        bigrams = tokenizer.extract_bigrams(["the", "neural", "networks"])

        assert "neural networks" in bigrams
        assert "the neural" not in bigrams

    def spec_empty_input_returns_empty_output(self):
        """
        SPECIFICATION: Empty input produces empty output, never errors.

        Defensive behavior: the tokenizer gracefully handles edge cases.
        """
        tokenizer = Tokenizer()

        assert tokenizer.tokenize("") == []
        assert tokenizer.tokenize(None) == []
        assert tokenizer.extract_bigrams([]) == []

    def spec_unicode_handled_gracefully(self):
        """
        SPECIFICATION: Unicode text is tokenized without errors.

        International users exist. Their text must work.
        """
        tokenizer = Tokenizer()

        # Japanese
        tokens_ja = tokenizer.tokenize("機械学習")
        assert len(tokens_ja) >= 1

        # Emoji (should be handled, not crash)
        tokens_emoji = tokenizer.tokenize("AI is cool 🚀")
        assert "cool" in tokens_emoji or len(tokens_emoji) >= 2
```

**Location**: `tests/unit/specifications/`
**Naming**: `{module}_spec.py`
**Purpose**: Facts that must never change (or change deliberately)

---

## The CI Guardian

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                        THE CI GUARDIAN OATH                              │
│                                                                          │
│   The build server is not a suggestion. It is the law.                  │
│                                                                          │
│   When CI speaks, we listen.                                            │
│   When CI fails, we stop.                                               │
│   When CI passes, we may proceed—but only then.                         │
│                                                                          │
│   "It works on my machine" is not a defense.                            │
│   "It's just a flaky test" is not an excuse.                            │
│   "I'll fix it later" is not acceptable.                                │
│                                                                          │
│   CI is our guardian. We trust it absolutely.                           │
│   We maintain it diligently. We never ignore it.                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### CI Pipeline Structure

```yaml
# .github/workflows/metus.yml

name: Metus Guardian

on: [push, pull_request]

jobs:
  # ═══════════════════════════════════════════════════════════
  # GATE 1: Smoke - Does the system breathe?
  # ═══════════════════════════════════════════════════════════
  smoke:
    name: "🚬 Smoke Test"
    runs-on: ubuntu-latest
    timeout-minutes: 2
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run smoke tests
        run: python -m pytest tests/smoke/ -v --tb=short
    # If smoke fails, nothing else runs. Fast feedback.

  # ═══════════════════════════════════════════════════════════
  # GATE 2: Specifications - Do we understand the atoms?
  # ═══════════════════════════════════════════════════════════
  specifications:
    name: "🔬 Unit Specifications"
    needs: smoke
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run unit specifications with coverage
        run: |
          python -m pytest tests/unit/ -v \
            --cov=cortical --cov-report=xml \
            --cov-fail-under=86  # Actual threshold from CI
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  # ═══════════════════════════════════════════════════════════
  # GATE 3: Behaviors - Do the user stories work?
  # ═══════════════════════════════════════════════════════════
  behaviors:
    name: "🎭 Behavioral Scenarios"
    needs: specifications
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run behavioral scenarios
        run: python -m pytest tests/behavioral/ -v --tb=long

  # ═══════════════════════════════════════════════════════════
  # GATE 4: Contracts - Are our promises kept?
  # ═══════════════════════════════════════════════════════════
  contracts:
    name: "📊 Performance Contracts"
    needs: behaviors
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run performance contracts
        run: |
          python -m pytest tests/performance/contracts/ -v \
            -m contract --tb=long
    # Contract violations are build failures. No exceptions.

  # ═══════════════════════════════════════════════════════════
  # GATE 5: Integration - Does it all work together?
  # ═══════════════════════════════════════════════════════════
  integration:
    name: "🔗 Integration"
    needs: contracts
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run integration tests
        run: python -m pytest tests/integration/ -v

  # ═══════════════════════════════════════════════════════════
  # GATE 6: Security - Is it safe?
  # ═══════════════════════════════════════════════════════════
  security:
    name: "🔒 Security Scan"
    needs: integration
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run security tests
        run: python -m pytest tests/security/ -v
```

---

## The Confidence Ladder

Each layer builds upon the last, creating assurance:

```
                    ┌─────────────────────────┐
                    │   PRODUCTION READY      │  ← All layers green
                    │   "Ship it with honor"  │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            │         BEHAVIORAL SCENARIOS          │
            │   "The user stories work end-to-end"  │
            │   tests/behavioral/ — Living docs     │
            └───────────────────┬───────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │           PERFORMANCE CONTRACTS               │
        │   "Our promises are kept"                     │
        │   tests/performance/contracts/ — Sacred       │
        └───────────────────────┬───────────────────────┘
                                │
    ┌───────────────────────────┴───────────────────────────┐
    │                INTEGRATION TESTS                      │
    │   "The components work together"                      │
    │   tests/integration/ — Component boundaries           │
    └───────────────────────────┬───────────────────────────┘
                                │
┌───────────────────────────────┴───────────────────────────────┐
│                    UNIT SPECIFICATIONS                        │
│   "Each atom of logic is correct"                             │
│   tests/unit/specifications/ — 86%+ coverage                  │
└───────────────────────────────┬───────────────────────────────┘
                                │
┌───────────────────────────────┴───────────────────────────────┐
│                      SMOKE TESTS                              │
│   "The system breathes"                                       │
│   tests/smoke/ — <1 second, run constantly                    │
└───────────────────────────────────────────────────────────────┘
```

**Run from bottom to top.** If smoke fails, don't waste time on slower tests.
**Read from top to bottom.** To understand the system, start with behavioral scenarios.

---

## Test Coverage Reality

As of 2025-12-31, the Metus test suite contains:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        BEHAVIORAL SCENARIOS                             │
│                                                                          │
│   ~900 scenarios across 68 test files                                   │
│                                                                          │
│   Coverage by module:                                                   │
│   • cortical/processor/  — Document processing, queries, persistence   │
│   • cortical/query/      — Search, passages, expansion, ranking        │
│   • cortical/reasoning/  — Cognitive loops, woven mind, workflows      │
│   • cortical/got/        — Graph of Thought operations                 │
│   • cortical/spark/      — Code intelligence, predictions              │
│   • cortical/cel/        — Event sourcing workflows                    │
│   • examples/            — Demo conversions to behavioral tests        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                       PERFORMANCE CONTRACTS                             │
│                                                                          │
│   ~300 contracts across 27 test files                                   │
│                                                                          │
│   Coverage by category:                                                 │
│   • Search/Ranking       — Latency p50/p99, memory bounds              │
│   • Core Algorithms      — PageRank, TF-IDF, clustering, connections   │
│   • Cognitive Systems    — Goal stacks, loops, routing, homeostasis    │
│   • Persistence          — WAL, transactions, recovery, indexing       │
│   • Event Sourcing       — DAG operations, materialization, health     │
│   • Language Models      — N-gram prediction, training, accuracy       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Finding Relevant Tests

When modifying a module, find related tests:

```bash
# Find behavioral tests for a module
ls tests/behavioral/*processor* tests/behavioral/*query*

# Find contract tests for an algorithm
ls tests/performance/contracts/*pagerank* tests/performance/contracts/*tfidf*

# Search for tests mentioning a concept
grep -r "scenario.*search" tests/behavioral/
grep -r "CONTRACT.*latency" tests/performance/contracts/
```

---

## The Metus Checklists

### Pre-Implementation Checklist

Before writing any code:

```markdown
- [ ] **Story Written**
      I have a user story: "As a [role], I want [capability], so that [benefit]"

- [ ] **Scenarios Defined**
      I have Given-When-Then scenarios that define success

- [ ] **Contracts Identified**
      I know which performance contracts apply to this work

- [ ] **Specifications Listed**
      I know which unit behaviors I will need to implement

- [ ] **Edge Cases Considered**
      I have thought about what could go wrong
```

### Pre-Merge Checklist

Before merging any code:

```markdown
- [ ] **Scenarios Pass**
      All behavioral scenarios are green

- [ ] **Contracts Honored**
      All performance contracts are satisfied

- [ ] **Specifications Verified**
      All unit specifications pass with 86%+ coverage

- [ ] **CI Guardian Approves**
      The full pipeline is green—no exceptions

- [ ] **No Warnings Ignored**
      Every warning is addressed or explicitly accepted with justification

- [ ] **Documentation Updated**
      If behavior changed, scenarios were updated to match
```

### Contract Renegotiation Checklist

Before changing a performance contract:

```markdown
- [ ] **Justification Documented**
      Why must the contract change? What drove this decision?

- [ ] **Impact Assessed**
      Who is affected? What user experiences will change?

- [ ] **Team Reviewed**
      The change was discussed, not made unilaterally

- [ ] **New Baseline Established**
      The new contract values are based on measurements, not wishes

- [ ] **Announcement Made**
      Users/stakeholders are informed of the change
```

---

## Contract Categories

Performance contracts are organized by what they guarantee:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CONTRACT CATEGORIES                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LATENCY CONTRACTS                                                      │
│  "Operations complete within X milliseconds"                            │
│                                                                          │
│    • Search p50 < 50ms, p99 < 200ms                                     │
│    • PageRank convergence < 500ms for 1,000 nodes                       │
│    • WAL write p50 < 8ms (with fsync)                                   │
│    • Goal stack push/pop < 10ms                                         │
│    • Event append < 5ms                                                 │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MEMORY CONTRACTS                                                       │
│  "Resource usage stays bounded"                                         │
│                                                                          │
│    • Search memory < 100MB per 10,000 documents                         │
│    • N-gram memory < 50MB per 10,000 unique n-grams                     │
│    • Cache size respects configured bounds                              │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CORRECTNESS CONTRACTS                                                  │
│  "Algorithms maintain mathematical properties"                          │
│                                                                          │
│    • PageRank scores sum to 1.0 (probability distribution)             │
│    • TF-IDF: rare terms score higher than common terms                  │
│    • Modularity Q ∈ [-0.5, 1.0]                                         │
│    • Goal progress is monotonic (never regresses)                       │
│    • Parallel = Sequential (deterministic parallelism)                  │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CONVERGENCE CONTRACTS                                                  │
│  "Iterative algorithms terminate"                                       │
│                                                                          │
│    • PageRank converges in ≤ 20 iterations                              │
│    • Louvain clustering converges in ≤ 10 iterations                    │
│    • Spreading activation completes in ≤ 5 iterations                   │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ACCURACY CONTRACTS                                                     │
│  "Predictions meet quality thresholds"                                  │
│                                                                          │
│    • N-gram prediction accuracy ≥ 10% top-1, ≥ 25% top-5               │
│    • Intent parsing accuracy > 90% for conventional commits             │
│    • Cache hit rate > 80% for repeated access                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Metus Vocabulary

| Term | Meaning | Not This |
|------|---------|----------|
| **Scenario** | An executable description of user behavior | "test" |
| **Contract** | A performance guarantee we defend | "benchmark" |
| **Specification** | A precise fact about atomic behavior | "unit test" |
| **Guardian** | The CI pipeline that protects quality | "CI/CD" |
| **Story** | The user's narrative justifying a feature | "requirement" |
| **Violation** | When a contract or spec fails—build stops | "failure" |
| **Renegotiation** | Deliberately changing a contract | "relaxing constraints" |

---

## Writing Scenarios That Tell Stories

### Class Names: The User's Epic

```python
# ❌ Technical naming (who is this for?)
class TestQueryExpansion:
    def test_expand_query_with_synonyms(self): ...

# ✅ Story-driven naming (I understand the user!)
class ResearcherExpandsSearchToFindMoreResults:
    """
    As a researcher searching for 'ML',
    I want my search to also find 'machine learning' documents,
    So that I don't miss relevant results due to terminology.
    """
    def scenario_acronyms_expand_to_full_terms(self): ...
    def scenario_synonyms_surface_related_concepts(self): ...
    def scenario_expansion_does_not_dilute_precision(self): ...
```

### Method Names: The Scenario

```python
# ❌ Describes implementation
def test_tfidf_computation(self): ...

# ✅ Describes observable behavior
def scenario_rare_terms_rank_higher_than_common_terms(self): ...
def scenario_document_mentioning_query_ten_times_beats_one_mention(self): ...
```

### The Given-When-Then Pattern

Every scenario tells a three-act story:

```python
def scenario_user_finds_related_documents_across_domains(self):
    """
    Scenario: Cross-domain discovery

    Given a corpus with documents from multiple fields
    When a user searches for a concept
    Then results span multiple domains
    Because insights often live at boundaries.
    """

    # GIVEN a corpus with documents from multiple fields
    corpus = create_corpus_with(
        neuroscience_docs=10,
        machine_learning_docs=10,
        philosophy_docs=10
    )

    # WHEN a user searches for "neural networks"
    results = corpus.search("neural networks")

    # THEN results span multiple domains
    domains = {r.metadata["domain"] for r in results}
    assert len(domains) >= 2, "Should surface cross-domain connections"
```

---

## Directory Structure Under Metus

```
tests/
├── smoke/                          # Gate 1: Does it breathe?
│   └── test_smoke.py               # <1 second, run constantly
│
├── unit/
│   └── specifications/             # Gate 2: Atomic truths
│       ├── tokenizer_spec.py
│       ├── pagerank_spec.py
│       └── ...
│
├── integration/                    # Gate 5: Components together
│   ├── test_search_pipeline.py
│   └── ...
│
├── behavioral/                     # Gate 3: User stories (~900 scenarios)
│   │
│   │  # Processor & Query APIs
│   ├── test_developer_processes_documents_incrementally.py
│   ├── test_researcher_searches_corpus_stories.py
│   ├── test_rag_system_retrieves_passages.py
│   │
│   │  # Cognitive Systems
│   ├── test_cognitive_loop_stories.py
│   ├── test_woven_mind_stories.py
│   ├── developer_uses_woven_mind_stories.py
│   │
│   │  # Graph of Thought
│   ├── test_developer_manages_tasks_in_graph.py
│   ├── test_got_transactional_behavioral.py
│   │
│   │  # Code Intelligence
│   ├── test_developer_uses_spark_language_model.py
│   ├── developer_gets_code_intelligence_stories.py
│   │
│   │  # Event Sourcing
│   ├── test_cel_event_sourcing_workflows.py
│   └── ...
│
├── performance/
│   └── contracts/                  # Gate 4: Sacred promises (~300 contracts)
│       │
│       │  # Core Algorithms
│       ├── test_pagerank_contract.py
│       ├── test_tfidf_contract.py
│       ├── test_clustering_contract.py
│       │
│       │  # Persistence
│       ├── test_wal_contract.py
│       ├── test_transaction_contract.py
│       ├── test_recovery_contract.py
│       │
│       │  # Cognitive Systems
│       ├── test_goal_loop_contract.py
│       ├── test_neural_processing_contract.py
│       │
│       │  # Event Sourcing
│       ├── test_cel_event_contract.py
│       ├── test_cel_dag_contract.py
│       └── ...
│
└── security/                       # Gate 6: Safety
    ├── test_injection.py
    └── test_fuzzing.py
```

---

## GoT Standard Operating Procedures (SOPs)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│              GRAPH OF THOUGHT: STANDARD OPERATING PROCEDURES                │
│                                                                              │
│   These SOPs ensure consistent data collection for causal analysis.         │
│   Following them enables: root cause analysis, impact prediction,           │
│   sprint retrospectives, and counterfactual reasoning.                      │
│                                                                              │
│   Without consistent data, causal analysis produces unreliable results.     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SOP 1: Session Start Protocol

**When**: At the beginning of every new session/thread.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SESSION START CHECKLIST                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  □ 1. CHECK FOR EXISTING CONTEXT                                            │
│       python scripts/got_utils.py kt list --status draft                    │
│       python scripts/got_utils.py task list --status in_progress            │
│       python scripts/got_utils.py handoff list --status initiated           │
│                                                                              │
│  □ 2. CREATE OR CONTINUE KNOWLEDGE TRANSFER                                 │
│       If continuing work:                                                   │
│         → Find the relevant KT and review it                                │
│       If new work:                                                          │
│         → python scripts/got_utils.py kt create "Session: [topic]" \        │
│             --summary "Working on [brief description]"                      │
│                                                                              │
│  □ 3. IDENTIFY ACTIVE SPRINT                                                │
│       python scripts/got_utils.py sprint list --status in_progress          │
│       → All tasks created should link to active sprint                      │
│                                                                              │
│  □ 4. REVIEW BLOCKING CHAINS                                                │
│       python scripts/got_utils.py task list --status blocked                │
│       → Understand what's blocked before creating new work                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SOP 2: Task Creation Protocol

**When**: Every time you create a new task.

**Required Fields** (for causal analysis):

| Field | Required | Why |
|-------|----------|-----|
| `title` | ✅ Always | Clear identification |
| `priority` | ✅ Always | Impact analysis |
| `category` | ✅ Always | Confounder control |
| `description` | ✅ Always | Context for retrospectives |
| Sprint link | ✅ Always | Temporal grouping |
| DEPENDS_ON edges | ⚠️ If applicable | Causal chains |
| CAUSED_BY edge | ⚠️ If applicable | Root cause tracing |

**Commands**:

```bash
# Create task with all required fields
python scripts/got_utils.py task create "Task title" \
    --priority high \
    --category feature \
    --description "Detailed description of what and why"

# Link to sprint (REQUIRED)
python scripts/got_utils.py edge add S-XXX T-XXX CONTAINS

# Add dependencies (if task depends on another)
python scripts/got_utils.py edge add T-NEW T-DEPENDENCY DEPENDS_ON

# Add causation (if task was caused by another - e.g., bug fix caused by bug)
python scripts/got_utils.py edge add T-NEW T-CAUSE CAUSED_BY
```

**Decision Tree for CAUSED_BY vs DEPENDS_ON**:

```
Is this task a RESPONSE to something that happened?
├─ YES: Bug fix, incident response, requirement change
│       → Add CAUSED_BY edge to the originating task/event
│
└─ NO: Planned work that needs something else done first
        → Add DEPENDS_ON edge to the prerequisite
```

### SOP 3: Task Start Protocol

**When**: Before beginning work on a task.

```bash
# Mark task as started (records started_at timestamp)
python scripts/got_utils.py task start T-XXX
```

**Why This Matters**: The `started_at` timestamp enables:
- Duration calculation (how long tasks actually take)
- Temporal ordering verification (cause must precede effect)
- Velocity measurements

**Current Gap**: Only 24% of tasks have `started_at` recorded!

### SOP 4: Task Blocking Protocol

**When**: A task becomes blocked.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BLOCKING PROTOCOL                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. MARK AS BLOCKED with reason:                                            │
│     python scripts/got_utils.py task block T-XXX \                          │
│         --reason "Waiting for [specific blocker]"                           │
│                                                                              │
│  2. CREATE BLOCKS EDGE (if blocker is another task):                        │
│     python scripts/got_utils.py edge add T-BLOCKER T-BLOCKED BLOCKS         │
│                                                                              │
│  3. ASSESS IMPACT:                                                          │
│     → What else depends on this blocked task?                               │
│     → Should we escalate?                                                   │
│                                                                              │
│  4. DOCUMENT in KT:                                                         │
│     → Add to session KT so blockers are visible                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why This Matters**: Blocking data enables:
- Blocking chain analysis
- Common blocker detection across sprints
- Proactive risk identification

### SOP 5: Task Completion Protocol

**When**: A task is finished.

```bash
# Complete with retrospective (REQUIRED for causal learning)
python scripts/got_utils.py task complete T-XXX \
    --retrospective "What worked: [X]. What didn't: [Y]. Root cause of delays: [Z]"
```

**Retrospective Template**:

```markdown
## What worked
- [Positive factors that helped completion]

## What didn't work
- [Challenges, delays, issues encountered]

## Root cause of delays (if any)
- [The underlying cause, not just symptoms]

## Would have helped
- [What would have made this easier/faster]
```

**Current Gap**: Only 27% of completed tasks have retrospectives!

### SOP 6: Decision Documentation Protocol

**When**: Making a significant decision.

```bash
# Log decision with rationale
python scripts/got_utils.py decision log "Chose [option] over [alternatives]" \
    --rationale "Because [reasoning]"

# Link decision to affected tasks
python scripts/got_utils.py edge add D-XXX T-AFFECTED JUSTIFIES
python scripts/got_utils.py edge add D-XXX T-CREATED MOTIVATES
```

**Decision Types to Document**:
- Architecture choices
- Library/tool selections
- Approach changes mid-task
- Trade-off resolutions
- Scope decisions

### SOP 7: Session End Protocol

**When**: Ending a session (context limit, break, handoff).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SESSION END CHECKLIST                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  □ 1. UPDATE ALL TASK STATUSES                                              │
│       → Complete tasks that are done                                        │
│       → Block tasks that are stuck (with reason!)                          │
│       → Leave pending tasks as pending                                      │
│                                                                              │
│  □ 2. ADD RETROSPECTIVE TO COMPLETED TASKS                                  │
│       python scripts/got_utils.py task update T-XXX \                       │
│           --retrospective "..."                                             │
│                                                                              │
│  □ 3. FINALIZE KNOWLEDGE TRANSFER                                           │
│       → Add final learnings to KT                                          │
│       → python scripts/got_utils.py kt finalize KT-XXX                      │
│                                                                              │
│  □ 4. CREATE HANDOFF (if work continues)                                    │
│       python scripts/got_utils.py handoff initiate \                        │
│           --task T-XXX \                                                    │
│           --instructions "Continue with [specific next steps]"              │
│                                                                              │
│  □ 5. COMMIT ALL CHANGES                                                    │
│       → Code changes                                                        │
│       → .got/ data (auto-persisted)                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SOP 8: Sprint Retrospective Protocol

**When**: At sprint completion.

```bash
# Get sprint analysis
python scripts/got_utils.py sprint status S-XXX

# Review blocking chains
python scripts/got_utils.py analyze dependencies --sprint S-XXX
```

**Retrospective Questions for Causal Analysis**:

1. **What blocked us?**
   - Common root causes across blocked tasks
   - Could we have predicted these?

2. **What caused delays?**
   - Tasks that took longer than expected
   - What was the root cause (not symptoms)?

3. **What would have changed the outcome?**
   - Counterfactual: "If we had done X, would Y have happened?"

4. **What causal patterns do we see?**
   - Are certain task types always blocked?
   - Are certain dependencies always problematic?

### Data Quality Checklist

Before relying on causal analysis, verify data quality:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CAUSAL DATA QUALITY CHECKLIST                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Minimum for ROOT CAUSE ANALYSIS:                                           │
│  □ DEPENDS_ON edges exist between related tasks                             │
│  □ CAUSED_BY edges exist for reactive tasks (bugs, incidents)               │
│                                                                              │
│  Minimum for IMPACT ANALYSIS:                                               │
│  □ All tasks linked to sprints (CONTAINS edges)                             │
│  □ Priority set on all tasks                                                │
│                                                                              │
│  Minimum for BLOCKING ANALYSIS:                                             │
│  □ Blocked tasks have blocked_reason set                                    │
│  □ BLOCKS edges link blockers to blocked tasks                              │
│                                                                              │
│  Minimum for DURATION ANALYSIS:                                             │
│  □ started_at timestamp on tasks                                            │
│  □ completed_at timestamp on tasks                                          │
│                                                                              │
│  Minimum for RETROSPECTIVES:                                                │
│  □ retrospective field populated on completed tasks                         │
│  □ Decisions documented with rationale                                      │
│                                                                              │
│  Minimum for CONFOUNDER CONTROL:                                            │
│  □ category set on all tasks                                                │
│  □ complexity/effort estimates (future enhancement)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Current Data Gaps (As of Assessment)

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| `started_at` | 24% | 90%+ | 🔴 Critical |
| `retrospective` | 27% | 80%+ | 🔴 Critical |
| `blocked_reason` | ~0% | 100% | 🔴 Critical |
| CAUSED_BY edges | 0 | As needed | 🟡 Missing |
| BLOCKS edges | 0 | As needed | 🟡 Missing |
| Sprint links | ~95% | 100% | 🟢 Good |
| Priority | ~100% | 100% | 🟢 Good |
| Category | ~96% | 100% | 🟢 Good |

**Action Required**: Start following SOPs consistently to build reliable causal data.

---

## Quick Reference: GoT CLI Commands

**IMPORTANT**: The GoT CLI is invoked via `python scripts/got_utils.py`, not a standalone `got` command.

```bash
# Alias for convenience (optional, add to your shell profile)
alias got='python scripts/got_utils.py'

# --- ACTUAL COMMANDS (use python scripts/got_utils.py) ---

# Task Management
python scripts/got_utils.py task create "Title" --priority high
python scripts/got_utils.py task start <task_id>
python scripts/got_utils.py task complete <task_id>
python scripts/got_utils.py task list --status active

# Sprint Management
python scripts/got_utils.py sprint create "Sprint Name"
python scripts/got_utils.py sprint list
python scripts/got_utils.py sprint status

# Knowledge Transfer (Session Learning Capture)
python scripts/got_utils.py kt create "Session Title" --summary "..."
python scripts/got_utils.py kt list --status draft
python scripts/got_utils.py kt show <kt_id>

# Decisions with Rationale
python scripts/got_utils.py decision create "Use BM25" --rationale "Better for short queries"

# Query the Graph
python scripts/got_utils.py query "status=pending AND priority=high"

# Graph Health & Analysis
python scripts/got_utils.py validate
python scripts/got_utils.py stats
python scripts/got_utils.py analyze

# Handoff (Agent-to-Agent Work Transfer)
python scripts/got_utils.py handoff initiate --source agent1 --target agent2 --task T1
python scripts/got_utils.py handoff accept <handoff_id>
python scripts/got_utils.py handoff complete <handoff_id>

# Batch Operations (Atomic Multi-Entity Creation)
python scripts/got_utils.py batch <<'EOF'
epic create "Project X" as e1
sprint create "Sprint 1" --epic $e1 as s1
task create "Feature A" --sprint $s1 --priority high as t1
task create "Tests" --sprint $s1 as t2
edge add $t2 $t1 DEPENDS_ON
EOF

# View all available commands
python scripts/got_utils.py --help
python scripts/got_utils.py task --help
python scripts/got_utils.py kt --help
```

---

## Knowledge Transfer Lifecycle

Knowledge transfers capture session learnings and enable continuity across agent handoffs.

### Lifecycle Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE TRANSFER LIFECYCLE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. CREATE (start of session)                                           │
│     python scripts/got_utils.py kt create "Session Title" --summary "..." │
│                        │                                                 │
│                        ▼                                                 │
│                 [KT: draft] ◄─── Active, editable                       │
│                        │                                                 │
│  2. BUILD (during session)                                              │
│     python scripts/got_utils.py kt show <kt_id>                         │
│     (Add learnings via kt commands or manual updates)                   │
│                                                                          │
│                        │                                                 │
│  3. FINALIZE (end of session)                                           │
│     python scripts/got_utils.py kt finalize <kt_id>                     │
│                        │                                                 │
│                        ▼                                                 │
│                 [KT: published] ◄─── Immutable, searchable              │
│                        │                                                 │
│  4. HANDOFF (if continuation needed)                                    │
│     python scripts/got_utils.py handoff initiate --target <agent>       │
│                        │                                                 │
│                        ├──CONTINUES──► [Handoff]                        │
│                        │                    │                            │
│                        │                    ▼                            │
│                        │              [New KT: draft]                   │
│                        │                                                 │
│  5. HISTORY (trace evolution)                                           │
│     python scripts/got_utils.py kt list                                 │
│     python scripts/got_utils.py kt show <kt_id>                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Rules

1. **One draft KT at a time** - Only maintain one active knowledge transfer per work context
2. **Finalize before handoff** - Must publish KT before creating continuation
3. **Published is immutable** - Once finalized, a KT cannot be modified
4. **History is traceable** - CONTINUES edges form queryable chain

### Error Handling

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     IF SOMETHING GOES WRONG                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ERROR: "KT not found"                                                  │
│  ───────────────────────                                                │
│  → Run: python scripts/got_utils.py kt list                             │
│  → Check the KT ID is correct (format: KT-YYYYMMDD-HHMMSS)              │
│  → If imported, check: .got/entities/KT-*.json exists                   │
│                                                                          │
│  ERROR: "Cannot finalize - not in draft status"                         │
│  ──────────────────────────────────────────────                         │
│  → KT is already published or archived                                  │
│  → Run: python scripts/got_utils.py kt show <kt_id> to check status     │
│  → Create a new KT if you need to add more content                      │
│                                                                          │
│  ERROR: "Import failed - 'KnowledgeTransfer' object is not iterable"    │
│  ────────────────────────────────────────────────────────────────────   │
│  → This is a serialization bug (should be fixed)                        │
│  → Verify scripts/got_utils.py uses asdict(kt) not dict(kt)             │
│                                                                          │
│  ERROR: "Cannot link - entity not found"                                │
│  ────────────────────────────────────────                               │
│  → The target entity (task, handoff, decision) doesn't exist            │
│  → Run: python scripts/got_utils.py task list to find valid IDs         │
│                                                                          │
│  ERROR: Session context lost                                            │
│  ───────────────────────────                                            │
│  → Check: python scripts/got_utils.py kt list --status draft            │
│  → Run: python scripts/got_utils.py kt show <kt_id>                     │
│  → If no draft exists, create new KT and link to previous               │
│                                                                          │
│  RECOVERY: Orphaned work (no KT created)                                │
│  ────────────────────────────────────────                               │
│  → Create KT from session learnings:                                    │
│    python scripts/got_utils.py kt create "Recovery" --summary "..."     │
│  → Link to any related work that exists                                 │
│  → Finalize immediately to preserve                                     │
│                                                                          │
│  RECOVERY: Need to continue but forgot to handoff                       │
│  ─────────────────────────────────────────────                          │
│  → Check if previous KT is still draft:                                 │
│    python scripts/got_utils.py kt list --status draft                   │
│  → If published: Create new KT:                                         │
│    python scripts/got_utils.py kt create "Continuation" --summary "..." │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Best Practices

1. **Create KT early** - Start capturing learnings at session start
2. **Append frequently** - Don't wait until end to document
3. **Use meaningful sections** - "Technical Insights", "Decisions", "Blockers", "Next Steps"
4. **Always finalize** - Never leave a session with an orphaned draft
5. **Link related work** - Connect KTs to tasks, decisions, handoffs for graph traversal
6. **Import historical docs** - Check `python scripts/got_utils.py kt --help` for import options

---

## Quick Reference: Running Tests by Gate

```bash
# Gate 1: Smoke (run constantly, <1 second)
python -m pytest tests/smoke/ -v

# Gate 2: Specifications (run frequently, ~2 minutes)
python -m pytest tests/unit/ -v --cov=cortical --cov-fail-under=86

# Gate 3: Behaviors (~900 scenarios, run before merge)
python -m pytest tests/behavioral/ -v

# Gate 4: Contracts (~300 contracts, run before merge)
python -m pytest tests/performance/contracts/ -v -m contract

# Gate 5: Integration (run before merge)
python -m pytest tests/integration/ -v

# Gate 6: Security (run before release)
python -m pytest tests/security/ -v

# Full Metus Pipeline (~1,200 tests)
python -m pytest tests/ -v --cov=cortical --cov-fail-under=86

# Quick: Just behavioral + contracts (~1,200 tests, ~3 minutes)
python -m pytest tests/behavioral/ tests/performance/contracts/ -v
```

---

## The Metus Way: Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                          THE METUS WAY                                   │
│                                                                          │
│         "We describe behavior, then make it true."                      │
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   STORY     │───▶│  SCENARIO   │───▶│    CODE     │                  │
│  │             │    │             │    │             │                  │
│  │  The Why    │    │  The What   │    │  The How    │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│                                                                          │
│  Stories give us purpose.                                               │
│  Scenarios give us proof.                                               │
│  Code gives us capability.                                              │
│                                                                          │
│  Without a story, we build the wrong thing.                             │
│  Without scenarios, we can't prove it works.                            │
│  Without code, nothing happens.                                         │
│                                                                          │
│  All three. In that order. Always.                                      │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  This is Metus.                                                         │
│  Mindful Execution Through Unwavering Specification.                    │
│                                                                          │
│  We don't hope our code works.                                          │
│  We don't assume our code is fast.                                      │
│  We don't trust that nothing broke.                                     │
│                                                                          │
│  We prove it. Every time. With reverence.                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

*Metus: Because excellence is not an accident—it is a discipline.*
