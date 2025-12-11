# Task List: Cortical Text Processor

Active backlog for the Cortical Text Processor project. Completed tasks are archived in [TASK_ARCHIVE.md](TASK_ARCHIVE.md).

**Last Updated:** 2025-12-11
**Pending Tasks:** 24
**Completed Tasks:** 75+ (see archive)

---

## Active Backlog

<!-- Machine-parseable format for automation -->

### 🔴 Critical (Do Now)

| # | Task | Category | Depends | Effort |
|---|------|----------|---------|--------|
| 88 | Create package installation files | DevEx | - | Small |

### 🟠 High (Do This Week)

| # | Task | Category | Depends | Effort |
|---|------|----------|---------|--------|
| 89 | Create CONTRIBUTING.md | DevEx | - | Small |
| 90 | Create docs/quickstart.md tutorial | Docs | 88 | Medium |
| 94 | Split query.py into focused modules | Arch | - | Large |
| 97 | Integrate CorticalConfig into processor | Arch | - | Medium |
| 103 | Add Priority Backlog Summary (this!) | TaskMgmt | - | Small |
| 104 | Create TASK_ARCHIVE.md | TaskMgmt | - | Medium |

### 🟡 Medium (Do This Month)

| # | Task | Category | Depends | Effort |
|---|------|----------|---------|--------|
| 91 | Create docs/README.md index | Docs | - | Small |
| 92 | Add badges to README.md | DevEx | - | Small |
| 93 | Update README with docs references | Docs | 91 | Small |
| 95 | Split processor.py into modules | Arch | 97 | Large |
| 96 | Centralize duplicate constants | CodeQual | - | Small |
| 98 | Replace print() with logging | CodeQual | - | Medium |
| 99 | Add input validation to public methods | CodeQual | - | Medium |
| 102 | Add tests for edge cases | Testing | - | Medium |
| 105 | Standardize task format | TaskMgmt | 103 | Medium |
| 107 | Add Quick Context to tasks | TaskMgmt | 105 | Medium |

### 🟢 Low (Backlog)

| # | Task | Category | Depends | Effort |
|---|------|----------|---------|--------|
| 73 | Add "Find Similar Code" command | DevEx | - | Medium |
| 74 | Add "Explain This Code" command | DevEx | - | Medium |
| 75 | Add "What Changed?" semantic diff | DevEx | - | Large |
| 76 | Add "Suggest Related Files" feature | DevEx | - | Medium |
| 78 | Add code pattern detection | DevEx | - | Large |
| 79 | Add corpus health dashboard | DevEx | - | Medium |
| 80 | Add "Learning Mode" for contributors | DevEx | - | Large |
| 100 | Implement plugin/extension registry | Arch | - | Large |
| 101 | Automate staleness tracking | Arch | - | Medium |
| 106 | Add task dependency graph | TaskMgmt | 105 | Small |
| 108 | Create task selection script | TaskMgmt | 103,105 | Medium |
| 109 | Add Recently Completed section | TaskMgmt | 104 | Small |

### ⏸️ Deferred

| # | Task | Reason |
|---|------|--------|
| 7 | Document magic numbers in gaps.py | Low priority, functional as-is |
| 42 | Add simple query language | Nice-to-have, not blocking |
| 44 | Remove deprecated feedforward_sources | Cleanup, low impact |
| 46 | Standardize return types with dataclasses | Nice-to-have |

### 🔄 In Progress

| # | Task | Started | Notes |
|---|------|---------|-------|
| 87 | Add Python code samples and showcase | 2025-12-11 | samples/*.py created |

---

## Recently Completed (Last 7 Days)

| # | Task | Completed | Notes |
|---|------|-----------|-------|
| 86 | Add semantic chunk boundaries for code | 2025-12-11 | In query.py |
| 85 | Improve test vs source ranking | 2025-12-11 | DOC_TYPE_BOOSTS |
| 84 | Add direct definition pattern search | 2025-12-11 | query.py |
| 83 | Add definition-aware boosting | 2025-12-11 | query.py |
| 82 | Add code stop words filter | 2025-12-11 | query.py |
| 81 | Fix tokenizer underscore identifiers | 2025-12-11 | tokenizer.py |
| 77 | Add interactive "Ask the Codebase" mode | 2025-12-11 | scripts/ |

*Full details in [TASK_ARCHIVE.md](TASK_ARCHIVE.md)*

---

## Pending Task Details

### 7. Document Magic Numbers

**Meta:** `status:deferred` `priority:low` `category:docs`
**Files:** `cortical/gaps.py:62,76,99`
**Effort:** Small

**Problem:** Magic numbers in gap detection lack documentation.

**Note:** Deferred - functional as-is, low priority.

---

### 42. Add Simple Query Language Support

**Meta:** `status:deferred` `priority:low` `category:feature`
**Files:** `cortical/query.py`
**Effort:** Large

**Problem:** Users must construct queries programmatically.

**Solution:** Add simple query syntax like `"neural AND networks NOT deep"`.

---

### 44. Remove Deprecated feedforward_sources

**Meta:** `status:deferred` `priority:low` `category:cleanup`
**Files:** `cortical/minicolumn.py:117`, `analysis.py:457`, `query.py:105`
**Effort:** Small

**Problem:** Deprecated field still exists in codebase.

---

### 46. Standardize Return Types with Dataclasses

**Meta:** `status:deferred` `priority:low` `category:code-quality`
**Files:** `cortical/query.py`
**Effort:** Medium

**Problem:** Query functions return tuples instead of typed dataclasses.

---

### 73. Add "Find Similar Code" Command

**Meta:** `status:pending` `priority:low` `category:devex`
**Files:** `scripts/similar_code.py` (new)
**Effort:** Medium

**Problem:** No easy way to find similar code blocks.

**Solution:** Create script using semantic fingerprinting.

---

### 74. Add "Explain This Code" Command

**Meta:** `status:pending` `priority:low` `category:devex`
**Files:** `scripts/explain_code.py` (new)
**Effort:** Medium

**Problem:** New contributors need help understanding code.

**Solution:** Script that retrieves relevant docs/comments for a code block.

---

### 75. Add "What Changed?" Semantic Diff

**Meta:** `status:pending` `priority:low` `category:devex`
**Files:** `scripts/what_changed.py` (new)
**Effort:** Large

**Problem:** Hard to understand semantic impact of changes.

**Solution:** Show how fingerprints changed between commits.

---

### 76. Add "Suggest Related Files" Feature

**Meta:** `status:pending` `priority:low` `category:devex`
**Files:** `scripts/related_files.py` (new)
**Effort:** Medium

**Problem:** When editing a file, don't know what else might need changes.

---

### 78. Add Code Pattern Detection

**Meta:** `status:pending` `priority:low` `category:devex`
**Files:** `cortical/patterns.py` (new), `scripts/find_patterns.py` (new)
**Effort:** Large

**Problem:** Hard to find common patterns (factory, singleton, etc.) in codebase.

---

### 79. Add Corpus Health Dashboard

**Meta:** `status:pending` `priority:low` `category:devex`
**Files:** `scripts/corpus_health.py` (new)
**Effort:** Medium

**Problem:** No quick overview of corpus health metrics.

---

### 80. Add "Learning Mode" for New Contributors

**Meta:** `status:pending` `priority:low` `category:devex`
**Files:** `scripts/learn_codebase.py` (new)
**Effort:** Large

**Problem:** Steep learning curve for new contributors.

**Solution:** Guided tour of codebase using semantic search.

---

### 87. Add Python Code Samples and Update Showcase

**Meta:** `status:in-progress` `priority:medium` `category:showcase`
**Files:** `showcase.py`, `samples/data_processor.py`, `samples/search_engine.py`, `samples/test_data_processor.py`
**Effort:** Medium

**Started:** 2025-12-11

**Progress:** Sample files created, showcase updates in progress.

---

### 88. Create Package Installation Files

**Meta:** `status:pending` `priority:critical` `category:devex`
**Files:** `pyproject.toml` (new), `requirements.txt` (new)
**Effort:** Small

**Quick Context:**
- README references `pip install -e .` but no setup.py exists
- Zero runtime dependencies (stdlib only)
- Dev dependency: coverage>=7.0

**Problem:** Can't install via pip - biggest adoption blocker.

**Solution:**
```toml
[project]
name = "cortical-text-processor"
version = "2.0.0"
requires-python = ">=3.9"
dependencies = []

[project.optional-dependencies]
dev = ["coverage>=7.0"]
```

**Acceptance Criteria:**
- [ ] `pip install -e .` works
- [ ] `pip install -e ".[dev]"` installs coverage
- [ ] README installation instructions work

---

### 89. Create CONTRIBUTING.md

**Meta:** `status:pending` `priority:high` `category:devex`
**Files:** `CONTRIBUTING.md` (new)
**Effort:** Small
**Depends:** 88

**Quick Context:**
- No contribution guide exists
- Testing: `python -m unittest discover -s tests -v`
- Code style: PEP 8, type hints, Google docstrings
- Reference: `docs/code-of-ethics.md`, `docs/definition-of-done.md`

**Problem:** Contributors don't know how to help.

**Solution:** Create CONTRIBUTING.md with fork/PR workflow, test instructions, style guide.

**Acceptance Criteria:**
- [ ] Fork/clone/PR workflow documented
- [ ] Test running instructions clear
- [ ] Links to ethics and definition-of-done docs

---

### 90. Create docs/quickstart.md Tutorial

**Meta:** `status:pending` `priority:high` `category:docs`
**Files:** `docs/quickstart.md` (new)
**Effort:** Medium
**Depends:** 88

**Quick Context:**
- showcase.py is 700+ lines - too intimidating
- Need 10-minute "Hello World" path
- Core API: `CorticalTextProcessor`, `process_document()`, `compute_all()`, `find_documents_for_query()`

**Problem:** No minimal example for newcomers.

**Solution:**
1. Installation (1 min)
2. First document (2 min)
3. First search (2 min)
4. Understanding results (3 min)
5. Next steps

**Acceptance Criteria:**
- [ ] Complete example in <50 lines
- [ ] Copy-paste runnable
- [ ] Links to advanced docs

---

### 91. Create docs/README.md Index

**Meta:** `status:pending` `priority:medium` `category:docs`
**Files:** `docs/README.md` (new)
**Effort:** Small

**Problem:** 11 docs files with no navigation.

**Solution:** Create index with recommended reading order.

---

### 92. Add Badges to README.md

**Meta:** `status:pending` `priority:medium` `category:devex`
**Files:** `README.md`
**Effort:** Small

**Problem:** No visual project health indicators.

**Solution:** Add CI, coverage, Python version, license badges.

---

### 93. Update README with Documentation References

**Meta:** `status:pending` `priority:medium` `category:docs`
**Files:** `README.md`
**Effort:** Small
**Depends:** 91

**Problem:** README doesn't mention docs/ folder.

---

### 94. Split query.py into Focused Modules

**Meta:** `status:pending` `priority:high` `category:architecture`
**Files:** `cortical/query.py` (2,719 lines) → `cortical/query/` package
**Effort:** Large

**Quick Context:**
- Current: 2,719 lines, 7+ responsibilities
- Key functions: `expand_query()` (L127), `find_documents_for_query()` (L450), `find_passages_for_query()` (L890)
- Imports: `processor.py` (L15), all test files

**Problem:** Violates Single Responsibility Principle.

**Solution:**
```
cortical/query/
├── __init__.py       # Re-export public API
├── expansion.py      # expand_query, expand_query_semantic
├── search.py         # find_documents_for_query, fast_find_documents
├── passages.py       # find_passages_for_query, chunking
├── intent.py         # parse_intent_query, search_by_intent
├── definitions.py    # is_definition_query, find_definition_passages
└── ranking.py        # multi_stage_rank
```

**Acceptance Criteria:**
- [ ] No file >500 lines
- [ ] All existing tests pass
- [ ] Backward-compatible imports from `cortical.query`

---

### 95. Split processor.py into Focused Modules

**Meta:** `status:pending` `priority:medium` `category:architecture`
**Files:** `cortical/processor.py` (2,301 lines)
**Effort:** Large
**Depends:** 97

**Problem:** God Object with 100+ methods.

**Solution:** Extract DocumentManager, ComputationManager as internal classes.

---

### 96. Centralize Duplicate Constants

**Meta:** `status:pending` `priority:medium` `category:code-quality`
**Files:** `cortical/constants.py` (new), `cortical/semantics.py`, `cortical/query.py`
**Effort:** Small

**Problem:** `RELATION_WEIGHTS`, `DOC_TYPE_BOOSTS` defined in multiple places.

**Solution:** Create `cortical/constants.py` as single source of truth.

---

### 97. Integrate CorticalConfig into CorticalTextProcessor

**Meta:** `status:pending` `priority:high` `category:architecture`
**Files:** `cortical/processor.py`, `cortical/config.py`
**Effort:** Medium

**Quick Context:**
- `CorticalConfig` exists with 20+ parameters
- `CorticalTextProcessor.__init__()` doesn't accept config
- All parameters passed at call time, scattered

**Problem:** Config exists but isn't used.

**Solution:**
```python
def __init__(
    self,
    config: Optional[CorticalConfig] = None,
    tokenizer: Optional[Tokenizer] = None
):
    self.config = config or CorticalConfig()
```

**Acceptance Criteria:**
- [ ] Config parameter accepted
- [ ] Methods use config defaults
- [ ] Override still possible per-call

---

### 98. Replace print() with Logging Module

**Meta:** `status:pending` `priority:medium` `category:code-quality`
**Files:** `cortical/processor.py`, other modules
**Effort:** Medium

**Problem:** Uses print() for progress - can't configure.

**Solution:** Use `logging.getLogger(__name__)`.

---

### 99. Add Input Validation to Public Methods

**Meta:** `status:pending` `priority:medium` `category:code-quality`
**Files:** `cortical/processor.py`, `cortical/query.py`
**Effort:** Medium

**Problem:** Some methods don't validate inputs.

**Solution:** Add validation at API boundary.

---

### 100. Implement Plugin/Extension Registry

**Meta:** `status:pending` `priority:low` `category:architecture`
**Files:** `cortical/registry.py` (new), `cortical/processor.py`
**Effort:** Large

**Problem:** Can't add algorithms without modifying core.

**Solution:** Registry pattern for PageRank, clustering, expansion strategies.

---

### 101. Automate Staleness Tracking with Decorators

**Meta:** `status:pending` `priority:low` `category:architecture`
**Files:** `cortical/processor.py`
**Effort:** Medium

**Problem:** Manual `_mark_stale()` calls are error-prone.

**Solution:** `@invalidates('tfidf', 'pagerank')` decorator.

---

### 102. Add Tests for Edge Cases

**Meta:** `status:pending` `priority:medium` `category:testing`
**Files:** `tests/test_edge_cases.py` (new)
**Effort:** Medium

**Problem:** Missing tests for Unicode, large docs, malformed inputs.

---

### 103. Add Priority Backlog Summary to Top of TASK_LIST.md

**Meta:** `status:in-progress` `priority:high` `category:task-mgmt`
**Files:** `TASK_LIST.md`
**Effort:** Small

**Problem:** Must read 3500+ lines to find pending tasks.

**Solution:** Machine-parseable backlog tables at top.

**Status:** This restructure implements Task 103.

---

### 104. Create TASK_ARCHIVE.md for Completed Tasks

**Meta:** `status:completed` `priority:high` `category:task-mgmt`
**Files:** `TASK_ARCHIVE.md` (new), `TASK_LIST.md`
**Effort:** Medium

**Solution:** Archive created, TASK_LIST.md restructured.

---

### 105. Standardize Task Format with Machine-Parseable Metadata

**Meta:** `status:pending` `priority:medium` `category:task-mgmt`
**Files:** `TASK_LIST.md`
**Effort:** Medium
**Depends:** 103

**Problem:** Inconsistent task formats.

**Solution:** `**Meta:** \`status:pending\` \`priority:high\`...` format.

---

### 106. Add Task Dependency Graph

**Meta:** `status:pending` `priority:low` `category:task-mgmt`
**Files:** Section in TASK_LIST.md
**Effort:** Small
**Depends:** 105

**Problem:** Dependencies not visualized.

---

### 107. Add "Quick Context" Section to Each Task

**Meta:** `status:pending` `priority:medium` `category:task-mgmt`
**Files:** `TASK_LIST.md`
**Effort:** Medium
**Depends:** 105

**Problem:** Must search codebase to understand task context.

---

### 108. Create Task Selection Helper Script

**Meta:** `status:pending` `priority:low` `category:task-mgmt`
**Files:** `scripts/next_task.py` (new)
**Effort:** Medium
**Depends:** 103, 105

**Problem:** Selecting next task is manual work.

---

### 109. Add "Recently Completed" Section for Context

**Meta:** `status:pending` `priority:low` `category:task-mgmt`
**Files:** `TASK_LIST.md`
**Effort:** Small
**Depends:** 104

**Problem:** No session context on recent work.

**Status:** Implemented as part of this restructure.

---

## Category Index

| Category | Pending | Description |
|----------|---------|-------------|
| DevEx | 8 | Developer experience (installation, contribution) |
| Docs | 3 | Documentation improvements |
| Arch | 4 | Architecture refactoring |
| CodeQual | 3 | Code quality improvements |
| Testing | 1 | Test coverage |
| TaskMgmt | 5 | Task management system |
| Deferred | 4 | Low priority, not blocking |

---

## Notes

- **Effort estimates:** Small (<1 hour), Medium (1-4 hours), Large (1+ days)
- **Dependencies:** Complete dependent tasks first
- **Quick Context:** Key info to start task without searching
- **Archive:** Full history in [TASK_ARCHIVE.md](TASK_ARCHIVE.md)

---

*Last restructured: 2025-12-11 (Tasks 103, 104, 109 implemented)*
