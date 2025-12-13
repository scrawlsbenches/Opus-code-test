# Task List: Cortical Text Processor

Active backlog for the Cortical Text Processor project. Completed tasks are archived in [TASK_ARCHIVE.md](TASK_ARCHIVE.md).

**Last Updated:** 2025-12-13
**Pending Tasks:** 23
**Completed Tasks:** 215 (see archive)

**Legacy Test Cleanup:** 16 duplicated legacy tests removed, 13 remaining need investigation
- See Tasks #198-205 for legacy test investigation

**Unit Test Initiative:** ✅ COMPLETE - 85% coverage from unit tests (1,729 tests)
- 19 modules at 90%+ coverage
- See [Coverage Baseline](#unit-test-coverage-baseline) for per-module status

---

## Active Backlog

<!-- Machine-parseable format for automation -->

### 🟠 High (Do This Week)

| # | Task | Category | Depends | Effort |
|---|------|----------|---------|--------|
| 184 | Implement MCP Server for Claude Desktop integration | Integration | - | Large |

### 🟡 Medium (Do This Month)

| # | Task | Category | Depends | Effort |
|---|------|----------|---------|--------|
| 133 | Implement WAL + snapshot persistence (fault-tolerant rebuild) | Arch | 132 | Large |
| 134 | Implement protobuf serialization for corpus | Arch | 132 | Medium |
| 135 | Implement chunked parallel processing for full-analysis | Arch | 132 | Large |
| 95 | Split processor.py into modules | Arch | - | Large |
| 99 | Add input validation to public methods | CodeQual | - | Medium |
| 107 | Add Quick Context to tasks | TaskMgmt | - | Medium |
| 198 | Investigate legacy test_coverage_gaps.py (91 tests) | Testing | - | Medium |
| 199 | Investigate legacy test_cli_wrapper.py (96 tests) | Testing | - | Medium |
| 200 | Investigate legacy test_edge_cases.py (53 tests) | Testing | - | Small |
| 201 | Investigate legacy test_incremental_indexing.py (47 tests) | Testing | - | Small |
| 202 | Investigate legacy test_intent_query.py (24 tests) | Testing | - | Small |
| 203 | Investigate legacy test_behavioral.py (9 tests) | Testing | - | Small |
| 204 | Investigate legacy test_query_optimization.py (20 tests) | Testing | - | Small |
| 205 | Investigate legacy script tests (6 files, 132 tests) | Testing | - | Medium |

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
| 106 | Add task dependency graph | TaskMgmt | - | Small |
| 108 | Create task selection script | TaskMgmt | - | Medium |
| 117 | Create debugging cookbook | AINav | - | Medium |
| 118 | Add function complexity annotations | AINav | - | Small |
| 140 | Analyze customer service cluster quality | Research | 127 | Small |
| 129 | Test customer service retrieval quality | Testing | - | Small |
| 130 | Expand customer service sample cluster | Samples | - | Medium |
| 131 | Investigate cross-domain semantic bridges | Research | - | Medium |

### ⏸️ Deferred

| # | Task | Reason |
|---|------|--------|
| 110 | Add section markers to large files | Superseded by #119 (AI metadata generator) |
| 111 | Add "See Also" cross-references | Superseded by #119 (AI metadata generator) |
| 112 | Add docstring examples | Superseded by #119 (AI metadata generator) |
| 7 | Document magic numbers in gaps.py | Low priority, functional as-is |
| 42 | Add simple query language | Nice-to-have, not blocking |
| 44 | Remove deprecated feedforward_sources | Cleanup, low impact |
| 46 | Standardize return types with dataclasses | Superseded by #185 |

### 🔮 Future (Async/Advanced)

| # | Task | Category | Notes |
|---|------|----------|-------|
| 187 | Add async API support (AsyncCorticalTextProcessor) | Architecture | Enables FastAPI, async frameworks |
| 188 | Add streaming query results | Architecture | Depends on #187 |
| 189 | Add observability hooks (timing, traces, metrics) | DevEx | OpenTelemetry integration |
| 190 | Create REST API wrapper (FastAPI) | Integration | Depends on #187 |
| 191 | Add Interactive REPL mode | DevEx | `python -m cortical --interactive` |

### 🔄 In Progress

*No tasks currently in progress*

<!-- Note: Task #87 was completed 2025-12-13, moved to archive -->

---

## Recently Completed

All completed tasks are now archived in [TASK_ARCHIVE.md](TASK_ARCHIVE.md).

**Latest completions (2025-12-13):**
- #192 Deduplicate connections storage - typed_connections is now single source of truth, lateral_connections is cached property (15 tests)
- #197 Task list validation in CI - Added validate-task-list job to workflow
- #186 Simplified facade methods - quick_search(), rag_retrieve(), explore() (23 tests)
- #196 Spectral embeddings warning - RuntimeWarning for large graphs (>5000 terms)
- #193 Unify alpha validation - retrofit_embeddings() now accepts [0,1] consistently
- #194 Layer validation - Added checks for invalid layer values (0-3) in persistence/layers
- #195 Stopwords import - semantics.py now uses Tokenizer.DEFAULT_STOP_WORDS
- #148 Performance test refactor - Moved to small synthetic corpus (25 docs)
- #149 Performance test fix - Tests now use small_corpus.py fixtures
- #182 Fluent API - FluentProcessor with method chaining (44 tests)
- #183 Progress Feedback - ConsoleProgressReporter, callbacks (30 tests)
- #185 Result Dataclasses - DocumentMatch, PassageMatch, QueryResult (56 tests)
- #179 Fix definition search - line boundary fix in `find_definition_in_text()`
- #180 Fix doc-type boosting - filename pattern + empty metadata fallback
- #181 Fix query ranking - hybrid boost strategy for exact name matches
- Unit Test Coverage Initiative: 1,729 tests, 85% coverage, 19 modules at 90%+
- Tasks #159-178 (unit tests for all modules)

---

## Pending Task Details

### 184. Implement MCP Server for Claude Desktop Integration

**Meta:** `status:pending` `priority:high` `category:integration`
**Files:** `cortical/mcp_server.py` (new), `mcp_config.json` (new)
**Effort:** Large

**Problem:** AI agents must call subprocess scripts instead of native integration. Claude Desktop users can't access the processor directly.

**Solution:** Create MCP (Model Context Protocol) server with tools:
- `search(query, top_n)` → document results
- `passages(query, top_n)` → RAG passages
- `expand_query(query)` → expansion terms
- `corpus_stats()` → statistics
- `add_document(doc_id, content)` → index document

**Acceptance:**
- [ ] Works in Claude Desktop
- [ ] 5+ core tools implemented
- [ ] Documentation for installation
- [ ] Example MCP config file

---

## Legacy Test Investigation Tasks

These tasks were created during the test coverage review (2025-12-13). 16 duplicated legacy tests were removed, and 13 remaining tests need investigation to determine if they should be migrated to categorized test directories or kept as-is. Check git history for any previous migration work.

### 198. Investigate test_coverage_gaps.py (91 tests)

**Meta:** `status:pending` `priority:medium` `category:testing`
**Files:** `tests/test_coverage_gaps.py`
**Effort:** Medium

**Problem:** 91 tests covering edge cases and coverage gaps. Investigate:
1. Check git history for previous migration attempts
2. Determine if tests should move to `tests/unit/` or `tests/regression/`
3. Check for overlap with existing unit tests

---

### 199. Investigate test_cli_wrapper.py (96 tests)

**Meta:** `status:pending` `priority:medium` `category:testing`
**Files:** `tests/test_cli_wrapper.py`
**Effort:** Medium

**Problem:** 96 tests for CLI wrapper. Investigate:
1. Check git history for migration attempts
2. Determine if these belong in `tests/integration/`
3. Verify no unit test coverage exists

---

### 200. Investigate test_edge_cases.py (53 tests)

**Meta:** `status:pending` `priority:medium` `category:testing`
**Files:** `tests/test_edge_cases.py`
**Effort:** Small

**Problem:** 53 tests for edge cases (Unicode, large docs, malformed inputs). Investigate:
1. Check git history
2. Consider moving to `tests/unit/` or creating `tests/robustness/`

---

### 201. Investigate test_incremental_indexing.py (47 tests)

**Meta:** `status:pending` `priority:medium` `category:testing`
**Files:** `tests/test_incremental_indexing.py`
**Effort:** Small

**Problem:** 47 tests for incremental document operations. Investigate:
1. Check git history
2. Verify no overlap with `tests/unit/test_processor_core.py`
3. Consider moving to `tests/integration/`

---

### 202. Investigate test_intent_query.py (24 tests)

**Meta:** `status:pending` `priority:medium` `category:testing`
**Files:** `tests/test_intent_query.py`
**Effort:** Small

**Problem:** 24 tests for intent-based query parsing. Investigate:
1. Check git history
2. Check if `tests/unit/test_query.py` covers this
3. Consider moving to `tests/unit/test_query_intent.py`

---

### 203. Investigate test_behavioral.py (9 tests)

**Meta:** `status:pending` `priority:medium` `category:testing`
**Files:** `tests/test_behavioral.py`
**Effort:** Small

**Problem:** 9 behavioral/acceptance tests. Investigate:
1. Check git history
2. Move to `tests/behavioral/` if appropriate

---

### 204. Investigate test_query_optimization.py (20 tests)

**Meta:** `status:pending` `priority:medium` `category:testing`
**Files:** `tests/test_query_optimization.py`
**Effort:** Small

**Problem:** 20 tests for query performance. Investigate:
1. Check git history
2. Consider moving to `tests/performance/`

---

### 205. Investigate legacy script tests (6 files)

**Meta:** `status:pending` `priority:medium` `category:testing`
**Files:** `tests/test_analyze_louvain_resolution.py`, `tests/test_ask_codebase.py`, `tests/test_evaluate_cluster.py`, `tests/test_generate_ai_metadata.py`, `tests/test_search_codebase.py`, `tests/test_showcase.py`
**Effort:** Medium

**Problem:** 6 test files (132 tests total) for scripts and showcase. Investigate:
1. Check git history for each
2. Consider creating `tests/scripts/` directory
3. Or moving to `tests/integration/`

---

## Unit Test Coverage Baseline

✅ **Unit test coverage as of 2025-12-13 (1,729 tests, 85% overall):**

| Module | Coverage | Status | Task |
|--------|----------|--------|------|
| config.py | 100% | ✅ | #168 |
| minicolumn.py | 100% | ✅ | #162 |
| definitions.py | 100% | ✅ | #173 |
| tokenizer.py | 99% | ✅ | #159 |
| layers.py | 99% | ✅ | #161 |
| ranking.py | 99% | ✅ | #175 |
| fingerprint.py | 99% | ✅ | #163 |
| chunk_index.py | 98% | ✅ | #167 |
| code_concepts.py | 98% | ✅ | #168 |
| embeddings.py | 98% | ✅ | #160 |
| gaps.py | 98% | ✅ | #164 |
| search.py | 95% | ✅ | #171 |
| persistence.py | 94% | ✅ | #178 |
| expansion.py | 94% | ✅ | #170 |
| analysis.py | 94% | ✅ | #176 |
| passages.py | 92% | ✅ | #172 |
| semantics.py | 91% | ✅ | #177 |
| chunking.py | 91% | ✅ | #172 |
| analogy.py | 90% | ✅ | #174 |
| intent.py | 87% | 🔶 | - |
| processor.py | 85% | 🔶 | #165-166 |

**19 of 21 modules at 90%+ coverage**

---

## Category Index

| Category | Pending | Description |
|----------|---------|-------------|
| Arch | 5 | Architecture refactoring (#133, 134, 135, 95, 100, 101) |
| CodeQual | 1 | Code quality improvements (#99) |
| Testing | 9 | Test coverage and legacy investigation (#129, 198-205) |
| TaskMgmt | 3 | Task management system (#106, 107, 108) |
| AINav | 2 | AI assistant navigation (#117, 118) |
| DevEx | 7 | Developer experience, scripts (#73-80) |
| Research | 2 | Research and analysis (#140, 131) |
| Samples | 1 | Sample document improvements (#130) |
| Integration | 1 | MCP Server (#184) |

*Updated 2025-12-13 - Unit test initiative COMPLETE (85% coverage, 1,729 tests)*

---

## Notes

- **Effort estimates:** Small (<1 hour), Medium (1-4 hours), Large (1+ days)
- **Dependencies:** Complete dependent tasks first
- **Quick Context:** Key info to start task without searching
- **Archive:** Full history in [TASK_ARCHIVE.md](TASK_ARCHIVE.md)

---

*Last restructured: 2025-12-13 (Major cleanup: removed 2,001 lines of stale completed task details)*
