# Task List: Cortical Text Processor

Active backlog for the Cortical Text Processor project. Completed tasks are archived in [TASK_ARCHIVE.md](TASK_ARCHIVE.md).

**Last Updated:** 2025-12-13
**Pending Tasks:** 16
**Completed Tasks:** 222 (see archive)

**Legacy Test Cleanup:** ✅ COMPLETE - All 8 tasks investigated (#198-205)
- **KEEP (7 files, 506 tests):** Provide unique coverage not duplicated in unit tests
  - #198 test_coverage_gaps.py (91 tests) - edge case coverage
  - #199 test_cli_wrapper.py (96 tests) - CLI wrapper framework
  - #200 test_edge_cases.py (53 tests) - robustness tests
  - #201 test_incremental_indexing.py (47 tests) - script integration
  - #205 Script tests: 6 files (132 tests) - scripts/ directory
- **DELETED (3 files, 53 tests):** Covered by unit tests
  - #202 test_intent_query.py - covered by tests/unit/test_query.py
  - #203 test_behavioral.py - superseded by tests/behavioral/
  - #204 test_query_optimization.py - covered by tests/unit/test_query_search.py

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
| 192 | Deduplicate lateral_connections and typed_connections storage | Memory | - | Medium |

### 🟡 Medium (Do This Month)

| # | Task | Category | Depends | Effort |
|---|------|----------|---------|--------|
| 133 | Implement WAL + snapshot persistence (fault-tolerant rebuild) | Arch | 132 | Large |
| 134 | Implement protobuf serialization for corpus | Arch | 132 | Medium |
| 135 | Implement chunked parallel processing for full-analysis | Arch | 132 | Large |
| 95 | Split processor.py into modules | Arch | - | Large |
| 99 | Add input validation to public methods | CodeQual | - | Medium |
| 107 | Add Quick Context to tasks | TaskMgmt | - | Medium |

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
- #198-205 Legacy test investigation COMPLETE - 8 tasks, 10 files reviewed
  - DELETED 3 duplicate files (53 tests): test_behavioral.py, test_intent_query.py, test_query_optimization.py
  - KEPT 7 unique files (506 tests): test_coverage_gaps.py, test_cli_wrapper.py, test_edge_cases.py, test_incremental_indexing.py, + 6 script tests
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

### 192. Deduplicate lateral_connections and typed_connections storage

**Meta:** `status:pending` `priority:high` `category:memory`
**Files:** `cortical/minicolumn.py`

**Problem:**
Every typed connection is duplicated in `lateral_connections` for backward compatibility (`minicolumn.py:209-212`). For large graphs, this doubles memory for edge weights.

**Options:**
1. Deprecate `lateral_connections` in favor of `typed_connections`
2. Make `lateral_connections` a property that derives from `typed_connections`
3. Keep both but document the trade-off

**Context from code review (2025-12-13):**
- Found in comprehensive code review of core classes
- Memory concern for large corpora with millions of edges

---

## Legacy Test Investigation Tasks

✅ **COMPLETE** - All 8 legacy test investigation tasks finished (Tasks #198-205).

### 198. ✅ test_coverage_gaps.py (91 tests) - KEEP

**Meta:** `status:completed` `priority:medium` `category:testing`
**Recommendation:** KEEP - comprehensive edge case coverage

**Findings:**
- Tests edge cases for 10+ modules (semantics, processor, persistence, chunk_index, layers, query, analysis, config, embeddings, minicolumn)
- Some overlap with unit tests, but specifically targets coverage gaps
- All 91 tests pass (3.026s)
- Provides valuable edge case coverage complementing unit tests

---

### 199. ✅ test_cli_wrapper.py (96 tests) - KEEP

**Meta:** `status:completed` `priority:medium` `category:testing`
**Recommendation:** KEEP - unique coverage for cortical/cli_wrapper.py

**Findings:**
- Tests ExecutionContext, GitContext, HookRegistry, CLIWrapper, TaskCompletionManager, ContextWindowManager
- **NO unit test coverage exists** for cli_wrapper module
- All 96 tests pass
- UNIQUE and valuable - only tests for CLI wrapper framework

---

### 200. ✅ test_edge_cases.py (53 tests) - KEEP

**Meta:** `status:completed` `priority:medium` `category:testing`
**Recommendation:** KEEP - unique edge case/robustness tests not covered elsewhere

**Findings:**
- Tests Unicode/i18n handling (Arabic, Chinese, emoji, mixed scripts)
- Tests large documents (10K+ words, very long lines)
- Tests malformed inputs (empty, None, non-string)
- Tests boundary conditions (single word, repeated words)
- Tests query edge cases (special chars, Unicode, negative top_n)
- **No significant overlap** with unit tests - these are comprehensive robustness tests

---

### 201. ✅ test_incremental_indexing.py (47 tests) - KEEP (partial)

**Meta:** `status:completed` `priority:medium` `category:testing`
**Recommendation:** KEEP script-specific tests, consider removing duplicated remove_document tests

**Findings:**
- remove_document/remove_minicolumn tests: COVERED by tests/unit/test_processor_core.py and tests/unit/test_layers.py
- Script-specific tests (UNIQUE, keep):
  - Manifest operations for scripts/index_codebase.py
  - File change detection
  - Progress tracker
  - Timeout handler
  - Full/Incremental index functions

---

### 202. ✅ test_intent_query.py (24 tests) - DELETE

**Meta:** `status:completed` `priority:medium` `category:testing`
**Recommendation:** DELETE - fully covered by tests/unit/test_query.py

**Findings:**
- parse_intent_query tests: COVERED by 12+ tests in tests/unit/test_query.py
- search_by_intent tests: COVERED by tests/unit/test_processor_core.py
- QUESTION_INTENTS/ACTION_VERBS constant tests: minor, can be added to unit tests if needed

---

### 203. ✅ test_behavioral.py (9 tests) - DELETE

**Meta:** `status:completed` `priority:medium` `category:testing`
**Recommendation:** DELETE - superseded by tests/behavioral/test_behavioral.py

**Findings:**
- Legacy file uses unittest + loads real samples/ corpus (256s runtime!)
- tests/behavioral/test_behavioral.py uses pytest fixtures + synthetic corpus (<5s)
- Same test categories: SearchBehavior, QualityBehavior, RobustnessBehavior
- Categorized version has MORE tests (18 vs 9) with better coverage

---

### 204. ✅ test_query_optimization.py (20 tests) - DELETE

**Meta:** `status:completed` `priority:medium` `category:testing`
**Recommendation:** DELETE - fully covered by tests/unit/test_query_search.py

**Findings:**
- fast_find_documents: COVERED by 15+ tests in tests/unit/test_query_search.py
- build_document_index: COVERED by 7+ tests in tests/unit/test_query_search.py
- search_with_index: COVERED by 8+ tests in tests/unit/test_query_search.py

---

### 205. ✅ Script tests (6 files, 132 tests) - KEEP

**Meta:** `status:completed` `priority:medium` `category:testing`
**Recommendation:** KEEP - unique coverage for scripts/ directory

**Files:**
- test_analyze_louvain_resolution.py (23 tests)
- test_ask_codebase.py (19 tests)
- test_evaluate_cluster.py (24 tests)
- test_generate_ai_metadata.py (29 tests)
- test_search_codebase.py (24 tests)
- test_showcase.py (13 tests)

**Findings:**
- **NO unit test coverage exists** for scripts/ directory
- All 129 tests pass (10.303s)
- Tests critical developer tools: codebase indexing, search, metadata generation
- UNIQUE and valuable - only tests for scripts/

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
| Testing | 1 | Test coverage (#129) |
| TaskMgmt | 3 | Task management system (#106, 107, 108) |
| AINav | 2 | AI assistant navigation (#117, 118) |
| DevEx | 7 | Developer experience, scripts (#73-80) |
| Research | 2 | Research and analysis (#140, 131) |
| Samples | 1 | Sample document improvements (#130) |
| Integration | 1 | MCP Server (#184) |
| Memory | 1 | Optimization (#192) |

*Updated 2025-12-13 - Unit test initiative COMPLETE (85% coverage, 1,729 tests)*

---

## Notes

- **Effort estimates:** Small (<1 hour), Medium (1-4 hours), Large (1+ days)
- **Dependencies:** Complete dependent tasks first
- **Quick Context:** Key info to start task without searching
- **Archive:** Full history in [TASK_ARCHIVE.md](TASK_ARCHIVE.md)

---

*Last restructured: 2025-12-13 (Major cleanup: removed 2,001 lines of stale completed task details)*
