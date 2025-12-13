# Task List: Cortical Text Processor

Active backlog for the Cortical Text Processor project. Completed tasks are archived in [TASK_ARCHIVE.md](TASK_ARCHIVE.md).

**Last Updated:** 2025-12-13
**Pending Tasks:** 13
**Completed Tasks:** 241 (see archive)

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
| 208 | Create Claude Code hooks documentation | Docs | - | Medium |
| 209 | Add session-start hook for auto-corpus indexing | DevEx | - | Small |

### 🟡 Medium (Do This Month)

| # | Task | Category | Depends | Effort |
|---|------|----------|---------|--------|
| 210 | Add pre-commit hook for auto-reindex after code changes | DevEx | - | Small |
| 212 | Create architectural roadmap diagram (Mermaid) | Docs | - | Medium |
| 133 | Implement WAL + snapshot persistence (fault-tolerant rebuild) | Arch | 132 | Large |
| 134 | Implement protobuf serialization for corpus | Arch | 132 | Medium |
| 135 | Implement chunked parallel processing for full-analysis | Arch | 132 | Large |
| 95 | Split processor.py into modules | Arch | - | Large |

### 🟢 Low (Backlog)

| # | Task | Category | Depends | Effort |
|---|------|----------|---------|--------|
| 75 | Add "What Changed?" semantic diff | DevEx | - | Large |
| 78 | Add code pattern detection | DevEx | - | Large |
| 80 | Add "Learning Mode" for contributors | DevEx | - | Large |
| 100 | Implement plugin/extension registry | Arch | - | Large |
| 130 | Expand customer service sample cluster | Samples | - | Medium |

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

**Latest completions (2025-12-13) - README Deep Analysis Update:**
- #206 README statistics updated - 161 docs, 8,789 tokens, 46,374 bigrams, 2.4M connections
- #207 MCP Server documented - 5 tools, Claude Desktop config example
- #211 Simplified facade methods documented - quick_search(), rag_retrieve(), explore()

**Previous completions (2025-12-13) - Parallel Sub-Agent Implementation:**
- #184 MCP Server for Claude Desktop - 5 tools (search, passages, expand_query, corpus_stats, add_document), 22 tests
- #73 "Find Similar Code" command - `scripts/find_similar.py` with fingerprint-based similarity
- #74 "Explain This Code" command - `scripts/explain_code.py` with concept/relation analysis
- #76 "Suggest Related Files" feature - `scripts/suggest_related.py` with import/semantic analysis
- #79 Corpus health dashboard - `scripts/corpus_health.py` with metrics and recommendations
- #99 Add input validation - `cortical/validation.py` module with decorators and validators
- #101 Automate staleness tracking - `@marks_stale`, `@marks_fresh` decorators
- #118 Function complexity annotations - O(n²) annotations on key analysis functions
- #106 Task dependency graph - `scripts/task_graph.py` with ASCII/Mermaid output
- #107 Quick Context to tasks - Added to all high/medium priority tasks in TASK_LIST.md
- #108 Create task selection script - `scripts/select_task.py` with scoring algorithm
- #117 Create debugging cookbook - `docs/debugging-cookbook.md` with 7 scenarios
- #129 Test customer service retrieval quality - `tests/behavioral/test_customer_service_quality.py`
- #131 Investigate cross-domain semantic bridges - `docs/research/cross-domain-bridges.md`
- #140 Analyze customer service cluster quality - `docs/research/customer-service-cluster-analysis.md`

**Previous completions (2025-12-13):**
- #192 Deduplicate connections storage - typed_connections is now single source of truth
- #198-205 Legacy test investigation COMPLETE - 8 tasks, 10 files reviewed
- #197 Task list validation in CI - Added validate-task-list job to workflow
- #186 Simplified facade methods - quick_search(), rag_retrieve(), explore() (23 tests)
- #196 Spectral embeddings warning - RuntimeWarning for large graphs (>5000 terms)
- Unit Test Coverage Initiative: 1,729 tests, 85% coverage, 19 modules at 90%+

---

## Pending Task Details

### 208. Create Claude Code Hooks Documentation

**Meta:** `status:pending` `priority:high` `category:docs`
**Files:** `docs/hooks.md` (new), `CLAUDE.md`
**Effort:** Medium

**Problem:** No hooks exist in `.claude/hooks/` and hooks are not documented.

**Solution:** Create documentation for Claude Code hooks:
- What hooks are and when they run
- Available hook types (session-start, pre-commit, etc.)
- Example implementations
- Integration with corpus indexing

---

### 209. Add Session-Start Hook for Auto-Corpus Indexing

**Meta:** `status:pending` `priority:high` `category:devex`
**Files:** `.claude/hooks/session_start.sh` (new)
**Effort:** Small

**Problem:** Corpus must be manually indexed before codebase search works.

**Solution:** Create session-start hook that:
- Checks if `corpus_dev.pkl` exists and is up-to-date
- Runs incremental indexing if needed
- Generates AI metadata if missing

**Example:**
```bash
#!/bin/bash
# .claude/hooks/session_start.sh
python scripts/index_codebase.py --incremental 2>/dev/null || true
```

---

### 133. Implement WAL + Snapshot Persistence (Fault-Tolerant Rebuild)

**Meta:** `status:pending` `priority:medium` `category:arch`
**Files:** `cortical/persistence.py`, `cortical/wal.py` (new)
**Effort:** Large
**Depends:** #132 (completed)

**Problem:** If `compute_all()` crashes mid-computation, all work is lost. Large corpora take minutes to rebuild from scratch.

**Solution:** Write-Ahead Logging (WAL) + periodic snapshots:
- Log each document addition to WAL file
- Save snapshots at checkpoints (every N documents or M minutes)
- On crash: load latest snapshot + replay WAL
- Similar to SQLite's WAL mode

**Quick Context:**
- Current save/load: `cortical/persistence.py::save_processor()` (line 25), `load_processor()` (line 78)
- Uses `pickle.dump()` for full state serialization (line 63)
- Add: `wal_append(operation, data)`, `wal_replay(from_snapshot)`
- Checkpoint: `save_snapshot(filepath)` periodically during `compute_all()`
- See `cortical/chunk_index.py` for append-only pattern (similar concept)

---

### 134. Implement Protobuf Serialization for Corpus

**Meta:** `status:pending` `priority:medium` `category:arch`
**Files:** `cortical/persistence.py`, `cortical/proto/` (new), `schema.proto` (new)
**Effort:** Medium
**Depends:** #132 (completed)

**Problem:** Pickle is Python-specific and fragile across versions. Can't share corpora with other languages or tools.

**Solution:** Protocol Buffers for cross-language serialization:
- Define `.proto` schema for Minicolumn, Layer, Processor state
- Add `to_proto()` and `from_proto()` methods
- Keep pickle for backward compatibility, add protobuf option
- Enable `processor.save(path, format='protobuf')`

**Quick Context:**
- Current serialization: `cortical/persistence.py::save_processor()` (line 25-76)
- State structure: `layers`, `documents`, `document_metadata`, `embeddings`, `semantic_relations`
- Minicolumn structure: `cortical/minicolumn.py::Minicolumn` (has `to_dict()/from_dict()`)
- Layer structure: `cortical/layers.py::HierarchicalLayer::to_dict()` (line ~200)
- Add `format` parameter to `save()`/`load()` methods

---

### 135. Implement Chunked Parallel Processing for Full-Analysis

**Meta:** `status:pending` `priority:medium` `category:arch`
**Files:** `cortical/processor.py`, `cortical/analysis.py`
**Effort:** Large
**Depends:** #132 (completed)

**Problem:** `compute_all()` processes entire corpus serially. For 10,000+ document corpora, this takes 10+ minutes.

**Solution:** Parallelize independent computations:
- Split TF-IDF computation across document chunks
- Parallelize PageRank iterations (graph partitioning)
- Use `multiprocessing.Pool` for CPU-bound tasks
- Add `parallel=True` option to `compute_all(parallel=True, workers=4)`

**Quick Context:**
- Entry point: `cortical/processor.py::compute_all()` (line 636)
- Phases: TF-IDF → bigram connections → PageRank → concepts → semantics
- TF-IDF: `analysis.py::compute_tfidf()` - can split by document chunks
- PageRank: `analysis.py::compute_pagerank()` - iterative, harder to parallelize
- Bigram connections: `processor.py::compute_bigram_connections()` (line 839) - parallelizable by document
- See `tests/performance/` for timing baselines

---

### 95. Split processor.py into Modules

**Meta:** `status:pending` `priority:medium` `category:arch`
**Files:** `cortical/processor.py` → `cortical/processor/` (directory)
**Effort:** Large

**Problem:** `processor.py` is 2,301 lines and handles too many responsibilities. Hard to navigate and test.

**Solution:** Split into focused modules:
- `processor/core.py` - main CorticalTextProcessor class
- `processor/documents.py` - document processing methods
- `processor/computation.py` - compute_all, staleness tracking
- `processor/query.py` - search/retrieval wrappers
- `processor/export.py` - export/visualization methods

**Quick Context:**
- Current file: `cortical/processor.py` (2,301 lines)
- Sections: __init__ → document processing → computation → queries → export
- Key class: `CorticalTextProcessor` (line ~40)
- Public API: ~60 methods, most are wrappers calling other modules
- Staleness tracking: `_stale_computations` set, `COMP_*` constants
- Keep public API unchanged - only internal reorganization

---

### 99. Add Input Validation to Public Methods

**Meta:** `status:pending` `priority:medium` `category:codequal`
**Files:** `cortical/processor.py`, `cortical/query/*.py`
**Effort:** Medium

**Problem:** Some public methods don't validate inputs, leading to confusing errors deep in call stack.

**Solution:** Add validation to all public methods:
- Type checks: `isinstance(doc_id, str)`, `isinstance(top_n, int)`
- Range checks: `top_n > 0`, `alpha in [0, 1]`
- Empty checks: `doc_id.strip()`, `query.strip()`
- Raise `ValueError` with clear messages

**Quick Context:**
- Example pattern: `cortical/processor.py::process_document()` (lines 98-103)
  ```python
  if not isinstance(doc_id, str) or not doc_id:
      raise ValueError("doc_id must be a non-empty string")
  if not isinstance(content, str):
      raise ValueError("content must be a string")
  ```
- Apply to: `find_documents_for_query()`, `expand_query()`, `compute_importance()`, etc.
- Check all methods with user-facing parameters
- Add tests in `tests/unit/test_validation.py` (new file)

---

### 107. Add Quick Context to Tasks

**Meta:** `status:pending` `priority:medium` `category:taskmgmt`
**Files:** `TASK_LIST.md`
**Effort:** Medium

**Problem:** Starting a task requires extensive code exploration to find entry points, key files, and patterns.

**Solution:** Add "Quick Context" section to high/medium priority tasks with:
- Entry point files and line numbers
- Key methods/functions to understand
- Related patterns in codebase
- Relevant test files

**Quick Context:**
- Target tasks: #184, #133, #134, #135, #95, #99 (high/medium priority)
- Format: See task #184 above for example
- Use `grep`, search_codebase.py, and .ai_meta files to find context
- Keep context brief (3-5 bullet points)

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
| Arch | 5 | Architecture refactoring (#133, 134, 135, 95, 100) |
| DevEx | 3 | Developer experience, scripts (#75, 78, 80) |
| Samples | 1 | Sample document improvements (#130) |

*Updated 2025-12-13 - 15 tasks completed via parallel sub-agents*

---

## Notes

- **Effort estimates:** Small (<1 hour), Medium (1-4 hours), Large (1+ days)
- **Dependencies:** Complete dependent tasks first
- **Quick Context:** Key info to start task without searching
- **Archive:** Full history in [TASK_ARCHIVE.md](TASK_ARCHIVE.md)

---

*Last restructured: 2025-12-13 (Major cleanup: removed 2,001 lines of stale completed task details)*
