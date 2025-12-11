# Task List: Cortical Text Processor

Active backlog for the Cortical Text Processor project. Completed tasks are archived in [TASK_ARCHIVE.md](TASK_ARCHIVE.md).

**Last Updated:** 2025-12-11
**Pending Tasks:** 25
**Completed Tasks:** 86+ (see archive)

---

## Active Backlog

<!-- Machine-parseable format for automation -->

### 🔴 Critical (Do Now)

*No critical tasks at this time.*

### 🟠 High (Do This Week)

| # | Task | Category | Depends | Effort |
|---|------|----------|---------|--------|
| 94 | Split query.py into focused modules | Arch | - | Large |
| 97 | Integrate CorticalConfig into processor | Arch | - | Medium |

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
| 107 | Add Quick Context to tasks | TaskMgmt | - | Medium |
| 113 | Document staleness tracking system | AINav | - | Small |
| 114 | Add type aliases for complex types | AINav | - | Small |
| 115 | Create component interaction diagram | AINav | - | Medium |
| 116 | Document return value semantics | AINav | - | Medium |

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

### ⏸️ Deferred

| # | Task | Reason |
|---|------|--------|
| 110 | Add section markers to large files | Superseded by #119 (AI metadata generator) |
| 111 | Add "See Also" cross-references | Superseded by #119 (AI metadata generator) |
| 112 | Add docstring examples | Superseded by #119 (AI metadata generator) |
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
| 122 | Investigate Concept Layer & Embeddings regressions | 2025-12-11 | Fixed inverted strictness, improved embeddings |
| 119 | Create AI metadata generator script | 2025-12-11 | scripts/generate_ai_metadata.py with tests |
| 120 | Add AI metadata loader to Claude skills | 2025-12-11 | ai-metadata skill created |
| 121 | Auto-regenerate AI metadata on changes | 2025-12-11 | Documented in CLAUDE.md, skills |
| 88 | Create package installation files | 2025-12-11 | pyproject.toml, requirements.txt |
| 89 | Create CONTRIBUTING.md | 2025-12-11 | Contribution guide |
| 90 | Create docs/quickstart.md | 2025-12-11 | 5-minute tutorial |
| 103 | Add Priority Backlog Summary | 2025-12-11 | TASK_LIST.md restructure |
| 104 | Create TASK_ARCHIVE.md | 2025-12-11 | 75+ tasks archived |
| 105 | Standardize task format | 2025-12-11 | Meta tags, effort estimates |
| 109 | Add Recently Completed section | 2025-12-11 | Session context |
| 86 | Add semantic chunk boundaries for code | 2025-12-11 | In query.py |
| 85 | Improve test vs source ranking | 2025-12-11 | DOC_TYPE_BOOSTS |

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

### 88. Create Package Installation Files ✓

**Meta:** `status:completed` `priority:critical` `category:devex`
**Files:** `pyproject.toml`, `requirements.txt`, `cortical/py.typed`
**Completed:** 2025-12-11

**Solution Applied:**
- Created `pyproject.toml` with modern Python packaging
- Created `requirements.txt` documenting dev dependencies
- Added `py.typed` marker for PEP 561 compliance
- Tested: `pip install -e .` and `pip install -e ".[dev]"` both work

---

### 89. Create CONTRIBUTING.md ✓

**Meta:** `status:completed` `priority:high` `category:devex`
**Files:** `CONTRIBUTING.md`
**Completed:** 2025-12-11

**Solution Applied:**
- Created comprehensive CONTRIBUTING.md with:
  - Quick start (8 steps from fork to PR)
  - Development setup instructions
  - Test running commands
  - Code style guidelines with examples
  - Project structure overview
  - Links to code-of-ethics.md and definition-of-done.md

---

### 90. Create docs/quickstart.md Tutorial ✓

**Meta:** `status:completed` `priority:high` `category:docs`
**Files:** `docs/quickstart.md`
**Completed:** 2025-12-11

**Solution Applied:**
- Created 5-minute quickstart tutorial with:
  - Installation instructions
  - "Your First 10 Lines" minimal example
  - Understanding results section
  - Query expansion demo
  - Passage retrieval for RAG
  - Key concepts table
  - Save/load instructions
  - Common patterns (batch, incremental, metadata)
  - Troubleshooting section

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

### 103. Add Priority Backlog Summary to Top of TASK_LIST.md ✓

**Meta:** `status:completed` `priority:high` `category:task-mgmt`
**Files:** `TASK_LIST.md`
**Completed:** 2025-12-11

**Solution Applied:** Added machine-parseable backlog with priority tiers (Critical/High/Medium/Low/Deferred) at top of file.

---

### 104. Create TASK_ARCHIVE.md for Completed Tasks ✓

**Meta:** `status:completed` `priority:high` `category:task-mgmt`
**Files:** `TASK_ARCHIVE.md`, `TASK_LIST.md`
**Completed:** 2025-12-11

**Solution Applied:** Created archive with 75+ completed tasks, reduced TASK_LIST.md from 3,565 to 604 lines (83% reduction).

---

### 105. Standardize Task Format with Machine-Parseable Metadata ✓

**Meta:** `status:completed` `priority:medium` `category:task-mgmt`
**Files:** `TASK_LIST.md`
**Completed:** 2025-12-11

**Solution Applied:** All tasks now use `**Meta:** \`status:...\` \`priority:...\`` format with effort estimates.

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

### 109. Add "Recently Completed" Section for Context ✓

**Meta:** `status:completed` `priority:low` `category:task-mgmt`
**Files:** `TASK_LIST.md`
**Completed:** 2025-12-11

**Solution Applied:** Added "Recently Completed (Last 7 Days)" section with table format showing task, date, and notes.

---

### 110. Add Section Markers to Large Files

**Meta:** `status:pending` `priority:high` `category:ai-nav`
**Files:** `cortical/processor.py`, `cortical/query.py`
**Effort:** Small

**Problem:** Large files (processor.py: 2,301 lines, query.py: 2,719 lines) are hard to navigate. AI assistants must scan large portions to find relevant sections.

**Solution:** Add clear section markers like:
```python
# =============================================================================
# DOCUMENT MANAGEMENT
# =============================================================================

# =============================================================================
# COMPUTATION METHODS
# =============================================================================
```

**Acceptance Criteria:**
- [ ] processor.py has 5-7 logical sections marked
- [ ] query.py has 5-7 logical sections marked
- [ ] Section names match CLAUDE.md terminology

---

### 111. Add "See Also" Cross-References to Docstrings

**Meta:** `status:pending` `priority:high` `category:ai-nav`
**Files:** `cortical/processor.py`, `cortical/query.py`, `cortical/analysis.py`
**Effort:** Medium

**Problem:** When reading a function, AI assistants don't know about related functions without searching.

**Solution:** Add "See Also" sections to docstrings:
```python
def find_documents_for_query(self, query: str, top_n: int = 5):
    """
    Find documents matching query.

    See Also:
        fast_find_documents: Faster search, document-level only
        find_passages_for_query: Chunk-level retrieval for RAG
        expand_query: Get expanded terms before searching
    """
```

**Target Functions:** Top 20 most-used public methods.

---

### 112. Add Docstring Examples for Complex Functions

**Meta:** `status:pending` `priority:high` `category:ai-nav`
**Files:** `cortical/query.py`, `cortical/analysis.py`, `cortical/processor.py`
**Effort:** Medium

**Problem:** Complex functions lack examples showing expected input/output.

**Solution:** Add Examples section to docstrings:
```python
def expand_query(self, query: str, max_expansions: int = 10):
    """
    Expand query with related terms.

    Example:
        >>> processor.expand_query("neural networks")
        {'neural': 1.0, 'networks': 1.0, 'network': 0.85,
         'learning': 0.72, 'deep': 0.68}
    """
```

**Target Functions:**
- `expand_query()`, `expand_query_semantic()`, `expand_query_multihop()`
- `find_documents_for_query()`, `find_passages_for_query()`
- `parse_intent_query()`, `search_by_intent()`
- `complete_analogy()`
- `compute_pagerank()`, `compute_tfidf()`

---

### 113. Document Staleness Tracking System

**Meta:** `status:pending` `priority:medium` `category:ai-nav`
**Files:** `CLAUDE.md` or `docs/staleness.md` (new)
**Effort:** Small

**Problem:** The staleness tracking system (`COMP_TFIDF`, `COMP_PAGERANK`, `is_stale()`, `_mark_all_stale()`) is powerful but not documented. AI assistants discover it through exploration.

**Solution:** Add documentation explaining:
- What staleness means and why it matters
- List of all `COMP_*` constants and what they track
- When staleness is automatically set (which methods call `_mark_all_stale()`)
- How to check and resolve staleness
- Example workflow showing stale → recompute → fresh

---

### 114. Add Type Aliases for Complex Types

**Meta:** `status:pending` `priority:medium` `category:ai-nav`
**Files:** `cortical/types.py` (new), update imports in other modules
**Effort:** Small

**Problem:** Complex return types like `List[Tuple[str, float, Dict[str, Any]]]` are hard to understand at a glance.

**Solution:** Create type aliases:
```python
# cortical/types.py
from typing import List, Tuple, Dict, Any

# Query results
DocumentScore = Tuple[str, float]  # (doc_id, score)
DocumentResults = List[DocumentScore]

PassageResult = Tuple[str, float, str]  # (doc_id, score, passage_text)
PassageResults = List[PassageResult]

IntentResult = Tuple[str, float, Dict[str, Any]]  # (doc_id, score, intent_info)
IntentResults = List[IntentResult]

# Graph types
ConnectionMap = Dict[str, float]  # {target_id: weight}
LayerDict = Dict[CorticalLayer, HierarchicalLayer]
```

---

### 115. Create Component Interaction Diagram

**Meta:** `status:pending` `priority:medium` `category:ai-nav`
**Files:** `docs/architecture.md` or `CLAUDE.md`
**Effort:** Medium

**Problem:** Understanding how modules call each other requires tracing imports and function calls.

**Solution:** Add ASCII or Mermaid diagram showing:
```
┌─────────────┐
│ processor.py│ ← Public API entry point
└──────┬──────┘
       │ calls
       ▼
┌──────────────────────────────────────────────┐
│ analysis.py │ query.py │ semantics.py │ ...  │
└──────────────────────────────────────────────┘
       │ operates on
       ▼
┌──────────────────────────────────────────────┐
│      layers.py  →  minicolumn.py             │
└──────────────────────────────────────────────┘
```

Include which module calls which, and data flow direction.

---

### 116. Document Return Value Semantics

**Meta:** `status:pending` `priority:medium` `category:ai-nav`
**Files:** `CLAUDE.md`
**Effort:** Medium

**Problem:** Inconsistent understanding of what functions return in edge cases (empty corpus, no matches, invalid input).

**Solution:** Add section to CLAUDE.md documenting:

| Scenario | Return | Example Functions |
|----------|--------|-------------------|
| Empty corpus | Empty list `[]` | `find_documents_for_query()` |
| No matches | Empty list `[]` | `find_passages_for_query()` |
| Invalid doc_id | `None` | `get_document_metadata()` |
| Invalid layer | Raises `KeyError` | `get_layer()` |

Also document:
- When functions return `None` vs raise exceptions
- Default values for optional parameters
- Score ranges (0.0-1.0 vs unbounded)

---

### 117. Create Debugging Cookbook

**Meta:** `status:pending` `priority:low` `category:ai-nav`
**Files:** `docs/debugging.md` (new)
**Effort:** Medium

**Problem:** Common debugging scenarios require discovering patterns through trial and error.

**Solution:** Create cookbook with scenarios:

1. **"Why is my query returning no results?"**
   - Check if corpus has documents
   - Check if terms exist in corpus
   - Use `--expand` to see what's being searched

2. **"Why are PageRank values all zero?"**
   - Check if `compute_all()` was called
   - Check staleness with `is_stale()`

3. **"Why is search slow?"**
   - Use `fast_find_documents()` for document-level
   - Pre-build index with `build_search_index()`

4. **"Why are bigrams not connecting?"**
   - Verify space separator (not underscore)
   - Check `compute_bigram_connections()` was called

---

### 118. Add Function Complexity Annotations

**Meta:** `status:pending` `priority:low` `category:ai-nav`
**Files:** `cortical/processor.py`, `cortical/analysis.py`
**Effort:** Small

**Problem:** AI assistants don't know which functions are expensive to call.

**Solution:** Add complexity notes to expensive functions:
```python
def compute_all(self, verbose: bool = True):
    """
    Compute all network properties.

    Complexity: O(n²) where n = total minicolumns across all layers.
    Typical time: 2-5 seconds for 10K documents.

    Note: For incremental updates, prefer add_document_incremental()
    which is O(m) where m = tokens in new document.
    """
```

**Target Functions:**
- `compute_all()` - O(n²)
- `compute_pagerank()` - O(iterations × edges)
- `build_concept_clusters()` - O(n × iterations)
- `find_passages_for_query()` - O(docs × chunks)

---

### 119. Create AI Metadata Generator Script ✅

**Meta:** `status:completed` `priority:high` `category:ai-nav`
**Files:** `scripts/generate_ai_metadata.py`, `tests/test_generate_ai_metadata.py`
**Effort:** Medium
**Completed:** 2025-12-11

**Problem:** AI navigation tasks (110-118) require modifying code files directly, cluttering them for human readers. We need a way to provide rich AI navigation aids without polluting the source code.

**Solution:** Generate companion `.ai_meta` files (YAML) that provide:
- Section markers with line ranges
- Function cross-references ("See Also")
- Complexity annotations
- Return value semantics
- Docstring examples (extracted or generated)
- Import/dependency information

**Output format:**
```yaml
# processor.py.ai_meta - Auto-generated
file: cortical/processor.py
lines: 2301
generated: 2025-12-11T14:30:00

sections:
  - name: "Document Management"
    lines: [54, 520]
    functions: [process_document, add_document_incremental, remove_document]
  - name: "Computation Methods"
    lines: [613, 1140]
    functions: [compute_all, compute_importance, compute_tfidf]

functions:
  find_documents_for_query:
    line: 1596
    signature: "(query: str, top_n: int = 5) -> List[Tuple[str, float]]"
    see_also:
      fast_find_documents: "~2-3x faster, document-level only"
      find_passages_for_query: "Chunk-level retrieval for RAG"
    complexity: "O(terms × documents)"
    returns:
      on_empty_corpus: "[]"
      on_no_matches: "[]"

dependencies:
  imports: [analysis, query, semantics]
  imported_by: [__init__]
```

**Acceptance Criteria:**
- [ ] Script generates .ai_meta for all cortical/*.py files
- [ ] *.ai_meta added to .gitignore
- [ ] Output is valid YAML (AI-parseable)
- [ ] Extracts sections by analyzing class/function groupings
- [ ] Extracts function signatures and line numbers
- [ ] Identifies related functions by naming patterns
- [ ] Can be run incrementally (only changed files)

---

### 120. Add AI Metadata Loader to Claude Skills ✅

**Meta:** `status:completed` `priority:high` `category:ai-nav`
**Files:** `.claude/skills/ai-metadata/SKILL.md`, `.claude/skills/corpus-indexer/SKILL.md`
**Effort:** Small
**Depends:** 119
**Completed:** 2025-12-11

**Problem:** Generated .ai_meta files need to be used by AI assistants during code navigation.

**Solution:** Created new `ai-metadata` skill that:
1. Provides structured documentation for using .ai_meta files
2. Explains metadata fields (sections, see_also, complexity hints)
3. Shows best practices for AI agent navigation
4. Updated corpus-indexer skill to include metadata generation commands

---

### 121. Auto-regenerate AI Metadata on File Changes ✅

**Meta:** `status:completed` `priority:high` `category:ai-nav`
**Files:** `CLAUDE.md`, `.claude/skills/corpus-indexer/SKILL.md`
**Effort:** Medium
**Depends:** 119
**Completed:** 2025-12-11

**Problem:** .ai_meta files become stale when source files change.

**Solution implemented:**
1. **Documented in CLAUDE.md** - AI Agent Onboarding section with startup command
2. **Incremental mode** - `python scripts/generate_ai_metadata.py --incremental` only updates changed files
3. **Combined workflow** - `python scripts/index_codebase.py --incremental && python scripts/generate_ai_metadata.py --incremental`
4. **Skills documentation** - corpus-indexer skill explains metadata regeneration

**Recommended workflow for new agents:**
```bash
# On arrival, check if metadata exists and regenerate if needed
ls cortical/*.ai_meta || python scripts/generate_ai_metadata.py
```

---

### 122. Investigate Concept Layer & Embeddings Regressions ✅

**Meta:** `status:completed` `priority:critical` `category:bugfix`
**Files:** `cortical/analysis.py`, `cortical/embeddings.py`, `showcase.py`
**Effort:** Medium
**Completed:** 2025-12-11

**Problem:** Showcase output reveals potential regressions or bugs:

1. **Concept Layer has only 3 clusters** for 95 documents
2. **Graph embeddings show nonsensical similarities** - "neural" similar to "blockchain"

**Root Cause Analysis:**

1. **Clustering strictness logic was inverted** (introduced in Task #4):
   - Bug: `change_threshold = (1.0 - cluster_strictness) * 0.3` meant high strictness → easy label changes
   - Fix: Changed to `change_threshold = cluster_strictness * 0.3` so high strictness → resist changes
   - Also fixed bonus logic that had the same inversion

2. **Adjacency embeddings were sparse** for large graphs:
   - Bug: Only captured direct connections to landmark nodes, resulting in mostly-zero vectors
   - Fix: Added multi-hop propagation to reach landmarks through neighbors
   - Also changed showcase.py to use `random_walk` method (better semantic results)

3. **Concept cluster count** (3-5 for 95 docs) is actually correct behavior:
   - The corpus is highly connected (avg 18.2 connections per token)
   - Label propagation correctly merges connected tokens into large clusters
   - One giant cluster (6656 tokens) with a few small clusters is expected

**Solution Applied:**

1. Fixed `cluster_strictness` logic in `analysis.py:585` and `analysis.py:601-603`
2. Improved `_adjacency_embeddings()` with multi-hop propagation in `embeddings.py`
3. Changed showcase.py to use `method='random_walk'` for embeddings
4. Added 3 regression tests:
   - `test_cluster_strictness_direction` - ensures higher strictness → more clusters
   - `test_random_walk_semantic_similarity` - ensures "neural" similar to "networks"
   - `test_adjacency_produces_nonzero_embeddings` - ensures dense embeddings

**Results After Fix:**
- Embeddings now show "neural" similar to "networks" (0.938), "learn" (0.928) ✅
- Clustering direction now matches documentation ✅
- All 820 tests pass ✅

**Acceptance Criteria:**
- [x] Root cause identified via git history
- [x] Embedding similarities semantically meaningful
- [x] Regression test added to prevent recurrence
- [~] Concept clusters > 10: Not achievable due to highly connected corpus (correct behavior)

---

## Category Index

| Category | Pending | Description |
|----------|---------|-------------|
| BugFix | 0 | Bug fixes and regressions |
| AINav | 6 | AI assistant navigation & usability |
| DevEx | 6 | Developer experience (scripts, tools) |
| Docs | 2 | Documentation improvements |
| Arch | 4 | Architecture refactoring |
| CodeQual | 3 | Code quality improvements |
| Testing | 1 | Test coverage |
| TaskMgmt | 2 | Task management system |
| Deferred | 7 | Low priority or superseded |

---

## Notes

- **Effort estimates:** Small (<1 hour), Medium (1-4 hours), Large (1+ days)
- **Dependencies:** Complete dependent tasks first
- **Quick Context:** Key info to start task without searching
- **Archive:** Full history in [TASK_ARCHIVE.md](TASK_ARCHIVE.md)

---

*Last restructured: 2025-12-11 (Tasks 88, 89, 90, 103, 104, 105, 109 implemented)*
