# Forensic Analysis: cortical/query/ Package

## Executive Summary

The cortical/query/ package was split from a monolithic 2,771-line file into 8 focused modules on **December 12, 2025** (commit 7cbe9d55). While the split improved organization, it introduced **three critical bugs within 24 hours** and revealed **significant code duplication** that should be refactored.

---

## 1. Git History Analysis

### Timeline of File Creation

**Split Event: December 12, 2025 00:42:35 UTC (commit 7cbe9d55)**

Original structure:
- `cortical/query.py` (2,771 lines) → **DELETED**

New structure (8 modules + package init):
- `cortical/query/__init__.py` (176 lines) - Re-exports for backward compatibility
- `cortical/query/analogy.py` (330 lines) - Analogy completion and relations
- `cortical/query/chunking.py` (335 lines) - Text chunking functions
- `cortical/query/definitions.py` (375 lines) - Definition search and boosting
- `cortical/query/expansion.py` (483 lines) - Query expansion (lateral, semantic, multihop)
- `cortical/query/intent.py` (220 lines) - Intent-based query parsing
- `cortical/query/passages.py` (407 lines) - Passage retrieval for RAG
- `cortical/query/ranking.py` (472 lines) - Multi-stage ranking and doc type boosting
- `cortical/query/search.py` (571 lines) - Document search functions

**Total: 3,369 lines** (598 lines added for module structure and separation)

### Post-Split Bug Fixes

| Date | Commit | Issue | Modules Affected |
|------|--------|-------|------------------|
| Dec 13, 2025 | ef9fe796 | **Three critical bugs** #179, #180, #181 | definitions.py, ranking.py, search.py |
| Dec 15, 2025 | fecd6dc4 | Query expansion weight domination | expansion.py |
| Dec 13, 2025 | 6cd9f595 | Misleading hardcoded values | (multiple) |

**Critical Bug #179**: Definition search showing wrong content
- Root cause: `find_definition_in_text()` used character offset instead of line boundaries
- Impact: Passages didn't start at the definition line

**Critical Bug #180**: Doc-type boosting showing identical results
- Root cause: `get_doc_type_boost()` only checked path, not filename for 'test'
- Impact: Test files weren't properly de-prioritized

**Critical Bug #181**: Query ranking for exact doc matches
- Root cause: Missing hybrid boosting strategy
- Impact: Documents with exact name matches didn't rank higher

### Refactoring Quality Assessment

**Positive:**
- Modules are well-sized (220-571 lines, target was <500)
- Clear separation of concerns
- Backward compatibility maintained via `__init__.py` re-exports
- All 1075 tests passed after the split

**Negative:**
- **Three critical bugs introduced within 24 hours**
- Significant code duplication (see Section 2)
- Inconsistent patterns across modules (see Section 3)

---

## 2. Code Duplication Patterns

### Pattern 1: TF-IDF Scoring (HIGH PRIORITY)

**Duplicated 10+ times across 6 modules:**

```python
# This exact pattern appears in:
# - search.py (lines 131, 251, 294, 496)
# - chunking.py (lines 287, 330)
# - intent.py (line 209)
# - passages.py (line 290)
# - ranking.py (lines 308, 447)

tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
doc_scores[doc_id] += tfidf * term_weight
```

**Recommendation:**
- Extract to shared helper function in a `query/utils.py` module:
  ```python
  def score_term_for_doc(col, doc_id, term_weight):
      tfidf = col.tfidf_per_doc.get(doc_id, col.tfidf)
      return tfidf * term_weight
  ```

### Pattern 2: Test File Detection (HIGH PRIORITY)

**Implemented THREE different ways:**

**Implementation 1: definitions.py (lines 282-310)**
- Has dedicated `is_test_file()` function
- Checks: `/tests/`, `/test/`, `test_*`, `*_test.py`, `mock`, `fixture`
- **Most comprehensive**

**Implementation 2: search.py (lines 143-147)**
- Inline code in `find_documents_for_query()`
- Checks: `tests/`, `test_*`, `/test_*/`, `/tests/`
- **Incomplete** - missing fixture/mock detection

**Implementation 3: ranking.py (line 87-88)**
- Inline code in `get_doc_type_boost()`
- Checks: `tests/` OR `'test' in doc_id.lower()`
- **Least precise** - could match "latest.py"

**Recommendation:**
- Consolidate on `definitions.is_test_file()` - move to `query/utils.py`
- Replace all inline checks with function call
- Add tests for edge cases (e.g., "latest.py" should not match)

### Pattern 3: defaultdict Imports

**Used in 5 modules:**
- expansion.py, intent.py, passages.py, ranking.py, search.py

This is acceptable (stdlib import), not a duplication issue.

### Pattern 4: Document Name Boosting

**Private function in search.py only:**
- `_apply_document_name_boost()` (lines 25-78)
- Complex logic: exact match = additive boost, partial = multiplicative
- Could be useful in other modules but currently isolated

**Recommendation:**
- Make public and move to `query/utils.py` if reused elsewhere
- Otherwise, keep as private implementation detail

---

## 3. Inconsistent Patterns

### Issue 1: Parameter Validation

**Only 2 modules validate parameters:**

**chunking.py (lines 58-62):**
```python
if chunk_size <= 0:
    raise ValueError(f"chunk_size must be positive, got {chunk_size}")
if overlap < 0:
    raise ValueError(f"overlap must be non-negative, got {overlap}")
if overlap >= chunk_size:
    raise ValueError(f"overlap must be less than chunk_size, ...")
```

**search.py (lines 465-467):**
```python
if not (0.0 <= pagerank_weight <= 1.0):
    raise ValueError(f"pagerank_weight must be in [0.0, 1.0], got {pagerank_weight}")
if not (0.0 <= proximity_weight <= 1.0):
    raise ValueError(f"proximity_weight must be in [0.0, 1.0], got {proximity_weight}")
```

**Other modules lack validation:**
- expansion.py: No validation for `max_expansions`, `tfidf_weight`, etc.
- ranking.py: No validation for boost factors
- passages.py: No validation for `chunk_size`, `overlap`, etc.

**Recommendation:**
- Add consistent parameter validation across all modules
- Consider validation utilities in `query/utils.py`

### Issue 2: Cross-Module Dependencies

**passages.py imports from 4 other query modules:**
```python
from .expansion import get_expanded_query_terms
from .search import find_documents_for_query
from .definitions import find_definition_passages, DEFINITION_BOOST
from .ranking import get_doc_type_boost, is_conceptual_query
from .chunking import (create_chunks, create_code_aware_chunks, ...)
```

This creates tight coupling. While functionally correct, it means:
- passages.py depends on 4 other modules
- Changes to expansion, search, definitions, or ranking could break passages
- Circular dependency risk (currently avoided)

**Current dependency graph:**
```
passages.py → expansion.py
           → search.py
           → definitions.py
           → ranking.py
           → chunking.py

ranking.py → expansion.py
           → search.py

search.py → expansion.py
```

**Recommendation:**
- Document these dependencies clearly
- Consider if passages.py should be the "orchestrator" module
- Add module-level docstrings explaining the dependency hierarchy

---

## 4. Cross-Reference with processor/query_api.py

### Architecture Verification

**processor/query_api.py (768 lines)** is a **wrapper/mixin** that delegates to query modules.

Pattern:
```python
class QueryMixin:
    def expand_query(self, query_text, ...):
        # Validation
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError("query_text must be a non-empty string")

        # Delegation to query module
        return query_module.expand_query(
            query_text, self.layers, self.tokenizer, ...
        )
```

**Findings:**
- ✅ No code duplication between query/ and processor/
- ✅ Clean separation: processor/ does validation + delegation
- ✅ All query logic lives in query/ package
- ✅ Correct architectural pattern

---

## 5. Potential Issues

### Issue 1: No Dead Code Found

✅ All 43 functions across the 9 files appear to be actively used.
✅ No TODO, FIXME, XXX, or HACK comments found.

### Issue 2: Module Size Balance

| Module | Lines | Functions | Ratio |
|--------|-------|-----------|-------|
| search.py | 571 | 8 | 71.4 lines/fn |
| expansion.py | 483 | 5 | 96.6 lines/fn |
| ranking.py | 472 | 7 | 67.4 lines/fn |
| passages.py | 407 | 3 | 135.7 lines/fn |
| definitions.py | 375 | 7 | 53.6 lines/fn |
| chunking.py | 335 | 7 | 47.9 lines/fn |
| analogy.py | 330 | 4 | 82.5 lines/fn |
| intent.py | 220 | 2 | 110.0 lines/fn |

**passages.py has the highest lines/function ratio** (135.7), indicating potentially complex functions.

### Issue 3: Import Overhead

All query modules import from `..layers`, `..tokenizer`, etc.
- This is expected and necessary
- No circular imports detected

---

## 6. Recommendations

### High Priority

1. **Extract TF-IDF Scoring Helper** (Effort: 2 hours)
   - Create `query/utils.py` with shared scoring functions
   - Replace 10+ duplicate TF-IDF patterns
   - Add unit tests

2. **Consolidate Test File Detection** (Effort: 1 hour)
   - Move `is_test_file()` from definitions.py to utils.py
   - Replace inline checks in search.py and ranking.py
   - Add comprehensive test coverage

3. **Add Parameter Validation** (Effort: 3 hours)
   - Add validation to expansion.py, ranking.py, passages.py
   - Ensure consistency with chunking.py and search.py patterns
   - Add negative test cases

### Medium Priority

4. **Document Module Dependencies** (Effort: 1 hour)
   - Add dependency graph to query/__init__.py docstring
   - Clarify that passages.py is the "orchestrator"
   - Document import order requirements

5. **Review passages.py Function Complexity** (Effort: 2 hours)
   - `find_passages_for_query()` is 135+ lines
   - Consider breaking into smaller helper functions
   - Improve readability

6. **Add Module-Level Tests** (Effort: 4 hours)
   - Ensure each module has comprehensive test coverage
   - Add integration tests for cross-module interactions
   - Regression tests for bugs #179, #180, #181 (already added)

### Low Priority

7. **Consider Making Document Name Boost Public** (Effort: 1 hour)
   - If other modules need name-based boosting
   - Currently private to search.py

8. **Evaluate Module Boundaries** (Effort: 2 hours)
   - Is passages.py in the right place given its dependencies?
   - Should it live in processor/ instead?

---

## 7. Conclusion

### Summary

The query package split was **structurally sound** but suffered from:
- **Quality control issues** (3 critical bugs within 24 hours)
- **Code duplication** (10+ instances of TF-IDF scoring)
- **Inconsistent patterns** (test file detection, parameter validation)

### Risk Assessment

**Current Risk: MEDIUM**
- Bugs have been fixed
- No known critical issues remain
- Code duplication increases maintenance burden

**Future Risk: MEDIUM-HIGH**
- Changes to scoring logic require updates in 6 files
- Test file detection inconsistencies could cause ranking bugs
- Tight coupling between modules increases fragility

### Recommended Action

**Phase 1 (Immediate - 6 hours):**
1. Extract TF-IDF scoring helper
2. Consolidate test file detection
3. Add parameter validation

**Phase 2 (Next sprint - 7 hours):**
4. Document dependencies
5. Review passages.py complexity
6. Add module-level tests

**Total effort: ~13 hours**

This refactoring would:
- ✅ Eliminate code duplication
- ✅ Reduce bug surface area
- ✅ Improve maintainability
- ✅ Establish consistent patterns

---

**Analysis Date:** 2025-12-22
**Analyzer:** Claude (Forensic Code Analysis)
**Codebase:** Cortical Text Processor v1.0
