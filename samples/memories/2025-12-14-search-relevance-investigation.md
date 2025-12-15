# Memory Entry: 2025-12-14 Search Relevance Investigation

**Tags:** `search`, `tfidf`, `query-expansion`, `code-search`, `relevance`
**Related:** [[CLAUDE.md]], [[cortical/query/expansion.py]], [[cortical/query/search.py]]

## Problem Statement

During dog-fooding, searching for "security test fuzzing" returned staleness tests instead of actual security-related code. The search seems to weight common terms like "test" too heavily, and query expansion pulls in unrelated terms.

## Root Cause Analysis

### 1. Code Stop Words Not Filtered in Query Expansion

**Issue:** Ubiquitous programming tokens dominate lateral connections and query expansion.

**Evidence:**
- `cortical/query/expansion.py:93` - `filter_code_stop_words` defaults to `False`
- `cortical/query/search.py:54-59` - `find_documents_for_query()` doesn't pass `filter_code_stop_words` to expansion
- Testing shows "def" gets expansion weight of 1.201 (higher than original query terms!)
- Common terms like "self", "return", "def", "pass" appear in almost every Python function

**Impact:**
```python
# Query: "security test fuzzing"
# Expansion adds:
#   def: 1.201        # ❌ Too high! Not relevant
#   staleness: 0.207  # ❌ Pulled in via "test" lateral connections
#   tracked: 0.119    # ❌ Also from "test" connections
```

**Code Location:**
- `cortical/tokenizer.py:16-25` - `CODE_EXPANSION_STOP_WORDS` defined but not used by default
- `cortical/query/expansion.py:199-203` - Filtering logic exists but gated by parameter

### 2. Over-Expansion of Common Terms

**Issue:** Terms like "test" appear in many documents, creating strong lateral connections to unrelated terms.

**How it happens:**
1. "test" appears in `test_staleness.py`, `test_security.py`, `test_performance.py`, etc.
2. Each file has different topic-specific terms (staleness, security, performance)
3. Lateral connections link "test" → "staleness", "test" → "security", "test" → "performance"
4. When searching for "security test", expansion adds "staleness" and other test-related noise

**Code Location:**
- `cortical/query/expansion.py:150-168` - Lateral connection expansion (Method 1)
- Line 164: `score = weight * neighbor.pagerank * 0.6` - doesn't account for term ubiquity

**TF-IDF should help but doesn't:**
- TF-IDF correctly penalizes "test" in individual documents
- But lateral connections are based on co-occurrence count, not TF-IDF
- So ubiquitous terms still get strong lateral connections

### 3. Test File Penalty Not Applied in Basic Search

**Issue:** Test files are penalized in `ranking.py` but not in `search.py`.

**Evidence:**
- `cortical/query/ranking.py:87-90` - Test file penalty of 0.8 exists
- `cortical/query/search.py:25-112` - `find_documents_for_query()` doesn't use doc_type boosting
- `cortical/constants.py:60-65` - `DOC_TYPE_BOOSTS` defines test penalty

**Workaround exists but not used by default:**
- `ranking.py:124-177` - `find_documents_with_boost()` applies penalties
- But most code uses `find_documents_for_query()` directly

### 4. No Security-Specific Concept Expansion

**Issue:** Code concepts are defined for common operations (get/fetch/load) but not for domain-specific terms like security.

**Evidence:**
- `cortical/code_concepts.py:36-40` - Auth concept group exists with generic terms
- Line 38: `'auth', 'authentication', 'login', 'logout', 'credentials'...`
- But no 'security' concept group with terms like: fuzzing, validation, sanitize, injection, xss, csrf

**Impact:**
- "security fuzzing" doesn't expand to related security terms
- "staleness" lateral connections get equal weight to "fuzzing" connections
- No domain knowledge to prioritize security context

## Specific Code Locations Needing Attention

### High Priority

1. **`cortical/query/search.py:25-112` - `find_documents_for_query()`**
   - Line 54-59: Add `filter_code_stop_words=True` parameter
   - Should filter ubiquitous code tokens from expansion

2. **`cortical/query/expansion.py:150-168` - Lateral expansion scoring**
   - Line 164: Incorporate term IDF into scoring: `score = weight * neighbor.pagerank * neighbor.tfidf * 0.6`
   - Penalize ubiquitous terms that connect to everything

3. **`cortical/query/search.py` - Default to test file penalty**
   - Detect test files and apply 0.8 penalty by default
   - Or route to `find_documents_with_boost()` instead

### Medium Priority

4. **`cortical/code_concepts.py:16-131` - Add security concept group**
   - Add fuzzing, validation, sanitize, injection, exploit, vulnerability, etc.
   - Enable domain-specific expansion

5. **`cortical/tokenizer.py:158-221` - Add code-specific stop words**
   - Consider adding "def", "class", "return" to DEFAULT_STOP_WORDS for code corpora
   - Or create a `code_stop_words` set that's automatically used for code files

### Low Priority

6. **`cortical/analysis.py:883-924` - TF-IDF computation**
   - Currently correct, but lateral connections don't use TF-IDF
   - Could add IDF-weighted lateral connections in `compute_bigram_connections()`

## Recommended Fixes (Prioritized)

### Fix 1: Enable Code Stop Word Filtering by Default (EASY)

**What:** Set `filter_code_stop_words=True` by default in code search functions.

**Where:** `cortical/query/search.py:54-59`

**Change:**
```python
# Before
query_terms = get_expanded_query_terms(
    query_text, layers, tokenizer,
    use_expansion=use_expansion,
    semantic_relations=semantic_relations,
    use_semantic=use_semantic
)

# After
query_terms = get_expanded_query_terms(
    query_text, layers, tokenizer,
    use_expansion=use_expansion,
    semantic_relations=semantic_relations,
    use_semantic=use_semantic,
    filter_code_stop_words=True  # Filter def, self, return, etc.
)
```

**Impact:** Immediate reduction in noise from ubiquitous code tokens.

**Risk:** Low - filtering is already implemented and tested.

### Fix 2: Weight Lateral Connections by TF-IDF (MEDIUM)

**What:** Incorporate IDF into lateral expansion scoring to penalize ubiquitous terms.

**Where:** `cortical/query/expansion.py:164`

**Change:**
```python
# Before
score = weight * neighbor.pagerank * 0.6

# After
# Penalize ubiquitous terms (low IDF)
idf_factor = neighbor.tfidf / (neighbor.pagerank + 0.1)  # Normalize by pagerank
score = weight * neighbor.pagerank * min(idf_factor, 1.0) * 0.6
```

**Impact:** Terms that appear everywhere (low TF-IDF) get lower expansion weights.

**Risk:** Medium - requires testing to ensure good queries aren't hurt.

### Fix 3: Apply Test File Penalty by Default (EASY)

**What:** Detect test files and apply penalty in `find_documents_for_query()`.

**Where:** `cortical/query/search.py:25-112`

**Change:**
```python
# After doc_scores calculation (around line 70)
for doc_id in list(doc_scores.keys()):
    if doc_id.startswith('tests/') or '/test_' in doc_id or doc_id.startswith('test_'):
        doc_scores[doc_id] *= 0.8  # Apply test file penalty
```

**Impact:** Test files naturally rank lower unless highly relevant.

**Risk:** Low - penalty is already defined in constants.

### Fix 4: Add Security Concept Group (EASY)

**What:** Add security-related terms to code concepts for better expansion.

**Where:** `cortical/code_concepts.py:16-131`

**Change:**
```python
CODE_CONCEPT_GROUPS = {
    # ... existing groups ...

    # Security and safety
    'security': frozenset([
        'security', 'secure', 'auth', 'authentication', 'authorize',
        'sanitize', 'validate', 'escape', 'injection', 'xss', 'csrf',
        'fuzzing', 'fuzz', 'exploit', 'vulnerability', 'attack',
        'defense', 'protect', 'encrypt', 'decrypt', 'hash', 'salt',
        'permission', 'access', 'credential', 'token', 'session'
    ]),
}
```

**Impact:** "security fuzzing" would expand to related security terms.

**Risk:** Low - additive change, doesn't affect existing queries.

## Sample Queries Demonstrating the Issue

### Query 1: "security test fuzzing"

**Expected:** Security implementation files, fuzzing utilities
**Actual:** Staleness tests ranked high due to "test" expansion to "staleness"

**Why:**
- "test" expands to "staleness", "tracked", "properly" via lateral connections
- "def" gets added with weight 1.201 (higher than query terms!)
- Test files match on many expanded terms

### Query 2: "authentication validation"

**Expected:** Auth validation code, security checks
**Actual:** Works better because "test" not in query

**Insight:** Problem is specific to queries containing common programming terms.

### Query 3: "where is PageRank computed"

**Expected:** `analysis.py` with compute_pagerank function
**Actual:** Works well because "where" triggers implementation intent detection

**Insight:** Intent detection helps, but only for certain query patterns.

## Next Steps

1. **Implement Fix 1** (filter code stop words) - immediate impact, low risk
2. **Implement Fix 3** (test file penalty) - easy win for test file noise
3. **Implement Fix 4** (security concepts) - addresses specific case
4. **Test and iterate on Fix 2** (IDF-weighted expansion) - needs careful tuning
5. **Run regression tests** on existing search quality benchmarks
6. **Consider long-term:** Domain-specific query expansion strategies

## Connections

This investigation reveals a fundamental tension in code search:
- Programming languages have ubiquitous tokens ("def", "test", "return")
- These tokens appear in almost every file, creating strong lateral connections
- TF-IDF correctly penalizes them in documents
- But lateral connections use co-occurrence count, not TF-IDF
- Result: noise dominates expansion for queries containing common terms

**Related design question:** Should lateral connections be IDF-weighted during construction, not just during scoring?

## Measurement

To validate fixes, test these queries before/after:
1. "security test fuzzing" - should rank security_fuzzer.py > test_staleness.py
2. "authentication validation" - should rank auth code > test files
3. "database connection pooling" - should rank DB code > test files
4. "error handling retry logic" - should rank implementation > tests

Success metric: Security/implementation files rank in top 3, test files rank lower unless truly relevant.
