# BDD Writing Guide

A comprehensive guide to writing behavioral scenarios in the Metus style.

---

## The Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   Scenarios are not tests. They are executable specifications.  │
│                                                                  │
│   They document behavior for humans while proving correctness   │
│   for machines. Write them for the person who will read them    │
│   in six months, not for the interpreter that runs them today.  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

### The Epic Class

Each file contains one or more epic classes. An epic groups related scenarios.

```python
"""
<Feature Name> Behavioral Tests
================================

Epic: <One-line epic summary>

As a <role>,
I want <capability>,
So that <benefit>.

Requirements:
- <Key requirement 1>
- <Key requirement 2>
- <Key requirement 3>

Run with: pytest tests/behavioral/<filename>.py -v
"""

import pytest
from datetime import datetime


class Test<EpicName>:
    """
    Epic: <Epic summary>

    As a <role>,
    I want <capability>,
    So that <benefit>.
    """

    def test_<scenario_name>(self, fixture):
        """
        Scenario: <Scenario title>

        Given <context>
        When <action>
        Then <outcome>
        Because <rationale>.
        """
        # Given
        ...

        # When
        ...

        # Then
        assert ...
```

---

## Naming Conventions

### File Names

Pattern: `test_<feature>_<type>.py` or `<role>_<action>_stories.py`

```
# Good names
test_document_freshness.py
test_query_expansion.py
researcher_searches_corpus.py
developer_indexes_documents.py

# Avoid
test_tfidf.py           # Too technical
test_stuff.py           # Too vague
my_tests.py             # Non-descriptive
```

### Class Names

Pattern: `Test<UserAction>` or `<Role><Action>`

```python
# Good - describes user action
class TestDocumentFreshness: ...
class ResearcherSearchesForKnowledge: ...
class SystemHandlesUnexpectedFailures: ...

# Avoid - describes implementation
class TestTFIDFScoring: ...
class TestPageRankAlgorithm: ...
```

### Method Names

Pattern: `test_<observable_behavior>` or `scenario_<what_happens>`

```python
# Good - describes what user observes
def test_recent_documents_rank_higher_than_older_ones(self): ...
def test_search_returns_results_within_100_milliseconds(self): ...
def scenario_typos_in_query_still_find_correct_documents(self): ...

# Avoid - describes implementation
def test_apply_freshness_boost(self): ...
def test_tfidf_weights(self): ...
```

---

## The Given-When-Then Pattern

### Given: The Context

Establishes the initial state. Use clear, declarative statements.

```python
# Good
def test_example(self, corpus):
    # Given a corpus with technical documentation
    corpus.add("api_guide.md", "REST API design patterns for services...")
    corpus.add("db_guide.md", "Database optimization and query tuning...")

# Also good - use fixtures for complex setup
@pytest.fixture
def corpus_with_technical_docs(corpus):
    """A corpus containing technical documentation."""
    corpus.add("api_guide.md", "REST API design patterns...")
    corpus.add("db_guide.md", "Database optimization...")
    return corpus
```

### When: The Action

The single action being tested. Keep it focused.

```python
# Good - single clear action
def test_example(self, corpus):
    # When I search for API documentation
    results = corpus.search("REST API patterns")

# Avoid - multiple actions
def test_example(self, corpus):
    # When (too many things happening)
    corpus.reindex()
    results = corpus.search("REST API patterns")
    filtered = [r for r in results if r.score > 0.5]
```

### Then: The Outcome

Observable, verifiable outcomes. Use descriptive assertion messages.

```python
# Good - clear assertion with helpful message
def test_example(self, corpus):
    # Then the API guide appears in results
    result_ids = [r.doc_id for r in results]
    assert "api_guide.md" in result_ids, (
        f"Expected 'api_guide.md' in results for 'REST API patterns' query. "
        f"Got: {result_ids}"
    )

# Avoid - assertion without context
def test_example(self, corpus):
    assert "api_guide.md" in result_ids  # What if this fails?
```

### Because: The Rationale (Optional but Encouraged)

Explains why this behavior matters. Helps future readers understand intent.

```python
def test_recent_documents_rank_higher(self, corpus):
    """
    Scenario: Recent documents outrank stale documents

    Given two documents with similar content
    And one was added 2 days ago
    And another was added 30 days ago
    When I search with freshness boost enabled
    Then the recent document appears before the older one
    Because users want current information first.  # <-- The why
    """
```

---

## Assertion Best Practices

### Use Descriptive Messages

```python
# Good - explains what went wrong
assert recent_idx < old_idx, (
    f"Recent document (index {recent_idx}) should rank higher than "
    f"old document (index {old_idx}). Results: {result_docs}"
)

# Avoid - cryptic failure
assert recent_idx < old_idx
```

### Test One Behavior Per Scenario

```python
# Good - focused scenario
def test_search_finds_exact_matches(self): ...
def test_search_finds_partial_matches(self): ...
def test_search_handles_typos(self): ...

# Avoid - testing multiple behaviors
def test_search_works(self):
    # Tests exact matches
    # Also tests partial matches
    # Also tests typos
    # If one fails, which behavior is broken?
```

### Use Pytest Markers for Categories

```python
@pytest.mark.slow
def test_large_corpus_search_performance(self): ...

@pytest.mark.contract
def test_p99_latency_honored(self): ...

@pytest.mark.integration
def test_full_indexing_pipeline(self): ...
```

---

## Content Sovereignty

Test content should reflect the project's philosophy of building everything ourselves.

```python
# WRONG - references external tools
corpus.add("infra.md", "Kubernetes orchestration with Helm charts...")
corpus.add("cache.md", "Redis caching with cluster mode...")

# RIGHT - reflects sovereignty
corpus.add("infra.md",
    "Custom task orchestration engine we built from first principles. "
    "Hand-crafted scheduler with in-house load balancing."
)
corpus.add("cache.md",
    "In-house caching layer we implemented ourselves. "
    "Custom eviction policies and memory management."
)
```

---

## Common Patterns

### Testing Ranking Order

```python
def test_ranking_order(self, corpus):
    results = corpus.search("query")
    result_docs = [doc_id for doc_id, _ in results]

    # Verify specific document ranks higher
    assert result_docs.index("expected_first") < result_docs.index("expected_second"), (
        f"Expected 'expected_first' before 'expected_second'. Got: {result_docs}"
    )
```

### Testing Score Relationships

```python
def test_score_relationships(self, corpus):
    results = corpus.search("query")
    scores = {doc_id: score for doc_id, score in results}

    # Verify score relationships
    assert scores["high_relevance"] > scores["low_relevance"], (
        f"High relevance doc should score higher. "
        f"High: {scores['high_relevance']}, Low: {scores['low_relevance']}"
    )
```

### Testing Presence/Absence

```python
def test_presence_absence(self, corpus):
    results = corpus.search("specific query")
    result_ids = {r.doc_id for r in results}

    # Should find relevant documents
    assert "relevant_doc.md" in result_ids, "Relevant doc should appear"

    # Should not find irrelevant documents (or they rank very low)
    assert "completely_unrelated.md" not in result_ids, (
        "Unrelated doc should not appear in results"
    )
```

### Testing Performance

```python
import time

def test_performance(self, corpus):
    start = time.perf_counter()
    results = corpus.search("query")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 100, (
        f"Search took {elapsed_ms:.1f}ms, expected < 100ms"
    )
```

---

## Fixture Patterns

### Parameterized Scenarios

```python
@pytest.mark.parametrize("query,expected_docs", [
    ("machine learning", ["ml_guide.md", "ai_overview.md"]),
    ("database tuning", ["db_optimization.md"]),
    ("REST API", ["api_patterns.md", "web_services.md"]),
])
def test_search_finds_relevant_documents(self, corpus, query, expected_docs):
    results = corpus.search(query)
    result_ids = {r.doc_id for r in results}

    for doc in expected_docs:
        assert doc in result_ids, f"'{doc}' should appear for '{query}'"
```

### Shared Setup with Fixtures

```python
@pytest.fixture
def corpus_with_mixed_ages(fresh_processor):
    """Corpus with documents of varying ages."""
    today = datetime.now()

    fresh_processor.process_document(
        "today_doc", "Content...",
        metadata={"timestamp": today.strftime("%Y-%m-%d")}
    )
    fresh_processor.process_document(
        "week_old_doc", "Content...",
        metadata={"timestamp": (today - timedelta(days=7)).strftime("%Y-%m-%d")}
    )
    fresh_processor.compute_all(verbose=False)
    return fresh_processor
```

---

## Checklist Before Commit

```
[ ] File follows naming convention (test_<feature>.py)
[ ] Class docstring contains user story
[ ] Method docstrings contain Given-When-Then scenarios
[ ] Assertions have descriptive failure messages
[ ] Test content reflects sovereignty (no external tool references)
[ ] Tests run in isolation (no shared mutable state)
[ ] Performance-related tests use appropriate fixtures
[ ] All tests pass locally: pytest tests/behavioral/ -v
```

---

## Further Reading

- `CLAUDE.md` - Complete Metus philosophy
- `tests/behavioral/test_document_freshness.py` - Production example
- `tests/performance/contracts/` - Performance contract examples

---

*"Understanding is demonstrated through automation."* - Metus Tenet IV
