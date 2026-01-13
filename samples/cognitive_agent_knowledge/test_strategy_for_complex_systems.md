# Test Strategy for Complex Systems

A guide for AI agents on testing complex systems effectively. Based on the Cortical project's battle-tested test architecture.

## The Test Pyramid

Complex systems need tests at multiple levels. Each level has a different purpose, speed, and scope.

```
                    /\
                   /  \        Performance/E2E (~5 min)
                  /----\       Integration (~2 min)
                 /------\      Behavioral (~2 min)
                /--------\     Unit (~30 sec)
               /----------\    Smoke (~1 sec)
              /____________\
```

### 1. Smoke Tests (Gate 1)

**Purpose:** Does the system fundamentally work?

**Location:** `tests/smoke/`

**Characteristics:**
- Execute in under 10 seconds total
- Catch critical breakage early
- Test imports, basic instantiation, core workflows

**What to test:**
- Core modules import successfully
- Main classes can be instantiated
- Basic happy-path workflow completes
- Persistence (save/load) works

**Example from this codebase:**

```python
class TestBasicWorkflow:
    """Verify the basic processing workflow works."""

    def test_process_single_document(self):
        """Single document can be processed."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        stats = processor.process_document("test", "Hello world test document.")

        assert stats['tokens'] > 0
        assert "test" in processor.documents

    def test_compute_all_completes(self):
        """compute_all() completes without error."""
        from cortical import CorticalTextProcessor

        processor = CorticalTextProcessor()
        processor.process_document("test", "Test document for computation.")
        processor.compute_all(verbose=False)

        from cortical import CorticalLayer
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        assert layer0.column_count() > 0
```

**When smoke tests fail:** Stop everything. Fix them before investigating other failures. A broken smoke test means the system is fundamentally broken.

### 2. Unit Tests (Gate 2)

**Purpose:** Does each component work correctly in isolation?

**Location:** `tests/unit/`

**Characteristics:**
- Fast (under 1 second each)
- Test single functions, methods, or classes
- Use mocks for external dependencies
- High coverage of edge cases

**What to test:**
- Individual functions with various inputs
- Edge cases (empty, null, boundary values)
- Error handling paths
- State transitions

**Example from this codebase:**

```python
class TestSplitIdentifier:
    """Tests for split_identifier function."""

    def test_empty_string(self):
        """Empty string returns empty list."""
        result = split_identifier("")
        assert result == []

    def test_camelcase(self):
        """camelCase splits into components."""
        result = split_identifier("getUserCredentials")
        assert result == ["get", "user", "credentials"]

    def test_snake_case(self):
        """snake_case splits on underscores."""
        result = split_identifier("get_user_data")
        assert result == ["get", "user", "data"]

    def test_acronym_at_start(self):
        """Acronym at start: XMLParser -> ['xml', 'parser']."""
        result = split_identifier("XMLParser")
        assert result == ["xml", "parser"]
```

### 3. Behavioral Tests (Gate 3)

**Purpose:** Do user stories work end-to-end?

**Location:** `tests/behavioral/`

**Characteristics:**
- Test from the user's perspective
- Written in story format (Given/When/Then)
- Test workflows, not individual functions
- May involve multiple components

**What to test:**
- User stories and acceptance criteria
- Workflow completions
- Feature integrations
- Quality attributes (accuracy, relevance)

**Example from this codebase:**

```python
class TestResearcherSearchesCorpusWithMultipleMethods:
    """
    Epic: Document Search and Discovery

    As a researcher exploring a knowledge base,
    I want flexible search methods optimized for different use cases,
    So that I find relevant documents efficiently.
    """

    def test_scenario_researcher_searches_with_query_expansion(self):
        """
        Scenario: Basic search expands query with related terms

        Given a corpus with interconnected concepts
        When I search for a term
        Then the system expands my query with related terms
        And returns documents matching both original and expanded terms
        Because query expansion improves recall without manual effort.
        """
        # GIVEN a corpus with interconnected concepts
        docs = {
            "ml_intro": "Machine learning trains models to recognize patterns in data.",
            "neural_overview": "Neural networks use layers to process information.",
            "dl_guide": "Deep learning employs multilayer architectures for learning.",
        }

        processor = CorticalTextProcessor()
        for doc_id, content in docs.items():
            processor.process_document(doc_id, content)
        processor.compute_all(verbose=False)

        # WHEN I search for a term
        results = processor.find_documents_for_query(
            "neural",
            top_n=3,
            use_expansion=True
        )

        # THEN the system returns relevant documents
        assert len(results) > 0, "Should find relevant documents"
        doc_ids = [doc_id for doc_id, _ in results]
        assert "neural_overview" in doc_ids, "Should find direct match"
```

### 4. Integration Tests (Gate 4)

**Purpose:** Do components work together correctly?

**Location:** `tests/integration/`

**Characteristics:**
- Test component interactions
- May use real dependencies (databases, files)
- Slower than unit tests, faster than E2E
- Focus on boundaries and interfaces

**What to test:**
- API integrations
- Database operations
- File I/O
- Inter-service communication
- Transaction boundaries

### 5. Performance Tests (Gate 5)

**Purpose:** Does the system meet performance requirements?

**Location:** `tests/performance/`

**Characteristics:**
- Test timing and resource usage
- Run without coverage (coverage adds 10x overhead)
- May take longer to complete
- Guard performance contracts

**What to test:**
- Response latency (p50, p95)
- Throughput limits
- Memory usage
- Index/computation time

---

## Testing for Correctness vs Testing for Regressions

### Correctness Tests

Verify the system does what it should:

```python
def test_search_returns_relevant_results(self):
    """Search returns documents matching the query."""
    results = processor.find_documents_for_query("machine learning")
    assert len(results) > 0
    # Results should be relevant
```

### Regression Tests

Prevent specific bugs from returning:

```python
class TestBigramSeparatorRegression:
    """
    Task #10 (2025-12-10): Bigram separators must be spaces, not underscores.

    Bug: Bigrams were inconsistently created with underscores ("neural_networks")
    but searched with spaces ("neural networks"), causing search failures.

    Fix: Standardized on space separators throughout.
    """

    def test_bigrams_use_space_separator(self, small_processor):
        """Bigrams should use space separators."""
        from cortical import CorticalLayer

        layer1 = small_processor.get_layer(CorticalLayer.BIGRAMS)
        bigram_contents = [col.content for col in layer1]

        # None should have underscores as separators
        underscore_bigrams = [b for b in bigram_contents if '_' in b and ' ' not in b]
        assert len(underscore_bigrams) == 0, (
            f"Found bigrams with underscore separators: {underscore_bigrams[:5]}"
        )
```

**Key pattern for regression tests:**
1. Document the task/issue number
2. Describe the bug that was fixed
3. Write a minimal test that would have caught the bug
4. Include the date the bug was fixed

---

## Performance Contracts and Benchmarks

### What is a Performance Contract?

A promise about system behavior that cannot be broken. If a contract is violated, the build fails.

```python
"""
=====================================================================
                    SEARCH PERFORMANCE CONTRACT
=====================================================================
  Ratified:     2024-12-30
  Guardian:     CI Pipeline
  Renegotiation: Requires team review + documented justification
=====================================================================

  We solemnly contract the following guarantees:

  - Search latency p50 < 50ms   for corpus <= 1,000 docs
  - Search latency p95 < 100ms  for corpus <= 1,000 docs
  - Memory usage < 50MB per 100 documents indexed
  - Index build time < 2 seconds for 100 documents
"""

@pytest.mark.contract
class TestSearchPerformanceContract:
    """
    Search Performance Contract

    These contracts are enforced on every CI run.
    Breaking a contract blocks the build. There are no exceptions.
    """

    # The sacred numbers - DO NOT CHANGE without team review
    P50_LATENCY_MS = 100
    P95_LATENCY_MS = 200
    SAMPLE_SEARCHES = 20

    def test_p50_latency_honored(self, small_processor):
        """CONTRACT: Half of all searches complete in under 50ms."""
        latencies = self._measure_searches(small_processor, n=self.SAMPLE_SEARCHES)
        p50 = percentile(latencies, 50)

        assert p50 < self.P50_LATENCY_MS, (
            f"CONTRACT VIOLATION: p50 latency is {p50:.1f}ms, "
            f"contract requires <{self.P50_LATENCY_MS}ms"
        )
```

### Contract vs Benchmark

| Contract | Benchmark |
|----------|-----------|
| Blocks build on failure | Informational only |
| Cannot be changed without review | Can be adjusted freely |
| Sacred promise to users | Internal tracking metric |
| Runs on every CI | May run periodically |

---

## Property-Based Testing for Edge Cases

Property-based testing generates random inputs to find edge cases that manual tests miss.

### Using Hypothesis

```python
from hypothesis import given, settings
from hypothesis import strategies as st

# Define strategies for generating test data
safe_text = st.text(
    alphabet=string.ascii_letters + string.digits + " .,!?'\"-",
    min_size=1,
    max_size=10000
)

doc_ids = st.text(min_size=1, max_size=100)

class TestProcessDocumentFuzzing:
    """Fuzz testing for process_document() method."""

    @given(doc_id=doc_ids, content=safe_text)
    @settings(max_examples=100)
    def test_process_document_never_crashes(self, doc_id, content):
        """process_document should never crash with valid-ish inputs."""
        processor = CorticalTextProcessor()

        # Should not raise any exception
        try:
            processor.process_document(doc_id, content)
        except ValueError:
            pass  # ValueError is acceptable for invalid inputs
        # Any other exception = test failure
```

### When to Use Property-Based Testing

- Input validation functions
- Parsers and serializers
- Mathematical operations
- State machines
- Anything that should "never crash"

---

## When Tests Are Worth the Investment

### Always Worth It

1. **Bug fixes** - Regression test prevents recurrence
2. **Core functionality** - Smoke tests for critical paths
3. **Public APIs** - Contract tests for stability
4. **Security-sensitive code** - Defense in depth

### Sometimes Worth It

1. **UI components** - When behavior is complex
2. **Configuration handling** - Edge cases matter
3. **Performance-critical paths** - When SLAs exist

### Rarely Worth It

1. **Trivial getters/setters** - Low value, high noise
2. **Framework-provided functionality** - Trust the framework
3. **One-off scripts** - Manual verification is fine

### Cost-Benefit Framework

```
Test Value = (Bug Severity * Bug Probability) / (Test Maintenance Cost)

Write the test when: Test Value > 1
```

---

## Maintaining Test Health

### Flaky Tests

**Symptoms:** Test passes sometimes, fails sometimes with no code change.

**Common causes:**
- Timing dependencies (`time.sleep()`)
- Race conditions
- External service dependencies
- Non-deterministic order

**Fixes:**
- Mock time instead of sleeping
- Add proper synchronization
- Use test doubles for external services
- Make tests deterministic

**Rule:** Never add `time.sleep()` to tests without explicit approval.

### Slow Tests

**Symptoms:** Test suite takes too long to run.

**Common causes:**
- Unnecessary setup/teardown
- Network calls
- Disk I/O
- Large data sets

**Fixes:**
- Use in-memory alternatives (fixtures)
- Share expensive setup across tests (class/session scope)
- Mock external services
- Use smaller representative datasets

**Example of good fixture design:**

```python
@pytest.fixture(scope="session")
def small_processor():
    """
    Session-scoped fixture providing a processor with small synthetic corpus.

    This is fast to create (~1s) and suitable for most tests.
    """
    from tests.fixtures.small_corpus import get_small_processor
    return get_small_processor()

@pytest.fixture
def fresh_processor():
    """
    Function-scoped fixture providing a fresh, empty processor.

    Use when tests need to modify processor state.
    """
    from cortical import CorticalTextProcessor
    return CorticalTextProcessor()
```

### Meaningless Tests

**Symptoms:** Tests pass but don't catch bugs. High coverage, low confidence.

**Common causes:**
- Testing implementation, not behavior
- Weak assertions
- Testing the wrong thing

**Signs of a good test:**
- Would fail if the feature is broken
- Tests behavior, not implementation details
- Has meaningful assertions
- Documents expected behavior

---

## TDD Workflow: Red, Green, Refactor

### Phase 1: RED

Write a failing test first. The test must:
- Be clear about what it's testing
- Fail for the right reason
- Document the expected behavior

```python
def test_empty_query_raises_value_error(self, processor):
    """Empty string query should raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        processor.find_documents_for_query("", top_n=5)

    assert "non-empty" in str(exc_info.value).lower()
```

### Phase 2: GREEN

Write the minimal code to make the test pass. No more.

```python
def find_documents_for_query(self, query: str, top_n: int = 10):
    if not query or not query.strip():
        raise ValueError("Query must be non-empty")
    # ... rest of implementation
```

### Phase 3: REFACTOR

Clean up while tests are green:
- Improve naming
- Extract duplicated code
- Simplify complex logic
- Add documentation

**Critical:** Run tests after each refactoring step.

### The TDD Mindset

1. **If you can't write the test, you don't understand the requirement**
2. **Tests are specifications, not afterthoughts**
3. **Green tests give you permission to refactor**
4. **Red tests tell you what to work on next**

---

## Test Organization in This Codebase

```
tests/
    smoke/              # Gate 1: Quick sanity (~1s)
    unit/               # Gate 2: Unit specs (~30s)
        got/            # GoT-specific unit tests
        cdg/            # CDG-specific unit tests
        algorithms/     # Algorithm tests
        specifications/ # Formal specs
    behavioral/         # Gate 3: User stories (~2m)
    integration/        # Gate 4: Component integration (~2m)
    performance/        # Gate 5: Performance contracts (~3m)
        contracts/      # Sacred promises
    regression/         # Bug-specific tests
    security/           # Security validation
        test_fuzzing.py # Property-based tests
    fixtures/           # Shared test data and setup
    conftest.py         # Pytest configuration and fixtures
```

---

## Summary: Test Strategy Checklist

Before shipping any change:

1. [ ] Smoke tests pass (`pytest tests/smoke/ -v`)
2. [ ] Unit tests pass (`pytest tests/unit/ -v`)
3. [ ] Behavioral tests pass (`pytest tests/behavioral/ -v`)
4. [ ] Integration tests pass (`pytest tests/integration/ -v`)
5. [ ] Performance contracts honored (`pytest tests/performance/contracts/ -v`)
6. [ ] Added regression test if fixing a bug
7. [ ] Coverage didn't drop below threshold

When writing new tests:

1. [ ] Test is at the appropriate level (unit/integration/behavioral)
2. [ ] Test has clear documentation
3. [ ] Test uses shared fixtures where appropriate
4. [ ] Test is deterministic (no flakiness)
5. [ ] Test completes quickly (no unnecessary sleeps)
6. [ ] Test will fail if the feature breaks

---

*This document captures hard-won lessons about testing complex systems. When in doubt, write the test.*
