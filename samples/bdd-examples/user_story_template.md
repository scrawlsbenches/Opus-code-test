# User Story Template

Use this template when creating new behavioral scenarios.

---

## The User Story Format

```
As a [role/persona],
I want [capability/feature],
So that [benefit/value].
```

### Components

| Component | Purpose | Example |
|-----------|---------|---------|
| **Role** | Who benefits | researcher, developer, analyst |
| **Capability** | What they need | search by concept, index documents |
| **Benefit** | Why it matters | find insights faster, save time |

---

## Story Examples

### Example 1: Search Discovery

```
As a researcher with a vast document collection,
I want to search using natural concepts,
So that I discover insights I didn't know to look for.
```

### Example 2: Performance Reliability

```
As a developer using the search API,
I want consistent sub-100ms response times,
So that my application remains responsive under load.
```

### Example 3: Fault Tolerance

```
As a system operator,
I want the system to recover gracefully from failures,
So that users experience minimal disruption.
```

---

## From Story to Scenarios

Each story expands into multiple scenarios:

```
STORY: As a researcher, I want to search using concepts...

SCENARIOS:
  ├── Scenario: Finding documents by concept, not just keywords
  ├── Scenario: Discovering related documents across domains
  ├── Scenario: Handling ambiguous queries gracefully
  └── Scenario: Search remains fast even with large corpus
```

---

## Scenario Template

```python
def scenario_<observable_behavior>(self, fixture):
    """
    Scenario: <Human-readable scenario title>

    Given <initial context>
    And <additional context if needed>
    When <action taken>
    Then <observable outcome>
    And <additional outcomes if needed>
    Because <why this matters>.
    """
    # Given
    <setup code>

    # When
    <action code>

    # Then
    assert <expected outcome>, "Helpful failure message"
```

---

## Complete Example

```python
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

        Given a corpus with documents about 'custom ML algorithms'
        And documents about 'hand-built statistical inference'
        When I search for 'AI prediction methods'
        Then I find documents from both domains
        Because the system understands conceptual relationships.
        """
        # Given
        corpus.add("ml_regression.md",
            "Custom machine learning regression models we built from scratch "
            "for prediction tasks using our own gradient descent implementation.")
        corpus.add("stats_bayes.md",
            "Hand-built Bayesian statistical inference engine enables prediction "
            "using probability distributions we implemented ourselves.")

        # When
        results = corpus.search("AI prediction methods")

        # Then
        found_ids = {r.doc_id for r in results}
        assert "ml_regression.md" in found_ids, (
            "Should find ML document via conceptual relationship"
        )
        assert "stats_bayes.md" in found_ids, (
            "Should find statistics document via conceptual relationship"
        )
```

---

## Acceptance Criteria Checklist

Before your story is complete, verify:

```
[ ] Story clearly states role, capability, and benefit
[ ] At least 3 scenarios cover the happy path
[ ] Edge cases have dedicated scenarios
[ ] Error conditions are explicitly tested
[ ] Performance expectations are documented (if applicable)
[ ] All scenarios follow Given-When-Then format
[ ] Assertion messages explain what went wrong
[ ] Scenarios run in isolation (no shared mutable state)
```

---

## Anti-Patterns to Avoid

### 1. Implementation-Focused Stories

```
# WRONG: Describes implementation
As a developer,
I want to use TF-IDF scoring,
So that search is better.

# RIGHT: Describes behavior
As a researcher,
I want rare terms to rank higher than common ones,
So that specific searches return precise results.
```

### 2. Missing the "So That"

```
# WRONG: No benefit stated
As a user,
I want to search documents.

# RIGHT: Clear benefit
As a user,
I want to search documents by topic,
So that I find relevant information without reading everything.
```

### 3. Vague Outcomes

```
# WRONG: Vague
Then the search works better

# RIGHT: Specific and measurable
Then documents mentioning the exact query term rank in the top 3
```

---

## File Naming Convention

Stories are grouped by role/feature:

```
tests/behavioral/
├── researcher_searches_corpus.py      # Search-related stories
├── developer_indexes_codebase.py      # Indexing stories
├── analyst_discovers_patterns.py      # Pattern discovery stories
└── system_handles_failures.py         # Reliability stories
```

---

*"If you can't write the scenario, you don't understand the requirement."*
