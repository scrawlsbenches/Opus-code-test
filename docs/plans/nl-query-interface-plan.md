# Natural Language Query Interface Plan

*Created: 2026-01-12*

## Problem Statement

When context is lost (new session, context window reset), we need a way to ask natural language questions about the codebase and get useful answers. Currently we have:

- Trained knowledge (vocabulary associations from samples)
- Code structure (atoms for classes, functions, files with relationships)
- REFERS_TO links bridging vocabulary to code

But no unified interface that takes a question and generates a complete answer.

## Desired Behavior

**Input:**
```
python -m cortical.cognitive ask "How does code indexing work?"
```

**Output:**
```
Code indexing is handled by CodeBridge in cortical/cognitive/code_bridge.py.

Key methods:
  index_file(path)           - Index single Python file (line 89)
  index_directory(path)      - Recursively index .py files (line 142)
  create_refers_to_links()   - Bridge vocabulary to code (line 447)

Related concepts: ast, indexstats, cognitivegraph, atomtype

CLI usage:
  python -m cortical.cognitive index-code cortical/ --link-text

See also: tests/behavioral/test_code_indexing_spec.py
```

## Design Principles

1. **No "ask again"** - Generate complete answer or honestly say "I don't know"
2. **Actionable** - Include file paths, line numbers, CLI commands
3. **Grounded** - Answers come from trained knowledge + code graph, not hallucination
4. **Extensible** - Easy to add new question types

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ask "question"                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Intent Parser                                │
│  - Extract key concepts from question                           │
│  - Identify question type (how, what, where, who, why)          │
│  - Map to query strategy                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Knowledge Gatherer                             │
│  - Query vocabulary associations (similar_to)                   │
│  - Query code structure (callers_of, methods_of, defined_in)    │
│  - Query text-code bridge (code_for_word)                       │
│  - Aggregate related information                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Response Generator                             │
│  - Format gathered knowledge into natural language              │
│  - Include file:line references                                 │
│  - Add CLI commands where relevant                              │
│  - Add "See also" for related topics                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Formatted Answer                            │
└─────────────────────────────────────────────────────────────────┘
```

## Question Types and Query Strategies

| Question Pattern | Intent | Query Strategy |
|------------------|--------|----------------|
| "How does X work?" | Understand mechanism | similar_to(X) + code_for_word(X) + methods_of(X) |
| "Where is X?" | Find location | code_for_word(X) + defined_in |
| "What calls X?" | Find callers | callers_of(X) |
| "What does X call?" | Find callees | (need to add: calls query) |
| "What inherits from X?" | Find subclasses | subclasses_of(X) |
| "What is X?" | Definition | similar_to(X) + code_for_word(X) |
| "How do I use X?" | Usage examples | similar_to(X) + look for test files |

## Implementation Steps

### Step 1: Intent Parser
**File:** `cortical/cognitive/nl_query.py` (NEW)

```python
@dataclass
class QueryIntent:
    question_type: str  # how, what, where, who, why
    concepts: List[str]  # extracted key concepts
    query_strategy: List[str]  # which queries to run

def parse_intent(question: str) -> QueryIntent:
    """Parse natural language question into structured intent."""
```

### Step 2: Knowledge Gatherer
**File:** `cortical/cognitive/nl_query.py`

```python
@dataclass
class GatheredKnowledge:
    associations: List[str]  # from similar_to
    code_entities: List[Atom]  # from code_for_word
    related_files: List[str]  # file paths
    methods: List[Atom]  # from methods_of
    callers: List[Atom]  # from callers_of

def gather_knowledge(agent: CognitiveAgent, intent: QueryIntent) -> GatheredKnowledge:
    """Execute queries based on intent and aggregate results."""
```

### Step 3: Response Generator
**File:** `cortical/cognitive/nl_query.py`

```python
def generate_response(intent: QueryIntent, knowledge: GatheredKnowledge) -> str:
    """Format gathered knowledge into natural language response."""
```

### Step 4: CLI Command
**File:** `cortical/cognitive/__main__.py`

```python
ask_parser = subparsers.add_parser("ask", help="Ask a question about the codebase")
ask_parser.add_argument("question", help="Natural language question")
ask_parser.add_argument("--verbose", "-v", action="store_true")
```

### Step 5: Behavioral Tests (TDD)
**File:** `tests/behavioral/test_nl_query_spec.py` (NEW)

```python
class TestAskCommand:
    def test_how_question_returns_mechanism(self, trained_agent):
        response = ask(trained_agent, "How does code indexing work?")
        assert "CodeBridge" in response
        assert "code_bridge.py" in response

    def test_where_question_returns_location(self, trained_agent):
        response = ask(trained_agent, "Where is the query method?")
        assert "graph.py" in response
        assert "line" in response.lower()

    def test_unknown_concept_says_so(self, trained_agent):
        response = ask(trained_agent, "How does foobar work?")
        assert "don't" in response.lower() or "unknown" in response.lower()
```

## Edge Cases

1. **Unknown concept** - Concept not in vocabulary
   - Response: "I don't have information about 'X'. Try training on relevant documents."

2. **Ambiguous concept** - Multiple meanings
   - Response: Include all relevant matches, let user disambiguate

3. **No code entities** - Only vocabulary associations
   - Response: Show associations, suggest indexing code

4. **No associations** - Only code entities
   - Response: Show code locations, note no trained knowledge

## Success Criteria

1. `ask "How does code indexing work?"` returns CodeBridge with file:line
2. `ask "Where is the query method?"` returns graph.py with line number
3. `ask "What calls compute_pagerank?"` returns list of callers
4. Unknown concepts get honest "I don't know" response
5. Response time < 1 second

## File Changes Summary

| File | Change |
|------|--------|
| `tests/behavioral/test_nl_query_spec.py` | NEW - behavioral tests (FIRST) |
| `cortical/cognitive/nl_query.py` | NEW - Intent parser, gatherer, generator |
| `cortical/cognitive/__main__.py` | Add `ask` command |

## Execution Order

1. Write behavioral tests (TDD: red first)
2. Implement Intent Parser
3. Implement Knowledge Gatherer
4. Implement Response Generator
5. Add CLI command
6. Verify all tests pass
7. Test on real questions

## Open Questions

1. Should we cache responses for common questions?
2. Should we log questions to learn what users ask?
3. Should responses include confidence scores?

## Dependencies

- CognitiveAgent with trained knowledge
- CodeBridge with indexed code
- agent.query() unified interface (already implemented)
