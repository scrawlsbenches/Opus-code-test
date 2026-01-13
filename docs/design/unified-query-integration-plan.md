# Integration Plan: Cognitive Ask + Audit Framework

*Created: 2026-01-13*
*Status: Planning*

## Current State Analysis

**What We Have:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Cognitive Agent (nl_query.py)                                   │
│ ├── parse_intent() → question type, concepts, strategy          │
│ ├── ToolRegistry → similar_to, callers_of, methods_of           │
│ └── Word graph → SIMILARITY + FOLLOWS links                     │
│                                                                 │
│ Audit Framework (audits/reasoning.py)                           │
│ ├── translate_audit_query() → AuditQuery (intent, filters)      │
│ ├── AuditReasoner → PLN rules + truth values                    │
│ └── abstraction_to_rule() → WovenMind → PLN rules               │
│                                                                 │
│ CDG Query (cdg/query/)                                          │
│ ├── Parser → SQL-like AST                                       │
│ ├── QueryExecutor → indexed retrieval                           │
│ └── FunctionRegistry → blockers(), depends_on()                 │
│                                                                 │
│ Query Intent (query/intent.py)                                  │
│ ├── parse_intent_query() → action, subject, intent type         │
│ └── Term expansion with synonyms                                │
└─────────────────────────────────────────────────────────────────┘
```

**The Gap:**
These systems don't talk to each other. The cognitive `ask` only uses word similarity.

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     UNIFIED ASK PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Question                                                  │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────┐                                            │
│  │ Query Router    │ ← Determines which backend to use          │
│  └────────┬────────┘                                            │
│           │                                                     │
│     ┌─────┼─────┬─────────┬─────────┐                          │
│     ▼     ▼     ▼         ▼         ▼                          │
│  ┌─────┐ ┌───┐ ┌───────┐ ┌───────┐ ┌────────┐                  │
│  │ CDG │ │PLN│ │ Audit │ │ Code  │ │ Graph  │                  │
│  │Query│ │   │ │Reason │ │Intent │ │Assoc.  │                  │
│  └──┬──┘ └─┬─┘ └───┬───┘ └───┬───┘ └───┬────┘                  │
│     │      │       │         │         │                        │
│     └──────┴───────┴────┬────┴─────────┘                        │
│                         ▼                                       │
│                  ┌─────────────┐                                │
│                  │  Aggregator │ ← Combines results             │
│                  └──────┬──────┘                                │
│                         ▼                                       │
│                  ┌─────────────┐                                │
│                  │  Formatter  │ ← Natural language response    │
│                  └─────────────┘                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 1: Query Router (Foundation)

**Goal:** Create a unified entry point that routes to the right backend.

**New File:** `cortical/cognitive/unified_query.py`

```python
@dataclass
class UnifiedQuery:
    """Unified query representation."""
    raw_question: str
    query_type: str  # cdg, audit, code, semantic
    parsed: Union[CDGQuery, AuditQuery, ParsedIntent, QueryIntent]
    confidence: float

class QueryRouter:
    """Routes questions to appropriate backends."""

    def route(self, question: str) -> UnifiedQuery:
        # 1. Try CDG query pattern (FROM ... WHERE ...)
        if self._looks_like_cdg_query(question):
            return UnifiedQuery(question, "cdg", parse_cdg(question), 0.9)

        # 2. Try audit query pattern (risky files, why is X flagged)
        audit_query = translate_audit_query(question)
        if audit_query.intent != "list" or audit_query.min_risk > 0:
            return UnifiedQuery(question, "audit", audit_query, 0.8)

        # 3. Try code intent (where do we handle X)
        intent = parse_intent_query(question)
        if intent['action'] or intent['intent'] != 'definition':
            return UnifiedQuery(question, "code", intent, 0.7)

        # 4. Fall back to semantic graph query
        return UnifiedQuery(question, "semantic", self._parse_semantic(question), 0.5)
```

**Tasks:**
- [ ] Create `UnifiedQuery` dataclass
- [ ] Create `QueryRouter` class
- [ ] Integrate existing parsers
- [ ] Add tests for routing logic

## Phase 2: Backend Executors

**Goal:** Each query type has a dedicated executor.

| Executor | Input | Output | Backend |
|----------|-------|--------|---------|
| `CDGExecutor` | CDGQuery | List[Entity] | CDGStore + indexes |
| `AuditExecutor` | AuditQuery | List[FileRisk] | AuditReasoner + PLN |
| `CodeExecutor` | ParsedIntent | List[CodeEntity] | CodeBridge + graph |
| `SemanticExecutor` | QueryIntent | List[Association] | TextBridge + graph |

**New Interface:**
```python
class QueryExecutorProtocol(Protocol):
    def execute(self, query: Any) -> ExecutionResult: ...
    def format_result(self, result: ExecutionResult) -> str: ...
```

**Tasks:**
- [ ] Define `QueryExecutorProtocol`
- [ ] Wrap existing `QueryExecutor` for CDG
- [ ] Wrap `AuditReasoner` for audit queries
- [ ] Create `CodeExecutor` using `CodeBridge`
- [ ] Refactor current `ask` into `SemanticExecutor`

## Phase 3: Result Aggregation

**Goal:** Combine results from multiple backends when appropriate.

```python
@dataclass
class ExecutionResult:
    """Result from a query executor."""
    items: List[Any]
    confidence: float
    source: str  # which executor
    explanation: Optional[str] = None

class ResultAggregator:
    """Combines results from multiple executors."""

    def aggregate(self, results: List[ExecutionResult]) -> AggregatedResult:
        # Deduplicate, rank by confidence, merge explanations
        ...
```

**Tasks:**
- [ ] Define `ExecutionResult` dataclass
- [ ] Create `ResultAggregator`
- [ ] Handle deduplication across sources
- [ ] Rank results by confidence

## Phase 4: Response Formatter

**Goal:** Generate natural language responses from structured results.

```python
class ResponseFormatter:
    """Formats execution results as natural language."""

    def format(self, query: UnifiedQuery, result: AggregatedResult) -> str:
        if query.query_type == "audit":
            return self._format_audit_response(result)
        elif query.query_type == "cdg":
            return self._format_cdg_response(result)
        # ...
```

**Tasks:**
- [ ] Create `ResponseFormatter`
- [ ] Template responses by query type
- [ ] Include explanations from PLN inference
- [ ] Handle "why" questions with trace output

## Phase 5: Integration & Testing

**Goal:** Wire everything into cognitive `ask`.

**Modified `ask` function:**
```python
def ask(self, question: str) -> str:
    # Route the question
    unified = self.router.route(question)

    # Execute against appropriate backend(s)
    executor = self.executors[unified.query_type]
    result = executor.execute(unified.parsed)

    # Format response
    return self.formatter.format(unified, result)
```

**Tasks:**
- [ ] Update `NLQuery.ask()` to use new pipeline
- [ ] Maintain backward compatibility
- [ ] Add integration tests
- [ ] Update CLI to use unified query

## File Changes Summary

| File | Change |
|------|--------|
| `cortical/cognitive/unified_query.py` | **NEW** - Router, UnifiedQuery |
| `cortical/cognitive/executors/` | **NEW** - CDG, Audit, Code, Semantic executors |
| `cortical/cognitive/aggregator.py` | **NEW** - Result aggregation |
| `cortical/cognitive/formatter.py` | **NEW** - Response formatting |
| `cortical/cognitive/nl_query.py` | **MODIFY** - Wire in unified pipeline |
| `cortical/cognitive/cli.py` | **MODIFY** - Update `ask` command |

## Testing Strategy

1. **Unit tests** for each executor
2. **Integration tests** for router → executor → formatter pipeline
3. **Behavioral tests** for natural language queries:
   - "FROM task WHERE status = 'pending'" → CDG executor
   - "risky files in cortical/" → Audit executor
   - "where do we handle transactions" → Code executor
   - "what is cognitive agent" → Semantic executor

## Questions to Resolve

1. **Should we run multiple executors in parallel?** (e.g., semantic + code for broader coverage)
2. **How do we handle confidence thresholds?** (when to fall back to another executor)
3. **Do we need caching?** (same question shouldn't re-execute)

---

## Implementation Log

*Track progress here as phases are completed.*

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| Phase 1 | Complete | 2026-01-13 | QueryRouter with 16 tests |
| Phase 2 | Complete | 2026-01-13 | AuditExecutor, SemanticExecutor, CodeExecutor, CDGExecutor with 29 tests |
| Phase 3 | Complete | 2026-01-13 | ResultAggregator with 38 tests (merge, best, weighted strategies) |
| Phase 4 | Pending | | |
| Phase 5 | Pending | | |
