# CDG Query Language Design

**Date:** 2026-01-09
**Status:** Approved for Implementation
**Author:** Claude (session claude/merge-got-commands-Cngol)

---

## Executive Summary

This document describes the design for a unified query language at the CDG (Cortical Data Graph) layer. The query language will be schema-driven, support any entity type, and leverage existing CDG infrastructure (SchemaRegistry, CDGIndexManager, CDGStore).

**Key Decision:** Move the query language from `cortical/got/expression/` to `cortical/cdg/query/`, making it a core CDG capability rather than a GoT-specific feature.

---

## Problem Statement

### Current State

The query system is fragmented across multiple locations:

| Component | Location | Purpose | Limitation |
|-----------|----------|---------|------------|
| `query` command | `got/cli/query.py:28` | Natural language keywords | 15 lines of keyword matching |
| `expr` command | `got/cli/query.py:333` | Expression DSL | Hardcoded to tasks |
| `infer` command | `got/cli/query.py:280` | Git analysis | Separate from query system |
| Expression parser | `got/expression/` | Parse SQL-like syntax | Lives in wrong layer |
| Graph functions | `got/expression/functions/` | blockers(), path() | Hardcoded `list_all_tasks()` |

### Problems

1. **Wrong Layer:** Query language lives in GoT but should be in CDG
2. **Task-Centric:** Graph functions hardcode `manager.list_all_tasks()`
3. **No Index Usage:** Queries don't leverage CDGIndexManager
4. **Duplicate Commands:** `query`, `expr`, `infer` should be unified
5. **Not Extensible:** Adding new entity types requires code changes

---

## Proposed Architecture

### Directory Structure

```
cortical/cdg/query/                    # NEW: Core query engine
├── __init__.py                        # Public API: parse(), execute(), CDGQuery
├── lexer.py                           # Tokenizer (moved from got/expression/)
├── parser.py                          # Parser (moved from got/expression/)
├── ast.py                             # AST nodes (moved from got/expression/)
├── planner.py                         # NEW: Query planner
├── executor.py                        # NEW: Schema-aware executor
├── validator.py                       # Schema-aware validation
├── errors.py                          # Query-specific errors
├── registry.py                        # Function registry (moved)
└── functions/
    └── core.py                        # count(), distinct(), exists(), type_of()

cortical/got/query/                    # GoT extensions only
├── __init__.py                        # Register GoT functions
└── functions/
    ├── graph.py                       # blockers(), path(), connected_to()
    ├── git.py                         # infer()
    └── filters.py                     # in_sprint(), stale(), recent()
```

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLI Layer                                       │
│  got query "FROM task WHERE status = 'pending'"                         │
│  got query "blockers('T-123')"                                          │
│  got query "infer(commits=10)"                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       GoT Query Extensions                               │
│  cortical/got/query/                                                    │
│  ├── Registers graph functions (blockers, path, connected_to)           │
│  ├── Registers git functions (infer)                                    │
│  └── Registers filter functions (in_sprint, stale, recent)              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CDG Query Engine                                   │
│  cortical/cdg/query/                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   Lexer     │→ │   Parser    │→ │  Planner    │→ │  Executor   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
│         │                                                 │             │
│         └────────────────────┬────────────────────────────┘             │
│                              ▼                                          │
│                    FunctionRegistry                                     │
│                    (extensible)                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CDG Storage Layer                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  SchemaRegistry │  │ CDGIndexManager │  │    CDGStore     │         │
│  │  - entity types │  │  - lookup()     │  │  - read()       │         │
│  │  - field defs   │  │  - lookup_multi │  │  - iter_entities│         │
│  │  - indexed flag │  │  - distinct()   │  │  - list_by_prefix│        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Query Language Specification

### Syntax

```sql
-- Basic entity query
FROM <entity_type> [WHERE <conditions>] [ORDER BY <field> [ASC|DESC]] [LIMIT n] [OFFSET n]

-- Function call (returns results directly)
<function_name>(<args>, <kwargs>)

-- Compound queries
<query> AND <query>
<query> OR <query>
NOT <query>
```

### Entity Types (Schema-Driven)

Entity types are discovered from SchemaRegistry, not hardcoded:

```python
# Current entity types (from cortical/got/entity_schemas.py)
task             # T-*
decision         # D-*
sprint           # S-*
epic             # EPIC-*
edge             # E-*
handoff          # H-*
knowledge_transfer  # KT-*
claudemd_layer   # CML-*
claudemd_version # CMV-*
team             # TEAM-*
persona_profile  # PP-*
document         # DOC-*
```

### Operators

| Operator | Example | Index-Optimizable |
|----------|---------|-------------------|
| `=` | `status = 'pending'` | Yes |
| `!=` | `status != 'completed'` | No (full scan) |
| `IN` | `priority IN ['high', 'critical']` | Yes (lookup_multi) |
| `NOT IN` | `status NOT IN ['done']` | No (full scan) |
| `LIKE` | `title LIKE '%bug%'` | No (full scan) |
| `>`, `<`, `>=`, `<=` | `created_at > '2026-01-01'` | Future (btree index) |
| `IS NULL` | `assignee IS NULL` | Future |
| `IS NOT NULL` | `assignee IS NOT NULL` | Future |

### Core Functions (CDG Layer)

```python
# Entity introspection
type_of(entity_id)              # Returns entity type from ID prefix
exists(entity_id)               # Boolean: entity exists?
fields(entity_type)             # List fields for entity type

# Aggregation
count()                         # Count results
distinct(field)                 # Distinct values for field

# Utility
all()                           # All entities (with FROM clause)
none()                          # Empty result set
```

### GoT Extension Functions

```python
# Graph traversal
connected_to(entity_id, entity_type='task', depth=1)
path(from_id, to_id)
children(entity_id)
parents(entity_id)
ancestors(entity_id)
descendants(entity_id)
blockers(entity_id)
blocking(entity_id)

# Filters
recent(days=7)
stale(days=30)
blocked()
in_sprint(sprint_id)
unassigned()
overdue()

# Git analysis
infer(commits=10)
infer(message="commit message")
```

---

## Query Execution

### Phase 1: Parse

Convert query string to AST:

```python
query = "FROM task WHERE status = 'pending' AND priority = 'high'"

# Parsed AST
CDGQuery(
    entity_type='task',
    expression=AndExpr([
        Comparison(Field('status'), Op.EQ, Literal('pending')),
        Comparison(Field('priority'), Op.EQ, Literal('high')),
    ]),
    order_by=None,
    limit=None,
    offset=None
)
```

### Phase 2: Plan

Determine execution strategy based on schema:

```python
class QueryPlan:
    strategy: Literal['index_intersect', 'index_scan', 'full_scan']
    index_lookups: List[IndexLookup]  # Fields to use index for
    post_filter: Optional[Expression]  # Remaining conditions

# Example plan for "status = 'pending' AND priority = 'high'"
QueryPlan(
    strategy='index_intersect',
    index_lookups=[
        IndexLookup('status', 'pending'),    # Use index
        IndexLookup('priority', 'high'),     # Use index
    ],
    post_filter=None  # No additional filtering needed
)

# Example plan for "status = 'pending' AND title LIKE '%bug%'"
QueryPlan(
    strategy='index_scan',
    index_lookups=[
        IndexLookup('status', 'pending'),    # Use index
    ],
    post_filter=LikeExpr(Field('title'), '%bug%')  # Post-filter in memory
)
```

### Phase 3: Execute

Execute plan against CDG storage:

```python
def execute(plan: QueryPlan, store: CDGStore, index_mgr: CDGIndexManager) -> List[Entity]:
    if plan.strategy == 'index_intersect':
        # Intersect all index lookups
        result_ids = None
        for lookup in plan.index_lookups:
            ids = index_mgr.lookup(plan.entity_type, lookup.field, lookup.value)
            result_ids = ids if result_ids is None else result_ids & ids

        # Load entities
        entities = [store.read(id) for id in result_ids]

    elif plan.strategy == 'index_scan':
        # Use first index, then filter
        first = plan.index_lookups[0]
        ids = index_mgr.lookup(plan.entity_type, first.field, first.value)
        entities = [store.read(id) for id in ids]

        # Apply post-filter
        if plan.post_filter:
            entities = [e for e in entities if evaluate(plan.post_filter, e)]

    elif plan.strategy == 'full_scan':
        # Load all entities of type
        prefix = schema_registry.get_schema(plan.entity_type).id_prefix
        entities = store.iter_entities(prefix)

        # Apply all conditions as post-filter
        entities = [e for e in entities if evaluate(plan.expression, e)]

    return entities
```

---

## Migration Plan

### Phase 1: Create CDG Query Infrastructure (This PR)

1. Create `cortical/cdg/query/` directory
2. Move lexer, parser, AST from `got/expression/`
3. Create new schema-aware executor
4. Create query planner
5. Keep `got/expression/` working (deprecated)

### Phase 2: Migrate GoT Functions

1. Create `cortical/got/query/functions/`
2. Move graph.py, filters.py, add git.py
3. Update functions to use `list_entities(entity_type)` pattern
4. Register GoT functions with CDG query engine at startup

### Phase 3: Unify CLI Commands

1. Make `got query` use CDG query engine
2. Deprecate `got expr` (alias to `got query`)
3. Remove `got infer` (use `got query "infer(commits=10)"`)
4. Keep `got blocked`, `got active` as shortcuts

### Phase 4: Cleanup (Future PR)

1. Remove `cortical/got/expression/` (after migration complete)
2. Remove deprecated `cmd_query` keyword matching
3. Update all documentation

---

## API Design

### Public API (cortical/cdg/query/__init__.py)

```python
from cortical.cdg.query import parse, execute, validate, CDGQuery

# Parse a query string
query = parse("FROM task WHERE status = 'pending'")

# Validate against schema
errors = validate(query, schema_registry)

# Execute query
results = execute(query, store, index_manager, schema_registry)

# Or use high-level API
from cortical.cdg.query import CDGQueryEngine

engine = CDGQueryEngine(store, index_manager, schema_registry)
results = engine.query("FROM task WHERE status = 'pending'")
```

### Function Registration

```python
from cortical.cdg.query.registry import FunctionRegistry

# Register a CDG core function
@FunctionRegistry.register(
    name='count',
    description='Count entities in result set',
    returns='Integer count'
)
def fn_count(entities: List[Entity]) -> int:
    return len(entities)

# Register a GoT extension function
@FunctionRegistry.register(
    name='blockers',
    description='Find tasks blocking this task',
    required_args=['entity_id'],
    returns='List of blocking tasks'
)
def fn_blockers(manager: GoTManager, entity_id: str) -> List[Entity]:
    # Implementation
    ...
```

---

## Schema Integration

### Indexed Fields (from entity_schemas.py)

The planner will check `Field.indexed` to determine query strategy:

```python
# TaskSchema
Field('status', FieldType.STRING, indexed=True)    # Use index
Field('priority', FieldType.STRING, indexed=True)  # Use index
Field('title', FieldType.STRING, indexed=False)    # Full scan
Field('content', FieldType.TEXT, indexed=False)    # Full scan

# SprintSchema
Field('status', FieldType.STRING, indexed=True)    # Use index
Field('name', FieldType.STRING, indexed=False)     # Full scan
```

### Entity Type Discovery

```python
def get_entity_type(self, entity_id: str) -> Optional[str]:
    """Get entity type from ID using SchemaRegistry."""
    prefix = entity_id.split('-')[0] + '-'
    return self.schema_registry.get_entity_type_by_prefix(prefix)
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/cdg/test_query_parser.py
def test_parse_simple_query():
    query = parse("FROM task WHERE status = 'pending'")
    assert query.entity_type == 'task'
    assert isinstance(query.expression, Comparison)

# tests/unit/cdg/test_query_planner.py
def test_plan_uses_index_for_indexed_field():
    query = parse("FROM task WHERE status = 'pending'")
    plan = planner.plan(query, schema_registry)
    assert plan.strategy == 'index_scan'
    assert plan.index_lookups[0].field == 'status'

# tests/unit/cdg/test_query_executor.py
def test_execute_with_index():
    # Setup mock index_manager
    index_manager.lookup.return_value = {'T-001', 'T-002'}

    results = execute(query, store, index_manager, schema_registry)

    index_manager.lookup.assert_called_once_with('task', 'status', 'pending')
```

### Integration Tests

```python
# tests/integration/test_cdg_query.py
def test_query_across_entity_types():
    """Query should work for any registered entity type."""
    engine = CDGQueryEngine(store, index_manager, schema_registry)

    # Query tasks
    tasks = engine.query("FROM task WHERE status = 'pending'")
    assert all(t.entity_type == 'task' for t in tasks)

    # Query decisions
    decisions = engine.query("FROM decision WHERE status = 'draft'")
    assert all(d.entity_type == 'decision' for d in decisions)

    # Query handoffs
    handoffs = engine.query("FROM handoff WHERE status = 'initiated'")
    assert all(h.entity_type == 'handoff' for h in handoffs)
```

---

## Backwards Compatibility

### Deprecation Path

```python
# cortical/got/expression/__init__.py
import warnings
from cortical.cdg.query import parse, execute, validate

warnings.warn(
    "cortical.got.expression is deprecated. Use cortical.cdg.query instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backwards compatibility
__all__ = ['parse', 'execute', 'validate']
```

### CLI Compatibility

```bash
# Old (deprecated but still works)
got expr "status = 'pending'"

# New (preferred)
got query "FROM task WHERE status = 'pending'"

# Shortcuts (unchanged)
got blocked
got active
```

---

## Success Criteria

1. **Generic:** Query any entity type without code changes
2. **Schema-Driven:** Entity types discovered from SchemaRegistry
3. **Indexed:** Uses CDGIndexManager when fields are indexed
4. **Extensible:** GoT can register domain-specific functions
5. **Unified:** One `query` command replaces `query`, `expr`, `infer`
6. **Backwards Compatible:** Old API works during migration period

---

## Open Questions

1. **JOIN support?** Should we support cross-entity queries like "tasks in sprint S-001"?
   - Recommendation: Defer. Use functions like `in_sprint('S-001')` instead.

2. **Subqueries?** Should we support nested queries?
   - Recommendation: Defer. Start with flat queries.

3. **Aggregation beyond count?** Sum, avg, group by?
   - Recommendation: Defer. Add when needed.

---

## Timeline

| Phase | Description | Estimate |
|-------|-------------|----------|
| 1 | Create CDG query infrastructure | This session |
| 2 | Migrate GoT functions | Next session |
| 3 | Unify CLI commands | Next session |
| 4 | Cleanup deprecated code | Future |

---

## Approval

- [ ] Architecture approved
- [ ] Ready to implement Phase 1

---

*This document will be updated as implementation progresses.*
