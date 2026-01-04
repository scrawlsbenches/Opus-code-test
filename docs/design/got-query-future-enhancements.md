# Graph of Thought Query System: Future Enhancements

**Status:** Deferred - For discussion after foundation is built
**Parent Document:** [got-query-audit-and-design.md](./got-query-audit-and-design.md)
**Date:** 2026-01-04
**Version:** 1.0

---

## Purpose

This document captures query language enhancements identified during gap analysis that are **not critical for the initial foundation**. These features should be discussed and prioritized after the core expression parser and function registry are operational.

The foundation must be built first:
1. Expression lexer and parser
2. Function registry pattern
3. Query builder integration
4. Core graph traversal functions
5. NOT/Negation support
6. Transitive closure operations

Only then should we extend with these additional capabilities.

---

## Enhancement Categories

### 1. Field Projection (Medium Priority)

**Current State:** Queries return full entities with all fields.

**Enhancement:** Allow selecting specific fields, like SQL's SELECT clause.

```
# Proposed syntax options:

# Option A: Function-style
select('id', 'title', 'status') WHERE status = 'pending'

# Option B: SQL-style
SELECT id, title, status WHERE status = 'pending'

# Option C: Projection function
project(['id', 'title']) WHERE status = 'pending'
```

**Benefits:**
- Reduced memory for large result sets
- Cleaner output for reporting
- Enables computed fields in future

**Implementation Considerations:**
- Should work with all entity types
- How to handle nested fields (properties.foo)?
- Should support aliasing? (title AS name)

---

### 2. Set Operations (Medium Priority)

**Current State:** No way to combine query results.

**Enhancement:** Support UNION, INTERSECT, EXCEPT operations.

```
# All tasks that are pending OR blocked (different from OR in WHERE)
status = 'pending' UNION status = 'blocked'

# Tasks that are both high priority AND in Sprint-5
priority = 'high' INTERSECT connected_to('S-005', via='CONTAINS')

# Pending tasks NOT in any sprint
status = 'pending' EXCEPT connected_to('S-*', via='CONTAINS')
```

**Benefits:**
- Complex query composition
- Enables "tasks not in X" patterns
- Foundation for saved query building

**Implementation Considerations:**
- UNION ALL vs UNION (duplicates)?
- Result ordering after set operations?
- Type compatibility across operands?

---

### 3. NULL/Missing Value Handling (Medium Priority)

**Current State:** No explicit NULL handling. Missing fields may cause runtime errors or silent failures.

**Enhancement:** Add IS NULL, IS NOT NULL, COALESCE operators.

```
# Grammar additions
<null_check>    ::= <field> 'IS' ['NOT'] 'NULL'
<coalesce>      ::= 'COALESCE' '(' <field> ',' <value> [',' <value>]* ')'

# Examples
description IS NOT NULL
sprint_id IS NULL
ORDER BY COALESCE(priority, 'low')
```

**Benefits:**
- Query entities with missing optional fields
- Handle data quality issues gracefully
- Default value support in ordering

**Implementation Considerations:**
- What constitutes "null" vs "empty string" vs missing field?
- COALESCE on non-existent fields?
- Performance of null checks across large datasets?

---

### 4. Temporal Query Functions (Medium Priority)

**Current State:** Limited date filtering, no date arithmetic.

**Enhancement:** Full temporal query support.

```
# Date arithmetic
created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
completed_at BETWEEN '2026-01-01' AND '2026-01-31'

# Relative time queries
AGE(created_at) > INTERVAL 30 DAY
updated_at IN LAST 24 HOURS

# Date extraction
YEAR(created_at) = 2026
MONTH(completed_at) = 1
```

**Benefits:**
- Time-based reporting ("tasks completed this week")
- SLA tracking ("overdue tasks")
- Historical analysis

**Implementation Considerations:**
- Timezone handling (store UTC, query local?)
- Date format parsing flexibility
- Performance of date comparisons
- NOW() caching within single query

---

### 5. Existence Subqueries (Medium Priority)

**Current State:** No subquery support.

**Enhancement:** WHERE EXISTS and WHERE NOT EXISTS patterns.

```
# Tasks that have at least one dependency
WHERE EXISTS (edges WHERE from_id = task.id AND type = 'DEPENDS_ON')

# Orphan tasks (no incoming or outgoing edges)
WHERE NOT EXISTS (edges WHERE from_id = task.id OR to_id = task.id)

# Tasks with completed dependencies
WHERE NOT EXISTS (
  SELECT 1 FROM tasks AS dep
  WHERE dep.id IN (dependencies of task.id)
  AND dep.status != 'completed'
)
```

**Benefits:**
- Complex filtering based on relationships
- "Orphan detection" queries
- "All dependencies satisfied" queries

**Implementation Considerations:**
- Correlation between outer and inner query
- Performance of correlated subqueries
- Syntax for referencing outer scope

---

### 6. Extended Aggregation (Medium Priority)

**Current State:** COUNT aggregation only. Single-field GROUP BY.

**Enhancement:** Full aggregation function suite.

```
# Additional aggregation functions
SUM(properties.story_points) GROUP BY sprint_id
AVG(AGE(completed_at - created_at)) GROUP BY priority
MIN(created_at), MAX(updated_at) GROUP BY status

# Multi-field grouping
COUNT(*) GROUP BY status, priority

# HAVING clause
COUNT(*) GROUP BY sprint_id HAVING COUNT(*) > 5

# Aggregation with filtering
COUNT(*) WHERE status = 'pending' GROUP BY priority
```

**Benefits:**
- Sprint velocity calculations
- Priority distribution analysis
- Workload balancing queries

**Implementation Considerations:**
- NULL handling in aggregations
- Type coercion for SUM/AVG
- HAVING vs WHERE execution order
- Aggregation over computed fields

---

### 7. DISTINCT Results (Low Priority)

**Current State:** No deduplication of results.

**Enhancement:** DISTINCT keyword to remove duplicates.

```
# Unique statuses in use
DISTINCT status FROM tasks

# Unique priority/status combinations
DISTINCT status, priority FROM tasks

# After set operations
DISTINCT (query1 UNION query2)
```

**Benefits:**
- Clean enumeration of values
- Result normalization
- Set semantics support

**Implementation Considerations:**
- Definition of "distinct" for complex objects
- DISTINCT with aggregation
- Performance of large distinct operations

---

### 8. Graph Metrics Exposure (Low Priority)

**Current State:** Graph analysis exists (PageRank, clustering) but not queryable.

**Enhancement:** Expose graph metrics as queryable fields.

```
# Sort by PageRank (importance)
ORDER BY PAGERANK(task.id) DESC LIMIT 10

# Find clusters
CLUSTER_ID(task.id) = CLUSTER_ID('T-001')

# Betweenness centrality (bottleneck detection)
BETWEENNESS(task.id) > 0.5

# Connected component queries
COMPONENT(task.id) = COMPONENT('T-001')
```

**Benefits:**
- "Most important tasks" queries
- Bottleneck identification
- Related task grouping

**Implementation Considerations:**
- When to compute metrics (on-demand vs pre-computed)?
- Staleness of pre-computed metrics
- Computational cost for large graphs
- Caching strategy

---

### 9. Variable Bindings / CTEs (Low Priority)

**Current State:** No variable support, no named subqueries.

**Enhancement:** Common Table Expressions for complex queries.

```
# CTE style
WITH blocked_tasks AS (
  SELECT * FROM tasks WHERE status = 'blocked'
),
their_blockers AS (
  SELECT from_id FROM edges WHERE to_id IN (blocked_tasks.id) AND type = 'BLOCKS'
)
SELECT * FROM tasks WHERE id IN (their_blockers.from_id)

# Variable binding style
$critical_tasks = priority = 'critical'
$in_sprint_5 = connected_to('S-005', via='CONTAINS')
$critical_tasks AND $in_sprint_5
```

**Benefits:**
- Complex query decomposition
- Query reuse within session
- Improved readability

**Implementation Considerations:**
- Scope of variable bindings (query vs session)
- Recursive CTEs (for transitive closure)?
- Memory management for materialized CTEs

---

### 10. All Paths vs Shortest Path (Low Priority)

**Current State:** `path()` function finds shortest path only.

**Enhancement:** Support for all paths enumeration.

```
# All paths (may be many)
all_paths('T-001', 'T-050', via='DEPENDS_ON')

# K-shortest paths
k_shortest_paths('T-001', 'T-050', k=3)

# Paths with constraints
paths_through('T-001', 'T-050', must_include='T-025')
```

**Benefits:**
- Dependency analysis ("how many ways to reach X?")
- Redundancy detection
- Critical path identification

**Implementation Considerations:**
- Exponential blowup of all-paths
- Output format (list of paths vs iterator)
- Memory limits and streaming
- Cycle handling in all-paths

---

### 11. Write Operations Exclusion (Documentation)

**Current State:** Implicit that query language is read-only.

**Enhancement:** Explicit documentation that query language is READ-ONLY.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUERY LANGUAGE SCOPE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  The GoT Query Language is READ-ONLY by design.                         │
│                                                                          │
│  WHY:                                                                   │
│  - Write operations require transactional guarantees                    │
│  - Modifications need audit trails (GoT system handles this)            │
│  - Prevents accidental data corruption from query typos                 │
│  - Separates concerns: query vs mutation                                │
│                                                                          │
│  FOR WRITES, USE:                                                       │
│  - got_utils.py CLI commands                                            │
│  - TransactionalGoTAdapter Python API                                   │
│  - GoT Manager direct methods                                           │
│                                                                          │
│  NEVER IMPLEMENT:                                                       │
│  - UPDATE expressions                                                   │
│  - DELETE expressions                                                   │
│  - INSERT/CREATE expressions                                            │
│                                                                          │
│  If someone requests write support in the query language,               │
│  direct them to the existing write APIs.                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Additional Design Thoughts

### On Query Language Philosophy

The expression system we're building sits at an interesting intersection: it must be powerful enough for developers to express complex graph queries, yet constrained enough that it compiles cleanly to the existing Query builder infrastructure.

**Key insight from the gap analysis:** The Query builder is already remarkably capable. The real value of the expression parser isn't adding new capabilities—it's providing a human-friendly surface syntax that maps to those capabilities.

This means our grammar should prioritize:
1. **Familiarity** - SQL-like syntax where possible
2. **Composability** - Boolean operators, parentheses, function chaining
3. **Discoverability** - Helpful errors, schema introspection, suggestions
4. **Debuggability** - EXPLAIN plans, position tracking, clear error messages

### On Future Extension Points

The function registry pattern provides clean extension without core modification. Future capabilities should be added as registered functions, not grammar changes:

```python
# New function = new capability, no parser changes needed
@FunctionRegistry.register("pagerank")
class PageRankFunction(QueryFunction):
    ...

@FunctionRegistry.register("cluster")
class ClusterFunction(QueryFunction):
    ...
```

### On NOT Being More Fundamental Than Expected

Adding NOT to the grammar revealed something interesting: negation is surprisingly fundamental to expressiveness. Without NOT, you can't express:
- "Tasks that are NOT in any sprint"
- "Entities without incoming edges"
- "Decisions that don't have rationale"

The grammar change from:
```
<primary> ::= <comparison> | <function_call> | '(' <expression> ')'
```
to:
```
<not_expr> ::= 'NOT' <not_expr> | <primary>
<and_expr> ::= <not_expr> ( 'AND' <not_expr> )*
```

This gives NOT higher precedence than AND (and AND higher than OR), matching SQL semantics and user expectations.

### On Hardcoded Limits

The principle "no hardcoded magic numbers" is worth emphasizing. Default depth=10 or max_length=5 might seem "safe," but:
1. They silently truncate results without user awareness
2. They're arbitrary—why 10 and not 20?
3. They assume "typical" use cases that may not apply
4. They mask performance problems rather than exposing them

If a transitive query on a 1000-node graph takes too long, the developer should see that and decide how to handle it—not have the system silently return incomplete results.

### On the Relationship to GraphQL

Someone might ask: "Why not just use GraphQL?" The answer is sovereignty and specificity:

1. **Sovereignty** - We don't adopt external dependencies we can't maintain
2. **Specificity** - Our query language knows about GoT concepts (tasks, dependencies, sprints) natively
3. **Simplicity** - GraphQL solves API evolution; we're solving internal querying
4. **Integration** - We compile to Query builder, preserving all optimizations

---

## Prioritization Recommendations

When the foundation is complete, here's my recommended priority order:

| Priority | Enhancement | Rationale |
|----------|-------------|-----------|
| 1 | Field Projection | Most requested, simplest addition |
| 2 | Temporal Functions | High value for reporting |
| 3 | Extended Aggregation | Analytics support |
| 4 | NULL Handling | Data quality, robustness |
| 5 | Existence Subqueries | Complex filtering patterns |
| 6 | Set Operations | Query composition |
| 7 | DISTINCT | Clean semantics |
| 8 | Graph Metrics | Advanced analysis |
| 9 | Variables/CTEs | Power user feature |
| 10 | All Paths | Specialized use cases |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-04 | Initial document from gap analysis |

---

*This document is intentionally separate from the main design to keep the foundation focused and buildable. These enhancements should not distract from completing the core system first.*
