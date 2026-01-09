# GoT Query Language Reference

> **Gate**: Writing GoT queries? Read this first.

The GoT has two query interfaces:
1. **Natural language** (`query`) - Simple, conversational queries
2. **Expression language** (`expr`) - Full SQL-like query language

---

## Natural Language Queries (Simple)

```bash
python -m cortical.got query "what blocks T-001"
python -m cortical.got query "what depends on T-001"
python -m cortical.got query "blocked tasks"
python -m cortical.got query "active tasks"
python -m cortical.got query "path from T-001 to T-010"
```

---

## Expression Language (Full Power)

The expression language is SQL-like with boolean logic, operators, and functions.

### Quick Examples

```bash
# Simple comparison
python -m cortical.got expr "status = 'pending'"

# Boolean logic
python -m cortical.got expr "status = 'pending' AND priority = 'high'"

# With functions
python -m cortical.got expr "blocked() AND category = 'bug'"

# Order and limit
python -m cortical.got expr "status = 'pending' ORDER BY created_at DESC LIMIT 10"
```

---

## Grammar (EBNF)

```
query       ::= expression [ORDER BY field [ASC|DESC]] [LIMIT n [OFFSET m]]
expression  ::= and_expr (OR and_expr)*
and_expr    ::= not_expr (AND not_expr)*
not_expr    ::= NOT not_expr | primary
primary     ::= comparison | function_call | '(' expression ')'
comparison  ::= field operator value
operator    ::= = | != | > | < | >= | <= | IN | NOT IN | LIKE | NOT LIKE
```

---

## Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `=` | `status = 'pending'` | Equality |
| `!=` | `status != 'deleted'` | Inequality |
| `>`, `<` | `priority > 3` | Greater/less than |
| `>=`, `<=` | `priority >= 3` | Greater/less or equal |
| `IN` | `status IN ['pending', 'active']` | In list |
| `NOT IN` | `status NOT IN ['deleted']` | Not in list |
| `LIKE` | `title LIKE '%bug%'` | Pattern match (% = wildcard) |

---

## Boolean Logic

```bash
# AND (both must match)
status = 'pending' AND priority = 'high'

# OR (either matches)
status = 'pending' OR status = 'blocked'

# NOT (negation)
NOT status = 'completed'

# Parentheses for grouping
(status = 'pending' OR status = 'blocked') AND priority = 'high'

# Precedence: NOT > AND > OR
```

---

## Filter Functions

| Function | Description | Example |
|----------|-------------|---------|
| `recent(days)` | Modified within N days | `recent(7)` |
| `stale(days)` | Not modified for N days | `stale(30)` |
| `blocked()` | Blocked tasks | `blocked()` |
| `blocking()` | Tasks blocking others | `blocking()` |
| `in_sprint(id)` | In specific sprint | `in_sprint(S-001)` |
| `unassigned()` | No assignment | `unassigned()` |
| `overdue()` | Past due date | `overdue()` |
| `has_edge(type)` | Has edge of type | `has_edge('DEPENDS_ON')` |
| `entity_type(type)` | Filter by type | `entity_type('task')` |

---

## Graph Functions

| Function | Description | Example |
|----------|-------------|---------|
| `connected_to(id)` | Connected entities | `connected_to(T-001)` |
| `path(from, to)` | Find path | `path(T-001, T-010)` |
| `children(id)` | Direct children | `children(S-001)` |
| `parents(id)` | Direct parents | `parents(T-001)` |
| `descendants(id)` | All descendants | `descendants(E-001)` |
| `ancestors(id)` | All ancestors | `ancestors(T-001)` |
| `orphan_nodes()` | Disconnected entities | `orphan_nodes()` |
| `blockers(id)` | What blocks task | `blockers(T-001)` |
| `dependents(id)` | What depends on task | `dependents(T-001)` |
| `all_dependencies(id)` | Full dependency tree | `all_dependencies(T-001)` |
| `cycle_detect()` | Find cycles | `cycle_detect()` |

---

## Order and Pagination

```bash
# Order ascending
status = 'pending' ORDER BY created_at ASC

# Order descending
status = 'pending' ORDER BY created_at DESC

# Limit results
category = 'bug' LIMIT 10

# Pagination
category = 'bug' LIMIT 10 OFFSET 20
```

---

## Function Arguments

```bash
# Positional
path(T-001, T-002)

# Keyword
path(from=T-001, to=T-002)

# Mixed (positional first)
path(T-001, T-002, max_depth=5)
```

---

## Entity Fields

### Task Fields
| Field | Type | Values |
|-------|------|--------|
| `id` | string | T-XXX |
| `title` | string | - |
| `status` | string | pending/in_progress/completed/blocked |
| `priority` | string | critical/high/medium/low |
| `category` | string | feature/bugfix/refactor/docs/test |
| `created_at` | datetime | - |
| `updated_at` | datetime | - |

---

## Complex Examples

### Sprint Planning

```bash
# Find candidates for next sprint: pending, high priority, not blocked, no dependencies
status = 'pending' AND priority IN ['critical', 'high'] AND NOT blocked() AND orphan_nodes()
ORDER BY priority DESC LIMIT 20

# Tasks ready to start: pending with all dependencies completed
status = 'pending' AND NOT blocked() AND NOT has_edge('DEPENDS_ON')

# Sprint capacity check: unfinished tasks in current sprint
in_sprint(S-005) AND status IN ['pending', 'in_progress', 'blocked']
ORDER BY priority DESC
```

### Dependency Analysis

```bash
# Critical path: what must complete before T-100 can start
ancestors(T-100) AND status != 'completed' ORDER BY created_at ASC

# Ripple effect: everything that depends on T-050 (directly or indirectly)
descendants(T-050) ORDER BY priority DESC

# Circular dependency detection
cycle_detect()

# Tasks with many blockers (complexity indicator)
blocked() AND has_edge('DEPENDS_ON') ORDER BY updated_at DESC LIMIT 10
```

### Risk Assessment

```bash
# Stale blocked tasks: blocked for 14+ days (needs escalation)
blocked() AND stale(14) ORDER BY updated_at ASC

# High-priority tasks with no recent activity
priority IN ['critical', 'high'] AND stale(7) AND status = 'in_progress'

# Overdue tasks blocking others
overdue() AND blocking() ORDER BY priority DESC

# Orphaned work: tasks not connected to any sprint or epic
orphan_nodes() AND entity_type('task') AND status != 'completed'
```

### Progress Tracking

```bash
# Recently completed high-value work
status = 'completed' AND priority IN ['critical', 'high'] AND recent(7)
ORDER BY updated_at DESC LIMIT 20

# Active work by category
status = 'in_progress' ORDER BY category ASC, priority DESC

# Bug fix velocity: completed bugs in last 30 days
status = 'completed' AND category = 'bug' AND recent(30)
```

### Cleanup & Maintenance

```bash
# Find disconnected decisions (no implementing tasks)
entity_type('decision') AND orphan_nodes()

# Knowledge transfers still in draft
entity_type('knowledge_transfer') AND status = 'draft' AND stale(7)

# Handoffs pending acceptance
entity_type('handoff') AND status = 'initiated' ORDER BY created_at ASC
```

### Combined Graph + Filter Queries

```bash
# All blockers of tasks in Sprint 5 that are themselves blocked
blockers(in_sprint(S-005)) AND blocked()

# Path between two tasks, excluding completed intermediate tasks
path(T-001, T-100) AND status != 'completed'

# Children of epic E-001 that are high priority and not started
children(E-001) AND priority = 'high' AND status = 'pending'
```

---

## CLI Options

```bash
# Expression query
python -m cortical.got expr "expression"

# Show execution plan
python -m cortical.got expr "expression" --explain

# JSON output
python -m cortical.got expr "expression" --json
```

---

## Source Files

| File | Purpose |
|------|---------|
| `cortical/got/expression/grammar.py` | EBNF grammar definition |
| `cortical/got/expression/parser.py` | Expression parser |
| `cortical/got/expression/executor.py` | Query execution |
| `cortical/got/expression/functions/` | Built-in functions |
