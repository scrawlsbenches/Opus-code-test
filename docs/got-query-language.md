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

```bash
# High-priority pending bugs
status = 'pending' AND priority = 'high' AND category = 'bug'

# Recently modified, not blocked
recent(7) AND NOT blocked()

# Tasks blocking Sprint 5
blocking() AND in_sprint(S-005)

# Orphaned high-priority tasks
orphan_nodes() AND priority = 'high'

# Full dependency tree, limited
all_dependencies(T-001) ORDER BY created_at LIMIT 50
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
