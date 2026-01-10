# CLI Reference

> **Gate**: Need a CLI command? Read this file first.

---

## GoT CLI Reference

The GoT CLI is the primary interface for Graph of Thought operations:

```bash
python -m cortical.got [command] [subcommand] [options]
```

### Available Commands

| Command | Description |
|---------|-------------|
| `task` | Task CRUD, lifecycle (create/start/complete/block) |
| `sprint` | Sprint management, claiming, goals |
| `epic` | Epic-level organization |
| `handoff` | Agent-to-agent handoffs |
| `decision` | Log decisions with rationale |
| `edge` | Create/manage relationships |
| `query` | Natural language queries |
| `expr` | Expression-based queries (e.g., `status = 'pending'`) |
| `analyze` | Graph analysis (dependencies, patterns, orphans) |
| `kt` / `knowledge` | Knowledge transfer documents |
| `failure` | Track failed approaches |
| `backup` | Backup and recovery |
| `validate` | Health checks |
| `stats` | Statistics overview |
| `dashboard` | Comprehensive metrics |
| `blocked` | List blocked tasks |
| `active` | List in-progress tasks |
| `infer` | Infer edges from git commits |
| `orphan` | Detect disconnected entities |
| `backlog` | Manage unassigned tasks |
| `batch` | Execute batch operations |

### Task Commands

```bash
# Create
python -m cortical.got task create "Title" --priority high --category feature

# Lifecycle
python -m cortical.got task start T-XXX
python -m cortical.got task complete T-XXX --retrospective "What worked..."
python -m cortical.got task block T-XXX --reason "Waiting for API"

# Query
python -m cortical.got task list --status in_progress
python -m cortical.got task show T-XXX
python -m cortical.got task next                    # Get recommended next task

# Dependencies
python -m cortical.got task depends T-XXX --on T-YYY
```

### Sprint Commands

```bash
# Create and manage
python -m cortical.got sprint create "Sprint 20" --number 20
python -m cortical.got sprint list
python -m cortical.got sprint status               # Current sprint

# Claiming (for parallel agents)
python -m cortical.got sprint claim S-XXX --agent "agent-1"
python -m cortical.got sprint release S-XXX --agent "agent-1"

# Task assignment
python -m cortical.got sprint link S-XXX T-YYY
python -m cortical.got sprint tasks S-XXX
python -m cortical.got sprint suggest              # AI-suggested tasks
```

### Decision Logging

```bash
python -m cortical.got decision log "Use PostgreSQL over SQLite" \
    --rationale "Need concurrent writes" \
    --affects T-XXX T-YYY
```

### Knowledge Transfers

```bash
python -m cortical.got kt create "Session: Auth refactor" --summary "..."
python -m cortical.got kt list --status draft
python -m cortical.got kt finalize KT-XXX
```

### Edge (Relationship) Management

```bash
python -m cortical.got edge add T-001 T-002 DEPENDS_ON
python -m cortical.got edge add S-001 T-001 CONTAINS
python -m cortical.got edge list --source T-XXX
```

Edge types: `DEPENDS_ON`, `BLOCKS`, `SIMILAR`, `CONTAINS`, `IMPLEMENTS`, `TESTS`, `JUSTIFIES`, `RELATED`

### Queries

```bash
# Natural language
python -m cortical.got query "what blocks T-XXX"
python -m cortical.got query "blocked tasks"
python -m cortical.got query "path from T-1 to T-2"

# Expression syntax
python -m cortical.got expr "status = 'pending' AND priority = 'high'"
```

### Analysis Commands

```bash
python -m cortical.got analyze summary             # Quick overview
python -m cortical.got analyze dependencies T-XXX  # Dependency chain
python -m cortical.got analyze patterns            # Find graph patterns
python -m cortical.got analyze orphans             # Disconnected clusters
```

### Handoffs

```bash
python -m cortical.got handoff initiate T-XXX --target agent-2 --instructions "..."
python -m cortical.got handoff accept H-XXX --agent agent-2
python -m cortical.got handoff complete H-XXX --agent agent-2
python -m cortical.got handoff list --status initiated
```

### Failed Approach Tracking

```bash
python -m cortical.got failure log T-XXX --attempt "Tried mutex lock" --error "Caused deadlock"
python -m cortical.got failure list T-XXX
```

### Health & Maintenance

```bash
python -m cortical.got validate                    # Check integrity
python -m cortical.got validate --check-refs       # Deep validation
python -m cortical.got stats                       # Statistics
python -m cortical.got dashboard                   # Full metrics
python -m cortical.got backup create              # Creates timestamped snapshot
python -m cortical.got infer --commits 10          # Infer edges from git
```

### Batch Operations

```bash
# From file
python -m cortical.got batch --file commands.yaml

# From stdin
cat <<EOF | python -m cortical.got batch
task create "Task 1" --priority high
task create "Task 2" --priority medium
edge add T-001 T-002 DEPENDS_ON
EOF
```

⚠️ **NEVER edit `.got/` files directly** - use these commands!

---

## Audit CLI Reference

The Audit CLI provides codebase quality analysis tools using algorithms like Bloom Filters, LSH, Suffix Arrays, and PLN reasoning:

```bash
# Entry point
python -m cortical.cli.audit [command]
```

### Available Commands

| Command | Purpose |
|---------|---------|
| `generate` | Generate training data from codebase comments |
| `train` | Train classifiers from labeled findings |
| `scan` | Scan for suspicious comments |
| `patterns` | Find repeated patterns in comments |
| `similar` | Find similar comments using LSH |
| `index` | Build search indexes |
| `health` | Analyze codebase health |
| `reason` | PLN-based audit reasoning |
| `discover` | WovenMind pattern discovery (experimental) |

### Typical Workflow

```bash
# 1. Generate training data from codebase
python -m cortical.cli.audit generate cortical/ -o docs/audits/

# 2. Train classifiers from labeled findings
python -m cortical.cli.audit train docs/audits/

# 3. Scan for suspicious comments
python -m cortical.cli.audit scan cortical/
```

### Health Analysis

Comprehensive codebase health check using pattern detection, duplicate detection, and git history:

```bash
# Basic health check
python -m cortical.cli.audit health cortical/

# Include git history analysis (stale TODOs, high churn files)
python -m cortical.cli.audit health cortical/ --git

# Verbose output with findings
python -m cortical.cli.audit health cortical/ --git -v

# JSON output for automation
python -m cortical.cli.audit health cortical/ --json
```

### Scanning for Issues

Uses Bloom Filter for pre-screening and Naive Bayes for classification:

```bash
# Scan directory for suspicious comments
python -m cortical.cli.audit scan cortical/

# Verbose with confidence threshold
python -m cortical.cli.audit scan cortical/ -v --confidence 0.5
```

### Pattern Discovery

Find repeated patterns and similar comments:

```bash
# Find repeated patterns in comments
python -m cortical.cli.audit patterns cortical/

# Find similar comments using LSH clustering
python -m cortical.cli.audit similar cortical/
```

### PLN Reasoning

Probabilistic Logic Networks for risk assessment:

```bash
# Analyze with natural language query
python -m cortical.cli.audit reason "risky files in reasoning/"

# Analyze specific directory
python -m cortical.cli.audit reason --directory cortical/

# Explain risk for a specific file
python -m cortical.cli.audit reason --explain cortical/reasoning/loop_validator.py

# Load WovenMind rules for advanced reasoning
python -m cortical.cli.audit reason cortical/ --load-rules

# Mark a file as Very Long Term Important (pinned)
python -m cortical.cli.audit reason --vlti cortical/cdg/transaction_manager.py
```

### WovenMind Discovery (Experimental)

Uses dual-process cognitive architecture for emergent pattern discovery:

```bash
# Run discovery analysis
python -m cortical.cli.audit discover cortical/

# Include git history
python -m cortical.cli.audit discover cortical/ --with-git

# Show learned mind state
python -m cortical.cli.audit discover --show-mind

# Run learning consolidation cycle
python -m cortical.cli.audit discover --consolidate

# Reset mind state for fresh start
python -m cortical.cli.audit discover --reset-mind
```

### Algorithms Used

| Algorithm | Used In | Purpose |
|-----------|---------|---------|
| Bloom Filter | scan | Fast pre-screening for suspicious patterns |
| Naive Bayes | scan, train | Classification of comment types |
| Trie | scan | Comment marker detection (TODO, FIXME, etc.) |
| Inverted Index | health, index | Pattern lookup |
| Suffix Array | health | Duplicate detection |
| LSH (Locality-Sensitive Hashing) | similar | Near-duplicate clustering |
| Union-Find | health | Grouping similar items |
| DAG | health | Import dependency analysis |
| PLN (Probabilistic Logic Networks) | reason | Multi-rule risk aggregation |

---

## GoT Deep Dive: Understanding the Data Model

This section teaches how the GoT CLI works internally.

### Entity Types and Storage

GoT stores entities as JSON files in `.got/entities/`:

| Entity | ID Prefix | File Pattern | Key Fields |
|--------|-----------|--------------|------------|
| Task | T- | `T-*.json` | title, status, priority, description, properties |
| Edge | E- | `E-*.json` | from_id, to_id, edge_type, weight |
| Decision | D- | `D-*.json` | content, rationale, status |
| Sprint | S- | `S-*.json` | name, goal, status, task_ids |
| Handoff | H- | `H-*.json` | task_id, target, instructions, status |
| KnowledgeTransfer | KT- | `KT-*.json` | title, summary, sections, status |

### The Task Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TASK STATE MACHINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    [pending] ──start──► [in_progress] ──complete──► [completed]         │
│        │                     │                                          │
│        └───────block────────►│◄────unblock────────                      │
│                              │                                          │
│                         [blocked]                                       │
│                                                                          │
│    Commands:                                                            │
│    - task create → pending                                              │
│    - task start T-XXX → in_progress                                     │
│    - task complete T-XXX → completed (requires retrospective!)          │
│    - task block T-XXX --reason "..." → blocked                          │
│    - task unblock T-XXX → in_progress                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Task Properties: The Extensibility Point

Every task has a `properties: Dict[str, Any]` field for storing arbitrary metadata:

```python
task.properties = {
    "retrospective": "What worked: X. What didn't: Y. Learned: Z.",
    "category": "feature",  # feature/bugfix/refactor/docs/test
    "estimated_effort": "2h",
    "actual_effort": "3h"
}
```

### Edge Types and When to Use Them

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EDGE TYPE GUIDE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  DEPENDENCY EDGES:                                                      │
│  - DEPENDS_ON: T-2 depends on T-1 (T-1 must complete first)            │
│  - BLOCKS: T-1 blocks T-2 (inverse of DEPENDS_ON)                       │
│                                                                          │
│  STRUCTURAL EDGES:                                                      │
│  - CONTAINS: Sprint S-1 contains Task T-1                               │
│  - BELONGS_TO: T-1 belongs to Epic E-1                                  │
│                                                                          │
│  RELATIONSHIP EDGES:                                                    │
│  - SIMILAR: T-1 is similar to T-2 (for guidance/learning)              │
│  - RELATED: Generic relationship                                        │
│  - IMPLEMENTS: T-1 implements Decision D-1                              │
│  - TESTS: T-1 tests feature in T-2                                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Query Language (Natural Language)

```bash
# Blocking relationships
python -m cortical.got query "what blocks T-001"
python -m cortical.got query "what does T-001 depend on"

# Status queries
python -m cortical.got query "blocked tasks"
python -m cortical.got query "high priority pending"

# Path queries
python -m cortical.got query "path from T-001 to T-010"

# Free-form search
python -m cortical.got query "authentication"
```

### GoTManager API

The `cortical/got/api.py` module provides the primary interface for GoT operations:

```python
from cortical.core.bootstrap import create_container
from cortical.got.api import GoTManager

container = create_container()
manager = container.resolve(GoTManager)
```

| Method | Purpose | Returns |
|--------|---------|---------|
| `create_task(title, **kwargs)` | Create new task | Task object |
| `get_task(task_id)` | Fetch task by ID | Task or None |
| `update_task(task_id, **updates)` | Update task fields | Task object |
| `complete_task(task_id, retrospective)` | Mark complete | bool |
| `list_tasks(status, priority)` | Query tasks | List[Task] |
| `get_blocked_tasks()` | Find blocked tasks | List[(Task, reason)] |
| `add_edge(from_id, to_id, edge_type)` | Create relationship | Edge object |
| `list_edges()` | Get all edges | List[Edge] |

### Common Patterns

**Pattern 1: Task Workflow**
```bash
T_ID=$(python -m cortical.got task create "Fix login bug" --priority high)
python -m cortical.got task start $T_ID
python -m cortical.got task complete $T_ID --retrospective "Fixed by extending TTL."
```

**Pattern 2: Dependency Chain**
```bash
python -m cortical.got edge add T-001 T-002 DEPENDS_ON
python -m cortical.got blocked
```

**Pattern 3: Session Handoff**
```bash
python -m cortical.got kt create "Session: Auth refactor" --summary "..."
python -m cortical.got handoff initiate T-001 --target "next-agent" --instructions "..."
```

**Pattern 4: Failed Approach Tracking**
```bash
python -m cortical.got failure log T-001 --attempt "Tried mutex lock" --error "Caused deadlock"
```

### Validation and Recovery

```bash
python -m cortical.got validate              # Basic validation
python -m cortical.got validate --check-refs # Deep validation
python -m cortical.got stats                 # Statistics
python -m cortical.got recover               # If validation fails
python -m cortical.got backup create         # Creates timestamped snapshot
python -m cortical.got backup restore BACKUP_ID
```
