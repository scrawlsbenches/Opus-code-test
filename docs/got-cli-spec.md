# GoT CLI Specification: Graph-Based Project Management

**Version:** 1.0.0
**Status:** Design Specification
**Author:** Claude Code (2025-12-20)

---

## Executive Summary

This document specifies the complete CLI interface for `scripts/got_utils.py`, a Graph of Thought (GoT) based project management system that replaces file-based task tracking with graph-native storage.

**Key Design Principles:**
1. **Graph-native storage** - Tasks, sprints, and epics are graph nodes, not files
2. **Rich relationships** - Edges capture dependencies, blockers, and hierarchy
3. **Persistent state** - WAL ensures durability, snapshots enable fast recovery
4. **Query flexibility** - Traverse graphs to answer complex questions
5. **Migration path** - Gradual transition from file-based system with `sync` command

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         GoT CLI LAYER                            │
│  (got_utils.py - User-facing command interface)                 │
├─────────────────────────────────────────────────────────────────┤
│                         GRAPH LAYER                              │
│  ThoughtGraph with nodes:                                        │
│  - NodeType.TASK → Individual work items                        │
│  - NodeType.GOAL → Sprint/Epic goals                            │
│  - NodeType.CONTEXT → Sprint/Epic metadata                      │
│                                                                  │
│  Edges:                                                          │
│  - EdgeType.DEPENDS_ON → Task dependencies                      │
│  - EdgeType.BLOCKS → Blocking relationships                     │
│  - EdgeType.CONTAINS → Sprint contains tasks                    │
│  - EdgeType.MOTIVATES → Goal motivates tasks                    │
├─────────────────────────────────────────────────────────────────┤
│                      PERSISTENCE LAYER                           │
│  - GraphWAL for operation logging                               │
│  - Snapshots for fast recovery                                  │
│  - GitAutoCommitter for versioning                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Node Schema Design

### Task Node
```python
node_id = "T-YYYYMMDD-HHMMSS-XXXX"  # Collision-free timestamp IDs
node_type = NodeType.TASK
content = "Task title/summary"
properties = {
    'status': 'pending|in_progress|completed|blocked|deferred',
    'priority': 'high|medium|low',
    'category': 'feature|bugfix|docs|refactor|test|devex|arch',
    'effort': 'small|medium|large',
    'created_at': '2025-12-20T10:30:00',
    'updated_at': '2025-12-20T11:45:00',
    'completed_at': '2025-12-20T15:00:00',  # None if not completed
    'sprint_id': 'S-001',  # Optional sprint assignment
}
metadata = {
    'description': 'Detailed task description',
    'retrospective': {
        'notes': 'What was learned',
        'duration_minutes': 120,
        'files_touched': ['file1.py', 'file2.py'],
        'tests_added': 5,
        'commits': ['abc123', 'def456'],
    },
    'context': {
        'files': ['relevant.py'],
        'methods': ['relevant_function'],
    }
}
```

### Sprint Node
```python
node_id = "S-001"  # Human-friendly sprint IDs
node_type = NodeType.CONTEXT
content = "Sprint name/title"
properties = {
    'status': 'available|in_progress|complete',
    'started': '2025-12-01',
    'ended': '2025-12-15',  # None if not complete
    'epic_id': 'E-AUTH',  # Optional epic linkage
}
metadata = {
    'notes': ['Sprint note 1', 'Sprint note 2'],
    'blocked': ['Blocked item 1'],
}
```

### Sprint Goal Node
```python
node_id = "G-{sprint_id}-001"  # e.g., "G-S-001-001"
node_type = NodeType.GOAL
content = "Goal text"
properties = {
    'completed': False,
    'sprint_id': 'S-001',
}
```

### Epic Node
```python
node_id = "E-{SLUG}"  # e.g., "E-AUTH", "E-PERF"
node_type = NodeType.CONTEXT
content = "Epic name/title"
properties = {
    'status': 'active|complete|archived',
    'created': '2025-12-01',
}
metadata = {
    'description': 'Epic description',
}
```

---

## Edge Schema Design

### Task Dependencies
```python
# Task T1 depends on Task T2
ThoughtEdge(
    source_id="T-20251220-100000-a1b2",
    target_id="T-20251220-095000-c3d4",
    edge_type=EdgeType.DEPENDS_ON,
    weight=1.0,
    confidence=1.0,
)
```

### Task Blockers
```python
# Task T1 is blocked by Task T2
ThoughtEdge(
    source_id="T-20251220-100000-a1b2",
    target_id="T-20251220-095000-c3d4",
    edge_type=EdgeType.BLOCKS,
    weight=1.0,
    confidence=1.0,
)
```

### Sprint Contains Tasks
```python
# Sprint S-001 contains Task T1
ThoughtEdge(
    source_id="S-001",
    target_id="T-20251220-100000-a1b2",
    edge_type=EdgeType.CONTAINS,
    weight=1.0,
    confidence=1.0,
)
```

### Goal Motivates Tasks
```python
# Goal G-S-001-001 motivates Task T1
ThoughtEdge(
    source_id="G-S-001-001",
    target_id="T-20251220-100000-a1b2",
    edge_type=EdgeType.MOTIVATES,
    weight=1.0,
    confidence=1.0,
)
```

### Epic Contains Sprints
```python
# Epic E-AUTH contains Sprint S-001
ThoughtEdge(
    source_id="E-AUTH",
    target_id="S-001",
    edge_type=EdgeType.CONTAINS,
    weight=1.0,
    confidence=1.0,
)
```

---

## CLI Command Reference

### Task Commands

#### `got task create`
Create a new task in the graph.

**Syntax:**
```bash
got task create "title" [OPTIONS]

Options:
  --priority high|medium|low    Task priority (default: medium)
  --category CATEGORY           Task category (default: general)
                                Values: feature, bugfix, docs, refactor, test, devex, arch
  --effort small|medium|large   Estimated effort (default: medium)
  --sprint S-XXX                Assign to sprint (creates CONTAINS edge)
  --depends T-XXX [T-YYY...]   Task dependencies (creates DEPENDS_ON edges)
  --description TEXT            Detailed description
  --context-file FILE          Add file to context
  --context-method METHOD      Add method to context
```

**Examples:**
```bash
# Basic task creation
got task create "Implement authentication"

# Task with full metadata
got task create "Add JWT validation" \
  --priority high \
  --category feature \
  --effort medium \
  --sprint S-001 \
  --depends T-20251220-100000-a1b2 \
  --description "Validate JWT tokens in middleware" \
  --context-file cortical/auth.py \
  --context-method validate_jwt

# Quick high-priority bugfix
got task create "Fix login timeout" --priority high --category bugfix
```

**Output:**
```
✓ Created task: T-20251220-143052-a1b2
  Title: Implement authentication
  Priority: medium
  Category: general
  Status: pending

Graph updated:
  + Node: T-20251220-143052-a1b2 (TASK)
  + Edge: S-001 --[CONTAINS]--> T-20251220-143052-a1b2
```

---

#### `got task list`
List tasks with filtering and graph queries.

**Syntax:**
```bash
got task list [OPTIONS]

Options:
  --status STATUS              Filter by status (pending|in_progress|completed|blocked|deferred)
  --priority PRIORITY          Filter by priority (high|medium|low)
  --category CATEGORY          Filter by category
  --sprint S-XXX               Tasks in specific sprint
  --blocked                    Show only blocked tasks
  --no-sprint                  Show tasks not assigned to any sprint
  --sort-by FIELD              Sort by: created|updated|priority (default: created)
  --limit N                    Limit results (default: 50)
  --format text|json|table     Output format (default: table)
```

**Examples:**
```bash
# All pending tasks
got task list --status pending

# High-priority tasks in current sprint
got task list --priority high --sprint S-001

# All blocked tasks across all sprints
got task list --blocked

# Tasks not in any sprint
got task list --no-sprint

# JSON output for scripting
got task list --status in_progress --format json

# Latest 10 completed tasks
got task list --status completed --sort-by updated --limit 10
```

**Output (table format):**
```
┌──────────────────────────┬────────────────────────────┬──────────┬──────────┬──────────┐
│ Task ID                  │ Title                      │ Status   │ Priority │ Sprint   │
├──────────────────────────┼────────────────────────────┼──────────┼──────────┼──────────┤
│ T-20251220-100000-a1b2   │ Implement authentication   │ pending  │ high     │ S-001    │
│ T-20251220-101500-c3d4   │ Add JWT validation         │ progress │ high     │ S-001    │
│ T-20251220-103000-e5f6   │ Write auth tests           │ pending  │ medium   │ S-001    │
└──────────────────────────┴────────────────────────────┴──────────┴──────────┴──────────┘

3 tasks found
```

---

#### `got task show`
Show detailed task information with graph context.

**Syntax:**
```bash
got task show TASK_ID [OPTIONS]

Options:
  --deps                Show dependency graph
  --blocked-by          Show what's blocking this task
  --blocking            Show what this task blocks
  --sprint              Show sprint context
  --files               Show files in context
  --retrospective       Show retrospective notes (if completed)
```

**Examples:**
```bash
# Basic task info
got task show T-20251220-100000-a1b2

# Show with full dependency graph
got task show T-20251220-100000-a1b2 --deps

# Show only blocking relationships
got task show T-20251220-100000-a1b2 --blocked-by --blocking
```

**Output:**
```
Task: T-20251220-100000-a1b2
Title: Implement authentication
Status: in_progress
Priority: high
Category: feature
Effort: medium

Created: 2025-12-20 10:00:00
Updated: 2025-12-20 14:30:52
Sprint: S-001 (Authentication & Authorization)

Description:
  Add JWT-based authentication to API endpoints. Include token
  validation, refresh token handling, and session management.

Context Files:
  - cortical/auth.py
  - cortical/middleware.py

Dependencies:
  ← DEPENDS_ON: T-20251220-095000-c3d4 (Set up auth database) [completed]
  ← DEPENDS_ON: T-20251220-094500-g7h8 (Install JWT library) [completed]

Blocks:
  → BLOCKS: T-20251220-103000-e5f6 (Write auth tests) [pending]
  → BLOCKS: T-20251220-104500-i9j0 (Add auth to endpoints) [pending]

Sprint Goals:
  ← MOTIVATES: G-S-001-001 "Complete basic auth flow"
  ← MOTIVATES: G-S-001-002 "Support OAuth providers"
```

---

#### `got task start`
Mark a task as in-progress.

**Syntax:**
```bash
got task start TASK_ID
```

**Examples:**
```bash
got task start T-20251220-100000-a1b2
```

**Output:**
```
✓ Task T-20251220-100000-a1b2 marked as in_progress
  Previous status: pending
  Updated: 2025-12-20 14:35:22

Dependency check:
  ✓ All dependencies completed
```

**Warning cases:**
```
⚠ Warning: Task has uncompleted dependencies:
  - T-20251220-095000-c3d4: Set up auth database [pending]
  - T-20251220-094500-g7h8: Install JWT library [in_progress]

Start anyway? [y/N]: _
```

---

#### `got task complete`
Mark a task as completed with optional retrospective.

**Syntax:**
```bash
got task complete TASK_ID [OPTIONS]

Options:
  --retrospective TEXT         Completion notes/learnings
  --duration MINUTES           Time spent on task
  --files FILE [FILE...]       Files modified
  --tests N                    Number of tests added
  --commits SHA [SHA...]       Related commit SHAs
  --create-memory              Create memory entry from retrospective
```

**Examples:**
```bash
# Simple completion
got task complete T-20251220-100000-a1b2

# Complete with full retrospective
got task complete T-20251220-100000-a1b2 \
  --retrospective "JWT validation working. Learned about token refresh patterns." \
  --duration 120 \
  --files cortical/auth.py cortical/middleware.py tests/test_auth.py \
  --tests 5 \
  --commits abc123 def456 \
  --create-memory

# Quick completion with notes
got task complete T-20251220-100000-a1b2 \
  --retrospective "Fixed issue with token expiry handling"
```

**Output:**
```
✓ Task T-20251220-100000-a1b2 marked as completed
  Completed: 2025-12-20 15:00:00
  Duration: 120 minutes (2h 0m)

Retrospective captured:
  - Notes: JWT validation working. Learned about token refresh patterns.
  - Files: 3 files modified
  - Tests: 5 tests added
  - Commits: 2 commits linked

Unblocking:
  ✓ Unblocked 2 tasks:
    - T-20251220-103000-e5f6: Write auth tests
    - T-20251220-104500-i9j0: Add auth to endpoints

✓ Memory entry created: samples/memories/2025-12-20_15-00-00_a1b2-task-implement-authentication.md
```

---

#### `got task block`
Mark a task as blocked with reason.

**Syntax:**
```bash
got task block TASK_ID --reason TEXT [OPTIONS]

Options:
  --blocker TASK_ID            Link to blocking task (creates BLOCKS edge)
  --external                   External blocker (not a task)
```

**Examples:**
```bash
# Block with task dependency
got task block T-20251220-100000-a1b2 \
  --reason "Waiting for database schema approval" \
  --blocker T-20251220-095000-c3d4

# Block with external reason
got task block T-20251220-100000-a1b2 \
  --reason "Blocked by vendor API downtime" \
  --external
```

**Output:**
```
✓ Task T-20251220-100000-a1b2 marked as blocked
  Status: in_progress → blocked
  Reason: Waiting for database schema approval

Graph updated:
  + Edge: T-20251220-095000-c3d4 --[BLOCKS]--> T-20251220-100000-a1b2

⚠ This task is now blocking:
  - T-20251220-103000-e5f6: Write auth tests
  - T-20251220-104500-i9j0: Add auth to endpoints
```

---

#### `got task unblock`
Remove blocked status from a task.

**Syntax:**
```bash
got task unblock TASK_ID [OPTIONS]

Options:
  --remove-blocker TASK_ID     Remove specific blocker edge
  --all                        Remove all blocker edges
  --resume                     Resume as in_progress (default: pending)
```

**Examples:**
```bash
# Unblock and return to pending
got task unblock T-20251220-100000-a1b2

# Unblock and resume work
got task unblock T-20251220-100000-a1b2 --resume

# Remove specific blocker
got task unblock T-20251220-100000-a1b2 --remove-blocker T-20251220-095000-c3d4
```

---

#### `got task deps`
Show dependency graph for a task.

**Syntax:**
```bash
got task deps TASK_ID [OPTIONS]

Options:
  --depth N                    Traversal depth (default: 3)
  --format ascii|json|mermaid  Output format (default: ascii)
  --reverse                    Show reverse dependencies (what depends on this)
```

**Examples:**
```bash
# ASCII tree of dependencies
got task deps T-20251220-100000-a1b2

# Mermaid diagram for documentation
got task deps T-20251220-100000-a1b2 --format mermaid > deps.md

# Show what depends on this task
got task deps T-20251220-100000-a1b2 --reverse
```

**Output (ASCII format):**
```
Task: T-20251220-100000-a1b2 (Implement authentication)
└── Dependencies:
    ├── T-20251220-095000-c3d4 (Set up auth database) [✓ completed]
    │   └── T-20251220-092000-k1l2 (Design schema) [✓ completed]
    └── T-20251220-094500-g7h8 (Install JWT library) [✓ completed]

Reverse Dependencies (3 tasks depend on this):
    ├── T-20251220-103000-e5f6 (Write auth tests) [pending]
    ├── T-20251220-104500-i9j0 (Add auth to endpoints) [pending]
    └── T-20251220-105500-m3n4 (Document auth API) [pending]
```

---

### Sprint Commands

#### `got sprint create`
Create a new sprint.

**Syntax:**
```bash
got sprint create "name" [OPTIONS]

Options:
  --id SPRINT_ID               Custom sprint ID (default: auto-generated S-NNN)
  --epic E-XXX                 Link to epic (creates CONTAINS edge)
  --start DATE                 Start date (default: today)
```

**Examples:**
```bash
# Create sprint with auto ID
got sprint create "Authentication & Authorization"

# Create sprint in epic with custom ID
got sprint create "OAuth Integration" \
  --id S-002 \
  --epic E-AUTH \
  --start 2025-12-22
```

**Output:**
```
✓ Created sprint: S-001
  Name: Authentication & Authorization
  Status: available
  Started: 2025-12-20

Graph updated:
  + Node: S-001 (CONTEXT)
  + Edge: E-AUTH --[CONTAINS]--> S-001
```

---

#### `got sprint list`
List all sprints.

**Syntax:**
```bash
got sprint list [OPTIONS]

Options:
  --status STATUS              Filter by status (available|in_progress|complete)
  --epic E-XXX                 Sprints in specific epic
  --format text|json|table     Output format (default: table)
```

**Examples:**
```bash
# All sprints
got sprint list

# Only active sprints
got sprint list --status in_progress

# Sprints in epic
got sprint list --epic E-AUTH
```

---

#### `got sprint status`
Show sprint progress and statistics.

**Syntax:**
```bash
got sprint status [SPRINT_ID] [OPTIONS]

Options:
  --tasks                      Show task breakdown
  --goals                      Show goal progress
  --blocked                    Show blocked items
  --velocity                   Calculate velocity metrics
```

**Examples:**
```bash
# Current sprint status (if only one in_progress)
got sprint status

# Specific sprint
got sprint status S-001

# Full breakdown
got sprint status S-001 --tasks --goals --blocked --velocity
```

**Output:**
```
Sprint: S-001
Name: Authentication & Authorization
Status: in_progress
Epic: E-AUTH (Authentication System)
Started: 2025-12-01
Duration: 19 days

Goals Progress: 2/5 completed (40%)
  [x] Complete basic auth flow
  [x] Support OAuth providers
  [ ] Add 2FA support
  [ ] Implement session management
  [ ] Document auth API

Tasks: 12 total
  ✓ Completed: 5 (42%)
  → In Progress: 3 (25%)
  ○ Pending: 3 (25%)
  ⚠ Blocked: 1 (8%)

Velocity:
  - Completed: 5 tasks / 19 days = 0.26 tasks/day
  - Estimated completion: 2025-12-27 (7 days remaining)

Blocked Items:
  - T-20251220-100000-a1b2: Implement authentication
    Blocked by: Waiting for vendor API access
```

---

#### `got sprint add-goal`
Add a goal to a sprint.

**Syntax:**
```bash
got sprint add-goal SPRINT_ID "goal text" [OPTIONS]

Options:
  --link-task TASK_ID          Link task to goal (creates MOTIVATES edge)
```

**Examples:**
```bash
# Add goal
got sprint add-goal S-001 "Complete basic auth flow"

# Add goal linked to task
got sprint add-goal S-001 "Add 2FA support" \
  --link-task T-20251220-100000-a1b2
```

**Output:**
```
✓ Added goal to sprint S-001: G-S-001-003
  Goal: Add 2FA support
  Completed: false

Graph updated:
  + Node: G-S-001-003 (GOAL)
  + Edge: G-S-001-003 --[MOTIVATES]--> T-20251220-100000-a1b2
```

---

#### `got sprint goals`
List goals for a sprint.

**Syntax:**
```bash
got sprint goals SPRINT_ID [OPTIONS]

Options:
  --completed                  Show only completed goals
  --pending                    Show only pending goals
  --tasks                      Show tasks motivated by each goal
```

**Examples:**
```bash
# All goals
got sprint goals S-001

# Only pending goals with linked tasks
got sprint goals S-001 --pending --tasks
```

---

### Epic Commands

#### `got epic create`
Create a new epic.

**Syntax:**
```bash
got epic create "name" [OPTIONS]

Options:
  --id EPIC_ID                 Custom epic ID (default: auto-generated from name)
  --description TEXT           Epic description
```

**Examples:**
```bash
# Auto-generate ID from name
got epic create "Authentication System"
# Creates E-AUTHENTICATION-SYSTEM

# Custom ID
got epic create "Performance Optimization" --id E-PERF
```

---

#### `got epic list`
List all epics.

**Syntax:**
```bash
got epic list [OPTIONS]

Options:
  --status STATUS              Filter by status (active|complete|archived)
  --format text|json|table     Output format (default: table)
```

---

#### `got epic status`
Show epic progress.

**Syntax:**
```bash
got epic status EPIC_ID [OPTIONS]

Options:
  --sprints                    Show sprint breakdown
  --tasks                      Show all tasks in epic (across sprints)
```

**Examples:**
```bash
got epic status E-AUTH --sprints --tasks
```

**Output:**
```
Epic: E-AUTH
Name: Authentication System
Status: active
Created: 2025-12-01

Sprints: 2 total
  → In Progress: 1 (S-001: Authentication & Authorization)
  ✓ Completed: 1 (S-000: Initial setup)

Tasks: 25 total (across all sprints)
  ✓ Completed: 12 (48%)
  → In Progress: 5 (20%)
  ○ Pending: 7 (28%)
  ⚠ Blocked: 1 (4%)

Completion Estimate: 72% complete
```

---

#### `got epic sprints`
List sprints in an epic.

**Syntax:**
```bash
got epic sprints EPIC_ID [OPTIONS]

Options:
  --status STATUS              Filter by sprint status
```

---

### Query Commands

#### `got blocked`
Show all blocked items across the graph.

**Syntax:**
```bash
got blocked [OPTIONS]

Options:
  --sprint SPRINT_ID           Only blocked items in sprint
  --epic EPIC_ID               Only blocked items in epic
  --format text|json|table     Output format (default: table)
```

**Examples:**
```bash
# All blocked tasks
got blocked

# Blocked tasks in current sprint
got blocked --sprint S-001
```

**Output:**
```
Blocked Tasks: 3 found

┌──────────────────────────┬────────────────────────────┬──────────────────────────────────┐
│ Task ID                  │ Title                      │ Blocked By                       │
├──────────────────────────┼────────────────────────────┼──────────────────────────────────┤
│ T-20251220-100000-a1b2   │ Implement authentication   │ External: Vendor API access      │
│ T-20251220-103000-e5f6   │ Write auth tests           │ Task: T-20251220-100000-a1b2     │
│ T-20251220-104500-i9j0   │ Add auth to endpoints      │ Task: T-20251220-100000-a1b2     │
└──────────────────────────┴────────────────────────────┴──────────────────────────────────┘
```

---

#### `got active`
Show all in-progress items.

**Syntax:**
```bash
got active [OPTIONS]

Options:
  --sprint SPRINT_ID           Only active items in sprint
  --epic EPIC_ID               Only active items in epic
  --format text|json|table     Output format (default: table)
```

---

#### `got graph`
Export the full graph for visualization or analysis.

**Syntax:**
```bash
got graph [OPTIONS]

Options:
  --output FILE                Output file (default: stdout)
  --format json|dot|mermaid    Export format (default: json)
  --filter TYPE                Filter by node type (task|sprint|epic|goal)
  --sprint SPRINT_ID           Only nodes related to sprint
  --epic EPIC_ID               Only nodes related to epic
```

**Examples:**
```bash
# Export full graph as JSON
got graph --output project.json

# Export task dependency graph as Mermaid
got graph --format mermaid --filter task > tasks.md

# Export sprint S-001 subgraph as DOT for Graphviz
got graph --format dot --sprint S-001 > sprint.dot
```

---

### Sync & Migration Commands

#### `got sync`
Bidirectional sync with file-based system during transition.

**Syntax:**
```bash
got sync [OPTIONS]

Options:
  --from files                 Import from tasks/*.json to graph
  --from graph                 Export from graph to tasks/*.json
  --dry-run                    Show what would change without applying
  --validate                   Validate consistency between systems
```

**Examples:**
```bash
# Import file-based tasks to graph
got sync --from files

# Export graph to files for compatibility
got sync --from graph

# Check consistency
got sync --validate
```

**Output (sync --from files):**
```
Syncing from files → graph...

Found 25 task files in tasks/
  - 12 tasks already in graph (no changes)
  - 8 tasks updated in graph
  - 5 new tasks added to graph

Graph operations logged to WAL:
  + 5 add_node operations
  + 8 update_node operations
  + 13 add_edge operations (dependencies)

✓ Sync complete
  Graph snapshot created: snap_20251220_150000
  Git commit: sync: Import 5 new tasks from files
```

---

#### `got migrate`
One-time migration from file-based system to graph.

**Syntax:**
```bash
got migrate [OPTIONS]

Options:
  --source DIR                 Source directory (default: tasks/)
  --backup                     Create backup before migration
  --preserve-files             Keep files after migration (default: remove)
  --validate                   Validate migration completeness
```

**Examples:**
```bash
# Full migration with backup
got migrate --backup --validate

# Migrate but keep files for safety
got migrate --preserve-files
```

**Output:**
```
Migration: tasks/*.json → GoT Graph

Pre-migration backup:
  ✓ Created: tasks_backup_20251220_150000.tar.gz

Analyzing source:
  - 25 task files found
  - 3 sprint tracking files found
  - 45 total tasks across all sessions

Migration plan:
  1. Create 45 task nodes (TASK)
  2. Create 3 sprint nodes (CONTEXT)
  3. Create 12 goal nodes (GOAL)
  4. Create 67 dependency edges (DEPENDS_ON)
  5. Create 38 containment edges (CONTAINS)
  6. Create 24 motivation edges (MOTIVATES)

Proceed? [y/N]: y

Migrating...
  ✓ Created 45 task nodes
  ✓ Created 3 sprint nodes
  ✓ Created 12 goal nodes
  ✓ Created 67 dependency edges
  ✓ Created 38 containment edges
  ✓ Created 24 motivation edges

Validation:
  ✓ All tasks migrated (45/45)
  ✓ All sprints migrated (3/3)
  ✓ All goals migrated (12/12)
  ✓ All dependencies preserved (67/67)
  ✓ Graph integrity check passed

Graph persisted:
  - WAL: project_wal/wal_20251220.log (189 entries)
  - Snapshot: project_wal/snapshots/snap_20251220_150000.json.gz
  - Git commit: migrate: Import project from file-based system

✓ Migration complete
  Old files preserved in: tasks_backup_20251220_150000.tar.gz
  Graph ready at: project_wal/
```

---

#### `got validate`
Validate graph integrity and consistency.

**Syntax:**
```bash
got validate [OPTIONS]

Options:
  --fix                        Auto-fix issues where possible
  --report FILE                Write validation report to file
```

**Examples:**
```bash
# Check integrity
got validate

# Check and fix issues
got validate --fix
```

**Output:**
```
Graph Integrity Validation

Checking node integrity...
  ✓ 45 task nodes verified
  ✓ 3 sprint nodes verified
  ✓ 12 goal nodes verified
  ✓ 2 epic nodes verified

Checking edge integrity...
  ✓ 67 DEPENDS_ON edges verified
  ✓ 38 CONTAINS edges verified
  ✓ 24 MOTIVATES edges verified
  ⚠ 1 orphaned edge found:
    - DEPENDS_ON: T-20251220-100000-a1b2 → T-DELETED (target missing)

Checking consistency...
  ✓ All tasks have unique IDs
  ✓ All sprint references valid
  ✓ No circular dependencies detected
  ⚠ 1 task assigned to non-existent sprint:
    - T-20251220-103000-e5f6 references S-999

Issues found: 2
  - 1 orphaned edge
  - 1 invalid sprint reference

Fix issues? [y/N]: y

Fixing...
  ✓ Removed orphaned edge: T-20251220-100000-a1b2 → T-DELETED
  ✓ Cleared invalid sprint reference: T-20251220-103000-e5f6

✓ Graph validated and fixed
  Integrity: 100%
  Snapshot created: snap_20251220_150500
```

---

## Configuration File

The CLI can be configured via `~/.config/got/config.toml`:

```toml
[storage]
# Graph storage directory
wal_dir = "project_wal"
chunks_dir = "project_chunks"

# Snapshot settings
max_snapshots = 3
auto_snapshot = true
snapshot_interval_operations = 100

[git]
# Auto-commit settings
auto_commit = true
commit_mode = "debounced"  # immediate, debounced, manual
debounce_seconds = 5
auto_push = false
protected_branches = ["main", "master"]

[defaults]
# Default values for task creation
task_priority = "medium"
task_category = "general"
task_effort = "medium"

# CLI behavior
output_format = "table"  # table, json, text
color = true

[sync]
# Sync with file-based system
enabled = true
tasks_dir = "tasks"
sprint_file = "tasks/CURRENT_SPRINT.md"
```

---

## Error Handling

All commands follow consistent error handling:

### Exit Codes
- `0` - Success
- `1` - General error
- `2` - Invalid arguments
- `3` - Graph integrity error
- `4` - Persistence failure
- `5` - Migration/sync error

### Error Output Format
```
Error: Task not found: T-20251220-100000-INVALID

Graph state:
  - 45 tasks in graph
  - Last snapshot: snap_20251220_150000
  - WAL entries: 189

Suggestions:
  - Check task ID with: got task list
  - Verify graph integrity: got validate
  - Restore from snapshot: got recover
```

---

## Performance Characteristics

**Task Creation:**
- Graph operations: O(1) node add + O(E) edge adds
- WAL append: O(1)
- Expected time: <10ms

**Task Listing:**
- Graph traversal: O(N) where N = filtered nodes
- With sprint filter: O(1) edge lookup + O(M) edge traversal
- Expected time: <50ms for 1000 tasks

**Dependency Queries:**
- DFS traversal: O(V + E) where V = nodes in subgraph
- Depth-limited: O(D * avg_degree)
- Expected time: <20ms for depth 5

**Sync Operations:**
- File-to-graph: O(N * M) where N = files, M = avg tasks per file
- Graph-to-file: O(T) where T = total tasks
- Expected time: <500ms for 100 tasks

**Migration:**
- Initial load: O(N * M)
- Graph construction: O(T + D) where D = dependencies
- Expected time: <2s for 500 tasks

---

## Recovery Procedures

### Recover from Crash
```bash
# Check if recovery needed
got validate

# If validation fails, attempt recovery
got recover

# Recovery uses 4-level cascade:
#   1. WAL replay (from latest snapshot)
#   2. Previous snapshots
#   3. Git history
#   4. Chunk reconstruction
```

### Restore from Snapshot
```bash
# List available snapshots
got snapshots

# Restore specific snapshot
got restore --snapshot snap_20251220_150000
```

### Export/Import
```bash
# Export entire graph
got export --output backup.json

# Import from backup
got import --input backup.json --merge
```

---

## Implementation Notes

### Key Files to Create
1. `scripts/got_utils.py` - Main CLI entry point
2. `scripts/got_graph_manager.py` - Graph management wrapper
3. `scripts/got_sync.py` - File-based sync logic
4. `scripts/got_migration.py` - Migration utilities

### Dependencies
- `cortical.reasoning.thought_graph` - Graph data structure
- `cortical.reasoning.graph_persistence` - WAL, snapshots, git
- `scripts.task_utils` - For migration/sync with file-based system

### Testing Strategy
1. **Unit tests** - Each command's graph operations
2. **Integration tests** - WAL persistence, recovery
3. **Migration tests** - File → Graph → File roundtrip
4. **Performance tests** - Large graphs (1000+ tasks)

---

## Migration Path

### Phase 1: Parallel Systems (Weeks 1-2)
- Implement `got` CLI with full feature parity
- Run both systems in parallel
- Use `got sync` to keep in sync
- Validate consistency

### Phase 2: Graph-Primary (Weeks 3-4)
- Switch to graph as source of truth
- File-based system becomes export target
- Continue sync for compatibility

### Phase 3: Graph-Only (Week 5+)
- Deprecate file-based commands
- Remove sync logic
- Archive old task files

---

## Future Enhancements

### Advanced Queries
```bash
# Find tasks by natural language
got query "authentication tasks that are blocked"

# Complex graph queries
got query --cypher "MATCH (t:TASK)-[:DEPENDS_ON*]->(d:TASK) WHERE d.status='pending' RETURN t"
```

### Analytics
```bash
# Task completion velocity
got analytics velocity --sprint S-001

# Dependency complexity metrics
got analytics complexity

# Blocked task trends
got analytics blocked --last 30d
```

### AI Integration
```bash
# Suggest next tasks based on graph
got suggest --sprint S-001

# Detect dependency cycles
got analyze cycles

# Estimate completion dates
got forecast --sprint S-001
```

---

**End of Specification**

This design provides a complete, graph-native project management CLI with:
- ✓ Full feature parity with file-based system
- ✓ Rich graph queries and traversals
- ✓ Durable persistence via WAL
- ✓ Safe migration path
- ✓ Extensible for future enhancements
