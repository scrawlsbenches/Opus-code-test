# Sprint Reasoning Graph Integration

This document describes how to use the sprint reasoning system that connects Graph of Thought (GoT) task tracking with the ThoughtGraph reasoning framework.

## Overview

The sprint reasoning integration provides structured reasoning during sprint execution by:

1. Loading sprint goals from `tasks/CURRENT_SPRINT.md`
2. Loading associated tasks from the GoT system
3. Creating a ThoughtGraph representing the sprint structure
4. Running a QAPV (Question-Answer-Produce-Verify) reasoning session

## Quick Start

```bash
# List all available sprints
python scripts/run_sprint_reasoning.py --list-sprints

# Run reasoning session for a sprint
python scripts/run_sprint_reasoning.py --sprint sprint-020-forensic-remediation

# Show the thought graph structure
python scripts/run_sprint_reasoning.py --sprint sprint-020-forensic-remediation --show-graph

# Output as JSON
python scripts/run_sprint_reasoning.py --sprint sprint-020-forensic-remediation --json
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--sprint SPRINT_ID` | Sprint ID to execute (e.g., `sprint-020-forensic-remediation`) |
| `--list-sprints` | List all available sprints from CURRENT_SPRINT.md |
| `--show-graph` | Display the ThoughtGraph structure in ASCII format |
| `--json` | Output sprint data in JSON format |
| `--spawn-agents` | Generate sub-agent Task tool configurations for parallel execution |
| `--show-boundaries` | Show work boundaries for conflict prevention (with `--spawn-agents`) |

## How It Works

### 1. Sprint Loading

The script parses `tasks/CURRENT_SPRINT.md` to extract:
- Sprint metadata (ID, status, epic)
- Goals with completion status (checkbox items)
- GoT task IDs referenced in the sprint

### 2. Graph Construction

A ThoughtGraph is created with:

**Node Types:**
- `GOAL` - The sprint itself (root node)
- `QUESTION` - Incomplete goals (need work)
- `DECISION` - Completed goals
- `HYPOTHESIS` - GoT tasks

**Edge Types:**
- `DEPENDS_ON` - Sprint → Goal relationships
- `SUPPORTS` - Sprint → Task relationships
- `PRECEDES` - Task → Task sequential dependencies

### 3. QAPV Reasoning Session

The session follows the cognitive loop phases:

```
┌─────────────────────────────────────────────────────────────┐
│ QUESTION PHASE                                              │
│ - Lists all incomplete goals                                │
│ - Identifies what needs to be done                          │
├─────────────────────────────────────────────────────────────┤
│ ANSWER PHASE                                                │
│ - Analyzes goals and breaks into tasks                      │
│ - Shows task count and recommended execution order          │
├─────────────────────────────────────────────────────────────┤
│ PRODUCE PHASE                                               │
│ - Provides actionable commands to start work                │
│ - Shows git branch and task start commands                  │
├─────────────────────────────────────────────────────────────┤
│ VERIFY PHASE                                                │
│ - Shows verification checklist                              │
│ - Defines success criteria for sprint completion            │
└─────────────────────────────────────────────────────────────┘
```

## Example Output

```
============================================================
REASONING SESSION
============================================================

Session started: 435a3f57

--- QUESTION PHASE ---
  ❓ Complete ID generation migration to canonical module
  ❓ Consolidate WAL implementations (got/wal.py → cortical/wal.py)
  ❓ Create cortical/utils/checksums.py (consolidate 6+ duplicates)

--- ANSWER PHASE ---
  Analysis: Breaking down goals into actionable tasks
  📋 6 tasks identified in GoT
  📊 8 goals remaining

  Recommended execution order:
    1. Complete ID generation migration to canonical module
    2. Consolidate WAL implementations
    ...

--- PRODUCE PHASE ---
  Ready to execute. Run the following to start work:

  1. Switch to sprint branch:
     git checkout -b claude/sprint-020-forensic-remediation-$(date +%s)

  2. Start first task:
     python scripts/got_utils.py task start T-20251222-025531-e6e222a1

--- VERIFY PHASE ---
  Verification checklist:
    □ All goals marked complete in CURRENT_SPRINT.md
    □ All GoT tasks marked complete
    □ Tests pass: python -m pytest tests/ -q
    □ Coverage maintained: 88%+
    □ GoT validation healthy: python scripts/got_utils.py validate

============================================================
SESSION COMPLETE
============================================================
```

## Graph Visualization

With `--show-graph`, you get ASCII visualization:

```
============================================================
THOUGHT GRAPH STRUCTURE
============================================================

Nodes: 14
  [GOAL] Sprint: sprint-020-forensic-remediation
  [QUESTION] Complete ID generation migration to canonical module
  [QUESTION] Consolidate WAL implementations
  [HYPOTHESIS] Task T-20251222-025531-e6e222a1
  ...

Edges: 18
  sprint-sprint-0 --DEPENDS_ON--> goal-0
  sprint-sprint-0 --SUPPORTS--> task-T-20251222
  task-T-20251222 --PRECEDES--> task-T-20251222
  ...

--- ASCII Graph ---
[GOAL] Sprint: sprint-020-forensic-remediation
├── [QUESTION] Complete ID generation migration... (depends_on)
├── [QUESTION] Consolidate WAL implementations... (depends_on)
└── [HYPOTHESIS] Task T-20251222-025531-e6e222a1 (supports)
```

## Integration with GoT

### Adding Tasks to a Sprint

1. Create tasks in GoT:
   ```bash
   python scripts/got_utils.py task create "Your task title" --priority high
   ```

2. Add the task ID to your sprint in `tasks/CURRENT_SPRINT.md`:
   ```markdown
   ### GoT Task IDs
   - T-20251222-025531-e6e222a1: Your task title
   ```

3. Run the reasoning session to see the integrated view.

### Task Lifecycle

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   pending   │ ──> │ in_progress │ ──> │  completed  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       │                   │                   │
       v                   v                   v
   HYPOTHESIS          HYPOTHESIS          DECISION
   (in graph)          (in graph)         (in graph)
```

## Creating a New Sprint

1. Add sprint section to `tasks/CURRENT_SPRINT.md`:
   ```markdown
   ## Sprint N: Your Sprint Name
   **Sprint ID:** sprint-NNN-your-sprint-name
   **Epic:** Your Epic (epic-id)
   **Status:** Available 🟢

   ### Goals
   - [ ] First goal
   - [ ] Second goal

   ### GoT Task IDs
   - T-XXXXXXXX-XXXXXX-XXXXXXXX: Task description
   ```

2. Create corresponding GoT tasks:
   ```bash
   python scripts/got_utils.py task create "First goal" --priority high
   python scripts/got_utils.py task create "Second goal" --priority medium
   ```

3. Run the reasoning session:
   ```bash
   python scripts/run_sprint_reasoning.py --sprint sprint-NNN-your-sprint-name
   ```

## Sub-Agent Spawning

The script can generate configurations for parallel sub-agents to execute sprint tasks:

```bash
# Generate sub-agent configurations
python scripts/run_sprint_reasoning.py --sprint sprint-020-forensic-remediation --spawn-agents

# Show work boundaries to prevent conflicts
python scripts/run_sprint_reasoning.py --sprint sprint-020-forensic-remediation --spawn-agents --show-boundaries
```

### Communication Infrastructure

When `--spawn-agents` is used, the script sets up:

1. **PubSub Broker** - Async messaging between agents
   - Topics: `task.started`, `task.completed`, `task.blocked`, `discovery.*`

2. **Context Pool** - Shared findings with confidence scores
   - Agents publish discoveries that other agents can query

3. **Collaboration Manager** - Work boundaries and blockers
   - Prevents multiple agents from modifying the same files

### Work Boundaries

File ownership is automatically assigned based on task content:

| Task Pattern | Owned Files |
|--------------|-------------|
| "id generation" | `cortical/utils/id_generation.py`, `scripts/orchestration_utils.py` |
| "wal" | `cortical/wal.py`, `cortical/got/wal.py` |
| "checksum" | `cortical/utils/checksums.py` |
| "query" | `cortical/query/` |
| "atomic" | `cortical/utils/persistence.py` |
| "slugify" | `cortical/utils/text.py` |

### Agent Configuration Output

Each agent configuration includes:
- **agent_id**: Unique identifier for tracking
- **description**: Short task description
- **prompt**: Full task instructions with sprint context
- **subagent_type**: Agent type (default: `general-purpose`)
- **boundary**: Optional file ownership for conflict prevention

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Sprint Reasoning System                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐    ┌──────────────────┐               │
│  │ CURRENT_SPRINT.md│    │   GoT System     │               │
│  │ (Sprint Goals)   │    │ (Task Tracking)  │               │
│  └────────┬─────────┘    └────────┬─────────┘               │
│           │                       │                          │
│           └───────────┬───────────┘                          │
│                       │                                      │
│                       v                                      │
│           ┌───────────────────────┐                          │
│           │    ThoughtGraph       │                          │
│           │ (Reasoning Structure) │                          │
│           └───────────┬───────────┘                          │
│                       │                                      │
│                       v                                      │
│           ┌───────────────────────┐                          │
│           │  ReasoningWorkflow    │                          │
│           │   (QAPV Session)      │                          │
│           └───────────┬───────────┘                          │
│                       │                                      │
│      ┌────────────────┼────────────────┐                     │
│      v                v                v                     │
│  ┌────────┐      ┌────────┐      ┌────────┐                  │
│  │Agent 1 │      │Agent 2 │      │Agent N │                  │
│  │(Task)  │      │(Task)  │      │(Task)  │                  │
│  └────┬───┘      └────┬───┘      └────┬───┘                  │
│       │               │               │                      │
│       └───────────────┼───────────────┘                      │
│                       v                                      │
│           ┌───────────────────────┐                          │
│           │  Communication Layer  │                          │
│           │  (PubSub + ContextPool)│                         │
│           └───────────────────────┘                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Related Documentation

- [Graph of Thought](graph-of-thought.md) - Reasoning framework details
- [Complex Reasoning Workflow](complex-reasoning-workflow.md) - QAPV cycle documentation
- [Merge-Friendly Tasks](merge-friendly-tasks.md) - Task management system
- [CURRENT_SPRINT.md](../tasks/CURRENT_SPRINT.md) - Active sprint definitions

## Troubleshooting

### "Sprint not found"

Ensure the sprint ID matches exactly what's in `CURRENT_SPRINT.md`:
```bash
python scripts/run_sprint_reasoning.py --list-sprints
```

### "Could not import GoT manager"

This warning is non-fatal. The script will still work with sprint goals, but won't load detailed task metadata from GoT.

### Empty graph

Check that your sprint has:
- Goals defined with checkbox format: `- [ ] Goal text`
- GoT task IDs in the expected format: `T-YYYYMMDD-HHMMSS-XXXXXXXX`
