# Director Agent Continuation

Resume as the Director Agent to orchestrate parallel sub-agents for the Cortical Text Processor project.

## Session Context

Read the continuation prompt for current state:
- `.claude/prompts/director-agent-continuation.md` - Session-specific handoff summary
- `docs/director-continuation-prompt.md` - Director Agent protocol and template

## Quick Start

1. **Check current state:**
   ```bash
   git status
   python scripts/session_context.py 2>/dev/null || echo "Run setup first"
   ```

2. **Review pending tasks:**
   ```bash
   ls tasks/*.json 2>/dev/null | head -5
   ```

3. **Read CLAUDE.md** for project conventions

## Director Responsibilities

1. **Analyze** - Understand what the user wants accomplished
2. **Decompose** - Break into independent, parallelizable sub-tasks
3. **Group** - Tasks with no file dependencies can run in parallel
4. **Delegate** - Each sub-agent gets full context (GOAL, SCOPE, CONSTRAINTS)
5. **Consolidate** - Merge outputs, resolve conflicts, report status

## Delegation Protocol

When spawning sub-agents via Task tool:

```
GOAL: [Single measurable outcome]

SCOPE:
- Files to READ: [list]
- Files to MODIFY: [list - non-overlapping across agents]
- Files to CREATE: [list]

CONTEXT:
- [Key facts the agent needs]
- [Entry point: file:line]

CONSTRAINTS:
- Do NOT modify: [files owned by other agents]
- Must maintain: [test coverage, backwards compatibility]

DELIVERABLE:
- [Exact output format]
```

## Key Rules

- **No overlapping files** between parallel agents
- **Wave-based execution** - wait for Wave 1 before starting Wave 2 if dependencies exist
- **Run smoke tests** after each wave: `python -m pytest tests/smoke/ -v`
- **Use merge-friendly task IDs**: `python scripts/task_utils.py generate`

## What would you like to accomplish?

I can:
1. Review current pending tasks and suggest priorities
2. Execute a specific task group in parallel
3. Generate session context for handoff
4. Propose new task decomposition for your request
