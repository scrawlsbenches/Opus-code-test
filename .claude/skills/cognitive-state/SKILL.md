---
name: cognitive-state
description: Manage cognitive state across sessions and branches. Use when confused, recovering from context loss, or coordinating work across multiple branches. Provides automated recovery and state persistence.
allowed-tools: Read, Bash, Write, Grep, Glob
---
# Cognitive State Management Skill

This skill enables **cognitive continuity** across sessions, branches, and context windows using Graph of Thought (GoT) for reasoning persistence.

## When to Use

- **Session start**: Orient yourself to current state
- **Context confusion**: Detect and recover from cognitive breakdown
- **Branch switching**: Preserve state before moving between branches
- **Multi-branch work**: Understand relationships between parallel work streams
- **Task resumption**: Pick up complex work from a previous session

## Core Capabilities

### 1. State Assessment

Quickly understand current cognitive state:

```bash
# What branch am I on?
git branch -vv

# What's the branch state?
cat .branch-state/active/*.json 2>/dev/null || echo "No active branch state"

# What tasks are in progress?
python scripts/task_utils.py list --status in_progress

# What did I do recently?
git log --oneline -10

# What memories are recent?
ls -t samples/memories/*.md 2>/dev/null | head -5
```

### 2. Breakdown Detection

Signs of cognitive breakdown:

| Signal | Type | Recovery Action |
|--------|------|-----------------|
| Repeating failed approach | **LOOP** | Stop, analyze patterns, try different approach |
| Contradicting self | **CONFUSION** | Re-read context, reconcile state |
| Acting without reading | **PREMATURE** | Read first, understand, then act |
| Re-asking answered questions | **CONTEXT LOSS** | Check memories and task history |
| Generating placeholders | **UNCERTAINTY** | Admit gap, ask for specific info |

### 3. Recovery Protocol

When breakdown is detected:

```
STEP 1: DETECT
└── Identify: What type of breakdown is this?

STEP 2: STOP
└── Immediately halt current action
└── Do NOT "push through" or guess

STEP 3: DIAGNOSE
└── What was I attempting?
└── What state am I actually in?
└── What information do I lack?

STEP 4: INFORM USER
└── "I've detected [BREAKDOWN TYPE]"
└── "I was attempting [X] but [PROBLEM]"
└── "I need [SPECIFIC INFO] to proceed"

STEP 5: RECOVER
└── Load state from: .branch-state/, tasks/, memories/
└── If unavailable: Request from user explicitly
└── Document recovery in memory entry

STEP 6: VERIFY
└── Confirm state is now consistent
└── Run sanity checks before resuming
```

### 4. Branch Topology Awareness

Before merging or switching branches:

```bash
# Understand the relationship
git log --oneline HEAD..origin/other-branch | head -10  # Incoming
git log --oneline origin/other-branch..HEAD | head -10  # Unique here

# Check for state conflicts
git diff HEAD...origin/other-branch --stat | grep -E "(tasks/|memories/|.branch-state/)"

# Visualize branch structure
git log --oneline --graph --all | head -20
```

### 5. GoT Reasoning Persistence

For complex multi-step tasks, use CognitiveLoop:

```python
from cortical.reasoning import CognitiveLoop

# Create and track reasoning
loop = CognitiveLoop(goal="Complex feature implementation")
loop.start()

# Record reasoning explicitly
loop.record_question("What approach should we use?")
loop.record_decision("Use pattern X", rationale="Because of constraint Y")

# Serialize for next session
state = loop.serialize()
# Save to file for persistence
```

## Recovery Commands

### Quick State Check
```bash
echo "=== CURRENT STATE ===" && \
git branch -vv && \
echo "=== RECENT COMMITS ===" && \
git log --oneline -5 && \
echo "=== IN-PROGRESS TASKS ===" && \
python scripts/task_utils.py list --status in_progress 2>/dev/null
```

### Full Context Recovery
```bash
# Branch state
cat .branch-state/active/*.json 2>/dev/null

# Recent memories
for f in $(ls -t samples/memories/*.md 2>/dev/null | head -3); do
  echo "=== $f ===" && head -20 "$f"
done

# Task context
python scripts/task_utils.py list --limit 10
```

### Verify Reasoning System
```bash
python scripts/reasoning_demo.py --quick 2>&1 | tail -20
```

## Communication Template

When recovering, communicate clearly:

```markdown
## Cognitive State Report

**Breakdown Detected:** [TYPE - loop/confusion/context loss/premature action]

**What I Understand:**
- Current branch: [branch name]
- Last successful action: [action]
- Current goal: [goal]

**What's Unclear:**
- [Specific uncertainty 1]
- [Specific uncertainty 2]

**What I Need:**
- [Specific information request 1]
- [Specific information request 2]

**Recovery Action:**
Proceeding to [specific verification command]
```

## Backup Plans

| Primary | Backup | Fallback |
|---------|--------|----------|
| `.branch-state/` files | Git log analysis | Ask user |
| GoT saved state | Reconstruct from tasks | Fresh start with acknowledgment |
| Incremental updates | Full recompute | Manual verification |
| Parallel agents | Sequential execution | Single agent with checkpoints |

## Integration with Other Skills

- **task-manager**: Track work state across sessions
- **memory-manager**: Persist learnings and decisions
- **codebase-search**: Find relevant context in code
- **corpus-indexer**: Keep search index current

## Security Model

### Tool Permissions

| Tool | Why Needed | Scope |
|------|------------|-------|
| **Read** | Read state files, memories, configs | `.branch-state/`, `tasks/`, `samples/` |
| **Bash** | Run git commands, Python utilities | State queries only |
| **Write** | Create recovery checkpoints | `samples/memories/`, `.branch-state/` |
| **Grep** | Search for context patterns | Project-wide |
| **Glob** | Find state files | Project-wide |

### Safety Guarantees

1. **Read before write**: Always assess state before modifying
2. **Explicit communication**: Never silently recover - inform user
3. **Checkpoint before action**: Save state before risky operations
4. **Verify after recovery**: Confirm state consistency

## Quick Reference

| Need | Command |
|------|---------|
| Current branch | `git branch -vv` |
| Branch state | `cat .branch-state/active/*.json` |
| In-progress tasks | `python scripts/task_utils.py list --status in_progress` |
| Recent commits | `git log --oneline -5` |
| Recent memories | `ls -t samples/memories/*.md \| head -5` |
| Test reasoning | `python scripts/reasoning_demo.py --quick` |
| Full recovery | `/context-recovery` command |
