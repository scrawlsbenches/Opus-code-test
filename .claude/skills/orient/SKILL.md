---
name: orient
description: Quick session orientation. Use at the start of any session to understand current state, pending handoffs, and recent context. Combines handoff check, cognitive agent query, and git status into one command.
allowed-tools: Read, Bash, Grep, Glob
---
# Session Orientation Skill

This skill provides **rapid context acquisition** at the start of a new session.

## When to Use

- **Every session start**: Run this first to orient yourself
- **After context loss**: When confused about what you were doing
- **Before major work**: Ensure you understand the current state

## What This Skill Does

1. **Checks pending handoffs** - Shows any initiated handoffs waiting for you
2. **Queries cognitive agent** - Asks what previous session was working on
3. **Shows git state** - Current branch, recent commits, modified files
4. **Validates system** - Runs GoT validate to ensure integrity

## Quick Orientation

Run these commands in sequence:

### Step 1: Check Pending Handoffs
```bash
echo "=== PENDING HANDOFFS ==="
python -m cortical.got handoff list --status initiated 2>/dev/null || echo "No handoff system or no pending handoffs"
```

### Step 2: Query Cognitive Agent
```bash
echo "=== COGNITIVE AGENT CONTEXT ==="
./scripts/bootstrap_cognitive.sh --check 2>/dev/null && \
python -m cortical.cognitive ask "What was the previous session working on?" 2>/dev/null || \
echo "Cognitive agent not available"
```

### Step 3: Git State
```bash
echo "=== GIT STATE ==="
git branch --show-current
git log --oneline -5
git status --short
```

### Step 4: System Validation
```bash
echo "=== SYSTEM VALIDATION ==="
python -m cortical.got validate 2>/dev/null | head -20
```

## Full Orientation Command

Copy and run this all-in-one command:

```bash
echo "╔══════════════════════════════════════════════════════════════╗" && \
echo "║              SESSION ORIENTATION REPORT                       ║" && \
echo "╚══════════════════════════════════════════════════════════════╝" && \
echo "" && \
echo "📋 PENDING HANDOFFS:" && \
python -m cortical.got handoff list --status initiated --limit 5 2>/dev/null || echo "  None" && \
echo "" && \
echo "🧠 COGNITIVE CONTEXT:" && \
python -m cortical.cognitive ask "What was previous session working on?" 2>/dev/null | head -5 || echo "  Not available" && \
echo "" && \
echo "📁 GIT STATE:" && \
echo "  Branch: $(git branch --show-current)" && \
echo "  Recent commits:" && \
git log --oneline -3 | sed 's/^/    /' && \
echo "" && \
echo "✅ VALIDATION:" && \
python -m cortical.got validate 2>/dev/null | grep -E "(HEALTHY|ERROR|WARNING)" || echo "  OK"
```

## Handling Results

### If Handoffs Found
```bash
# Review the handoff
python -m cortical.got handoff show H-XXXXXXXX

# Accept it
python -m cortical.got handoff accept H-XXXXXXXX --agent "me"
```

### If Cognitive Agent Reports Issues
```bash
# Get more context
python -m cortical.cognitive ask "What bugs were found recently?"
python -m cortical.cognitive query "current task"
```

### If Validation Fails
```bash
# Run recovery
python -m cortical.got recover

# Re-validate
python -m cortical.got validate
```

## Integration with Entry Gate

This skill implements steps from the Entry Gate Checklist in CLAUDE.md:

```
□ 1. SMOKE TESTS PASS (skip if scratchpad says not to)
□ 2. GOT VALIDATES CLEANLY  ← /orient checks this
□ 3. CHECK FOR EXISTING WORK ← /orient checks this
□ 4. QUERY COGNITIVE AGENT  ← /orient does this
```

## Output Template

After running orientation, you should be able to fill in:

```markdown
## Session Orientation Complete

**Current Branch:** [branch name]
**Pending Handoffs:** [count] - [most important one]
**Previous Work:** [summary from cognitive agent]
**System Status:** [healthy/issues]

**Ready to proceed:** [yes/no, with blockers if any]
```

## Tool Permissions

| Tool | Why Needed | Scope |
|------|------------|-------|
| **Read** | Read handoff details, cognitive results | `.got/`, `samples/` |
| **Bash** | Run git, python commands | Read-only queries |
| **Grep** | Search for context | Project-wide |
| **Glob** | Find relevant files | Project-wide |

## Best Practices

1. **Run at session start** - Before making any changes
2. **Accept pending handoffs** - Don't leave them hanging
3. **Note blockers** - If validation fails, fix before proceeding
4. **Query cognitive agent** - It knows more than you think

## Quick Reference

| Need | Command |
|------|---------|
| Pending handoffs | `python -m cortical.got handoff list --status initiated` |
| Previous context | `python -m cortical.cognitive ask "What was previous session working on?"` |
| Current branch | `git branch --show-current` |
| Recent commits | `git log --oneline -5` |
| System health | `python -m cortical.got validate` |
| Full orientation | Run the all-in-one command above |
