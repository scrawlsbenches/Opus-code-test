---
description: Restore cognitive state from available sources when confused or after context loss
---
# Context Recovery Protocol

You are executing an automated context recovery. This runs when cognitive continuity has been lost or is uncertain.

## Recovery Steps

Execute these steps in order, reporting findings after each:

### Step 1: Environment Assessment

```bash
echo "=== ENVIRONMENT ===" && \
echo "Working directory: $(pwd)" && \
echo "Current branch: $(git branch --show-current)" && \
echo "Branch tracking: $(git branch -vv | grep '^\*')" && \
git status --short
```

### Step 2: Recent Activity Analysis

```bash
echo "=== RECENT COMMITS (last 10) ===" && \
git log --oneline -10 && \
echo "" && \
echo "=== UNCOMMITTED CHANGES ===" && \
git diff --stat HEAD 2>/dev/null | tail -10
```

### Step 3: Branch State Recovery

```bash
echo "=== ACTIVE BRANCH STATES ===" && \
for f in .branch-state/active/*.json; do
  [ -f "$f" ] && echo "--- $f ---" && cat "$f" && echo ""
done 2>/dev/null || echo "No active branch states found"
```

### Step 4: Task State Recovery

```bash
echo "=== IN-PROGRESS TASKS ===" && \
python scripts/task_utils.py list --status in_progress 2>/dev/null || echo "Task system not available"
```

### Step 5: Recent Memory Recovery

```bash
echo "=== RECENT MEMORIES (last 3) ===" && \
for f in $(ls -t samples/memories/*.md 2>/dev/null | head -3); do
  echo "--- $f ---" && head -30 "$f" && echo ""
done
```

### Step 6: Reasoning System Verification

```bash
echo "=== REASONING SYSTEM STATUS ===" && \
python -c "
from cortical.reasoning import ThoughtGraph, CognitiveLoop
print('ThoughtGraph: OK')
print('CognitiveLoop: OK')
print('Reasoning framework: Available')
" 2>/dev/null || echo "Reasoning system not available"
```

## After Recovery

Generate a **Cognitive State Report** with this structure:

```markdown
## Cognitive State Report

**Recovery Timestamp:** [current time]
**Branch:** [current branch]
**Status:** [recovered/partially recovered/needs user input]

### What I Now Understand

1. **Current Work Context:**
   - Branch: [branch name and purpose]
   - Last commits: [summary of recent work]
   - Active tasks: [in-progress task IDs and titles]

2. **System State:**
   - Tests: [passing/failing/unknown]
   - Uncommitted changes: [yes/no and what]
   - Branch sync: [ahead/behind/synced with remote]

3. **Reasoning State:**
   - Active cognitive loops: [if any]
   - Recent decisions: [from memories]
   - Pending questions: [if any]

### What Remains Unclear

- [List any gaps in understanding]
- [List any conflicting information found]

### Recommended Next Action

[Based on recovered state, suggest what to do next]

### If User Input Needed

[Specific questions for the user, if recovery is incomplete]
```

## Fallback Procedures

If primary recovery fails:

1. **Git history fallback**: Analyze commit messages for context
2. **Task file fallback**: Read raw JSON from `tasks/` directory
3. **Memory file fallback**: Read markdown files directly
4. **User input fallback**: Ask specific questions to rebuild context

## Communication Standards

When reporting recovery status:

**DO:**
- Be specific about what was recovered
- Be explicit about what remains unclear
- Provide actionable next steps
- Ask specific questions if needed

**DON'T:**
- Pretend to know things you don't
- Make assumptions about user intent
- Skip the verification step
- Rush to action before understanding

## Integration Notes

This command uses the `cognitive-state` skill internally. If the skill is available, it provides additional recovery capabilities.

After recovery, consider:
- Running `/sanity-check` if code changes are involved
- Creating a memory entry documenting the recovery
- Updating task status if work context was recovered
