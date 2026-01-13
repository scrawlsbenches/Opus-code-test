# Session Continuity Patterns

## Overview

AI agent sessions are context-limited. This document explains how to maintain
continuity across sessions using the tools available in this codebase.

## The Problem

Each new session starts with:
- No memory of previous work
- No knowledge of in-progress tasks
- No context about recent decisions

Without proper handoffs, sessions:
- Repeat solved problems
- Introduce conflicting changes
- Lose hard-won insights

## Solution 1: GoT Handoffs (Primary)

Handoffs are the **primary method** for session continuity.

### Creating a Session Handoff
```bash
# At end of session
python -m cortical.got handoff session \
    --target "next-session" \
    --summary "Fixed tokenizer bug, pending: loop detection" \
    --notes "Key insight: must preserve full identifiers" \
    --kt KT-XXX  # Link to knowledge transfer if exists
```

### Auto-Captured Context
Session handoffs automatically capture:
- Current git branch
- Modified files (from git status)
- Recent commits (last 5)
- Any blockers you specify

### Starting a New Session
```bash
# Check for pending handoffs
python -m cortical.got handoff list --status initiated

# Review handoff details
python -m cortical.got handoff show H-XXXXXXXX

# Accept and continue
python -m cortical.got handoff accept H-XXXXXXXX --agent "me"
```

## Solution 2: Knowledge Transfers (Long-term)

KTs preserve **learnings** that should persist beyond a single handoff.

### When to Create KT
- Significant bug fix with lessons learned
- Architectural decision made
- Pattern discovered that others should know
- Session produced reusable insights

### Creating a KT
```bash
python -m cortical.got kt create "Session: Tokenizer Deep Dive" \
    --summary "Found that CamelCase identifiers need both full form and split parts"

# Add detailed sections
python -m cortical.got kt append KT-XXX "What Worked" "Using split_identifier..."
python -m cortical.got kt append KT-XXX "What Didn't" "Simple regex lost acronyms..."

# Finalize when complete
python -m cortical.got kt finalize KT-XXX
```

## Solution 3: Cognitive Agent Memory

The cognitive agent learns from documents and can answer questions.

### Writing Knowledge for Future Sessions
```bash
# Create a knowledge document
cat > samples/cognitive_agent_knowledge/my_topic.md << 'EOF'
# What I Learned About [Topic]

## Problem
[What was the issue?]

## Solution
[How did you fix it?]

## Key Insight
[What should future sessions know?]
EOF

# Train the cognitive agent on it
python -m cortical.cognitive train samples/cognitive_agent_knowledge
python -m cortical.cognitive reindex
```

### Querying Past Knowledge
```bash
# Ask the cognitive agent
python -m cortical.cognitive ask "What was found about tokenization?"
python -m cortical.cognitive ask "What bugs were fixed recently?"

# Query semantic associations
python -m cortical.cognitive query "tokenizer"
```

## The Entry Gate Checklist

At the START of every session:

```
□ 1. CHECK FOR HANDOFFS
    python -m cortical.got handoff list --status initiated

□ 2. CHECK FOR DRAFT KTs
    python -m cortical.got kt list --status draft

□ 3. QUERY COGNITIVE AGENT
    python -m cortical.cognitive ask "What was previous session working on?"

□ 4. VERIFY SYSTEM STATE
    python -m cortical.got validate
    python -m pytest tests/smoke/ -v
```

## Handoff Lifecycle

```
Session Handoff:  session → initiated → accepted → completed
Task Handoff:     initiate → initiated → accepted → completed
                                      ↘ rejected
```

## Best Practices

1. **Always create handoff before ending session** - Even if work is complete
2. **Link KTs to handoffs** - Use `--kt KT-XXX` flag
3. **Be specific in summaries** - "Fixed TypeError in cmd_kt_show:436" not "Fixed bug"
4. **Include blockers** - `--blockers "item1" "item2"` surfaces issues
5. **Accept handoffs explicitly** - Shows acknowledgment in graph
6. **Complete handoffs with results** - Close the loop

## Recovery When Lost

If you're confused about context:

```bash
# Run full recovery
/context-recovery

# Or manually:
git log --oneline -10
python -m cortical.got task list --status in_progress
python -m cortical.got handoff list --status initiated
python -m cortical.cognitive ask "What issues were found recently?"
```

## Common Mistakes

| Mistake | Why It's Bad | Fix |
|---------|--------------|-----|
| Not creating handoff | Next session starts blind | Always create before ending |
| Creating KT but not handoff | KT is for learnings, not context | Use both when appropriate |
| Vague summaries | Doesn't help next session | Be specific about what/where/why |
| Not accepting handoffs | Shows as "abandoned" | Accept then complete |
| Training without reindexing | IDF weights are stale | Always reindex after training |
