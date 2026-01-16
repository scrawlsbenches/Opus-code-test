# Thought Branches: Modeling Conversational Continuity

*Created: 2026-01-16*

---

## The Problem

Current intent model is linear:
```
Intent A → complete → Intent B → complete → ...
```

But real conversations branch:
```
Intent A (in progress)
    └── User redirects: "Actually, first do Y"
        └── Intent B (branch)
            └── Complete B
                └── Return to A
```

When context compacts (daydreaming), this tree structure is lost. Claude sees recent messages but loses the *shape* of the conversation - what branched from what, what's paused, what to return to.

---

## The Human Model

When humans converse, they hold a mental stack:
- "We were improving Claude.md"
- "Then you asked about git history"
- "Then you asked about memory frequency"
- "Now we're designing thought branches"

Humans naturally track this. When a branch completes, they say "so, back to the memory question..." or "actually, let's stay on branches."

This is natural working memory - hold context, branch, return.

---

## Intent Stack Concept

Instead of flat task tracking:
> "Pending: Add dark mode, Fix build"

Model as a tree with depth:
> "We paused 'dark mode' to fix the build. Build's done. Back to dark mode?"

```
INTENT STACK:
├─ [0] "Add dark mode"      ← PAUSED (waiting for branch)
└─ [1] "Fix build first"    ← ACTIVE (branch)
```

When branch completes, pop back to paused intent.

---

## Data Model

### Paused Intent
```python
{
    "type": "intention",
    "id": "intent_abc",
    "goal": "Add dark mode",
    "status": "paused",           # New status
    "paused_for": "intent_xyz"    # What branched from it
}
```

### Active Branch
```python
{
    "type": "intention",
    "id": "intent_xyz",
    "goal": "Fix build first",
    "status": "pending",
    "parent_intent": "intent_abc"  # What to return to
}
```

### Nested Branches
Branches can nest arbitrarily:
```
CONVERSATION TREE:
└── "Improve Claude.md" (paused)
    ├── "Show git history" ✓
    ├── "Revert to 9858ea77" ✓
    ├── "List public methods" ✓
    └── "Design memory workflow" (paused)
        └── "Thought branches concept" ← HERE
```

---

## Proposed Interface

```python
# User redirects mid-conversation
memory.branch("Fix the build first")
# → Pauses current intent, creates new with parent reference

# Branch completes
memory.complete_branch("Build fixed")
# → Marks branch complete
# → Prompts: "Return to 'Add dark mode'?"

# Abandon branch without completing
memory.abandon_branch("User changed mind")
# → Marks branch abandoned
# → Prompts: "Return to 'Add dark mode'?"

# View current state
memory.intent_tree()
# → Shows nested structure with current position

# View stack depth
memory.intent_depth()
# → Returns integer: how deep in branches

# Recovery shows tree, not flat list
memory.recover()
# → "You are at branch depth 2: ..."
# → Shows full tree with position marker
```

---

## Updated Per-Prompt Workflow

```
USER PROMPT ARRIVES
        │
        ▼
┌─────────────────────────┐
│ New / Continue /        │
│ REDIRECT?               │
└─────────────────────────┘
    │       │        │
   New   Continue  Redirect
    │       │        │
    ▼       │        ▼
anchor_     │    branch()
intent()    │    - Pause current
    │       │    - Create child intent
    └───────┴────────┘
            │
        ... work ...
            │
            ▼
┌─────────────────────────┐
│ Task complete.          │
│ Is this a branch?       │───Yes───► complete_branch()
└─────────────────────────┘           │
            │ No                      ▼
            │              ┌─────────────────────┐
            │              │ Return to parent?   │
            │              │ (Ask user)          │
            │              └─────────────────────┘
            ▼
       Respond to user
```

---

## Design Decisions

1. **Branches can nest** - User can redirect within a redirect
2. **Abandonment prompts return** - If user abandons branch AND original, Claude asks
3. **complete_branch() auto-prompts** - "Return to [paused intent]?"
4. **Recovery shows tree** - Not flat list of pending intents

---

## Behavior Change

**Before (flat):**
```
Pending intentions:
- Add dark mode
- Fix build
- Update tests
```

**After (tree):**
```
Intent tree:
└── Add dark mode (PAUSED at step 2/5)
    └── Fix build (ACTIVE)

You are at depth 1. When "Fix build" completes,
return to "Add dark mode" step 2?
```

---

## Why This Matters

This models how humans naturally hold conversational context - as branches that can be returned to, not a queue to process linearly.

For Claude, this means:
- **During work**: Natural flow with redirects
- **After daydreaming**: Recovery shows where we are in the conversation tree
- **For the user**: Feels like working with someone who remembers "where we were"

The goal: Conversations that feel continuous even across context loss.

---

## Open Questions

1. How deep should branches be allowed to nest? (Practical limit?)
2. Should old completed branches be pruned from the tree?
3. How does this interact with `handoff()` across sessions?
4. Visual representation in `recover()` output - ASCII tree vs prose?

---

## Next Steps

1. Add `status: "paused"` to intention model
2. Add `parent_intent` field for branch tracking
3. Implement `branch()`, `complete_branch()`, `abandon_branch()`
4. Implement `intent_tree()` and `intent_depth()`
5. Update `recover()` to show tree structure
6. Update CLAUDE.md with branching workflow
