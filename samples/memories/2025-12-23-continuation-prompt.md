# Continuation Prompt

> Use this to start a new session and test the continuation system.

---

## Step 1: Generate Your Context

Run this command first:

```bash
python scripts/claudemd_generation_demo.py --generate-layer4
```

## Step 2: Review What You See

You should see output like:

```markdown
## Observable Facts

**Current Branch:** `claude/[YOUR-NEW-BRANCH]`
**Saved Branch:** `claude/verify-gitignore-LpJe4` (saved: 2025-12-23T...)
**Note:** Branches differ. Investigate before acting.
...

## Verify State

Run these commands to understand current state:
```bash
git log --oneline origin/claude/verify-gitignore-LpJe4 -5
...
```

## Questions to Resolve

1. **Branch Continuity**
   - Current: `claude/[YOUR-NEW-BRANCH]`
   - Saved: `claude/verify-gitignore-LpJe4`
   - *Is this a continuation? Should I pull from previous branch?*
```

## Step 3: Investigate

Run the git commands it suggests. See what's on the previous branch.

## Step 4: Ask Me

Based on what you find, ask me:
- Should you pull from the previous branch?
- Should you wait for a PR merge?
- Should you proceed independently?

## What This Tests

The refactored script should:
- ✅ Show facts (branches differ)
- ✅ Provide investigation commands
- ✅ Ask questions, not mandate actions
- ❌ NOT say "you MUST merge"
- ❌ NOT make determinations like "needs_merge"

## Context Files

- **Knowledge Transfer:** `samples/memories/2025-12-23-knowledge-transfer-continuation-refactor.md`
- **Script:** `scripts/claudemd_generation_demo.py`
- **Previous Branch:** `claude/verify-gitignore-LpJe4`

---

**Philosophy:** Facts from GoT. You investigate. You decide.
