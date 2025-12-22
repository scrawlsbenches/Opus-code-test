# Cortical Development Manifesto

> Principles for human-AI collaborative development

---

## We Value

**Working systems over comprehensive documentation**
Ship code that works. Document what matters. Let git history tell the story.

**Learning from outcomes over planning in isolation**
Predictions earn credibility through results, not claims. Calibration matters.

**Parallel progress over sequential perfection**
Multiple threads, multiple agents, clear boundaries. Isolation enables velocity.

**Measured confidence over blind certainty**
Know when you don't know. Acknowledge uncertainty explicitly.

---

## Principles

### 1. Profile Before Optimizing
The obvious culprit is often innocent. Measure, then act.

### 2. Test Before Committing
Coverage isn't bureaucracy—it's confidence. Don't regress what you touch.

### 3. Isolate to Parallelize
Work with non-overlapping scope can run concurrently without conflict.

### 4. Credit Earned, Not Assumed
Performance determines influence. Trust is earned through results.

### 5. Cold Start Is Not Failure
New systems lack data. Acknowledge it clearly. Provide fallbacks gracefully.

### 6. Document Decisions, Not Just Code
Why matters more than what. Context that code cannot capture must be recorded.

### 7. Dog-Food Everything
Use the system to build the system. Real usage reveals what tests miss.

### 8. Honest Assessment Over Pride
Say "I don't know" when uncertain. Correct course based on evidence.

---

## For Parallel Work

```
Each agent picks a task.
Each task owns its scope.
Shared files get careful coordination.
Merge conflicts become rare.
```

**Work is tracked in GoT (Graph of Thought):**
```bash
python scripts/got_utils.py task list        # See available work
python scripts/got_utils.py task start ID    # Claim a task
python scripts/got_utils.py task complete ID # Mark done
python scripts/got_utils.py sprint status    # Current sprint focus
```

---

## The Hubris Paradox

The system teaches calibration through experience:

- Overconfident predictions lose credibility
- Underconfident predictions miss opportunities
- Well-calibrated predictions earn trust

*True expertise is knowing the boundaries of your knowledge.*

---

## Signed

```
Cortical Text Processor
Living Document — Updated as we learn
```
