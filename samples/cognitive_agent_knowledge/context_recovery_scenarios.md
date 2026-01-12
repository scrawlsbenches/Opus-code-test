# Context Recovery Scenarios

## How to Recover When You're Lost

This document describes common scenarios where context is lost and how to recover.

---

## Scenario 1: Fresh Session, No Prior Knowledge

**Situation**: You just started and know nothing about this codebase.

**Recovery Steps**:
1. Run quick orientation:
   ```bash
   git log --oneline -5
   python -m cortical.cognitive status
   python -m cortical.got task list --status in_progress
   ```

2. Ask the cognitive agent:
   ```bash
   python -m cortical.cognitive ask "What is this codebase about?"
   python -m cortical.cognitive ask "What are the main components?"
   ```

3. Read session notes:
   ```bash
   ls docs/sessions/
   cat docs/sessions/2026-01-12-cognitive-agent-roadmap.md
   ```

4. Check CLAUDE.md for project conventions.

---

## Scenario 2: Mid-Task Context Loss

**Situation**: You were working on something but lost track of what.

**Recovery Steps**:
1. Check in-progress tasks:
   ```bash
   python -m cortical.got task list --status in_progress
   ```

2. Look at recent commits for clues:
   ```bash
   git log --oneline -10
   git diff HEAD~1 --stat
   ```

3. Check for handoffs:
   ```bash
   python -m cortical.got handoff list --status initiated
   ```

4. Ask about recent work:
   ```bash
   python -m cortical.cognitive ask "What was being worked on recently?"
   ```

---

## Scenario 3: Encountering Unfamiliar Code

**Situation**: You need to understand a module you've never seen.

**Recovery Steps**:
1. Query for related concepts:
   ```bash
   python -m cortical.cognitive query "module_name"
   ```

2. Find associations:
   ```python
   agent.get_associations("transaction", top_k=15)
   ```

3. Predict common patterns:
   ```python
   agent.predict_next("transaction")  # What follows?
   ```

4. Check if there's documentation:
   ```bash
   grep -r "module_name" docs/
   ```

---

## Scenario 4: Understanding Design Decisions

**Situation**: You see code and don't understand WHY it's built that way.

**Recovery Steps**:
1. Check design documents:
   ```bash
   cat samples/cognitive_agent_knowledge/design_decisions.md
   ```

2. Ask specific questions:
   ```bash
   python -m cortical.cognitive ask "Why does IDF weighting matter?"
   python -m cortical.cognitive ask "Why are there two types of links?"
   ```

3. Check commit messages for rationale:
   ```bash
   git log --grep="design" --oneline
   git log --grep="why" --oneline
   ```

---

## Scenario 5: Something Broke and You Don't Know Why

**Situation**: Tests fail or behavior changed unexpectedly.

**Recovery Steps**:
1. Run diagnostic tests:
   ```bash
   python -m pytest tests/smoke/ -v --tb=short
   ```

2. Check model health:
   ```bash
   python -m cortical.cognitive status
   python -m cortical.got validate
   ```

3. Look for recent changes:
   ```bash
   git log --oneline -5
   git diff HEAD~3
   ```

4. Ask about the failing component:
   ```bash
   python -m cortical.cognitive ask "How does predict_next work?"
   ```

---

## Scenario 6: Performance Seems Slow

**Situation**: Operations take longer than expected.

**Recovery Steps**:
1. Check expected performance:
   - get_associations: <100ms
   - predict_next: <50ms
   - Save (no changes): <1s

2. Profile the slow operation:
   ```python
   import time
   start = time.perf_counter()
   result = agent.get_associations("test")
   print(f"Took: {(time.perf_counter() - start)*1000:.1f}ms")
   ```

3. Check if indexes exist:
   ```python
   storage = agent.graph._storage
   print(f"_outgoing entries: {len(storage._outgoing)}")
   print(f"_incoming entries: {len(storage._incoming)}")
   ```

4. Check staleness (may need reindex):
   ```bash
   python -m cortical.cognitive status
   # If staleness > 20%, run:
   python -m cortical.cognitive reindex
   ```

---

## Scenario 7: Need to Add New Functionality

**Situation**: You need to extend the cognitive agent.

**Recovery Steps**:
1. Understand current architecture:
   ```bash
   cat samples/cognitive_agent_knowledge/architecture_overview.md
   ```

2. Check existing patterns:
   ```bash
   python -m cortical.cognitive ask "How are new atom types added?"
   ```

3. Look at similar implementations:
   ```bash
   grep -r "AtomType\." cortical/cognitive/
   ```

4. Run tests before and after:
   ```bash
   python -m pytest tests/integration/test_cognitive_agent_queries.py -v
   ```

---

## Scenario 8: Preparing for Session Handoff

**Situation**: You're ending a session and want future agents to continue.

**Recovery Steps**:
1. Create a knowledge transfer:
   ```bash
   python -m cortical.got kt create "Session: topic" --summary "What was done"
   ```

2. Create session handoff:
   ```bash
   python -m cortical.got handoff session --target "next-agent" --summary "Context here"
   ```

3. Commit work with clear messages:
   ```bash
   git add -A
   git commit -m "feat: Description of what changed"
   ```

4. Update session notes if significant:
   ```bash
   # Add to docs/sessions/YYYY-MM-DD-topic.md
   ```

---

## Quick Recovery Commands

When completely lost, run these in order:

```bash
# 1. Where am I?
pwd
git branch --show-current

# 2. What's the project?
cat CLAUDE.md | head -50

# 3. What's recent?
git log --oneline -5

# 4. What's in progress?
python -m cortical.got task list --status in_progress 2>/dev/null || echo "No GoT"

# 5. What does the model know?
python -m cortical.cognitive status 2>/dev/null || echo "No model"

# 6. Any handoffs?
ls docs/sessions/ 2>/dev/null | tail -5
```

This sequence takes 30 seconds and provides enough context to start working.
