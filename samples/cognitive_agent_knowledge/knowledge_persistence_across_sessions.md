# Knowledge Persistence Across Sessions

## The Core Challenge

AI agents lose all context between sessions. Every new session starts with a blank slate - no memory of previous work, decisions made, bugs fixed, or patterns discovered. This creates a fundamental problem: **valuable knowledge dies when sessions end**.

Without deliberate knowledge persistence:
- The same bugs get reintroduced
- The same architectural mistakes get made
- The same questions get re-investigated
- The same "gotchas" catch agents repeatedly

This document teaches how to preserve knowledge so future sessions inherit wisdom rather than starting from scratch.

---

## Types of Knowledge to Persist

Not all knowledge is worth preserving. Focus on these categories:

### 1. Decisions and Rationale

**What**: Architectural choices, design trade-offs, why X was chosen over Y.

**Why persist**: Without rationale, future agents may undo good decisions or repeat bad experiments.

**Example from this repo** (from `samples/cognitive_agent_knowledge/design_decisions.md`):
```markdown
## IDF Weighting: Why It Matters

**Problem**: Common words like "the", "and", "is" appear everywhere but carry little meaning.

**Solution**: IDF (Inverse Document Frequency) weighting.

**Why smoothed formula?** Adding +1 prevents division by zero and log(0).
```

### 2. Bugs and Fixes

**What**: What broke, why it broke, how it was fixed, how to prevent recurrence.

**Why persist**: Bugs have a tendency to return. Documenting them prevents regression.

**Example from CLAUDE.md** (Critical Bugs table):
```markdown
| Bug | Root Cause | Fix | File Reference |
|-----|------------|-----|----------------|
| WAL commit order | Wrote entities before WAL fsync | WAL-first: commit -> fsync -> writes | cdg/transaction_manager.py:293 |
| Bigram separators | Underscore instead of space | SPACE only: "neural networks" | tokenizer.py:319-332 |
```

### 3. Patterns and Anti-Patterns

**What**: What works well, what doesn't, common mistakes to avoid.

**Why persist**: Patterns encode hard-won experience. Anti-patterns prevent repeated failures.

**Example**: The Container pattern in this codebase:
```markdown
DO: Receive dependencies through constructor injection
DON'T: Hardcode dependencies in constructors
DON'T: Use Path(".got") or other magic paths
```

### 4. Gotchas and Surprises

**What**: Non-obvious behaviors, edge cases, things that "shouldn't" happen but do.

**Why persist**: Gotchas waste hours when rediscovered. One sentence can save a session.

**Example from `training_process_findings_and_enhancements.md`**:
```markdown
The model was trained on ~669 documents, almost exclusively from samples/.
Key directories had 0% coverage: cortical/, tests/, docs/.

Impact: When asked "What is TextToAtomsBridge?", the model couldn't answer
because it had never seen cortical/cognitive/text_bridge.py.
```

### 5. Investigation Findings

**What**: What was explored, what was learned, what questions were answered.

**Why persist**: Deep investigations are expensive. Preserve the insights.

---

## Where to Persist Knowledge

Different types of knowledge belong in different places:

### CLAUDE.md - Project-Wide Truths

**Use for**:
- Core workflows and protocols
- Critical bugs that must never recur
- Architectural principles
- Entry gate checklists

**Characteristics**:
- Read at session start
- Authoritative (overrides other sources)
- Stable (changes rarely)
- Concise (agents have limited context)

**Example sections in this repo's CLAUDE.md**:
- "Critical Bugs (Don't Reintroduce)" - table of fixed bugs
- "Container: First-Class Citizen" - DI requirements
- "Safeguards: Protecting Against Bad Requests" - request classification

### samples/cognitive_agent_knowledge/ - Long-Term Memory

**Use for**:
- Design decisions and rationale
- Concept explanations
- Recovery scenarios
- Work history and context

**Characteristics**:
- Queryable via CognitiveAgent
- Detailed (can be thorough)
- Discoverable (train the model on it)
- Self-teaching (written for AI agents)

**Existing documents in this repo**:
| Document | Purpose |
|----------|---------|
| `what_is_cognitive_agent.md` | Identity and purpose |
| `design_decisions.md` | Rationale for choices |
| `context_recovery_scenarios.md` | How to recover when lost |
| `training_process_findings_and_enhancements.md` | Lessons from debugging |

### docs/sessions/ - Session Context

**Use for**:
- Session-specific scratchpads
- Handoff documents
- Investigation notes
- Temporary context

**Characteristics**:
- Date-stamped
- May become stale
- Detailed for specific work
- Bridge between sessions

**Pattern**: `docs/sessions/YYYY-MM-DD-topic.md`

### Code Comments - Inline Warnings

**Use for**:
- Non-obvious behavior that might confuse
- "Don't remove this" explanations
- Performance-critical sections
- Bug-fix context

**Example**:
```python
# IMPORTANT: WAL must be fsync'd BEFORE writing entities.
# See Critical Bugs table in CLAUDE.md - "WAL commit order" bug.
# Reversing this order causes data corruption on crash.
await self._wal.fsync()
await self._write_entities(entities)
```

### GoT (Graph of Thought) - Structured Data

**Use for**:
- Task tracking (what's in progress, what's done)
- Handoff state (initiated, accepted, completed)
- Knowledge transfers (linked to tasks)
- Decision records

**Commands**:
```bash
# Create knowledge transfer
python -m cortical.got kt create "Session: topic" --summary "Key learnings"

# Create handoff for next session
python -m cortical.got handoff session --target "next-agent" --summary "Context"

# Log a decision
python -m cortical.got decision log "Chose X over Y" --rationale "Because..."
```

---

## How to Write for Future Sessions

Writing for AI agents requires a different style than writing for humans.

### Be Searchable

Future agents will search for keywords. Use terms they'll likely search for.

**Bad**: "The thing that converts stuff broke"
**Good**: "TextToAtomsBridge failed during tokenization due to empty vocabulary"

### Be Specific

Vague descriptions waste investigation time.

**Bad**: "Fixed the bug in storage"
**Good**: "Fixed race condition in VersionedStore.save() by adding threading.Lock before fcntl.flock (process-level lock was insufficient for multi-threaded access)"

### Be Actionable

Tell future agents what to DO, not just what happened.

**Bad**: "IDF weights were wrong"
**Good**: "If IDF weights seem wrong, run `python -m cortical.cognitive reindex` to refresh them. Check staleness with `status` command - above 20% triggers reindex need."

### Include Commands

Future agents need copy-pasteable commands.

**Good pattern**:
```markdown
## How to Verify This Still Works

```bash
# 1. Check model state
python -m cortical.cognitive status

# 2. Run relevant tests
python -m pytest tests/integration/test_cognitive_agent_queries.py -v

# 3. Verify specific behavior
python -m cortical.cognitive ask "What is TextToAtomsBridge?"
```
```

### Use Tables for Reference Data

Tables are scannable and information-dense.

**Example**:
```markdown
| Component | Location | Purpose |
|-----------|----------|---------|
| CognitiveGraph | cortical/cognitive/graph.py | Hypergraph storage |
| BPETokenizer | cortical/cognitive/text_bridge.py | Vocabulary and IDF |
| IncrementalTrainer | cortical/cognitive/training.py | Document training |
```

### State the "Why"

Without "why", knowledge is brittle. Future agents may undo good decisions.

**Bad**: "Use spaces for bigram separators"
**Good**: "Use spaces for bigram separators because underscores prevent phrase matching. 'neural_networks' won't match searches for 'neural networks'. See tokenizer.py:319-332."

---

## Knowledge Transfer Protocols

### Session Handoffs (GoT)

Use when ending a session with incomplete work:

```bash
# 1. Create knowledge transfer for learnings
python -m cortical.got kt create "Session: [topic]" \
    --summary "Key outcomes and learnings"

# 2. Create session handoff with context
python -m cortical.got handoff session \
    --target "next-agent" \
    --summary "Fixed X, pending Y, blocked on Z" \
    --kt KT-XXX \
    --blockers "item1" "item2"

# 3. Commit everything
git add -A && git commit -m "chore: Session checkpoint"
```

### Accepting Handoffs

When starting a session, check for handoffs:

```bash
# Check for pending handoffs
python -m cortical.got handoff list --status initiated

# Review handoff details
python -m cortical.got handoff show H-XXX

# Accept and continue
python -m cortical.got handoff accept H-XXX --agent "me"
```

### Markdown Handoffs (Legacy)

For detailed context that doesn't fit structured fields:

```markdown
# Agent Handoff Document - YYYY-MM-DD

**From:** [Previous agent]
**Branch:** `branch-name`
**Status:** [Ready for continuation]

## Quick Validation Checklist
[Commands to verify state]

## What Was Done
[Summary of completed work]

## Known Issues
[Pre-existing problems]

## Next Steps
[Prioritized options for continuation]
```

---

## Making Knowledge Queryable

Knowledge is only useful if it can be found. The CognitiveAgent provides semantic search.

### Training on Knowledge Documents

```bash
# Train on cognitive agent knowledge
python -m cortical.cognitive train samples/cognitive_agent_knowledge --pattern "*.md"

# Train on source code (for self-awareness)
python -m cortical.cognitive train cortical/ --pattern "*.py"

# Verify training
python -m cortical.cognitive status
```

### Querying Knowledge

```bash
# Ask questions
python -m cortical.cognitive ask "What is TextToAtomsBridge?"
python -m cortical.cognitive ask "Why does IDF weighting matter?"
python -m cortical.cognitive ask "What bugs were fixed recently?"

# Find related concepts
python -m cortical.cognitive query "transaction"
python -m cortical.cognitive query "persistence"
```

### Writing Queryable Documents

Structure documents so key terms appear naturally:

```markdown
# What is TextToAtomsBridge?

TextToAtomsBridge is a component that converts text into atoms for the
CognitiveGraph. It uses BPETokenizer for tokenization and tracks IDF
weights for semantic relevance.

Key methods:
- process_text() - Main entry point
- _tokenize() - Splits text into tokens
- _create_atoms() - Converts tokens to graph atoms
```

This document will now respond to queries for: "TextToAtomsBridge", "convert text atoms", "tokenization", "IDF weights".

---

## Anti-Patterns to Avoid

### 1. Knowledge Silos

**Problem**: Knowledge exists but is hard to find.

**Symptoms**:
- Important context buried in old commits
- Decisions recorded only in chat logs
- Patterns known to one agent but not documented

**Solution**: Centralize in queryable locations (samples/, CLAUDE.md)

### 2. Tribal Knowledge

**Problem**: "Everyone knows that" - but new sessions don't.

**Symptoms**:
- Assumptions not documented
- Implicit conventions
- "Just ask X" mentality (but X is a previous session)

**Solution**: Document assumptions explicitly. If it's not written down, it doesn't exist for future sessions.

### 3. Outdated Documentation

**Problem**: Docs say one thing, code does another.

**Symptoms**:
- Commands that don't work
- File paths that don't exist
- Patterns that have changed

**Solution**: Update docs when changing code. Delete stale docs. Date-stamp volatile information.

### 4. Write-Only Documentation

**Problem**: Documents created but never trained on.

**Symptoms**:
- CognitiveAgent can't answer questions about documented topics
- Knowledge exists but isn't queryable

**Solution**: Always train after adding documents:
```bash
# After creating knowledge documents
python -m cortical.cognitive train samples/cognitive_agent_knowledge/ --pattern "*.md"
```

### 5. Context-Free Bug Fixes

**Problem**: Bug is fixed but not documented.

**Symptoms**:
- Same bug returns later
- Future agent removes "unnecessary" code that was the fix
- No regression test exists

**Solution**: For significant bugs:
1. Add to Critical Bugs table in CLAUDE.md
2. Add regression test
3. Add code comment explaining the fix

### 6. Implicit Decisions

**Problem**: Architecture evolved but rationale is lost.

**Symptoms**:
- "Why is it built this way?" has no answer
- Future agent "improves" by undoing good decisions
- Design discussions lost in ephemeral chat

**Solution**: Log decisions explicitly:
```bash
python -m cortical.got decision log "Chose X over Y" \
    --rationale "X has O(1) lookup, Y is O(n). Performance matters here."
```

---

## Practical Checklist: Before Ending a Session

```
[ ] Is there incomplete work? -> Create GoT handoff
[ ] Did I learn something non-obvious? -> Write to samples/cognitive_agent_knowledge/
[ ] Did I fix a significant bug? -> Add to Critical Bugs table in CLAUDE.md
[ ] Did I make an architectural decision? -> Log in GoT or design_decisions.md
[ ] Are there gotchas future agents should know? -> Document them
[ ] Did I update docs? -> Train CognitiveAgent on changes
```

---

## Summary

Knowledge persistence is a discipline, not an afterthought. Every session should leave the codebase smarter than it found it.

**The core principles**:
1. **Write it down** - If it's not documented, it doesn't exist
2. **Make it findable** - Train the CognitiveAgent, use searchable terms
3. **Make it actionable** - Include commands, file paths, specific steps
4. **Explain the why** - Rationale protects decisions from reversal
5. **Keep it current** - Delete stale docs, update when code changes

Future sessions inherit two things: code and knowledge. Make sure both are valuable.

---

*This document is training data for the CognitiveAgent. Future sessions can ask "How do I preserve knowledge?" or "What are knowledge anti-patterns?" to recover this context.*
