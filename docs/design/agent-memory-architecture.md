# Agent Memory Architecture Design

*Draft: 2026-01-08*
*Status: PROPOSAL - pending discussion*

---

## Problem Statement

Agents lose focus across branches and sessions because:

1. **Scratchpads are branch-specific** - tied to the branch where they were created
2. **No persistent identity** - each session starts fresh
3. **No searchable history** - can't find what was done before
4. **No self-communication over time** - agents can't talk to themselves across sessions

### The Core Insight

> Each agent needs its own file for communication across branches so we can go back and understand what was being done and how. Agents need to be able to communicate with themselves over time. Changes to be able to search history is very valuable.

---

## Current State

```
docs/sessions/
└── file-access-audit-scratchpad.md   # Branch-specific, lost on switch
```

**Problems:**
- Tied to specific branch (`claude/code-review-fixes-J4A3H`)
- Left behind when switching branches
- No agent identity - just session notes
- No searchable history across sessions
- No way to backfill chat data

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENT MEMORY ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CURRENT PROBLEM:                                                       │
│  ─────────────────                                                      │
│  Session 1 (branch A) → scratchpad.md → loses focus                    │
│  Session 2 (branch B) → new scratchpad → no history                    │
│  Session 3 (branch A) → "what was I doing?"                            │
│                                                                          │
│  PROPOSED SOLUTION:                                                     │
│  ──────────────────                                                     │
│  Each agent → own file → persists across branches                       │
│  Chat history → backfilled → searchable over time                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
.claude/
├── agent-history/                      # Git-tracked, searchable
│   ├── 2026-01-07-session-abc123.md    # Summary of session
│   ├── 2026-01-08-session-def456.md    # Summary of session
│   ├── by-branch/                      # Branch-indexed views
│   │   ├── main.md                     # What happened on main
│   │   └── claude-fix-xyz.md           # What happened on feature branch
│   └── index.json                      # Searchable metadata
│
└── working/                            # .gitignore'd, per-machine
    └── current-context.md              # Active scratchpad (ephemeral)
```

### Key Design Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Session summaries | Git-tracked | Survive branch switches, searchable, shared |
| Working state | .gitignore'd | Doesn't pollute git history, ephemeral |
| Session identity | Timestamp + UUID | Unique, sortable, no conflicts |
| Branch awareness | Explicit tracking | Know what happened on each branch |
| Chat backfill | Structured summaries | Searchable key points from conversations |

---

## Session Summary Format

Each session creates a summary file:

```markdown
# Session: 2026-01-08-abc123

**Branch:** claude/fix-scratchpad-focus-SUJkx
**Started:** 2026-01-08T14:30:00Z
**Duration:** ~2 hours

## Context
What was the starting state? What was the goal?

## Key Decisions
- Decision 1: Why and what was decided
- Decision 2: Why and what was decided

## Work Completed
- [ ] Task 1 - status
- [x] Task 2 - completed

## Handoff Notes
What does the next session need to know?

## Chat Highlights
Key exchanges that provide context:
- User asked about X, resolution was Y
- Discovered issue with Z, documented in FILE

## Files Touched
- path/to/file1.py - what changed
- path/to/file2.py - what changed
```

---

## Branch-Aware Context

Each branch gets an aggregated view:

```markdown
# Branch: claude/fix-scratchpad-focus-SUJkx

## Sessions on this branch
1. 2026-01-08-abc123 - Initial work on agent memory
2. 2026-01-08-def456 - Continued implementation

## Current State
- What's done
- What's in progress
- What's blocked

## Key Decisions Made
- Links to session summaries with decisions

## Files Changed
- Aggregated from all sessions
```

---

## Chat Backfill Mechanism

### What to Capture

| Type | Example | Value |
|------|---------|-------|
| User requests | "Fix the scratchpad focus issue" | Intent |
| Key insights | "Each agent needs its own file" | Understanding |
| Decisions | "We'll use timestamp-based IDs" | Architecture |
| Blockers | "Tests are failing, need to fix first" | Context |
| Handoffs | "Save design, discuss later" | Continuity |

### How to Capture

Options to explore:
1. **Manual summary** - Agent writes summary at session end
2. **Structured prompts** - Template questions to answer
3. **Automatic extraction** - Parse conversation for key points
4. **Hybrid** - Auto-extract + manual refinement

---

## Search Capabilities

### What We Want to Search

- "What was done on branch X?"
- "When did we decide to use approach Y?"
- "What files were touched for feature Z?"
- "What was the user's feedback on W?"

### Implementation Options

1. **Simple grep** - Text search over markdown files
2. **JSON index** - Structured metadata for filtering
3. **Semantic search** - Use existing audit_indexer.py pattern
4. **GoT integration** - Store sessions as entities with edges

---

## Open Questions

1. **Granularity**: Per-session vs per-day vs per-task summaries?
2. **Automation**: How much can be auto-generated vs manual?
3. **Storage**: All git-tracked, or hybrid with local state?
4. **Identity**: How to identify "same agent" across sessions?
5. **Merge conflicts**: How to handle when branches merge?
6. **Privacy**: What should NOT be captured in searchable history?

---

## Implementation Phases

### Phase 1: Manual Session Summaries
- Create `.claude/agent-history/` structure
- Write session summary at end of each session
- Simple grep-based search

### Phase 2: Branch Indexing
- Auto-generate branch summary files
- Track which sessions touched which branches
- Cross-reference with git log

### Phase 3: Chat Backfill
- Define extraction patterns
- Build structured capture mechanism
- Enable semantic search

### Phase 4: GoT Integration
- Sessions as entities
- Edges: SESSION → BRANCH, SESSION → FILE, SESSION → DECISION
- Query: "path from session-1 to current state"

---

## Related Work

- `docs/design/sub-agent-communication-patterns.md` - How to delegate to sub-agents
- `docs/sessions/file-access-audit-scratchpad.md` - Current scratchpad approach
- `.got/` - Graph of Thought for entity tracking
- `docs/audits/experiments/learnings.md` - Pattern for capturing learnings

---

*To be discussed after resolving automated test issues.*
