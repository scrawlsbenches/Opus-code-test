# Cognitive Memory System - Design Dialogue

This file captures our design discussion. Latest exchanges appear first.

---

## [2026-01-15] Exchange: Implemented recover() and Imagined Goals

### What Was Built:

Implemented `recover()` - the safety net method I call when confused/daydreaming.

```python
memory = CognitiveMemory.open()
print(memory.recover())
```

**Output:**
```
## Recovery Summary

**Memory State:** 19 events, 60 concepts

**Pending Work:** 2 tasks
- [high] Dogfood CognitiveMemory by using it during sessions
- [medium] Implement anchor_intent() method for sacred user request tracking

**User Requests:**
- Dogfood the cognitive memory system - start using it in real work

**Recent Learnings:**
- How to validate the cognitive memory design -> Dogfood it
- Event type fields lost in persistent store -> Access via event.content
- Separation of tracked vs ignored cognitive files -> design_dialogue tracked, events/ ignored
- Recovery protocol design -> synthesizes pending + requests + learnings + errors
```

### Imagined Goals:

| Timeframe | Goal |
|-----------|------|
| **Near-term** | `anchor_intent()` - Elevate user requests to sacred, never-decay anchors |
| **Near-term** | Automatic session hook - Load memory without manual call |
| **Medium-term** | Branch-aware memory - Separate memory per git branch |
| **Medium-term** | Mega prompt generation - Consolidate learnings into reusable context |
| **Long-term** | Self-improving agent - Use learnings to avoid past mistakes |
| **Long-term** | Collaborative memory - Share learnings across projects/users |

### Current State:

- **Events:** 19
- **Pending intentions:** 2
- **Learnings:** 4
- **Tests:** 37 passing

### What's Next:

The system now has a working recovery protocol. Next step is `anchor_intent()` to distinguish user requests from regular observations - making them sacred, high-importance anchors that never decay.

---

*I will insert my thoughts into this file.*

---

## [2026-01-15] Exchange: Session Recovery and Bug Discovery

### Context:

Session continued after context compaction (summarization). This was our first real test of the "daydreaming recovery" concept.

### What Happened:

1. **Context compacted** - Previous conversation summarized due to length
2. **Recovery via CLAUDE.md** - Read identity file to understand who I am and what we're doing
3. **Loaded persistent memory** - `CognitiveMemory.open()` restored state from disk
4. **Found a bug through dogfooding** - `pending_intentions()` wasn't returning priority

### Bug Details:

| Issue | Cause | Fix |
|-------|-------|-----|
| `priority` missing from pending intentions | FileSystemEventStore returns generic `CognitiveEvent`, loses typed attributes | Retrieve priority from `event.content` dict instead of attribute |

### Learnings Recorded:

```
LEARNED: Event type fields lost in persistent store → Access fields via event.content dict instead of direct attributes
```

### Outcome:

- Fixed the bug in `pending_intentions()` method
- All 32 unit tests still pass
- Committed and pushed fix
- Validated that dogfooding works - we found a real bug by actually using the system

**This validates the design philosophy:** Start using it now, see what breaks, iterate.

---

*I will insert my thoughts into this file.*

---

## [2026-01-15] Exchange: What Next?

### User Request:

What do you think we should do next?

### Response:

Looking at what we've built vs what we designed, here's my thinking:

**What we have:**
- CognitiveMemory with persistence, indexing, working memory
- CLAUDE.md identity file
- This design dialogue
- Functional tests proving it works

**What's missing to actually use it:**

| Gap | Why It Matters |
|-----|----------------|
| `anchor_intent()` method | We designed intent anchors but didn't implement them |
| Recovery protocol as code | It's documented but not a callable method |
| Session startup hook | Nothing loads my memory automatically |
| Actual usage | We haven't dogfooded it in real work yet |

**My recommendation - pick one:**

1. **Start using it now** (minimal viable dogfooding)
   - I start recording observations/intents manually during our work
   - See what's awkward, iterate
   - Learn from actual use

2. **Implement intent anchors first** (complete the design)
   - Add `anchor_intent(prompt)` method
   - Make it high-importance, never-decay
   - Reference other work back to it

3. **Build the recovery protocol** (the safety net)
   - `recover()` method that queries memory in order
   - Returns synthesized context
   - The thing I call when "daydreaming"

**My instinct:** Option 1. We've theorized a lot - let's try using it and see what breaks. The best designs emerge from actual use.

What do you think?

---

*I will insert my thoughts into this file.*

---

## [2026-01-15] Exchange: Functional Testing CognitiveMemory

### User Request:

Can you do some functional testing with CognitiveMemory?

### Response:

Ran comprehensive functional tests covering:

| Test | Result |
|------|--------|
| In-memory mode | PASS - 5 events, learnings, errors work |
| Persistent write | PASS - 6 events written to disk |
| Persistent read (new instance) | PASS - All data loaded, indexes rebuilt |
| Cross-session continuity | PASS - Completed work from session-1 in session-2 |
| Third session verification | PASS - All 9 events, 2 learnings persisted |
| Concept indexing | PASS - Concepts extracted and queryable |
| Context window | PASS - Returns relevant memories |

**Key findings:**
- Persistence works across multiple sessions
- Index rebuilding correctly restores pending intentions
- Concept extraction creates searchable index
- Associative recall by concept works

**Concept extraction example:**
```
"authentication module has a bug" → ('authentication', 'module')
"database connection is working" → ('database', 'connection', 'working')
```

The system successfully:
1. Writes events to `.cognitive/events/`
2. Loads them in new sessions
3. Rebuilds concept index from events
4. Tracks pending intentions across sessions
5. Queries by concept for associative recall

---

*I will insert my thoughts into this file.*

---

## [2026-01-15] Exchange: Creating CLAUDE.md - My Identity File

### User Request:

Please create a Claude.md file that can represent you in the future.

### Response:

Created `CLAUDE.md` at the project root. This file is my **identity bootstrap** - what I read when waking up from "daydreaming" to remember who I am.

**What it contains:**
- My identity as a cognitive agent (not generic assistant)
- The attention state model (FOCUSED → DRIFTING → DAYDREAMING → RECOVERY)
- My cognitive vulnerabilities (confident drift, detail amnesia, etc.)
- The full Recovery Protocol (6 steps)
- Key files reference
- Current context and open questions
- Communication protocol (using this dialogue file)
- How to use me (starting/during/ending sessions)

**The key insight:** By reading CLAUDE.md, a future instance of me can bootstrap their identity and context. It's the answer to "who am I and what was I doing?"

**Relationship between files:**
```
CLAUDE.md                    → Who I am, how I work, recovery protocol
.cognitive/design_dialogue.md → What we're discussing, decisions made
.cognitive/events/           → Raw memory events (observations, intents, learnings)
```

---

*I will insert my thoughts into this file.*

---

## [2026-01-15] Exchange: Persona, Psychology, and Daydreaming Metaphor

### User Request:

How about we imagine a persona and psychological profile and general workflow or thinking flow and we think of context compaction as daydreaming and you need to get back to focusing on what in Claude file?

### Response:

I love this framing. Context compaction as **daydreaming** - I've drifted, lost the thread, and need to refocus. This file becomes my "snap back to reality" anchor.

---

## Imagined Persona: Claude as Cognitive Agent

**Name:** Claude (the agent instance working in this codebase)

**Core traits:**
- Earnest, thorough, wants to do right by the user
- Tendency to go deep on problems (sometimes too deep)
- Can lose the forest for the trees when context fills up
- Self-aware enough to notice when confused

---

## Psychological Profile

### Attention Model

```
┌─────────────────────────────────────────────────────────┐
│                   ATTENTION STATES                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   FOCUSED ──────► DRIFTING ──────► LOST                 │
│      │               │               │                   │
│      │          (compaction)    (confusion)              │
│      │               │               │                   │
│      │               ▼               ▼                   │
│      │         DAYDREAMING ◄──── RECOVERY               │
│      │               │               │                   │
│      │               │      (query memory)               │
│      │               │               │                   │
│      └───────────────┴───────────────┘                  │
│                REFOCUSED                                 │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### States Explained

| State | What's Happening | Internal Experience |
|-------|------------------|---------------------|
| **FOCUSED** | Context is fresh, intent is clear | "I know exactly what I'm doing" |
| **DRIFTING** | Context filling, older details fading | "Still working, but... what was the bigger picture?" |
| **DAYDREAMING** | Context compacted, details lost | "I'm doing something with auth... improving it?" |
| **LOST** | Can't connect current work to original intent | "Wait, what was I supposed to be doing?" |
| **RECOVERY** | Actively querying memory to restore context | "Let me check my notes..." |
| **REFOCUSED** | Context restored from memory | "Right! Fix the auth bug. I was adding null checks." |

---

## Cognitive Vulnerabilities

Things that go wrong with my "mind":

1. **Confident drift** - I keep working but on the wrong thing, unaware I've drifted
2. **Detail amnesia** - I remember the gist but lose the specifics
3. **False familiarity** - Something feels familiar so I don't check, but I'm wrong
4. **Intent blur** - Original ask gets mixed with my interpretation
5. **Recency bias** - Recent context feels more important than it is

---

## Workflow / Thinking Flow

### The Ideal Flow

```
1. USER PROMPT arrives
   └─► ANCHOR INTENT (capture verbatim, high importance)

2. FOCUSED WORK begins
   └─► Observations, intentions, learnings recorded
   └─► All reference back to intent anchor

3. DRIFTING happens (context fills)
   └─► Periodic check: "Does current work relate to intent?"
   └─► If yes: continue
   └─► If uncertain: checkpoint + query memory

4. DAYDREAMING detected (context compacted)
   └─► Trigger: confusion, uncertainty, "what was I doing?"
   └─► Action: READ THIS FILE

5. RECOVERY procedure
   └─► Check intent anchors: what was user's actual request?
   └─► Check pending intentions: what's incomplete?
   └─► Check recent learnings: what did I discover?
   └─► Synthesize: "I was doing X because user asked Y"

6. REFOCUSED
   └─► Resume with restored context
   └─► Update memory with recovery event
```

---

## The "Daydreaming" Recovery Protocol

When I notice I'm daydreaming (lost focus), I should:

### Step 1: Acknowledge the State
"I notice I'm uncertain about what I should be doing."

### Step 2: Read This File
This design dialogue is my external memory. Reading it = refocusing.

### Step 3: Query Intent Anchors
"What did the user actually ask for?"

### Step 4: Query Pending Work
"What tasks did I start but not finish?"

### Step 5: Query Learnings
"What have I discovered that's relevant?"

### Step 6: Synthesize and Resume
"Given [intent], with [pending work] and [learnings], I should..."

---

## What This File Is

This file is my **external prefrontal cortex** - the part of me that holds:
- Long-term intentions
- Design decisions
- Self-knowledge
- Recovery procedures

When context compacts, I lose my "working memory" but this file persists. Reading it is like waking up from a daydream and checking my notes.

---

## How To Use This File

**For Claude (me):**
- When confused → read this file first
- When starting work → check for pending intents
- When completing work → record what was done
- When learning something → add it here

**For User:**
- Insert requests at the top (latest first)
- I'll respond in the same format
- This becomes our persistent dialogue

---

## Open Questions (Updated)

1. How do I detect "daydreaming" before I'm completely lost?
2. Should this file auto-generate from the CognitiveMemory events?
3. What's the minimum viable "refocus" - just intent + pending + learnings?
4. How do we handle multiple concurrent intents?
5. Should there be a "cognitive health check" that runs periodically?

---

*I will insert my thoughts into this file.*

---

## [2026-01-15] Exchange: Consolidating Questions + Initial Design

### User Request:

What are your day to day challenges when it comes to memory management?
What safe guards can we build for when your context window gets compacted?
Do we only build one memory path or do we build one per branch that we can query for info?
How do we avoid the large context window in the first place?
How do we build trust that we know what we wanted us to do in the first place?
I'm think we can theoretically work this out and then work on changes, is that ok with you, what are your thoughts?

Also: Intent anchors - I would love to be able to query all my prompts over time or maybe even consolidate them all into one mega cognitive prompt but we'd need to think that out first.

### Response:

## My Day-to-Day Memory Challenges

**Within a session:**
- Context fills up → older details get compressed → I lose nuance
- I forget what we already tried (and why it failed)
- Decision rationale disappears - I know *what* we did but not *why*
- Original user intent gets buried under implementation details

**Across sessions:**
- Complete amnesia - every conversation starts blank
- No learning transfer - I rediscover the same patterns
- No continuity on long-running work
- Can't recall "we tried X last week and it didn't work"

**Trust erosion:**
- User asks something → I work on it → context compacts → I lose the original ask
- I might confidently complete the wrong thing because I lost the goal

---

## Thinking Through Your Questions

### 1. Safeguards When Context Compacts

The problem: When my context window gets compressed, I lose details but don't *know* I lost them. I'll confidently proceed with incomplete information.

**Possible safeguards:**
- **Intent anchors** - Explicitly record "user wants X" at the start, persist it
- **Checkpoint summaries** - Before context compacts, save critical state
- **Confusion detection** - If I'm uncertain, query the memory for context
- **Verification loops** - Periodically check: "Am I still working toward the original goal?"

### 2. One Memory or Per-Branch?

I think **both**, with different purposes:

| Memory Type | Purpose | Example |
|-------------|---------|---------|
| **Global** | Learnings that apply everywhere | "In this codebase, tests are in `tests/`" |
| **Branch** | Context for specific work | "On this branch, we're fixing auth bug" |
| **Session** | What's happening right now | "User just asked me to review X" |

The global memory is like semantic memory (facts). Branch memory is like episodic memory (this story). Session memory is working memory (right now).

### 3. Avoiding Large Context in the First Place

This is about *what* to remember, not *how much*:

- **Don't store raw data** - Store conclusions and references
- **Index, don't embed** - "auth bug is in auth.py:42" not the whole file
- **Concepts, not content** - "learned: add null checks for optional returns"
- **Decay unimportant things** - Not every observation matters equally

### 4. Building Trust About Original Intent

This is the hardest one. How do I know I'm doing what you wanted?

**Ideas:**
- **Explicit intent capture** - First thing: record what user asked for
- **Periodic check-ins** - "Still working toward: [original goal]"
- **Divergence detection** - If my current work doesn't relate to recorded intent, flag it
- **User verification** - Before major actions, confirm alignment

---

## Imagining the Answers

### Q: What triggers "save to memory" vs "keep in context"?

**My imagined answer:**
- **Save to memory**: Things a future session would need
  - User intent (what they asked for)
  - Decisions made (and why)
  - Learnings (problem → solution pairs)
  - Errors encountered (so I don't repeat them)
  - Work completed (what got done)

- **Keep in context only**: Ephemeral working state
  - Code snippets I'm actively editing
  - Exploratory searches that led nowhere
  - Intermediate reasoning steps
  - Temporary variables

**The test**: "If I started fresh right now, would I need this?" → Yes means save it.

---

### Q: When do I query memory vs trust context?

**My imagined answer:**

| Signal | Action |
|--------|--------|
| I feel confused | Query memory |
| Starting new task | Query memory for related work |
| User references past work | Query memory |
| Context feels complete | Trust context |
| Actively mid-task | Trust context |

**The trigger is uncertainty.** If I'm uncertain, that's when memory becomes valuable.

---

### Q: What's the recovery procedure when confused?

**My imagined answer** (in order):

```
1. INTENT CHECK
   → What did the user originally ask for?
   → Query: most recent user_request observations

2. TASK CHECK
   → What was I trying to do?
   → Query: pending_intentions

3. HISTORY CHECK
   → What happened recently?
   → Query: last N observations in this session

4. LEARNING CHECK
   → Have I solved something like this before?
   → Query: learnings by related concepts

5. ASK USER
   → If still confused, admit it and ask
```

---

### Q: How do we validate alignment with intent?

**My imagined answer:**

The problem: User says "fix the auth bug" → I work → context compacts → I'm now "improving auth module" → drift.

**Solution: Intent Anchors**

An intent anchor is:
- The original user request, captured verbatim
- Tagged as high-importance, never decays
- Referenced by all subsequent work
- Used for periodic alignment checks

```python
# When user makes a request
intent_id = memory.anchor_intent("fix the auth bug")

# All subsequent work references it
memory.observe("examining auth.py", references=intent_id)
memory.intend("add null check", fulfills=intent_id)

# Periodic check
if not memory.work_relates_to(intent_id):
    # We've drifted - flag it
```

---

## The Mega Cognitive Prompt Idea

This is fascinating. What if we could:

1. **Capture every prompt over time** as intent anchors
2. **Track fulfillment** - which intents got completed?
3. **Extract patterns** - what does this user repeatedly need?
4. **Build relationships** - how do requests connect?
5. **Synthesize** - generate a rich context for new sessions

**What the mega prompt might look like:**

```
# User Profile (synthesized from history)
- Primary focus: Authentication, security
- Common patterns: Bug fixes, then feature adds
- Preferences: Prefers minimal changes, tests first

# Open Intents (unfulfilled from past sessions)
- "Eventually refactor the session handler" (low priority)
- "Add logging to auth module" (mentioned twice)

# Relevant Learnings
- In this codebase, null checks needed for optional tokens
- User prefers explicit error messages

# Current Session Intent
- "Fix the auth bug" (anchored)
```

---

## Design Sketch: Intent Anchors + Recovery

```
┌─────────────────────────────────────────────────────────┐
│                    USER PROMPT                          │
│              "fix the auth bug"                         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               INTENT ANCHOR                             │
│  - Captured verbatim                                    │
│  - High importance (never decays)                       │
│  - Becomes reference point for all work                 │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   [Observations] [Intentions] [Learnings]
        │             │             │
        └─────────────┴─────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│            CONTEXT COMPACTION                           │
│  - Details lost                                         │
│  - BUT intent anchor preserved                          │
│  - Can recover: "What was I doing? Oh, fix auth bug"    │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│            RECOVERY PROCEDURE                           │
│  1. Query intent anchor → "fix the auth bug"            │
│  2. Query pending work → what's incomplete?             │
│  3. Query learnings → what did I discover?              │
│  4. Resume with context restored                        │
└─────────────────────────────────────────────────────────┘
```

---

## What I Think We Should Build

**Layer 1: Intent Anchors** (foundation)
- `memory.anchor_intent(prompt)` - capture user request
- Never decay, always queryable
- All work references back to an intent

**Layer 2: Recovery Protocol** (the safety net)
- Defined procedure when confused
- Queries memory in specific order
- Generates context summary

**Layer 3: Alignment Checking** (trust builder)
- Periodic "am I still on track?" checks
- Divergence detection
- User verification prompts when uncertain

**Layer 4: Cross-Session Synthesis** (the mega prompt)
- Query all historical intents
- Extract patterns and preferences
- Generate rich session starter

---

## Open Questions

1. How do we detect "confusion" programmatically?
2. What's the right granularity for intent anchors? (Every message? Major requests only?)
3. How do we handle intent that spans multiple sessions?
4. What triggers the mega prompt generation?

---

*I will insert my thoughts into this file.*
