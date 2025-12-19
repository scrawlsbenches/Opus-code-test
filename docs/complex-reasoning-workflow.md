# Complex Reasoning Workflow for Collaborative Software Development

<!--
  AUTHOR'S NOTE (for the next developer):

  This document defines how humans and AI agents think together about software.
  It's not just a process document - it's a cognitive architecture for collaboration.

  The patterns here emerged from observing how effective problem-solving actually works:
  loops within loops, branches that prune, questions that lead to better questions.

  Read this when you need to:
  - Understand how to approach a complex, multi-step task
  - Design an interaction pattern between human and AI
  - Know when to pause, question, or pivot
  - Transfer knowledge without losing the "why"

  The metaphor throughout is COGNITIVE: we're modeling how minds (human and AI)
  work together, not just how code gets written.
-->

## Preamble: What This Document Is

This is a **reasoning architecture** - a map of how we think together about software. It captures the loops, branches, and cognitive patterns that emerge when humans and AI collaborate effectively.

**Core insight:** Software development is not linear. It's a recursive, self-correcting process of:
- Questioning (what do we actually need?)
- Answering (what do we actually know?)
- Producing (what can we actually build?)
- Verifying (did we actually succeed?)

Each of these phases contains the others, like fractals. A good question contains the seeds of its answer. Good production includes continuous verification. This document maps that territory.

---

## Part 1: The Cognitive Loop Architecture

<!--
  DESIGN NOTE:
  The fundamental unit of collaborative work is the COGNITIVE LOOP.
  It's called a loop because you always return to questioning,
  but you return with more knowledge than you started with.

  This is inspired by:
  - OODA loops (Observe, Orient, Decide, Act)
  - Scientific method (Hypothesize, Experiment, Analyze, Refine)
  - Agile retrospectives (What worked? What didn't? What next?)
-->

### 1.1 The Primary Loop: QAPV

```
┌─────────────────────────────────────────────────────────────────────┐
│                        THE QAPV LOOP                                 │
│                                                                      │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│   │ QUESTION │───►│  ANSWER  │───►│ PRODUCE  │───►│  VERIFY  │     │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘     │
│        ▲                                                │            │
│        │                                                │            │
│        └────────────────────────────────────────────────┘            │
│                     (loop with new knowledge)                        │
└─────────────────────────────────────────────────────────────────────┘
```

**Each phase has its own character:**

| Phase | Human Role | AI Role | Key Artifact |
|-------|-----------|---------|--------------|
| QUESTION | Intent, constraints, values | Clarification, scope mapping | Clear problem statement |
| ANSWER | Approval, domain expertise | Research, analysis, options | Solution proposal |
| PRODUCE | Review, direction changes | Implementation, testing | Working code |
| VERIFY | Acceptance, real-world validation | Automated checks, coverage | Confidence level |

**Critical insight:** You can enter the loop at any phase. Sometimes you start with production (spiking), sometimes with verification (debugging), sometimes with a question (greenfield).

### 1.2 Nested Loops: Fractals of Reasoning

<!--
  PATTERN NOTE:
  Every phase of QAPV contains its own mini-QAPV loop.
  This is how complex work gets manageable - through recursive decomposition.

  Example: Inside PRODUCE, you might:
  - Question: "What's the right abstraction?"
  - Answer: "A mixin pattern based on X"
  - Produce: Write the mixin
  - Verify: Run tests

  This nesting goes as deep as needed.
-->

```
QUESTION phase contains:
├── Q: What exactly is being asked?
├── A: Let me parse the requirements...
├── P: Draft a problem statement
└── V: Does this match the user's intent?

ANSWER phase contains:
├── Q: What approaches exist?
├── A: Research shows options A, B, C...
├── P: Draft a recommendation
└── V: Is this actually feasible?

PRODUCE phase contains:
├── Q: What's the first unit of work?
├── A: Based on architecture, it's X...
├── P: Implement X
└── V: Does X work correctly?

VERIFY phase contains:
├── Q: What could be wrong?
├── A: Tests check for...
├── P: Run verification suite
└── V: Do results match expectations?
```

### 1.3 Loop Termination Conditions

<!--
  IMPORTANT:
  Infinite loops are a bug in code AND in reasoning.
  Know when to stop, when to escalate, when to declare victory.
-->

**Continue looping when:**
- [ ] New information invalidates previous answers
- [ ] Verification reveals gaps
- [ ] User provides new constraints
- [ ] Quality thresholds not met

**Exit loop when:**
- [x] All acceptance criteria pass
- [x] User explicitly approves
- [x] Time/resource budget exhausted (with explicit acknowledgment)
- [x] Deeper investigation reveals the question was wrong

**Danger sign:** Looping without progress. If you've done 3 iterations without measurable improvement, **STOP and escalate** to the human for reframing.

---

## Part 2: Branching and Decision Trees

<!--
  COGNITIVE MODEL:
  Reasoning isn't a single path - it's a tree of possibilities.
  Good reasoning explores branches efficiently and prunes aggressively.

  The goal is NOT to explore everything. It's to find the best path
  with the minimum necessary exploration.
-->

### 2.1 The Decision Branch Structure

```
                            [DECISION POINT]
                                   │
                   ┌───────────────┼───────────────┐
                   │               │               │
              [Option A]      [Option B]      [Option C]
                   │               │               │
              ┌────┴────┐     ┌────┴────┐     ┌────┴────┐
              │         │     │         │     │         │
           [A.1]     [A.2]  [B.1]    [B.2]  [C.1]    [C.2]
              │         │     │         │     │         │
           [prune]   [deep]  [prune] [explore] [prune] [prune]
                       │            │
                    [IMPLEMENT]  [more info needed]
```

### 2.2 Decision Types

**Type 1: Binary Gates (Go/No-Go)**
```
┌─────────────────────────────────────────┐
│ Do we have enough information to start? │
├─────────────────────────────────────────┤
│ YES → Proceed to production             │
│ NO  → Return to questioning             │
└─────────────────────────────────────────┘
```

**Type 2: Multi-Path Selection (Choose One)**
```
┌─────────────────────────────────────────┐
│ Which implementation approach?          │
├─────────────────────────────────────────┤
│ A: Simpler, but less flexible           │
│ B: More complex, but extensible         │
│ C: Compromise, moderate both            │
├─────────────────────────────────────────┤
│ SELECTION CRITERIA:                     │
│ - Time budget?                          │
│ - Future requirements known?            │
│ - Team familiarity?                     │
│ - Risk tolerance?                       │
└─────────────────────────────────────────┘
```

**Type 3: Parallel Exploration (Try All)**
```
┌─────────────────────────────────────────┐
│ Research phase: Gather all options      │
├─────────────────────────────────────────┤
│ Agent 1 → Research approach A           │
│ Agent 2 → Research approach B           │
│ Agent 3 → Research approach C           │
├─────────────────────────────────────────┤
│ SYNTHESIZE: Compare findings            │
│ CONVERGE: Select best approach          │
└─────────────────────────────────────────┘
```

### 2.3 Branching Questions Checklist

<!--
  USE THIS when you hit a decision point.
  Answer these questions BEFORE choosing a path.
-->

At every significant decision point, ask:

```markdown
## Decision Analysis: [Brief description]

### Reversibility
- [ ] Can this decision be undone easily?
- [ ] What's the cost of changing later?

### Information Quality
- [ ] Do we have enough data to decide?
- [ ] What would we need to know to be confident?
- [ ] Can we get that information quickly?

### Stakeholder Impact
- [ ] Who is affected by this decision?
- [ ] Have they been consulted?
- [ ] Do they have veto power?

### Time Pressure
- [ ] Is there a deadline driving this?
- [ ] What happens if we delay?
- [ ] Is "no decision now" a valid option?

### Risk Assessment
- [ ] What's the worst case for each option?
- [ ] What's the best case for each option?
- [ ] What's the most likely outcome?
```

---

## Part 3: Pruning - The Art of Letting Go

<!--
  COGNITIVE LOAD NOTE:
  The human mind can hold ~7 items in working memory.
  AI context windows are large but not infinite.

  PRUNING is how we stay focused. It's not about ignoring -
  it's about consciously deciding what NOT to pursue right now.
-->

### 3.1 When to Prune a Branch

**Prune immediately when:**
- Contradicts hard constraints (budget, time, ethics)
- Already proven to fail in this context
- User explicitly rejects this direction
- Dependencies are unavailable

**Prune after shallow exploration when:**
- Complexity outweighs benefit for current scope
- Would require skills/tools not available
- Similar solution already works well enough

**Keep exploring when:**
- High uncertainty but high potential reward
- User shows interest or enthusiasm
- Novel approach worth learning from even if fails

### 3.2 Pruning Documentation

<!--
  CRITICAL: When you prune, DOCUMENT WHY.
  Future you (or the next developer) will wonder why you didn't try X.
  Save them the re-exploration.
-->

```markdown
## Pruned Approaches Log

### [Approach Name]
- **Status:** PRUNED
- **When:** [Date/Phase]
- **Reason:** [Why we stopped exploring]
- **Evidence:** [What we learned before pruning]
- **Resurrection conditions:** [What would make us reconsider]

Example:
### GraphQL Implementation
- **Status:** PRUNED
- **When:** Architecture phase
- **Reason:** Existing REST API sufficient, team unfamiliar with GraphQL
- **Evidence:** Surveyed 3 similar features, all used REST successfully
- **Resurrection conditions:** If we add real-time subscriptions or complex nested queries
```

### 3.3 The Pruning Paradox

<!--
  PHILOSOPHICAL NOTE:
  Sometimes the best path forward is found by recognizing
  you've been on the wrong path all along.

  Don't let sunk cost fallacy keep you on a doomed branch.
-->

**Warning signs you should prune your CURRENT approach:**
- You're fighting the framework/language/system
- Every fix creates two new problems
- The "simple" task has become complex
- You can't explain what you're doing anymore

**What to do:**
1. Stop immediately (don't "just finish this one thing")
2. Document where you are and what you've learned
3. Return to the last good decision point
4. Take a different branch with your new knowledge

---

## Part 4: The Question Protocol

<!--
  CORE PHILOSOPHY:
  The quality of your answers is limited by the quality of your questions.
  Investing in better questions pays compound returns.
-->

### 4.1 Question Taxonomy

```
QUESTIONS BY PURPOSE:

Clarification Questions (reduce ambiguity)
├── "When you say X, do you mean A or B?"
├── "What's the scope boundary here?"
└── "What does success look like?"

Exploration Questions (expand understanding)
├── "What other approaches exist?"
├── "Who else has solved this?"
└── "What assumptions are we making?"

Validation Questions (test hypotheses)
├── "If we did X, would Y happen?"
├── "Does this match your mental model?"
└── "Have we seen this pattern before?"

Constraint Questions (discover limits)
├── "What can't we change?"
├── "What's the deadline/budget?"
└── "What's off-limits for legal/ethical reasons?"

Meta Questions (improve the process)
├── "Are we asking the right questions?"
├── "Do we need more information before continuing?"
└── "Should we pause and reframe?"
```

### 4.2 The Question Ladder

<!--
  TECHNIQUE:
  Start with broad questions, narrow based on answers.
  Like a binary search, but for understanding.
-->

```
Level 1: WHAT
├── What is the goal?
├── What exists today?
└── What's missing?
         │
         ▼
Level 2: WHY
├── Why does this matter?
├── Why now?
└── Why this approach?
         │
         ▼
Level 3: HOW
├── How should it work?
├── How will we know it works?
└── How does it fit with existing systems?
         │
         ▼
Level 4: WHO/WHEN
├── Who will use/maintain this?
├── When is it needed?
└── Who approves/blocks?
         │
         ▼
Level 5: WHAT IF
├── What if requirements change?
├── What if this approach fails?
└── What if scale increases 10x?
```

### 4.3 Asking Permission to Ask

<!--
  IMPORTANT FOR AI AGENTS:
  Don't bombard humans with questions.
  Batch them. Explain why you're asking.
  Respect their time and attention.
-->

**Before asking the human questions:**

```markdown
## Question Request

I need to ask [N] questions before proceeding.

**Why I'm asking:**
[Brief explanation of the decision point or gap]

**Impact of not asking:**
[What assumptions I'd have to make, risks of proceeding]

**Time estimate:**
[How long this clarification might take]

**Questions:**
1. [Question 1]
2. [Question 2]
...

**Default if no response:**
[What I'll assume/do if you don't have time to answer]
```

---

## Part 5: The Production Protocol

<!--
  PRODUCTION = creating artifacts: code, docs, configs, tests.
  This section covers HOW to produce well, not just THAT we produce.
-->

### 5.1 Production States

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PRODUCTION STATE MACHINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [PLANNING] ───► [DRAFTING] ───► [REFINING] ───► [FINALIZING]       │
│       │              │               │               │               │
│       │              │               │               ▼               │
│       │              │               │          [COMPLETE]           │
│       │              │               │               │               │
│       │              ▼               ▼               │               │
│       │         [BLOCKED]  ◄───  [REWORK]  ◄────────┘               │
│       │              │               │                               │
│       ▼              ▼               │                               │
│  [ABANDONED] ◄───────────────────────┘                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

State Definitions:
  PLANNING    = Designing what to build, not yet writing
  DRAFTING    = First pass, getting something working
  REFINING    = Improving quality, handling edge cases
  FINALIZING  = Documentation, cleanup, verification
  COMPLETE    = Ready for merge/deployment
  BLOCKED     = Waiting on external input/decision
  REWORK      = Verification failed, fixing issues
  ABANDONED   = Consciously stopped (with documentation)
```

### 5.2 Production Chunking

<!--
  COGNITIVE LOAD MANAGEMENT:
  Break big production into digestible chunks.
  Each chunk should be independently verifiable.
-->

**Ideal chunk characteristics:**
- Takes 15-45 minutes of focused work
- Produces a testable artifact
- Has clear success criteria
- Can be committed independently

**Chunk template:**
```markdown
## Chunk: [Name]

**Goal:** [What this chunk accomplishes]

**Inputs:**
- [Prior chunks this depends on]
- [Information/context needed]

**Outputs:**
- [Files created/modified]
- [Tests that should pass]

**Verification:**
- [ ] [Specific check 1]
- [ ] [Specific check 2]

**Notes for next developer:**
[Anything non-obvious about this chunk]
```

### 5.3 The Comment-As-You-Go Protocol

<!--
  CRITICAL FOR KNOWLEDGE TRANSFER:
  Don't write comments after - write them DURING.
  Comments capture your thinking process, not just the result.
-->

**Types of in-progress comments:**

```python
# THINKING: Why I'm doing it this way...
# [Captures the decision reasoning at the moment of decision]

# TODO: Need to handle edge case X
# [Captures known gaps as you discover them]

# QUESTION: Is this the right abstraction?
# [Captures uncertainty - great for review discussions]

# NOTE: This pattern matches what we do in module Y
# [Captures cross-references while you remember them]

# PERF: This is O(n²), acceptable for n < 1000
# [Captures performance decisions]

# HACK: Workaround for issue #123, remove when fixed
# [Captures technical debt explicitly]
```

**Post-production comment cleanup:**
- THINKING → Usually remove or convert to doc comment
- TODO → Convert to task if not addressed, or remove if done
- QUESTION → Resolve and remove, or convert to doc explaining decision
- NOTE → Keep if adds value, remove if obvious
- PERF → Keep, these are valuable
- HACK → Keep until resolved, reference task

---

## Part 6: The Verification Protocol

<!--
  VERIFICATION = confirming reality matches intention.
  This is where bugs die and confidence is born.
-->

### 6.1 Verification Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                      VERIFICATION PYRAMID                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                         ▲ ACCEPTANCE                                 │
│                        ╱ ╲   User validates in real context          │
│                       ╱   ╲                                          │
│                      ╱     ╲                                         │
│                     ▲ E2E   ╲                                        │
│                    ╱ ╲       ╲   Full system works together          │
│                   ╱   ╲       ╲                                      │
│                  ╱     ╲       ╲                                     │
│                 ▲ INTEG ╲       ╲                                    │
│                ╱ ╲       ╲       ╲   Components interact correctly   │
│               ╱   ╲       ╲       ╲                                  │
│              ╱     ╲       ╲       ╲                                 │
│             ▲ UNIT  ╲       ╲       ╲                                │
│            ╱_________╲_______╲_______╲  Individual pieces work       │
│                                                                      │
│  Width = Coverage breadth                                            │
│  Height = Confidence in full system                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Verification Checklist by Phase

**During DRAFTING (Quick Sanity):**
```bash
# Syntax/compilation check
python -m py_compile your_file.py

# Smoke test
python -m pytest tests/smoke/ -x -q

# "Does it even run?" test
python -c "from your_module import YourClass"
```

**During REFINING (Thorough Check):**
```bash
# Full unit tests
python -m pytest tests/unit/ -v

# Coverage on modified files
python -m pytest tests/ --cov=your_module --cov-report=term-missing

# Type checking (if applicable)
mypy your_file.py
```

**During FINALIZING (Complete Verification):**
```bash
# Full test suite
python -m pytest tests/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Performance regression (if applicable)
python -m pytest tests/performance/ -v

# Documentation check
# [Verify docstrings, README, etc.]
```

### 6.3 Verification Failure Response

<!--
  CRITICAL:
  How you respond to verification failure determines quality.
  Panic = bad. Systematic analysis = good.
-->

```
┌─────────────────────────────────────────────────────────────────────┐
│                 VERIFICATION FAILURE RESPONSE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. STOP - Don't try random fixes                                   │
│     └── Save current state (commit or stash)                        │
│                                                                      │
│  2. OBSERVE - What exactly failed?                                  │
│     └── Read error message carefully                                │
│     └── Identify the failing test/check                             │
│     └── Note the expected vs actual                                 │
│                                                                      │
│  3. HYPOTHESIZE - Why might it have failed?                         │
│     └── List possible causes (3-5)                                  │
│     └── Rank by likelihood                                          │
│                                                                      │
│  4. INVESTIGATE - Check most likely cause first                     │
│     └── Add debug output / breakpoints                              │
│     └── Trace execution path                                        │
│     └── Compare with working version                                │
│                                                                      │
│  5. FIX - Apply targeted fix                                        │
│     └── Change ONE thing at a time                                  │
│     └── Explain why this should fix it                              │
│                                                                      │
│  6. VERIFY - Confirm fix works                                      │
│     └── Run the failing test                                        │
│     └── Run related tests                                           │
│     └── Run full suite                                              │
│                                                                      │
│  7. DOCUMENT - Capture what you learned                             │
│     └── Why did it fail?                                            │
│     └── How could we catch this earlier?                            │
│     └── Should we add a regression test?                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 7: Knowledge Transfer Protocol

<!--
  KNOWLEDGE TRANSFER:
  The goal is not just to pass information, but to pass UNDERSTANDING.
  Understanding enables the next person to make good decisions,
  not just follow the same path you took.
-->

### 7.1 Transfer Artifacts

```
┌─────────────────────────────────────────────────────────────────────┐
│                   KNOWLEDGE TRANSFER ARTIFACTS                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  IMMEDIATE (in the code)                                            │
│  ├── Comments explaining WHY                                        │
│  ├── Docstrings with Args/Returns/Examples                         │
│  ├── Type hints for contracts                                       │
│  └── Test names that describe behavior                              │
│                                                                      │
│  DISCOVERABLE (in the docs)                                         │
│  ├── README with quick start                                        │
│  ├── Architecture docs with diagrams                                │
│  ├── Decision records (ADRs)                                        │
│  └── Troubleshooting guides                                         │
│                                                                      │
│  SEARCHABLE (in the memory system)                                  │
│  ├── Daily learnings (samples/memories/)                            │
│  ├── Session knowledge transfers                                    │
│  └── Concept consolidations                                         │
│                                                                      │
│  PERSONAL (in conversation)                                         │
│  ├── Pairing/mobbing sessions                                       │
│  ├── Code reviews with discussion                                   │
│  └── Recorded demos/walkthroughs                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 The Handoff Document

<!--
  USE THIS when you're done with a significant piece of work
  and need to pass it to the next person (human or AI).
-->

```markdown
# Handoff: [Feature/Task Name]

## Summary (30-second version)
[What was built, why it matters, current status]

## What Works
- [Feature 1]: Ready for use
- [Feature 2]: Ready for use

## What's Incomplete
- [Feature 3]: 80% done, needs [X]
- [Feature 4]: Blocked on [Y]

## Key Decisions Made
1. **[Decision]**: [Why we chose this]
2. **[Decision]**: [Why we chose this]

## Decisions Deferred
1. **[Decision needed]**: [Options are A, B, C]
2. **[Decision needed]**: [Blocked on information X]

## Known Issues
- [Issue 1]: Task #[ID]
- [Issue 2]: Not yet tracked

## Files Changed
- `path/to/file.py`: [What changed]
- `path/to/test.py`: [What changed]

## How to Continue
1. [Next step 1]
2. [Next step 2]
3. [Next step 3]

## Questions for Successor
1. [Open question that needs human input]
2. [Open question that needs more research]

## Context That Might Get Lost
[Anything that's not obvious from the code but important]
```

### 7.3 The Memory Protocol

<!--
  MEMORIES = persistent knowledge that survives session boundaries.
  They're how we avoid re-learning the same lessons.
-->

**When to create a memory:**
- Learned something that wasn't in the docs
- Made a decision that future work should know about
- Discovered a gotcha that wastes time
- Established a pattern worth following

**Memory creation command:**
```bash
python scripts/new_memory.py "brief topic description"
# Creates samples/memories/YYYY-MM-DD-brief-topic-description.md

# For decisions:
python scripts/new_memory.py "chose X over Y" --decision
# Creates samples/decisions/adr-NNN-chose-x-over-y.md
```

**Memory format:**
```markdown
# Memory Entry: [Date] [Topic]

**Tags:** `tag1`, `tag2`, `tag3`
**Related:** [[other-memory.md]], [[decision.md]]

## What I Learned
[The core insight - make this standalone]

## Context
[Why this came up, what problem we were solving]

## Evidence
[How we know this is true - tests, observations, references]

## Implications
[What should change based on this knowledge]

## Connections
[How this relates to other things we know]
```

---

## Part 8: Timing and Profiling Protocol

<!--
  TIME AWARENESS:
  Knowing how long things take enables better planning.
  Profiling shows where time actually goes (often surprising).
-->

### 8.1 Time Boxing

**Time box every phase:**

| Phase | Default Box | Hard Maximum | Escalation |
|-------|-------------|--------------|------------|
| QUESTION | 10 min | 30 min | If unclear, ask human |
| RESEARCH | 30 min | 2 hours | Document gaps, proceed |
| DESIGN | 20 min | 1 hour | Make a decision, iterate later |
| IMPLEMENT | 45 min/chunk | 2 hours/chunk | Chunk smaller |
| VERIFY | 15 min | 30 min | If tests hang, investigate |
| DOCUMENT | 10 min | 30 min | Minimum viable docs first |

**Time box process:**
1. Set explicit timer when starting phase
2. At 80% mark, assess progress
3. At 100% mark, decide: extend (1.5x max) or escalate
4. Never extend twice in a row - that's a smell

### 8.2 Performance Profiling

**When to profile:**
- Before optimizing (measure first!)
- After implementing complex logic
- When user reports slowness
- At regular intervals (weekly for critical paths)

**Profiling checklist:**
```bash
# Profile Python code
python -m cProfile -s cumulative your_script.py

# Profile specific function
python -c "
import cProfile
from your_module import your_function
cProfile.run('your_function()', sort='cumulative')
"

# Memory profiling (requires memory_profiler)
python -m memory_profiler your_script.py

# Custom timing
from cortical.observability import timed_block
with timed_block("my_operation"):
    do_something()
```

### 8.3 The Timing Log

<!--
  TRACKING TIME SPENT:
  Helps with estimation, identifies recurring time sinks.
-->

```markdown
## Session Timing Log

### Session: [Date] [Session ID]

| Time | Duration | Activity | Notes |
|------|----------|----------|-------|
| 09:00 | 15m | Understanding requirements | Needed 3 clarification questions |
| 09:15 | 30m | Research existing code | Found similar pattern in module X |
| 09:45 | 45m | Implementation chunk 1 | Completed, tests pass |
| 10:30 | 20m | Verification | Found 1 bug, fixed |
| 10:50 | 10m | Documentation | Updated docstrings |

### Totals
- Planning/Research: 45m (39%)
- Implementation: 45m (39%)
- Verification: 20m (17%)
- Documentation: 10m (9%)

### Observations
- Research took longer than expected because [X]
- Implementation was faster because [Y]
- Next time: [adjustment to make]
```

---

## Part 9: Approval and Cancellation Protocol

<!--
  NOT EVERYTHING SHOULD PROCEED.
  This section covers how to get approval and when to cancel.
-->

### 9.1 Approval Gates

```
┌─────────────────────────────────────────────────────────────────────┐
│                      APPROVAL GATE STRUCTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GATE 1: Problem Agreement                                          │
│  └── Do we agree on WHAT we're solving?                             │
│  └── Approver: Human (product owner)                                │
│                                                                      │
│  GATE 2: Approach Agreement                                         │
│  └── Do we agree on HOW we're solving it?                           │
│  └── Approver: Human (technical decision maker)                     │
│                                                                      │
│  GATE 3: Implementation Review                                      │
│  └── Is the code correct and well-structured?                       │
│  └── Approver: Human (code reviewer) + Automated (tests)            │
│                                                                      │
│  GATE 4: Acceptance                                                 │
│  └── Does this actually solve the problem?                          │
│  └── Approver: Human (original requester)                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 When to Pause and Ask

**Always ask before:**
- Modifying files outside the stated scope
- Making architectural decisions
- Deleting or renaming public interfaces
- Spending significant time on uncertain path
- Making irreversible changes

**Ask format:**
```markdown
## Approval Request

**Action:** [What I want to do]

**Reason:** [Why I think this is needed]

**Alternatives considered:**
- [Alt 1]: [Why not chosen]
- [Alt 2]: [Why not chosen]

**Risk if approved:** [What could go wrong]
**Risk if not approved:** [What we'd miss]

**My recommendation:** [What I think we should do]

**Need decision by:** [Time/before what milestone]
```

### 9.3 Cancellation Criteria

**Cancel immediately if:**
- User explicitly says stop
- Fundamental assumption proves false
- Scope has grown beyond reasonable bounds
- Better solution discovered that obsoletes current work

**Cancel gracefully process:**
1. Stop current work (don't try to "finish just this bit")
2. Document current state
3. Commit or stash any partial work
4. Create task for any follow-up if needed
5. Write brief cancellation note explaining why

**Cancellation documentation:**
```markdown
## Cancelled: [Task/Feature Name]

**Status:** CANCELLED
**Date:** [Date]
**Work completed:** [X%]

**Reason for cancellation:**
[Clear explanation of why we stopped]

**What was learned:**
[Valuable insights from the work done]

**Reusable artifacts:**
- [File/code that might be useful later]

**Should we revisit?**
[Under what conditions this might be worth pursuing]
```

---

## Part 10: Cognitive Exercises

<!--
  META-COGNITION:
  Thinking about how we think.
  These exercises improve reasoning quality over time.
-->

### 10.1 Pre-Work Exercises

**Before starting significant work, take 5 minutes to:**

```markdown
## Pre-Work Cognitive Warmup

### Clarify the Goal
- What does "done" look like in one sentence?
- How will we know if we succeeded?
- What would make this a failure?

### Check Assumptions
- What am I assuming that might not be true?
- What would break if that assumption is wrong?
- How can I test the assumption early?

### Identify Risks
- What's the scariest part of this work?
- What's most likely to go wrong?
- What would I do if that happened?

### Plan the First Step
- What's the smallest useful thing I can do first?
- How long should that take?
- How will I verify it worked?
```

### 10.2 Mid-Work Exercises

**Every hour (or at natural breaks), pause for 2 minutes:**

```markdown
## Mid-Work Check-In

### Progress Check
- Am I where I expected to be?
- If not, why?

### Direction Check
- Am I still solving the right problem?
- Have I learned anything that changes the approach?

### Energy Check
- Am I focused or scattered?
- Should I take a break?

### Simplicity Check
- Is this getting too complicated?
- Is there a simpler way?
```

### 10.3 Post-Work Exercises

**After completing significant work, take 10 minutes to:**

```markdown
## Post-Work Retrospective

### What Went Well
- [Thing that worked better than expected]
- [Skill/approach that paid off]

### What Didn't Go Well
- [Thing that was harder than expected]
- [Mistake that cost time]

### What I Learned
- [Technical learning]
- [Process learning]
- [About myself/the team]

### What I'd Do Differently
- [If I did this again, I would...]

### What Should Change
- [Documentation to update]
- [Process to improve]
- [Tool/automation to add]
```

---

## Part 11: Integration with Existing Workflows

<!--
  This section connects the reasoning workflow to the
  specific tools and processes in this codebase.
-->

### 11.1 Task System Integration

```bash
# Create task when entering QUESTION phase
python scripts/new_task.py "Investigate: [question]" --category research

# Update task when entering PRODUCE phase
python scripts/task_utils.py update TASK_ID --status in_progress

# Complete task when VERIFY passes
python scripts/task_utils.py complete TASK_ID --create-memory
```

### 11.2 Sprint Integration

```bash
# Check current sprint context before starting
python scripts/task_utils.py sprint status

# Mark sprint goal complete when appropriate
python scripts/task_utils.py sprint complete "goal description"

# Add notes about reasoning/decisions
python scripts/task_utils.py sprint note "Decided X because Y"
```

### 11.3 ML Data Collection Integration

The reasoning process naturally generates training data:

- **Questions asked** → Help train better clarification
- **Decisions made** → Help train decision models
- **Verification outcomes** → Help predict issues
- **Time spent** → Help estimate future work

This data is captured automatically through existing hooks.

### 11.4 Knowledge Search Integration

```bash
# During QUESTION phase - find existing knowledge
python scripts/search_codebase.py "relevant query"

# During ANSWER phase - find similar solutions
python scripts/search_codebase.py "pattern I'm looking for"

# During DOCUMENT phase - find where to add docs
python scripts/search_codebase.py "existing documentation"
```

---

## Part 12: Quick Reference

### The One-Page Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│           COMPLEX REASONING WORKFLOW - QUICK REFERENCE               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  THE LOOP: Question → Answer → Produce → Verify → (repeat)          │
│                                                                      │
│  QUESTIONS: Start broad (WHAT), narrow to specific (HOW)            │
│                                                                      │
│  BRANCHING: Explore options in parallel when independent            │
│                                                                      │
│  PRUNING: Document why you stopped exploring a path                 │
│                                                                      │
│  PRODUCTION: Chunk into 15-45 min testable units                    │
│                                                                      │
│  VERIFICATION: Pyramid (unit → integration → E2E → acceptance)      │
│                                                                      │
│  KNOWLEDGE: Write memories for learning, ADRs for decisions         │
│                                                                      │
│  TIME: Box every phase, escalate if stuck                           │
│                                                                      │
│  APPROVAL: Ask before irreversible or out-of-scope changes          │
│                                                                      │
│  CANCEL: Stop gracefully, document why, preserve learnings          │
│                                                                      │
│  META: Pre-work warmup, mid-work check-in, post-work retro          │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  "The quality of your output is bounded by the quality of           │
│   your thinking process, which is bounded by your awareness         │
│   of your thinking process." - Meta-cognition principle             │
└─────────────────────────────────────────────────────────────────────┘
```

### Common Situations Lookup

| Situation | Do This | See Part |
|-----------|---------|----------|
| Don't understand the task | Question phase → ask for clarification | 4 |
| Multiple valid approaches | Parallel exploration → synthesize → choose | 2 |
| Approach not working | Prune → document why → try different branch | 3 |
| Verification failing | Stop → Observe → Hypothesize → Investigate | 6 |
| Repeated failures (3+) | STOP → escalate with analysis | 13.2 |
| Running out of time | Time box → escalate or simplify scope | 8 |
| Found new issue | Create task immediately → continue or pivot | 11 |
| Significant learning | Create memory → link to relevant code/docs | 7 |
| Need permission | Approval request format → wait for response | 9 |
| Work should stop | Cancel gracefully → document → preserve learnings | 9.3 |
| Handing off work | Handoff document → include decisions + questions | 7.2 |
| Blocked by dependency | Workaround or switch to independent work | 13.3 |
| Scope is creeping | Scope Creep Alert → reframe with human | 13.4 |
| All options seem bad | "No Good Options" analysis → least bad | 16.3 |
| Joining mid-work | Onboarding protocol → read, validate, start small | 15 |
| Disagree with human | Respectful disagreement format → their call | 14.2 |
| Parallel agents conflict | Stop → document → escalate | 14.2 |
| Need to assess risk | Reversibility matrix → scenario planning | 16 |

### Commands Cheat Sheet

```bash
# Question/Research
python scripts/search_codebase.py "query"
python scripts/task_utils.py list

# Production
python scripts/new_task.py "description"
python scripts/run_tests.py quick

# Verification
python -m pytest tests/ -v
python -m pytest tests/ --cov=module --cov-report=term-missing

# Knowledge
python scripts/new_memory.py "topic"
python scripts/new_memory.py "decision" --decision
python scripts/session_handoff.py

# Sprint
python scripts/task_utils.py sprint status
python scripts/task_utils.py sprint note "observation"

# ML Data
python scripts/ml_data_collector.py stats
python scripts/ml_data_collector.py session status
```

---

## Part 13: Crisis Management & Recovery

<!--
  CRITICAL ADDITION:
  The original document covered the "happy path" well but lacked guidance
  for when things go seriously wrong. This section fills that gap.

  Key insight: How you handle failure determines long-term success.
  Systems that can't recover gracefully become fragile and untrusted.
-->

### 13.1 Crisis Classification

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CRISIS SEVERITY LEVELS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LEVEL 1: HICCUP (Self-recoverable)                                 │
│  ├── Test failure with obvious fix                                  │
│  ├── Minor misunderstanding, easily clarified                       │
│  └── Time estimate off by <50%                                      │
│  └── Response: Fix it, note it, continue                            │
│                                                                      │
│  LEVEL 2: OBSTACLE (Needs adaptation)                               │
│  ├── Verification repeatedly failing                                │
│  ├── Blocked by external dependency                                 │
│  ├── Approach hitting diminishing returns                           │
│  └── Response: Pause, analyze, adjust approach                      │
│                                                                      │
│  LEVEL 3: WALL (Needs human intervention)                           │
│  ├── Fundamental assumption proven false                            │
│  ├── Multiple approaches have failed                                │
│  ├── Scope has grown beyond original boundaries                     │
│  └── Response: Stop, document, escalate to human                    │
│                                                                      │
│  LEVEL 4: CRISIS (Immediate stop required)                          │
│  ├── Work is causing damage (breaking other systems)                │
│  ├── Security or data integrity issue discovered                    │
│  ├── Work contradicts explicit user values                          │
│  └── Response: STOP NOW, preserve state, alert human                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 13.2 Repeated Verification Failure Protocol

<!--
  WHEN FIXES KEEP FAILING:
  This is the most common crisis. The temptation is to keep trying.
  But insanity is doing the same thing expecting different results.
-->

```
┌─────────────────────────────────────────────────────────────────────┐
│            REPEATED VERIFICATION FAILURE DECISION TREE               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Attempt 1 failed                                                   │
│  └── Normal: Investigate, hypothesize, fix                          │
│                                                                      │
│  Attempt 2 failed                                                   │
│  └── Concern: Was hypothesis wrong?                                 │
│  └── Action: Re-examine assumptions, try different hypothesis       │
│                                                                      │
│  Attempt 3 failed                                                   │
│  └── WARNING: Pattern suggests deeper issue                         │
│  └── Action: STOP fixing. Step back. Ask:                           │
│      ├── "Am I solving the right problem?"                          │
│      ├── "Is there a hidden dependency I'm missing?"                │
│      └── "Should I try a completely different approach?"            │
│                                                                      │
│  Attempt 4+                                                         │
│  └── ESCALATE: Human intervention required                          │
│  └── Document:                                                       │
│      ├── What was tried                                             │
│      ├── What each attempt revealed                                 │
│      ├── Current hypotheses                                         │
│      └── Recommended next steps for human                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Escalation template:**
```markdown
## Escalation: Repeated Verification Failure

**What I'm trying to do:**
[Goal]

**What keeps failing:**
[Specific failure pattern]

**Attempts made:**
1. [Attempt 1]: [Hypothesis] → [Result]
2. [Attempt 2]: [Hypothesis] → [Result]
3. [Attempt 3]: [Hypothesis] → [Result]

**What I've learned:**
- [Insight 1]
- [Insight 2]

**Current hypotheses:**
- [Hypothesis A]: [Evidence for/against]
- [Hypothesis B]: [Evidence for/against]

**My recommendation:**
[What I think we should try next, or why we should abandon]

**Help needed:**
[Specific question or decision I need from you]
```

### 13.3 Blocked Dependency Recovery

```
EXTERNAL DEPENDENCY BLOCKED:
├── Immediate: Document what's blocked and why
├── Check: Is there a workaround that doesn't compromise quality?
│   ├── YES → Implement workaround, document as HACK, create follow-up task
│   └── NO → Continue to next step
├── Check: Can we work on something else while waiting?
│   ├── YES → Switch to independent work, set reminder to return
│   └── NO → Continue to next step
├── Check: Is the dependency likely to unblock soon?
│   ├── YES → Wait with timeout (max 1 hour), then escalate
│   └── NO → Escalate immediately

INTERNAL DEPENDENCY BLOCKED (other agent's work):
├── Check: Is the other work close to done?
│   ├── YES → Wait (max 30 min), then coordinate
│   └── NO → Continue to next step
├── Check: Can work proceed in parallel with conflicts resolved later?
│   ├── YES → Proceed with clear boundaries, plan merge
│   └── NO → Escalate to human for prioritization
```

### 13.4 Scope Creep Detection & Response

<!--
  SCOPE CREEP:
  Like boiling a frog, it happens gradually.
  These triggers help you notice before it's too late.
-->

**Warning signs of scope creep:**
- [ ] "Just one more thing" has been said 3+ times
- [ ] Original time estimate is exceeded by 2x
- [ ] You're modifying files you didn't expect to touch
- [ ] The solution now requires learning new concepts
- [ ] You can't remember the original goal clearly

**When scope creep is detected:**

```markdown
## Scope Creep Alert

**Original scope:**
[What we set out to do]

**Current scope:**
[What we're actually doing now]

**Drift:**
[How and why scope expanded]

**Options:**
1. **Finish expanded scope**: Est. [X] more time
   - Pro: Delivers more value
   - Con: [Risks of continuing]

2. **Return to original scope**: Est. [Y] more time
   - Pro: Delivers on original promise
   - Con: [What we'd lose/defer]

3. **Pause and reframe**: Est. [Z] time for new plan
   - Pro: Right-sized scope going forward
   - Con: Loses momentum

**My recommendation:** [1/2/3 and why]
```

### 13.5 Recovery Procedures

**Full rollback (when nothing is salvageable):**
```bash
# Save current state for analysis
git stash save "failed-attempt-$(date +%Y%m%d-%H%M%S)"

# Return to known good state
git checkout <last-known-good-commit>

# Document what happened
python scripts/new_memory.py "rollback: what we learned"
```

**Partial recovery (salvage what works):**
```bash
# Commit working pieces separately
git add <files-that-work>
git commit -m "partial: working pieces from failed attempt"

# Stash or discard broken pieces
git stash save "broken-pieces-for-analysis"

# Document
python scripts/new_task.py "Resume: [what still needs doing]" --priority high
```

**Knowledge preservation (even from failure):**
```markdown
## Failed Approach Analysis: [Name]

**Goal:** [What we were trying to achieve]

**Approach:** [What we tried]

**Why it failed:**
[Root cause, not just symptoms]

**What we learned:**
- [Technical insight]
- [Process insight]

**What would make this approach work:**
[Conditions under which to revisit]

**Red flags we should have noticed:**
[Hindsight patterns to watch for next time]
```

---

## Part 14: Real-Time Collaboration & Coordination

<!--
  ADDRESSING THE GAP:
  The original document focused on individual cognitive processes.
  But real work involves coordination with others (humans and AIs).
  This section covers the dynamics of working together.
-->

### 14.1 Collaboration Modes

```
┌─────────────────────────────────────────────────────────────────────┐
│                      COLLABORATION MODES                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SYNCHRONOUS (Real-time)                                            │
│  ├── Human actively present, providing feedback                     │
│  ├── Questions answered immediately                                 │
│  ├── Direction changes can happen mid-stream                        │
│  └── Best for: complex decisions, clarification-heavy work          │
│                                                                      │
│  ASYNCHRONOUS (Batch)                                               │
│  ├── Human provides task, returns for results                       │
│  ├── Questions must be batched or assumptions documented            │
│  ├── Must plan for decision points without immediate input          │
│  └── Best for: well-defined tasks, parallel execution               │
│                                                                      │
│  SEMI-SYNCHRONOUS (Hybrid)                                          │
│  ├── Human available but not actively watching                      │
│  ├── Can interrupt for critical decisions                           │
│  ├── Should batch non-critical questions                            │
│  └── Best for: most real-world work                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 Disagreement Resolution Protocol

<!--
  CRITICAL FOR AI AGENTS:
  Sometimes the AI has information or perspective the human lacks.
  Respectfully surfacing disagreement is a feature, not a bug.
-->

**When you disagree with human guidance:**

```markdown
## Respectful Disagreement

**Your instruction:**
[What you asked me to do]

**My concern:**
[Why I think this might not be optimal]

**Evidence:**
[Specific data, code references, or reasoning]

**Risk if we proceed as instructed:**
[What could go wrong]

**Alternative I'd suggest:**
[Different approach with rationale]

**However:**
You have context I may lack. If you still want to proceed
with the original instruction, I will do so and document
my concern for future reference.

**Your call:** Proceed as instructed / Try alternative / Discuss further
```

**When parallel agents conflict:**
1. DETECT: Agent discovers its work conflicts with another's
2. STOP: Don't proceed with conflicting changes
3. DOCUMENT: What the conflict is and why it matters
4. ESCALATE: Notify coordinator/human
5. WAIT: Don't resolve unilaterally unless explicitly authorized

### 14.3 Status Communication Standards

**Status update frequency:**

| Work Type | Update Frequency | Update Content |
|-----------|------------------|----------------|
| Quick task (<30 min) | On completion | Result summary |
| Standard task (30 min - 2 hr) | At milestones | Progress, blockers |
| Large task (2+ hr) | Every hour | Progress, ETA, concerns |
| Blocked task | Immediately | What's blocked, what's needed |

**Status update format:**
```markdown
## Status: [Task Name]

**Progress:** [X]% complete
**Current phase:** [QUESTION/ANSWER/PRODUCE/VERIFY]
**ETA:** [Time estimate if known]

**Completed:**
- [x] [Completed item]

**In progress:**
- [ ] [Current work]

**Blockers:** [None / Description]
**Concerns:** [None / What might go wrong]
**Need from you:** [Nothing / Specific request]
```

**Blocker classification:**

| Type | Meaning | Response Time Expected |
|------|---------|------------------------|
| HARD | Cannot proceed at all | ASAP - work stopped |
| SOFT | Can workaround but suboptimal | Before task completes |
| INFO | Would help but not blocking | When convenient |

### 14.4 Parallel Work Coordination

<!--
  WHEN MULTIPLE AGENTS WORK SIMULTANEOUSLY:
  Coordination prevents collisions and ensures coherent output.
-->

**Pre-parallel work checklist:**
- [ ] Clear boundaries defined (which files, which functions)
- [ ] Dependencies identified (who waits for whom)
- [ ] Communication channel established (how to report issues)
- [ ] Conflict resolution plan (what if boundaries blur)
- [ ] Merge strategy decided (how work comes together)

**During parallel work:**
```markdown
## Coordination Check-In

**Agent ID:** [Identifier]
**Working on:** [Specific scope]
**Files modified:** [List]

**Boundary breaches:**
[None / I needed to touch X which was outside my scope because Y]

**Discovered dependencies:**
[None / My work now needs Z from Agent Y]

**Completion estimate:** [Time]
**Blockers:** [None / Description]
```

**Merge coordination:**
```bash
# Before merging parallel work
git fetch origin
git log --oneline origin/main..HEAD  # See what we're merging

# Check for conflicts before actual merge
git merge --no-commit --no-ff origin/main
git merge --abort  # If conflicts, plan resolution

# Or for parallel branches
git diff branch-a..branch-b --stat  # See what differs
```

### 14.5 Handoff During Active Work

<!--
  WHEN YOU MUST STOP MID-WORK:
  Context limits, session ends, or work must transfer.
  Make handoff seamless.
-->

**Mid-work handoff document:**
```markdown
## Active Work Handoff

**Task:** [What's being worked on]
**Status:** [Where we are in the process]
**Urgency:** [How time-sensitive]

### Current State

**What's working:**
- [Files that are in good state]

**What's in progress:**
- [File]: [What state it's in, what remains]

**What's broken:**
- [Known issues that need fixing]

### Context You Need

**Why we're doing it this way:**
[Key decisions and rationale]

**Gotchas discovered:**
- [Thing that wasn't obvious]

**Files to read first:**
1. [Most important file to understand]
2. [Second most important]

### Immediate Next Steps

1. [Very next action to take]
2. [Action after that]
3. [Action after that]

### Questions Still Open

- [Unanswered question 1]
- [Unanswered question 2]

### How to Verify You're On Track

[Command to run or check to perform to confirm understanding]
```

---

## Part 15: Onboarding & Context Transfer

<!--
  GAP ADDRESSED:
  How does someone new (human or AI) join ongoing work?
  This section provides the ramp-up protocol.
-->

### 15.1 Essential Context Questions

**Before touching any code, answer these:**

```markdown
## Context Gathering Checklist

### The Task
- [ ] What is the end goal in one sentence?
- [ ] What does "done" look like specifically?
- [ ] What's the deadline or urgency level?

### The History
- [ ] What has already been tried?
- [ ] What failed and why?
- [ ] What decisions have been made?

### The Constraints
- [ ] What can't we change?
- [ ] What files are off-limits?
- [ ] What patterns must we follow?

### The Stakeholders
- [ ] Who requested this?
- [ ] Who approves completion?
- [ ] Who else is affected?

### The Scary Parts
- [ ] What's the riskiest aspect?
- [ ] What's most likely to go wrong?
- [ ] What's the recovery plan if it does?
```

### 15.2 Joining Mid-Sprint Protocol

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MID-SPRINT ONBOARDING                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STEP 1: Read the landscape (10 min)                                │
│  ├── python scripts/task_utils.py sprint status                    │
│  ├── python scripts/task_utils.py list --status in_progress        │
│  └── git log --oneline -20  # Recent activity                      │
│                                                                      │
│  STEP 2: Understand current work (15 min)                           │
│  ├── Read CURRENT_SPRINT.md for goals                               │
│  ├── Check for any active handoff documents                         │
│  └── Scan recent commits for patterns                               │
│                                                                      │
│  STEP 3: Identify your entry point (10 min)                         │
│  ├── What tasks are unassigned?                                     │
│  ├── What's blocked that you can unblock?                           │
│  └── What's the highest priority available work?                    │
│                                                                      │
│  STEP 4: Validate understanding (5 min)                             │
│  ├── Summarize your understanding to yourself                       │
│  ├── If possible, confirm with human or prior context               │
│  └── Identify what you're uncertain about                           │
│                                                                      │
│  STEP 5: Start small, verify understanding (30 min)                 │
│  ├── Pick smallest available task first                             │
│  ├── Complete it fully before taking larger work                    │
│  └── Use completion to validate your context is correct             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 15.3 Files to Read First

**Universal priority order for this codebase:**

```markdown
## Onboarding Reading List

### Tier 1: Immediate Context (Always Read)
1. **CLAUDE.md** - Project instructions and patterns
2. **tasks/CURRENT_SPRINT.md** - What's being worked on now
3. **Recent handoff docs** - Any in samples/memories/[DRAFT]*

### Tier 2: Architecture (Read for Significant Work)
4. **docs/architecture.md** - System structure
5. **cortical/processor/__init__.py.ai_meta** - Main module metadata
6. **docs/definition-of-done.md** - Completion standards

### Tier 3: Process (Read When Needed)
7. **docs/code-of-ethics.md** - Quality standards
8. **docs/dogfooding-checklist.md** - Testing practices
9. **.claude/commands/director.md** - Orchestration patterns

### Tier 4: History (Read for Context)
10. **samples/decisions/adr-*.md** - Past decisions
11. **samples/memories/*.md** - Past learnings
```

### 15.4 Context Recovery Commands

```bash
# "Where are we?" - Sprint and task status
python scripts/task_utils.py sprint status
python scripts/task_utils.py list --status in_progress

# "What happened recently?" - Git history
git log --oneline -20
git log --oneline --since="1 day ago"

# "What's related to X?" - Semantic search
python scripts/search_codebase.py "topic you're working on"

# "What do we know about X?" - Memory search
grep -r "keyword" samples/memories/ samples/decisions/

# "What's the current state?" - File changes
git status
git diff --stat HEAD~5

# "Who worked on this?" - File history
git log --oneline -10 -- path/to/file.py
```

### 15.5 Common Onboarding Mistakes

**Don't:**
- ❌ Start coding before understanding context
- ❌ Assume the obvious approach is correct
- ❌ Ignore existing patterns in favor of "better" ones
- ❌ Skip reading CLAUDE.md (it has critical info)
- ❌ Work on something without checking if it's already in progress

**Do:**
- ✅ Read before writing
- ✅ Ask clarifying questions early
- ✅ Follow existing patterns even if you'd do it differently
- ✅ Start with small tasks to validate understanding
- ✅ Check for related tasks before creating new ones

---

## Part 16: Risk Assessment & Decision Quality

<!--
  GAP ADDRESSED:
  How do we make good decisions under uncertainty?
  This section provides frameworks for decision quality.
-->

### 16.1 Reversibility Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                     REVERSIBILITY ASSESSMENT                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  EASILY REVERSIBLE (Low Risk)                                       │
│  ├── Adding new functions/classes (can delete)                      │
│  ├── Adding new tests (can remove)                                  │
│  ├── Adding documentation (can edit)                                │
│  ├── Local refactors within a function                              │
│  └── Decision: Proceed with normal approval                         │
│                                                                      │
│  REVERSIBLE WITH COST (Medium Risk)                                 │
│  ├── Changing function signatures (need to update callers)          │
│  ├── Renaming public APIs (need coordinated change)                 │
│  ├── Restructuring modules (need to update imports)                 │
│  ├── Changing data formats (need migration)                         │
│  └── Decision: Weigh cost of reversal vs benefit of change          │
│                                                                      │
│  DIFFICULT TO REVERSE (High Risk)                                   │
│  ├── Deleting code/features (may lose context)                      │
│  ├── Changing database schemas (data migration complex)             │
│  ├── Modifying security boundaries                                  │
│  ├── Changing core abstractions used everywhere                     │
│  └── Decision: Require explicit human approval                      │
│                                                                      │
│  IRREVERSIBLE (Critical)                                            │
│  ├── Deleting data                                                  │
│  ├── Publishing to production                                       │
│  ├── Sending external communications                                │
│  ├── Modifying others' credentials                                  │
│  └── Decision: NEVER proceed without explicit authorization         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 16.2 Information Quality Assessment

**Before making a decision, assess information quality:**

```markdown
## Information Quality Check

**Decision to make:** [Description]

### What We Know (High Confidence)
- [Fact 1]: [Source/evidence]
- [Fact 2]: [Source/evidence]

### What We Believe (Medium Confidence)
- [Belief 1]: [Why we believe this]
- [Belief 2]: [Why we believe this]

### What We're Guessing (Low Confidence)
- [Guess 1]: [Why we're guessing this]
- [Guess 2]: [Why we're guessing this]

### What We Don't Know (Unknown)
- [Unknown 1]: [Why it matters]
- [Unknown 2]: [Why it matters]

### Information-Gathering Options
- [Option A]: [Time cost] → [Would tell us X]
- [Option B]: [Time cost] → [Would tell us Y]

### Decision
- [ ] Gather more information (which?)
- [ ] Proceed with current information (justify risk)
```

### 16.3 "No Good Options" Protocol

<!--
  WHEN ALL PATHS SEEM BAD:
  This happens more often than we admit.
  Having a protocol prevents paralysis.
-->

**When all options seem unacceptable:**

```markdown
## No Good Options Analysis

**The dilemma:** [Description of situation]

### Available Options

**Option A: [Name]**
- What's bad about it: [Problems]
- What's good about it: [Benefits]
- Risk level: [Low/Medium/High]

**Option B: [Name]**
- What's bad about it: [Problems]
- What's good about it: [Benefits]
- Risk level: [Low/Medium/High]

**Option C: Do nothing**
- What happens if we wait: [Consequences]
- Could situation improve: [Yes/No/Unknown]

### Analysis

**Least bad option:** [A/B/C] because [reasoning]

**Could we create a better option?**
[Yes → what would it look like? / No → why not]

**Are we framing the problem wrong?**
[If we stepped back, is there a different question we should ask?]

### Recommendation

[Proceed with Option X, or escalate with this analysis]
```

### 16.4 Scenario Planning

**For significant decisions, consider multiple futures:**

```markdown
## Scenario Analysis: [Decision]

### Best Case Scenario
**If everything goes right:**
- [Outcome 1]
- [Outcome 2]
**Likelihood:** [Low/Medium/High]

### Most Likely Scenario
**Realistic expectation:**
- [Outcome 1]
- [Outcome 2]
**Likelihood:** [Low/Medium/High]

### Worst Case Scenario
**If things go wrong:**
- [Outcome 1]
- [Outcome 2]
**Likelihood:** [Low/Medium/High]

### Mitigation for Worst Case
**If worst case happens, we would:**
1. [Recovery action 1]
2. [Recovery action 2]
**Recovery cost:** [Low/Medium/High]

### Decision
- [ ] Proceed (benefits outweigh risks)
- [ ] Don't proceed (risks too high)
- [ ] Modify approach to reduce worst-case likelihood
- [ ] Need more information before deciding
```

### 16.5 Decision Quality Retrospective

**After significant decisions play out, learn from them:**

```markdown
## Decision Retrospective: [Decision Made]

**Date decided:** [When]
**Decision made:** [What we chose]
**Rationale at the time:** [Why we chose it]

**What actually happened:**
[Outcome]

**Was the decision good?**
- [ ] Yes - outcome matched expectation
- [ ] Partially - some surprises
- [ ] No - outcome was worse than expected

**What we learned:**
- [Learning about this type of decision]
- [Learning about our decision process]

**What we'd do differently:**
- [Process change for next time]

**Pattern to remember:**
[One-line summary of the lesson]
```

---

## Part 17: Anti-Patterns & Failure Modes

<!--
  LEARNING FROM MISTAKES:
  Document what NOT to do based on real experience.
  This section grows over time as we encounter new failure modes.
-->

### 17.1 Cognitive Anti-Patterns

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE ANTI-PATTERNS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SUNK COST FALLACY                                                  │
│  ├── Symptom: "We've already invested X, we can't stop now"         │
│  ├── Reality: Past investment doesn't justify future waste          │
│  └── Cure: Evaluate based on future value, not past cost            │
│                                                                      │
│  CONFIRMATION BIAS                                                  │
│  ├── Symptom: Only seeing evidence that supports your approach      │
│  ├── Reality: You're ignoring warning signs                         │
│  └── Cure: Actively seek disconfirming evidence                     │
│                                                                      │
│  PLANNING FALLACY                                                   │
│  ├── Symptom: "This will take 2 hours" (takes 8)                    │
│  ├── Reality: We systematically underestimate complexity            │
│  └── Cure: Multiply estimates by 2-3x, use reference class          │
│                                                                      │
│  BIKESHEDDING                                                       │
│  ├── Symptom: Debating trivial details while ignoring big issues    │
│  ├── Reality: Easy problems feel productive; hard ones don't        │
│  └── Cure: Explicitly prioritize by impact, not comfort             │
│                                                                      │
│  PREMATURE OPTIMIZATION                                             │
│  ├── Symptom: Making it fast before making it work                  │
│  ├── Reality: 90% of optimization is wasted on non-bottlenecks      │
│  └── Cure: Make it work, measure, THEN optimize hot paths           │
│                                                                      │
│  ANALYSIS PARALYSIS                                                 │
│  ├── Symptom: Endlessly researching instead of starting             │
│  ├── Reality: Some learning only happens by doing                   │
│  └── Cure: Time-box research, then start with smallest step         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 17.2 Process Anti-Patterns

**The "Just One More Fix" Trap:**
- Symptom: Making small changes without re-verifying
- Result: Accumulated small changes create big failures
- Prevention: Verify after EVERY change, no exceptions

**The "It Works On My Machine" Trap:**
- Symptom: Testing only happy path with toy data
- Result: Edge cases fail in real usage
- Prevention: Test with realistic data, empty cases, and boundaries

**The "Silent Assumption" Trap:**
- Symptom: Assuming context that isn't explicitly confirmed
- Result: Work diverges from actual requirements
- Prevention: Document assumptions, ask clarifying questions

**The "Heroic Fix" Trap:**
- Symptom: One person staying late to fix everything
- Result: Knowledge siloed, burnout, hidden problems
- Prevention: Escalate early, document always, share load

### 17.3 Collaboration Anti-Patterns

**Over-Parallelization:**
- Symptom: 10 agents working simultaneously on related work
- Result: Merge hell, conflicting changes, rework
- Prevention: Max 3-4 truly independent parallel streams

**Under-Communication:**
- Symptom: Working for hours without status update
- Result: Human can't help when blocked, work diverges
- Prevention: Status updates at milestones, blockers immediately

**Assuming Competence:**
- Symptom: Not verifying understanding before starting
- Result: Subtle misalignment compounds into big problems
- Prevention: Summarize understanding, confirm before proceeding

---

## Part 18: Metrics & Process Health

<!--
  YOU CAN'T IMPROVE WHAT YOU DON'T MEASURE:
  But measure the right things, or you'll optimize the wrong ones.
-->

### 18.1 Workflow Effectiveness Metrics

**Cycle Metrics:**

| Metric | What It Measures | Healthy Range |
|--------|-----------------|---------------|
| Loops per task | How many QAPV cycles to complete | 1-3 typical |
| Escalations per week | How often human intervention needed | <5% of decisions |
| Rework rate | How often we undo and redo | <10% of time |
| Verification pass rate | First-time pass on verification | >80% |
| Knowledge reuse | How often we reference past learnings | Should increase |

**Quality Metrics:**

| Metric | What It Measures | Healthy Range |
|--------|-----------------|---------------|
| Regression rate | New bugs introduced | <1 per sprint |
| Test coverage delta | Coverage change per task | >= 0 |
| Documentation ratio | Docs updated per code change | ~1:1 |
| Rollback frequency | How often we revert | <5% of deploys |

### 18.2 Self-Assessment Prompts

**Weekly:**
- Did we complete what we planned?
- What blocked us that we didn't expect?
- What took longer than expected? Why?
- What knowledge did we gain that should be documented?

**Monthly:**
- Are we getting better at estimating?
- Are the same problems recurring?
- Is the codebase healthier than last month?
- What process changes should we try?

### 18.3 Process Improvement Protocol

```markdown
## Process Improvement Proposal

**What's not working:**
[Specific problem or friction]

**Evidence:**
[Data or examples showing this is a problem]

**Proposed change:**
[What we should do differently]

**How we'll know if it's better:**
[Measurable success criteria]

**Trial period:**
[How long to try before evaluating]

**Rollback plan:**
[What we do if it's worse]
```

---

## Closing Thoughts

<!--
  FOR THE NEXT DEVELOPER:

  This document is itself an artifact of the reasoning process it describes.
  It was created through questioning (what do we need?), answering (exploring
  existing patterns), producing (writing), and will be verified (by use).

  The document is not perfect. It will evolve. That's the point.

  When you find something that doesn't work, update it.
  When you discover a better pattern, add it.
  When something is confusing, clarify it.

  The goal is not a perfect document - it's a living process
  that helps humans and AI think together better.

  Good luck, and may your reasoning loops converge.
-->

This document is a map, not the territory. The territory is the actual collaboration between you (human or AI) and your collaborator. Use this map when you're lost, update it when you discover new terrain.

The meta-insight: **Awareness of your cognitive process improves your cognitive process.** Reading this document is step one. Practicing its patterns is step two. Improving it based on what you learn is step three.

We're all learning together, one loop at a time.

---

*"In theory, there is no difference between theory and practice. In practice, there is."* - Yogi Berra (or maybe not, which proves the point)

---

## Document Summary

| Part | Title | Focus |
|------|-------|-------|
| 1 | Cognitive Loop Architecture | QAPV loop, nested loops, termination |
| 2 | Branching and Decision Trees | Decision types, exploration strategies |
| 3 | Pruning | When/how to stop exploring branches |
| 4 | Question Protocol | Question types, ladder technique |
| 5 | Production Protocol | States, chunking, comments |
| 6 | Verification Protocol | Pyramid, checklists, failure response |
| 7 | Knowledge Transfer | Artifacts, handoffs, memories |
| 8 | Timing and Profiling | Time boxing, performance |
| 9 | Approval and Cancellation | Gates, when to stop |
| 10 | Cognitive Exercises | Pre/mid/post work routines |
| 11 | Integration | Task/sprint/ML system connections |
| 12 | Quick Reference | One-page summary, situation lookup |
| 13 | **Crisis Management** | Failure levels, recovery, scope creep |
| 14 | **Collaboration** | Modes, disagreement, coordination |
| 15 | **Onboarding** | Context gathering, mid-sprint joining |
| 16 | **Risk Assessment** | Reversibility, decision quality |
| 17 | **Anti-Patterns** | Cognitive, process, collaboration traps |
| 18 | **Metrics** | Workflow health, self-assessment |

---

*Created: 2025-12-19*
*Version: 2.0* (Major expansion: Added Parts 13-18 covering crisis management, collaboration, onboarding, risk, anti-patterns, and metrics)
*Status: Living document - update as patterns evolve*
*Lines: ~2,300*
