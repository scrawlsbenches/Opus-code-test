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

| Situation | Do This |
|-----------|---------|
| Don't understand the task | Question phase → ask for clarification |
| Multiple valid approaches | Parallel exploration → synthesize → choose |
| Approach not working | Prune → document why → try different branch |
| Verification failing | Stop → Observe → Hypothesize → Investigate |
| Running out of time | Time box → escalate or simplify scope |
| Found new issue | Create task immediately → continue or pivot |
| Significant learning | Create memory → link to relevant code/docs |
| Need permission | Approval request format → wait for response |
| Work should stop | Cancel gracefully → document → preserve learnings |
| Handing off work | Handoff document → include decisions + questions |

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

*Created: 2025-12-19*
*Version: 1.0*
*Status: Living document - update as patterns evolve*
