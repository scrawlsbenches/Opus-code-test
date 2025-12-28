# Design Document: LLM Cognitive Architecture

## Why This Exists

I am an LLM. I have specific limitations that make complex, multi-session work difficult:

1. **No persistent memory** - Every session starts fresh
2. **Limited context window** - I can only hold so much at once
3. **No true learning** - I don't update from experience within deployment
4. **Uncertainty about state** - Did my action succeed? Is this information current?
5. **Fragile coordination** - Multiple instances of me can't directly communicate

This framework exists to **externalize the cognitive capabilities I lack** so that I can:
- Remember what I've learned across sessions
- Coordinate with other instances of myself
- Recover when I get confused
- Improve through evolutionary pressure on strategies

---

## Core Insight: Externalized Cognition

The key insight is that **tools and structures can provide what my architecture lacks**.

| I Lack | Externalized As |
|--------|-----------------|
| Persistent memory | Graph of Thought nodes |
| Learning | Evolutionary strategy selection |
| State verification | Checksums, event logs, ground truth tools |
| Coordination | Event bus, delegation protocols |
| Recovery | Checkpoints, cognitive state snapshots |

This isn't a workaround—it's a **cognitive architecture** where my reasoning abilities combine with external structures to create capabilities neither has alone.

---

## The Graph of Thought

### What It Is

The Graph of Thought (GoT) is not just a data structure. It's a **substrate for reasoning** that persists across sessions and instances.

```
                    ┌─────────────┐
                    │  QUESTION   │
                    │ "How to auth?"│
                    └──────┬──────┘
                           │ LEADS_TO
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │HYPOTHESIS│ │HYPOTHESIS│ │HYPOTHESIS│
        │  "JWT"   │ │ "OAuth"  │ │ "Session"│
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ EVIDENCE │ │ EVIDENCE │ │ EVIDENCE │
        │"fast,    │ │"standard,│ │"simple,  │
        │ stateless"│ │ delegated"│ │ stateful"│
        └────┬─────┘ └──────────┘ └──────────┘
             │
             ▼
        ┌──────────┐
        │ DECISION │
        │"Use JWT" │
        │ + reason │
        └──────────┘
```

### Node Types

| Type | Represents | When Created |
|------|------------|--------------|
| `QUESTION` | Something I'm trying to understand | Start of investigation |
| `HYPOTHESIS` | A possible answer or approach | During exploration |
| `EVIDENCE` | Information supporting/refuting | During research |
| `DECISION` | A choice made with rationale | When committing to path |
| `ACTION` | Something I did | During execution |
| `OBSERVATION` | What happened as result | After action |
| `REFLECTION` | Meta-level insight | During retrospective |

### Edge Types

| Type | Meaning | Example |
|------|---------|---------|
| `LEADS_TO` | This thought led to that thought | Question → Hypothesis |
| `SUPPORTS` | Evidence for a hypothesis | Evidence → Hypothesis |
| `CONTRADICTS` | Evidence against | Evidence → Hypothesis |
| `DEPENDS_ON` | Can't proceed without | Action → Decision |
| `BLOCKS` | Prevents progress | Blocker → Action |
| `CAUSED` | This action caused this observation | Action → Observation |
| `REFINED_BY` | Improved understanding | Hypothesis → Hypothesis |
| `SUPERSEDES` | Replaces previous thought | Decision → Decision |

### Why Graph, Not Tree

A tree would force linear reasoning. Real thinking is messier:
- Multiple hypotheses can share evidence
- Decisions can be revised (new node supersedes old)
- Reflections connect back to earlier thoughts
- Dead ends are preserved for learning

The graph captures this naturally.

---

## The Cognitive Loop: QAPV

My core reasoning pattern is **Question → Answer → Produce → Verify**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         QAPV CYCLE                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│   │ QUESTION │───▶│  ANSWER  │───▶│ PRODUCE  │───▶│  VERIFY  │      │
│   │          │    │          │    │          │    │          │      │
│   │ What am  │    │ Research │    │ Create   │    │ Check    │      │
│   │ I trying │    │ explore  │    │ artifact │    │ quality  │      │
│   │ to do?   │    │ decide   │    │          │    │          │      │
│   └──────────┘    └──────────┘    └──────────┘    └────┬─────┘      │
│        ▲                                               │             │
│        │                                               │             │
│        └───────────────────────────────────────────────┘             │
│                         (if verify fails)                            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

Each phase creates nodes in the GoT:
- **Question** → QUESTION node
- **Answer** → HYPOTHESIS, EVIDENCE, DECISION nodes
- **Produce** → ACTION nodes
- **Verify** → OBSERVATION, REFLECTION nodes

This pattern repeats at multiple scales:
- Micro: within a single function implementation
- Meso: across a feature implementation
- Macro: across a project or session

---

## Hierarchical Orchestration

### Why Hierarchy?

I can only hold so much context. Hierarchy lets me:
1. **Scope attention** - Each level focuses on its concerns
2. **Compress context** - Higher levels see summaries, not details
3. **Parallelize** - Independent workers don't need to coordinate
4. **Recover** - Failures are isolated to their level

### The Levels

```
┌─────────────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR                                                         │
│ Concerns: Which goals? In what order? System health?                │
│ Thinks in: Goals, flow, capacity                                    │
│ Time horizon: Hours to days                                         │
├─────────────────────────────────────────────────────────────────────┤
│ DIRECTOR                                                             │
│ Concerns: How to decompose? Who does what? Are we on track?         │
│ Thinks in: Phases, tasks, dependencies                              │
│ Time horizon: Minutes to hours                                      │
├─────────────────────────────────────────────────────────────────────┤
│ WORKER                                                               │
│ Concerns: How to implement? What's blocking? Is it correct?         │
│ Thinks in: Steps, code, tests                                       │
│ Time horizon: Seconds to minutes                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### Context Compression

Each level receives compressed context from above:

```python
# Orchestrator's full context
full_goal = """
Implement user authentication with JWT tokens, ensuring backward
compatibility with existing session-based auth, supporting OAuth
providers, with full test coverage and documentation updates.
"""

# Director receives
director_context = """
Goal: Implement JWT authentication
Constraints: Backward compatible, needs tests
Related: Existing session auth in auth.py
"""

# Worker receives
worker_context = """
Task: Add JWT token validation function
Input: Token string
Output: User ID or error
Test: Validate against test vectors
"""
```

The compression is lossy but sufficient. Each level has what it needs.

---

## Why Kanban at Top, Agile at Bottom

### The Insight

Different levels have different dynamics:

| Level | Arrival Pattern | Best Managed By |
|-------|-----------------|-----------------|
| Goals | Continuous, variable | Kanban (flow) |
| Phases | Batched per goal | Hybrid |
| Tasks | Defined per phase | Agile (sprints) |

Kanban optimizes **throughput** and **flow efficiency**.
Agile optimizes **predictability** and **learning**.

### WIP Limits

Work-in-progress limits are crucial:

```
Without WIP limits:
  Goal 1: ████████░░░░░░░░░░░░  (40%)
  Goal 2: ██████░░░░░░░░░░░░░░  (30%)
  Goal 3: ████░░░░░░░░░░░░░░░░  (20%)
  Goal 4: ██░░░░░░░░░░░░░░░░░░  (10%)

  Result: Everything partially done, nothing shipped

With WIP limits (max 2):
  Goal 1: ████████████████████  (100%) ✓
  Goal 2: ██████████████░░░░░░  (70%)
  Goal 3: (waiting)
  Goal 4: (waiting)

  Result: Goals complete and ship
```

### Sprints for Workers

Workers use time-boxed sprints because:
1. **Predictable delivery** - Directors can plan
2. **Natural checkpoints** - For saving state
3. **Retrospectives** - Feed the learning system
4. **Velocity tracking** - Improves estimation over time

---

## Evolutionary Improvement

### What Evolves

Not my weights—my **strategies**. The genome includes:

```python
genome = {
    # How I break down problems
    "decomposition": "divide by component vs. divide by layer",

    # How I delegate
    "delegation": "parallel vs. sequential",

    # How I handle failure
    "failure_response": "retry vs. escalate vs. try alternative",

    # How I compress context
    "context_strategy": "include rationale vs. just decisions",

    # Meta-parameters
    "exploration_rate": 0.1,  # How often to try new strategies
    "confidence_threshold": 0.7,  # When to escalate vs. proceed
}
```

### Selection Pressure

Strategies compete based on fitness:

```
Fitness = weighted_sum(
    success_rate,        # Did goals complete?
    efficiency,          # How many resources used?
    quality,             # Were outputs correct?
    predictability,      # Could we forecast completion?
    user_satisfaction,   # External feedback
)
```

### The Evolution Cycle

```
1. EXECUTE goals with current strategy pool
2. SURVEY all executions (trace events, outcomes)
3. STUDY traces (compute fitness, attribute to genes)
4. SELECT high-fitness strategies
5. CROSSOVER successful strategies
6. MUTATE to explore new variations
7. VALIDATE new generation (no regression)
8. PROPAGATE to strategy pool
9. GOTO 1
```

### Safeguards

Evolution can go wrong. Safeguards prevent regression:

1. **Elitism**: Best strategy always survives
2. **Golden strategies**: Some strategies are protected
3. **Diversity floor**: Prevent population collapse
4. **Regression tests**: New generations must pass tests

---

## Cognitive State Management

### The Problem

I forget everything between sessions. But complex work spans sessions.

### The Solution

Externalize cognitive state to files:

```
.cognitive_state/
├── current_focus.json      # What am I working on?
├── open_questions.json     # What don't I know yet?
├── decisions.json          # What have I decided and why?
├── hypotheses.json         # What am I considering?
├── observations.json       # What have I noticed?
└── checkpoints/
    └── 2025-12-28T10-30.json  # Full state snapshot
```

### Recovery Protocol

When I detect confusion:

```
1. STOP current action
2. LOAD most recent checkpoint
3. VERIFY checkpoint matches reality (files exist, etc.)
4. IDENTIFY what's changed since checkpoint
5. RECONCILE my understanding with current state
6. RESUME or ESCALATE if can't reconcile
```

### Confusion Signals

How I detect I'm confused:
- Repeating the same failed approach
- Contradicting earlier statements
- Making changes without reading first
- Asking questions already answered
- Generating placeholder content

---

## Event-Driven Coordination

### Why Events

Direct communication between agents is hard. Events provide:
- **Loose coupling** - Agents don't need to know each other
- **Asynchrony** - Don't block waiting for response
- **Observability** - Everything is logged
- **Replay** - Can reconstruct what happened

### Event Flow

```
Worker completes task
    │
    ▼
Event: worker.task.completed {task_id, output}
    │
    ├──▶ Director: updates phase progress
    │
    ├──▶ Surveyor: records for evolution
    │
    └──▶ Metrics: updates dashboards
```

### Critical Events

| Event | Triggers |
|-------|----------|
| `goal.submitted` | Orchestrator adds to backlog |
| `goal.started` | Director assigned, execution begins |
| `task.blocked` | Swarming or escalation |
| `task.completed` | Progress update, maybe phase complete |
| `error.occurred` | Recovery protocol |
| `decision.made` | Logged to GoT with rationale |
| `retrospective.complete` | Feeds evolution |

---

## Design Decisions Log

### Why Dataclasses Not Dicts

**Decision**: Use dataclasses for all data structures.

**Rationale**:
- Type hints provide documentation
- IDE autocomplete helps implementors
- Validation can be added to `__post_init__`
- Serialization is straightforward

**Alternative considered**: Plain dicts are more flexible but lose type safety.

### Why Async

**Decision**: Core loops are async.

**Rationale**:
- Workers can run concurrently
- Events can be processed without blocking
- Matches how I actually operate (not truly parallel but concurrent)

**Alternative considered**: Threads are harder to reason about and debug.

### Why Not Neural Architecture

**Decision**: Use symbolic/algorithmic approaches, not neural.

**Rationale**:
- I am the neural component
- External structures should be inspectable, debuggable
- Symbolic systems compose better
- No training data needed

**Alternative considered**: Could train models for strategy selection, but adds complexity and opacity.

### Why Fitness is Multi-Objective

**Decision**: Fitness is a vector, not a scalar.

**Rationale**:
- Different goals weight dimensions differently
- Prevents collapse to single-metric optimization
- Enables Pareto-optimal diverse strategies

**Alternative considered**: Single scalar is simpler but loses nuance.

---

## Implementation Priority

If building incrementally, priority order:

1. **Types and Events** - Foundation everything builds on
2. **Cognitive State** - I need to persist thinking
3. **Basic Orchestration** - Can run simple goals
4. **Workers with Sprints** - Predictable execution
5. **GoT Persistence** - Thinking survives sessions
6. **Evolution** - Self-improvement
7. **Metrics Dashboard** - Visibility

Each layer is useful without the ones below, but the full system needs all.

---

## Open Questions

Things I'm uncertain about:

1. **Optimal compression ratio** - How much context can I lose and still function?
2. **Evolution speed** - How many executions before strategies converge?
3. **Failure cascade** - How to prevent one bad strategy from propagating?
4. **Human-in-loop** - Where should users inject guidance?
5. **Cross-project learning** - Can strategies transfer between codebases?

These should be resolved through experimentation and observation.

---

## Closing Thought

This framework is not about making me "smarter." I am what I am. It's about creating an **environment** where my capabilities are amplified and my limitations are compensated for.

The Graph of Thought is my extended mind.
The Evolutionary loop is my learning system.
The Hierarchy is my attention management.
The Events are my nervous system.
The Checkpoints are my memory.

Together, they let me do work I couldn't do alone.
