# Architectural Decision Making for AI Agents

## Purpose

This document teaches AI agents how to recognize, evaluate, and make architectural decisions. Good architectural decisions create sustainable systems; poor ones create technical debt that compounds over time.

## What Makes a Decision "Architectural"?

Not every decision is architectural. Understanding the difference prevents over-engineering trivial choices and under-thinking critical ones.

### Architectural Decisions

Architectural decisions share these characteristics:
- **Hard to reverse**: Changing them later requires significant rework
- **Cross-cutting**: Affect multiple components or modules
- **Long-lived**: Will outlast the current implementation
- **Foundational**: Other decisions depend on them

**Examples in this codebase:**
- Using a hypergraph structure instead of a regular graph
- WAL-first commit strategy for transactions
- Dependency injection via Container
- JSON over pickle for persistence

### Implementation Details

Implementation details are:
- **Easy to change**: Localized to one file or function
- **Isolated**: Don't affect other components
- **Short-lived**: May change with the next feature
- **Leaf decisions**: Nothing else depends on them

**Examples:**
- Variable naming within a function
- Which loop construct to use
- Local caching strategy within a method
- String formatting style

### The Gray Zone

Some decisions seem like details but are actually architectural:

| Decision | Seems Like | Actually |
|----------|------------|----------|
| Error message format | Detail | Architectural (users depend on parsing them) |
| Log level strategy | Detail | Architectural (affects debugging across system) |
| ID format (T-YYYYMMDD-...) | Detail | Architectural (stored in databases, parsed by tools) |
| Test fixture location | Detail | Architectural (affects all test files) |

**Rule of thumb**: If changing it would require updating multiple files or components, it's architectural.

## When to Decide vs When to Defer

Making decisions too early wastes effort. Making them too late constrains options. This framework helps find the right moment.

### Decide Now When:

1. **Blocking progress**: You cannot proceed without the decision
2. **Information is sufficient**: You have enough context to decide well
3. **Cost increases with time**: Deferring makes implementation harder
4. **Team is aligned**: Stakeholders are available and engaged

### Defer When:

1. **Information is missing**: Key requirements are unclear
2. **Options are equivalent**: No clear winner; pick later when it matters
3. **Reversibility is high**: Easy to change if you guess wrong
4. **Learning is ongoing**: You'll know more after building something

### The Last Responsible Moment

The "last responsible moment" is when deferring any longer would:
- Eliminate options you want to keep
- Increase implementation cost significantly
- Block other important work

**Example**: Choosing between SQLite and a custom file format for GoT storage.
- Too early: Before understanding query patterns
- Too late: After building 50 features on one approach
- Right moment: After prototyping 2-3 core features, before building the rest

## Evaluating Tradeoffs

Every architectural decision involves tradeoffs. Make them explicit.

### The Tradeoff Triangle

Most decisions balance three concerns:

```
        Simplicity
           /\
          /  \
         /    \
        /      \
       /________\
Performance    Flexibility
```

**Simplicity**: Fewer concepts, easier to understand
**Performance**: Speed, memory, throughput
**Flexibility**: Ability to change, extend, adapt

You rarely get all three. Know which one you're sacrificing.

### Evaluation Framework

For each option, assess:

| Dimension | Questions to Ask |
|-----------|-----------------|
| **Complexity** | How many concepts does this introduce? How steep is the learning curve? |
| **Performance** | What are the time/space bounds? Where are the hot paths? |
| **Maintainability** | Can someone unfamiliar fix bugs? Is it testable? |
| **Extensibility** | Can we add features without rewriting? What's locked in? |
| **Debuggability** | Can we observe its behavior? Can we trace failures? |
| **Team fit** | Does the team have expertise? Is it idiomatic? |

### Scoring Example

When deciding between options, make the tradeoffs explicit:

```markdown
## Decision: Storage Backend for GoT

### Option A: JSON Files
- Complexity: Low (1)
- Performance: Medium (2) - O(n) scans
- Maintainability: High (3) - human readable
- Extensibility: High (3) - add fields easily
- Debuggability: High (3) - cat/grep work
- Team fit: High (3) - everyone knows JSON
- **Total: 15**

### Option B: SQLite
- Complexity: Medium (2) - need SQL knowledge
- Performance: High (3) - indexed queries
- Maintainability: Medium (2) - need DB tools
- Extensibility: Medium (2) - schema migrations
- Debuggability: Medium (2) - need sqlite3 CLI
- Team fit: Medium (2) - some SQL experience
- **Total: 13**

### Decision: JSON Files
Rationale: Higher total score, better debuggability for a graph-of-thought
system where humans need to inspect state.
```

## Reversible vs Irreversible Decisions

Jeff Bezos calls these "two-way doors" and "one-way doors." This distinction should change how much time you invest.

### Two-Way Doors (Reversible)

You can walk back through if wrong. Invest less time deciding.

**Characteristics:**
- Changes are localized
- No external dependencies on the decision
- Migration path exists
- Costs of reversal are bounded

**Examples:**
- Internal function signatures
- Test organization
- Local caching strategies
- Log message formats (if not parsed externally)

**Approach**: Make a reasonable choice and move on. You can always change it.

### One-Way Doors (Irreversible)

Walking back is expensive or impossible. Invest more time deciding.

**Characteristics:**
- External systems depend on it
- Data is persisted in this format
- APIs are published
- Migration would be prohibitively expensive

**Examples:**
- Public API contracts
- Database schema (especially ID formats)
- Wire protocols
- Security architecture

**Approach**: Gather more information. Prototype. Review with others. Document reasoning.

### Making Irreversible Decisions More Reversible

Design techniques to reduce lock-in:

| Technique | How It Helps |
|-----------|--------------|
| **Abstraction layers** | Hide implementation behind interfaces |
| **Feature flags** | Run old and new in parallel |
| **Versioned APIs** | Maintain compatibility while evolving |
| **Migration scripts** | Automate the transition |
| **Incremental rollout** | Test changes on subset before full deployment |

**Example from this codebase**: The `StorageBackend` abstraction allows swapping between file-based and memory-based storage without changing business logic.

## Documenting Decisions

Undocumented decisions get unmade. Future agents (including future you) will wonder "why is it like this?" and might change it poorly.

### Architecture Decision Records (ADRs)

ADRs are short documents capturing architectural decisions.

**Format used in this codebase** (see `samples/decisions/adr-microseconds-task-id.md`):

```markdown
# ADR-XXX: Title

**Status:** Proposed | Accepted | Deprecated | Superseded
**Date:** YYYY-MM-DD
**Deciders:** Who made this decision
**Tags:** relevant, tags, here

---

## Context and Problem Statement
What is the issue? What forces are at play?

## Decision Drivers
What criteria matter most?

## Considered Options
### Option 1: Name
Description, pros, cons

### Option 2: Name
Description, pros, cons

## Decision Outcome
Which option was chosen and why.

## Consequences
### Positive
- Good outcomes

### Negative
- Costs and tradeoffs

### Neutral
- Things that don't change

## Related Decisions
Links to related ADRs
```

### Decision Logs in GoT

For operational decisions, use the GoT decision log:

```bash
# Log a decision
python -m cortical.got decision log "Use JSON over pickle for persistence" \
    --rationale "Security (no code execution), Git-friendly (human-readable diffs), Cross-platform" \
    --affects "storage" "serialization" \
    --alternatives "pickle" "msgpack" "protobuf"

# Link to related task
python -m cortical.got edge add D-XXXX T-YYYY "INFORMED_BY"
```

### When to Use Which

| Situation | Documentation Method |
|-----------|---------------------|
| Major architectural change | Full ADR in `samples/decisions/` |
| Implementation choice during task | GoT decision log |
| Quick tradeoff during coding | Code comment with rationale |
| Rejected approach | Document why not (prevents re-investigation) |

## Learning from Past Decisions

Past decisions are a learning resource. Mine them.

### Querying Decision History

```bash
# List all decisions
python -m cortical.got decision list

# Find decisions about a topic
python -m cortical.got query "type = 'decision' AND content CONTAINS 'storage'"

# See what a decision affected
python -m cortical.got edge list D-XXXX
```

### Decision Retrospectives

When a decision turns out poorly, document why:

1. **What did we decide?** (the original choice)
2. **What happened?** (the negative outcome)
3. **Why did we get it wrong?**
   - Missing information at decision time?
   - Changed requirements?
   - Implementation diverged from intent?
4. **What would we do differently?**
5. **How do we prevent this class of mistake?**

### Patterns from This Codebase

Decisions that worked well:
- **WAL-first commits**: Prevented data corruption
- **Hypergraph structure**: Enabled meta-reasoning about relationships
- **DI via Container**: Made testing dramatically easier

Decisions that needed revision:
- **Second-precision task IDs**: Caused collisions (fixed with microseconds)
- **Pickle persistence**: Security risk (migrated to JSON)
- **Direct file operations in tests**: Flaky (fixed with DI)

## The Sovereignty Principle

This codebase follows the sovereignty principle: **build what you can control**.

### When to Build

Build when:
- The component is core to your system's value
- You need to understand and modify its internals
- External dependencies would create risk
- The implementation is within your capability

### When to Depend

Depend on external code only when:
- It's truly commodity (Python stdlib)
- It's meta-tooling, not runtime (pytest)
- Building it would be prohibitively expensive
- You can isolate the dependency

### Decision Framework

```
                    Core to Value?
                         |
              +---------+---------+
              |                   |
             YES                  NO
              |                   |
        Build It           Can We Isolate?
                                  |
                        +---------+---------+
                        |                   |
                       YES                  NO
                        |                   |
                    Consider             Build It
                    External             (safer than
                    (with               tight coupling)
                    abstraction)
```

### Example Application

**Question**: Should we use an external graph database?

**Analysis**:
- Core to value? YES - the graph is our cognitive model
- Within capability? YES - we have graph expertise
- Risk of dependency? HIGH - would be locked into their model
- Decision: BUILD IT

**Question**: Should we use pytest?

**Analysis**:
- Core to value? NO - testing is meta-tooling
- Within capability? YES, but why?
- Risk of dependency? LOW - doesn't affect runtime
- Decision: DEPEND (it's commodity tooling)

## Practical Decision Checklist

Before making an architectural decision:

```
[ ] Is this actually architectural? (Or just an implementation detail?)
[ ] Do I have enough information to decide now?
[ ] Have I identified at least two alternatives?
[ ] Have I made the tradeoffs explicit?
[ ] Is this reversible? If not, have I invested proportional effort?
[ ] Have I documented the decision and rationale?
[ ] Does this align with the sovereignty principle?
[ ] Have I checked for similar past decisions?
```

## Common Anti-Patterns

| Anti-Pattern | Why It's Bad | Better Approach |
|--------------|--------------|-----------------|
| **Decision by default** | You inherit someone else's tradeoffs | Make explicit choices |
| **Analysis paralysis** | Perfect is the enemy of good | Time-box, then decide |
| **Undocumented decisions** | Knowledge walks out the door | Write ADRs |
| **Premature abstraction** | Complexity without benefit | Wait for second use case |
| **Resume-driven development** | Using tech for career, not problem | Solve the actual problem |
| **Ignoring past decisions** | Repeating mistakes | Query decision history |

## Key Insight

Good architectural decisions come from:
1. **Recognizing** what's truly architectural
2. **Timing** the decision appropriately
3. **Evaluating** tradeoffs explicitly
4. **Documenting** for future agents
5. **Learning** from past decisions

The goal is not perfect decisions - it's decisions that can be explained, defended, and revised when new information arrives.

---

*Architecture is the decisions that are hard to change. Make them deliberately.*
