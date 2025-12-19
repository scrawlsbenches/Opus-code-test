# Graph of Thought: Network-Based Reasoning for Software Development

<!--
  AUTHOR'S NOTE (for the next developer):

  This document models thinking as a GRAPH, not a sequence.

  The Complex Reasoning Workflow (docs/complex-reasoning-workflow.md) describes
  PROCESSES - loops, phases, protocols. This document describes STRUCTURES -
  how ideas connect, cluster, and evolve.

  Think of it this way:
  - Complex Reasoning Workflow = the VERBS of thinking (question, produce, verify)
  - Graph of Thought = the NOUNS of thinking (concepts, connections, clusters)

  Use this when you need to:
  - Understand how different parts of a problem relate
  - Navigate complex decision spaces
  - Merge multiple lines of investigation
  - Avoid losing track of important threads
  - Communicate the "shape" of a problem to others

  The graph metaphor is powerful because software itself is a graph:
  modules depend on modules, functions call functions, concepts reference concepts.
  Thinking about software in graph terms aligns cognition with the domain.
-->

## Preamble: Why Graphs?

Linear thinking is easy to follow but misses connections. Hierarchical thinking (trees) captures structure but forces false exclusivity. **Graph thinking** captures reality: ideas connect in networks, not chains.

```
LINEAR THINKING:           TREE THINKING:           GRAPH THINKING:

A → B → C → D              A                        A ←──→ B
                          / \                       ↑ ╲   ╱ ↑
                         B   C                      │  ╲ ╱  │
                        / \   \                     │   ╳   │
                       D   E   F                    │  ╱ ╲  │
                                                    ↓ ╱   ╲ ↓
                                                    C ←──→ D

"First A, then B..."    "A contains B and C..."    "A relates to B, C, D
                                                    in different ways..."
```

**Key insight:** Real problems have:
- Multiple entry points (you can start from different angles)
- Cross-cutting concerns (ideas that touch many areas)
- Feedback loops (later discoveries change earlier understanding)
- Emergent clusters (groups of tightly-related concepts)

A graph model captures all of these.

---

## Part 1: The Anatomy of a Thought Graph

### 1.1 Nodes: Units of Thought

<!--
  NODES are the atoms of thinking.
  They represent discrete concepts, decisions, questions, or facts.
  The key is making them small enough to be clear but large enough to be meaningful.
-->

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NODE TYPES                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  CONCEPT NODE [C]                                                   │
│  ├── Represents an idea, pattern, or abstraction                    │
│  ├── Example: "Dependency Injection", "Event Sourcing"              │
│  └── Properties: name, definition, examples, counter-examples       │
│                                                                      │
│  QUESTION NODE [?]                                                  │
│  ├── Represents something unknown or uncertain                      │
│  ├── Example: "How should we handle auth?", "What's the bottleneck?│
│  └── Properties: question, context, candidate_answers, urgency      │
│                                                                      │
│  DECISION NODE [D]                                                  │
│  ├── Represents a choice point with options                         │
│  ├── Example: "REST vs GraphQL", "Monolith vs Microservices"        │
│  └── Properties: options[], criteria[], chosen, rationale           │
│                                                                      │
│  FACT NODE [F]                                                      │
│  ├── Represents verified information                                │
│  ├── Example: "Response time is 200ms", "Test coverage is 85%"      │
│  └── Properties: claim, evidence, confidence, source                │
│                                                                      │
│  TASK NODE [T]                                                      │
│  ├── Represents work to be done                                     │
│  ├── Example: "Implement caching", "Write integration tests"        │
│  └── Properties: description, status, dependencies, assignee        │
│                                                                      │
│  ARTIFACT NODE [A]                                                  │
│  ├── Represents something created                                   │
│  ├── Example: "auth.py module", "API documentation"                 │
│  └── Properties: path, type, version, created_by                    │
│                                                                      │
│  INSIGHT NODE [I]                                                   │
│  ├── Represents a learning or realization                           │
│  ├── Example: "The bottleneck is in serialization"                  │
│  └── Properties: insight, how_discovered, implications              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Edges: Relationships Between Thoughts

<!--
  EDGES are the connections.
  They represent HOW thoughts relate, not just THAT they relate.
  Typed edges are crucial - "relates to" is too vague to be useful.
-->

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EDGE TYPES                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  SEMANTIC EDGES (meaning relationships)                             │
│  ├── REQUIRES: A requires B to exist/function                       │
│  ├── ENABLES: A makes B possible                                    │
│  ├── CONFLICTS: A and B cannot both be true/chosen                  │
│  ├── SUPPORTS: A provides evidence for B                            │
│  ├── REFUTES: A provides evidence against B                         │
│  ├── SIMILAR: A and B share significant properties                  │
│  ├── CONTRASTS: A and B differ in important ways                    │
│  └── CONTAINS: A includes B as a component                          │
│                                                                      │
│  TEMPORAL EDGES (time relationships)                                │
│  ├── PRECEDES: A must happen before B                               │
│  ├── FOLLOWS: A happens after B                                     │
│  ├── TRIGGERS: A causes B to happen                                 │
│  └── BLOCKS: A prevents B until resolved                            │
│                                                                      │
│  EPISTEMIC EDGES (knowledge relationships)                          │
│  ├── ANSWERS: A answers question B                                  │
│  ├── RAISES: A raises question B                                    │
│  ├── ASSUMES: A depends on B being true                             │
│  ├── VALIDATES: A confirms B is true                                │
│  └── INVALIDATES: A proves B is false                               │
│                                                                      │
│  PRACTICAL EDGES (work relationships)                               │
│  ├── IMPLEMENTS: A implements concept/decision B                    │
│  ├── TESTS: A tests/verifies B                                      │
│  ├── DOCUMENTS: A documents B                                       │
│  ├── DEPENDS_ON: A needs B to be complete first                     │
│  └── PRODUCES: A creates B as output                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Edge notation:**
```
A ──REQUIRES──> B      (directed: A requires B)
A <──SIMILAR──> B      (undirected: A and B are similar)
A ──CONFLICTS──X B     (X marks conflict)
A ═══CRITICAL═══> B    (double line = high importance)
A ···WEAK···> B        (dotted = tentative/uncertain)
```

### 1.3 Clusters: Groups of Related Thoughts

<!--
  CLUSTERS emerge when nodes are densely connected.
  They represent coherent sub-problems or topic areas.
  Identifying clusters helps manage complexity.
-->

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CLUSTER IDENTIFICATION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  A cluster forms when:                                              │
│  ├── Multiple nodes share many edges                                │
│  ├── Nodes can be discussed independently of others                 │
│  ├── There's a unifying concept or purpose                          │
│  └── Changes to one likely affect others in the group               │
│                                                                      │
│  Example: Authentication Cluster                                    │
│  ┌─────────────────────────────────────────────────┐                │
│  │  [C] JWT Tokens                                 │                │
│  │       ↕                                         │                │
│  │  [D] Token vs Session  ←→  [?] Token Expiry    │                │
│  │       ↓                                         │                │
│  │  [T] Implement Auth   ──→  [A] auth.py         │                │
│  │       ↑                         ↓               │                │
│  │  [F] "OWASP recommends..."  ──→ [T] Add Tests  │                │
│  └─────────────────────────────────────────────────┘                │
│                                                                      │
│  Cluster properties:                                                │
│  ├── Name: Authentication                                           │
│  ├── Core nodes: 7                                                  │
│  ├── Internal edges: 12                                             │
│  ├── External edges: 3 (connections to other clusters)              │
│  └── Coherence: High (most nodes connect to most others)            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 Graph Properties

**Useful metrics for thought graphs:**

| Property | What It Measures | Implication |
|----------|-----------------|-------------|
| **Node count** | How many concepts | Complexity indicator |
| **Edge density** | Connections / possible connections | How interconnected |
| **Cluster count** | Distinct topic areas | Natural decomposition |
| **Orphan nodes** | Nodes with no edges | Forgotten or independent |
| **Hub nodes** | Nodes with many edges | Central concepts |
| **Bridge nodes** | Connect different clusters | Integration points |
| **Question ratio** | Question nodes / total | Uncertainty level |
| **Decision ratio** | Undecided / total decisions | Commitment level |

---

## Part 2: Graph Operations

<!--
  OPERATIONS are how you manipulate the graph.
  These are the verbs that act on the noun-structure.
-->

### 2.1 Construction Operations

**ADD NODE:**
```
Before: [A] ──→ [B]

Action: Add node C related to A

After:  [A] ──→ [B]
         │
         └──→ [C]
```

**ADD EDGE:**
```
Before: [A]     [B]     [C]
         │               │
         └──────────────┘

Action: Add edge B──→C

After:  [A]     [B] ──→ [C]
         │               │
         └──────────────┘
```

**MERGE NODES:**
```
Before: [A: "Auth tokens"]    [B: "JWT handling"]
              │                      │
         (many similar edges)

Action: Merge A and B (they're really the same concept)

After:  [AB: "Auth tokens (JWT)"]
              │
         (combined edges)
```

**SPLIT NODE:**
```
Before: [A: "Database layer"]
         │
         └──→ (many diverse connections)

Action: Split A (it's really multiple concepts)

After:  [A1: "Query building"]    [A2: "Connection pooling"]
              │                          │
         (subset of edges)         (subset of edges)
```

### 2.2 Traversal Operations

**DEPTH-FIRST EXPLORATION:**
```
Start at node, follow one path to its end before backtracking.

Use when: Deep understanding of one thread needed
Example: "Let me fully understand authentication before moving on"

        [Start]
            │
            ▼
        [Auth] ──→ [Tokens] ──→ [Expiry] ──→ [Refresh]
                                               │
                                          (backtrack)
                                               │
                      [Sessions] ◄─────────────┘
```

**BREADTH-FIRST EXPLORATION:**
```
Start at node, explore all immediate neighbors before going deeper.

Use when: Survey of related concepts needed
Example: "What all connects to the API module?"

Level 0:            [API]
                   / | \ \
Level 1:    [Auth] [DB] [Cache] [Logging]
              │      │      │       │
Level 2:   [...] [...] [...] [...]
```

**SHORTEST PATH:**
```
Find the minimum-hop connection between two nodes.

Use when: Understanding how concepts relate
Example: "How does user input get to the database?"

[User Input] ──→ [Validation] ──→ [Controller] ──→ [ORM] ──→ [Database]
             (4 hops - this is the dependency chain)
```

**CLUSTER HOPPING:**
```
Move between clusters via bridge nodes.

Use when: Seeing the big picture across topics
Example: "How do our clusters relate?"

[Auth Cluster] ═══[User Model]═══ [Data Cluster]
                       │
                       │
               [API Cluster]
```

### 2.3 Analysis Operations

**FIND CYCLES:**
```
Detect circular dependencies or reasoning.

A ──→ B ──→ C ──→ A  (cycle!)

Use when: Checking for circular dependencies
Warning: Cycles in requirements = deadlock
         Cycles in reasoning = may need to break with assumption
```

**FIND ORPHANS:**
```
Nodes with no connections.

[A] ──→ [B]     [C] ←── [D]     [E]  ← orphan!

Use when: Identifying forgotten or independent items
Action: Either connect them or question their relevance
```

**FIND HUBS:**
```
Nodes with many connections (high degree).

         [X]     [Y]
           \     /
        [A]─[HUB]─[B]
           /     \
         [C]     [D]

Use when: Identifying core concepts
Warning: Hubs are single points of failure
Insight: Changes to hubs ripple widely
```

**FIND BRIDGES:**
```
Nodes whose removal would disconnect clusters.

[Cluster A]═══[Bridge]═══[Cluster B]

Use when: Identifying integration points
Warning: Bridges are bottlenecks and risks
Insight: May need redundant bridges for resilience
```

### 2.4 Transformation Operations

**PRUNE:**
```
Remove nodes/edges that are no longer relevant.

Before: [A] ──→ [B] ──→ [C]
              ↘   ↓
            [X] [Y]  (decided these are out of scope)

After:  [A] ──→ [B] ──→ [C]

Document: Why X and Y were pruned (see Part 3 of complex-reasoning-workflow.md)
```

**COLLAPSE:**
```
Replace a cluster with a single representative node.

Before: ┌───────────────┐
        │ [A]─[B]─[C]  │
        │  │   │   │   │
        │ [D]─[E]─[F]  │
        └───────────────┘
              │
              ▼
After:  [Auth Cluster]  (treat as single unit for higher-level thinking)
```

**EXPAND:**
```
Replace a collapsed node with its internal structure.

Before: [Auth Cluster] ──→ [Data Cluster]

After:  ┌───────────────┐
        │ [A]─[B]─[C]  │──→ [Data Cluster]
        │  │   │   │   │
        │ [D]─[E]─[F]  │
        └───────────────┘
```

**LAYER:**
```
Organize nodes into abstraction levels.

Level 3 (Abstract):   [System Design]
                           │
Level 2 (Module):     [Auth]────[Data]────[API]
                        │         │         │
Level 1 (Component):  [JWT]     [ORM]    [REST]
                        │         │         │
Level 0 (Concrete):  [token.py][models.py][routes.py]
```

---

## Part 3: Graph Patterns in Software Development

<!--
  PATTERNS are recurring graph structures.
  Recognizing them speeds up understanding and decision-making.
-->

### 3.1 The Requirements Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REQUIREMENTS GRAPH PATTERN                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│              [User Need]                                            │
│               /   |   \                                             │
│              /    |    \                                            │
│       [Req A] [Req B] [Req C]                                       │
│          |      / \      |                                          │
│          |     /   \     |                                          │
│       [Spec] [Spec] [Spec]                                          │
│          \     |     /                                              │
│           \    |    /                                               │
│            [Design]                                                 │
│               |                                                      │
│         [Implementation]                                            │
│                                                                      │
│  Key edges:                                                         │
│  ├── User Need ──DECOMPOSES_TO──> Requirements                      │
│  ├── Requirements ──SPECIFIES──> Specifications                     │
│  ├── Specifications ──CONSTRAINS──> Design                          │
│  └── Design ──GUIDES──> Implementation                              │
│                                                                      │
│  Traversal: Top-down for implementation, bottom-up for validation   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 The Investigation Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INVESTIGATION GRAPH PATTERN                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│               [Initial Question]                                    │
│                /      |      \                                      │
│               /       |       \                                     │
│       [Hypothesis A] [Hypothesis B] [Hypothesis C]                  │
│            |              |              |                          │
│       [Evidence]     [Evidence]     [Evidence]                      │
│            |              X              |                          │
│       [Supported]    [Refuted]     [Inconclusive]                   │
│            |                             |                          │
│            └──────────┬─────────────────┘                           │
│                       |                                              │
│                [More Questions]                                      │
│                       |                                              │
│                 [Conclusion]                                        │
│                                                                      │
│  Key edges:                                                         │
│  ├── Question ──GENERATES──> Hypotheses                             │
│  ├── Hypothesis ──PREDICTS──> Expected Evidence                     │
│  ├── Evidence ──SUPPORTS/REFUTES──> Hypothesis                      │
│  └── Conclusion ──ANSWERS──> Question                               │
│                                                                      │
│  Traversal: Breadth-first to generate hypotheses,                   │
│             depth-first to test each one                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 The Decision Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DECISION GRAPH PATTERN                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│               [Decision Point]                                      │
│                /      |      \                                      │
│               /       |       \                                     │
│       [Option A]  [Option B]  [Option C]                            │
│           |           |           |                                  │
│      ┌────┴────┐ ┌────┴────┐ ┌────┴────┐                            │
│      │Pro  Con│ │Pro  Con│ │Pro  Con│                               │
│      └────┬────┘ └────┬────┘ └────┬────┘                            │
│           |           |           |                                  │
│       [Criteria Evaluation]──────┘                                   │
│               |                                                      │
│         [Selected: B]                                               │
│               |                                                      │
│         [Rationale]                                                 │
│               |                                                      │
│      [Implementation Plan]                                          │
│                                                                      │
│  Key edges:                                                         │
│  ├── Decision ──HAS_OPTION──> Option                                │
│  ├── Option ──HAS_PRO/CON──> Argument                               │
│  ├── Criteria ──EVALUATES──> Options                                │
│  ├── Selected ──BECAUSE──> Rationale                                │
│  └── Selection ──ENABLES──> Implementation                          │
│                                                                      │
│  Traversal: Breadth-first to enumerate options,                     │
│             depth-first to evaluate each                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4 The Debug Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                       DEBUG GRAPH PATTERN                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│               [Symptom]                                             │
│                   |                                                  │
│           [Reproduce Steps]                                         │
│                   |                                                  │
│          [Observed Behavior]                                        │
│               /       \                                             │
│              /         \                                            │
│    [Expected]    [Difference]                                       │
│                       |                                              │
│              [Possible Causes]                                      │
│              /    |    |    \                                       │
│           [A]   [B]   [C]   [D]                                     │
│            |     |     X     |                                      │
│         [Test] [Test]     [Test]                                    │
│            |     X           |                                      │
│      [Not It] [ROOT CAUSE!]  |                                      │
│                   |         [Not It]                                │
│               [Fix]                                                 │
│                   |                                                  │
│             [Verify]                                                │
│                   |                                                  │
│             [Resolved]                                              │
│                                                                      │
│  Key edges:                                                         │
│  ├── Symptom ──MANIFESTS_AS──> Observed Behavior                    │
│  ├── Difference ──SUGGESTS──> Possible Causes                       │
│  ├── Test ──CONFIRMS/ELIMINATES──> Cause                            │
│  └── Fix ──RESOLVES──> Root Cause                                   │
│                                                                      │
│  Traversal: Likelihood-ordered depth-first on causes                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.5 The Knowledge Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH PATTERN                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│       [Concept A]═══SIMILAR═══[Concept B]                           │
│            │                       │                                 │
│        CONTAINS                REQUIRES                              │
│            │                       │                                 │
│       [Detail 1]              [Prereq]                              │
│            │                       │                                 │
│        EXAMPLE                 ENABLES                               │
│            │                       │                                 │
│       [Instance]              [Concept C]                           │
│            │                       │                                 │
│       DOCUMENTED_IN           IMPLEMENTED_BY                         │
│            │                       │                                 │
│       [docs/X.md]             [module.py]                           │
│                                                                      │
│  Key relationship types:                                            │
│  ├── Taxonomic: IS_A, CONTAINS, PART_OF                             │
│  ├── Semantic: SIMILAR, CONTRASTS, RELATES                          │
│  ├── Causal: ENABLES, REQUIRES, PREVENTS                            │
│  └── Referential: DOCUMENTED_IN, IMPLEMENTED_BY, TESTED_BY          │
│                                                                      │
│  Use for: Building understanding, finding connections,              │
│           identifying gaps in knowledge                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Practical Graph Templates

### 4.1 Feature Planning Graph

```markdown
## Feature Graph: [Feature Name]

### Core Nodes

**Goal:** [What we're trying to achieve]
**Type:** CONCEPT

**User Story:** [As a... I want... So that...]
**Type:** REQUIREMENT
**Edges:** ACHIEVES → Goal

**Acceptance Criteria:**
1. [Criterion 1] **Type:** FACT (when verified)
2. [Criterion 2]
3. [Criterion 3]
**Edges:** VALIDATES → User Story

### Design Nodes

**Approach:** [High-level approach]
**Type:** DECISION
**Options explored:** [A, B, C]
**Selected:** [B]
**Edges:** IMPLEMENTS → User Story

**Components:**
- [Component 1] **Type:** ARTIFACT
- [Component 2] **Type:** ARTIFACT
**Edges:** PART_OF → Approach

### Implementation Nodes

**Tasks:**
- [ ] [Task 1] **Type:** TASK **Edges:** PRODUCES → Component 1
- [ ] [Task 2] **Type:** TASK **Edges:** PRODUCES → Component 2
- [ ] [Task 3] **Type:** TASK **Edges:** DEPENDS_ON → Task 1, Task 2

### Open Questions

- [?] [Question 1] **Type:** QUESTION **Edges:** BLOCKS → Task 2
- [?] [Question 2] **Type:** QUESTION **Edges:** RAISED_BY → Component 1

### Graph Visualization

Goal
  │
  └──→ User Story
         │
         ├──→ Acceptance Criteria (validates)
         │
         └──→ Approach (implements)
                │
                ├──→ Component 1 ←── Task 1
                │         ↑
                │         └── [?] Question 2
                │
                └──→ Component 2 ←── Task 2 ←── [?] Question 1
                          ↑
                          └── Task 3 (depends on Task 1 & 2)
```

### 4.2 Bug Investigation Graph

```markdown
## Bug Graph: [Bug Title]

### Symptom Cluster

**Observed:** [What user/test sees]
**Type:** FACT

**Expected:** [What should happen]
**Type:** FACT
**Edges:** CONTRASTS → Observed

**Reproduce:**
1. [Step 1]
2. [Step 2]
3. [Bug manifests]
**Type:** FACT
**Edges:** PRODUCES → Observed

### Hypothesis Cluster

**Hypothesis A:** [Potential cause]
**Type:** QUESTION
**Prior likelihood:** [High/Medium/Low]
**Test:** [How to verify]
**Result:** [Confirmed/Refuted/Inconclusive]
**Edges:** COULD_EXPLAIN → Observed

**Hypothesis B:** [Potential cause]
**Type:** QUESTION
**Prior likelihood:** [High/Medium/Low]
**Test:** [How to verify]
**Result:** [Confirmed/Refuted/Inconclusive]
**Edges:** COULD_EXPLAIN → Observed

### Resolution Cluster

**Root Cause:** [The actual cause]
**Type:** INSIGHT
**Edges:** EXPLAINS → Observed, ANSWERS → Hypothesis B

**Fix:** [What we changed]
**Type:** TASK
**Edges:** RESOLVES → Root Cause

**Verification:** [How we confirmed fix]
**Type:** FACT
**Edges:** VALIDATES → Fix

### Graph Visualization

[Observed] ←─CONTRASTS─→ [Expected]
     ↑
     │
[Reproduce]
     │
     └──COULD_EXPLAIN──→ [Hypothesis A] ──TEST──→ [Refuted]
     │
     └──COULD_EXPLAIN──→ [Hypothesis B] ──TEST──→ [Confirmed]
                              │                        │
                              └────────────────────────┘
                                         │
                                   [Root Cause]
                                         │
                                       [Fix]
                                         │
                                   [Verification]
```

### 4.3 Architecture Decision Graph

```markdown
## Architecture Decision Graph: [Decision Title]

### Context Cluster

**Problem:** [What architectural challenge we face]
**Type:** CONCEPT

**Constraints:**
- [Constraint 1] **Type:** FACT **Edges:** LIMITS → Options
- [Constraint 2] **Type:** FACT **Edges:** LIMITS → Options

**Goals:**
- [Goal 1] **Type:** CONCEPT **Edges:** EVALUATES → Options
- [Goal 2] **Type:** CONCEPT **Edges:** EVALUATES → Options

### Options Cluster

**Option A: [Name]**
**Type:** DECISION_OPTION
**Description:** [How it works]
**Pros:**
- [Pro 1] **Edges:** SUPPORTS → Option A
- [Pro 2] **Edges:** SUPPORTS → Option A
**Cons:**
- [Con 1] **Edges:** AGAINST → Option A
**Edges:** SATISFIES/VIOLATES → Constraints, ACHIEVES → Goals

**Option B: [Name]**
(same structure)

**Option C: [Name]**
(same structure)

### Decision Cluster

**Selected:** [Option B]
**Type:** DECISION

**Rationale:**
[Why this option was chosen]
**Type:** INSIGHT
**Edges:** JUSTIFIES → Selected

**Trade-offs accepted:**
- [Trade-off 1]
- [Trade-off 2]

**Consequences:**
- [Consequence 1] **Type:** FACT **Edges:** FOLLOWS_FROM → Selected
- [Consequence 2] **Type:** TASK **Edges:** REQUIRED_BY → Selected

### Graph Visualization

[Problem]
    │
    ├──→ [Constraint 1]──┐
    ├──→ [Constraint 2]──┼──LIMITS──→ [Options]
    │                    │
    ├──→ [Goal 1]────────┼──EVALUATES──→ [Option A]
    └──→ [Goal 2]────────┘              [Option B] ← SELECTED
                                        [Option C]
                                             │
                                        [Rationale]
                                             │
                                      [Consequences]
```

---

## Part 5: Graph Navigation Strategies

### 5.1 Entry Point Selection

<!--
  WHERE you start exploring affects WHAT you discover.
  Choose entry points strategically.
-->

**Entry point strategies:**

| Strategy | When to Use | How |
|----------|-------------|-----|
| **Start from symptom** | Debugging | Begin at observable problem, trace backward |
| **Start from goal** | Planning | Begin at desired outcome, decompose backward |
| **Start from constraint** | Architecture | Begin at hard limits, explore what's possible |
| **Start from hub** | Understanding | Begin at most-connected node, radiate outward |
| **Start from orphan** | Cleanup | Begin at disconnected nodes, integrate or prune |
| **Start from question** | Research | Begin at unknown, seek answers |

### 5.2 Traversal Heuristics

**The "Follow the Energy" Heuristic:**
```
When exploring, prioritize paths that:
1. Have high uncertainty (questions over facts)
2. Have high impact (hubs over leaves)
3. Are blocking other work (dependencies)
4. Are time-sensitive (deadlines approaching)

Don't get lost in interesting-but-irrelevant branches.
```

**The "Complete the Cluster" Heuristic:**
```
When you enter a cluster:
1. Identify all nodes in the cluster
2. Understand internal relationships
3. Identify edges leaving the cluster
4. Summarize before leaving

Don't half-understand a topic.
```

**The "Bridge First" Heuristic:**
```
When connecting multiple areas:
1. Identify bridge nodes between clusters
2. Understand what crosses each bridge
3. Ensure bridges are solid before depending on them

Bridges are high-value targets for understanding.
```

### 5.3 Getting Unstuck

**When lost in the graph:**
```
1. ZOOM OUT: Collapse detail, see clusters
2. FIND HUB: Go to most-connected node
3. TRACE BACK: How did you get here?
4. CHECK GOAL: Is current path relevant to objective?
5. PRUNE: Remove nodes that don't serve the goal
```

**When graph is too complex:**
```
1. COUNT: How many nodes? How many clusters?
2. LAYER: Separate into abstraction levels
3. SCOPE: Define boundaries (what's in/out)
4. DELEGATE: Assign clusters to different agents/times
5. SERIALIZE: Convert to sequence for current focus
```

**When connections are unclear:**
```
1. EXPLICIT EDGES: Name every relationship
2. TEST EDGES: "Does A really require B?"
3. FIND MISSING: "What should connect to this?"
4. CHALLENGE CYCLES: "Is this circular dependency real?"
5. VALIDATE: Have someone else check your graph
```

---

## Part 6: Graph Collaboration

<!--
  GRAPHS ARE SHARED ARTIFACTS.
  Multiple people/agents can contribute to the same graph.
  This requires coordination.
-->

### 6.1 Collaborative Graph Protocols

**Adding to shared graph:**
```markdown
## Proposed Addition

**New node(s):**
- [Node]: [Type] [Description]

**New edge(s):**
- [From] ──[Relationship]──> [To]

**Rationale:**
[Why this should be added]

**Impact:**
[What existing nodes/edges this affects]
```

**Modifying shared graph:**
```markdown
## Proposed Modification

**Current state:**
- [Node/Edge as it exists]

**Proposed change:**
- [How it should change]

**Rationale:**
[Why this change]

**Migration:**
[How to update dependent understanding]
```

**Challenging graph structure:**
```markdown
## Graph Challenge

**Target:**
- [Node/Edge/Cluster being challenged]

**Challenge:**
[Why current structure may be wrong]

**Evidence:**
- [Supporting evidence for challenge]

**Proposed alternative:**
[Different structure]

**Impact if accepted:**
[What would change]
```

### 6.2 Graph Merging

**When parallel work creates divergent graphs:**

```
Graph A (Person 1):        Graph B (Person 2):

[X]──→[Y]──→[Z]            [X]──→[Y]──→[Z]
       │                          │    │
       └──→[A]                    │    └──→[B]
                                  └──→[C]

Merge questions:
1. Is [A] from Graph A valid? Keep/Discard?
2. Is [B] from Graph B valid? Keep/Discard?
3. Is [C] from Graph B valid? Keep/Discard?
4. Any conflicts? [A] and [B] seem similar - merge?

Merged Graph:
[X]──→[Y]──→[Z]
       │    │
       │    └──→[AB] (merged similar nodes)
       └──→[C]
```

### 6.3 Graph Versioning

**Track graph evolution:**
```markdown
## Graph Version History

### v3 (current)
- Added: [Node X], [Edge Y→Z]
- Removed: [Node W] (obsolete)
- Changed: [Node Q] type from QUESTION to FACT

### v2
- Major restructure: Split [Big Cluster] into [A] and [B]
- Added bridge node [Bridge] between clusters

### v1
- Initial graph with [N] nodes, [M] edges
```

---

## Part 7: Integration with Reasoning Workflow

<!--
  This document complements complex-reasoning-workflow.md
  Here's how they connect.
-->

### 7.1 QAPV Loop as Graph Operations

| QAPV Phase | Graph Operations |
|------------|------------------|
| **QUESTION** | Add QUESTION nodes, identify gaps, find orphans |
| **ANSWER** | Traverse for information, add FACT/INSIGHT nodes, resolve questions |
| **PRODUCE** | Add TASK/ARTIFACT nodes, connect to requirements, track dependencies |
| **VERIFY** | Add validation edges, mark questions as answered, prune invalid paths |

### 7.2 Decision Trees as Subgraphs

The decision patterns from complex-reasoning-workflow.md Part 2 are subgraphs:
- Decision Point = hub node
- Options = connected nodes
- Selection = edge weight/highlight
- Pruned options = removed nodes (documented)

### 7.3 Knowledge Transfer as Graph Export

The handoff document (complex-reasoning-workflow.md Part 7) is a graph serialization:
- "What works" = validated ARTIFACT nodes
- "What's incomplete" = TASK nodes with status
- "Key decisions" = DECISION nodes with rationale
- "Questions for successor" = unresolved QUESTION nodes

---

## Part 8: Quick Reference

### Node Type Symbols
```
[C] = Concept      [?] = Question    [D] = Decision
[F] = Fact         [T] = Task        [A] = Artifact
[I] = Insight
```

### Edge Type Symbols
```
──→     Directed relationship
←──→    Bidirectional
══→     Critical/strong
···→    Tentative/weak
──X     Conflict
```

### Common Operations
```
ADD:     Create new node/edge
REMOVE:  Delete node/edge (document why)
MERGE:   Combine similar nodes
SPLIT:   Separate overloaded node
PRUNE:   Remove irrelevant subgraph
COLLAPSE: Summarize cluster as single node
EXPAND:  Show cluster internal structure
LAYER:   Organize by abstraction level
```

### Traversal Commands
```
DFS(node):     Explore depth-first from node
BFS(node):     Explore breadth-first from node
PATH(a, b):    Find connections between a and b
CLUSTER(node): Find all nodes in same cluster
HUBS():        Find most-connected nodes
ORPHANS():     Find disconnected nodes
BRIDGES():     Find cluster connectors
```

### Graph Health Checks
```
☑ No orphan nodes (or orphans are intentional)
☑ No unexplained cycles
☑ All QUESTION nodes have investigation status
☑ All DECISION nodes have rationale
☑ Clusters are cohesive (internal edges > external)
☑ Bridges are documented
☑ Graph is navigable from multiple entry points
```

---

## Closing Thoughts

<!--
  FOR THE NEXT DEVELOPER:

  Graphs are maps of thought territory.
  Like any map, they simplify reality.
  Like any map, they become outdated.
  Like any map, they're more useful when shared.

  The value isn't in the graph itself - it's in the
  clarity that comes from making connections explicit.

  When you find your graph doesn't match reality,
  update the graph. That's how knowledge grows.
-->

A thought graph is a living model of understanding. It grows as you learn, shrinks as you prune, and restructures as you gain insight. The goal isn't a perfect graph - it's a useful one.

**Key principles:**
1. **Explicit > Implicit**: Name nodes and edges
2. **Structure > Sequence**: Capture connections, not just order
3. **Evolving > Fixed**: Update as understanding changes
4. **Shared > Private**: Graphs enable collaboration
5. **Useful > Complete**: Serve the goal, not the model

Use this alongside the Complex Reasoning Workflow. The workflow tells you **how to think**; the graph shows you **what you're thinking about**.

---

*"The map is not the territory, but a good map makes the territory navigable."*

---

## Document Summary

| Part | Title | Focus |
|------|-------|-------|
| 1 | Anatomy of a Thought Graph | Nodes, edges, clusters, properties |
| 2 | Graph Operations | Construction, traversal, analysis, transformation |
| 3 | Graph Patterns | Requirements, investigation, decision, debug, knowledge |
| 4 | Practical Templates | Feature planning, bug investigation, architecture decision |
| 5 | Navigation Strategies | Entry points, traversal heuristics, getting unstuck |
| 6 | Graph Collaboration | Protocols, merging, versioning |
| 7 | Integration | Connection to QAPV workflow |
| 8 | Quick Reference | Symbols, operations, health checks |

---

*Created: 2025-12-19*
*Version: 1.0*
*Companion to: docs/complex-reasoning-workflow.md*
*Status: Living document - update as patterns evolve*
