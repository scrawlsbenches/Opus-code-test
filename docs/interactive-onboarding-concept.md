# Interactive Onboarding Through GoT

## Vision

Rather than static documentation, new Claude sessions begin with a **dynamic, interactive process** that adapts to the user's request. The Graph of Thought becomes the backbone of this experience.

## How It Works

### 1. Request Analysis Phase

When a new session starts, the system analyzes the user's initial request:

```
User: "I want to add a new verification strategy to the reasoning module"
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    REQUEST TOPOLOGY SELECTOR                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Detected concepts: [verification, reasoning, strategy, add]       │
│  Task type: Implementation (new feature)                           │
│  Complexity: Medium (single module, integration required)          │
│                                                                     │
│  Suggested topology: TREE OF THOUGHT                               │
│  Rationale: Clear parent goal with parallel sub-tasks              │
│                                                                     │
│  Auto-generated structure:                                         │
│  ├── Understand existing verification patterns                     │
│  ├── Design new strategy interface                                 │
│  ├── Implement strategy                                            │
│  └── Verify integration with reasoning loop                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. Context Fetching Phase

The system automatically fetches relevant context:

```python
# 1. Query the knowledge graph for related concepts
expanded = processor.expand_query("verification strategy", max=20)
# → verification, phase, checksum, failure, loop, pyramid...

# 2. Find relevant files
relevant_files = processor.find_documents_for_query("verification strategy")
# → verification.py, qapv_verification.py, loop_validator.py

# 3. Surface related decisions from GoT
decisions = got_manager.query()
    .decisions()
    .where(affects__contains="verification")
    .execute()
# → Past decisions about verification design

# 4. Check for blocking dependencies
blockers = got_manager.find_related(
    "verification",
    edge_type="BLOCKS"
)
```

### 3. Interactive Question Phase

Based on the analysis, the system asks targeted questions:

```
Based on analyzing the codebase, I have some questions:

1. VERIFICATION LEVELS
   The existing verification pyramid has 4 levels:
   - UNIT, INTEGRATION, E2E, ACCEPTANCE

   Where does your new strategy fit?
   [ ] New level    [ ] Extends existing level    [ ] Cross-cutting

2. INTEGRATION POINT
   Verification currently plugs into these phases:
   - DRAFTING (quick sanity checks)
   - REFINING (thorough checks)
   - FINALIZING (complete verification)

   When should your strategy run?

3. PRIOR DECISIONS
   I found a related decision:
   "D-20251220-xxxx: Use verification pyramid pattern"
   Rationale: "Layered verification catches issues at appropriate levels"

   Should your strategy follow this pattern or propose an alternative?
```

### 4. Dynamic Task Graph Generation

Based on responses, the system generates a task graph in GoT:

```
T-001: Implement Custom Verification Strategy
├── T-002: Analyze existing verification.py patterns
│   └── [auto-populated with relevant code sections]
├── T-003: Design strategy interface
│   ├── T-003a: Define strategy protocol
│   └── T-003b: Document integration points
├── T-004: Implement strategy
│   ├── T-004a: Write failing tests
│   ├── T-004b: Implement core logic
│   └── T-004c: Add to verification pyramid
└── T-005: Verify integration
    ├── T-005a: Test with cognitive loop
    └── T-005b: Test with crisis scenarios
```

### 5. Adaptive Guidance During Work

As the user works, the system adapts:

```
┌─────────────────────────────────────────────────────────────────────┐
│ WORKING ON: T-004a Write failing tests                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ CONTEXT SURFACED:                                                   │
│ • tests/test_verification.py has 47 existing test patterns         │
│ • VerificationCheck dataclass expects: name, description, level    │
│ • Similar test: test_verification_phase_transition (line 234)      │
│                                                                     │
│ RELATED DECISIONS:                                                  │
│ • "Always test state transitions explicitly"                       │
│ • "Mock external services, test internal logic directly"           │
│                                                                     │
│ SUGGESTED STRUCTURE (based on codebase patterns):                  │
│ ```python                                                          │
│ class TestCustomStrategy:                                          │
│     def test_strategy_interface_contract(self): ...                │
│     def test_strategy_execution_success(self): ...                 │
│     def test_strategy_execution_failure(self): ...                 │
│     def test_integration_with_pyramid(self): ...                   │
│ ```                                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Request Types and Their Flows

### Investigation Request
```
"Why does authentication fail intermittently?"
→ GRAPH topology (non-linear evidence gathering)
→ Auto-creates: Observations, Hypotheses, Evidence nodes
→ Interactive: "What symptoms have you observed?"
→ Surfaces: Related error logs, similar past issues, connected systems
```

### Refactoring Request
```
"Split the monolithic processor into modules"
→ TREE topology (hierarchical decomposition)
→ Auto-analyzes: Import dependencies, coupling metrics
→ Interactive: "Which responsibilities should be separate?"
→ Surfaces: Dependency graph, suggested module boundaries
```

### Learning Request
```
"How does the reasoning framework work?"
→ CHAIN topology (sequential explanation)
→ Auto-surfaces: Core files, concept map, example workflows
→ Interactive: "What's your current understanding level?"
→ Adapts: Depth of explanation, code examples shown
```

### Bug Fix Request
```
"Fix the edge rebuild issue"
→ CHAIN topology (reproduce → analyze → fix → verify)
→ Auto-surfaces: Related code, similar past bugs, test cases
→ Interactive: "Can you reproduce it? What error message?"
→ Creates: Bug task with reproduction steps, linked to fix task
```

## The Onboarding State Machine

```
                    ┌──────────────────┐
                    │   NEW SESSION    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ ANALYZE REQUEST  │ ←─ Parse intent, concepts
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │  CHAIN   │   │   TREE   │   │  GRAPH   │
       │ (simple) │   │(complex) │   │(explore) │
       └────┬─────┘   └────┬─────┘   └────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼─────────┐
                    │  FETCH CONTEXT   │ ←─ Knowledge graph + GoT
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ ASK QUESTIONS    │ ←─ Targeted clarification
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ GENERATE TASKS   │ ←─ Create GoT task graph
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  EXECUTE WORK    │ ←─ With adaptive guidance
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ CAPTURE LEARNING │ ←─ Update profiles, habits
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  SESSION END     │ ←─ Knowledge transfer doc
                    └──────────────────┘
```

## Implementation Notes

### Required Components

1. **TopologySelector** - Analyzes requests and suggests structures
2. **ContextFetcher** - Queries knowledge graph for relevant info
3. **QuestionGenerator** - Creates targeted questions based on gaps
4. **TaskGraphBuilder** - Generates GoT task structures from plans
5. **AdaptiveGuide** - Surfaces relevant context during work
6. **LearningCapture** - Records decisions, corrections, preferences

### Integration Points

- **CorticalTextProcessor** - Semantic search and query expansion
- **GoTManager** - Task/decision tracking and relationships
- **UserProfile** - Preferences, learnings, habits
- **PubSubBroker** - Event notifications for state changes

### Key Insight

The onboarding process itself becomes a **QAPV loop**:
- **Question**: What does the user need?
- **Answer**: Analyze request + fetch context + ask clarifying questions
- **Produce**: Generate appropriate task structure
- **Verify**: Check with user before proceeding

This creates a self-documenting, adaptive experience where each session builds on previous learnings and leaves behind artifacts (tasks, decisions, knowledge transfers) that make future sessions more effective.
