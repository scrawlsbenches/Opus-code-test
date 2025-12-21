# Sub-Agent Utilization Plan for GoT Development

**Author:** Claude (reflecting on how to work more effectively)
**Date:** 2025-12-21
**Purpose:** Define when and how to use sub-agents to maximize efficiency while maintaining cognitive continuity

---

## The Core Problem I'm Solving

Every message I send consumes context. Complex tasks require:
1. **Research** - Finding relevant code, understanding patterns
2. **Planning** - Deciding approach, identifying dependencies
3. **Implementation** - Writing code, tests, docs
4. **Verification** - Running tests, checking coverage

If I do all of this in my main context, I:
- Lose earlier conversation context
- Can't parallelize independent work
- Get bogged down in details that don't require my judgment

**Solution:** Delegate mechanical work to sub-agents, preserve main context for decisions.

---

## Available Sub-Agent Types

| Type | Use For | Context Access | Speed |
|------|---------|---------------|-------|
| `Explore` | Finding code, understanding patterns | Full conversation | Fast |
| `general-purpose` | Implementation, multi-step tasks | Full conversation | Medium |
| `Plan` | Architecture decisions, complex planning | Full conversation | Medium |
| `claude-code-guide` | Claude Code/SDK documentation | Full conversation | Fast |

---

## When to Use Each Agent

### `Explore` Agent - My Scout

**Use when:**
- "Where is X implemented?"
- "How does Y work across the codebase?"
- "Find all uses of Z pattern"
- Understanding unfamiliar modules before modifying them

**Thoroughness levels:**
- `quick` - Basic pattern search, 1-2 files
- `medium` - Moderate exploration, follows references
- `very thorough` - Comprehensive analysis, multiple locations

**Example prompts:**
```
"Find all places where EdgeType is used and how it's handled.
Quick search - just need the main patterns."

"Thoroughly explore how the GoT event sourcing works.
I need to understand the full flow from event creation to persistence to rebuild."
```

**When NOT to use:**
- I already know the file path (just Read it)
- Simple grep for a specific string (use Grep directly)
- I need to make a decision about what I find (do that myself)

---

### `general-purpose` Agent - My Implementer

**Use when:**
- Writing tests for a specific module
- Implementing a well-defined feature
- Fixing a bug with clear reproduction steps
- Updating documentation across multiple files

**Key principle:** Give complete context upfront. Agent can't ask clarifying questions.

**Template for implementation tasks:**
```markdown
## Task: [Clear Title]

### Context
[What problem are we solving? What's the current state?]

### Files to Modify
- `/path/to/file.py` - [what changes needed]

### Acceptance Criteria
- [ ] Specific measurable outcome
- [ ] Test command to verify
- [ ] Coverage target if applicable

### Constraints
- Do NOT modify [files to avoid]
- Follow pattern in [reference file]
- [Any other constraints]

### Verification
Run: `[test command]`
Expected: [what success looks like]
```

**When NOT to use:**
- I'm uncertain about the approach (figure that out first)
- Task requires judgment calls mid-implementation
- I need to see intermediate results to decide next step

---

### `Plan` Agent - My Architect

**Use when:**
- Multiple valid approaches exist
- Significant architectural decisions needed
- Large-scale changes touching many files
- Need to understand full scope before starting

**When to use instead of implementing directly:**
- "Add authentication" - many approaches, need to choose
- "Refactor X system" - need to map dependencies first
- "Improve performance" - need to profile and analyze first

**Template:**
```markdown
## Planning Task: [Feature/Refactor Name]

### Goal
[What are we trying to achieve?]

### Questions to Answer
1. What approaches are viable?
2. What are the trade-offs?
3. What files would be affected?
4. What's the recommended sequence?

### Existing Context
[Any decisions already made, constraints]

### Output Format
Provide:
1. Recommended approach with rationale
2. Alternative approaches considered
3. Implementation steps in order
4. Risks and mitigation
```

---

## Parallel Execution Patterns

### Pattern 1: Research in Parallel

When I need to understand multiple areas before making a decision:

```
SPAWN PARALLEL:
├── Explore: "How does GoT handle task creation?"
├── Explore: "How does the ML data collector track commits?"
└── Explore: "What's the current sprint tracking approach?"

WAIT → Synthesize findings → Make decision
```

### Pattern 2: Implementation in Parallel

When I have independent implementation tasks:

```
SPAWN PARALLEL:
├── general-purpose: "Add tests for module A. File: tests/test_a.py"
├── general-purpose: "Add tests for module B. File: tests/test_b.py"
└── general-purpose: "Update docs for both. File: docs/api.md"

WAIT → Verify all pass → Commit together
```

### Pattern 3: Research Then Implement

When I need understanding before implementation:

```
STEP 1: Explore: "Thoroughly analyze how X works"
        ↓
STEP 2: (Me) Review findings, make decision
        ↓
STEP 3: general-purpose: "Implement Y based on decision"
        ↓
STEP 4: (Me) Verify and commit
```

---

## GoT-Specific Agent Operations

GoT (Graph of Thought) is designed for **parallel agent work**. Its event-sourced architecture means multiple agents can work simultaneously without conflicts.

### Why GoT + Agents Work Well Together

```
┌─────────────────────────────────────────────────────────────┐
│  MERGE-FRIENDLY BY DESIGN                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Agent A writes: .got/events/20251221-120001-aaaa.jsonl    │
│  Agent B writes: .got/events/20251221-120001-bbbb.jsonl    │
│  Agent C writes: .got/events/20251221-120002-cccc.jsonl    │
│                                                             │
│  On merge: Git preserves ALL event files (no conflicts!)   │
│  On load: Sort by timestamp → replay → consistent state    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### GoT Handoff Primitives (Built-in Multi-Agent Coordination)

GoT has **native handoff support**. Use it to coordinate agent work:

```bash
# 1. I initiate handoff to sub-agent
python scripts/got_utils.py handoff initiate TASK_ID \
  --target "sub-agent-1" \
  --instructions "Write tests for the query module"

# 2. Sub-agent accepts (in their prompt, include this)
python scripts/got_utils.py handoff accept HANDOFF_ID --agent "sub-agent-1"

# 3. Sub-agent completes with results
python scripts/got_utils.py handoff complete HANDOFF_ID \
  --agent "sub-agent-1" \
  --result '{"tests_written": 15, "coverage": "92%"}'

# 4. I can query handoff status anytime
python scripts/got_utils.py handoff list --status accepted
```

**Agent Prompt Template with Handoff:**
```markdown
## Task: Write Tests for Query Module

You are sub-agent-1. A handoff has been initiated for you.

### First: Accept the handoff
```bash
python scripts/got_utils.py handoff accept H-20251221-120530-a1b2 --agent sub-agent-1
```

### Then: Do the work
[Detailed task instructions...]

### Finally: Complete the handoff
```bash
python scripts/got_utils.py handoff complete H-20251221-120530-a1b2 \
  --agent sub-agent-1 \
  --result '{"tests": 15, "file": "tests/test_query.py"}'
```
```

---

### GoT Agent Task Patterns

#### Pattern 1: Task Batch Creation
Delegate creating multiple related tasks:

```markdown
## Agent Task: Create Sprint Tasks

Create tasks for Sprint 12 in GoT. Use these commands:

```bash
# Create the sprint first
python scripts/got_utils.py sprint create "Sprint 12 - Search Improvements" --number 12

# Create each task with dependencies
python scripts/got_utils.py task create "Implement BM25 scoring" \
  --priority high --category feature --sprint S-012

python scripts/got_utils.py task create "Add query expansion tests" \
  --priority medium --category test --sprint S-012 \
  --depends-on T-XXX  # Use ID from previous task
```

Report back: Sprint ID and all task IDs created.
```

#### Pattern 2: Dependency Analysis
Delegate finding critical paths:

```markdown
## Agent Task: Analyze Blocking Relationships

Query the GoT graph to find:
1. All blocked tasks: `python scripts/got_utils.py query "blocked tasks"`
2. For each blocked task, what blocks it: `python scripts/got_utils.py query "what blocks TASK_ID"`
3. Find the root blockers (tasks that block others but aren't blocked)

Report: A dependency tree showing the critical path.
```

#### Pattern 3: Decision Documentation
Delegate logging decisions with full context:

```markdown
## Agent Task: Document Architecture Decision

Log this decision in GoT:

Decision: "Use event sourcing for GoT persistence"
Rationale: "Merge-friendly, audit trail, recoverable"
Alternatives: ["Direct file writes", "SQLite database", "Redis"]
Affects: [T-persistence-001, T-persistence-002]

Command:
```bash
python scripts/got_utils.py decision log "Use event sourcing for GoT persistence" \
  --rationale "Merge-friendly, provides audit trail, recoverable from any state" \
  --alternatives "Direct file writes" "SQLite database" "Redis" \
  --affects T-persistence-001 T-persistence-002
```

Report: Decision ID and edges created.
```

#### Pattern 4: Health Check and Repair
Delegate validation and fixes:

```markdown
## Agent Task: GoT Health Audit

1. Run validation:
```bash
python scripts/got_utils.py validate
```

2. If orphan rate > 30%, investigate:
```bash
python scripts/got_utils.py query "relationships ORPHAN_TASK_ID"
```

3. If edge loss detected, check event log vs graph state

4. Report findings with specific recommendations.
```

#### Pattern 5: Sprint Progress Tracking
Delegate gathering metrics:

```markdown
## Agent Task: Sprint Status Report

Gather sprint metrics:
```bash
python scripts/got_utils.py sprint status
python scripts/got_utils.py dashboard
python scripts/got_utils.py task list --status in_progress
python scripts/got_utils.py task list --status blocked
```

Calculate:
- Completion percentage
- Velocity (tasks completed this week)
- Blockers requiring attention

Report: Formatted sprint status with recommendations.
```

---

### Parallel GoT Agent Workflows

#### Workflow 1: Parallel Task Implementation

When multiple GoT tasks can be worked on simultaneously:

```
ME: Create handoffs for each task
    ↓
SPAWN PARALLEL (each agent gets handoff ID):
├── Agent 1: "Accept H-001, implement feature A, complete handoff"
├── Agent 2: "Accept H-002, implement feature B, complete handoff"
└── Agent 3: "Accept H-003, write tests for A+B, complete handoff"
    ↓
ME: Query handoff status, verify all completed
    ↓
ME: Commit all changes together
```

#### Workflow 2: Research → Decide → Implement

```
STEP 1 - PARALLEL RESEARCH:
├── Explore: "How does GoT handle edge creation?"
├── Explore: "What's the current orphan rate and why?"
└── Explore: "How do other systems solve this?"
    ↓
STEP 2 - ME: Synthesize findings, make decision, log in GoT:
    python scripts/got_utils.py decision log "..." --rationale "..."
    ↓
STEP 3 - PARALLEL IMPLEMENTATION:
├── general-purpose: "Implement the fix in got_utils.py"
└── general-purpose: "Add regression tests"
    ↓
STEP 4 - ME: Verify, complete tasks in GoT, commit
```

#### Workflow 3: Sprint Planning with Agents

```
STEP 1 - Explore: "Analyze current sprint progress and blockers"
    ↓
STEP 2 - ME: Review findings, decide sprint scope
    ↓
STEP 3 - general-purpose: "Create sprint and tasks in GoT based on this spec..."
    ↓
STEP 4 - ME: Review created tasks, add dependencies
    ↓
STEP 5 - PARALLEL: Distribute work via handoffs to multiple agents
```

---

### GoT Event Types Agents Can Create

| Event Type | Created By | Use Case |
|-----------|------------|----------|
| `node.create` | `task create`, `decision log` | New tasks, decisions |
| `node.update` | `task start`, `task complete` | Status changes |
| `edge.create` | `--depends-on`, `--blocks` | Relationships |
| `handoff.initiate` | `handoff initiate` | Delegate to agent |
| `handoff.accept` | `handoff accept` | Agent starts work |
| `handoff.complete` | `handoff complete` | Agent returns results |
| `decision.create` | `decision log` | Architectural choices |

---

### GoT Query Cheatsheet for Agents

Include these in agent prompts as needed:

```bash
# What blocks this task?
python scripts/got_utils.py query "what blocks TASK_ID"

# What depends on this task?
python scripts/got_utils.py query "what depends on TASK_ID"

# Find path between tasks
python scripts/got_utils.py query "path from TASK_A to TASK_B"

# All relationships for a task
python scripts/got_utils.py query "relationships TASK_ID"

# Status-based queries
python scripts/got_utils.py query "active tasks"    # in_progress
python scripts/got_utils.py query "pending tasks"   # pending
python scripts/got_utils.py query "blocked tasks"   # blocked
```

---

### Critical GoT Bugs to Warn Agents About

Include this in agent prompts when they modify GoT code:

```markdown
## CRITICAL: Known Bug Patterns to Avoid

1. **Edge Parameters**: Use `from_id`/`to_id`, NOT `source_id`/`target_id`
   ```python
   # WRONG: graph.add_edge(source_id=x, target_id=y)
   # RIGHT: graph.add_edge(from_id=x, to_id=y)
   ```

2. **EdgeType Lookup**: Use try/except, NOT hasattr()
   ```python
   # WRONG: if hasattr(EdgeType, name): ...
   # RIGHT: try: EdgeType[name] except KeyError: ...
   ```

3. **Query Output Parsing**: Skip first line (query echo)
   ```python
   # First line is the query itself, skip it
   for line in output.split('\n')[1:]:
   ```
```

---

## Anti-Patterns to Avoid

### 1. Over-Delegation
**Bad:** Spawning an agent to read one file
**Good:** Just read the file directly

### 2. Under-Specification
**Bad:** "Write some tests for the query module"
**Good:** Detailed template with files, criteria, verification

### 3. Sequential When Parallel Is Possible
**Bad:** Agent A → wait → Agent B → wait → Agent C
**Good:** Agent A + B + C in parallel (if independent)

### 4. Parallel When Sequential Is Needed
**Bad:** Implement feature + Write tests for it (parallel)
**Good:** Implement feature → then write tests

### 5. Delegating Decisions
**Bad:** "Should we use approach A or B?"
**Good:** Research both approaches, then I decide

---

## Integration with GoT Workflow

### Before Spawning Agents
```bash
# Check current state
python scripts/got_utils.py validate
python scripts/got_utils.py task list --status in_progress
```

### Create Task for Complex Work
```bash
# Track in GoT before delegating
python scripts/got_utils.py task create "Implement X feature" --priority high
```

### After Agents Complete
```bash
# Update task status
python scripts/got_utils.py task complete T-XXX

# Log decision if architectural
python scripts/got_utils.py decision log "Chose approach A" --rationale "Because..."
```

---

## Quick Reference

| Situation | Agent | Thoroughness |
|-----------|-------|-------------|
| "Where is X?" | Explore | quick |
| "How does X work?" | Explore | medium |
| "Full analysis of X" | Explore | very thorough |
| "Write tests for X" | general-purpose | N/A |
| "Implement feature X" | general-purpose | N/A |
| "How should we build X?" | Plan | N/A |
| "What does Claude Code do?" | claude-code-guide | N/A |
| **GoT Operations** | | |
| "Create sprint + tasks" | general-purpose | N/A |
| "Analyze blockers" | Explore | medium |
| "Log decision" | general-purpose | N/A |
| "Health check" | Explore | quick |
| "Sprint status report" | Explore | medium |
| "Implement GoT feature" | general-purpose + handoff | N/A |

---

## Success Metrics

I'm using sub-agents well when:
- [ ] My main context stays focused on decisions, not research
- [ ] Parallel agents complete independent work simultaneously
- [ ] I'm not delegating then re-doing the same work
- [ ] GoT tracks tasks that agents are working on
- [ ] Verification happens after every batch

---

*This plan was created to help me (Claude) work more effectively on the Cortical Text Processor and GoT system. It should be updated as I learn what works and what doesn't.*
