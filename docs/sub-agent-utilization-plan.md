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

## GoT-Specific Agent Usage

### Validating GoT State
```
Explore (quick): "Check GoT event log integrity.
Run 'python scripts/got_utils.py validate' and report any issues."
```

### Finding Orphan Tasks
```
Explore (medium): "Find tasks in GoT that have no edges connecting them.
Query the graph and identify which tasks are isolated."
```

### Implementing GoT Features
```
general-purpose: "Add a new 'got compact' command to scripts/got_utils.py.
It should merge old event files while preserving the current graph state.
Follow the pattern of existing cmd_* functions.
Test with: python scripts/got_utils.py compact --dry-run"
```

### Planning GoT Architecture Changes
```
Plan: "We need to add transaction support to GoT.
Currently events are written immediately with no rollback.
Analyze the current persistence approach and propose
a transactional wrapper that can commit or rollback batches."
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
