# Parallel Agent Orchestration Pattern

A guide for efficiently completing large-scale tasks using parallel sub-agents.

## Overview

When facing a large initiative with many independent subtasks (like achieving 90% unit test coverage across 20 modules), orchestrating multiple sub-agents in parallel can dramatically reduce completion time while maintaining quality.

**Case Study:** Unit Test Coverage Initiative
- **Goal:** 20 modules from 16% → 90% coverage
- **Result:** 1,729 tests, 85% coverage, 19 modules at 90%+
- **Time:** ~2 hours of orchestrated parallel work

## The Pattern

### 1. Assess and Batch

Before launching agents, understand the full scope:

```
1. Read task definitions: `python scripts/task_utils.py list`
2. Check existing infrastructure (mocks, fixtures, patterns)
3. Identify dependencies between tasks
4. Group into batches by:
   - Independence (can run in parallel)
   - Complexity (similar effort levels)
   - Dependencies (sequential when needed)
```

**Example Batching:**
```
Batch 1 (Small, Quick):     #168 config.py, #169 code_concepts.py
Batch 2 (Core Structures):  #159-162 tokenizer, embeddings, layers, minicolumn
Batch 3 (Core Modules):     #163-164, #178 fingerprint, gaps, persistence
Batch 4 (Query Part 1):     #170-172 expansion, search, passages
Batch 5 (Query Part 2):     #173-175 definitions, analogy, ranking
Batch 6 (Large Modules):    #176-177, #165 analysis, semantics, processor
Batch 7 (Dependent):        #166-167 processor Phase 2, chunk_index
```

### 2. Prepare Agent Context

Each agent needs sufficient context to work independently:

```markdown
## Task #XXX: [Description]

### Source File
`/path/to/source.py` - Brief description

### Output File
Create/Extend: `/path/to/test_file.py`

### Test Infrastructure Available
- What mocks exist (MockMinicolumn, MockLayers, etc.)
- What patterns to follow (reference existing test file)

### Functions to Test
- List specific functions
- Note any already-covered areas

### Test Categories Needed
1. Category A (X+ tests): specifics
2. Category B (Y+ tests): specifics

### Acceptance Criteria
- [ ] N+ tests
- [ ] Coverage ≥ X%
- [ ] Tests run in <Ns

### Instructions
1. Read source file first
2. Check existing coverage
3. Create/extend tests
4. Verify with pytest
5. Report results
```

### 3. Launch in Parallel

Use the Task tool with multiple invocations in a single message:

```
<Task subagent_type="general-purpose">
  Task #168: config.py tests...
</Task>
<Task subagent_type="general-purpose">
  Task #169: code_concepts.py tests...
</Task>
```

**Key principles:**
- Launch all independent tasks in one message
- Wait for batch completion before next batch
- Track progress with TodoWrite

### 4. Handle Results

As agents complete:
1. Check reported coverage/test counts
2. Commit successful work immediately
3. Note any partial completions for follow-up
4. Update tracking (TodoWrite, TASK_LIST.md)

### 5. Iterate on Gaps

After initial passes:
1. Run coverage to find remaining gaps
2. Launch targeted follow-up agents
3. Focus prompts on specific uncovered lines

## Best Practices

### DO:
- **Batch by independence** - Maximize parallelism
- **Provide complete context** - Agents can't ask clarifying questions
- **Reference existing patterns** - "Follow the pattern in test_analysis.py"
- **Set clear acceptance criteria** - Quantifiable goals
- **Commit frequently** - Don't lose work to context limits or caps
- **Track with TodoWrite** - Visibility into progress

### DON'T:
- **Don't launch dependent tasks in parallel** - They'll have conflicts
- **Don't skimp on context** - Agents need full picture
- **Don't forget to verify** - Run tests after each batch
- **Don't batch too large** - 3-4 agents per batch is manageable

## Handling Issues

### Spending Caps
If agents hit spending caps mid-batch:
1. Check git status for partial work
2. Commit any completed files
3. Continue remaining work directly or in new session

### Test Failures
If agents produce failing tests:
1. Run locally to see actual errors
2. Fix directly or re-prompt with error context

### Coverage Gaps
If coverage targets not met:
1. Run coverage with `--cov-report=term-missing`
2. Launch focused agents on specific uncovered lines

## Metrics from Unit Test Initiative

| Metric | Value |
|--------|-------|
| Tasks completed | 20 |
| Batches | 7 |
| Total tests created | 1,729 |
| Coverage improvement | 16% → 85% |
| Modules at 90%+ | 19 of 21 |
| Lines of test code | ~21,000 |

## Template: Agent Task Prompt

```markdown
## Task #[NUMBER]: [TITLE]

You are implementing [WHAT] for `[SOURCE_FILE]`. Goal: [METRIC].

### Source File
`/full/path/to/source.py` - [DESCRIPTION]

### Output File
[Create/Extend]: `/full/path/to/test_file.py`

### Test Infrastructure
Use mocks from `tests/unit/mocks.py`:
- [LIST AVAILABLE MOCKS]

### Test Pattern
Follow `/path/to/example_test.py`:
- [STYLE NOTES]

### Functions to Test
- `function_one()` - [DESCRIPTION]
- `function_two()` - [DESCRIPTION]

### Test Categories
1. **[CATEGORY]** ([N]+ tests): [SPECIFICS]
2. **[CATEGORY]** ([N]+ tests): [SPECIFICS]

### Acceptance Criteria
- [ ] [N]+ unit tests
- [ ] Coverage ≥ [X]%
- [ ] Tests run in <[N] seconds
- [ ] All tests passing

### Instructions
1. Read source file completely
2. [CHECK EXISTING IF EXTENDING]
3. Create comprehensive test classes
4. Run: `python -m pytest [TEST_FILE] -v`
5. Check: `python -m pytest [TEST_FILE] --cov=[MODULE] --cov-report=term-missing`

Report final test count and coverage percentage.
```

---

*This pattern emerged from the unit test coverage initiative completed on 2025-12-13, where 20 tasks were completed in parallel batches to achieve 85% coverage with 1,729 tests.*
