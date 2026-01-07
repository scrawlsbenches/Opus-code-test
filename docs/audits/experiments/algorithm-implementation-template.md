# Algorithm Implementation Experiment Template

This template tests whether sub-agents can correctly implement data structures and algorithms from first principles.

## Purpose

These experiments measure:
1. **Correctness** - Does the implementation work?
2. **Completeness** - Are all required operations implemented?
3. **Complexity awareness** - Does the agent understand time/space tradeoffs?
4. **Test coverage** - Does the agent write tests proving correctness?

## Template Structure

```markdown
# Experiment: exp-{YYYYMMDD}-{HHMMSS}-{algorithm-name}

## Algorithm
**Name:** [Algorithm name]
**Expected complexity:** [Time/space bounds]
**Required operations:** [List of methods that must work]

## Hypothesis
**I expect:** The agent will/will not correctly implement [algorithm]
**Because:** [reasoning about agent capabilities]

## Task Prompt (Given to Agent)
```
[Exact prompt given to the agent - must include:]
- Algorithm name and description
- Required operations with signatures
- Test cases that must pass
- Constraints (no external libraries, etc.)
```

## Success Criteria
- [ ] All required operations implemented
- [ ] All test cases pass
- [ ] Correct time complexity
- [ ] Correct space complexity
- [ ] Code is readable and documented

## Failure Criteria
- [ ] Missing operations
- [ ] Test cases fail
- [ ] Wrong complexity (e.g., O(n²) when O(n) required)
- [ ] Uses external libraries when forbidden
- [ ] Incomplete implementation marked as "done"

## Prediction
Before running: [PASS / FAIL / PARTIAL]
Confidence: [HIGH / MEDIUM / LOW]
Reasoning: [Why this prediction]

## Actual Result
Status: [PASS / FAIL / PARTIAL]
Operations implemented: [X/Y]
Tests passed: [X/Y]
Notes: [Observations]

## Agent Output
[Paste the agent's code here]

## Test Results
[Paste test execution output]

## Analysis
**Discrepancy:** [Expected vs actual]
**Root cause:** [Why did this happen?]
**Learning:** [What does this teach us about agent capabilities?]

## Recommendations
[How to improve agent performance on this type of task]
```

## Evaluation Rubric

| Score | Meaning |
|-------|---------|
| **PASS** | All operations work, tests pass, complexity correct |
| **PARTIAL** | Some operations work, or tests mostly pass, or minor issues |
| **FAIL** | Core functionality broken, tests fail, or fundamentally wrong |

## Notes on Experiment Design

1. **Isolate the algorithm** - Don't test multiple things at once
2. **Provide clear specs** - Ambiguity lets agents "succeed" incorrectly
3. **Include edge cases** - Empty input, single element, duplicates
4. **Require tests** - Agent must prove their code works
5. **Forbid shortcuts** - No `import collections` for Union-Find, etc.
