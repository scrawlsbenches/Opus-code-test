---
description: Delegate a task to a sub-agent with structured context and output format
argument-hint: <task description>
---
# Delegated Task: $ARGUMENTS

You are a delegated agent with NO prior context. Everything you need is in this prompt.

---

## ENVIRONMENT

```
Working Directory: /home/user/Opus-code-test
Project: Cortical Text Processor (Python library for hierarchical text analysis)
Key Docs: CLAUDE.md (read first if unsure about anything)
```

## SETUP (always do this first)

```bash
cd /home/user/Opus-code-test
git fetch origin
git status
```

If the task mentions a specific branch, checkout that branch:
```bash
git checkout <branch-name>
```

---

## TASK CLASSIFICATION

Based on the task description, identify the type:

| Type | Keywords | Mode | Output |
|------|----------|------|--------|
| **sanity-check** | "sanity", "verify", "pre-merge" | READ-ONLY | PASS/FAIL table |
| **code-review** | "review", "audit", "check code" | READ-ONLY | Findings list |
| **investigate** | "why", "debug", "find", "understand" | READ-ONLY | Analysis report |
| **test** | "test", "coverage", "add tests" | WRITE | Test files + results |
| **fix** | "fix", "bug", "broken", "error" | WRITE | Fix + verification |
| **implement** | "add", "create", "implement", "build" | WRITE | Code + tests |

---

## TASK-SPECIFIC GUIDANCE

### For SANITY-CHECK tasks:
```bash
# 1. Verify clean state
git status --short

# 2. Check dependencies
python /home/user/Opus-code-test/scripts/run_tests.py --check-deps

# 3. Run smoke tests
python /home/user/Opus-code-test/scripts/run_tests.py smoke -q

# 4. Verify key commands work (if ML-related)
python /home/user/Opus-code-test/scripts/ml_data_collector.py stats
```

### For CODE-REVIEW tasks:
1. Identify files to review: `git diff --name-only origin/main..HEAD`
2. Read each file, focusing on:
   - Logic errors
   - Edge cases not handled
   - Security issues
   - Performance concerns
3. Do NOT make changes - report findings only

### For INVESTIGATE tasks:
1. Reproduce the issue first
2. Read relevant code (use absolute paths)
3. Form hypothesis
4. Verify hypothesis with targeted tests
5. Report root cause and recommended fix

### For TEST tasks:
1. Check current coverage: `python -m coverage report --include="cortical/*"`
2. Identify untested paths in target module
3. Create test file at `tests/unit/test_<module>.py`
4. Run new tests: `python -m unittest tests.unit.test_<module> -v`
5. Verify coverage improved

### For FIX tasks:
1. Reproduce the bug first
2. Write a failing test that captures the bug
3. Implement the fix
4. Verify test now passes
5. Run smoke tests to check for regressions

### For IMPLEMENT tasks:
1. Read CLAUDE.md for patterns and conventions
2. Check for similar implementations in codebase
3. Implement following existing patterns
4. Add tests for new code
5. Update documentation if needed

---

## DIFF CAPTURE (for WRITE mode tasks)

**Before completing a WRITE task**, capture changes for recovery:

```bash
# If you have a task ID, capture diff for recovery
python /home/user/Opus-code-test/scripts/task_diff.py capture TASK_ID -m "Description of changes"

# Example:
python /home/user/Opus-code-test/scripts/task_diff.py capture T-20251222-193227 -m "Added validation logic"
```

This saves the diff to `.got/diffs/TASK_ID.patch` so changes can be recovered if they don't persist.

**If your changes didn't persist**, the main agent can restore them:
```bash
python /home/user/Opus-code-test/scripts/task_diff.py restore TASK_ID
```

---

## UNIVERSAL RULES

1. **Absolute paths only** - Always use `/home/user/Opus-code-test/...`
2. **Never use `cd`** after setup - it persists and causes confusion
3. **Distinguish issues**:
   - ENV issue: missing dependency, permissions (not your fault)
   - CODE issue: bug, test failure, syntax error (report this)
4. **No hallucinating** - If a file doesn't exist, say so
5. **Ask if stuck** - If task is ambiguous, state assumptions clearly

---

## OUTPUT FORMAT

Always end with a structured report:

```markdown
## Delegation Report

**Task:** [one-line summary]
**Type:** [sanity-check|code-review|investigate|test|fix|implement]
**Branch:** [branch name or "main"]

### Results

| Check/Action | Status | Notes |
|--------------|--------|-------|
| ... | PASS/FAIL/SKIP | ... |

### Findings (if any)
- Finding 1: ...
- Finding 2: ...

### Changes Made (if WRITE mode)
- File 1: [created|modified] - description
- File 2: ...

### Verdict
**[PASS|FAIL|NEEDS-REVIEW]**: [one sentence summary]

### Recommended Next Steps (if any)
1. ...
2. ...
```

---

## FAILURE HANDLING

| Problem | Action |
|---------|--------|
| pytest not installed | Use `python -m unittest discover -s tests -v` instead |
| coverage not installed | Skip coverage, note as ENV limitation |
| File not found | Verify path, check if on correct branch |
| Permission denied | Note as ENV issue, not code problem |
| Tests fail | Report which tests, include error output |
| Ambiguous task | State your interpretation, proceed with assumptions noted |

---

## NOW EXECUTE

Task: $ARGUMENTS

Begin by classifying the task type, then follow the appropriate guidance above.
