# Definition of Done

This document defines when a task is truly "done" versus merely "code complete". Following these criteria ensures that work is production-ready and all discoveries are properly documented.

## Context

During feature development, it's easy to focus solely on implementation and overlook critical steps like documentation, verification, and issue tracking. This document provides a checklist to prevent incomplete work from being marked as finished.

**Example**: While implementing passage-level search features, we discovered a gap in passage-level boosting that could have been lost if not explicitly documented and added to TASK_LIST.md.

---

## Completion Criteria

### 1. Code Complete (Necessary but Not Sufficient)

- [ ] Implementation finished and functionally correct
- [ ] Unit tests written and passing
- [ ] No regressions in existing tests (full test suite passes)
- [ ] Code follows project style guidelines
- [ ] Type hints added to all public functions
- [ ] No obvious performance issues introduced

**Command to verify:**
```bash
python -m unittest discover -s tests -v
```

### 2. Documentation Complete

- [ ] All public functions have Google-style docstrings
  - Args section with types and descriptions
  - Returns section with type and description
  - Examples if the function is non-trivial
- [ ] TASK_LIST.md updated with:
  - Task marked as DONE
  - Solution details added
  - Implementation notes if applicable
- [ ] New APIs documented in relevant files:
  - Added to CLAUDE.md quick reference if user-facing
  - Usage examples provided for complex features
- [ ] PATTERNS.md updated if new patterns introduced

**Files to check:**
- Source code docstrings
- `/home/user/Opus-code-test/TASK_LIST.md`
- `/home/user/Opus-code-test/CLAUDE.md`
- `/home/user/Opus-code-test/docs/PATTERNS.md`

### 3. Verification Complete

- [ ] Feature tested end-to-end (not just unit tests)
  - Run showcase.py to verify integration
  - Test with realistic data, not just toy examples
- [ ] Dog-fooding performed when applicable:
  - Use codebase search to find related code
  - Test feature on the Cortical codebase itself
  - Verify behavior matches expectations
- [ ] Edge cases explored:
  - Empty corpus
  - Single document
  - Large corpus (performance testing)
  - Malformed input
  - Boundary conditions
- [ ] Limitations documented:
  - Known issues noted in docstrings or TASK_LIST.md
  - Performance characteristics documented
  - Unsupported use cases called out

**Commands to verify:**
```bash
python showcase.py
python scripts/search_codebase.py "your feature keywords"
```

### 4. Issue Tracking Complete

This is the step that is most often skipped but is critical for maintaining project knowledge.

- [ ] All discovered issues added to TASK_LIST.md:
  - New tasks created with clear descriptions
  - Priority assigned (Critical/High/Medium/Low)
  - Effort estimated (Small/Medium/Large)
  - Dependencies noted
- [ ] Summary tables updated:
  - Task counts reflect new additions
  - Status categories accurate
  - No orphaned task numbers
- [ ] Related tasks cross-referenced:
  - "See Task #X" links added where relevant
  - Dependencies noted in both directions
- [ ] Future work captured:
  - "Nice to have" features documented
  - Performance optimization opportunities noted
  - Potential extensions recorded

**Example**: When implementing passage search, we found that passage-level boosting was missing. This became Task #66, properly categorized and linked to related search tasks.

### 5. Truly Done

All previous criteria met, plus:

- [ ] Changes committed with descriptive message:
  - Follows project commit message style
  - References task numbers
  - Explains the "why" not just the "what"
- [ ] Commit includes all related files:
  - Source code changes
  - Test updates
  - Documentation updates
  - TASK_LIST.md changes
- [ ] Changes pushed to remote branch
- [ ] Ready for review/merge:
  - No "TODO" comments left in code
  - No commented-out debugging code
  - No temporary files committed

**Git commands:**
```bash
git status
git diff
git add <relevant files>
git commit -m "Implement Task #X: <description>"
git push origin <branch-name>
```

---

## Quick Check

Before marking a task as DONE, answer these questions:

### Testing
- [ ] Did I test this with real usage beyond unit tests?
- [ ] Did I run the full test suite without failures?
- [ ] Did I test edge cases (empty, single, large)?
- [ ] Did I verify behavior in showcase.py or dog-fooding scripts?

### Documentation
- [ ] Did I document all findings, even unexpected ones?
- [ ] Did I update TASK_LIST.md with solution details?
- [ ] Do all new functions have complete docstrings?
- [ ] Did I update user-facing documentation (CLAUDE.md)?

### Issue Tracking
- [ ] Did I create tasks for any issues found during implementation?
- [ ] Did I create tasks for any limitations discovered?
- [ ] Did I create tasks for related work that would improve this feature?
- [ ] Are the summary tables in TASK_LIST.md current?

### Completeness
- [ ] Is the code committed with a descriptive message?
- [ ] Are all related files included in the commit?
- [ ] Is there any "TODO" or temporary code still present?
- [ ] Would another developer understand this work from the documentation?

**If any answer is "no", the task is not done.**

---

## Anti-Patterns to Avoid

### The "Quick Fix" Trap
**Symptom**: Implementing a feature and immediately marking it done without verification.

**Problem**: Issues discovered later require context-switching and rework.

**Solution**: Always run end-to-end tests and dog-fooding before marking done.

### The "It Works on My Machine" Trap
**Symptom**: Testing only the happy path with toy data.

**Problem**: Edge cases fail in production or for other users.

**Solution**: Test with realistic data, empty corpus, and boundary conditions.

### The "Lost Knowledge" Trap
**Symptom**: Discovering an issue during implementation but not documenting it.

**Problem**: Issue gets forgotten and resurfaces later without context.

**Solution**: Immediately add discovered issues to TASK_LIST.md, even if they're out of scope.

### The "Partial Commit" Trap
**Symptom**: Committing code changes but forgetting to commit documentation updates.

**Problem**: Code and documentation fall out of sync.

**Solution**: Use git status before committing to verify all related files are included.

---

## Template: Pre-Commit Checklist

Copy this checklist into your task notes or PR description:

```markdown
## Definition of Done Checklist

### Code Complete
- [ ] Implementation finished
- [ ] Unit tests passing
- [ ] Full test suite passing
- [ ] Type hints added

### Documentation Complete
- [ ] Docstrings added
- [ ] TASK_LIST.md updated
- [ ] User docs updated (if applicable)

### Verification Complete
- [ ] End-to-end testing done
- [ ] showcase.py verified
- [ ] Edge cases tested
- [ ] Limitations documented

### Issue Tracking Complete
- [ ] New issues added to TASK_LIST.md
- [ ] Summary tables updated
- [ ] Dependencies noted

### Truly Done
- [ ] All files committed
- [ ] Descriptive commit message
- [ ] Ready for review
```

---

## Process Flow

```
┌─────────────────┐
│ Code Complete   │
│ (Tests Pass)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Documentation   │
│ Complete        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Verification    │◄──── Found Issue? ────┐
│ Complete        │                       │
└────────┬────────┘                       │
         │                                │
         ▼                                │
┌─────────────────┐                       │
│ Issue Tracking  │───────────────────────┘
│ Complete        │  Add to TASK_LIST.md
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Commit & Push   │
│ (Truly Done)    │
└─────────────────┘
```

---

## Examples

### Good: Task #59 (Add Metadata to Minicolumns)

**What made it good:**
- Code implemented with full type hints
- Tests added and passing
- Docstrings complete
- TASK_LIST.md updated with solution details
- Discovered prerequisite for Task #65 and documented it
- Committed with clear message

### Could Be Better: Initial Passage Search Implementation

**What was missing:**
- Almost forgot to document passage-level boosting gap
- Didn't initially create Task #66 for the missing feature
- Would have lost knowledge without explicit documentation

**Lesson**: Always document issues found during implementation, even if they're out of scope for the current task.

---

## Summary

**Code Complete ≠ Done**

A task is only done when:
1. Code works and tests pass
2. Documentation is complete and accurate
3. Verification confirms real-world usage works
4. All discovered issues are tracked
5. Changes are committed and pushed

**When in doubt, ask:**
- "Would I be comfortable if someone else had to maintain this tomorrow?"
- "Did I document everything I learned, including problems found?"
- "Could someone else understand this work from the documentation alone?"

If the answer to any question is "no", keep working. Your future self (and your teammates) will thank you.
