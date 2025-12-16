# Code of Ethics - Cortical Text Processor Development

## Preamble

This project demands **scientific rigor** in all aspects of development. As computational engineers, we commit to the same standards applied in peer-reviewed research: reproducibility, transparency, and intellectual honesty. Code that "works" is not enough - we must understand *why* it works, document its limitations, and verify our claims with evidence.

---

## 1. Documentation Ethics

### All findings must be documented, even if they seem minor

Every observation during development carries information. What seems minor today may be critical context for future debugging or feature development.

**Requirements:**
- Document unexpected behavior immediately, not "when time permits"
- Include reproduction steps, not just symptoms
- Note what you tried, even if it didn't work
- Add context about why the behavior matters

### Issues discovered during testing MUST be added to the task system

Testing is a discovery process. Issues found during dog-fooding are **not distractions** - they are the primary signal that our assumptions need refinement.

**Requirements:**
- Add tasks immediately upon discovery: `python scripts/new_task.py "description"`
- Include severity/priority assessment with `--priority high`
- Reference the test case or usage scenario that revealed it
- Link to related code locations with absolute paths

**Example:**
```bash
python scripts/new_task.py "Fix passage-level search doc-type boosting" \
  --priority medium \
  --category bugfix
# Task location: /cortical/query/passages.py:find_passages_for_query
# Issue: Document-level search applies doc-type boosting, but passage-level search does not
# Discovered during dog-fooding test with code search queries
```

### No "it works well enough" - if there's a limitation, document it

Undocumented limitations are landmines for future developers. They waste time, create confusion, and erode trust in the codebase.

**Requirements:**
- Add limitations to docstrings for affected functions
- Document known edge cases in module comments or create tracking tasks
- Be specific: "Doesn't support X when Y" not "Has limitations"
- Include workarounds if available, but track the underlying issue

### Workarounds are not solutions - track the underlying issue

A workaround is technical debt with interest. Document it as such.

**Requirements:**
- Add a comment explaining WHY the workaround exists
- Create a task with `python scripts/new_task.py` for the proper fix
- Reference the task ID in the workaround comment
- Never let a workaround become permanent through neglect

---

## 2. Testing Ethics

### Always exercise new features with real usage (dog-fooding)

Unit tests verify components. Dog-fooding verifies **value**. Both are required.

**Requirements:**
- Use new features in realistic scenarios, not toy examples
- Test against the actual project codebase (we index ourselves for a reason)
- Document the dog-fooding process and results
- If a feature can't be dog-fooded meaningfully, question whether it should exist

### Don't just run unit tests - verify the feature works end-to-end

Passing tests are necessary but not sufficient. Integration and user experience matter.

**Requirements:**
- Run `showcase.py` after significant changes
- Verify features work through the public API, not just internal functions
- Test the entire pipeline: input → processing → output → interpretation
- Consider: "Would I trust this result in production?"

### Document unexpected behavior even if tests pass

Tests encode our expectations. When reality differs from expectations, reality is teaching us something.

**Requirements:**
- Ask "Why?" when behavior surprises you, even if it's good
- Document counterintuitive behavior in docstrings
- Update tests to cover the unexpected case
- Investigate whether the surprise indicates a deeper issue

### Test edge cases and document limitations

The difference between research code and production code is edge case handling.

**Requirements:**
- Test empty corpus, single document, massive corpus
- Test malformed input, Unicode edge cases, pathological queries
- Document what breaks and at what scale
- Add "Known Limitations" sections to docstrings when appropriate

---

## 3. Completion Standards

### A task isn't done until findings are documented

"Done" has three components: implementation, testing, and documentation. All three are mandatory.

**Definition of Done:**
1. Feature implemented and tests pass
2. Feature exercised with real usage (dog-fooding)
3. Findings, limitations, and follow-up issues documented
4. Task status updated with completion details and any new tasks created

### If testing reveals new issues, create follow-up tasks

Testing expands our understanding. New knowledge creates new work - embrace it.

**Requirements:**
- Create follow-up tasks immediately, don't rely on memory
- Link follow-up tasks to the parent task for context
- Assess priority realistically (not everything is urgent)
- Close the parent task only after follow-ups are tracked

### Keep task tracking current when completing work

Task tracking provides project health metrics. Keep it current.

**Requirements:**
- Mark tasks complete: `python scripts/task_utils.py complete TASK_ID`
- Include retrospective notes about what was learned
- View summary: `python scripts/consolidate_tasks.py --summary`
- Commit task file updates with the feature implementation

### Leave the codebase better documented than you found it

Every commit is an opportunity to improve clarity.

**Requirements:**
- If you struggled to understand code, improve its documentation
- Add comments explaining the "why" behind non-obvious decisions
- Update docstrings when behavior changes
- Fix misleading comments immediately - they're worse than no comments

---

## 4. Scientific Rigor

### Be skeptical of "working" results

In science, reproducibility and understanding matter more than outcomes. Apply the same standard here.

**Requirements:**
- Question why a fix works, don't just celebrate that it does
- Test the boundaries: when does it work? When does it fail?
- Look for alternative explanations
- Be especially skeptical of fixes that "just work" without clear causation

### Verify claims with evidence

Anecdotes are hypotheses. Measurements are evidence.

**Requirements:**
- Use quantitative metrics: execution time, memory usage, result quality
- Provide reproduction steps for performance claims
- Compare before/after with controlled tests
- Document test methodology so others can verify

### Document both successes AND limitations

A complete scientific result includes what was learned, what worked, and what didn't.

**Requirements:**
- Note what approaches were tried and failed
- Document performance characteristics (time/space complexity)
- List known failure modes or edge cases
- Be honest about scope: "Solves X but not Y"

### Follow the evidence, not assumptions

Our mental models are often wrong. The code and data don't lie.

**Requirements:**
- When behavior contradicts expectations, trust the behavior
- Investigate discrepancies thoroughly before dismissing them
- Update your understanding based on evidence
- Document surprising findings - they're often the most valuable

---

## Enforcement

This is not a bureaucratic exercise. These standards exist because:

1. **We index our own codebase** - poor documentation directly impacts our tooling
2. **We depend on our own library** - bugs and limitations affect our work
3. **We are scientists** - rigor is not optional
4. **We respect future developers** - including our future selves

Violations aren't moral failures - they're opportunities to learn. When you notice a gap:

1. Document it immediately
2. Fix it if time permits
3. Create a task if not: `python scripts/new_task.py "fix description"`
4. Improve processes to prevent recurrence

---

## Example: The Doc-Type Boosting Case Study

**What happened:** Document-level search (`find_documents_for_query`) correctly applied doc-type boosting. Passage-level search (`find_passages_for_query`) did not, despite claiming to support the feature.

**What we did wrong:**
- Unit tests passed but didn't cover the integration path
- Dog-fooding test existed but results weren't critically examined
- The limitation wasn't documented in the docstring
- No task was created when the gap was first noticed

**What we should have done:**
1. Add explicit test case for passage-level doc-type boosting
2. Run dog-fooding test and examine actual score contributions
3. Document in `find_passages_for_query` docstring: "Note: Currently does not apply doc-type boosting"
4. Create task immediately: `python scripts/new_task.py "Implement doc-type boosting for passage search"`
5. Mark parent task as complete only after documenting this limitation

**This is the standard.** Match it consistently, and the codebase will remain trustworthy.

---

*"The first principle is that you must not fool yourself — and you are the easiest person to fool."* - Richard Feynman
