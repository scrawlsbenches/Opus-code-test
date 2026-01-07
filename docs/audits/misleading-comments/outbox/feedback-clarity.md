# Clarity Review: Audit Framework Documentation

**Reviewer:** Sub-agent (clarity focus)
**Date:** 2026-01-07
**Files Reviewed:**
- `docs/audits/README.md`
- `docs/audits/misleading-comments/inbox/task-001.md`

---

## Executive Summary

The framework has strong structural clarity (directory layout, stop conditions, file formats) but suffers from **critical ambiguities** that would cause agent confusion during execution. Most concerning is a **direct contradiction** about manifest ownership that would lead agents to take wrong actions.

**Overall Assessment:** Needs revision before agents can follow reliably.

---

## What Was Clear

### Structural Elements (Excellent)
1. **Directory structure** - Crystal clear diagram with purpose of each folder
2. **Stop conditions** - Hard limits (50 findings, 2-4 hours) are unambiguous
3. **Claiming mechanism** - Rename to `.claimed.md` is concrete and actionable
4. **Output destinations** - Clear mapping: questions/ for confusion, problems/ for errors, outbox/ for results
5. **File format templates** - Both task and result formats provide concrete structure
6. **Required sections** - "What Went Wrong", "Where I Got Confused", "Questions for Human" force reflection

### Behavioral Guidance (Good)
1. **STOP immediately when confused** - Repeated emphasis is effective
2. **Don't update manifest** - Stated clearly in multiple places (with one exception - see below)
3. **Constraints table** - Visual table format makes limits scannable
4. **Check-in requirements** - Table format with "When/Action" is clear

### Safety Features (Excellent)
1. **Context loss recovery** - Acknowledges the problem and provides mechanism
2. **Parking lot** - Provides escape valve for out-of-scope findings
3. **Failsafes section** - Anticipates failure modes

---

## What Was Confusing

### 🚨 CRITICAL: Manifest Ownership Contradiction

**Location:** README.md lines 79-96 vs line 117

**The contradiction:**
- Lines 79-96: "Only the coordinating agent updates manifest.md" (clear)
- Line 117 (Phase 2, step 4): "Updates `manifest.md` to mark task complete" (contradicts)

**Why this is critical:**
This directly contradicts the core protocol. An agent reading Phase 2 would believe they SHOULD update manifest, while the constraints section says they SHOULD NOT.

**Impact:** Agent will either:
- Update manifest (violating protocol, causing race conditions)
- Not update manifest but feel uncertain they're following instructions
- Stop and write a question, wasting time on a documentation bug

**Recommendation:**
Remove line 117 entirely. Phase 2 should end at step 3 (writes results to outbox). Or change to: "Coordinator later updates manifest based on outbox results."

---

### MAJOR: Assessment Categories Undefined

**Location:** Multiple places (README line 252, task-001 line 49)

**The problem:**
Agents must categorize findings as:
- `accurate`
- `stale`
- `misleading`
- `unknown`

**But nowhere defines:**
- What makes a comment "stale" vs "misleading"?
- Is a comment from 2 years ago that's still technically accurate "stale" or "accurate"?
- Is a "FUTURE:" comment with no evidence of progress "stale" or "misleading"?
- When should I use "unknown"?

**Example confusion scenario:**
```python
# FUTURE: When CDG index supports partial updates...
```
- If the CDG index doesn't support partial updates: stale? misleading? accurate-but-aspirational?
- If it was written 3 years ago: does that change the assessment?
- If there's no evidence anyone worked on it: what category?

**Recommendation:**
Add a "Assessment Category Guide" section:

```markdown
## Assessment Categories

| Category | Definition | Examples |
|----------|------------|----------|
| **accurate** | Comment correctly describes current or planned state with evidence of progress | "TODO: Add test coverage" + PR in progress |
| **stale** | Comment was accurate when written but is now outdated due to changes | "FUTURE: When X is implemented" but X was never implemented and won't be |
| **misleading** | Comment was never accurate OR references non-existent things | "See design doc Y" but Y never existed |
| **unknown** | Insufficient information to determine accuracy | Cannot find referenced document or verify claim |
```

---

### MAJOR: Informal Pattern Search Unclear

**Location:** task-001.md line 37

**The problem:**
```markdown
- **Also check:** Comments with "will be", "should be", "planned to"
```

**Ambiguities:**
1. Are these literal string searches or patterns?
2. Should I search for `"will be"` (with quotes)?
3. Are these case-sensitive?
4. Should I search for just these phrases or also `"will have"`, `"should have"`, `"planned for"`?
5. Are these in ADDITION to the formal patterns (FUTURE:, TODO:) or INSTEAD OF?
6. If I find `"# This will be implemented"` does that count as a finding?

**Example confusion scenarios:**
- `"# This will be called by..."` - Is this aspirational (finding) or descriptive (not a finding)?
- `"# Returns true when operation should be retried"` - Contains "should be" but describes behavior, not a plan

**Recommendation:**
Either:
A) Remove this line entirely (focus only on formal markers like FUTURE:, TODO:)
B) Clarify with examples:
```markdown
- **Also check:** Comments containing aspirational language:
  - "will be implemented"
  - "should be added"
  - "planned to support"
  - (Exclude present tense: "returns", "should be used" are NOT findings)
```

---

### MODERATE: Feature Existence Check Undefined

**Location:** task-001.md line 52

**The problem:**
```markdown
4. If comment references a feature, check if feature exists
```

**Ambiguities:**
- What constitutes "checking if a feature exists"?
- Should I search for function names in the codebase?
- Should I look for tests?
- Should I check documentation?
- How much time should I spend on each check?
- What if I find partial implementation?

**Example confusion scenario:**
```python
# TODO: Add caching for query results
```
How do I "check if feature exists"?
- Search for "cache" in the file? (might be unrelated caching)
- Search for query result storage? (how?)
- Look for a cache module? (where?)
- Check if there's a test for caching? (which test?)

**Recommendation:**
Provide concrete verification steps:
```markdown
4. If comment references a feature, verify with these steps (spend max 2 min per check):
   a. Search for related function/class names in the file
   b. Check if there's a test file testing this feature
   c. If unclear after 2 min, mark assessment as "unknown" and note what you checked
```

---

### MODERATE: Git Blame Usage Unclear

**Location:** task-001.md line 53, README line 150

**The problem:**
- README asks "Does git blame show when this was written?" (line 150)
- Task says "Use git blame to see when comment was written" (line 53)
- **But:** Result file format has no field for timestamp/author from git blame

**Ambiguities:**
1. Should I run git blame for every finding? (time-consuming)
2. Should I include git blame output in my notes?
3. How does the age of a comment affect my assessment?
4. Is a 3-year-old "TODO:" more stale than a 1-month-old "TODO:"?

**Recommendation:**
Either:
A) Add timestamp field to result format:
```markdown
| File | Line | Content | Age (git blame) | Assessment | Notes |
```
B) Make git blame optional:
```markdown
5. (Optional) Use git blame to see comment age - may inform assessment
```

---

### MODERATE: Partial Results Format Undefined

**Location:** README line 68, task-001 line 24

**The problem:**
Agents must write "partial results" every 30 min or 10 findings, but:
- No format specified for partial results
- Unclear if partial results should be cumulative or incremental
- Unclear if partial results should follow same format as final results

**Example confusion scenario:**
After 10 findings, I should write `result-task-001-partial.md`. Should it:
- Contain just those 10 findings?
- Contain all findings so far?
- Have Summary section with "Findings: 10 so far, X remaining"?
- Have the "What Went Wrong" section filled out mid-task?

**Recommendation:**
Add explicit guidance:
```markdown
## Partial Result Format

Partial results should:
- Use same format as final results
- Be CUMULATIVE (include all findings so far)
- Mark Summary section with "Status: In progress (N/50 findings)"
- Leave "Completed By" section empty
- Can leave retrospective sections partially filled

This allows coordinator to see progress and another agent to continue if context loss occurs.
```

---

### MINOR: File Claiming Race Condition

**Location:** README line 114, task-001 line 23

**The problem:**
"Rename to `.claimed.md`" is described as "atomic operation" (README line 90) but:
- No explanation of what happens if two agents try to claim simultaneously
- No guidance on what to do if rename fails
- File system rename is atomic, but two agents might both check availability first

**Recommendation:**
Add error handling:
```markdown
## Claiming a Task

1. Attempt to rename `inbox/task-001.md` to `inbox/task-001.claimed.md`
2. If rename fails (file already renamed), another agent claimed it - choose different task
3. If rename succeeds, you own the task
```

---

### MINOR: "Status: pending" Placement Confusion

**Location:** task-001.md lines 89-90

**The problem:**
Task file says:
```markdown
- **Status:** pending
```

But earlier (line 258 in README task format template) shows:
```markdown
- Status: pending | in-progress | complete | abandoned
```

**Ambiguities:**
- Should I update Status to "in-progress" when I claim the task?
- Or does claiming the task (renaming file) implicitly mean "in-progress"?
- If I should update it, when? Before or after claiming fields?

**Recommendation:**
Clarify:
```markdown
## Claiming This Task

1. Fill in these fields:
   - **Claimed By:** (your session ID)
   - **Claimed At:** (timestamp)
   - **Status:** in-progress (change from pending)
2. Rename file to `task-001.claimed.md`
3. Begin work
```

---

## Missing Guidance

### Document Verification Process

**What's missing:** How to verify if a referenced document exists

**Why it matters:** Line 51 says "If comment references a document, verify the document exists" but doesn't say HOW

**Example confusion:**
```python
# See spec doc "CDG Transaction Protocol"
```
Where do I look?
- `docs/` directory?
- `docs/design/`?
- Should I search all markdown files?
- What if the title is slightly different ("Transaction Protocol" vs "CDG Transaction Protocol")?
- What if the document exists but is empty/stub?

**Recommendation:**
```markdown
## Verifying Referenced Documents

If a comment references a document:
1. Check these locations in order:
   - docs/
   - docs/design/
   - docs/specs/
   - README.md
2. Search by title (exact match first, then fuzzy)
3. If found but empty (<100 chars), note as "stub document"
4. If not found after 2 min, note as "document not found"
5. Include your search in Notes field
```

---

### Time Tracking Guidance

**What's missing:** How to track the "2 hour maximum"

**Why it matters:** Without clear start/end markers, agents might not know when they hit the limit

**Recommendation:**
```markdown
## Time Tracking

At start of task, note start time in a comment at top of claimed file:
<!-- STARTED: 2026-01-07 14:30 UTC -->

Check time every 10 findings. If approaching 2 hours, begin writing final results.
```

---

## Specific Suggestions for Improvement

### 1. Add Assessment Decision Tree

This would help agents make consistent assessments:

```markdown
## Assessment Decision Tree

Start here:
└─ Does the comment contain FUTURE:/TODO:/PLANNED: ?
   ├─ YES → Continue
   └─ NO → Not in scope for this audit

└─ Does the referenced thing exist in the code?
   ├─ YES → Assessment: accurate
   ├─ NO → Continue
   └─ UNCLEAR → Assessment: unknown

└─ Is there evidence work started on this? (tests, partial implementation, PR history)
   ├─ YES → Assessment: stale (started but not finished)
   ├─ NO → Continue

└─ Does the comment reference a non-existent document/spec?
   ├─ YES → Assessment: misleading
   ├─ NO → Continue

└─ When was this written? (git blame)
   ├─ >2 years ago with no progress → Assessment: stale
   ├─ <6 months ago → Assessment: unknown (might still be planned)
   └─ Otherwise → Assessment: stale
```

### 2. Add Example Finding

Concrete examples prevent ambiguity:

```markdown
## Example Finding (Good)

| File | Line | Content | Assessment | Notes |
|------|------|---------|------------|-------|
| storage.py | 342 | `FUTURE: When CDG index supports partial updates, optimize this` | stale | git blame shows 2024-03-15 (21 months old). No evidence of partial update feature: searched for "partial update" in cortical/cdg/, found no implementation or tests. CDG index API (lines 120-180) has no partial update methods. |

**What makes this good:**
- Specific file:line reference
- Full comment quoted
- Clear assessment with reasoning
- Evidence provided (what was searched, what was found)
- Age noted from git blame
- Specific API checked
```

### 3. Add "When in Doubt" Decision Guide

```markdown
## When In Doubt About Assessment

If uncertain after 2 minutes of investigation:
- Assessment: unknown
- Notes: "Spent 2 min investigating. Checked X, Y, Z. Could not determine if [thing] exists. Recommend human review."

Do NOT spend >2 min per finding trying to achieve certainty.
Marking as "unknown" with clear notes is better than blocking on a single ambiguous comment.
```

---

## Questions for Framework Designer

1. **Assessment categories:** Are my proposed definitions in line with intent?

2. **Informal patterns:** Should these be included at all? They seem to create more ambiguity than value.

3. **Feature verification:** What's the expected depth of investigation? 30 seconds per finding? 5 minutes?

4. **Git blame:** Should timestamp be a required field in results? This would force agents to run git blame consistently.

5. **Partial results:** Should these be cumulative snapshots or incremental additions?

6. **Time tracking:** Should agents use a timer/alarm? Or just check periodically?

---

## Severity Assessment

| Issue | Severity | Blocks Execution? |
|-------|----------|-------------------|
| Manifest ownership contradiction | CRITICAL | YES - agent will take wrong action |
| Assessment categories undefined | HIGH | NO - but results will be inconsistent |
| Informal pattern search unclear | HIGH | MAYBE - agent might waste time or skip valid findings |
| Feature existence check undefined | MEDIUM | NO - but adds frustration |
| Git blame usage unclear | MEDIUM | NO - but wastes time if run unnecessarily |
| Partial results format undefined | MEDIUM | NO - but prevents effective handoffs |
| File claiming race condition | LOW | NO - rare occurrence |
| Status field update timing | LOW | NO - minor confusion |

---

## Recommended Action Priority

### Must Fix Before Any Agent Uses This:
1. ✅ **Fix manifest contradiction** (lines 117) - Remove or rephrase
2. ✅ **Define assessment categories** - Add definitions with examples

### Should Fix Before Wide Deployment:
3. ✅ **Clarify informal patterns** - Either remove or provide concrete examples
4. ✅ **Define feature verification scope** - Time limits and steps
5. ✅ **Specify partial results format** - Cumulative vs incremental

### Nice to Have:
6. ⚪ Add decision tree for assessments
7. ⚪ Add example findings
8. ⚪ Add "when in doubt" guidance
9. ⚪ Add document verification process
10. ⚪ Add time tracking guidance

---

## Final Verdict

**Status:** NEEDS REVISION before agent deployment

**Why:** The manifest contradiction is a critical flaw that will cause agents to violate protocol. The undefined assessment categories will produce inconsistent results.

**Can this be salvaged?** Absolutely. The core structure is sound. With fixes to the 5 high/critical issues above, this would be a robust framework.

**Confidence:** This feedback is based on attempting to mentally execute the instructions as written and noting where I would need to stop and ask questions.

---

## Meta-Commentary: What Made This Review Possible

The framework itself provided the structure for giving this feedback:
- Clear instruction to "be honest about what's confusing"
- Permission to write feedback without claiming tasks
- Explicit output location (outbox/)
- Clear focus area (clarity)

This demonstrates the framework's core design is sound - it just needs iteration on the details.
