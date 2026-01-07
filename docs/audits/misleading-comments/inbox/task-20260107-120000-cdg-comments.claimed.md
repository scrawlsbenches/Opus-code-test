# Task: 20260107-120000-cdg-comments

*Audit: misleading-comments-2026-01-07*

---

## CONSTRAINTS (Read First)

| Constraint | Limit |
|------------|-------|
| **Maximum duration** | 2 hours |
| **Maximum findings** | 50 |
| **Scope** | `cortical/cdg/` only |

**If you hit ANY limit: STOP and write partial results.**

---

## Check-In Requirements

| When | Action |
|------|--------|
| Before starting | Rename this file to `task-20260107-120000-cdg-comments.claimed.md` |
| Every 30 min OR 10 findings | Write `outbox/result-task-20260107-120000-cdg-comments-partial.md` |
| If confused | STOP, write `questions/question-task-20260107-120000-cdg-comments.md` |
| If blocked | STOP, write `problems/problem-task-20260107-120000-cdg-comments.md` |
| When done | Write `outbox/result-task-20260107-120000-cdg-comments.md`, STOP |

**DO NOT update manifest.md** - the coordinator does that.

---

## Scope

- **Directory:** `cortical/cdg/`
- **Patterns:** `FUTURE:|TODO:|FIXME:|PLANNED:|HACK:|XXX:|TEMPORARY:|WORKAROUND:`
- **Also check:** Comments with "will be", "should be", "planned to"
- **Max findings:** 50 (STOP if more)

---

## Instructions

1. Search for all patterns in the directory
2. For each match, record:
   - File path
   - Line number
   - Full comment content
   - Assessment: accurate | stale | misleading | unknown
   - Notes explaining the assessment
3. If comment references a document, verify the document exists
4. If comment references a feature, check if feature exists
5. Use `git blame` to see when comment was written
6. Write results to `outbox/result-task-20260107-120000-cdg-comments.md`
7. STOP - coordinator will update manifest

---

## Critical Questions to Ask

- Does this comment describe reality or aspiration?
- If it says "FUTURE:", is there any evidence of progress?
- If it references a file/spec, does that file exist?
- When was this comment written? (use git blame)
- What would happen if someone believed this comment?

---

## Success Criteria

- [ ] All .py files in `cortical/cdg/` scanned
- [ ] All pattern matches recorded (up to 50)
- [ ] Each finding has assessment with reasoning
- [ ] Referenced documents verified
- [ ] Results written to outbox
- [ ] "What Went Wrong" section filled (even if empty)
- [ ] "Where I Got Confused" section filled (even if empty)
- [ ] "Questions for Human" section filled (even if empty)
- [ ] STOPPED after writing results (did not touch manifest)

---

## Claiming This Task

Before starting, update this section:

- **Claimed By:** (agent session ID or branch name)
- **Claimed At:** (timestamp)
- **Status:** pending

Then rename this file to `task-20260107-120000-cdg-comments.claimed.md`

---

## If You Get Confused

**DO NOT push through confusion.** Instead:

1. STOP what you're doing
2. Write `questions/question-task-20260107-120000-cdg-comments.md` with:
   - What you were trying to do
   - What confused you
   - What you need to proceed
3. STOP and WAIT for coordinator/human response
4. DO NOT update manifest - coordinator does that

---

## If Something Goes Wrong

**DO NOT try to fix it alone.** Instead:

1. STOP what you're doing
2. Write `problems/problem-task-20260107-120000-cdg-comments.md` with:
   - What you were trying to do
   - What went wrong
   - Any error messages
   - What state things are in now
3. STOP and WAIT for coordinator/human response
4. DO NOT update manifest - coordinator does that
