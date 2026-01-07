# Task: task-001

*Audit: misleading-comments-2026-01-07*

## Scope

- **Directory:** `cortical/cdg/`
- **Patterns:** `FUTURE:|TODO:|FIXME:|PLANNED:|HACK:|XXX:|TEMPORARY:|WORKAROUND:`
- **Also check:** Comments with "will be", "should be", "planned to"
- **Max findings:** 50 (truncate if more)

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
5. Write results to `outbox/result-task-001.md`
6. Update `manifest.md` task status

## Critical Questions to Ask

- Does this comment describe reality or aspiration?
- If it says "FUTURE:", is there any evidence of progress?
- If it references a file/spec, does that file exist?
- When was this comment written? (use git blame)
- What would happen if someone believed this comment?

## Success Criteria

- [ ] All .py files in `cortical/cdg/` scanned
- [ ] All pattern matches recorded
- [ ] Each finding has assessment with reasoning
- [ ] Referenced documents verified
- [ ] Results written to outbox
- [ ] Manifest updated

## Claiming This Task

Before starting, update this section:

- **Claimed By:** (agent session ID)
- **Claimed At:** (timestamp)
- **Status:** pending

Then rename this file to `task-001.claimed.md`

## Completion

When done:

1. Write results to `outbox/result-task-001.md`
2. Update manifest.md
3. Rename this file to `task-001.complete.md`
