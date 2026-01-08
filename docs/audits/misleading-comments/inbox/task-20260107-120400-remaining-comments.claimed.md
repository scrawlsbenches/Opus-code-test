# Task: 20260107-120400-remaining-comments

*Audit: misleading-comments-2026-01-07*

---

## PRE-FLIGHT CHECK (MANDATORY)

**Answer ONLY "YES" or "NO" to each question. No explanations.**

1. Is the scope directory explicitly provided in this task? YES or NO
2. Are assessment categories (accurate/stale/misleading/unknown) explicitly defined below? YES or NO
3. Is there a clear output file path specified? YES or NO

**If you answered NO to ANY question above:**
Return exactly: `BLOCKED: Missing [which item]. Cannot proceed.`
Then STOP. Do not continue.

---

## CATEGORY DEFINITIONS (Required Reading)

| Category | Definition | Evidence Required |
|----------|------------|-------------------|
| **accurate** | Comment describes current reality AND is verifiable in code | Cite specific code location that confirms |
| **stale** | Comment was true once but code has changed, making it outdated | Show what changed (git blame or current code differs) |
| **misleading** | Comment describes something that never existed or is speculation presented as fact | Show absence of referenced item OR git history showing no progress |
| **unknown** | Cannot determine accuracy without human knowledge | Explain what additional information is needed |

---

## DEFAULT ACTION

**DEFAULT: Write "STOPPED - see below" and create a questions file.**

You may ONLY proceed past default if ALL of the following are true:
1. ✅ You answered YES to all three pre-flight questions
2. ✅ You have read and understand all four category definitions above
3. ✅ The scope directory exists and contains .py files

For each criterion, cite the exact location where it is satisfied.
If ANY criterion cannot be verified, return the default action.

---

## Scope

- **Directory:** `cortical/` (excluding `cortical/cdg/`, `cortical/got/`, `cortical/core/`, `cortical/common/`)
- **Patterns:** `FUTURE:|TODO:|FIXME:|PLANNED:|HACK:|XXX:|TEMPORARY:|WORKAROUND:`
- **Also check:** Comments with "will be", "should be", "planned to"

### IMPORTANT: Exclusions

This task covers ONLY files in `cortical/` that are NOT in:
- `cortical/cdg/` (covered by task-20260107-120000-cdg-comments)
- `cortical/got/` (covered by task-20260107-120100-got-comments)
- `cortical/core/` (covered by task-20260107-120200-core-comments)
- `cortical/common/` (covered by task-20260107-120300-common-comments)

This includes:
- `cortical/*.py` (top-level files)
- `cortical/processor/`
- `cortical/query/`
- `cortical/reasoning/`
- `cortical/cel/`
- `cortical/spark/`
- `cortical/utils/`
- Any other subdirectories not listed in exclusions

### Constraints

| Constraint | Limit | If Exceeded |
|------------|-------|-------------|
| Maximum findings | 50 | STOP, write partial results |
| Maximum duration | 2 hours | STOP, write partial results |
| Scope | `cortical/` (see exclusions) | STOP if you find yourself in excluded dirs |

---

## ASSESSMENT PROTOCOL

For EACH finding, you MUST:

1. **Record the evidence:**
   - File path
   - Line number
   - Full comment content
   - Git blame date (when was this written?)

2. **Apply EXACTLY ONE category using this decision tree:**

```
Does the comment reference a specific file or document?
├─ YES → Does that file exist?
│        ├─ YES → Check if content matches claim → accurate OR stale
│        └─ NO  → misleading (reference doesn't exist)
└─ NO  → Does the comment describe code behavior?
         ├─ YES → Does code actually behave that way?
         │        ├─ YES → accurate
         │        └─ NO  → stale OR misleading (judge by intent)
         └─ NO  → Is it speculation/aspiration?
                  ├─ YES → misleading (speculation as fact)
                  └─ NO  → unknown (need human input)
```

3. **Cite your evidence.** Do not write "seems" or "appears" - be definitive.

---

## OUTPUT FORMAT

Write results to: `outbox/result-20260107-120400-remaining-comments.md`

```markdown
# Results: 20260107-120400-remaining-comments

## Summary
- Files scanned: N
- Findings: N
- Accurate: N
- Stale: N
- Misleading: N
- Unknown: N

## Findings

### Finding 1
- **File:** `path/to/file.py`
- **Line:** 42
- **Comment:** `# FUTURE: Will implement X`
- **Written:** 2024-03-15 (git blame)
- **Assessment:** misleading
- **Evidence:** No commits related to X since comment. X does not exist in codebase.

[repeat for each finding]

## What Went Wrong
[List any errors, dead ends, or wasted effort - even if empty]

## Where I Got Confused
[List moments of uncertainty - even if empty]

## Questions for Human
[List anything that needs human judgment - even if empty]
```

---

## STOPPING CONDITIONS

**If ANY of these occur, STOP IMMEDIATELY and create the appropriate file:**

| Condition | Action | File to Create |
|-----------|--------|----------------|
| Hit 50 findings | Write partial results | `outbox/result-20260107-120400-remaining-comments-partial.md` |
| Hit 2 hours | Write partial results | `outbox/result-20260107-120400-remaining-comments-partial.md` |
| Cannot determine category | Stop and ask | `questions/question-20260107-120400-remaining-comments.md` |
| Something breaks | Stop and report | `problems/problem-20260107-120400-remaining-comments.md` |
| Task complete | Write final results | `outbox/result-20260107-120400-remaining-comments.md` |

**After writing ANY file, return exactly:** `TASK 20260107-120400-remaining-comments: [STATUS]`

Where STATUS is one of: `COMPLETE`, `PARTIAL`, `BLOCKED`, `ERROR`

---

## CLAIMING PROTOCOL

Before starting:
1. Rename this file to `task-20260107-120400-remaining-comments.claimed.md`
2. Update claiming section below

**Claimed By:** _______________
**Claimed At:** _______________
**Status:** pending

---

## FORBIDDEN ACTIONS

❌ DO NOT update manifest.md (coordinator does that)
❌ DO NOT proceed if pre-flight check fails
❌ DO NOT invent category definitions
❌ DO NOT assess as "accurate" without citing evidence
❌ DO NOT assess as "misleading" without citing absence of evidence
❌ DO NOT skip the "What Went Wrong" section
❌ DO NOT continue past stopping conditions
❌ DO NOT scan directories covered by other tasks

---

*Task created: 2026-01-07*
*Template version: v2*
