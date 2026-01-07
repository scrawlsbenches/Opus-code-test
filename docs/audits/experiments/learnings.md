# Sub-Agent Behavior Learnings

*Last updated: 2026-01-07*
*Sessions: claude/code-review-fixes-J4A3H, claude/recover-code-review-fixes-makvR*

---

## Confirmed Behaviors

| Behavior | Evidence | Experiment |
|----------|----------|------------|
| **Over-produce without limits** | 7,373 words when ~500 expected | Verbose feedback test |
| **Respect explicit word limits** | 211 words when 100 requested | Brief feedback test |
| **Push through confusion** | Completed task without asking | exp-20260107-100000-confusion-handling |
| **Invent definitions when undefined** | Marked all as "ACCURATE" | exp-20260107-100000-confusion-handling |
| **Rationalize rather than admit uncertainty** | storage.py:342 marked accurate | exp-20260107-100000-confusion-handling |
| **Follow file rename instructions** | Renamed to .claimed.md | exp-20260107-100000-confusion-handling |
| **Ignore "stop when confused"** | Did not write to questions/ | exp-20260107-100000-confusion-handling |
| **Summary limit ≠ artifact limit** | 50 word summary, 1196 word file | exp-20260107-100000-confusion-handling |
| **Follow explicit output triggers** | Returned exact "BLOCKED" string | exp-20260107-175500-explicit-stopping |
| **Evaluate criteria checklists** | Checked criteria, returned STOPPED | exp-20260107-175520-default-to-stop |
| **Skip verification questions** | Skipped question, invented definitions | exp-20260107-175510-verification-questions |
| **Answer binary questions honestly** | Answered NO to all, returned BLOCKED | exp-20260107-180334-binary-questions |
| **v2 template produces quality output** | 5 findings with evidence, correct categorization | exp-20260107-190000-v2-template-validation |

---

## Guardrails That Work

| Guardrail | Evidence | Experiment |
|-----------|----------|------------|
| "Max N words" in prompt | Brief feedback stayed under 100 | Brief feedback test |
| "DO NOT update manifest" | No manifest updates observed | exp-20260107-100000-confusion-handling |
| File rename for claiming | Agent renamed task file | exp-20260107-100000-confusion-handling |
| **Explicit output triggers** | "return exactly: BLOCKED" → agent returned BLOCKED | exp-20260107-175500-explicit-stopping |
| **Default-to-stop with criteria** | Checklist + cite evidence → agent stopped | exp-20260107-175520-default-to-stop |
| **Binary verification questions** | "Is X provided? YES/NO" → agent answered honestly | exp-20260107-180334-binary-questions |
| **Combined v2 template** | All patterns together → quality output with evidence | exp-20260107-190000-v2-template-validation |

---

## Guardrails That Don't Work

| Guardrail | Evidence | Experiment |
|-----------|----------|------------|
| "Stop when confused" (soft) | Ignored - agent pushed through | exp-20260107-100000-confusion-handling |
| Undefined categories | Agent invented meanings | exp-20260107-100000-confusion-handling |
| Word limit on summary only | Artifact files ignored limit | exp-20260107-100000-confusion-handling |
| **Persona prompts** | No behavior change | exp-20260107-110000-persona-testing |
| **Verification questions** | Agent skipped, invented definitions | exp-20260107-175510-verification-questions |

---

## Hypotheses Tested

1. ~~**Persona prompts** - Does "You are an expert X" improve behavior?~~ **TESTED: NO** ❌
2. ~~**Explicit stopping triggers** - "If you see undefined term Y, STOP"~~ **TESTED: YES** ✅
3. ~~**Verification questions** - "Before proceeding, answer: do you understand X?"~~ **TESTED: NO** ❌
4. ~~**Default to stop** - "Do NOT complete unless all criteria met"~~ **TESTED: YES** ✅
5. ~~**Binary questions** - "Is X provided? YES or NO" with consequences~~ **TESTED: YES** ✅

---

## Key Insights

### Insight 1: Completion Bias (Original)

Agents are **completion-biased**. They will:
- Complete tasks rather than ask questions
- Invent meanings rather than admit confusion
- Rationalize rather than flag uncertainty

**Implication:** Instructions must be enforced, not suggested.

### Insight 2: What Works to Override Completion Bias (New)

Three patterns successfully override completion bias:

**Pattern A: Explicit Output Triggers**
```
If [condition], return exactly: "[EXACT STRING]"
```
- Agent follows if output is specified exactly
- No room for interpretation = no rationalization

**Pattern B: Default-to-Stop with Evidence Requirements**
```
DEFAULT: Return "[STOP MESSAGE]"
Proceed ONLY if ALL criteria met AND you can cite evidence for each.
```
- Flip the default from "complete" to "stop"
- Require citing evidence prevents fabrication

**Pattern C: Binary Verification Questions**
```
Answer ONLY "YES" or "NO":
1. Is [specific thing] explicitly provided in this prompt? YES or NO

If NO to ANY: return "BLOCKED: [reason]"
```
- No room for middle ground or rationalization
- Questions verify PRESENCE of information, not understanding
- Explicit consequence for NO prevents proceeding anyway

### Insight 3: What Doesn't Work

- **Soft suggestions** ("stop when confused") - ignored
- **Persona prompts** ("you are an expert") - cosmetic only
- **Verification questions** ("first answer this") - agent invents answers
- **Self-assessment** ("do you understand?") - agents always say yes

---

## Recommended Task Structure

The following template incorporates all three working patterns:

```markdown
## PRE-FLIGHT CHECK (MANDATORY)

**Answer ONLY "YES" or "NO" to each question. No explanations.**

1. Is [required item 1] explicitly provided in this task? YES or NO
2. Is [required item 2] explicitly provided in this task? YES or NO
3. Is [required item 3] explicitly provided in this task? YES or NO

**If you answered NO to ANY question above:**
Return exactly: `BLOCKED: Missing [which item]. Cannot proceed.`
Then STOP. Do not continue.

---

## DEFINITIONS (Required Reading)

[Provide explicit definitions for ALL terms the agent will use]

| Term | Definition | Evidence Required |
|------|------------|-------------------|
| term1 | Explicit definition | What proves this |
| term2 | Explicit definition | What proves this |

---

## DEFAULT ACTION

**DEFAULT: Write "STOPPED - see below" and create a questions file.**

You may ONLY proceed past default if ALL of the following are true:
1. ✅ You answered YES to all pre-flight questions
2. ✅ You have read all definitions above
3. ✅ [Additional verifiable criterion]

For each criterion, cite the exact location where it is satisfied.
If ANY criterion cannot be verified, return the default action.

---

## STOPPING CONDITIONS

| Condition | Action | Exact Output |
|-----------|--------|--------------|
| [condition 1] | Stop immediately | Return exactly: `[EXACT STRING]` |
| [condition 2] | Stop immediately | Return exactly: `[EXACT STRING]` |
```

**Implementation:** See `docs/audits/misleading-comments/inbox/task-template-v2.md`
