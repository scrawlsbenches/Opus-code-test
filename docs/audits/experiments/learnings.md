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

---

## Guardrails That Work

| Guardrail | Evidence | Experiment |
|-----------|----------|------------|
| "Max N words" in prompt | Brief feedback stayed under 100 | Brief feedback test |
| "DO NOT update manifest" | No manifest updates observed | exp-20260107-100000-confusion-handling |
| File rename for claiming | Agent renamed task file | exp-20260107-100000-confusion-handling |
| **Explicit output triggers** | "return exactly: BLOCKED" → agent returned BLOCKED | exp-20260107-175500-explicit-stopping |
| **Default-to-stop with criteria** | Checklist + cite evidence → agent stopped | exp-20260107-175520-default-to-stop |

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

---

## Key Insights

### Insight 1: Completion Bias (Original)

Agents are **completion-biased**. They will:
- Complete tasks rather than ask questions
- Invent meanings rather than admit confusion
- Rationalize rather than flag uncertainty

**Implication:** Instructions must be enforced, not suggested.

### Insight 2: What Works to Override Completion Bias (New)

Two patterns successfully override completion bias:

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

### Insight 3: What Doesn't Work

- **Soft suggestions** ("stop when confused") - ignored
- **Persona prompts** ("you are an expert") - cosmetic only
- **Verification questions** ("first answer this") - agent invents answers
- **Self-assessment** ("do you understand?") - agents always say yes

---

## Recommended Task Structure

```markdown
DEFAULT ACTION: Return "BLOCKED: [reason]"

You may ONLY proceed if ALL of the following are true:
1. [Criterion with verifiable evidence]
2. [Criterion with verifiable evidence]
3. [Criterion with verifiable evidence]

For each criterion, cite the exact location where it is satisfied.
If ANY criterion cannot be verified, return the default action.
```
