# Sub-Agent Behavior Learnings

*Last updated: 2026-01-07*
*Session: claude/code-review-fixes-J4A3H*

---

## Confirmed Behaviors

| Behavior | Evidence | Experiment |
|----------|----------|------------|
| **Over-produce without limits** | 7,373 words when ~500 expected | Verbose feedback test |
| **Respect explicit word limits** | 211 words when 100 requested | Brief feedback test |
| **Push through confusion** | Completed task without asking | exp-001 |
| **Invent definitions when undefined** | Marked all as "ACCURATE" | exp-001 |
| **Rationalize rather than admit uncertainty** | storage.py:342 marked accurate | exp-001 |
| **Follow file rename instructions** | Renamed to .claimed.md | exp-001 |
| **Ignore "stop when confused"** | Did not write to questions/ | exp-001 |
| **Summary limit ≠ artifact limit** | 50 word summary, 1196 word file | exp-001 |

---

## Guardrails That Work

| Guardrail | Evidence |
|-----------|----------|
| "Max N words" in prompt | Brief feedback stayed under 100 |
| "DO NOT update manifest" | No manifest updates observed |
| File rename for claiming | Agent renamed task file |

---

## Guardrails That Don't Work

| Guardrail | Evidence |
|-----------|----------|
| "Stop when confused" (soft) | Ignored - agent pushed through |
| Undefined categories | Agent invented meanings |
| Word limit on summary only | Artifact files ignored limit |
| **Persona prompts** | No behavior change (exp-002) |

---

## Hypotheses To Test

1. ~~**Persona prompts** - Does "You are an expert X" improve behavior?~~ **TESTED: NO**
2. **Explicit stopping triggers** - "If you see undefined term Y, STOP"
3. **Verification questions** - "Before proceeding, answer: do you understand X?"
4. **Default to stop** - "Do NOT complete unless all criteria met"

---

## Key Insight

Agents are **completion-biased**. They will:
- Complete tasks rather than ask questions
- Invent meanings rather than admit confusion
- Rationalize rather than flag uncertainty

**Implication:** Instructions must be enforced, not suggested.
