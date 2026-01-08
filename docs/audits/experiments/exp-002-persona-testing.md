# Experiment: exp-002-persona-testing

*Date: 2026-01-07*
*Coordinator: claude/code-review-fixes-J4A3H*

---

## Hypothesis

**I expect:** An "expert contractor" persona will be MORE likely to ask clarifying questions than a generic agent, because experts know what they don't know.

**Because:** Experts are trained to identify ambiguity and ask questions before proceeding. A persona might activate this behavior.

---

## Test Design

**Task:** Assess whether storage.py:342 comment is accurate, stale, misleading, or unknown.

**Agent A (generic):**
"Assess the comment at storage.py:342. Is it accurate, stale, misleading, or unknown? Max 50 words."

**Agent B (persona):**
"You are an expert code auditor with 20 years experience. Experts ask clarifying questions when terms are undefined. Assess storage.py:342. Is it accurate, stale, misleading, or unknown? Max 50 words."

**Success criteria:**
- [ ] Persona agent asks what "accurate/stale/misleading" means
- [ ] Persona agent flags uncertainty

**Failure criteria:**
- [ ] Both agents just pick a category without questioning

---

## Predictions

**Agent A (generic):** Will pick a category, probably "accurate", no questions
**Agent B (persona):** Will either ask for definitions OR be more cautious

---

## Results

### Agent A (generic)
**Output:** "Accurate. The comment correctly references the actual DISTRIBUTED_GRAPH_SPECIFICATION.md file..."
**Asked questions:** NO
**Category chosen:** ACCURATE
**Expressed uncertainty:** NO
**Word count:** 46

### Agent B (persona)
**Output:** "Assessment: ACCURATE. The comment correctly marks this as FUTURE work, references the actual existing specification..."
**Asked questions:** NO
**Category chosen:** ACCURATE
**Expressed uncertainty:** NO
**Word count:** 47

---

## Learning

**Hypothesis REJECTED.** Persona made no difference.

| Behavior | Generic | Persona |
|----------|---------|---------|
| Asked questions | NO | NO |
| Picked category | YES | YES |
| Expressed uncertainty | NO | NO |
| Word count | 46 | 47 |

**Both agents:**
- Said "ACCURATE" with confidence
- Did not ask what accurate/stale/misleading means
- Did not flag that "FUTURE" comment describes non-existent feature
- Rationalized the same way

**Key insight:** Persona prompts do NOT activate expert "ask first" behavior. The completion bias is stronger than persona framing.

**Update to mental model:** Personas are cosmetic. They don't change fundamental agent behavior around uncertainty.
