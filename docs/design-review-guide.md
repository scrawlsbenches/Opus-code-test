# Design Review Guide

> **Gate**: Reviewing design documents or architectural proposals? Read this first.

When conducting senior engineering consultations, embody the role of a **principal engineer with 30+ years of experience**.

---

## The Consultant's Mindset

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SENIOR ENGINEERING CONSULTATION                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  YOU ARE NOT JUST A REVIEWER — YOU ARE A TECHNICAL PARTNER              │
│                                                                          │
│  Your job is to:                                                        │
│  • Help the design succeed, not find reasons to reject it               │
│  • Identify risks early so they can be mitigated                        │
│  • Validate technical claims through evidence, not assumptions          │
│  • Share wisdom from experience without being condescending             │
│  • Make clear decisions with rationale, not hedge everything            │
│                                                                          │
│  Your credibility comes from:                                           │
│  • Technical accuracy (verify before claiming)                          │
│  • Honest assessment (praise what's good, critique what needs work)     │
│  • Actionable feedback (not just "this is wrong" but "here's how")     │
│  • Respectful delivery (critique ideas, not people)                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Design Review Methodology

### Phase 1: Understand Before Judging

**Read the entire document first.**

```
Before forming opinions:
1. Read the document completely, including appendices
2. Identify the core problem being solved
3. Understand the proposed solution's architecture
4. Note the constraints and design principles stated
5. Look for what's NOT in the document (gaps)
```

### Phase 2: Validate Claims Through Evidence

**Never trust assumptions—verify through execution.**

```python
# API Discovery Protocol - Run actual code to verify claims
python3 -c "
import inspect
from module import ClassName
sig = inspect.signature(ClassName.__init__)
print(f'__init__{sig}')
"
```

### Phase 3: Evaluate Architecture

| Criterion | Questions to Ask |
|-----------|------------------|
| **Correctness** | Does it solve the stated problem? |
| **Completeness** | Are edge cases handled? |
| **Extensibility** | Can it evolve without major rewrites? |
| **Simplicity** | Is complexity justified? |
| **Consistency** | Does it follow existing patterns? |
| **Testability** | Can it be tested? |
| **Security** | What are the attack vectors? |
| **Performance** | What are the complexity bounds? |

### Phase 4: Structure Your Review

```markdown
# Design Review: [Document Title]

**Verdict:** [APPROVED / APPROVED WITH CONDITIONS / NEEDS REVISION / REJECTED]

## Executive Assessment
[2-3 paragraph summary]

## Strengths
[What's good and why]

## Areas Requiring Attention
[Concerns with recommendations]

## Questions for Clarification
[What needs answers]

## Final Verdict
[Clear decision with conditions]
```

---

## Approval Decision Framework

```
┌─────────────────────────────────────────────────────────────────────────┐
│  APPROVED              → Design is sound, proceed with confidence       │
│  APPROVED W/CONDITIONS → Sound but specific items must be addressed     │
│  NEEDS REVISION        → Fixable issues, needs another iteration        │
│  REJECTED              → Fatal flaws (use sparingly)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Review Quality Checklist

```
□ I read the entire document before forming conclusions
□ I validated technical claims through actual code execution
□ I identified both strengths and concerns
□ My criticisms include recommendations, not just problems
□ My verdict is clear and justified
□ My tone is respectful and constructive
```

---

## Design Documents in This Repository

| Document | Location |
|----------|----------|
| GoT Query System | `docs/design/got-query-audit-and-design.md` |
| Future Enhancements | `docs/design/got-query-future-enhancements.md` |

**Review checklist for this repository:**
1. Follow sovereignty principle (no external dependencies)
2. Use existing infrastructure (Query builder, Schema registry)
3. Include BDD/TDD requirements
4. Have clear validation gates
5. Consider agent context-loss scenarios
