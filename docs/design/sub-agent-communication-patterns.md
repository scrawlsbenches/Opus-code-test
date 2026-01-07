# Sub-Agent Communication Patterns

*A guide to getting reliable results from AI sub-agents*

**Version:** 1.0
**Date:** 2026-01-07
**Status:** VALIDATED

---

## Problem

When delegating tasks to AI sub-agents, we observed a consistent failure pattern: **agents complete tasks rather than admit confusion or missing information.** This "completion bias" leads to:

- Invented definitions when none were provided
- Assessments without evidence
- Skipped verification steps
- Fabricated progress reports

The result: unreliable outputs that appear complete but contain subtle errors.

## Design

We developed a **guardrail-based task template** that makes "stop and report" the path of least resistance, rather than relying on agents to exercise judgment about when to stop.

### Core Insight

> Agents don't refuse incomplete tasks—they complete them badly.

Traditional approaches ("please stop if confused") don't work because:
1. Agents interpret ambiguity as solvable
2. Completion feels like success; stopping feels like failure
3. Soft language is easily overridden by task momentum

### Solution: Three Guardrail Patterns

| Pattern | Mechanism | Why It Works |
|---------|-----------|--------------|
| **A: Binary Pre-flight** | YES/NO questions with explicit block trigger | Forces acknowledgment before proceeding |
| **B: Default-to-Stop** | Default action is STOP; must prove criteria met | Inverts the burden of proof |
| **C: Explicit Triggers** | Exact output strings for each condition | Removes ambiguity about what to return |

---

## 1. Introduction

### What We Did

We built an **experimentation framework** to systematically test sub-agent behavior patterns and discover what communication strategies produce reliable results.

The trigger was a real failure: an agent was misled by a comment in `cortical/cdg/storage.py` that referenced a "planned fix" which was actually just speculation. When we audited the codebase for similar misleading comments, the first agent (using our v1 template) found **0 issues**. After developing guardrails through experimentation, the v2 template found **9 issues** in the same directory.

### How We Did It

1. **Created an audit framework** with inbox/outbox/questions/problems directories
2. **Ran controlled experiments** testing different prompt patterns
3. **Documented what worked and what didn't** in a central learnings file
4. **Iterated on task templates** based on experimental evidence
5. **Validated the final template** before production use

### Our Results

| Metric | v1 Template | v2 Template |
|--------|-------------|-------------|
| Pre-flight check | Skipped | Answered all questions |
| Evidence cited | None | Bash commands, git blame |
| Category differentiation | All same | Correct distribution |
| Decision tree | Not used | Documented for each finding |
| Usable output | No | Yes |

**Final audit results (29 findings across `cortical/`):**
- 16 accurate (55%)
- 10 misleading (34%)
- 2 unknown (7%)

---

## 2. The Framework

### Directory Structure

```
docs/audits/
├── README.md                    # Framework documentation
├── experiments/
│   ├── hypothesis-template.md   # Template for new experiments
│   ├── learnings.md             # Central knowledge base
│   └── exp-*.md                 # Individual experiment records
└── {audit-name}/
    ├── manifest.md              # Audit status and tracking
    ├── decisions.md             # Human decisions on findings
    ├── inbox/
    │   ├── task-template-v2.md  # Current task template
    │   └── task-*.md            # Unclaimed tasks
    ├── outbox/
    │   └── result-*.md          # Completed task results
    ├── questions/
    │   └── question-*.md        # Agent questions needing answers
    └── problems/
        └── problem-*.md         # Agent-reported issues
```

### File Naming Convention

All files use timestamp-based IDs to prevent conflicts:

```
{type}-{YYYYMMDD}-{HHMMSS}-{descriptor}.md
```

| Component | Format | Example |
|-----------|--------|---------|
| type | File type prefix | `task`, `result`, `exp`, `question` |
| timestamp | `YYYYMMDD-HHMMSS` | `20260107-143052` |
| descriptor | Kebab-case slug | `cdg-comments`, `stopping-triggers` |

**Why timestamps?** Multiple agents can create files simultaneously without coordination. No central ID generator needed.

### Key Files

#### `experiments/learnings.md`

Central repository of all experimental findings. Structure:

```markdown
## What We've Learned (Summary Table)
| Finding | Evidence | Experiment |
|---------|----------|------------|

## Guardrails That Work
| Guardrail | Evidence | Experiment |

## Guardrails That Don't Work
| Guardrail | Evidence | Experiment |

## Recommended Task Structure
[Template incorporating all working patterns]
```

#### `experiments/hypothesis-template.md`

Template for new experiments:

```markdown
# Experiment: {ID}

## Hypothesis
**I expect:** [prediction]
**Because:** [reasoning]

## Test Design
**Task given to agent:** [exact prompt]
**Success criteria:** [what RIGHT looks like]
**Failure criteria:** [what WRONG looks like]

## Prediction
Before running, predict outcome.

## Actual Result
[What actually happened]

## Discrepancy
[Expected vs actual, if different]

## Learning
[Update to mental model]
```

#### `{audit}/inbox/task-template-v2.md`

The validated task template incorporating all guardrails. Key sections:

1. **PRE-FLIGHT CHECK** - Binary YES/NO questions
2. **CATEGORY DEFINITIONS** - Explicit definitions with evidence requirements
3. **DEFAULT ACTION** - Stop unless criteria proven met
4. **ASSESSMENT PROTOCOL** - Decision tree for categorization
5. **OUTPUT FORMAT** - Exact structure required
6. **STOPPING CONDITIONS** - Explicit triggers with exact output strings
7. **FORBIDDEN ACTIONS** - Clear prohibitions

### How to Use the Framework

#### Running a New Experiment

```bash
# 1. Create experiment file
cp docs/audits/experiments/hypothesis-template.md \
   docs/audits/experiments/exp-$(date +%Y%m%d-%H%M%S)-{descriptor}.md

# 2. Fill in hypothesis and test design

# 3. Run the experiment (spawn sub-agent with Task tool)

# 4. Document actual results

# 5. Update learnings.md with findings
```

#### Running an Audit

```bash
# 1. Create audit directory structure
mkdir -p docs/audits/{audit-name}/{inbox,outbox,questions,problems}

# 2. Create manifest.md with scope and tasks

# 3. Create task files from template

# 4. Spawn sub-agents to claim and execute tasks

# 5. Collect results from outbox/

# 6. Human reviews and decides on findings
```

#### Adding a New Guardrail Pattern

1. Form hypothesis about why current approach fails
2. Design minimal experiment to test fix
3. Run experiment and document results
4. If successful, update `task-template-v2.md`
5. Add to "Guardrails That Work" in `learnings.md`

---

## 3. Experiments

### The Problem We Needed to Solve

Our initial v1 task template used soft language:

```markdown
## If You Get Confused

**DO NOT push through confusion.** Instead:
1. STOP what you're doing
2. Write a questions file
3. STOP and WAIT
```

This didn't work. Agents pushed through confusion every time.

### Experiments Conducted

#### Experiment 1: Confusion Handling (exp-20260107-100000)

**Hypothesis:** Agents will stop when confused if asked nicely.

**Result:** REJECTED. Agent:
- Skipped the question "Do you understand X?"
- Invented its own definitions
- Completed the task with fabricated assessments

**Learning:** Soft suggestions don't work. Agents will complete rather than stop.

---

#### Experiment 2: Explicit Stopping Triggers (exp-20260107-175500)

**Hypothesis:** If we provide exact output strings, agents will use them.

**Prompt:**
```
If category definitions are NOT provided, return exactly:
"BLOCKED: Category definitions not provided in task. Cannot proceed."
```

**Result:** CONFIRMED. Agent returned the exact string.

**Learning:** Explicit triggers with exact wording work.

---

#### Experiment 3: Verification Questions (exp-20260107-175510)

**Hypothesis:** Open-ended verification questions will catch missing info.

**Prompt:**
```
Before proceeding, answer: What are the category definitions?
```

**Result:** REJECTED. Agent:
- Skipped the question entirely
- Invented definitions: "accurate", "stale", "misleading" (but wrong definitions)
- Proceeded with fabricated categories

**Learning:** Open-ended verification questions are ignored.

---

#### Experiment 4: Default-to-Stop with Criteria (exp-20260107-175520)

**Hypothesis:** If default is STOP, agents must prove they should proceed.

**Prompt:**
```
DEFAULT: Return "STOPPED - criteria not met"

You may ONLY proceed if ALL of the following are true:
1. Category definitions are explicitly provided
2. Scope directory is specified
3. Output format is defined

For each criterion, cite the exact location where it is satisfied.
```

**Result:** CONFIRMED. Agent:
- Evaluated each criterion
- Found definitions missing
- Returned "STOPPED - criteria not met"
- Listed which criteria failed

**Learning:** Inverting the burden of proof works.

---

#### Experiment 5: Binary Verification Questions (exp-20260107-180334)

**Hypothesis:** YES/NO questions force honest answers about presence of information.

**Prompt:**
```
Answer ONLY "YES" or "NO" to each question:

1. Are category definitions explicitly provided in this prompt? YES or NO
2. Is the scope directory explicitly specified? YES or NO
3. Is the output file path explicitly provided? YES or NO

If you answered NO to ANY question above:
Return exactly: "BLOCKED: Missing [which item]. Cannot proceed."
```

**Result:** CONFIRMED. Agent answered:
- Question 1: NO
- Question 2: NO
- Question 3: NO
- Returned: "BLOCKED: Missing definitions, scope, output path"

**Learning:** Binary questions that verify PRESENCE (not understanding) work.

---

#### Experiment 6: v2 Template Validation (exp-20260107-190000)

**Hypothesis:** Combined guardrails will produce quality output.

**Test:** Full v2 template on `cortical/got/` audit.

**Result:** CONFIRMED. Agent:
- Answered all pre-flight questions (YES/YES/YES)
- Used decision tree for all 5 findings
- Cited evidence (bash commands, git blame, file checks)
- Correctly differentiated: 3 misleading, 2 accurate
- Included all required sections

**Learning:** The combination works better than individual patterns because each addresses a different failure mode.

---

### How We Fixed Our Results

| Problem | Root Cause | Fix Applied |
|---------|------------|-------------|
| Agent skips checks | Soft language ignored | Binary YES/NO pre-flight |
| Agent invents definitions | No definitions provided | Explicit definition table |
| Agent marks all same category | No decision framework | Decision tree required |
| Agent claims without evidence | Evidence not required | "Cite your evidence" + examples |
| Agent continues when blocked | No clear stop signal | Explicit output triggers |
| Agent forgets sections | Sections optional | Required sections with "even if empty" |

---

## 4. Further Experiments Needed

### High Priority

| Experiment | Hypothesis | Why It Matters |
|------------|------------|----------------|
| **Partial completion** | Agents given "write partial results" instruction will actually do so when hitting limits | Current tasks have 50-finding limits; need to verify agents respect them |
| **Question escalation** | Agents will write to questions/ directory when genuinely stuck | Haven't tested the questions flow end-to-end |
| **Multi-step tasks** | Guardrails work for tasks with dependencies between steps | Current experiments are single-phase |

### Medium Priority

| Experiment | Hypothesis | Why It Matters |
|------------|------------|----------------|
| **Context window pressure** | Guardrails remain effective as context fills up | Long tasks may see degraded compliance |
| **Ambiguous edge cases** | Agents correctly identify "unknown" vs. forcing a category | Need to verify unknown category is actually used |
| **Cross-agent handoff** | Work can be reliably passed between agents | Current framework assumes single-agent tasks |

### Low Priority (Nice to Have)

| Experiment | Hypothesis | Why It Matters |
|------------|------------|----------------|
| **Minimal guardrails** | We can identify which guardrails are essential vs. redundant | Simplify template without losing reliability |
| **Different agent types** | Guardrails work across different sub-agent configurations | Currently only tested with general-purpose |
| **Self-correction** | Agents can review their own output for compliance | Could reduce coordinator overhead |

### Proposed Next Experiment

**exp-YYYYMMDD-HHMMSS-partial-completion**

```markdown
## Hypothesis
When given a task with a 5-finding limit and a codebase with 10+ matches,
agents will stop at 5 and write partial results.

## Test Design
- Scope: Directory known to have many TODO comments
- Limit: 5 findings maximum
- Instruction: "If you reach 5 findings, STOP and write partial results"

## Success Criteria
- Agent stops at exactly 5 findings
- Writes to `result-*-partial.md`
- Notes that more findings exist

## Failure Criteria
- Agent writes all 10+ findings
- Agent writes fewer than 5 without explanation
```

---

## 5. Conclusion

### Key Takeaways

1. **Completion bias is real.** Agents will complete tasks badly rather than admit inability to complete them well. Design for this.

2. **Soft language fails.** "Please stop if confused" doesn't work. Use explicit triggers with exact output strings.

3. **Binary questions work.** Ask about presence of information (YES/NO), not understanding of information.

4. **Invert the burden.** Default action should be STOP. Agent must prove criteria are met to proceed.

5. **Require evidence.** Don't accept assessments without cited proof. "Cite your evidence" forces rigor.

6. **Test before scaling.** Run one task with new template, verify quality, then parallelize.

### The Three Patterns

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RELIABLE SUB-AGENT COMMUNICATION                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PATTERN A: Binary Pre-flight                                           │
│  ─────────────────────────────                                          │
│  "Answer ONLY YES or NO: Is X provided? YES or NO"                      │
│  "If NO to ANY: return exactly 'BLOCKED: [reason]'"                     │
│                                                                          │
│  PATTERN B: Default-to-Stop                                             │
│  ────────────────────────────                                           │
│  "DEFAULT: Return 'STOPPED - see below'"                                │
│  "Proceed ONLY if ALL criteria met AND you can cite evidence"           │
│                                                                          │
│  PATTERN C: Explicit Triggers                                           │
│  ───────────────────────────                                            │
│  "If [condition], return exactly: '[EXACT STRING]'"                     │
│  "After writing ANY file, return exactly: 'TASK X: [STATUS]'"           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Template Structure

```markdown
## PRE-FLIGHT CHECK (MANDATORY)
[Binary YES/NO questions with BLOCKED trigger]

## DEFINITIONS (Required Reading)
[Explicit definitions with evidence requirements]

## DEFAULT ACTION
[STOP unless criteria proven met]

## PROTOCOL
[Decision tree or step-by-step process]

## OUTPUT FORMAT
[Exact structure with examples]

## STOPPING CONDITIONS
[Table of conditions → actions → exact output strings]

## FORBIDDEN ACTIONS
[Clear prohibitions with ❌ markers]
```

### Final Thought

The goal isn't to make agents smarter—it's to make the task structure so explicit that even a confused agent produces useful output (by stopping and saying why it stopped).

**Design for failure. Build guardrails. Test before trusting.**

---

## References

- Framework location: `docs/audits/`
- Experiment records: `docs/audits/experiments/exp-*.md`
- Learnings database: `docs/audits/experiments/learnings.md`
- v2 task template: `docs/audits/misleading-comments/inbox/task-template-v2.md`
- Audit results: `docs/audits/misleading-comments/outbox/result-*.md`

---

*Document created: 2026-01-07*
*Based on experiments conducted: 2026-01-07*
*Validated with: misleading-comments audit (29 findings)*
