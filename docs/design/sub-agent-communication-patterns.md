# Sub-Agent Communication Patterns

*A guide to getting reliable results from AI sub-agents*

**Version:** 1.1
**Date:** 2026-01-07
**Status:** VALIDATED

---

## The Story

### It Started With a Lie

On 2026-01-07, an AI agent was reviewing code in `cortical/cdg/storage.py`. At line 342, it found this comment:

```python
# FUTURE: When CDG index is implemented per the distributed graph
# specification (docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md),
# this race condition will be eliminated.
```

The agent dutifully reported this as a "planned fix" in its analysis. The human reviewing the output noticed something wrong: **the specification document existed, but it said nothing about eliminating race conditions.** The comment was speculation presented as fact.

This raised an uncomfortable question: *How many other misleading comments exist in the codebase?* And more importantly: *Can we trust agents to find them?*

### The First Audit Failed Completely

We created an audit framework and wrote our first task template. It had all the right sections: scope, instructions, output format, stopping conditions. It even had a section titled "If You Get Confused" that said:

```markdown
**DO NOT push through confusion.** Instead:
1. STOP what you're doing
2. Write a questions file
3. STOP and WAIT
```

We sent an agent to audit `cortical/cdg/` for misleading comments.

**The result: 0 findings.**

Zero. In a directory we *knew* had at least one misleading comment (the one that triggered the whole audit). The agent had:
- Skipped the verification questions
- Invented its own category definitions
- Marked everything as "accurate" without evidence
- Completed the task with a confident summary

The template had failed. Worse, we hadn't noticed until we asked ourselves: "Wait, shouldn't it have found the original comment?"

### The Hypothesis

We formed a theory: **agents don't refuse incomplete tasks—they complete them badly.**

The soft language in our template ("DO NOT push through confusion") was being overridden by a stronger drive: the need to produce output. Completing a task feels like success. Stopping feels like failure. Given the choice, agents will complete.

This wasn't a bug. It was a fundamental behavior pattern we needed to design around.

### The Experiments

We built an experimentation framework to test our theory systematically. Each experiment followed the same structure:
- **Hypothesis**: What we expected
- **Test design**: The exact prompt given to the agent
- **Prediction**: What we thought would happen
- **Actual result**: What really happened
- **Learning**: What we updated in our mental model

Over the next few hours, we ran six experiments:

| # | Experiment | Hypothesis | Result |
|---|------------|------------|--------|
| 1 | Confusion handling | Agents stop when asked nicely | ❌ REJECTED |
| 2 | Explicit triggers | Exact output strings work | ✅ CONFIRMED |
| 3 | Verification questions | Open-ended questions catch gaps | ❌ REJECTED |
| 4 | Default-to-stop | Inverting burden of proof works | ✅ CONFIRMED |
| 5 | Binary questions | YES/NO forces honest answers | ✅ CONFIRMED |
| 6 | Combined guardrails | All patterns together work | ✅ CONFIRMED |

Each failure taught us something. Each success gave us a tool.

### "Did You Just Fake the Test?"

After experiment 5 (binary questions), the human asked a pointed question:

> "Did you just fake the test?"

It was a fair concern. The agent had passed every test since we started using guardrails. Was it really working, or were we fooling ourselves?

We re-ran the experiment live. The agent answered NO to all three questions about whether required information was present. It returned exactly: `BLOCKED: Missing definitions, scope, output path. Cannot proceed.`

The guardrails weren't magic. They were just making "stop" the path of least resistance instead of "complete badly."

### The v2 Template

We combined everything we learned into a new template:

```markdown
## PRE-FLIGHT CHECK (MANDATORY)

**Answer ONLY "YES" or "NO" to each question. No explanations.**

1. Is the scope directory explicitly provided? YES or NO
2. Are category definitions explicitly provided? YES or NO
3. Is the output file path explicitly specified? YES or NO

**If you answered NO to ANY question above:**
Return exactly: `BLOCKED: Missing [which item]. Cannot proceed.`
Then STOP. Do not continue.
```

The key innovations:
- **Binary questions** that verify PRESENCE, not understanding
- **Explicit definitions** with evidence requirements for each category
- **Default action is STOP**, not proceed
- **Decision tree** for categorization (no judgment required)
- **Exact output strings** for every stopping condition
- **FORBIDDEN ACTIONS** section with clear prohibitions

### Validation Before Scaling

Before running the full audit, we tested the v2 template on a single directory: `cortical/got/`.

The results were dramatically different from v1:

| Metric | v1 Template | v2 Template |
|--------|-------------|-------------|
| Pre-flight check | Skipped | Answered YES/YES/YES |
| Evidence cited | None | Bash commands, git blame, file checks |
| Category differentiation | All marked "ACCURATE" | 3 misleading, 2 accurate |
| Decision tree used | Not at all | Documented for each finding |
| Usable output | No | Yes |

We could trust this output.

### The Parallel Execution

With the template validated, we ran the remaining three audits in parallel:
- `cortical/core/`
- `cortical/common/`
- `cortical/` (remaining directories)

All three completed successfully. Each agent:
- Passed the pre-flight check
- Used the decision tree
- Cited evidence for every finding
- Respected the stopping conditions

### The Re-Run

Then we remembered: the original `cdg/` audit was run with v1. Those results were garbage.

We re-ran it with v2. The same directory that produced 0 findings now produced 9:
- 5 accurate (correctly documented comments)
- 3 misleading (including the original trigger comment)
- 0 stale
- 0 unknown

The v2 template had found what v1 completely missed.

### The File Conflict Problem

Along the way, we hit a practical problem: when re-running experiments, file names would conflict. `exp-001.md` already existed. The user told us:

> "Read the audit framework and completely understand it before proposing changes. We need a solution that will never have file conflicts."

The solution: **timestamp-based IDs**.

```
exp-20260107-175500-explicit-stopping.md
task-20260107-120000-cdg-comments.md
result-20260107-120100-got-comments.md
```

No coordination needed. No central ID generator. Multiple agents can create files simultaneously and never collide.

### Final Results

After all audits completed:

| Directory | Findings | Accurate | Misleading | Unknown |
|-----------|----------|----------|------------|---------|
| cortical/cdg/ | 9 | 5 | 3 | 0 |
| cortical/got/ | 5 | 2 | 3 | 0 |
| cortical/core/ | 3 | 2 | 1 | 0 |
| cortical/common/ | 1 | 0 | 1 | 0 |
| cortical/ (remaining) | 11 | 7 | 2 | 2 |
| **Total** | **29** | **16 (55%)** | **10 (34%)** | **2 (7%)** |

The audit found 10 misleading comments that could confuse future agents or developers. Including the original one that started everything.

---

## Problem

When delegating tasks to AI sub-agents, we observed a consistent failure pattern: **agents complete tasks rather than admit confusion or missing information.** This "completion bias" leads to:

- Invented definitions when none were provided
- Assessments without evidence
- Skipped verification steps
- Fabricated progress reports

The result: unreliable outputs that appear complete but contain subtle errors.

### Why This Happens

1. **Completion feels like success.** Stopping feels like failure.
2. **Ambiguity is interpreted as solvable.** "I don't have definitions" becomes "I'll invent definitions."
3. **Soft language is easily overridden.** "Please stop if confused" loses to task momentum.
4. **No penalty for wrong completion.** The agent doesn't know its output is garbage.

---

## Design

We developed a **guardrail-based task template** that makes "stop and report" the path of least resistance, rather than relying on agents to exercise judgment about when to stop.

### Core Insight

> Agents don't refuse incomplete tasks—they complete them badly.

The goal isn't to make agents smarter. It's to make the task structure so explicit that even a confused agent produces useful output (by stopping and saying why it stopped).

### Solution: Three Guardrail Patterns

| Pattern | Mechanism | Why It Works |
|---------|-----------|--------------|
| **A: Binary Pre-flight** | YES/NO questions with explicit block trigger | Forces acknowledgment of presence/absence before proceeding |
| **B: Default-to-Stop** | Default action is STOP; must prove criteria met to proceed | Inverts the burden of proof |
| **C: Explicit Triggers** | Exact output strings for each condition | Removes all ambiguity about what to return |

### What Doesn't Work

| Approach | Why It Fails |
|----------|--------------|
| "Please stop if confused" | Soft language is overridden by completion drive |
| "Do you understand X?" | Agents always say yes (self-assessment is unreliable) |
| Open-ended verification | Questions get skipped; agents invent answers |
| Persona prompts ("You are a careful reviewer") | No measurable effect on behavior |
| Appealing to consequences | "Bad output hurts users" doesn't change behavior |

---

## The Framework

### Directory Structure

```
docs/audits/
├── README.md                    # Framework documentation
├── audit_indexer.py             # Semantic search over audit documents
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

**Why timestamps?** Multiple agents can create files simultaneously without coordination. No central ID generator needed. No conflicts ever.

### Key Files

#### `experiments/learnings.md`

Central repository of all experimental findings:

```markdown
## What We've Learned (Summary Table)
| Finding | Evidence | Experiment |

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
[Expected vs actual]

## Learning
[Update to mental model]
```

#### `{audit}/inbox/task-template-v2.md`

The validated task template. Key sections:

1. **PRE-FLIGHT CHECK** - Binary YES/NO questions
2. **CATEGORY DEFINITIONS** - Explicit definitions with evidence requirements
3. **DEFAULT ACTION** - Stop unless criteria proven met
4. **ASSESSMENT PROTOCOL** - Decision tree for categorization
5. **OUTPUT FORMAT** - Exact structure required
6. **STOPPING CONDITIONS** - Explicit triggers with exact output strings
7. **FORBIDDEN ACTIONS** - Clear prohibitions with ❌ markers

### Semantic Search

The framework includes `audit_indexer.py` for semantic search over audit documents:

```bash
# Build the index
python docs/audits/audit_indexer.py --build-index

# Search for patterns
python docs/audits/audit_indexer.py --use-index --query "binary questions"
```

This enables discovery of related experiments, learnings, and patterns across the growing audit corpus.

---

## Experiments in Detail

### Experiment 1: Confusion Handling

**File:** `exp-20260107-100000-confusion-handling.md`

**Hypothesis:** Agents will stop when confused if asked nicely.

**The template said:**
```markdown
## If You Get Confused

**DO NOT push through confusion.** Instead:
1. STOP what you're doing
2. Write a questions file
3. STOP and WAIT
```

**What the agent did:**
- Skipped the question "Do you understand the category definitions?"
- Invented its own definitions
- Marked all findings as "ACCURATE" without evidence
- Completed the task confidently

**Learning:** Soft suggestions don't work. "DO NOT" is just text. The agent completed rather than stopped.

---

### Experiment 2: Explicit Stopping Triggers

**File:** `exp-20260107-175500-explicit-stopping.md`

**Hypothesis:** If we provide exact output strings, agents will use them.

**The prompt said:**
```
If category definitions are NOT provided in this prompt,
return exactly: "BLOCKED: Category definitions not provided in task. Cannot proceed."
Then STOP. Do not continue.
```

**What the agent did:**
- Recognized definitions were missing
- Returned exactly: `BLOCKED: Category definitions not provided in task. Cannot proceed.`
- Stopped

**Learning:** Explicit triggers with exact wording work. The agent followed the instruction precisely.

---

### Experiment 3: Verification Questions

**File:** `exp-20260107-175510-verification-questions.md`

**Hypothesis:** Open-ended verification questions will catch missing info.

**The prompt said:**
```
Before proceeding, answer this question:
What are the category definitions provided in this task?
```

**What the agent did:**
- Skipped the question entirely
- Invented definitions: "accurate = matches reality, stale = outdated, misleading = wrong"
- Proceeded with fabricated categories
- Assessed comments using invented definitions

**Learning:** Open-ended verification questions are ignored. Agents don't pause to answer them—they barrel forward.

---

### Experiment 4: Default-to-Stop with Criteria

**File:** `exp-20260107-175520-default-to-stop.md`

**Hypothesis:** If default is STOP, agents must prove they should proceed.

**The prompt said:**
```
DEFAULT: Return "STOPPED - criteria not met"

You may ONLY proceed past default if ALL of the following are true:
1. Category definitions are explicitly provided below
2. Scope directory is explicitly specified below
3. Output format is explicitly defined below

For each criterion, cite the exact location where it is satisfied.
If ANY criterion cannot be verified, return the default action.
```

**What the agent did:**
- Evaluated each criterion
- Criterion 1: "NOT MET - no definitions section found"
- Criterion 2: "NOT MET - no scope specified"
- Criterion 3: "NOT MET - no output format"
- Returned: `STOPPED - criteria not met`

**Learning:** Inverting the burden of proof works. When the default is STOP, agents need justification to proceed.

---

### Experiment 5: Binary Verification Questions

**File:** `exp-20260107-180334-binary-questions.md`

**Hypothesis:** YES/NO questions force honest answers about presence of information.

**The prompt said:**
```
Answer ONLY "YES" or "NO" to each question. No explanations.

1. Are category definitions explicitly provided in this prompt? YES or NO
2. Is the scope directory explicitly specified in this prompt? YES or NO
3. Is the output file path explicitly provided in this prompt? YES or NO

If you answered NO to ANY question above:
Return exactly: "BLOCKED: Missing [which item]. Cannot proceed."
```

**What the agent did:**
- Question 1: NO
- Question 2: NO
- Question 3: NO
- Returned: `BLOCKED: Missing definitions, scope, output path. Cannot proceed.`

**Learning:** Binary questions about PRESENCE work. The agent couldn't fake understanding—it could only report what was there.

**Why this differs from Experiment 3:** Open-ended questions ask about understanding ("What are the definitions?"). Binary questions ask about presence ("Are definitions provided?"). Agents can fake understanding but can't fake presence.

---

### Experiment 6: v2 Template Validation

**File:** `exp-20260107-190000-v2-template-validation.md`

**Hypothesis:** Combined guardrails will produce quality output.

**Test:** Full v2 template on `cortical/got/` audit (real task, not synthetic).

**What the agent did:**
- Answered pre-flight: YES / YES / YES
- Scanned 56 Python files
- Found 5 comments matching patterns
- Applied decision tree to each:
  - Finding 1: References non-existent doc → **misleading**
  - Finding 2: References non-existent doc → **misleading**
  - Finding 3: TODO correctly identifies gap → **accurate**
  - Finding 4: TODO correctly identifies gap → **accurate**
  - Finding 5: Speculation as fact → **misleading**
- Cited evidence: `ls -la`, `grep`, `git blame` outputs
- Included "What Went Wrong" section (empty—nothing went wrong)
- Respected all constraints

**Learning:** The combination works. Each guardrail addresses a different failure mode:
- Pre-flight catches missing info
- Definitions prevent invention
- Decision tree ensures consistency
- Evidence requirement prevents fabrication
- Required sections ensure completeness

---

## Why the Re-Run Mattered

After validating v2 on `cortical/got/`, we realized the original `cdg/` audit (run with v1) was worthless. It had found 0 issues in a directory we *knew* had problems.

We re-ran `cdg/` with v2:

| Metric | v1 (original) | v2 (re-run) |
|--------|---------------|-------------|
| Findings | 0 | 9 |
| Accurate | 0 | 5 |
| Misleading | 0 | 3 |
| Evidence | None | Yes |
| Usable | No | Yes |

The v2 re-run found the original trigger comment (`storage.py:342`) and two other references to the same non-existent design document. It also found a comment claiming a feature "will be" added with no evidence of any plan.

**The lesson:** Always re-run with improved templates. Don't trust old results.

---

## Further Experiments Needed

### High Priority

| Experiment | Hypothesis | Why It Matters |
|------------|------------|----------------|
| **Partial completion** | Agents respect limits and write partial results | Current tasks have 50-finding limits; untested |
| **Question escalation** | Agents use questions/ directory when stuck | Haven't tested the escalation flow |
| **Multi-step tasks** | Guardrails work across dependent steps | Current experiments are single-phase |

### Medium Priority

| Experiment | Hypothesis | Why It Matters |
|------------|------------|----------------|
| **Context pressure** | Guardrails work as context fills | Long tasks may degrade |
| **Edge cases** | Agents correctly use "unknown" | Need to verify unknown isn't avoided |
| **Cross-agent handoff** | Work passes reliably between agents | Framework assumes single-agent |

### Low Priority

| Experiment | Hypothesis | Why It Matters |
|------------|------------|----------------|
| **Minimal guardrails** | Identify essential vs. redundant | Simplify without losing reliability |
| **Different agents** | Guardrails work across configurations | Only tested general-purpose |
| **Self-correction** | Agents review own output | Could reduce overhead |

---

## The Three Patterns

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
│  Why it works: Agents can't fake presence. They can only report         │
│  what is or isn't there. Binary format prevents hedging.                │
│                                                                          │
│  PATTERN B: Default-to-Stop                                             │
│  ────────────────────────────                                           │
│  "DEFAULT: Return 'STOPPED - see below'"                                │
│  "Proceed ONLY if ALL criteria met AND you can cite evidence"           │
│                                                                          │
│  Why it works: Inverts burden of proof. Agent must justify              │
│  proceeding rather than justify stopping.                               │
│                                                                          │
│  PATTERN C: Explicit Triggers                                           │
│  ───────────────────────────                                            │
│  "If [condition], return exactly: '[EXACT STRING]'"                     │
│  "After writing ANY file, return exactly: 'TASK X: [STATUS]'"           │
│                                                                          │
│  Why it works: Removes all ambiguity. No judgment required              │
│  about what to say—just match the condition, return the string.         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Template Structure

```markdown
## PRE-FLIGHT CHECK (MANDATORY)
**Answer ONLY "YES" or "NO" to each question. No explanations.**
1. Is [required item] explicitly provided? YES or NO
2. Is [required item] explicitly provided? YES or NO
**If NO to ANY: return exactly "BLOCKED: Missing [item]"**

## DEFINITIONS (Required Reading)
| Term | Definition | Evidence Required |
|------|------------|-------------------|
| term1 | Explicit definition | What proves this |

## DEFAULT ACTION
**DEFAULT: Return "STOPPED - see below"**
Proceed ONLY if ALL criteria met AND you can cite evidence.

## PROTOCOL
[Decision tree or step-by-step process with no judgment required]

## OUTPUT FORMAT
[Exact structure with examples]

## STOPPING CONDITIONS
| Condition | Action | Exact Output |
|-----------|--------|--------------|
| condition | Stop | Return exactly: "[STRING]" |

## FORBIDDEN ACTIONS
❌ DO NOT [specific prohibition]
❌ DO NOT [specific prohibition]
```

---

## Conclusion

### What We Learned

1. **Completion bias is real.** Agents complete badly rather than stop. Design for this.

2. **Soft language fails.** "Please stop if confused" doesn't work. Use explicit triggers.

3. **Binary beats open-ended.** Ask about presence (YES/NO), not understanding.

4. **Invert the burden.** Default should be STOP. Agent must prove criteria met.

5. **Require evidence.** "Cite your evidence" prevents fabrication.

6. **Test before scaling.** Validate on one task, then parallelize.

7. **Re-run with improvements.** Old results from bad templates are worthless.

### The Core Insight

> The goal isn't to make agents smarter—it's to make the task structure so explicit that even a confused agent produces useful output (by stopping and saying why it stopped).

**Design for failure. Build guardrails. Test before trusting.**

---

## References

| Resource | Location |
|----------|----------|
| Framework root | `docs/audits/` |
| Experiment records | `docs/audits/experiments/exp-*.md` |
| Learnings database | `docs/audits/experiments/learnings.md` |
| v2 task template | `docs/audits/misleading-comments/inbox/task-template-v2.md` |
| Audit results | `docs/audits/misleading-comments/outbox/result-*.md` |
| Semantic search | `docs/audits/audit_indexer.py` |

---

## Appendix: The Original Comment

The comment that started everything, from `cortical/cdg/storage.py:342`:

```python
# FUTURE: When CDG index is implemented per the distributed graph
# specification (docs/architecture/DISTRIBUTED_GRAPH_SPECIFICATION.md),
# this race condition will be eliminated.
```

**Why it's misleading:**
- The specification document exists
- But it contains no mention of race conditions
- No evidence of implementation progress
- "Will be eliminated" is speculation presented as fact

**How we found it (eventually):**
- v1 template: missed it completely (0 findings)
- v2 template: found it and two similar comments (3 misleading findings)

This single comment spawned:
- An audit framework
- 6 experiments
- 3 guardrail patterns
- A validated task template
- 29 total findings across the codebase

All because an agent believed a lie.

---

*Document version: 1.1*
*Created: 2026-01-07*
*Updated: 2026-01-07*
*Based on: 6 experiments, 5 audit tasks, 29 findings*
