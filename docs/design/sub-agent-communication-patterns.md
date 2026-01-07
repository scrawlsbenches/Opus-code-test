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

## Appendix: Algorithms for Learning from Audits

This section explores data structures and algorithms—from first principles—that could help us learn patterns from audit data.

### The Problem Domain

We have:
- **Text data** (comments) with **labels** (misleading/accurate)
- **Temporal metadata** (when written, when code changed)
- **Structural relationships** (file → function → comment)
- **Goal:** Detect patterns, predict categories, find anomalies

---

### Algorithms Ranked by Effectiveness × Usefulness ÷ Complexity

| Rank | Structure/Algorithm | E | U | C | Why |
|------|---------------------|---|---|---|-----|
| **1** | Inverted Index | ★★★★★ | ★★★★★ | ★★☆☆☆ | Foundation for everything |
| **2** | Decision Tree | ★★★★★ | ★★★★★ | ★★☆☆☆ | Interpretable rules |
| **3** | Trie | ★★★★☆ | ★★★★★ | ★★☆☆☆ | Pattern prefix matching |
| **4** | Naive Bayes | ★★★★☆ | ★★★★☆ | ★★☆☆☆ | Probabilistic classification |
| **5** | Union-Find | ★★★★☆ | ★★★★☆ | ★☆☆☆☆ | Clustering with near O(1) |
| **6** | Bloom Filter | ★★★☆☆ | ★★★★★ | ★☆☆☆☆ | Fast "probably suspicious" |
| **7** | Suffix Array | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | Find repeated patterns |
| **8** | DAG (Directed Acyclic Graph) | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | Dependency tracking |
| **9** | Markov Chain | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | Sequential patterns |
| **10** | LSH | ★★★★☆ | ★★★☆☆ | ★★★★☆ | Similarity at scale |
| **11** | Count-Min Sketch | ★★☆☆☆ | ★★★☆☆ | ★★☆☆☆ | Streaming frequency |

---

### 1. Inverted Index

**What:** Maps terms → list of (document, position) pairs

```
"will" → [(doc3, pos12), (doc7, pos45), (doc7, pos89)]
"be"   → [(doc3, pos13), (doc7, pos46), (doc12, pos3)]
```

**Why #1:**
- O(1) term lookup
- Foundation for TF-IDF, phrase search, boolean queries
- Simple to build, simple to query
- Enables: "Find all comments containing 'will be implemented'"

**Complexity:** O(n) build, O(1) term lookup, O(k) for k results

---

### 2. Decision Tree

**What:** Binary tree where each node splits on a feature

```
                    [contains "See:"]
                    /              \
                  YES              NO
                  /                  \
        [file exists?]        [contains "TODO"]
        /          \              /        \
      YES          NO           YES        NO
       ↓            ↓            ↓          ↓
   accurate    MISLEADING    accurate    unknown
```

**Why #2:**
- Directly interpretable ("IF See: AND file missing THEN misleading")
- Works well with small labeled datasets (29 examples)
- Produces rules you can audit and explain
- No black box

**Complexity:** O(n log n) build, O(log n) classify

---

### 3. Trie (Prefix Tree)

**What:** Tree where edges are characters, paths spell words/patterns

```
         root
        /    \
       S      T
       |      |
       e      O
       |      |
       e      D
       |      |
       :      O
       |      |
      (leaf)  :
              |
            (leaf)
```

**Why #3:**
- O(m) lookup for pattern of length m
- Natural for "starts with" patterns ("See:", "TODO:", "FUTURE:")
- Compact storage for shared prefixes
- Enables autocomplete, pattern enumeration

**Complexity:** O(m) insert, O(m) search, O(alphabet × m) space worst case

---

### 4. Naive Bayes Classifier

**What:** Apply Bayes theorem assuming feature independence

```
P(misleading | "will", "be", "implemented") ∝
    P("will" | misleading) × P("be" | misleading) × P("implemented" | misleading) × P(misleading)
```

**Why #4:**
- Surprisingly effective for text classification
- Works with small training sets
- Fast training and inference
- Mathematically principled

**The math:**
```
P(class | document) = P(document | class) × P(class) / P(document)

With independence assumption:
P(document | class) = ∏ P(word_i | class)
```

**Complexity:** O(n × v) train, O(v) classify (v = vocabulary size)

---

### 5. Union-Find (Disjoint Set)

**What:** Tracks equivalence classes with near-constant operations

```
Initially: {a}, {b}, {c}, {d}, {e}

union(a, b) → {a,b}, {c}, {d}, {e}
union(c, d) → {a,b}, {c,d}, {e}
union(b, c) → {a,b,c,d}, {e}

find(a) = find(d)  # Same cluster
find(a) ≠ find(e)  # Different clusters
```

**Why #5:**
- Incrementally cluster similar comments
- O(α(n)) ≈ O(1) for union and find (inverse Ackermann)
- Simple implementation (< 30 lines)
- Merge similar patterns as you discover them

**Complexity:** O(α(n)) per operation, effectively constant

---

### 6. Bloom Filter

**What:** Probabilistic set membership using k hash functions

```
Insert "See: docs/":
  h1("See: docs/") mod m = 3  → set bit 3
  h2("See: docs/") mod m = 7  → set bit 7
  h3("See: docs/") mod m = 12 → set bit 12

Query "See: docs/":
  All bits 3,7,12 set? → Probably in set

Query "FIXME:":
  Bit 5 not set? → Definitely NOT in set
```

**Why #6:**
- O(k) insert and query
- Space efficient (10 bits per element for 1% false positive)
- No false negatives
- Fast first-pass filter: "Is this comment pattern known-suspicious?"

**Complexity:** O(k) operations, O(m) space, tunable false positive rate

---

### 7. Suffix Array

**What:** Sorted array of all suffixes of a string

```
Text: "TODO: fix TODO: add"
Suffixes sorted:
  ": add"
  ": fix TODO: add"
  "D: add"
  "D: fix TODO: add"
  "DO: add"
  "DO: fix TODO: add"
  "O: add"
  "O: fix TODO: add"
  "ODO: add"
  "ODO: fix TODO: add"
  ...
```

**Why #7:**
- Find all occurrences of any pattern in O(m log n)
- Find longest repeated substrings
- Discover common patterns you didn't anticipate
- More space-efficient than suffix tree

**Complexity:** O(n log n) build (or O(n) with DC3), O(m log n) search

---

### 8. DAG for Dependencies

**What:** Directed acyclic graph tracking relationships

```
comment_123 ──references──→ file_456
    │                           │
    └──written_by──→ commit_abc │
                         │      │
                         └──modifies──┘
```

**Why #8:**
- Track comment → code → change relationships
- Detect "comment references file that was deleted"
- Enable traversals: "What changed since this comment was written?"
- Foundation for staleness detection

**Complexity:** O(V + E) traversal, O(1) edge lookup with adjacency list

---

### 9. Markov Chain

**What:** Probabilistic transitions between states (words)

```
P(next_word | current_word):

"will" → {"be": 0.7, "not": 0.2, "work": 0.1}
"be"   → {"implemented": 0.5, "fixed": 0.3, "removed": 0.2}
```

**Why #9:**
- Model "style" of misleading vs accurate comments
- Detect: "This comment follows the misleading pattern"
- Generate examples of suspicious comments
- Requires minimal training data

**Complexity:** O(n) build, O(1) transition lookup

---

### 10. LSH (Locality Sensitive Hashing)

**What:** Hash similar items to same buckets with high probability

```
Similar comments hash to same bucket:
  bucket_42: ["will be implemented soon", "will be added later"]
  bucket_17: ["TODO: add tests", "TODO: write tests"]
```

**Why #10:**
- Approximate nearest neighbor in O(1)
- Find similar comments without O(n²) comparison
- Scales to large codebases
- Tunable precision/recall tradeoff

**Complexity:** O(1) query, O(n) build, sub-linear search

---

### 11. Count-Min Sketch

**What:** Space-efficient frequency estimation

```
Query: "How often does 'will be' appear?"
Answer: "Approximately 47 times (±5)"
```

**Why #11:**
- Streaming algorithm (single pass over data)
- Fixed memory regardless of data size
- Good for "what are the most common patterns?"
- Trade exactness for space

**Complexity:** O(1) update and query, O(w × d) space

---

### The "Build This First" Stack

If starting from scratch:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: INDEXING                                                      │
│  Inverted Index + Trie                                                  │
│  "Find all documents with pattern X"                                    │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: SIMILARITY                                                    │
│  Union-Find + Bloom Filter                                              │
│  "Group similar items, fast membership check"                           │
└─────────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: CLASSIFICATION                                                │
│  Decision Tree + Naive Bayes                                            │
│  "Predict category for new items"                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

This gives you interpretable rules (Decision Tree), probabilistic confidence (Naive Bayes), fast filtering (Bloom), and efficient grouping (Union-Find)—all with moderate implementation complexity.

---

*Document version: 1.2*
*Created: 2026-01-07*
*Updated: 2026-01-07*
*Based on: 6 experiments, 5 audit tasks, 29 findings*
