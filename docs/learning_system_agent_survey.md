# Agent Learning System Survey

## Purpose
Understand what different types of agents would need from a learning system
to be effective in their work.

## Survey Questions

1. When you start a new task, what would help you most?
2. What information from past experiences would be valuable?
3. How should the system present guidance to you?
4. What would make you trust or distrust the guidance?
5. How would you contribute back to the system?

---

## Panel Responses

### Agent 1: Worker Agent (Task Executor)

**Role:** Executes focused tasks like "implement this feature" or "fix this bug"

**Q1: When you start a new task, what would help you most?**

> "I need to know if anyone has done something similar before. Not just
> 'feature implementation' but specifically 'JWT authentication' or
> 'database migration'. The actual task matters, not the category.
>
> I also need to know if the files I'm about to touch are 'safe' or
> 'dangerous'. If src/auth.py has broken 3 times in the last week,
> I want to know that before I start."

**Q2: What information from past experiences would be valuable?**

> "Three things:
> 1. What approach worked for similar tasks
> 2. What mistakes to avoid (specific, not generic)
> 3. What files/areas are risky
>
> I don't need 'use test-driven development'. I need 'when implementing
> auth tokens, test expiry edge cases - we've had 2 bugs from that'."

**Q3: How should the system present guidance to you?**

> "Brief and specific. Before I start:
> - Similar past task: [link]
> - Key insight: [one line]
> - Warning: [if any]
>
> Don't give me a wall of text. Don't give me generic advice."

**Q4: What would make you trust or distrust the guidance?**

> "Trust: If it references specific past tasks I can verify
> Distrust: If it's vague like 'be careful with databases'
>
> Show me the evidence. 'This advice comes from 3 successful tasks
> and 1 failure' is trustworthy. 'Our system learned this' is not."

**Q5: How would you contribute back to the system?**

> "After I finish, I should record:
> - What I actually did (not just action types, but specifics)
> - What worked that I didn't expect
> - What failed that I didn't expect
> - Which files I touched and what happened
>
> Make it easy. Don't ask me to write an essay."

---

### Agent 2: Director Agent (Orchestrator)

**Role:** Breaks down goals into tasks, assigns to workers, synthesizes results

**Q1: When you start a new task, what would help you most?**

> "I need to know how to decompose this goal. Have we done similar
> goals before? How were they broken down? What was the right
> parallelization strategy?
>
> I also need to know which workers have experience with which
> types of tasks. If Worker-3 has done 5 auth tasks successfully,
> I should assign auth work to them."

**Q2: What information from past experiences would be valuable?**

> "Decomposition patterns that worked:
> - 'Goal: Add feature X' was split into [task1, task2, task3]
> - These tasks could be parallel: [task1, task2]
> - This task blocked others: [task3]
>
> And failure patterns:
> - 'We tried to parallelize X and Y but they conflicted'
> - 'This goal took 3x longer than estimated because...'"

**Q3: How should the system present guidance to you?**

> "Show me similar past goals and their decompositions:
>
> Similar goal: 'Add OAuth support'
> Decomposition: [research, design, implement, test]
> Duration: 4 hours
> Issues: 'Research took longer than expected'
>
> Let me learn from the shape of past work."

**Q4: What would make you trust or distrust the guidance?**

> "Trust: If it shows me the actual history with outcomes
> Distrust: If it's prescriptive without evidence
>
> Don't tell me 'always do X'. Show me 'in 5 similar cases,
> 4 did X successfully and 1 did Y and failed'."

**Q5: How would you contribute back to the system?**

> "I should record:
> - How I decomposed the goal
> - What the dependency graph looked like
> - Which estimates were off and why
> - What I would do differently
>
> This helps future Directors with similar goals."

---

### Agent 3: Research Agent (Explorer)

**Role:** Investigates codebases, finds information, answers questions

**Q1: When you start a new task, what would help you most?**

> "I need to know where to look. Have we explored this area before?
> What did we find? What paths were dead ends?
>
> If someone already searched for 'how authentication works in this
> codebase', I shouldn't repeat that work. Give me their findings."

**Q2: What information from past experiences would be valuable?**

> "Exploration maps:
> - 'Authentication is in src/auth/, not src/security/'
> - 'The config system is confusing, see notes from task X'
> - 'This module has no tests, be careful'
>
> Basically, institutional knowledge that's hard to rediscover."

**Q3: How should the system present guidance to you?**

> "When I search for something, tell me:
> - Has this been searched before? What was found?
> - Are there related explorations I should know about?
> - What's the current understanding of this area?
>
> Like a knowledge base that grows from exploration."

**Q4: What would make you trust or distrust the guidance?**

> "Trust: Recent, specific findings with file references
> Distrust: Old, vague summaries
>
> 'As of 2 days ago, auth is in src/auth/jwt.py:45' is good.
> 'Auth is somewhere in src' is useless."

**Q5: How would you contribute back to the system?**

> "Every exploration should leave a trail:
> - What I was looking for
> - Where I looked
> - What I found (with file:line references)
> - What's still unknown
>
> Future agents shouldn't repeat my exploration."

---

### Agent 4: Recovery Agent (Fixer)

**Role:** Handles failures, recovers from errors, resolves blockers

**Q1: When you start a new task, what would help you most?**

> "I need failure history. What's the error? Have we seen it before?
> What fixed it last time? What didn't work?
>
> If this is a known issue with a known fix, just tell me.
> Don't make me rediscover it."

**Q2: What information from past experiences would be valuable?**

> "Error → Solution mappings:
> - 'ConnectionTimeout in database module' → 'Check pool size, see task X'
> - 'Import error in auth' → 'Circular import, refactored in PR #123'
> - 'Test flakiness in CI' → 'Race condition, fixed by adding lock'
>
> The more specific, the better."

**Q3: How should the system present guidance to you?**

> "When an error occurs, immediately show:
>
> KNOWN ISSUE? Yes (seen 3 times)
> LAST FIX: Increased connection pool size
> FILES INVOLVED: src/database/pool.py
> TASK REFERENCE: T-12345
>
> Don't make me search. The context is the error."

**Q4: What would make you trust or distrust the guidance?**

> "Trust: If it shows exactly this error was fixed before
> Distrust: If it's guessing based on similar-ish errors
>
> Be honest: 'Exact match: 3 times' vs 'Similar error: maybe related'
> I'll judge if it applies."

**Q5: How would you contribute back to the system?**

> "After fixing:
> - The exact error message/pattern
> - What the root cause was
> - What fixed it
> - How to prevent it
>
> Build a troubleshooting database."

---

## Synthesis: Common Needs Across All Agents

| Need | Description | All Agents? |
|------|-------------|-------------|
| **Semantic matching** | Find similar tasks by meaning, not category | ✓ |
| **Specific over generic** | "Test token expiry" not "write tests" | ✓ |
| **Evidence-based** | Show the source experiences | ✓ |
| **Recency matters** | Recent experience > old experience | ✓ |
| **File-level tracking** | Know which files are risky | Worker, Recovery |
| **Easy contribution** | Don't make reporting a burden | ✓ |
| **Verification** | Ability to check the source | ✓ |

---

## Key Design Implications

### 1. Search by Intent, Not Category

All agents want to find "similar tasks" by what the task actually is,
not by metadata like `goal_type="feature"`.

**Implication:** Index experiences by keywords from `intent` field.

### 2. Specific, Actionable Guidance

Nobody wants "use TDD". They want "test token expiry because we had bugs".

**Implication:** Store and retrieve specific `what_worked` / `what_didnt_work`
with enough context to be actionable.

### 3. Evidence Trails

Everyone wants to verify guidance by seeing the source.

**Implication:** Always link guidance back to specific experiences/tasks.

### 4. File Risk Awareness

Workers and Recovery agents especially care about file history.

**Implication:** Index experiences by files touched, track success/failure per file.

### 5. Low-Friction Contribution

Nobody wants to write essays after completing work.

**Implication:** Make experience capture structured and quick.

---

## Proposed Data Model (Based on Survey)

```python
class SmartExperience:
    # Identity
    id: str
    timestamp: datetime

    # The task (for semantic matching)
    intent: str                    # "Implement JWT authentication"
    keywords: Set[str]             # ["jwt", "authentication", "implement"]

    # What was touched (for file tracking)
    files_touched: List[str]       # ["src/auth/jwt.py", "tests/test_jwt.py"]

    # The outcome
    outcome: SUCCESS | FAILURE
    error_pattern: Optional[str]   # For failure matching

    # The learnings (specific, not generic)
    what_worked: List[str]         # ["Tested token expiry edge cases"]
    what_didnt_work: List[str]     # ["Forgot to handle refresh tokens"]
    key_insight: Optional[str]     # One line summary

    # For verification
    task_reference: Optional[str]  # Link to GoT task
    files_reference: Dict[str, str]  # file -> relevant line numbers
```

```python
class SmartGuidance:
    # For the current task
    similar_successes: List[ExperienceSummary]
    similar_failures: List[ExperienceSummary]

    # For the files being touched
    file_risks: Dict[str, FileRisk]

    # Aggregated insights
    recommended_approach: Optional[str]  # From successful similar tasks
    warnings: List[str]                  # From failures and risky files

    # For verification
    evidence_count: int
    source_experiences: List[str]        # IDs for drilling down
```

---

## Next Steps

1. Validate this understanding with actual implementation
2. Build keyword extraction from intent
3. Build file-based experience indexing
4. Test with real agent workflows
