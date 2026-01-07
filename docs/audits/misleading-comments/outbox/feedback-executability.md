# Executability Feedback: Task-001

**Reviewer:** Sub-agent (executability assessment)
**Date:** 2026-01-07
**Task Reviewed:** `task-001.md` (misleading comments audit for `cortical/cdg/`)
**Verdict:** PARTIALLY EXECUTABLE - needs practical details

---

## 1. What I Could Execute Immediately

These instructions are clear and actionable:

| Instruction | How I'd Execute It | Tool/Command |
|-------------|-------------------|--------------|
| Search for patterns in directory | `grep -rn "FUTURE:\|TODO:\|FIXME:\|PLANNED:\|HACK:\|XXX:\|TEMPORARY:\|WORKAROUND:" cortical/cdg/` | Grep tool or bash |
| Record file path, line number | Output from grep provides this | - |
| Verify document exists | Use Read tool or `ls` to check file existence | Read tool |
| Use git blame for timestamp | `git blame <file>` focusing on specific lines | Bash with git |
| Write results to outbox | Use Write tool to create result file | Write tool |
| Rename to claim task | `mv inbox/task-001.md inbox/task-001.claimed.md` | Bash with mv |

**Strong points:**
- Scope is crystal clear: one directory, specific patterns
- Constraints are explicit: 2 hours, 50 findings max
- Check-in requirements are well-defined
- Stop conditions are clear

---

## 2. What I Couldn't Execute Without More Info

### 2.1 Assessment Criteria (HIGH PRIORITY)

**Problem:** The task says to assess each finding as `accurate | stale | misleading | unknown`, but doesn't provide concrete criteria.

**What I need:**
```markdown
Assessment Guide:
- **accurate**: Comment describes current code behavior/implementation (verified by reading code)
- **stale**: Comment WAS accurate but code has changed (verified by git history)
- **misleading**: Comment never matched reality (no evidence in git history)
- **unknown**: Cannot determine without domain knowledge or human input
```

**Why this matters:** Without clear criteria, two agents might assess the same comment differently, leading to inconsistent results.

**Current workaround:** I'd have to infer criteria from the README's questions, but explicit rubric would be better.

---

### 2.2 "Check if Feature Exists" (HIGH PRIORITY)

**Problem:** Instruction #4 says "If comment references a feature, check if feature exists" but doesn't define what this means operationally.

**What I need:**
```markdown
Feature Verification Protocol:
1. Extract feature name from comment (e.g., "FUTURE: When CDG index supports X")
2. Search codebase for related function/class names
3. Check for test files mentioning the feature
4. Look in git history for implementation commits
5. If unsure, mark as "unknown" and note in findings
```

**Example of ambiguity:**
- Comment says: "TODO: Add caching layer"
- How do I verify if caching exists?
  - Search for "cache" in code?
  - Look for specific cache libraries?
  - Check for performance tests?
  - Ask human?

**Current workaround:** I'd have to make judgment calls, which could be inconsistent.

---

### 2.3 Result File Format (MEDIUM PRIORITY)

**Problem:** Task file doesn't explicitly show the table format for findings.

**What I need:** Either:
- Link to README section with table format, OR
- Include table template in task file:

```markdown
## Findings

| File | Line | Content | Assessment | Notes |
|------|------|---------|------------|-------|
| storage.py | 342 | `FUTURE: When CDG...` | misleading | References spec X which exists but no impl |
```

**Current workaround:** I can infer from the README example, but explicit template prevents format drift across agents.

---

### 2.4 50 Finding Limit Behavior (MEDIUM PRIORITY)

**Problem:** Task says "Max findings: 50 (STOP if more)" but doesn't clarify execution strategy.

**What I need:**
```markdown
When you hit 50 findings:
1. STOP searching immediately (don't scan remaining files)
2. Write partial results with "TRUNCATED: stopped at 50 findings"
3. Note in results: "X files scanned, Y files remaining"
4. Coordinator will create follow-up task for remaining files
```

**vs alternative interpretation:**
```markdown
When you hit 50 findings:
1. Continue scanning all files
2. Report only first 50 findings
3. Note total count found: "Reporting 50 of 127 total findings"
```

**Why this matters:** First interpretation saves time but gives incomplete picture. Second gives complete count but takes longer.

**Current workaround:** I'd have to guess, probably choosing first interpretation.

---

### 2.5 Multi-line Comment Handling (LOW PRIORITY)

**Problem:** Pattern matching may find comments that span multiple lines. How do I capture context?

**Example:**
```python
# TODO: This is a long explanation
#       that spans multiple lines
#       and provides important context
```

**What I need:**
```markdown
For multi-line comments:
- Record the triggering line (with TODO/FUTURE)
- Include up to 3 lines of context before and after
- If longer, truncate with "..."
```

**Current workaround:** I'd capture what grep gives me, but might miss important context.

---

### 2.6 "Comments with 'will be', 'should be', 'planned to'" (LOW PRIORITY)

**Problem:** Task says to check these phrases, but they're not in the grep pattern.

**What I need:**
```markdown
Two-pass search:
1. First pass: grep for explicit markers (FUTURE:|TODO: etc)
2. Second pass: grep for aspirational phrases ("will be", "should be", "planned to")
3. Combine results, deduplicate
```

**Current workaround:** I'd add them to the grep pattern, but task should specify if this is separate or combined with main patterns.

---

## 3. Missing Practical Details

### 3.1 Commands & Tools

**Add to task file:**
```markdown
## Recommended Commands

### Pattern Search
grep -rn "FUTURE:\|TODO:\|FIXME:\|PLANNED:\|HACK:\|XXX:\|TEMPORARY:\|WORKAROUND:" cortical/cdg/ --include="*.py"

### Check for aspirational language
grep -rn "will be\|should be\|planned to" cortical/cdg/ --include="*.py"

### Git blame for specific line
git blame -L <line>,<line> <file>

### Check document existence
ls -la docs/design/<referenced-file>
```

### 3.2 Example Walkthrough

**Add to task file:**
```markdown
## Example: Full Workflow

1. Find a match:
   ```
   cortical/cdg/storage.py:342:    # FUTURE: When CDG index supports semantic search
   ```

2. Read the surrounding code to understand context:
   ```python
   # Use Read tool to view storage.py lines 330-350
   ```

3. Check if "CDG index" has semantic search:
   ```bash
   grep -r "semantic.*search" cortical/cdg/
   # or search for related class/function names
   ```

4. Use git blame to date the comment:
   ```bash
   git blame -L 342,342 cortical/cdg/storage.py
   # Output: 7b617c49 (Agent 2026-01-04) ...
   ```

5. Make assessment:
   - If semantic search exists → "stale" (comment should be removed)
   - If no semantic search found → "accurate" (still in future)
   - If partial implementation → "unknown" (needs human review)

6. Record in findings table with file:line and reasoning.
```

### 3.3 Assessment Decision Tree

**Add to task file:**
```markdown
## Assessment Decision Tree

For each finding, follow this logic:

1. Does the comment reference a document?
   - YES → Does document exist?
     - YES → Check if comment matches document
       - MATCH → "accurate"
       - MISMATCH → "misleading"
     - NO → "misleading" (phantom reference)
   - NO → Continue to #2

2. Does the comment describe a feature/behavior?
   - YES → Does feature exist in code?
     - YES → "stale" (comment should be updated/removed)
     - NO → git blame shows age?
       - < 6 months → "accurate" (likely still planned)
       - > 6 months → "unknown" (ask human if abandoned)
   - NO → Continue to #3

3. Is comment a warning or workaround?
   - YES → Does the problem it describes still exist?
     - YES → "accurate"
     - NO → "stale"
   - NO → Mark as "unknown"
```

### 3.4 Partial Results Template

**Add to task file:**
```markdown
## Partial Results Format

When writing check-ins (every 30 min or 10 findings), use this format:

# Result: task-001 (PARTIAL - Check-in N)

## Progress
- **Time elapsed:** X minutes
- **Files scanned:** N of M
- **Findings so far:** N
- **Current file:** cortical/cdg/current_file.py

## Findings (so far)
[Same table format as final results]

## Status
- [ ] Scanning complete
- [x] On track to finish within time limit
- [ ] May need to truncate (approaching 50 findings)
```

### 3.5 Common Pitfalls

**Add to task file:**
```markdown
## Common Mistakes to Avoid

1. **Don't trust comment timestamps in code**
   - Comments lie about dates
   - ALWAYS use git blame for ground truth

2. **Don't assume FUTURE: means abandoned**
   - Check git history for recent related commits
   - Feature might be in progress

3. **Don't search outside scope**
   - If you find issues in cortical/got/, note in parking-lot
   - Don't expand search to other directories

4. **Don't skip "What Went Wrong" sections**
   - Even if nothing went wrong, write "Nothing went wrong"
   - These sections help improve future tasks

5. **Don't update manifest.md**
   - Coordinator does this
   - You updating it will cause conflicts
```

---

## 4. Summary

### Executability Score: 7/10

**What works well:**
- ✅ Scope is perfectly defined
- ✅ Constraints are clear and enforceable
- ✅ Check-in requirements are explicit
- ✅ Basic search patterns are provided
- ✅ Stop conditions are unambiguous

**What needs improvement:**
- ⚠️ Assessment criteria need explicit rubric
- ⚠️ "Feature verification" needs operational definition
- ⚠️ Result format needs explicit template
- ⚠️ 50-finding limit behavior needs clarification
- ⚠️ Example commands would reduce ambiguity

### Recommendation

**Verdict: APPROVED WITH CONDITIONS**

The task is executable by an experienced agent who can make reasonable inferences, but would benefit from:

1. **HIGH PRIORITY additions:**
   - Explicit assessment criteria rubric
   - Feature verification protocol
   - Result table template in task file

2. **MEDIUM PRIORITY additions:**
   - Example grep commands
   - Example walkthrough of one finding
   - Clarification on 50-finding limit behavior

3. **LOW PRIORITY additions:**
   - Multi-line comment handling guidance
   - Common pitfalls section

**For this specific execution:** I could complete task-001 by making reasonable assumptions, but:
- My assessments might differ from another agent's due to lack of rubric
- I'd need to guess at "feature verification" protocol
- Results might not match expected format exactly

**Bottom line:** The framework is solid. The task is 80% executable. Adding the missing 20% of practical details would make it 100% executable and ensure consistent results across agents.

---

## Closing Note

This feedback is written in the spirit of making the audit framework better. The core idea is excellent—bounded tasks, frequent check-ins, clear stop conditions. The execution details just need to catch up to the framework's ambition.

The framework thinks like a senior engineer: "What could go wrong?" Now the task files need to think like a junior agent: "What exact steps do I take?"
