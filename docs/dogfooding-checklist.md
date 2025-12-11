# Dog-Fooding Checklist

This checklist ensures we systematically test features using the Cortical Text Processor itself. The goal is to catch issues like the passage-level search doc-type boosting bug **before** they make it into production.

## Context

When implementing search features, it's easy to test individual components in isolation but miss integration issues. By actually using the system to index and search its own codebase, we catch problems that only appear in real-world usage.

---

## 1. Pre-Testing Setup

- [ ] **Re-index the codebase after changes**
  - Why: New code/docs won't appear in search results if the index is stale
  - Command: `python scripts/index_codebase.py --incremental`

- [ ] **Verify index completed successfully**
  - Why: Partial failures can lead to inconsistent state
  - Check: Look for "✓ Indexing complete" message, no exceptions

- [ ] **Check document count matches expectations**
  - Why: Missing or duplicate documents indicate indexing problems
  - Command: `python scripts/search_codebase.py "/stats"` in interactive mode
  - Verify: Count matches `find cortical tests -name "*.py" | wc -l` (or similar)

---

## 2. Search Quality Checks

- [ ] **Test document-level search with known queries**
  - Why: Baseline for comparing against passage-level search
  - Example queries:
    - `"PageRank algorithm"`
    - `"bigram separator"`
    - `"compute TF-IDF"`
  - Verify: Relevant files appear in top 5 results

- [ ] **Test passage-level search with same queries**
  - Why: Passage-level should return focused context from same documents
  - Command: Use `find_passages_for_query()` or `search_codebase.py` with passage mode
  - Verify: Results point to correct files and line ranges

- [ ] **Compare results - are they consistent?**
  - Why: Document and passage results should be complementary, not contradictory
  - Check: If `analysis.py` is #1 for doc search, it should appear in passage results too

- [ ] **Test conceptual queries ("what is X")**
  - Why: Should surface documentation and explanatory comments
  - Example queries:
    - `"what is a minicolumn"`
    - `"how does PageRank work"`
    - `"concept clustering algorithm"`
  - Expected: `.md` files and docstrings rank highly

- [ ] **Test implementation queries ("where is X")**
  - Why: Should surface actual code implementations
  - Example queries:
    - `"where is PageRank computed"`
    - `"implementation of TF-IDF"`
    - `"add document incremental"`
  - Expected: `.py` files with actual functions rank highly

- [ ] **Verify doc-type boosting is working**
  - Why: Catches the exact bug we found (passage search ignoring doc-type boosts)
  - Test: For conceptual query, check if `.md` files are boosted
  - Test: For implementation query, check if `.py` files are boosted
  - Evidence: Compare scores with/without doc-type filter

- [ ] **Check if documentation surfaces for conceptual queries**
  - Why: Users asking "what" questions need docs, not raw code
  - Query: `"hierarchical layer structure"`
  - Expected: `CLAUDE.md` or relevant docs appear in top 3

---

## 3. New Feature Verification

- [ ] **Search for terms from new code/docs**
  - Why: Ensures new content is indexed and retrievable
  - Action: Identify 2-3 unique terms from your new code
  - Query: Search for those terms
  - Verify: New file appears in results

- [ ] **Verify new files appear in results**
  - Why: New files might not be indexed if patterns are wrong
  - Check: Search for filename or unique content
  - Verify: File is in top results

- [ ] **Test the specific feature end-to-end**
  - Why: Unit tests may pass but integration fails
  - Action: Use the exact workflow a user would follow
  - Example: If you added intent parsing, run `search_by_intent()` on real corpus

- [ ] **Try edge cases**
  - Why: Edge cases reveal assumptions in the code
  - Examples:
    - Empty query
    - Very long query (50+ words)
    - Query with special characters
    - Query matching zero documents
    - Query matching all documents

---

## 4. Issue Discovery Protocol

- [ ] **Document any unexpected behavior**
  - Why: Memory is fallible; write it down immediately
  - Format: Query → Expected → Actual → Why it matters

- [ ] **Add new tasks to TASK_LIST.md immediately**
  - Why: Issues discovered during testing are easy to forget
  - Template:
    ```markdown
    ## Task #XX: Fix [brief description]
    **Status**: Not Started
    **Priority**: [High/Medium/Low]
    **Created**: [date]

    **Description**:
    When testing [feature], discovered [issue].

    Query: `[search query]`
    Expected: [what should happen]
    Actual: [what happened]

    **Root Cause** (if known):
    [explanation]

    **Proposed Fix**:
    [how to fix it]
    ```

- [ ] **Include evidence (query, results, expected vs actual)**
  - Why: Makes debugging easier when you return to the task
  - Save: Query strings, top 5 results, scores, file paths

- [ ] **Update summary tables**
  - Why: Keeps TASK_LIST.md organized and scannable
  - Tables to update:
    - Status summary (count by status)
    - Priority breakdown
    - Category summary

---

## 5. Final Verification

- [ ] **All issues documented in TASK_LIST.md?**
  - Why: Un-documented issues will be forgotten
  - Check: Review your testing notes and ensure every issue has a task

- [ ] **Summary tables updated?**
  - Why: Tables provide quick overview of project health
  - Verify: Counts match number of tasks in each section

- [ ] **Changes committed and pushed?**
  - Why: Sharing findings with team prevents duplicate work
  - Check: `git status` shows clean working directory
  - Verify: Latest commit includes test findings and new tasks

---

## Quick Example

Here's what a complete dog-fooding session looks like:

```bash
# 1. Re-index
python scripts/index_codebase.py --incremental

# 2. Test known queries
python scripts/search_codebase.py "PageRank algorithm" --verbose
python scripts/search_codebase.py "what is a minicolumn" --verbose

# 3. Test new feature
python scripts/search_codebase.py "my new function name" --verbose

# 4. Document issues
# (Open TASK_LIST.md and add any problems found)

# 5. Commit findings
git add docs/ TASK_LIST.md
git commit -m "Add dog-fooding findings from feature X testing"
```

---

## Tips

- **Test early, test often**: Don't wait until feature is "done" to dog-food
- **Use interactive mode**: `python scripts/search_codebase.py --interactive` for rapid iteration
- **Compare with grep**: If search misses obvious results, something is broken
- **Think like a user**: What would someone actually search for?
- **Document surprises**: Even if it's "working as designed", unexpected behavior may indicate UX issues

---

## Common Issues to Watch For

| Symptom | Likely Cause |
|---------|--------------|
| New file not in results | Not re-indexed, or file pattern excluded |
| Zero results for obvious query | Tokenization issue, or term not in corpus |
| Wrong files ranked #1 | Scoring bug (TF-IDF, doc-type, etc.) |
| Passage and doc results diverge | Passage search missing a boost/filter |
| Docs don't surface for "what is" | Doc-type boosting not applied |
| Code doesn't surface for "where is" | Same as above |

---

*Remember: The best way to ensure quality is to actually use what we build.*
