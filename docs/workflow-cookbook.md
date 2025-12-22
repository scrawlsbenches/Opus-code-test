# Workflow Cookbook: Cognitive Load Optimized Development

A practical guide for reducing cognitive burden during development, based on cognitive load theory principles.

---

## Quick Reference

| Cognitive Load Type | Goal | Key Practice |
|---------------------|------|--------------|
| **Intrinsic** | Manage complexity | Break tasks into chunks |
| **Extraneous** | Eliminate waste | Use templates and conventions |
| **Germane** | Maximize learning | Immediate feedback loops |

---

## Recipe 1: Task Decomposition

**Problem:** Complex tasks overwhelm working memory (~4 chunks max).

**Solution:** Break work into checkpoints where context can safely reset.

### Pattern

```
1. Identify natural boundaries (file, function, test)
2. Create checkpoint after each boundary
3. Verify checkpoint before continuing
4. Document discoveries at each checkpoint
```

### Verification Script

```bash
# scripts/verify_task_chunks.sh
# Checks that recent commits follow chunk boundaries

#!/bin/bash
# Verify commits touch related files (low context switching)

MAX_FILES_PER_COMMIT=5
RECENT_COMMITS=10

echo "Checking last $RECENT_COMMITS commits for chunking..."

git log --oneline -n $RECENT_COMMITS --format="%h" | while read commit; do
    file_count=$(git show --stat --format="" $commit | grep -c '|')
    if [ "$file_count" -gt "$MAX_FILES_PER_COMMIT" ]; then
        echo "⚠️  $commit touches $file_count files (threshold: $MAX_FILES_PER_COMMIT)"
    else
        echo "✓  $commit: $file_count files"
    fi
done
```

---

## Recipe 2: Context Preservation

**Problem:** Context switches flush working memory, requiring reconstruction.

**Solution:** Batch similar tasks; document context before switching.

### Pattern

```
1. Group related files in single session
2. Before switching: write 2-3 sentence summary
3. Use todo lists to offload tracking
4. Minimize open tabs/windows
```

### Quick Context Dump

```bash
# Capture current context before switching tasks
cat > /tmp/context-$(date +%Y%m%d-%H%M).md << EOF
## Context: $(date)

**Working on:** [describe current task]

**Key files:**
$(git diff --name-only HEAD~1)

**Next step:** [what to do when returning]

**Open questions:** [anything unresolved]
EOF
```

---

## Recipe 3: Template-Driven Work

**Problem:** Inconsistent formats force cognitive reconstruction each time.

**Solution:** Use templates to make structure automatic.

### Available Templates

| Template | Location | Use For |
|----------|----------|---------|
| Memory entry | `scripts/new_memory.py` | Daily learnings |
| Task creation | `scripts/new_task.py` | New work items |
| Test file | `tests/unit/` pattern | New test modules |

### Template Usage

```bash
# Create memory (template auto-applied)
python scripts/new_memory.py "what I learned today"

# Create task (structured fields)
python scripts/new_task.py "Fix the bug" --priority high --category bugfix

# Check templates before creating new files
ls -la samples/memories/*.md | head -3
```

---

## Recipe 4: Immediate Feedback Loops

**Problem:** Delayed feedback wastes capacity on context reconstruction.

**Solution:** Verify immediately; automate verification where possible.

### Feedback Hierarchy

| Action | Feedback Time | Method |
|--------|---------------|--------|
| Edit code | Immediate | Syntax highlighting, LSP |
| Run test | <30 seconds | `pytest -x` (fail fast) |
| Full suite | <5 minutes | `python scripts/run_tests.py quick` |
| Coverage | <10 minutes | CI or local coverage run |

### Quick Test Loop

```bash
# Fast feedback during development
pytest tests/unit/ -x -q --tb=short

# Watch mode (if pytest-watch installed)
# ptw tests/unit/ -- -x -q
```

---

## Recipe 5: Decision Pre-Commitment

**Problem:** Deliberation at decision points consumes working memory.

**Solution:** Pre-commit to criteria; automate repetitive decisions.

### Pre-Committed Decisions

| Decision | Pre-Committed Answer |
|----------|---------------------|
| Where to add test? | Same directory as feature, `test_` prefix |
| Commit message format? | `type: description` (feat, fix, docs, refactor) |
| When to create memory? | After any non-trivial learning |
| When to run full tests? | Before commit, after merge |

### Decision Checklist Script

```bash
# scripts/pre_commit_checklist.sh
#!/bin/bash
# Run before committing to reduce decision load

echo "Pre-commit checklist:"
echo ""

# Check for uncommitted test files
STAGED=$(git diff --cached --name-only)
SRC_FILES=$(echo "$STAGED" | grep -E "^cortical/.*\.py$" | grep -v test)
TEST_FILES=$(echo "$STAGED" | grep -E "^tests/.*\.py$")

if [ -n "$SRC_FILES" ] && [ -z "$TEST_FILES" ]; then
    echo "⚠️  Source files staged without tests:"
    echo "$SRC_FILES"
    echo ""
fi

# Check commit message format
echo "✓ Remember commit format: type: description"
echo "  Types: feat, fix, docs, refactor, test, chore"
```

---

## Recipe 6: Environment Design

**Problem:** Environmental distractions consume cognitive capacity.

**Solution:** Minimize visual noise; organize by cognitive similarity.

### Workspace Organization

```
Terminal Layout (recommended):
┌─────────────────┬─────────────────┐
│                 │                 │
│   Editor        │   Terminal      │
│   (main file)   │   (tests/git)   │
│                 │                 │
├─────────────────┴─────────────────┤
│         Reference (docs/logs)      │
└────────────────────────────────────┘
```

### Clean Start Script

```bash
# scripts/clean_workspace.sh
#!/bin/bash
# Reset workspace to reduce visual noise

# Clear terminal
clear

# Show only relevant status
echo "=== Workspace Status ==="
echo ""
git status --short
echo ""
echo "=== Recent Activity ==="
git log --oneline -5
echo ""
echo "=== Current Tasks ==="
python scripts/task_utils.py list --status in_progress 2>/dev/null || echo "No active tasks"
```

---

## Recipe 7: Cognitive Load Metrics

**Problem:** Can't improve what you don't measure.

**Solution:** Track indicators of cognitive overload.

### Warning Signs

| Indicator | Measurement | Threshold |
|-----------|-------------|-----------|
| Files per commit | `git show --stat` | >5 = review chunking |
| Time to resume | Self-report | >10 min = document better |
| Error rate | Test failures | Increasing = slow down |
| Context switches | Tab/window count | >10 = consolidate |

### Metrics Script

```bash
# scripts/cognitive_metrics.sh
#!/bin/bash
# Track cognitive load indicators

echo "=== Cognitive Load Metrics ==="
echo ""

# Files per commit (last 10)
echo "Avg files per commit (last 10):"
git log --oneline -10 --format="%h" | while read commit; do
    git show --stat --format="" $commit | grep -c '|'
done | awk '{sum+=$1; count++} END {print "  " sum/count " files/commit"}'

# Commit frequency (proxy for flow state)
echo ""
echo "Commits today:"
git log --oneline --since="midnight" | wc -l | awk '{print "  " $1 " commits"}'

# Test pass rate
echo ""
echo "Recent test results:"
if [ -f .git-ml/ci_results.json ]; then
    grep -o '"result":"[^"]*"' .git-ml/ci_results.json | tail -5
else
    echo "  No CI data available"
fi
```

---

## Daily Workflow

### Morning Startup

```bash
# 1. Clean start
./scripts/clean_workspace.sh 2>/dev/null || git status --short

# 2. Check sprint context
cat tasks/CURRENT_SPRINT.md 2>/dev/null | head -30

# 3. Review active tasks
python scripts/task_utils.py list --status pending | head -10
```

### Before Each Task

```bash
# 1. Read relevant code first
cat cortical/relevant_file.py.ai_meta 2>/dev/null || head -50 cortical/relevant_file.py

# 2. Run quick tests to establish baseline
python scripts/run_tests.py smoke
```

### After Each Task

```bash
# 1. Verify with tests
pytest tests/unit/ -x -q

# 2. Commit if passing
git add -p && git commit

# 3. Document if learned something
# python scripts/new_memory.py "topic" (if applicable)
```

### End of Session

```bash
# 1. Commit all work
git status && git add -A && git commit -m "wip: end of session"

# 2. Document context for next session
python scripts/session_handoff.py 2>/dev/null || git log --oneline -5
```

---

## Verification

All scripts in this cookbook can be verified:

```bash
# Test that scripts are syntactically correct
bash -n scripts/verify_task_chunks.sh 2>/dev/null && echo "✓ verify_task_chunks.sh"
bash -n scripts/pre_commit_checklist.sh 2>/dev/null && echo "✓ pre_commit_checklist.sh"
bash -n scripts/clean_workspace.sh 2>/dev/null && echo "✓ clean_workspace.sh"
bash -n scripts/cognitive_metrics.sh 2>/dev/null && echo "✓ cognitive_metrics.sh"
```

---

*Based on cognitive load theory principles. See `samples/cross_domain/workflow_cognitive_load_bridge.md` for theoretical foundation.*
