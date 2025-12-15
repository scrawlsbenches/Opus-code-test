---
description: Pre-merge sanity check for a branch with tests and verification
argument-hint: <branch-name>
---
# Pre-merge sanity check for branch: $ARGUMENTS

You have NO prior context. This is a fresh thread.

## SETUP (do this first)

```bash
cd /home/user/Opus-code-test
git fetch origin $ARGUMENTS
git checkout $ARGUMENTS
```

## CHECKS (use absolute paths throughout)

Run these in order, report results for each:

1. **Git status** - should be clean
   ```bash
   git status --short
   ```

2. **Check test dependencies**
   ```bash
   python /home/user/Opus-code-test/scripts/run_tests.py --check-deps
   ```

3. **Run smoke tests**
   ```bash
   python /home/user/Opus-code-test/scripts/run_tests.py smoke -q
   ```
   - If pytest missing and can't install, try: `python -m unittest discover -s tests/smoke -v`

4. **Verify ML collector commands work** (if applicable)
   ```bash
   python /home/user/Opus-code-test/scripts/ml_data_collector.py stats
   python /home/user/Opus-code-test/scripts/ml_data_collector.py quality-report
   ```

5. **Check for syntax errors in new/modified files**
   ```bash
   git diff --name-only origin/main..HEAD -- '*.py' | head -10
   # Then for each: python -m py_compile <file>
   ```

## RULES

- Use ABSOLUTE paths only (never relative paths)
- Do NOT use `cd` after setup - it persists and causes confusion
- Distinguish between:
  - **ENV issues**: missing pytest, permissions, etc. (not PR's fault)
  - **CODE issues**: actual bugs, syntax errors, failing tests (PR's fault)
- Do NOT make any changes - this is read-only verification

## REPORT FORMAT

```
## Sanity Check: <branch-name>

| Check | Result | Notes |
|-------|--------|-------|
| Git status | PASS/FAIL | |
| Dependencies | PASS/FAIL | |
| Smoke tests | PASS/FAIL | |
| ML commands | PASS/FAIL/SKIP | |
| Syntax check | PASS/FAIL | |

**Verdict: PASS** (ready to merge)
-- or --
**Verdict: FAIL** (issues found)
- Issue 1: ...
- Issue 2: ...
```

## CLEANUP

When done, return to original branch:
```bash
git checkout -
```
