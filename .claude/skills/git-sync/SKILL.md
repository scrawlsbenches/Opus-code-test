# Git Sync Skill

Safely synchronize with remote branch. Use when user says "pull from main", "sync with remote", "git sync", etc.

## Critical Rule

**NEVER check filesystem for file existence while git is in a conflicted state.**

Always: `git status` first → resolve issues → then check files.

## Quick Sync (90% of cases)

```bash
git status
git fetch origin
git pull --no-rebase origin BRANCH
git push -u origin BRANCH
```

## Handling Bad Git States

### Check for issues first:
```bash
git status
```

### If rebase in progress:
```bash
# Ask user: abort or continue?
git rebase --abort   # Safe default - can re-pull after
# OR
git rebase --continue  # If user wants to keep changes
```

### If merge in progress:
```bash
# Ask user: abort or continue?
git merge --abort    # Safe default
# OR
git merge --continue # If user resolved conflicts
```

### If detached HEAD:
```bash
# Show recent history
git reflog -5

# Find previous branch and checkout
git checkout -       # Previous branch
# OR
git checkout BRANCH  # Specific branch
```

## ML Data Conflict Resolution

ML data files (`.git-ml/`) use unique session IDs, so conflicts are rare. If they occur:

1. **commits.jsonl conflicts**: Take remote, then backfill missing entries
   ```bash
   git checkout --theirs .git-ml/tracked/commits.jsonl
   python scripts/ml_data_collector.py backfill -n 50
   ```

2. **Session files**: Keep both (unique names, no real conflict)
   ```bash
   git checkout --ours .git-ml/sessions/
   git checkout --theirs .git-ml/sessions/
   # Both versions are valid
   ```

## After Sync: Re-index if Code Changed

If remote had code changes (not just ML data):
```bash
python scripts/index_codebase.py --incremental
python scripts/generate_ai_metadata.py --incremental
```

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| "Files don't exist" but they should | Checked during conflict | Resolve git state first, then check |
| Push rejected | Remote has changes | `git pull --no-rebase` then push |
| Rebase conflicts keep appearing | Rebase replays commits | Use merge instead: `git pull --no-rebase` |

## The Simple Rule

When resuming a session:
1. `git status` - see the truth
2. Fix any issues (abort stale operations)
3. `git pull --no-rebase` - merge, don't rebase
4. Work
5. `git push`
