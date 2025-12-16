# Git Sync Skill

Safely synchronize with remote branch, handling edge cases like conflicts and interrupted operations.

## Usage

When the user says "pull from main", "sync with remote", "git sync", or similar:

1. First, check the current git state
2. If in a bad state (rebase conflict, detached HEAD, etc.), offer recovery
3. Run the sync script with appropriate mode

## Steps

### Step 1: Diagnose Current State

Run these checks in order:

```bash
# Check for rebase in progress
if [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; then
    echo "REBASE_IN_PROGRESS"
fi

# Check for merge in progress
if [ -f .git/MERGE_HEAD ]; then
    echo "MERGE_IN_PROGRESS"
fi

# Check for detached HEAD
if ! git symbolic-ref -q HEAD >/dev/null; then
    echo "DETACHED_HEAD"
fi

# Check working tree status
git status --porcelain
```

### Step 2: Recovery (if needed)

**If REBASE_IN_PROGRESS:**
- Ask user: "Rebase in progress. Options: (1) Abort rebase, (2) Continue rebase, (3) Show status"
- Default for auto mode: abort rebase with `git rebase --abort`

**If MERGE_IN_PROGRESS:**
- Ask user: "Merge in progress. Options: (1) Abort merge, (2) Continue merge"
- Default for auto mode: abort merge with `git merge --abort`

**If DETACHED_HEAD:**
- Ask user: "Detached HEAD. Options: (1) Checkout previous branch, (2) Create new branch"
- Show: `git reflog -5` to help identify where they were
- Default for auto mode: `git checkout -` (previous branch)

### Step 3: Run Sync

After recovery (or if state was clean):

```bash
./scripts/git-sync-ml.sh --auto
```

Or for more control:
```bash
./scripts/git-sync-ml.sh --semi
```

### Step 4: Verify

After sync, always verify:
```bash
git status
git log --oneline -3
```

Confirm to user: "Sync complete. You are on branch X, up to date with origin/X"

## Important Notes

- NEVER check filesystem for file existence during git conflicts
- ALWAYS resolve git state FIRST, then verify files
- The sync script handles ML data conflicts automatically
- If non-ML files have conflicts, pause and ask user

## Recovery Commands Reference

```bash
# Abort operations
git rebase --abort
git merge --abort
git cherry-pick --abort

# Reset to remote (DESTRUCTIVE - loses local changes)
git fetch origin
git reset --hard origin/BRANCH

# Reset to remote but keep local changes
git fetch origin
git stash
git reset --hard origin/BRANCH
git stash pop

# Checkout previous branch
git checkout -

# See recent history
git reflog -10
```
