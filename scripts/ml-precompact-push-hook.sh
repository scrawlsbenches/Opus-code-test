#!/bin/bash
#
# Pre-Compaction Auto-Push Hook
#
# This hook runs before context window compaction to push any uncommitted
# changes as a recovery checkpoint. If compaction causes issues, we can
# recover from the last pushed state.
#
# Task: T-20251218-012842-dbf8-006
#

# Read JSON input from stdin
input=$(cat)

# Extract working directory
cwd=$(echo "$input" | jq -r '.cwd // empty' 2>/dev/null)
if [[ -z "$cwd" ]]; then
    cwd="$(pwd)"
fi

cd "$cwd" || exit 0

# Only run if we're in a git repository
if [[ ! -d ".git" ]]; then
    exit 0
fi

# Get current branch
current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [[ -z "$current_branch" ]]; then
    exit 0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💾 Pre-Compaction Recovery Checkpoint"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check for uncommitted changes
if [[ -n $(git status --porcelain 2>/dev/null) ]]; then
    echo "   Found uncommitted changes, creating checkpoint..."

    # Stage tracked files only (not untracked)
    git add -u 2>/dev/null

    # Commit with auto-message
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    git commit -m "checkpoint: Pre-compaction auto-save at $timestamp" 2>/dev/null

    if [[ $? -eq 0 ]]; then
        echo "   ✅ Created checkpoint commit"
    fi
fi

# Check if we have commits to push
ahead=$(git rev-list --count origin/${current_branch}..HEAD 2>/dev/null || echo "0")
if [[ "$ahead" -gt 0 ]]; then
    echo "   Pushing $ahead commit(s) to origin..."

    # Push with timeout (10 seconds)
    timeout 10 git push origin "$current_branch" 2>/dev/null

    if [[ $? -eq 0 ]]; then
        echo "   ✅ Pushed to origin/$current_branch"
    else
        echo "   ⚠️  Push failed (network issue?) - local commits preserved"
    fi
else
    echo "   ✅ Already in sync with origin"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit 0
