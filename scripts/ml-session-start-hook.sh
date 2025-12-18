#!/bin/bash
#
# ML Session Start Hook for Claude Code
#
# This SessionStart hook automatically initializes ML data collection.
# It runs at the start of every Claude Code session in this project.
#
# What it does:
# 1. Starts a new ML session for commit-chat linking
# 2. Installs git hooks if not present
# 3. Shows collection stats
#

# Read JSON input from stdin (Claude Code provides session context)
input=$(cat)

# Extract working directory
cwd=$(echo "$input" | jq -r '.cwd // empty' 2>/dev/null)
if [[ -z "$cwd" ]]; then
    cwd="$(pwd)"
fi

# Only run for this specific project
if [[ ! -f "${cwd}/scripts/ml_data_collector.py" ]]; then
    exit 0
fi

cd "$cwd" || exit 0

# Function: Check and clean stale git locks
check_and_clean_git_locks() {
    local lock_file=".git/index.lock"

    if [ -f "$lock_file" ]; then
        # Get lock file age in seconds
        local lock_age=$(( $(date +%s) - $(stat -c %Y "$lock_file" 2>/dev/null || echo $(date +%s)) ))

        if [ "$lock_age" -gt 60 ]; then
            echo "⚠️  Stale git lock detected (${lock_age}s old)"
            echo "    Likely from interrupted operation or context compaction"
            rm -f "$lock_file"
            echo "    ✓ Cleaned up .git/index.lock"
        else
            echo "⚠️  Recent git lock exists (${lock_age}s old)"
            echo "    Another git process may be running"
            echo "    Will NOT auto-remove (too recent)"
        fi
    fi

    # Also check for other common lock files
    for lock in .git/refs/heads/*.lock .git/HEAD.lock; do
        if ls $lock 1>/dev/null 2>&1; then
            echo "⚠️  Found additional lock: $lock"
        fi
    done
}

# Function: Check session continuity (detect long gaps indicating compaction)
check_session_continuity() {
    local last_activity_file=".git-ml/.last_activity"
    local current_time=$(date +%s)

    if [ -f "$last_activity_file" ]; then
        local last_time=$(cat "$last_activity_file" 2>/dev/null || echo $current_time)
        local gap=$(( current_time - last_time ))

        if [ "$gap" -gt 300 ]; then  # 5 minutes
            echo "⏰ Long gap detected: ${gap}s since last activity"
            echo "   Session may have been compacted or interrupted"
        fi
    fi

    # Update last activity
    mkdir -p .git-ml
    echo "$current_time" > "$last_activity_file"
}

# Run lock and continuity checks
check_and_clean_git_locks
check_session_continuity

# Check if ML collection is disabled
if [[ "${ML_COLLECTION_ENABLED:-1}" == "0" ]]; then
    echo "📊 ML collection disabled (ML_COLLECTION_ENABLED=0)"
    exit 0
fi

# Ensure .git-ml directory exists
mkdir -p .git-ml

# Install git hooks if not present
if [[ ! -f ".git/hooks/post-commit" ]] || ! grep -q "ML-DATA-COLLECTOR" ".git/hooks/post-commit" 2>/dev/null; then
    python3 scripts/ml_data_collector.py install-hooks 2>/dev/null
fi

# Initialize branch manifest for conflict tracking
python3 scripts/branch_manifest.py init 2>/dev/null || true

# Start a new session
session_output=$(python3 scripts/ml_data_collector.py session start 2>/dev/null)
session_id=$(echo "$session_output" | grep -oP 'Started session: \K.*' || echo "")

# Get current stats
stats=$(python3 scripts/ml_data_collector.py stats 2>/dev/null | grep -E "Commits:|Chats:|Sessions:" | head -3)

# Output session info
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 ML Data Collection Active"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ -n "$session_id" ]]; then
    echo "   Session: $session_id"
fi
echo "$stats" | while read line; do
    echo "   $line"
done
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check sync status with origin
echo ""
echo "🔄 Checking origin sync status..."
current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [[ -n "$current_branch" ]]; then
    # Fetch origin (timeout after 5 seconds to not block)
    timeout 5 git fetch origin main 2>/dev/null || true

    # Count commits behind origin/main
    behind=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo "0")
    ahead=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo "0")

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Branch: $current_branch"
    if [[ "$behind" -gt 0 && "$ahead" -gt 0 ]]; then
        echo "   ⚠️  $behind commits behind, $ahead commits ahead of origin/main"
        echo "   Consider: git fetch origin && git rebase origin/main"
    elif [[ "$behind" -gt 0 ]]; then
        echo "   ⚠️  $behind commits behind origin/main"
        echo "   Consider: git pull --rebase origin main"
    elif [[ "$ahead" -gt 0 ]]; then
        echo "   📤 $ahead commits ahead of origin/main (ready to push/PR)"
    else
        echo "   ✅ In sync with origin/main"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

# Run test suite at session start
echo ""
echo "🧪 Running test suite..."
test_output=$(python3 -m pytest tests/ -x --tb=no -q 2>&1)
test_exit=$?

if [[ $test_exit -eq 0 ]]; then
    echo "✅ All tests passing"
else
    echo "⚠️  Tests failing - consider fixing before proceeding"
    echo ""
    echo "Failed tests:"
    echo "$test_output" | grep -E "FAILED|ERROR" | head -5
    echo ""
    echo "Run 'python -m pytest tests/ -v' for details"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit 0
