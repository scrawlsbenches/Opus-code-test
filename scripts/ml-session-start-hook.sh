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

# ============================================================
# SESSION TYPE DETECTION - Is this new or continuation?
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 SESSION START"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check for pending handoffs - indicates this might be a continuation
pending_count=$(python3 scripts/got_utils.py handoff list --status initiated 2>/dev/null | grep -c "→ H-" || echo "0")
in_progress_count=$(python3 scripts/got_utils.py task list --status in_progress 2>/dev/null | grep -c "T-" || echo "0")

if [[ "$pending_count" -gt 0 ]]; then
    echo "   📨 CONTINUATION SESSION - Pending handoffs detected"
    echo "   You may be picking up work from a previous agent."
    echo ""
    echo "   Questions to answer:"
    echo "   1. Do you accept the pending handoff?"
    echo "   2. Which task are you working on?"
    echo "   3. What's your agent name? (for handoff tracking)"
elif [[ "$in_progress_count" -gt 0 ]]; then
    echo "   🔄 RESUMPTION SESSION - In-progress tasks found"
    echo "   You may be resuming your own previous work."
    echo ""
    echo "   Questions to answer:"
    echo "   1. Which task are you continuing?"
    echo "   2. What was the last state of work?"
else
    echo "   ✨ FRESH SESSION - No pending work detected"
    echo ""
    echo "   Questions to answer:"
    echo "   1. What task will you work on?"
    echo "   2. Is there a sprint or epic context?"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ============================================================
# SPRINT CONTEXT - What sprint are we in?
# ============================================================
echo ""
echo "📅 Current Sprint:"
current_sprint=$(python3 scripts/got_utils.py sprint status 2>/dev/null | head -5)
if [[ -n "$current_sprint" ]]; then
    echo "$current_sprint" | while read line; do
        echo "   $line"
    done
else
    echo "   (no active sprint)"
    echo "   💡 Create one: python scripts/got_utils.py sprint create \"Sprint Name\" --number N"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Output session info
echo ""
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

# ============================================================
# GoT (Graph of Thought) Context - Critical for session continuity
# ============================================================
echo ""
echo "🧠 GoT Context (Task & Decision Tracking)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Run GoT validation (quick health check)
got_validate=$(python3 scripts/got_utils.py validate 2>&1)
got_exit=$?

if [[ $got_exit -eq 0 ]]; then
    echo "   ✅ GoT healthy"
else
    echo "   ⚠️  GoT issues detected - run 'python scripts/got_utils.py validate'"
fi

# Show task summary
task_stats=$(python3 scripts/got_utils.py dashboard 2>/dev/null | grep -E "Tasks:|Completion:" | head -2)
echo "$task_stats" | while read line; do
    echo "   $line"
done

# ============================================================
# HANDOFF DETECTION - Check for pending handoffs awaiting acceptance
# ============================================================
echo ""
echo "   🤝 Pending Handoffs:"
pending_handoffs=$(python3 scripts/got_utils.py handoff list --status initiated 2>/dev/null | grep -E "^  → H-" | head -5)
if [[ -n "$pending_handoffs" ]]; then
    echo ""
    echo "   ⚠️  HANDOFFS AWAITING YOUR ACCEPTANCE:"
    echo "$pending_handoffs" | while read line; do
        handoff_id=$(echo "$line" | grep -oP 'H-\S+')
        echo "      $line"
    done
    echo ""
    echo "   💡 To accept a handoff: python scripts/got_utils.py handoff accept <HANDOFF_ID> --agent <YOUR_AGENT_NAME>"
    echo ""

    # Get details of first pending handoff
    first_handoff=$(echo "$pending_handoffs" | head -1 | grep -oP 'H-\S+')
    if [[ -n "$first_handoff" ]]; then
        handoff_details=$(python3 scripts/got_utils.py handoff list --status initiated 2>/dev/null | grep -A5 "$first_handoff")
        task_id=$(echo "$handoff_details" | grep "Task:" | sed 's/.*Task: //')
        if [[ -n "$task_id" ]]; then
            echo "   📋 First pending handoff task ($task_id):"
            python3 scripts/got_utils.py task show "$task_id" 2>/dev/null | head -10 | while read line; do
                echo "      $line"
            done
        fi
    fi
else
    echo "      (no pending handoffs)"
fi

# Show in-progress tasks
echo ""
echo "   📌 In Progress:"
in_progress=$(python3 scripts/got_utils.py task list --status in_progress 2>/dev/null | grep "T-" | head -3)
if [[ -n "$in_progress" ]]; then
    echo "$in_progress" | while read line; do
        echo "      $line"
    done
else
    echo "      (none)"
fi

# Show high priority pending tasks
echo ""
echo "   🔥 High Priority Pending:"
high_priority=$(python3 scripts/got_utils.py task list --status pending --priority high 2>/dev/null | grep "T-" | head -3)
if [[ -n "$high_priority" ]]; then
    echo "$high_priority" | while read line; do
        echo "      $line"
    done
else
    echo "      (none)"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Point to recent knowledge transfer docs
echo ""
echo "📚 Recent Knowledge Transfer Docs:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
recent_transfers=$(ls -t samples/memories/*knowledge-transfer*.md samples/memories/*session*.md 2>/dev/null | head -3)
if [[ -n "$recent_transfers" ]]; then
    echo "$recent_transfers" | while read file; do
        basename "$file"
    done
else
    echo "   (none found)"
fi
echo ""
echo "   💡 Tip: Read these to restore context from previous sessions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit 0
