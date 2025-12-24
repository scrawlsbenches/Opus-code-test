#!/bin/bash
#
# ML Session Capture Hook for Claude Code
#
# This Stop hook automatically captures session transcripts for ML training.
# It reads the transcript_path from stdin and processes it via ml_data_collector.py
#
# Installation: Add to ~/.claude/settings.json hooks.Stop array
#

# Read JSON input from stdin
input=$(cat)

# Check if stop hook is already active (recursion prevention)
stop_hook_active=$(echo "$input" | jq -r '.stop_hook_active // "false"')
if [[ "$stop_hook_active" == "true" ]]; then
    exit 0
fi

# Check if ML collection is disabled
if [[ "${ML_COLLECTION_ENABLED:-1}" == "0" ]]; then
    exit 0
fi

# Extract transcript path from input
transcript_path=$(echo "$input" | jq -r '.transcript_path // empty')
session_id=$(echo "$input" | jq -r '.session_id // empty')
cwd=$(echo "$input" | jq -r '.cwd // empty')

# Bail if no transcript path
if [[ -z "$transcript_path" ]] || [[ ! -f "$transcript_path" ]]; then
    exit 0
fi

# Find the ml_data_collector.py script
# Try current working directory first, then the cwd from input
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR=""

if [[ -f "${cwd}/scripts/ml_data_collector.py" ]]; then
    COLLECTOR="${cwd}/scripts/ml_data_collector.py"
elif [[ -f "${SCRIPT_DIR}/../scripts/ml_data_collector.py" ]]; then
    COLLECTOR="${SCRIPT_DIR}/../scripts/ml_data_collector.py"
elif [[ -f "./scripts/ml_data_collector.py" ]]; then
    COLLECTOR="./scripts/ml_data_collector.py"
fi

# Only proceed if we found the collector and we're in a project that uses it
if [[ -z "$COLLECTOR" ]]; then
    exit 0
fi

# Check if this project has ML collection enabled (has .git-ml directory or is the right project)
if [[ ! -d "${cwd}/.git-ml" ]] && [[ ! -f "${cwd}/scripts/ml_data_collector.py" ]]; then
    exit 0
fi

# Process the transcript
cd "$cwd" 2>/dev/null || exit 0

# Error log location
error_log="$HOME/.claude/ml-capture-errors.log"
mkdir -p "$(dirname "$error_log")" 2>/dev/null

# ============================================================
# SESSION END CONTEXT - Show work summary and handoff options
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Session End Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Show recent commits from this session
echo ""
echo "📝 Recent Commits (this session):"
recent_commits=$(git log --oneline --since="1 hour ago" 2>/dev/null | head -5)
if [[ -n "$recent_commits" ]]; then
    echo "$recent_commits" | while read line; do
        echo "   $line"
    done
else
    echo "   (no commits this session)"
fi

# Show files modified
echo ""
echo "📁 Files Modified:"
modified_files=$(git status --porcelain 2>/dev/null | head -10)
if [[ -n "$modified_files" ]]; then
    echo "$modified_files" | while read line; do
        echo "   $line"
    done
else
    echo "   (working tree clean)"
fi

# Show sprint progress
echo ""
echo "📅 Sprint Status:"
sprint_status=$(python3 scripts/got_utils.py sprint status 2>/dev/null | grep -E "Sprint:|Progress:" | head -2)
if [[ -n "$sprint_status" ]]; then
    echo "$sprint_status" | while read line; do
        echo "   $line"
    done
else
    echo "   (no active sprint)"
fi

# Show in-progress tasks that may need handoff
echo ""
echo "📌 Tasks Still In Progress:"
in_progress_tasks=$(python3 scripts/got_utils.py task list --status in_progress 2>/dev/null | grep "T-" | head -5)
if [[ -n "$in_progress_tasks" ]]; then
    echo "$in_progress_tasks" | while read line; do
        echo "   $line"
    done
    echo ""
    echo "   💡 Consider creating a handoff if work is incomplete:"
    echo "      python scripts/got_utils.py handoff initiate <TASK_ID> --target next-session --instructions \"...\""
else
    echo "   (none - all tasks completed or pending)"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Call the transcript processor (log errors but don't block session end)
python3 "$COLLECTOR" transcript \
    --file "$transcript_path" \
    --session-id "$session_id" \
    2>>"$error_log" || {
    echo "[$(date -Iseconds)] ML capture failed for session $session_id (transcript: $transcript_path)" >> "$error_log"
    true  # Don't block Claude Code shutdown
}

# Archive branch manifest (records files touched during session)
python3 scripts/branch_manifest.py archive 2>/dev/null || true

# Generate draft memory from session activity (Sprint 2.3)
if [[ -f scripts/session_memory_generator.py ]]; then
    echo "📝 Generating session memory draft..."
    python3 scripts/session_memory_generator.py \
        --session-id "$session_id" \
        --output samples/memories \
        2>/dev/null || {
        echo "[$(date -Iseconds)] Memory generation failed for session $session_id" >> "$error_log"
        true  # Don't block session end
    }
fi

# Run test suite before committing session data
echo "🧪 Running test suite before session end..."
test_output=$(python3 -m pytest tests/ -x --tb=no -q 2>&1)
test_exit=$?

if [[ $test_exit -eq 0 ]]; then
    echo "✅ All tests passing"
else
    echo "⚠️  WARNING: Tests failing at session end"
    echo ""
    echo "Failed tests:"
    echo "$test_output" | grep -E "FAILED|ERROR" | head -5
    echo ""
    echo "You may want to fix these before merging."
    echo "Proceeding with session capture anyway..."
fi

# Commit tracked ML data (sessions.jsonl and commits.jsonl)
# This ensures session data is persisted in git for team/branch sharing
if [[ -d .git-ml/tracked ]] && [[ -n "$(ls -A .git-ml/tracked 2>/dev/null)" ]]; then
    git add .git-ml/tracked/ 2>/dev/null || true
    # Also add branch state if present
    if [[ -d .branch-state ]]; then
        git add .branch-state/ 2>/dev/null || true
    fi
    git commit -m "ml: Capture session data" --no-verify 2>/dev/null || true
fi

exit 0
