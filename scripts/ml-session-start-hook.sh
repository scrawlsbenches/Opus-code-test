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

exit 0
