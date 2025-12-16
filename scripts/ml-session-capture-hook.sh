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

# Call the transcript processor (log errors but don't block session end)
python3 "$COLLECTOR" transcript \
    --file "$transcript_path" \
    --session-id "$session_id" \
    2>>"$error_log" || {
    echo "[$(date -Iseconds)] ML capture failed for session $session_id (transcript: $transcript_path)" >> "$error_log"
    true  # Don't block Claude Code shutdown
}

exit 0
