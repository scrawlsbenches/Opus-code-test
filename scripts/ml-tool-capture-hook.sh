#!/bin/bash
#
# ML Tool Capture Hook for Claude Code
#
# This PostToolUse hook captures tool invocations for ML training.
# It runs after each tool execution and logs the tool name, input, and output.
#
# IMPORTANT: This hook must be fast (<100ms) to not slow down Claude Code.
# We write to a temp file and batch process at session end.
#

# Read JSON input from stdin
input=$(cat)

# Check if ML collection is disabled
if [[ "${ML_COLLECTION_ENABLED:-1}" == "0" ]]; then
    exit 0
fi

# Extract tool information
tool_name=$(echo "$input" | jq -r '.tool_name // empty' 2>/dev/null)
tool_input=$(echo "$input" | jq -c '.tool_input // {}' 2>/dev/null)
tool_output=$(echo "$input" | jq -c '.tool_output // {}' 2>/dev/null)
session_id=$(echo "$input" | jq -r '.session_id // empty' 2>/dev/null)
cwd=$(echo "$input" | jq -r '.cwd // empty' 2>/dev/null)

# Skip if no tool name
if [[ -z "$tool_name" ]]; then
    exit 0
fi

# Only proceed if this is our project
if [[ ! -d "${cwd}/.git-ml" ]]; then
    exit 0
fi

# Create tool log directory
tool_log_dir="${cwd}/.git-ml/tool_uses"
mkdir -p "$tool_log_dir" 2>/dev/null

# Timestamp
timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Create a compact log entry (append to session file for batching)
session_file="${tool_log_dir}/${session_id:-unknown}.jsonl"

# Truncate large outputs (>10KB) to avoid bloating the log
if [[ ${#tool_output} -gt 10240 ]]; then
    tool_output='"[TRUNCATED - output > 10KB]"'
fi

# Write log entry (async to not block)
{
    echo "{\"ts\":\"$timestamp\",\"tool\":\"$tool_name\",\"input\":$tool_input,\"output\":$tool_output}"
} >> "$session_file" 2>/dev/null &

exit 0
