#!/bin/bash
#
# ML Post-Test Hook for Claude Code
#
# This hook captures pytest results and feeds them to TestExpert
# for continuous learning from test outcomes.
#
# Integration options:
#
# 1. Manual invocation after pytest:
#    pytest tests/ -v --tb=short 2>&1 | tee .pytest-output.txt
#    ./scripts/ml-post-test-hook.sh
#
# 2. Automatic from SessionStart/Stop hooks:
#    - SessionStart runs tests and saves output
#    - Stop hook calls this script to process results
#
# 3. Pytest wrapper (pytest.ini addopts):
#    addopts = --verbose --tb=short -rA
#    Then: pytest tests/ 2>&1 | tee .pytest-output.txt && ./scripts/ml-post-test-hook.sh
#

set -e

# Find project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

# Check if ML collection is enabled
if [[ "${ML_COLLECTION_ENABLED:-1}" == "0" ]]; then
    exit 0
fi

# Check if this project has test_feedback.py
if [[ ! -f scripts/hubris/test_feedback.py ]]; then
    exit 0
fi

# Function: Find pytest output file
find_pytest_output() {
    # Priority order: explicit file, then auto-detect
    if [[ -n "${PYTEST_OUTPUT_FILE:-}" ]] && [[ -f "$PYTEST_OUTPUT_FILE" ]]; then
        echo "$PYTEST_OUTPUT_FILE"
        return 0
    fi

    # Auto-detect
    for candidate in .pytest-output.txt .pytest-results.xml pytest-results.xml test-results.xml; do
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done

    return 1
}

# Function: Check if output is recent (within last 5 minutes)
is_recent_output() {
    local file="$1"
    local max_age_seconds=300  # 5 minutes

    if [[ ! -f "$file" ]]; then
        return 1
    fi

    local file_time
    file_time=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null || echo 0)
    local current_time
    current_time=$(date +%s)
    local age=$((current_time - file_time))

    [[ $age -lt $max_age_seconds ]]
}

# Function: Process test results
process_test_results() {
    local output_file="$1"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧪 Processing Test Results"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "   Output file: $output_file"

    # Determine file type
    local parse_flag=""
    if [[ "$output_file" == *.xml ]]; then
        parse_flag="--parse-xml"
    else
        parse_flag="--parse-output"
    fi

    # Process with test_feedback.py
    python3 scripts/hubris/test_feedback.py \
        "$parse_flag" "$output_file" \
        --verbose

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ Test feedback processed successfully"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    else
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "⚠️  Test feedback processing failed (exit code: $exit_code)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    fi

    return $exit_code
}

# Main execution

# Try to find output file
output_file=$(find_pytest_output)
if [[ -z "$output_file" ]]; then
    # No output file found - not an error, just nothing to process
    exit 0
fi

# Check if output is recent
if ! is_recent_output "$output_file"; then
    # Output file is stale - skip processing
    # This prevents processing old results multiple times
    exit 0
fi

# Process the results
process_test_results "$output_file"

exit $?
