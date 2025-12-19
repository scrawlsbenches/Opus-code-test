#!/bin/bash
#
# ML File Prediction Suggestion Hook
#
# This prepare-commit-msg hook suggests potentially missing files based on
# the commit message using ML file prediction.
#
# What it does:
# 1. Reads the commit message
# 2. Runs ML file prediction on the message
# 3. Compares predictions with staged files
# 4. Warns if high-confidence files are missing
# 5. Allows commit to proceed (non-blocking by default)
#
# Environment variables:
#   ML_SUGGEST_ENABLED=0    - Disable suggestions (default: 1)
#   ML_SUGGEST_THRESHOLD=0.5 - Confidence threshold for warnings (default: 0.5)
#   ML_SUGGEST_BLOCKING=1    - Block commit if missing files (default: 0)
#   ML_SUGGEST_TOP_N=5       - Number of predictions to check (default: 5)
#
# Usage (git installs this automatically via ml_data_collector.py):
#   cp scripts/ml-precommit-suggest.sh .git/hooks/prepare-commit-msg
#   chmod +x .git/hooks/prepare-commit-msg
#

# Get hook arguments
COMMIT_MSG_FILE=$1
COMMIT_SOURCE=$2
SHA1=$3

# Only run for regular commits (not merge, amend, etc.)
# This prevents noise during merge commits or rebases
if [[ -n "$COMMIT_SOURCE" ]]; then
    exit 0
fi

# Check if suggestions are disabled
if [[ "${ML_SUGGEST_ENABLED:-1}" == "0" ]]; then
    exit 0
fi

# Check if we're in the right project
if [[ ! -f "scripts/ml_file_prediction.py" ]]; then
    exit 0
fi

# Configuration
THRESHOLD=${ML_SUGGEST_THRESHOLD:-0.5}
BLOCKING=${ML_SUGGEST_BLOCKING:-0}
TOP_N=${ML_SUGGEST_TOP_N:-5}

# Check if model is trained
MODEL_FILE=".git-ml/models/file_prediction.json"
if [[ ! -f "$MODEL_FILE" ]]; then
    # Model not trained yet - silently skip
    exit 0
fi

# Read commit message (skip comment lines)
COMMIT_MSG=$(grep -v '^#' "$COMMIT_MSG_FILE" | tr '\n' ' ')

# Skip if commit message is empty
if [[ -z "$COMMIT_MSG" ]]; then
    exit 0
fi

# Skip ML data commits to prevent noise
if [[ "$COMMIT_MSG" == "data: ML"* ]]; then
    exit 0
fi

# Get staged files
STAGED_FILES=$(git diff --cached --name-only)
if [[ -z "$STAGED_FILES" ]]; then
    # No staged files - probably amending or unusual workflow
    exit 0
fi

# Run ML prediction
PREDICTIONS=$(python3 scripts/ml_file_prediction.py predict "$COMMIT_MSG" --top "$TOP_N" 2>/dev/null)
PREDICT_EXIT_CODE=$?

# Check if prediction succeeded
if [[ $PREDICT_EXIT_CODE -ne 0 ]]; then
    # Prediction failed - silently skip
    exit 0
fi

# Parse predictions and compare with staged files
MISSING_FILES=()
MISSING_SCORES=()

while IFS= read -r line; do
    # Parse prediction line format: "  1. path/to/file.py              (0.123)"
    if [[ "$line" =~ ^[[:space:]]*[0-9]+\.[[:space:]]+([^[:space:]]+).*\(([0-9.]+)\) ]]; then
        PREDICTED_FILE="${BASH_REMATCH[1]}"
        SCORE="${BASH_REMATCH[2]}"

        # Check if score meets threshold
        if (( $(echo "$SCORE >= $THRESHOLD" | bc -l 2>/dev/null || echo 0) )); then
            # Check if file is staged
            if ! echo "$STAGED_FILES" | grep -q "^${PREDICTED_FILE}$"; then
                MISSING_FILES+=("$PREDICTED_FILE")
                MISSING_SCORES+=("$SCORE")
            fi
        fi
    fi
done <<< "$PREDICTIONS"

# If no missing files, exit quietly
if [[ ${#MISSING_FILES[@]} -eq 0 ]]; then
    exit 0
fi

# Display warning
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 ML File Prediction Suggestion"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Based on your commit message, these files might need changes:"
echo ""

for i in "${!MISSING_FILES[@]}"; do
    FILE="${MISSING_FILES[$i]}"
    SCORE="${MISSING_SCORES[$i]}"
    printf "  • %-45s (confidence: %.3f)\n" "$FILE" "$SCORE"
done

echo ""
echo "Staged files:"
echo "$STAGED_FILES" | sed 's/^/  ✓ /'
echo ""

if [[ "$BLOCKING" == "1" ]]; then
    echo "❌ Commit blocked. Add missing files or disable with:"
    echo "   export ML_SUGGEST_BLOCKING=0"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
else
    echo "ℹ️  Tip: Review the suggestions above. To block commits with missing files:"
    echo "   export ML_SUGGEST_BLOCKING=1"
    echo ""
    echo "To disable suggestions: export ML_SUGGEST_ENABLED=0"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
fi
