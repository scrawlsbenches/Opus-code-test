#!/bin/bash
#
# Test script for ML pre-commit hook
#
# This script demonstrates the ML file prediction suggestion hook
# without actually creating a commit.
#

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 Testing ML Pre-Commit File Suggestion Hook"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if model is trained
if [[ ! -f ".git-ml/models/file_prediction.json" ]]; then
    echo "❌ Model not trained. Please run:"
    echo "   python scripts/ml_file_prediction.py train"
    echo ""
    exit 1
fi

echo "✓ Model found"
echo ""

# Test 1: Feature commit
echo "Test 1: Feature commit (feat: Add authentication)"
echo "─────────────────────────────────────────────────"
python3 scripts/ml_file_prediction.py predict "feat: Add authentication feature" --top 5
echo ""

# Test 2: Test commit
echo "Test 2: Test commit (test: Add unit tests)"
echo "─────────────────────────────────────────────────"
python3 scripts/ml_file_prediction.py predict "test: Add unit tests for query expansion" --top 5
echo ""

# Test 3: Documentation commit
echo "Test 3: Documentation commit (docs: Update README)"
echo "─────────────────────────────────────────────────"
python3 scripts/ml_file_prediction.py predict "docs: Update README with new features" --top 5
echo ""

# Test 4: Fix commit
echo "Test 4: Fix commit (fix: Correct validation logic)"
echo "─────────────────────────────────────────────────"
python3 scripts/ml_file_prediction.py predict "fix: Correct validation logic in tokenizer" --top 5
echo ""

# Test 5: Refactor commit
echo "Test 5: Refactor commit (refactor: Split processor module)"
echo "─────────────────────────────────────────────────"
python3 scripts/ml_file_prediction.py predict "refactor: Split processor into smaller modules" --top 5
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Tests complete!"
echo ""
echo "To see the hook in action during a real commit:"
echo "  1. Stage a file: git add <file>"
echo "  2. Commit: git commit -m \"feat: Your commit message\""
echo ""
echo "To disable suggestions temporarily:"
echo "  export ML_SUGGEST_ENABLED=0"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
