#!/bin/bash
# Verify commits follow cognitive load chunking principles
# Checks that recent commits touch a reasonable number of files

MAX_FILES_PER_COMMIT=${MAX_FILES:-5}
RECENT_COMMITS=${COMMITS:-10}

echo "Checking last $RECENT_COMMITS commits for chunking..."
echo "Threshold: $MAX_FILES_PER_COMMIT files per commit"
echo ""

warnings=0
total=0

git log --oneline -n "$RECENT_COMMITS" --format="%h %s" | while read -r line; do
    commit=$(echo "$line" | cut -d' ' -f1)
    message=$(echo "$line" | cut -d' ' -f2-)
    file_count=$(git show --stat --format="" "$commit" 2>/dev/null | grep -c '|')
    total=$((total + 1))

    if [ "$file_count" -gt "$MAX_FILES_PER_COMMIT" ]; then
        echo "⚠️  $commit ($file_count files): $message"
        warnings=$((warnings + 1))
    else
        echo "✓  $commit ($file_count files): $message"
    fi
done

echo ""
if [ "$warnings" -gt 0 ]; then
    echo "Consider breaking large commits into smaller chunks."
    exit 1
else
    echo "All commits follow chunking guidelines."
    exit 0
fi
