#!/bin/bash
# Track cognitive load indicators
# Use to identify when to adjust workflow

echo "═══════════════════════════════════════════"
echo "  Cognitive Load Metrics"
echo "═══════════════════════════════════════════"
echo ""

# 1. Files per commit (chunking indicator)
echo "📊 Commit Chunking (last 10 commits):"
total_files=0
commit_count=0

while read -r commit; do
    files=$(git show --stat --format="" "$commit" 2>/dev/null | grep -c '|')
    total_files=$((total_files + files))
    commit_count=$((commit_count + 1))
done < <(git log --oneline -10 --format="%h" 2>/dev/null)

if [ "$commit_count" -gt 0 ]; then
    avg=$(echo "scale=1; $total_files / $commit_count" | bc 2>/dev/null || echo "$((total_files / commit_count))")
    echo "   Average: $avg files/commit"
    if [ "${avg%.*}" -gt 5 ]; then
        echo "   ⚠️  Consider smaller commits"
    else
        echo "   ✓  Good chunking"
    fi
else
    echo "   No commits found"
fi
echo ""

# 2. Commit frequency (flow state indicator)
echo "⏱️  Commit Frequency:"
today_commits=$(git log --oneline --since="midnight" 2>/dev/null | wc -l)
week_commits=$(git log --oneline --since="1 week ago" 2>/dev/null | wc -l)
echo "   Today: $today_commits commits"
echo "   This week: $week_commits commits"
echo ""

# 3. Context switching (file diversity)
echo "🔄 Context Switches (files touched today):"
today_files=$(git log --since="midnight" --name-only --format="" 2>/dev/null | sort -u | wc -l)
unique_dirs=$(git log --since="midnight" --name-only --format="" 2>/dev/null | xargs -I {} dirname {} 2>/dev/null | sort -u | wc -l)
echo "   Files: $today_files"
echo "   Directories: $unique_dirs"
if [ "$unique_dirs" -gt 5 ]; then
    echo "   ⚠️  High context switching"
else
    echo "   ✓  Focused work"
fi
echo ""

# 4. Test health (error rate proxy)
echo "🧪 Test Health:"
if [ -f ".git-ml/ci_results.json" ]; then
    recent_results=$(tail -5 .git-ml/ci_results.json 2>/dev/null | grep -c '"result":"pass"')
    total_recent=$(tail -5 .git-ml/ci_results.json 2>/dev/null | wc -l)
    if [ "$total_recent" -gt 0 ]; then
        echo "   Recent CI: $recent_results/$total_recent passing"
    fi
else
    # Quick local test check
    if command -v pytest &>/dev/null; then
        echo "   Run: pytest tests/smoke/ -q"
    else
        echo "   No CI data available"
    fi
fi
echo ""

# 5. Session duration estimate
echo "📅 Session Activity:"
first_commit=$(git log --oneline --since="midnight" --format="%ar" 2>/dev/null | tail -1)
last_commit=$(git log --oneline -1 --format="%ar" 2>/dev/null)
if [ -n "$first_commit" ]; then
    echo "   First commit today: $first_commit"
    echo "   Last commit: $last_commit"
fi
echo ""

echo "═══════════════════════════════════════════"
echo "  Recommendations"
echo "═══════════════════════════════════════════"

# Generate recommendations based on metrics
if [ "${avg%.*}" -gt 5 ] 2>/dev/null; then
    echo "• Break commits into smaller chunks"
fi
if [ "$unique_dirs" -gt 5 ]; then
    echo "• Consider batching similar file changes"
fi
if [ "$today_commits" -eq 0 ]; then
    echo "• Start with a small commit to build momentum"
fi
if [ "$today_commits" -gt 0 ] && [ "$unique_dirs" -le 3 ]; then
    echo "• Good focus! Keep the momentum"
fi
echo ""
