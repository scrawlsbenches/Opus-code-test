#!/bin/bash
# Test script for audit_tool.py

set -e

echo "========================================="
echo "Testing Audit Tool"
echo "========================================="
echo ""

# Test 1: Help
echo "Test 1: Help message"
python scripts/audit_tool.py --help > /dev/null
echo "✓ Help works"
echo ""

# Test 2: Patterns command
echo "Test 2: Patterns command"
python scripts/audit_tool.py patterns cortical/audits/algorithms/ --min-length 20 > /dev/null
echo "✓ Patterns works"
echo ""

# Test 3: Index command
echo "Test 3: Index command"
python scripts/audit_tool.py index cortical/audits/algorithms/ > /dev/null
echo "✓ Index works"
echo ""

# Test 4: Similar command
echo "Test 4: Similar command"
python scripts/audit_tool.py similar "Test implementation" --threshold 0.3 > /dev/null
echo "✓ Similar works"
echo ""

# Test 5: Train command
echo "Test 5: Train command"
mkdir -p /tmp/audit_test_train
cat > /tmp/audit_test_train/misleading.txt << 'EOF'
will be implemented
coming soon
TBD
placeholder
EOF

cat > /tmp/audit_test_train/accurate.txt << 'EOF'
Calculate the total
Returns the result
Validate input
Close the connection
EOF

python scripts/audit_tool.py train /tmp/audit_test_train/ > /dev/null
echo "✓ Train works"
echo ""

# Test 6: Scan command
echo "Test 6: Scan command"
python scripts/audit_tool.py scan cortical/audits/algorithms/ > /dev/null
echo "✓ Scan works"
echo ""

echo "========================================="
echo "All tests passed!"
echo "========================================="
