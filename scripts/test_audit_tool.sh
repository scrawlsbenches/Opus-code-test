#!/bin/bash
# Test script for cortical.cli.audit

set -e

echo "========================================="
echo "Testing Audit CLI"
echo "========================================="
echo ""

# Test 1: Help
echo "Test 1: Help message"
python -m cortical.cli.audit --help > /dev/null
echo "✓ Help works"
echo ""

# Test 2: Patterns command
echo "Test 2: Patterns command"
python -m cortical.cli.audit patterns cortical/audits/algorithms/ --min-length 20 > /dev/null
echo "✓ Patterns works"
echo ""

# Test 3: Index command
echo "Test 3: Index command"
python -m cortical.cli.audit index cortical/audits/algorithms/ > /dev/null
echo "✓ Index works"
echo ""

# Test 4: Similar command
echo "Test 4: Similar command"
python -m cortical.cli.audit similar "Test implementation" --threshold 0.3 > /dev/null
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

python -m cortical.cli.audit train /tmp/audit_test_train/ > /dev/null
echo "✓ Train works"
echo ""

# Test 6: Scan command
echo "Test 6: Scan command"
python -m cortical.cli.audit scan cortical/audits/algorithms/ > /dev/null
echo "✓ Scan works"
echo ""

# Test 7: Health command
echo "Test 7: Health command"
python -m cortical.cli.audit health cortical/audits/algorithms/ > /dev/null
echo "✓ Health works"
echo ""

# Test 8: Reason command
echo "Test 8: Reason command"
python -m cortical.cli.audit reason cortical/audits/algorithms/ > /dev/null
echo "✓ Reason works"
echo ""

echo "========================================="
echo "All tests passed!"
echo "========================================="
