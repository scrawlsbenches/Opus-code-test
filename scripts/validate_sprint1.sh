#!/bin/bash
# Sprint 1 Validation Script
# Validates: Module structure, AST, Lexer, Registry

set -e  # Exit on first error

echo "=========================================="
echo "Sprint 1: Foundation"
echo "=========================================="
echo "Validating: Module structure, AST, Lexer, Registry"
echo ""

# Test 1: Module imports
echo "Test 1: Module structure and imports..."
python3 -c "
from cortical.got.expression import parse, execute
from cortical.got.expression.ast import *
from cortical.got.expression.lexer import Lexer
from cortical.got.expression.registry import FunctionRegistry
from cortical.got.expression.errors import *
print('  ✓ All modules importable')
"

# Test 2: Public API exists
echo "Test 2: Public API availability..."
python3 -c "
from cortical.got.expression import parse, execute
import inspect
# Verify parse() exists and is callable
assert callable(parse), 'parse() must be callable'
# Verify execute() exists and is callable
assert callable(execute), 'execute() must be callable'
print('  ✓ Public API functions available')
"

# Test 3: FunctionRegistry exists and has core structure
echo "Test 3: FunctionRegistry structure..."
python3 -c "
from cortical.got.expression.registry import FunctionRegistry
# FunctionRegistry is a singleton with class methods
# Check registry has register method (class method decorator)
assert hasattr(FunctionRegistry, 'register'), 'Registry must have register() method'
# Check registry has get method
assert hasattr(FunctionRegistry, 'get'), 'Registry must have get() method'
# Check registry has list_functions method
assert hasattr(FunctionRegistry, 'list_functions'), 'Registry must have list_functions() method'
print('  ✓ FunctionRegistry structure valid')
"

# Test 4: Behavioral tests
echo "Test 4: Running behavioral tests..."
pytest /home/user/Opus-code-test/tests/behavioral/test_expression_module.py -v --tb=short || exit 1
pytest /home/user/Opus-code-test/tests/behavioral/test_ast_nodes.py -v --tb=short || exit 1
pytest /home/user/Opus-code-test/tests/behavioral/test_lexer_behavioral.py -v --tb=short || exit 1
pytest /home/user/Opus-code-test/tests/behavioral/test_registry.py -v --tb=short || exit 1
echo "  ✓ All behavioral tests passed"

# Test 5: Unit tests with coverage
echo "Test 5: Running unit tests..."
pytest /home/user/Opus-code-test/tests/unit/test_ast.py -v --tb=short || exit 1
pytest /home/user/Opus-code-test/tests/unit/test_lexer.py -v --tb=short || exit 1
pytest /home/user/Opus-code-test/tests/unit/test_registry.py -v --tb=short || exit 1
echo "  ✓ All unit tests passed"

# Test 6: Smoke tests still pass
echo "Test 6: Verifying smoke tests (no regressions)..."
pytest /home/user/Opus-code-test/tests/smoke/ -v --tb=short || exit 1
echo "  ✓ Smoke tests passed"

echo ""
echo "=========================================="
echo "Sprint 1: ALL TESTS PASSED ✓"
echo "=========================================="
