# Cortical Text Processor - Development Commands
# ================================================
#
# Quick Reference:
#   make test-smoke     # ~1s  - sanity check
#   make test-fast      # ~5s  - quick feedback
#   make test-quick     # ~30s - before commit
#   make test-precommit # ~2m  - before push
#   make help           # show all commands

.PHONY: help install test test-smoke test-fast test-quick test-precommit \
        test-unit test-integration test-coverage test-parallel \
        lint clean deps-check

# Default target
help:
	@echo "Cortical Text Processor - Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "Test Commands (use these!):"
	@echo "  make test-smoke      Run smoke tests (~1s)"
	@echo "  make test-fast       Run fast tests (~5s, no slow tests)"
	@echo "  make test-quick      Run quick suite (~30s, smoke + unit)"
	@echo "  make test-precommit  Run pre-commit suite (~2m)"
	@echo "  make test-parallel   Run unit tests with 4 workers"
	@echo "  make test-coverage   Run with coverage report"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install         Install as editable with dev deps"
	@echo "  make deps-check      Check if test dependencies are installed"
	@echo ""
	@echo "Individual Categories:"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo ""
	@echo "See docs/testing-strategy.md for full documentation."

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e ".[dev]"

deps-check:
	@python scripts/run_tests.py --check-deps

# =============================================================================
# Test Tiers (ordered by speed)
# =============================================================================

# Tier 1: Smoke tests (~1s)
# Use after every code change
test-smoke:
	@echo "Running smoke tests..."
	@python -m pytest tests/smoke/ -q --tb=line || \
		(echo "Tip: Run 'make install' if pytest not found" && exit 1)

# Tier 2: Fast tests (~5s)
# Smoke + critical unit tests, no slow markers
test-fast:
	@echo "Running fast tests (no slow tests)..."
	@python -m pytest tests/smoke/ tests/unit/ -q --tb=short \
		-m "not slow" --ignore=tests/unit/test_got_cli.py 2>/dev/null || \
		python scripts/run_tests.py smoke

# Tier 3: Quick tests (~30s)
# Full smoke + unit (default marker exclusions apply)
test-quick:
	@echo "Running quick tests (smoke + unit)..."
	@python scripts/run_tests.py quick -q

# Tier 4: Pre-commit (~2m)
# Full smoke + unit + integration
test-precommit:
	@echo "Running pre-commit suite..."
	@python scripts/run_tests.py precommit -q

# =============================================================================
# Individual Test Categories
# =============================================================================

test-unit:
	@python -m pytest tests/unit/ -v --tb=short

test-integration:
	@python -m pytest tests/integration/ -v --tb=short

test-regression:
	@python -m pytest tests/regression/ -v --tb=short

test-performance:
	@python -m pytest tests/performance/ -v --tb=short -s

test-behavioral:
	@python -m pytest tests/behavioral/ -v --tb=short

# =============================================================================
# Special Test Modes
# =============================================================================

# Run with coverage report
test-coverage:
	@echo "Running tests with coverage..."
	@python -m coverage run --source=cortical -m pytest tests/unit/ tests/integration/ -q
	@python -m coverage report --include="cortical/*"

# Run with parallel workers (requires pytest-xdist)
test-parallel:
	@echo "Running unit tests with 4 parallel workers..."
	@python -m pytest tests/unit/ -n 4 -q --tb=short 2>/dev/null || \
		(echo "Installing pytest-xdist..." && pip install pytest-xdist -q && \
		 python -m pytest tests/unit/ -n 4 -q --tb=short)

# Run all tests (let CI do this normally)
test-all:
	@echo "Running ALL tests (this takes a while)..."
	@python scripts/run_tests.py all

# Alias for muscle memory
test: test-quick

# =============================================================================
# Utility
# =============================================================================

# Show slowest tests
test-profile:
	@python -m pytest tests/unit/ --durations=20 -q

# Clean up test artifacts
clean:
	@rm -rf .pytest_cache .coverage htmlcov coverage.xml
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned test artifacts"

# Lint with built-in tools (no external deps)
lint:
	@python -m py_compile cortical/*.py
	@echo "Syntax check passed"
