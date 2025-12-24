# Testing Strategy Guide

> **TL;DR**: Use `make test-fast` for instant feedback (~1s), `make test-quick` for pre-commit (~30s).

## The Problem

Developers are frustrated because:
1. Full test suite takes 3+ minutes
2. Installing pytest/coverage every session is annoying
3. They don't know about the tiered test system

## The Solution: Tiered Testing Pyramid

```
                          ┌─────────────────┐
                          │  ALL (3+ min)   │  ← CI only
                          ├─────────────────┤
                      ┌───┤ PRECOMMIT (2m)  │  ← Before pushing
                      │   ├─────────────────┤
                  ┌───┤   │  QUICK (30s)    │  ← Before committing
                  │   │   ├─────────────────┤
              ┌───┤   │   │  FAST (<5s)     │  ← During development
              │   │   │   ├─────────────────┤
          ┌───┤   │   │   │  SMOKE (<1s)    │  ← After every change
          │   │   │   │   └─────────────────┘
          ▼   ▼   ▼   ▼
        Use frequency during development
```

## Quick Reference

| Command | Time | When to Use |
|---------|------|-------------|
| `make test-smoke` | ~1s | After every code change |
| `make test-fast` | ~5s | Before switching tasks |
| `make test-quick` | ~30s | Before committing |
| `make test-precommit` | ~2m | Before pushing |
| `make test-all` | ~3m+ | Let CI do this |

## Setup: One-Time Dependency Installation

### Option 1: Install as Dev Package (Recommended)
```bash
pip install -e ".[dev]"
```

### Option 2: Minimal Installation
```bash
pip install pytest coverage pytest-timeout pytest-xdist
```

### Option 3: Let run_tests.py Handle It
```bash
python scripts/run_tests.py smoke  # Auto-installs if needed
```

## Daily Workflow

### During Active Development

```bash
# Write code...
make test-smoke      # ~1s sanity check

# Write more code...
make test-smoke      # Still working?

# Ready to commit feature...
make test-quick      # Run smoke + fast unit tests

# Ready to push...
make test-precommit  # Full local validation
git push
```

### Using pytest Directly

```bash
# Smoke tests only
pytest tests/smoke/ -q

# Unit tests (no slow tests)
pytest tests/unit/ -q

# Include slow tests
pytest tests/unit/ -m "" -q

# Parallel execution (4 workers)
pytest tests/unit/ -n 4 -q

# Stop on first failure
pytest tests/ -x

# Show slowest 10 tests
pytest tests/ --durations=10
```

## Test Categories Explained

### Smoke Tests (`tests/smoke/`)
- **Purpose**: Basic sanity check - "does it even import?"
- **Time**: <1 second
- **When**: After every code change
- **What**: Core imports, basic processor creation, simple operations

### Fast Tests
- **Purpose**: Quick feedback on core functionality
- **Time**: <5 seconds
- **When**: Before switching context
- **What**: Unit tests excluding slow markers

### Unit Tests (`tests/unit/`)
- **Purpose**: Test individual components in isolation
- **Time**: ~70 seconds (all), ~30s (excluding slow)
- **When**: Before committing
- **What**: 7,795 fast, isolated tests

### Integration Tests (`tests/integration/`)
- **Purpose**: Test component interactions
- **Time**: ~60 seconds
- **When**: Before pushing
- **What**: Cross-module functionality

### Performance Tests (`tests/performance/`)
- **Purpose**: Catch performance regressions
- **Time**: ~25 seconds
- **When**: When changing algorithms
- **What**: Timing-based assertions

### Behavioral Tests (`tests/behavioral/`)
- **Purpose**: Verify user-facing quality
- **Time**: ~15 seconds (without shared_processor)
- **When**: When changing search/relevance
- **What**: Quality metrics, relevance tests

## Understanding Slow Tests

Some tests are intentionally slow because they verify timing behavior:

| Test Pattern | Why Slow | Time |
|--------------|----------|------|
| `test_network_error_retries` | Tests exponential backoff | 6s |
| `test_detect_stuck_phase` | Tests anomaly detection timing | 2.5s |
| `test_debounced_*` | Tests debounce behavior | 1-2s |
| `test_ttl_*` | Tests time-to-live expiration | 0.6s |
| `test_lock_*` | Tests lock acquisition timing | 0.5s |

These are marked with `@pytest.mark.slow` and excluded by default during development.

## Environment-Specific Strategies

### Ephemeral Environments (Containers, Cloud IDEs)

Each session starts fresh, so dependencies must be installed:

```bash
# Add to shell startup or session hook
pip install -e ".[dev]" --quiet 2>/dev/null

# Or use the auto-install feature
python scripts/run_tests.py smoke  # Installs if needed
```

### VS Code / IDE Integration

Add to `.vscode/settings.json`:
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests/smoke/",
        "-v"
    ]
}
```

### Pre-commit Hook

Already configured in `.claude/settings.local.json`:
```json
{
    "hooks": {
        "PreCommit": ["python scripts/run_tests.py quick"]
    }
}
```

## Parallel Execution

Install pytest-xdist for parallel test execution:

```bash
pip install pytest-xdist

# Run with 4 workers
pytest tests/unit/ -n 4

# Auto-detect CPU count
pytest tests/unit/ -n auto
```

**Speedup**: ~2-3x faster on multi-core machines.

## Troubleshooting

### "pytest not found"

```bash
# Auto-install
python scripts/run_tests.py --check-deps

# Or manual
pip install pytest coverage
```

### Tests timing out

```bash
# Increase timeout
pytest tests/ --timeout=60

# Or skip slow tests
pytest tests/ -m "not slow"
```

### shared_processor takes forever

The `shared_processor` fixture loads the full sample corpus (~125 docs).
Use `small_processor` or `fresh_processor` fixtures instead for faster tests.

### Too many tests collecting

```bash
# Run specific category only
pytest tests/smoke/ -q

# Or specific file
pytest tests/unit/test_tokenizer.py -q
```

## CI/CD Integration

CI runs the full test suite with coverage:

```yaml
# .github/workflows/ci.yml (simplified)
- smoke-tests: ~30s (gate for other jobs)
- unit-tests: ~2m (parallel with integration)
- integration-tests: ~3m (parallel with unit)
- coverage-report: Combines unit + integration
```

**Local vs CI**:
- Local: Skip slow tests, no coverage overhead
- CI: All tests, full coverage

## Metrics and Monitoring

Track test performance over time:

```bash
# Show slowest tests
pytest tests/ --durations=20

# Profile collection time
time pytest tests/ --collect-only -q
```

## Adding New Tests

1. **Place in correct category**:
   - Unit test? → `tests/unit/`
   - Integration? → `tests/integration/`
   - Bug fix? → `tests/regression/`

2. **Mark slow tests**:
   ```python
   @pytest.mark.slow
   def test_something_with_timing():
       time.sleep(1)  # Intentional delay
   ```

3. **Use appropriate fixtures**:
   - `fresh_processor` for isolated tests
   - `small_processor` for realistic data
   - Avoid `shared_processor` unless necessary
