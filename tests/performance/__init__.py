"""
Performance Tests
=================

Timing-based tests that catch performance regressions.
These tests:
- Should NOT run under coverage (adds 10x+ overhead)
- Use the small synthetic corpus fixture for speed
- Have explicit timing thresholds
- Are isolated from other test categories

Run with: python -m pytest tests/performance/ -v --no-cov
"""
