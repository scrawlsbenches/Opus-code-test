#!/usr/bin/env python3
"""
Test Runner Script
==================

Convenient script for running different test categories locally.

Usage:
    python scripts/run_tests.py              # Run all tests
    python scripts/run_tests.py smoke        # Run smoke tests only
    python scripts/run_tests.py unit         # Run unit tests
    python scripts/run_tests.py integration  # Run integration tests
    python scripts/run_tests.py performance  # Run performance tests (no coverage)
    python scripts/run_tests.py regression   # Run regression tests
    python scripts/run_tests.py behavioral   # Run behavioral tests
    python scripts/run_tests.py quick        # Run smoke + unit (fast feedback)
    python scripts/run_tests.py precommit    # Run smoke + unit + integration
    python scripts/run_tests.py coverage     # Run with coverage report

Options:
    -v, --verbose    Show verbose output
    -q, --quiet      Show minimal output
    --no-capture     Show print statements (pytest -s)
    --failfast       Stop on first failure
"""

import argparse
import subprocess
import sys
import os
import time

# Ensure we're running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)


# Test category definitions
CATEGORIES = {
    'smoke': {
        'description': 'Quick sanity checks',
        'paths': ['tests/smoke/'],
        'expected_time': '< 30s',
    },
    'unit': {
        'description': 'Fast isolated unit tests',
        'paths': ['tests/unit/'],
        'expected_time': '< 1 min',
        'also_run': [
            'tests/test_tokenizer.py',
            'tests/test_layers.py',
            'tests/test_config.py',
            'tests/test_code_concepts.py',
            'tests/test_embeddings.py',
            'tests/test_fingerprint.py',
            'tests/test_gaps.py',
        ],
    },
    'integration': {
        'description': 'Component interaction tests',
        'paths': ['tests/integration/'],
        'expected_time': '< 3 min',
        'also_run': [
            'tests/test_processor.py',
            'tests/test_query.py',
            'tests/test_analysis.py',
            'tests/test_semantics.py',
            'tests/test_persistence.py',
            'tests/test_incremental_indexing.py',
            'tests/test_chunk_indexing.py',
        ],
    },
    'performance': {
        'description': 'Timing-based performance tests',
        'paths': ['tests/performance/'],
        'expected_time': '< 1 min',
        'no_coverage': True,
    },
    'regression': {
        'description': 'Bug-specific regression tests',
        'paths': ['tests/regression/'],
        'expected_time': '< 1 min',
    },
    'behavioral': {
        'description': 'User workflow quality tests',
        'paths': ['tests/behavioral/'],
        'expected_time': '< 2 min',
    },
}

# Composite test suites
SUITES = {
    'quick': ['smoke', 'unit'],
    'precommit': ['smoke', 'unit', 'integration'],
    'full': ['smoke', 'unit', 'integration', 'regression', 'behavioral', 'performance'],
    'all': ['smoke', 'unit', 'integration', 'regression', 'behavioral', 'performance'],
}


def print_header(text, char='='):
    """Print a formatted header."""
    width = 70
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def run_pytest(paths, verbose=False, quiet=False, no_capture=False,
               failfast=False, no_coverage=False):
    """Run pytest with the given paths and options."""
    cmd = [sys.executable, '-m', 'pytest']

    # Add paths
    cmd.extend(paths)

    # Add options
    if verbose:
        cmd.append('-v')
    elif quiet:
        cmd.append('-q')
    else:
        cmd.append('-v')
        cmd.append('--tb=short')

    if no_capture:
        cmd.append('-s')

    if failfast:
        cmd.append('-x')

    # Performance tests should not run under coverage
    if no_coverage:
        cmd.append('--no-cov')

    return subprocess.run(cmd).returncode


def run_unittest(paths, verbose=False):
    """Run unittest for legacy test files."""
    cmd = [sys.executable, '-m', 'unittest']

    if verbose:
        cmd.append('-v')

    cmd.extend(paths)

    return subprocess.run(cmd).returncode


def run_category(category, verbose=False, quiet=False, no_capture=False,
                 failfast=False):
    """Run a test category."""
    config = CATEGORIES[category]

    print_header(f"Running {category.upper()} Tests: {config['description']}")
    print(f"Expected time: {config['expected_time']}")

    start_time = time.time()

    # Run pytest tests
    paths = config['paths']
    no_coverage = config.get('no_coverage', False)

    result = run_pytest(
        paths,
        verbose=verbose,
        quiet=quiet,
        no_capture=no_capture,
        failfast=failfast,
        no_coverage=no_coverage
    )

    # Run legacy unittest tests if configured
    if 'also_run' in config and result == 0:
        existing_files = [f for f in config['also_run'] if os.path.exists(f)]
        if existing_files:
            print(f"\n--- Also running {len(existing_files)} legacy test files ---")
            for test_file in existing_files:
                legacy_result = subprocess.run([
                    sys.executable, '-m', 'unittest', test_file, '-v' if verbose else ''
                ]).returncode
                if legacy_result != 0:
                    result = legacy_result
                    if failfast:
                        break

    elapsed = time.time() - start_time

    if result == 0:
        print(f"\n✅ {category.upper()} tests PASSED in {elapsed:.1f}s")
    else:
        print(f"\n❌ {category.upper()} tests FAILED in {elapsed:.1f}s")

    return result


def run_with_coverage():
    """Run full test suite with coverage."""
    print_header("Running Full Test Suite with Coverage")

    cmd = [
        sys.executable, '-m', 'coverage', 'run', '--source=cortical',
        '-m', 'unittest', 'discover', '-s', 'tests', '-v'
    ]

    result = subprocess.run(cmd).returncode

    if result == 0:
        print("\n--- Coverage Report ---")
        subprocess.run([
            sys.executable, '-m', 'coverage', 'report',
            '-m', '--include=cortical/*'
        ])

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run Cortical Text Processor tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'category',
        nargs='?',
        default='all',
        choices=list(CATEGORIES.keys()) + list(SUITES.keys()) + ['coverage'],
        help='Test category or suite to run (default: all)'
    )

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Quiet output')
    parser.add_argument('--no-capture', action='store_true',
                        help='Show print statements')
    parser.add_argument('--failfast', '-x', action='store_true',
                        help='Stop on first failure')

    args = parser.parse_args()

    # Handle coverage mode
    if args.category == 'coverage':
        sys.exit(run_with_coverage())

    # Handle suites
    if args.category in SUITES:
        categories = SUITES[args.category]
    else:
        categories = [args.category]

    print_header(f"Test Runner: {args.category.upper()}", char='#')
    print(f"Categories: {', '.join(categories)}")

    total_start = time.time()
    results = {}

    for category in categories:
        result = run_category(
            category,
            verbose=args.verbose,
            quiet=args.quiet,
            no_capture=args.no_capture,
            failfast=args.failfast
        )
        results[category] = result

        if result != 0 and args.failfast:
            break

    total_elapsed = time.time() - total_start

    # Summary
    print_header("Test Summary", char='#')

    all_passed = True
    for category, result in results.items():
        status = "✅ PASS" if result == 0 else "❌ FAIL"
        print(f"  {category:15} {status}")
        if result != 0:
            all_passed = False

    print(f"\nTotal time: {total_elapsed:.1f}s")

    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
