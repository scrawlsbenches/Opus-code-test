#!/usr/bin/env python3
"""
Test Runner Script
==================

Convenient script for running different test categories locally.

Automatically installs pytest/coverage if missing (can be disabled).

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
    -v, --verbose       Show verbose output
    -q, --quiet         Show minimal output
    --no-capture        Show print statements (pytest -s)
    --failfast, -x      Stop on first failure
    --no-auto-install   Do not auto-install missing dependencies
    --check-deps        Only check dependencies, do not run tests

Dependency Handling:
    By default, this script will attempt to install pytest and coverage
    if they are not available. Use --no-auto-install to disable this.
    Use --check-deps to see what's installed without running tests.
"""

import argparse
import subprocess
import sys
import os
import time

# Test dependencies with their pip package names
TEST_DEPENDENCIES = {
    'pytest': 'pytest',
    'coverage': 'coverage',
}

# Cache for dependency check results
_deps_checked = False
_deps_available = {}


def check_dependency(module_name):
    """Check if a Python module is importable."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def install_dependency(package_name, quiet=True):
    """Attempt to install a package via pip."""
    cmd = [sys.executable, '-m', 'pip', 'install', package_name]
    if quiet:
        cmd.append('-q')

    try:
        result = subprocess.run(cmd, capture_output=quiet, timeout=120)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False


def ensure_test_dependencies(auto_install=True, verbose=False):
    """
    Ensure test dependencies are available.

    Args:
        auto_install: If True, attempt to install missing dependencies
        verbose: If True, print status messages

    Returns:
        dict: {module_name: bool} indicating availability

    This function:
    1. Checks if pytest/coverage are installed
    2. If missing and auto_install=True, attempts to install them
    3. Returns availability status for each dependency
    4. Caches results to avoid repeated checks
    """
    global _deps_checked, _deps_available

    # Return cached results if already checked
    if _deps_checked:
        return _deps_available

    missing = []

    for module_name, package_name in TEST_DEPENDENCIES.items():
        if check_dependency(module_name):
            _deps_available[module_name] = True
            if verbose:
                print(f"  ✓ {module_name} available")
        else:
            missing.append((module_name, package_name))
            _deps_available[module_name] = False

    # Attempt to install missing dependencies
    if missing and auto_install:
        if verbose:
            print(f"\n📦 Installing missing test dependencies...")

        for module_name, package_name in missing:
            if verbose:
                print(f"  Installing {package_name}...", end=' ', flush=True)

            if install_dependency(package_name, quiet=not verbose):
                _deps_available[module_name] = True
                if verbose:
                    print("✓")
            else:
                if verbose:
                    print("✗")
                # Don't print error here - let caller handle it

    _deps_checked = True
    return _deps_available


def get_fallback_command(category):
    """
    Get a fallback unittest command when pytest is unavailable.

    Returns a command that will work without pytest.
    """
    # Map categories to unittest-compatible test discovery
    paths = []

    if category in CATEGORIES:
        paths = list(CATEGORIES[category].get('paths', []))
        paths.extend(CATEGORIES[category].get('also_run', []))
    elif category in SUITES:
        for cat in SUITES[category]:
            paths.extend(CATEGORIES[cat].get('paths', []))
            paths.extend(CATEGORIES[cat].get('also_run', []))

    return [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']


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
    """Run pytest with the given paths and options.

    If pytest is not available, falls back to unittest with a warning.
    """
    deps = ensure_test_dependencies(auto_install=True, verbose=verbose)

    if not deps.get('pytest', False):
        print("\n⚠️  pytest not available and could not be installed.")
        print("   Falling back to unittest (some features may not work).")
        print("   To install manually: pip install pytest\n")
        return subprocess.run(
            [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
        ).returncode

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
    # Only add --no-cov if pytest-cov is installed
    if no_coverage:
        try:
            import pytest_cov
            cmd.append('--no-cov')
        except ImportError:
            pass  # pytest-cov not installed, coverage is already disabled

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

    # Collect all test paths (directory + legacy files)
    paths = list(config['paths'])

    # Add legacy test files if they exist
    if 'also_run' in config:
        existing_files = [f for f in config['also_run'] if os.path.exists(f)]
        paths.extend(existing_files)

    no_coverage = config.get('no_coverage', False)

    # Run all tests together in a single pytest call
    # This matches CI behavior and avoids issues with empty directories
    result = run_pytest(
        paths,
        verbose=verbose,
        quiet=quiet,
        no_capture=no_capture,
        failfast=failfast,
        no_coverage=no_coverage
    )

    elapsed = time.time() - start_time

    if result == 0:
        print(f"\n✅ {category.upper()} tests PASSED in {elapsed:.1f}s")
    else:
        print(f"\n❌ {category.upper()} tests FAILED in {elapsed:.1f}s")

    return result


def run_with_coverage():
    """Run full test suite with coverage.

    If coverage is not available, attempts to install it first.
    Falls back to running tests without coverage if installation fails.
    """
    print_header("Running Full Test Suite with Coverage")

    deps = ensure_test_dependencies(auto_install=True, verbose=True)

    if not deps.get('coverage', False):
        print("\n⚠️  coverage not available and could not be installed.")
        print("   Running tests without coverage.")
        print("   To install manually: pip install coverage\n")
        return subprocess.run(
            [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v']
        ).returncode

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
    parser.add_argument('--no-auto-install', action='store_true',
                        help='Do not auto-install missing dependencies')
    parser.add_argument('--check-deps', action='store_true',
                        help='Only check dependencies, do not run tests')

    args = parser.parse_args()

    # Handle dependency check mode
    if args.check_deps:
        print("Checking test dependencies...")
        deps = ensure_test_dependencies(auto_install=False, verbose=True)
        missing = [k for k, v in deps.items() if not v]
        if missing:
            print(f"\n❌ Missing: {', '.join(missing)}")
            print(f"   Install with: pip install {' '.join(missing)}")
            sys.exit(1)
        else:
            print("\n✅ All test dependencies available")
            sys.exit(0)

    # Pre-check dependencies (silently auto-install unless --no-auto-install)
    if not args.no_auto_install and not args.quiet:
        deps = ensure_test_dependencies(auto_install=True, verbose=args.verbose)
        missing = [k for k, v in deps.items() if not v]
        if missing and not args.quiet:
            print(f"⚠️  Some dependencies unavailable: {', '.join(missing)}")
            print("   Tests will run with reduced functionality.\n")

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
