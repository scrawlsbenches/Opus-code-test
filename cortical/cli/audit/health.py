"""
Health command - Analyze codebase health.

Provides comprehensive health analysis using:
- Pattern detection (Trie + Inverted Index)
- Duplicate detection (Suffix Array)
- Similar comment clustering (LSH + Union-Find)
- Git history analysis (blame, churn, stale TODOs)
- Import dependency analysis (DAG)
"""

import os
from typing import Any

from ._base import (
    print_header,
    print_separator,
)


def setup_args(subparsers) -> None:
    """Set up command arguments."""
    parser = subparsers.add_parser(
        'health',
        help='Analyze codebase health'
    )
    parser.add_argument('directory', help='Directory to analyze')
    parser.add_argument(
        '--git',
        action='store_true',
        help='Include git history analysis (requires git)'
    )
    parser.add_argument(
        '--churn-days',
        type=int,
        default=90,
        help='Number of days for churn analysis (default: 90)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )


def run(args: Any) -> None:
    """Execute the health command."""
    import json as json_module
    from cortical.audits import (
        analyze_directory,
    )

    directory = args.directory
    verbose = getattr(args, 'verbose', False)
    with_git = getattr(args, 'git', False)
    as_json = getattr(args, 'json', False)

    if not as_json:
        print(f"Analyzing codebase health: {directory}")
        print_separator()

    # Run analysis
    result = analyze_directory(
        directory=directory,
        with_git=with_git,
        verbose=not as_json,
    )

    if result.error:
        print(f"Error: {result.error}")
        return

    # Output as JSON if requested
    if as_json:
        print(json_module.dumps(result.to_dict(), indent=2))
        return

    # Summary
    print(f"\nFiles analyzed: {result.files_analyzed}")
    print(f"Comments analyzed: {result.comments_analyzed}")
    print(f"Findings: {len(result.findings)}")

    # Pattern counts
    if result.pattern_counts:
        print("\nPATTERN COUNTS:")
        print_separator()
        for pattern, count in sorted(result.pattern_counts.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count}")

    # Findings
    if result.findings and verbose:
        print("\nFINDINGS:")
        print_separator()
        for finding in result.findings[:20]:  # Limit to first 20
            rel_path = os.path.relpath(finding.get('file', ''), directory)
            line = finding.get('line', '?')
            pattern = finding.get('pattern', 'unknown')
            comment = finding.get('comment', '')[:60]
            print(f"\n  [{pattern}] {rel_path}:{line}")
            print(f"    {comment}...")

        if len(result.findings) > 20:
            print(f"\n  ... and {len(result.findings) - 20} more findings")

    # Similar groups
    if result.similar_groups:
        print(f"\nSIMILAR COMMENT GROUPS: {len(result.similar_groups)}")
        if verbose:
            print_separator()
            for i, (rep, members) in enumerate(result.similar_groups[:5]):
                print(f"\n  Group {i+1} ({len(members)} similar):")
                print(f"    Representative: {rep[:60]}...")

    # Repeated substrings
    if result.repeated_substrings:
        print(f"\nREPEATED PATTERNS: {len(result.repeated_substrings)}")
        if verbose:
            print_separator()
            for pattern, count in result.repeated_substrings[:10]:
                if len(pattern) > 10:  # Skip short matches
                    print(f"  \"{pattern[:40]}...\" (x{count})")

    # Git analysis
    if result.git_analysis:
        print("\nGIT ANALYSIS:")
        print_separator()

        if 'high_churn_files' in result.git_analysis:
            high_churn = result.git_analysis['high_churn_files']
            print(f"  High churn files: {len(high_churn)}")
            if verbose and high_churn:
                for f, count in high_churn[:5]:
                    print(f"    {f}: {count} changes")

        if 'stale_todos' in result.git_analysis:
            stale = result.git_analysis['stale_todos']
            print(f"  Stale TODOs (>30 days): {len(stale)}")
            if verbose and stale:
                for todo in stale[:3]:
                    print(f"    {todo['file']}:{todo['line']} - {todo['age_days']} days old")

    print_separator()
    print("Health analysis complete!")
