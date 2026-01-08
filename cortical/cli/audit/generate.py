"""
Generate command - Extract training data from codebase comments.

This command scans Python files and extracts comments that can be used
to train the Naive Bayes classifier for misleading vs accurate classification.
"""

from pathlib import Path
from typing import Any

from ._base import print_header, print_separator


def setup_args(subparsers) -> None:
    """Set up command arguments."""
    parser = subparsers.add_parser(
        'generate',
        help='Generate training data from codebase comments'
    )
    parser.add_argument('directory', help='Directory to scan for comments')
    parser.add_argument(
        '-o', '--output',
        default='docs/audits',
        help='Output directory for training files (default: docs/audits)'
    )
    parser.add_argument(
        '-m', '--max-per-class',
        type=int,
        default=50,
        help='Maximum examples per class (default: 50)'
    )
    parser.add_argument(
        '--include-scripts',
        action='store_true',
        help='Also scan scripts/ directory'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without writing files'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show all extracted comments'
    )


def run(args: Any) -> None:
    """Execute the generate command."""
    from cortical.audits import (
        generate_training_data,
        write_training_files,
    )

    print_header("Training Data Generator")

    directories = [args.directory]
    if getattr(args, 'include_scripts', False):
        directories.append("scripts/")

    max_per_class = getattr(args, 'max_per_class', 50)
    verbose = getattr(args, 'verbose', False)

    # Generate training data
    training_data = generate_training_data(
        directories=directories,
        max_per_class=max_per_class,
        verbose=verbose,
    )

    # Print statistics
    stats = training_data.stats
    print(f"\nExtraction Statistics:")
    print(f"  Total comments scanned: {stats.total_comments}")
    print(f"  Excluded (noise): {stats.excluded}")
    print(f"  Classified as misleading: {stats.misleading}")
    print(f"  Classified as accurate: {stats.accurate}")
    print(f"  Unknown (not used): {stats.unknown}")

    if stats.by_pattern:
        print(f"\nPattern breakdown:")
        for pattern_type, count in sorted(
            stats.by_pattern.items(),
            key=lambda x: -x[1]
        ):
            print(f"    {pattern_type}: {count}")

    print(f"\nFinal counts:")
    print(f"  Misleading examples: {len(training_data.misleading)}")
    print(f"  Accurate examples: {len(training_data.accurate)}")

    # Handle dry-run
    if getattr(args, 'dry_run', False):
        print("\n[DRY RUN] Would write to:")
        print(f"  - {args.output}/misleading.txt")
        print(f"  - {args.output}/accurate.txt")
        print("\nSample misleading comments:")
        for c in training_data.misleading[:5]:
            display = c[:70] + "..." if len(c) > 70 else c
            print(f"  - {display}")
        print("\nSample accurate comments:")
        for c in training_data.accurate[:5]:
            display = c[:70] + "..." if len(c) > 70 else c
            print(f"  - {display}")
        return

    # Write training files
    output_path = Path(args.output)
    write_training_files(training_data, output_path)

    print_separator()
    print("Generation complete!")
    print(f"\nNext step: audit train {args.output}")
