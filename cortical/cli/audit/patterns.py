"""
Patterns command - Find repeated patterns in comments.

Uses Suffix Array to find copy-pasted text and Count-Min Sketch for
frequency tracking.
"""

from typing import Any, List, Tuple

from ._base import print_header, print_separator


def is_separator_line(text: str) -> bool:
    """
    Check if text is a separator line (>80% repeated chars like =, -, #).

    These lines are noise in pattern analysis - they appear frequently
    but carry no semantic meaning.
    """
    if len(text) < 10:
        return False
    # Count occurrences of each character
    char_counts: dict = {}
    for c in text:
        char_counts[c] = char_counts.get(c, 0) + 1
    most_common_count = max(char_counts.values())
    return most_common_count / len(text) > 0.8


def filter_separator_patterns(
    patterns: List[Tuple[str, int]]
) -> List[Tuple[str, int]]:
    """Filter out separator line patterns from results."""
    return [(p, c) for p, c in patterns if not is_separator_line(p)]


def setup_args(subparsers) -> None:
    """Set up command arguments."""
    parser = subparsers.add_parser(
        'patterns',
        help='Find repeated patterns in comments'
    )
    parser.add_argument('directory', help='Directory to analyze')
    parser.add_argument(
        '--min-length',
        type=int,
        default=15,
        help='Minimum pattern length (default: 15)'
    )


def run(args: Any) -> None:
    """Execute the patterns command."""
    from cortical.audits import (
        find_python_files,
        extract_comments_from_file,
        CommentPatternFinder,
        PatternFrequencySketch,
    )

    print(f"Finding repeated patterns in {args.directory}...")
    print_separator()

    # Collect all comments
    python_files = find_python_files(args.directory)
    print(f"Found {len(python_files)} Python files\n")

    all_comments = []
    comment_map = {}  # Map position to (file, line)

    current_pos = 0
    for file_path in python_files:
        comments = extract_comments_from_file(file_path)

        for line_num, comment in comments:
            all_comments.append(comment)
            comment_map[current_pos] = (file_path, line_num)
            current_pos += len(comment) + 1  # +1 for separator

    if not all_comments:
        print("No comments found.")
        return

    # Build combined text with separator
    combined_text = " ".join(all_comments)

    print(f"Analyzing {len(all_comments)} comments "
          f"({len(combined_text)} characters)...")

    # Build Suffix Array
    print("\nBuilding suffix array...")
    pattern_finder = CommentPatternFinder(combined_text)

    # Find repeated substrings
    min_length = getattr(args, 'min_length', 15)
    print(f"Finding patterns (min length: {min_length})...\n")

    repeated = pattern_finder.repeated_substrings(min_length=min_length)

    # Filter out separator lines (noise like ====, ----, ####)
    repeated = filter_separator_patterns(repeated)

    if not repeated:
        print(f"No repeated patterns of length >= {min_length} found.")
        return

    # Initialize Count-Min Sketch for frequency tracking
    # Use larger width (10000) to reduce hash collisions
    sketch = PatternFrequencySketch(width=10000, depth=5)

    for pattern, count in repeated:
        sketch.add(pattern, count)

    # Report top patterns
    print(f"Found {len(repeated)} repeated patterns\n")
    print("TOP REPEATED PATTERNS:")
    print_separator()

    # Show top 20 patterns
    for i, (pattern, count) in enumerate(repeated[:20], 1):
        # Clean up pattern for display
        display_pattern = pattern[:60] + "..." if len(pattern) > 60 else pattern
        estimated_freq = sketch.query(pattern)

        print(f"\n{i}. Pattern (length={len(pattern)}, count={count}, "
              f"est={estimated_freq}):")
        print(f'   "{display_pattern}"')

        # Find where it appears
        positions = pattern_finder.search(pattern)
        if positions:
            print(f"   Appears at {len(positions)} locations")

    print_separator()
