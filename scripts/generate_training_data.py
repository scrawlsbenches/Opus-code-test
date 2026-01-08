#!/usr/bin/env python3
"""
Training Data Generator for Audit Comment Classifier

Generates labeled training data for the Naive Bayes classifier by extracting
real comments from the codebase and categorizing them based on patterns.

Output files:
  - docs/audits/misleading.txt - Comments that are speculative, outdated, or misleading
  - docs/audits/accurate.txt - Comments that are factual and accurate

Usage:
    python scripts/generate_training_data.py                    # Generate from cortical/
    python scripts/generate_training_data.py --directory src/   # Generate from specific dir
    python scripts/generate_training_data.py --dry-run          # Preview without writing
    python scripts/generate_training_data.py --verbose          # Show all extracted comments

After generation, train the classifier:
    python scripts/audit_tool.py train docs/audits/
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# PATTERN DEFINITIONS
# =============================================================================

# Patterns that indicate MISLEADING comments (speculative, outdated, uncertain)
MISLEADING_PATTERNS = [
    # Speculative "will be" patterns
    (r'will be implemented', 'speculative'),
    (r'will be added', 'speculative'),
    (r'will be handled', 'speculative'),
    (r'will be replaced', 'speculative'),
    (r'will be done', 'speculative'),
    (r'will be fixed', 'speculative'),

    # FUTURE markers (often outdated)
    (r'^FUTURE:', 'future_marker'),
    (r'FUTURE\s*when', 'future_marker'),

    # "When X is implemented" patterns
    (r'when .* is implemented', 'speculative'),
    (r'when .* is ready', 'speculative'),
    (r'when .* is done', 'speculative'),
    (r'when feature is', 'speculative'),

    # Placeholder/stub markers
    (r'placeholder', 'placeholder'),
    (r'\bstub\b', 'placeholder'),
    (r'not implemented yet', 'placeholder'),

    # Vague future references
    (r'eventually', 'vague'),
    (r'someday', 'vague'),
    (r'in the future', 'vague'),
    (r'later we', 'vague'),
    (r'planned to', 'vague'),

    # Potentially stale references
    (r'See:.*\.md', 'doc_reference'),  # May reference deleted docs
    (r'see docs/', 'doc_reference'),
]

# Patterns that indicate ACCURATE comments (factual, documented behavior)
ACCURATE_PATTERNS = [
    # Return value documentation
    (r'^Returns?\s+', 'returns'),
    (r'^Returns:\s+', 'returns'),
    (r'returns\s+(True|False|None|the|a|an)\b', 'returns'),

    # Parameter documentation
    (r'^Args?:\s*', 'args'),
    (r'^Parameters?:\s*', 'args'),
    (r'^Params?:\s*', 'args'),

    # Exception documentation
    (r'^Raises?\s+', 'raises'),
    (r'^Raises:\s+', 'raises'),
    (r'raises\s+(ValueError|TypeError|KeyError|RuntimeError)', 'raises'),

    # Implementation facts
    (r'^This (is|uses|implements|creates|computes|validates)', 'implementation'),
    (r'^Implements\s+', 'implementation'),
    (r'^Uses\s+', 'implementation'),
    (r'^Creates\s+', 'implementation'),

    # Complexity/performance notes
    (r'O\([nN1]\)', 'complexity'),
    (r'O\(n\s*(log\s*n)?\)', 'complexity'),
    (r'runs in O\(', 'complexity'),
    (r'time complexity', 'complexity'),

    # Type annotations
    (r'^type:\s*', 'type_hint'),

    # Factual notes
    (r'^NOTE:\s+\w', 'note'),
    (r'^IMPORTANT:\s+', 'note'),

    # Valid TODOs with specific actions
    (r'^TODO:\s+[A-Z]', 'todo'),  # TODO with capitalized action
    (r'^FIXME:\s+[A-Z]', 'todo'),  # FIXME with capitalized action
]

# Comments to EXCLUDE (noise, not useful for training)
EXCLUDE_PATTERNS = [
    r'^-+$',           # Separator lines
    r'^=+$',           # Separator lines
    r'^\s*$',          # Empty
    r'^#\s*$',         # Just hash
    r'^\d+$',          # Just numbers
    r'^[a-z]$',        # Single letter
    r'^type:\s*ignore', # Type ignore comments
    r'^noqa',          # Linter ignores
    r'^pylint',        # Linter directives
    r'^pragma',        # Pragma directives
]


# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_comments_from_file(file_path: str) -> List[Tuple[int, str]]:
    """Extract comments from a Python file."""
    comments = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Find Python comments (# ...)
                match = re.search(r'#\s*(.+)$', line)
                if match:
                    comment_text = match.group(1).strip()
                    if comment_text and len(comment_text) > 3:
                        comments.append((line_num, comment_text))
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
    return comments


def find_python_files(directory: str) -> List[str]:
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def should_exclude(comment: str) -> bool:
    """Check if comment should be excluded from training."""
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, comment, re.IGNORECASE):
            return True
    return False


def classify_comment(comment: str) -> Tuple[str, str, str]:
    """
    Classify a comment as misleading, accurate, or unknown.

    Returns: (classification, pattern_type, matched_pattern)
    """
    comment_lower = comment.lower()

    # Check misleading patterns first
    for pattern, pattern_type in MISLEADING_PATTERNS:
        if re.search(pattern, comment_lower, re.IGNORECASE):
            return ('misleading', pattern_type, pattern)

    # Check accurate patterns
    for pattern, pattern_type in ACCURATE_PATTERNS:
        if re.search(pattern, comment, re.IGNORECASE):  # Some patterns are case-sensitive
            return ('accurate', pattern_type, pattern)

    return ('unknown', '', '')


def generate_training_data(
    directory: str,
    max_per_class: int = 50,
    verbose: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Generate training data from codebase.

    Returns: (misleading_comments, accurate_comments)
    """
    misleading = []
    accurate = []
    seen = set()  # Avoid duplicates

    # Track statistics
    stats = {
        'total_comments': 0,
        'excluded': 0,
        'misleading': 0,
        'accurate': 0,
        'unknown': 0,
        'by_pattern': {}
    }

    python_files = find_python_files(directory)
    print(f"Scanning {len(python_files)} Python files in {directory}...")

    for file_path in python_files:
        comments = extract_comments_from_file(file_path)

        for line_num, comment in comments:
            stats['total_comments'] += 1

            # Skip excluded patterns
            if should_exclude(comment):
                stats['excluded'] += 1
                continue

            # Skip duplicates
            comment_normalized = comment.lower().strip()
            if comment_normalized in seen:
                continue
            seen.add(comment_normalized)

            # Classify
            classification, pattern_type, pattern = classify_comment(comment)

            if classification == 'misleading' and len(misleading) < max_per_class:
                misleading.append(comment)
                stats['misleading'] += 1
                stats['by_pattern'][pattern_type] = stats['by_pattern'].get(pattern_type, 0) + 1
                if verbose:
                    print(f"  [MISLEADING] {file_path}:{line_num} - {comment[:60]}...")

            elif classification == 'accurate' and len(accurate) < max_per_class:
                accurate.append(comment)
                stats['accurate'] += 1
                stats['by_pattern'][pattern_type] = stats['by_pattern'].get(pattern_type, 0) + 1
                if verbose:
                    print(f"  [ACCURATE] {file_path}:{line_num} - {comment[:60]}...")
            else:
                stats['unknown'] += 1

    # Print statistics
    print(f"\nExtraction Statistics:")
    print(f"  Total comments scanned: {stats['total_comments']}")
    print(f"  Excluded (noise): {stats['excluded']}")
    print(f"  Classified as misleading: {stats['misleading']}")
    print(f"  Classified as accurate: {stats['accurate']}")
    print(f"  Unknown (not used): {stats['unknown']}")
    print(f"\nPattern breakdown:")
    for pattern_type, count in sorted(stats['by_pattern'].items(), key=lambda x: -x[1]):
        print(f"    {pattern_type}: {count}")

    return misleading, accurate


def write_training_files(
    misleading: List[str],
    accurate: List[str],
    output_dir: str = "docs/audits"
) -> None:
    """Write training data to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write misleading.txt
    misleading_file = output_path / "misleading.txt"
    with open(misleading_file, 'w', encoding='utf-8') as f:
        f.write("# Misleading comments - speculative, outdated, or referencing non-existent resources\n")
        f.write("# Auto-generated by scripts/generate_training_data.py\n")
        f.write("# One comment per line\n\n")
        for comment in misleading:
            # Clean up comment for training
            clean = comment.strip()
            if clean:
                f.write(clean + "\n")
    print(f"Wrote {len(misleading)} examples to {misleading_file}")

    # Write accurate.txt
    accurate_file = output_path / "accurate.txt"
    with open(accurate_file, 'w', encoding='utf-8') as f:
        f.write("# Accurate comments - factual descriptions, valid documentation\n")
        f.write("# Auto-generated by scripts/generate_training_data.py\n")
        f.write("# One comment per line\n\n")
        for comment in accurate:
            clean = comment.strip()
            if clean:
                f.write(clean + "\n")
    print(f"Wrote {len(accurate)} examples to {accurate_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for audit comment classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--directory", "-d",
        default="cortical/",
        help="Directory to scan for comments (default: cortical/)"
    )
    parser.add_argument(
        "--output", "-o",
        default="docs/audits",
        help="Output directory for training files (default: docs/audits)"
    )
    parser.add_argument(
        "--max-per-class", "-m",
        type=int,
        default=50,
        help="Maximum examples per class (default: 50)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview extraction without writing files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show all extracted comments"
    )
    parser.add_argument(
        "--include-scripts",
        action="store_true",
        help="Also scan scripts/ directory"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Training Data Generator for Audit Comment Classifier")
    print("=" * 60)
    print()

    # Collect directories to scan
    directories = [args.directory]
    if args.include_scripts:
        directories.append("scripts/")

    all_misleading = []
    all_accurate = []

    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Warning: Directory not found: {directory}")
            continue

        misleading, accurate = generate_training_data(
            directory,
            max_per_class=args.max_per_class,
            verbose=args.verbose
        )
        all_misleading.extend(misleading)
        all_accurate.extend(accurate)

    # Deduplicate
    all_misleading = list(dict.fromkeys(all_misleading))[:args.max_per_class]
    all_accurate = list(dict.fromkeys(all_accurate))[:args.max_per_class]

    print(f"\nFinal counts:")
    print(f"  Misleading examples: {len(all_misleading)}")
    print(f"  Accurate examples: {len(all_accurate)}")

    if args.dry_run:
        print("\n[DRY RUN] Would write to:")
        print(f"  - {args.output}/misleading.txt")
        print(f"  - {args.output}/accurate.txt")
        print("\nSample misleading comments:")
        for c in all_misleading[:5]:
            print(f"  - {c[:70]}...")
        print("\nSample accurate comments:")
        for c in all_accurate[:5]:
            print(f"  - {c[:70]}...")
    else:
        write_training_files(all_misleading, all_accurate, args.output)
        print(f"\nTraining data written to {args.output}/")
        print("\nNext steps:")
        print("  python scripts/audit_tool.py train docs/audits/")


if __name__ == "__main__":
    main()
