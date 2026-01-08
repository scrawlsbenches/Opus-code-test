"""
Generate command - Extract training data from codebase comments.
"""

import os
import re
from pathlib import Path

from ._base import (
    find_python_files,
    extract_comments_from_file,
    MISLEADING_PATTERNS,
    ACCURATE_PATTERNS,
    EXCLUDE_PATTERNS,
)

# ==============================================================================
# COMMAND METADATA
# ==============================================================================

NAME = 'generate'
HELP = 'Generate training data from codebase comments'


def setup_args(parser):
    """Set up command arguments."""
    parser.add_argument('directory', help='Directory to scan for comments')
    parser.add_argument('-o', '--output', default='docs/audits',
                        help='Output directory for training files (default: docs/audits)')
    parser.add_argument('-m', '--max-per-class', type=int, default=50,
                        help='Maximum examples per class (default: 50)')
    parser.add_argument('--include-scripts', action='store_true',
                        help='Also scan scripts/ directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without writing files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show all extracted comments')


# ==============================================================================
# HELPERS
# ==============================================================================

def should_exclude_comment(comment: str) -> bool:
    """Check if comment should be excluded from training."""
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, comment, re.IGNORECASE):
            return True
    return False


def classify_comment(comment: str) -> tuple:
    """Classify a comment as misleading, accurate, or unknown."""
    comment_lower = comment.lower()

    for pattern, pattern_type in MISLEADING_PATTERNS:
        if re.search(pattern, comment_lower, re.IGNORECASE):
            return ('misleading', pattern_type, pattern)

    for pattern, pattern_type in ACCURATE_PATTERNS:
        if re.search(pattern, comment, re.IGNORECASE):
            return ('accurate', pattern_type, pattern)

    return ('unknown', '', '')


# ==============================================================================
# COMMAND IMPLEMENTATION
# ==============================================================================

def run(args):
    """Execute the generate command."""
    print("=" * 70)
    print("  Training Data Generator")
    print("=" * 70)

    directories = [args.directory]
    if getattr(args, 'include_scripts', False):
        directories.append("scripts/")

    misleading = []
    accurate = []
    seen = set()
    max_per_class = getattr(args, 'max_per_class', 50)

    stats = {
        'total_comments': 0,
        'excluded': 0,
        'misleading': 0,
        'accurate': 0,
        'unknown': 0,
        'by_pattern': {}
    }

    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Warning: Directory not found: {directory}")
            continue

        python_files = find_python_files(directory)
        print(f"\nScanning {len(python_files)} Python files in {directory}...")

        for file_path in python_files:
            comments = extract_comments_from_file(file_path)

            for line_num, comment in comments:
                stats['total_comments'] += 1

                if len(comment) <= 3:
                    stats['excluded'] += 1
                    continue

                if should_exclude_comment(comment):
                    stats['excluded'] += 1
                    continue

                comment_normalized = comment.lower().strip()
                if comment_normalized in seen:
                    continue
                seen.add(comment_normalized)

                classification, pattern_type, _ = classify_comment(comment)

                if classification == 'misleading' and len(misleading) < max_per_class:
                    misleading.append(comment)
                    stats['misleading'] += 1
                    stats['by_pattern'][pattern_type] = stats['by_pattern'].get(pattern_type, 0) + 1
                    if getattr(args, 'verbose', False):
                        print(f"  [MISLEADING] {file_path}:{line_num}")

                elif classification == 'accurate' and len(accurate) < max_per_class:
                    accurate.append(comment)
                    stats['accurate'] += 1
                    stats['by_pattern'][pattern_type] = stats['by_pattern'].get(pattern_type, 0) + 1
                    if getattr(args, 'verbose', False):
                        print(f"  [ACCURATE] {file_path}:{line_num}")
                else:
                    stats['unknown'] += 1

    # Print statistics
    print(f"\nExtraction Statistics:")
    print(f"  Total comments scanned: {stats['total_comments']}")
    print(f"  Excluded (noise): {stats['excluded']}")
    print(f"  Classified as misleading: {stats['misleading']}")
    print(f"  Classified as accurate: {stats['accurate']}")
    print(f"  Unknown (not used): {stats['unknown']}")

    if stats['by_pattern']:
        print(f"\nPattern breakdown:")
        for pattern_type, count in sorted(stats['by_pattern'].items(), key=lambda x: -x[1]):
            print(f"    {pattern_type}: {count}")

    misleading = list(dict.fromkeys(misleading))[:max_per_class]
    accurate = list(dict.fromkeys(accurate))[:max_per_class]

    print(f"\nFinal counts:")
    print(f"  Misleading examples: {len(misleading)}")
    print(f"  Accurate examples: {len(accurate)}")

    if getattr(args, 'dry_run', False):
        print("\n[DRY RUN] Would write to:")
        print(f"  - {args.output}/misleading.txt")
        print(f"  - {args.output}/accurate.txt")
        return

    # Write training files
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    misleading_file = output_path / "misleading.txt"
    with open(misleading_file, 'w', encoding='utf-8') as f:
        f.write("# Misleading comments - speculative, outdated, or referencing non-existent resources\n")
        f.write("# Auto-generated by: python scripts/audit_tool.py generate\n")
        f.write("# One comment per line\n\n")
        for comment in misleading:
            f.write(comment.strip() + "\n")
    print(f"\nWrote {len(misleading)} examples to {misleading_file}")

    accurate_file = output_path / "accurate.txt"
    with open(accurate_file, 'w', encoding='utf-8') as f:
        f.write("# Accurate comments - factual descriptions, valid documentation\n")
        f.write("# Auto-generated by: python scripts/audit_tool.py generate\n")
        f.write("# One comment per line\n\n")
        for comment in accurate:
            f.write(comment.strip() + "\n")
    print(f"Wrote {len(accurate)} examples to {accurate_file}")

    print(f"\n" + "=" * 70)
    print("Generation complete!")
    print(f"\nNext step: python scripts/audit_tool.py train {args.output}")
