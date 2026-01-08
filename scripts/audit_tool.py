#!/usr/bin/env python3
"""
Unified Audit Tool for Cortical Codebase Maintenance

This is a thin wrapper around the cortical.cli.audit package.
The actual implementation lives in cortical/cli/audit/.

Commands:
    generate <directory>  - Generate training data from codebase comments
    train <findings_dir>  - Train classifiers from labeled findings
    scan <directory>      - Scan for suspicious comments using Bloom filter + Naive Bayes
    patterns <directory>  - Find repeated patterns using Suffix Array + Count-Min Sketch
    similar <comment>     - Find similar comments using LSH
    index <directory>     - Build search index using Inverted Index + Trie

Workflow:
    # 1. Generate training data from your codebase
    python scripts/audit_tool.py generate cortical/ --include-scripts

    # 2. Train the classifier
    python scripts/audit_tool.py train docs/audits/

    # 3. Scan for suspicious comments
    python scripts/audit_tool.py scan cortical/

Examples:
    python scripts/audit_tool.py generate cortical/ -o docs/audits/
    python scripts/audit_tool.py train docs/audits/
    python scripts/audit_tool.py scan cortical/
    python scripts/audit_tool.py patterns cortical/got/
    python scripts/audit_tool.py similar "FUTURE: When CDG index is implemented"
    python scripts/audit_tool.py index cortical/
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Main entry point - delegates to cortical.cli.audit."""
    parser = argparse.ArgumentParser(
        description="Unified Audit Tool for Cortical Codebase Maintenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Import and set up audit commands
    from cortical.cli.audit import generate, train, scan, patterns, similar, index

    generate.setup_args(subparsers)
    train.setup_args(subparsers)
    scan.setup_args(subparsers)
    patterns.setup_args(subparsers)
    similar.setup_args(subparsers)
    index.setup_args(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    command_handlers = {
        'generate': generate.run,
        'train': train.run,
        'scan': scan.run,
        'patterns': patterns.run,
        'similar': similar.run,
        'index': index.run,
    }

    handler = command_handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        handler(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
