"""
Entry point for python -m cortical.cognitive.

This module exists to properly handle CLI execution without class identity
issues that occur when a module is both imported (via __init__.py) and
executed as __main__.

Why this pattern?
-----------------
When running `python -m cortical.cognitive.training`:
1. Python imports the package, running __init__.py
2. If __init__.py imports IncrementalTrainer, it creates the class
3. Then Python runs training.py as __main__, potentially creating a
   DIFFERENT class object with the same name
4. DI containers use class objects as keys, so resolution fails

Solution:
---------
- This __main__.py is the ONLY entry point for CLI execution
- __init__.py uses lazy imports (doesn't import training at module level)
- training.py has no `if __name__ == "__main__"` block
- Class identity is preserved because there's only one import path

Usage:
    python -m cortical.cognitive --help
    python -m cortical.cognitive train samples/
    python -m cortical.cognitive status
    python -m cortical.cognitive reindex
"""

import argparse
import sys


def main() -> int:
    """Main entry point for cognitive CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m cortical.cognitive",
        description="Cognitive Agent training and management CLI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train on documents")
    train_parser.add_argument(
        "directory",
        nargs="?",
        default="samples/",
        help="Directory containing training documents (default: samples/)",
    )
    train_parser.add_argument(
        "--model-dir",
        default="models/cognitive_agent",
        help="Directory for model storage (default: models/cognitive_agent)",
    )
    train_parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob pattern for files (default: *.txt)",
    )
    train_parser.add_argument(
        "--files",
        nargs="+",
        help="Train on specific files instead of directory",
    )
    train_parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain all documents",
    )
    train_parser.add_argument(
        "--batch-size", "-n",
        type=int,
        help="Limit training to N documents (for controlled batch training)",
    )
    train_parser.add_argument(
        "--checkpoint", "-c",
        type=int,
        help="Checkpoint interval (save every N documents)",
    )
    train_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show training status")
    status_parser.add_argument(
        "--model-dir",
        default="models/cognitive_agent",
        help="Directory for model storage",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List trained documents")
    list_parser.add_argument(
        "--model-dir",
        default="models/cognitive_agent",
        help="Directory for model storage",
    )

    # Reindex command
    reindex_parser = subparsers.add_parser("reindex", help="Recalculate IDF weights")
    reindex_parser.add_argument(
        "--model-dir",
        default="models/cognitive_agent",
        help="Directory for model storage",
    )
    reindex_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Import here to avoid class identity issues
    # This is the ONLY place these classes should be imported for CLI use
    from cortical.cognitive.training import IncrementalTrainer, run_cli_command

    return run_cli_command(args.command, args)


if __name__ == "__main__":
    sys.exit(main())
