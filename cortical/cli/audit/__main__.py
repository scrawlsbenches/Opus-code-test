"""
Entry point for running audit CLI as a module.

Usage:
    python -m cortical.cli.audit scan cortical/
    python -m cortical.cli.audit generate cortical/ -o docs/audits/
    python -m cortical.cli.audit train docs/audits/
"""

import argparse
import sys

from . import generate, train, scan, patterns, similar, index, health, reason, discover, pattern


def main():
    """Main entry point for audit CLI."""
    parser = argparse.ArgumentParser(
        prog='python -m cortical.cli.audit',
        description='Audit CLI - Codebase Quality Analysis Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
    generate   Generate training data from codebase comments
    train      Train classifiers from labeled findings
    scan       Scan for suspicious comments
    patterns   Find repeated patterns in comments
    similar    Find similar comments using LSH
    index      Build search indexes
    health     Analyze codebase health
    reason     PLN-based audit reasoning
    discover   WovenMind pattern discovery (experimental)

Workflow:
    python -m cortical.cli.audit generate cortical/ -o docs/audits/
    python -m cortical.cli.audit train docs/audits/
    python -m cortical.cli.audit scan cortical/
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Set up all command parsers
    generate.setup_args(subparsers)
    train.setup_args(subparsers)
    scan.setup_args(subparsers)
    patterns.setup_args(subparsers)
    similar.setup_args(subparsers)
    index.setup_args(subparsers)
    health.setup_args(subparsers)
    reason.setup_args(subparsers)
    discover.setup_args(subparsers)
    pattern.setup_args(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handler
    command_handlers = {
        'generate': generate.run,
        'train': train.run,
        'scan': scan.run,
        'patterns': patterns.run,
        'similar': similar.run,
        'index': index.run,
        'health': health.run,
        'reason': reason.run,
        'discover': discover.run,
        'pattern': pattern.run,
    }

    handler = command_handlers.get(args.command)
    if handler is None:
        print(f"Error: Unknown command: {args.command}")
        sys.exit(1)

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
