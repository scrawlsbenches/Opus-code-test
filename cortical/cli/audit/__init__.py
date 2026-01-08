"""
Audit CLI Commands - Codebase Quality Analysis Tools.

This package provides CLI commands for the audit system:
    - generate: Generate training data from codebase comments
    - train: Train classifiers from labeled findings
    - scan: Scan for suspicious comments
    - patterns: Find repeated patterns in comments
    - similar: Find similar comments using LSH
    - index: Build search indexes
    - health: Analyze codebase health
    - reason: PLN-based audit reasoning
    - discover: WovenMind pattern discovery (experimental)

Usage:
    from cortical.cli.audit import setup_parser, handle_command

    # In main CLI setup
    setup_parser(subparsers)

    # Dispatch command
    if args.command == 'audit':
        handle_command(args)
"""

from typing import Any
import argparse


def setup_parser(subparsers) -> None:
    """Set up the audit command and its subcommands."""
    audit_parser = subparsers.add_parser(
        'audit',
        help='Codebase quality analysis tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Audit commands for codebase quality analysis.

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
    1. Generate training data:  audit generate cortical/
    2. Train the classifier:    audit train docs/audits/
    3. Scan for issues:         audit scan cortical/
"""
    )

    audit_subparsers = audit_parser.add_subparsers(
        dest='audit_command',
        help='Audit subcommand'
    )

    # Import command modules and set up parsers
    from . import generate, train, scan, patterns, similar, index, health, reason, discover

    generate.setup_args(audit_subparsers)
    train.setup_args(audit_subparsers)
    scan.setup_args(audit_subparsers)
    patterns.setup_args(audit_subparsers)
    similar.setup_args(audit_subparsers)
    index.setup_args(audit_subparsers)
    health.setup_args(audit_subparsers)
    reason.setup_args(audit_subparsers)
    discover.setup_args(audit_subparsers)


def handle_command(args: Any) -> int:
    """
    Handle an audit command.

    Args:
        args: Parsed command line arguments with audit_command set.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    if not hasattr(args, 'audit_command') or args.audit_command is None:
        print("Error: No audit subcommand specified.")
        print("Use 'audit --help' to see available commands.")
        return 1

    # Import command modules
    from . import generate, train, scan, patterns, similar, index, health, reason, discover

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
    }

    handler = command_handlers.get(args.audit_command)
    if handler is None:
        print(f"Error: Unknown audit command: {args.audit_command}")
        return 1

    try:
        handler(args)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1
