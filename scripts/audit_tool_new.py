#!/usr/bin/env python3
"""
Unified Audit Tool for Cortical Codebase Maintenance

This tool integrates algorithm implementations from cortical/audits/algorithms
to help maintain code quality during refactoring.

Commands are auto-discovered from the audit_commands/ package.

Workflow:
    python scripts/audit_tool.py generate cortical/ --include-scripts
    python scripts/audit_tool.py train docs/audits/
    python scripts/audit_tool.py scan cortical/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import command registry
from audit_commands import get_commands, setup_all_parsers, run_command


def main():
    parser = argparse.ArgumentParser(
        description="Unified Audit Tool for Cortical Codebase Maintenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Auto-register all discovered commands
    setup_all_parsers(subparsers)

    # Show available commands in help
    commands = get_commands()
    if commands:
        epilog_lines = ["\nAvailable commands:"]
        for name, cmd in sorted(commands.items()):
            epilog_lines.append(f"  {name:12} - {cmd['help']}")
        parser.epilog = "\n".join(epilog_lines)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return run_command(args.command, args)


if __name__ == '__main__':
    sys.exit(main())
