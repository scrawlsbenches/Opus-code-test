#!/usr/bin/env python3
"""
Example CLI wrappers for common development tasks.

Philosophy: QUIET BY DEFAULT. These wrappers collect context silently
and only speak when asked.

Usage:
    # Simple - just use run()
    from cortical.cli_wrapper import run, Session

    result = run("pytest tests/")
    if not result.success:
        print(result.stderr)

    # Session tracking
    with Session() as s:
        s.run("pytest tests/")
        s.run("git commit -m 'fix'")
        if s.should_reindex():
            print("Consider re-indexing")

    # As CLI
    python scripts/cli_wrappers.py run pytest tests/ -v
    python scripts/cli_wrappers.py --summary
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cortical.cli_wrapper import (
    CLIWrapper,
    ExecutionContext,
    TaskCompletionManager,
    ContextWindowManager,
    Session,
    run,
)


# =============================================================================
# Context-Aware Wrapper Configuration
# =============================================================================

class DevWrapper:
    """
    Development-focused CLI wrapper - QUIET BY DEFAULT.

    No emoji. No unsolicited advice. Just runs commands and tracks context.
    Ask for information when you want it.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_to_file: bool = False,
    ):
        self._session = Session(git=True)
        self.log_to_file = log_to_file

        if log_to_file:
            if log_dir:
                self.log_dir = Path(log_dir)
            else:
                self.log_dir = Path('.cli_wrapper_logs')
            self.log_dir.mkdir(exist_ok=True)

            # Register file logging hook
            @self._session._wrapper.on_complete()
            def log_to_file(ctx: ExecutionContext):
                self._log_result(ctx)

    def _log_result(self, ctx: ExecutionContext):
        """Log result to file (if enabled)."""
        if not self.log_to_file:
            return

        log_file = self.log_dir / 'commands.jsonl'
        entry = {
            'timestamp': ctx.timestamp,
            'command': ctx.command_str,
            'success': ctx.success,
            'duration': ctx.duration,
            'exit_code': ctx.exit_code,
            'git_branch': ctx.git.branch,
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def run(self, command: List[str], **kwargs) -> ExecutionContext:
        """Execute a command."""
        return self._session.run(command, **kwargs)

    def summary(self) -> dict:
        """Get summary of session activity."""
        return self._session.summary()

    def should_reindex(self) -> bool:
        """Check if re-indexing is recommended."""
        return self._session.should_reindex()

    @property
    def all_passed(self) -> bool:
        """True if all commands succeeded."""
        return self._session.all_passed

    @property
    def results(self) -> List[ExecutionContext]:
        """All command results."""
        return self._session.results


# =============================================================================
# CLI Interface (simplified)
# =============================================================================

def main():
    """
    CLI entry point - just run commands with optional context.

    This is mostly for demonstration. In practice, you'd use the
    Python API directly: run() or Session.
    """
    parser = argparse.ArgumentParser(
        description='Run commands with context collection (quiet by default)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s run pytest tests/ -v
    %(prog)s run git status
    %(prog)s run echo hello --json
        """
    )

    parser.add_argument(
        'mode',
        nargs='?',
        choices=['run'],
        default='run',
        help='Mode (currently only "run" is supported)'
    )
    parser.add_argument(
        'command',
        nargs='*',
        help='Command to execute'
    )
    parser.add_argument(
        '--git',
        action='store_true',
        help='Collect git context'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output result as JSON'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run the command
    result = run(args.command, git=args.git)

    # Output
    if args.json:
        print(result.to_json())
    else:
        if result.stdout:
            print(result.stdout, end='')
        if result.stderr:
            print(result.stderr, end='', file=sys.stderr)

    return result.exit_code


if __name__ == '__main__':
    sys.exit(main())
