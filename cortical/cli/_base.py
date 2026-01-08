"""
Base utilities for CLI commands.

This module provides shared functionality used across all CLI domains.
Domain-specific utilities should live in their respective packages.
"""

import sys
from typing import Optional, Any
from pathlib import Path


def print_header(title: str, width: int = 70) -> None:
    """Print a formatted header."""
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_section(title: str, width: int = 70) -> None:
    """Print a section divider."""
    print(f"\n{title}")
    print("-" * len(title))


def print_footer(width: int = 70) -> None:
    """Print a footer line."""
    print("=" * width)


def confirm_action(prompt: str, default: bool = False) -> bool:
    """
    Ask user for confirmation.

    Args:
        prompt: Question to ask
        default: Default answer if user just presses Enter

    Returns:
        True if user confirmed, False otherwise
    """
    suffix = " [Y/n]: " if default else " [y/N]: "
    try:
        response = input(prompt + suffix).strip().lower()
        if not response:
            return default
        return response in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path_str: str, must_exist: bool = False) -> Optional[Path]:
    """
    Resolve a path string to an absolute Path.

    Args:
        path_str: Path string (can be relative)
        must_exist: If True, return None if path doesn't exist

    Returns:
        Resolved Path, or None if must_exist and doesn't exist
    """
    path = Path(path_str).resolve()
    if must_exist and not path.exists():
        return None
    return path


class CLIError(Exception):
    """
    Exception for CLI-level errors.

    These are user-facing errors that should be displayed cleanly
    without a full stack trace.
    """

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def handle_cli_error(func):
    """
    Decorator to handle CLIError exceptions cleanly.

    Usage:
        @handle_cli_error
        def cmd_something(args):
            if error_condition:
                raise CLIError("Something went wrong")
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CLIError as e:
            print(f"Error: {e}", file=sys.stderr)
            return e.exit_code
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            return 130
    return wrapper
