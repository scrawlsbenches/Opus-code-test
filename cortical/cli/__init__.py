"""
Cortical CLI - Unified Command Line Interface.

This package provides a unified CLI framework for all cortical tools.
Commands are organized by domain (audit, got, cel, etc.) and auto-discovered.

Architecture:
    cortical/cli/
    ├── __init__.py      # This file - registry and utilities
    ├── _base.py         # Base classes and shared utilities
    ├── audit/           # Audit commands (scan, train, generate, etc.)
    └── ...              # Future: got/, cel/, spark/ commands

Usage:
    # From Python
    from cortical.cli import run_command
    run_command('audit', 'scan', ['cortical/'])

    # From shell (via scripts/audit_tool.py wrapper)
    python scripts/audit_tool.py scan cortical/

Adding New Command Domains:
    1. Create a new package under cortical/cli/ (e.g., cortical/cli/spark/)
    2. Implement __init__.py with setup_parser() and handle_command()
    3. Register in COMMAND_DOMAINS below

Design Principles:
    - Commands are thin wrappers around business logic in cortical/
    - Business logic lives in domain packages (cortical/audits/, cortical/got/, etc.)
    - Container is used for dependency injection where appropriate
    - Each domain handles its own argument parsing
"""

from typing import Dict, Any, Callable, Optional, List
import importlib

# Registry of command domains
# Each domain module must export: setup_parser(subparsers), handle_command(args)
COMMAND_DOMAINS: Dict[str, str] = {
    'audit': 'cortical.cli.audit',
    # 'got': 'cortical.cli.got',
    # 'cel': 'cortical.cli.cel',
}


def get_domain_module(domain: str):
    """
    Get the module for a command domain.

    Args:
        domain: Domain name (e.g., 'audit')

    Returns:
        The imported module, or None if not found
    """
    if domain not in COMMAND_DOMAINS:
        return None

    module_path = COMMAND_DOMAINS[domain]
    try:
        return importlib.import_module(module_path)
    except ImportError as e:
        print(f"Warning: Could not import {module_path}: {e}")
        return None


def setup_all_parsers(subparsers) -> None:
    """
    Set up argument parsers for all registered command domains.

    Args:
        subparsers: argparse subparsers object
    """
    for domain, module_path in COMMAND_DOMAINS.items():
        module = get_domain_module(domain)
        if module and hasattr(module, 'setup_parser'):
            module.setup_parser(subparsers)


def run_command(domain: str, command: str, args: List[str]) -> int:
    """
    Run a command programmatically.

    Args:
        domain: Domain name (e.g., 'audit')
        command: Command name (e.g., 'scan')
        args: Command arguments

    Returns:
        Exit code (0 for success)
    """
    module = get_domain_module(domain)
    if not module:
        print(f"Unknown domain: {domain}")
        return 1

    if hasattr(module, 'run_command'):
        return module.run_command(command, args)

    print(f"Domain {domain} does not support programmatic execution")
    return 1


__all__ = [
    'COMMAND_DOMAINS',
    'get_domain_module',
    'setup_all_parsers',
    'run_command',
]
