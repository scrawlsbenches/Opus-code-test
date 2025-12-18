"""
CLI Project - Command-line interface tools.

This project provides CLI utilities for task management and workflow automation.

Installation:
    pip install cortical-text-processor[cli]

Usage:
    from cortical.projects.cli import CLIWrapper, Session

    wrapper = CLIWrapper()
    result = wrapper.run("pytest tests/")

Dependencies:
    - click (optional, for enhanced CLI)

Note:
    The CLI wrapper is currently in the core library (cortical/cli_wrapper.py).
    It may be moved here in a future sprint if it becomes problematic.
"""

# Currently CLI is in core - this is a placeholder for future migration
__all__ = []
