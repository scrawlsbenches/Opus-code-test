"""
Cortical Projects - Optional extensions to the core library.

Projects are isolated, opt-in features that:
- Have their own dependencies (not required by core)
- Have their own test suites (can fail independently)
- Follow the pattern: cortical.projects.<name>

Available Projects:
- cli: Command-line interface tools
"""

__all__ = ['cli']
