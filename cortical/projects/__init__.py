"""
Cortical Projects - Optional extensions to the core library.

Projects are isolated, opt-in features that:
- Have their own dependencies (not required by core)
- Have their own test suites (can fail independently)
- Follow the pattern: cortical.projects.<name>

Available Projects:
- mcp: Model Context Protocol server integration
- proto: Protobuf serialization support
- cli: Command-line interface tools

Usage:
    # Install specific project dependencies
    pip install cortical-text-processor[mcp]
    pip install cortical-text-processor[proto]

    # Import from projects
    from cortical.projects.mcp import CorticalMCPServer
    from cortical.projects.proto import to_proto, from_proto
"""

__all__ = ['mcp', 'proto', 'cli']
