"""
MCP Project - Model Context Protocol server integration.

This project provides an MCP server that exposes the Cortical Text Processor
as tools for AI assistants like Claude Desktop.

Installation:
    pip install cortical-text-processor[mcp]

Usage:
    from cortical.projects.mcp import CorticalMCPServer

    server = CorticalMCPServer(processor)
    server.run()

Dependencies:
    - mcp>=1.0
    - pydantic
    - httpx
    - uvicorn
    - starlette
"""

try:
    from .server import CorticalMCPServer, main
    __all__ = ['CorticalMCPServer', 'main']
except ImportError as e:
    # MCP dependencies not installed
    def _missing_deps(*args, **kwargs):
        raise ImportError(
            "MCP dependencies not installed. "
            "Install with: pip install cortical-text-processor[mcp]"
        ) from e
    CorticalMCPServer = _missing_deps
    main = _missing_deps
    __all__ = ['CorticalMCPServer', 'main']
