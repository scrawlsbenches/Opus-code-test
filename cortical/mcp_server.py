"""
Backward compatibility shim for MCP server.

The MCP server has been moved to cortical.projects.mcp.
This module re-exports for backward compatibility.

New code should use:
    from cortical.projects.mcp import CorticalMCPServer, main

Deprecated usage (still works but shows warning when used):
    from cortical.mcp_server import CorticalMCPServer, main
"""

import warnings

def _warn_deprecated():
    """Issue deprecation warning (called on first use, not import)."""
    warnings.warn(
        "cortical.mcp_server is deprecated. "
        "Use cortical.projects.mcp instead. "
        "See docs/projects-architecture.md for details.",
        DeprecationWarning,
        stacklevel=3
    )

# Re-export from new location
try:
    from cortical.projects.mcp import CorticalMCPServer as _CorticalMCPServer
    from cortical.projects.mcp import main as _main

    # Check if we got the actual class or the error function
    # (projects gracefully degrade to functions when deps are missing)
    if isinstance(_CorticalMCPServer, type):
        # We have the real class - create wrapper with deprecation warning
        class CorticalMCPServer(_CorticalMCPServer):
            """Deprecated: Use cortical.projects.mcp.CorticalMCPServer instead."""
            def __init__(self, *args, **kwargs):
                _warn_deprecated()
                super().__init__(*args, **kwargs)

        def main(*args, **kwargs):
            """Deprecated: Use cortical.projects.mcp.main instead."""
            _warn_deprecated()
            return _main(*args, **kwargs)

        def create_mcp_server(*args, **kwargs):
            """Deprecated: Use cortical.projects.mcp.CorticalMCPServer instead."""
            _warn_deprecated()
            return _CorticalMCPServer(*args, **kwargs)
    else:
        # We got the error function - just re-export it
        CorticalMCPServer = _CorticalMCPServer
        main = _main
        create_mcp_server = _CorticalMCPServer

    __all__ = ['CorticalMCPServer', 'main', 'create_mcp_server']
except ImportError:
    # MCP project itself couldn't be imported
    def _missing(*args, **kwargs):
        raise ImportError(
            "MCP dependencies not installed. "
            "Install with: pip install cortical-text-processor[mcp]"
        )
    CorticalMCPServer = _missing
    main = _missing
    create_mcp_server = _missing
    __all__ = ['CorticalMCPServer', 'main', 'create_mcp_server']

if __name__ == "__main__":
    _warn_deprecated()
    from cortical.projects.mcp import main as _main
    _main()
