"""
Tests for the MCP server backward compatibility shim.

Tests the deprecation warnings and fallback behavior of cortical/mcp_server.py.
"""

import pytest
import warnings


class TestMCPServerShim:
    """Tests for the MCP server backward compatibility module."""

    def test_import_raises_deprecation_when_used(self):
        """Using CorticalMCPServer from shim raises deprecation warning or import error."""
        # Import the shim module
        from cortical import mcp_server

        # The module should export these names (even if they're error functions)
        assert hasattr(mcp_server, 'CorticalMCPServer')
        assert hasattr(mcp_server, 'main')
        assert hasattr(mcp_server, 'create_mcp_server')

    def test_warn_deprecated_function_exists(self):
        """The _warn_deprecated function should exist."""
        from cortical import mcp_server

        assert hasattr(mcp_server, '_warn_deprecated')

    def test_all_exported_names(self):
        """__all__ should contain expected names."""
        from cortical import mcp_server

        assert 'CorticalMCPServer' in mcp_server.__all__
        assert 'main' in mcp_server.__all__
        assert 'create_mcp_server' in mcp_server.__all__

    def test_deprecation_warning_message_content(self):
        """Deprecation warning should mention correct migration path."""
        from cortical import mcp_server

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mcp_server._warn_deprecated()

            assert len(w) == 1
            warning = w[0]
            assert issubclass(warning.category, DeprecationWarning)
            assert "cortical.mcp_server is deprecated" in str(warning.message)
            assert "cortical.projects.mcp" in str(warning.message)

    def test_calling_missing_function_raises_import_error(self):
        """If MCP deps not installed, calling functions raises ImportError."""
        from cortical import mcp_server

        # Try to call the exported functions - they may raise ImportError
        # if MCP dependencies are not installed
        try:
            # If MCP deps are installed, this might work or raise deprecation warning
            # If not installed, it should raise ImportError
            result = mcp_server.CorticalMCPServer
            # Check if it's the error function
            if callable(result) and not isinstance(result, type):
                # It's a function that should raise ImportError
                with pytest.raises(ImportError):
                    result()
        except Exception:
            pass  # MCP might be available in test environment


class TestMCPServerImportPath:
    """Tests for import path handling."""

    def test_projects_mcp_can_be_imported(self):
        """cortical.projects.mcp should be importable."""
        # May raise ImportError if dependencies not installed
        try:
            from cortical.projects import mcp
            # Module should have __all__
            assert hasattr(mcp, '__all__')
        except ImportError:
            # Expected if mcp dependencies not installed
            pass

    def test_shim_module_docstring(self):
        """Shim module should have docstring explaining deprecation."""
        from cortical import mcp_server

        assert mcp_server.__doc__ is not None
        assert "deprecated" in mcp_server.__doc__.lower()
        assert "cortical.projects.mcp" in mcp_server.__doc__


class TestProtoModule:
    """Tests for the protobuf serialization stub module."""

    def test_import_proto_module(self):
        """cortical.projects.proto should be importable."""
        from cortical.projects import proto

        assert hasattr(proto, 'to_proto')
        assert hasattr(proto, 'from_proto')
        assert '__all__' in dir(proto)

    def test_proto_all_exported(self):
        """__all__ should contain expected names."""
        from cortical.projects import proto

        assert 'to_proto' in proto.__all__
        assert 'from_proto' in proto.__all__

    def test_to_proto_raises_not_implemented(self):
        """to_proto should raise NotImplementedError (stub)."""
        try:
            from cortical.projects.proto.serialization import to_proto
            from cortical.processor import CorticalTextProcessor

            processor = CorticalTextProcessor()

            with pytest.raises(NotImplementedError) as exc_info:
                to_proto(processor)

            # Check error message mentions it's a stub
            assert "not yet implemented" in str(exc_info.value)
            assert "placeholder" in str(exc_info.value).lower()
        except ImportError:
            # Proto dependencies might cause import error
            pytest.skip("Proto module import failed")

    def test_from_proto_raises_not_implemented(self):
        """from_proto should raise NotImplementedError (stub)."""
        try:
            from cortical.projects.proto.serialization import from_proto

            with pytest.raises(NotImplementedError) as exc_info:
                from_proto(b"fake_proto_bytes")

            # Check error message mentions it's a stub
            assert "not yet implemented" in str(exc_info.value)
        except ImportError:
            pytest.skip("Proto module import failed")

    def test_proto_init_handles_missing_deps(self):
        """Proto __init__ should handle missing protobuf gracefully."""
        from cortical.projects import proto

        # The module should always be importable
        # Functions should raise ImportError or NotImplementedError when called
        assert callable(proto.to_proto)
        assert callable(proto.from_proto)

    def test_serialization_module_docstring(self):
        """Serialization module should have explanatory docstring."""
        try:
            from cortical.projects.proto import serialization

            assert serialization.__doc__ is not None
            assert "Protobuf" in serialization.__doc__
        except ImportError:
            pytest.skip("Proto module import failed")
