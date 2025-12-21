"""
Tests for protobuf serialization.

These tests verify the proto project's API and error messages.
Full implementation tests will be added when protobuf serialization is implemented.
"""

import pytest
from cortical.processor import CorticalTextProcessor


@pytest.mark.optional
class TestProtobufSerialization:
    """Tests for protobuf serialization functionality."""

    def test_import_succeeds_with_protobuf_installed(self):
        """Verify the proto module can be imported when protobuf is installed."""
        try:
            from cortical.projects.proto import to_proto, from_proto
            # Verify functions exist
            assert callable(to_proto)
            assert callable(from_proto)
        except ImportError as e:
            # If protobuf is not installed, expect a clear error message
            assert "pip install cortical-text-processor[proto]" in str(e)

    def test_to_proto_not_implemented(self):
        """Verify to_proto raises NotImplementedError with helpful message."""
        pytest.importorskip("google.protobuf", reason="protobuf not installed")

        from cortical.projects.proto import to_proto

        processor = CorticalTextProcessor()

        with pytest.raises(NotImplementedError) as exc_info:
            to_proto(processor)

        # Verify helpful error message
        error_msg = str(exc_info.value)
        assert "not yet implemented" in error_msg
        assert ".proto schema" in error_msg or "JSON serialization" in error_msg

    def test_from_proto_not_implemented(self):
        """Verify from_proto raises NotImplementedError with helpful message."""
        pytest.importorskip("google.protobuf", reason="protobuf not installed")

        from cortical.projects.proto import from_proto

        proto_bytes = b"dummy protobuf data"

        with pytest.raises(NotImplementedError) as exc_info:
            from_proto(proto_bytes)

        # Verify helpful error message
        error_msg = str(exc_info.value)
        assert "not yet implemented" in error_msg
        assert "ParseFromString" in error_msg or "JSON serialization" in error_msg

    def test_missing_protobuf_dependency_error(self):
        """Verify clear error when protobuf dependencies are missing."""
        # Note: Our current implementation allows import but raises NotImplementedError
        # when functions are called. This is different from the graceful degradation
        # pattern in MCP which uses ImportError.

        # If protobuf is installed, skip this test
        try:
            import google.protobuf
            pytest.skip("protobuf is installed, cannot test stub behavior")
        except ImportError:
            # protobuf not installed - verify stub functions work
            from cortical.projects.proto import to_proto, from_proto

            # Functions should be callable but raise NotImplementedError
            processor = CorticalTextProcessor()

            with pytest.raises(NotImplementedError) as exc_info:
                to_proto(processor)

            error_msg = str(exc_info.value)
            assert "not yet implemented" in error_msg.lower()
