"""
Proto Project - Protobuf serialization support.

This project provides protobuf serialization for cross-language interoperability.

Installation:
    pip install cortical-text-processor[proto]

Usage:
    from cortical.projects.proto import to_proto, from_proto

    # Serialize processor state
    proto_bytes = to_proto(processor)

    # Deserialize
    processor = from_proto(proto_bytes)

Dependencies:
    - protobuf>=4.0
"""

try:
    from .serialization import to_proto, from_proto
    __all__ = ['to_proto', 'from_proto']
except ImportError as e:
    # Protobuf dependencies not installed
    def _missing_deps(*args, **kwargs):
        raise ImportError(
            "Protobuf dependencies not installed. "
            "Install with: pip install cortical-text-processor[proto]"
        ) from e
    to_proto = _missing_deps
    from_proto = _missing_deps
    __all__ = ['to_proto', 'from_proto']
