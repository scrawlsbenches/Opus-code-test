"""
Protocol Buffers Serialization Module
=====================================

Provides Protocol Buffers serialization for cross-language corpus sharing.

This module enables the Cortical Text Processor to serialize and deserialize
its state using Protocol Buffers TEXT FORMAT, allowing corpus data to be:
- Human-readable and diffable in git
- Shared across different programming languages
- Reviewed in pull requests

Usage:
    from cortical.proto.serialization import to_proto, from_proto
    from google.protobuf import text_format

    # Convert processor state to protobuf
    proto_state = to_proto(layers, documents, document_metadata,
                           embeddings, semantic_relations, metadata)

    # Serialize to text format (git-friendly)
    text_output = text_format.MessageToString(proto_state)

    # Deserialize from text format
    proto_state = ProcessorState()
    text_format.Parse(text_input, proto_state)

    # Convert back to Python objects
    state = from_proto(proto_state)
"""

try:
    from .serialization import to_proto, from_proto
    __all__ = ['to_proto', 'from_proto']
except ImportError:
    # Protobuf not installed - this is OK for core library functionality
    __all__ = []
