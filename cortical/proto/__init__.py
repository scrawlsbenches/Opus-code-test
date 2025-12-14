"""
Protocol Buffers Serialization Module
=====================================

Provides Protocol Buffers serialization for cross-language corpus sharing.

This module enables the Cortical Text Processor to serialize and deserialize
its state using Protocol Buffers, allowing corpus data to be shared across
different programming languages and platforms.

Usage:
    from cortical.proto.serialization import to_proto, from_proto

    # Convert processor state to protobuf
    proto_state = to_proto(layers, documents, document_metadata,
                           embeddings, semantic_relations, metadata)

    # Serialize to bytes
    serialized = proto_state.SerializeToString()

    # Deserialize from bytes
    proto_state = ProcessorState()
    proto_state.ParseFromString(serialized)

    # Convert back to Python objects
    state = from_proto(proto_state)
"""

try:
    from .serialization import to_proto, from_proto
    __all__ = ['to_proto', 'from_proto']
except ImportError:
    # Protobuf not installed - this is OK for core library functionality
    __all__ = []
