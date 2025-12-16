"""
Protocol Buffers Serialization Facade
=====================================

Provides a clean API for Protocol Buffers text-format serialization.

This module hides protobuf implementation details and provides simple
functions for serializing/deserializing processor state to git-friendly
text format.

Usage:
    from cortical.proto import serialize_state, deserialize_state, PROTOBUF_AVAILABLE

    if PROTOBUF_AVAILABLE:
        # Serialize state to text
        text = serialize_state(layers, documents, metadata, ...)

        # Deserialize text back to state
        layers, docs, meta, emb, rels, info = deserialize_state(text)
"""

from typing import Dict, List, Optional, Any, Tuple

# Check if protobuf is available
# Note: We check for the google.protobuf package, but don't try to compile
# protos at import time. Compilation happens lazily when serialization is used.
PROTOBUF_AVAILABLE = False
_text_format = None
_to_proto = None
_from_proto = None
_get_proto_class = None

try:
    from google.protobuf import text_format as _text_format
    from .serialization import to_proto as _to_proto, from_proto as _from_proto
    from .serialization import _get_proto_class
    PROTOBUF_AVAILABLE = True
except ImportError:
    pass


def serialize_state(
    layers: Dict,
    documents: Dict[str, str],
    document_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    embeddings: Optional[Dict[str, list]] = None,
    semantic_relations: Optional[list] = None,
    metadata: Optional[Dict] = None
) -> str:
    """
    Serialize processor state to protobuf text format.

    Args:
        layers: Dictionary of CorticalLayer -> HierarchicalLayer
        documents: Document collection {doc_id: content}
        document_metadata: Per-document metadata
        embeddings: Graph embeddings for terms
        semantic_relations: Extracted semantic relations
        metadata: Processor metadata (version, settings)

    Returns:
        Human-readable protobuf text format string

    Raises:
        ImportError: If protobuf package is not installed
    """
    if not PROTOBUF_AVAILABLE:
        raise ImportError(
            "protobuf package required for serialization. "
            "Install with: pip install protobuf"
        )

    proto_state = _to_proto(
        layers, documents, document_metadata,
        embeddings, semantic_relations, metadata
    )
    return _text_format.MessageToString(proto_state)


def deserialize_state(text: str) -> Tuple:
    """
    Deserialize protobuf text format to processor state.

    Args:
        text: Protobuf text format string

    Returns:
        Tuple of (layers, documents, document_metadata,
                  embeddings, semantic_relations, metadata)

    Raises:
        ImportError: If protobuf package is not installed
    """
    if not PROTOBUF_AVAILABLE:
        raise ImportError(
            "protobuf package required for deserialization. "
            "Install with: pip install protobuf"
        )

    # Get ProcessorState class lazily to avoid import-time compilation
    ProcessorStateProto = _get_proto_class('ProcessorState')
    proto_state = ProcessorStateProto()
    _text_format.Parse(text, proto_state)
    return _from_proto(proto_state)


# Export public API
__all__ = [
    'PROTOBUF_AVAILABLE',
    'serialize_state',
    'deserialize_state',
]
