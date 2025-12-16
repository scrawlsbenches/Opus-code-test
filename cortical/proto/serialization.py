"""
Protocol Buffers Serialization
===============================

Converts between Python data structures and Protocol Buffer messages.

This module provides serialization and deserialization functions for the
Cortical Text Processor state, enabling cross-language corpus sharing.
"""

import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

try:
    import google.protobuf
    from google.protobuf import descriptor_pool
    from google.protobuf import symbol_database
    from google.protobuf.descriptor import FileDescriptor
    from google.protobuf import message_factory
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False
    # Don't raise here - let the module import succeed
    # Functions will raise ImportError when actually called

from ..layers import CorticalLayer, HierarchicalLayer
from ..minicolumn import Minicolumn, Edge


# Dynamically compile and load proto definitions
def _load_proto_definitions():
    """
    Load Protocol Buffer definitions from schema.proto.

    Uses runtime proto compilation to avoid requiring protoc at build time.

    Returns:
        Tuple of (ProcessorState class, message factory)
    """
    proto_dir = Path(__file__).parent
    proto_file = proto_dir / "schema.proto"

    if not proto_file.exists():
        raise FileNotFoundError(f"schema.proto not found at {proto_file}")

    # Read the proto file content
    with open(proto_file, 'r') as f:
        proto_content = f.read()

    # Try to use compiled protos first (if available)
    try:
        # Import generated pb2 module if it exists
        from . import schema_pb2
        return (
            schema_pb2.ProcessorState,
            schema_pb2.Edge,
            schema_pb2.Minicolumn,
            schema_pb2.HierarchicalLayer,
            schema_pb2.FloatList,
            schema_pb2.SemanticRelation,
            schema_pb2.AnyValue,
            schema_pb2.AnyDict,
            schema_pb2.AnyList
        )
    except ImportError:
        # Compiled protos not available - use runtime compilation
        # This is a fallback that requires the 'protoc' compiler
        import subprocess
        import tempfile

        # Create temporary directory for compilation
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write proto file to temp location
            temp_proto = Path(tmpdir) / "schema.proto"
            temp_proto.write_text(proto_content)

            # Compile proto file
            try:
                subprocess.run(
                    [
                        'protoc',
                        f'--python_out={tmpdir}',
                        f'--proto_path={tmpdir}',
                        str(temp_proto)
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except FileNotFoundError:
                raise RuntimeError(
                    "protoc compiler not found. Please install Protocol Buffers compiler:\n"
                    "  Ubuntu/Debian: sudo apt-get install protobuf-compiler\n"
                    "  macOS: brew install protobuf\n"
                    "  Or compile protos manually: protoc --python_out=. schema.proto"
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to compile schema.proto: {e.stderr}")

            # Import the generated module
            sys.path.insert(0, tmpdir)
            try:
                import schema_pb2
                return (
                    schema_pb2.ProcessorState,
                    schema_pb2.Edge,
                    schema_pb2.Minicolumn,
                    schema_pb2.HierarchicalLayer,
                    schema_pb2.FloatList,
                    schema_pb2.SemanticRelation,
                    schema_pb2.AnyValue,
                    schema_pb2.AnyDict,
                    schema_pb2.AnyList
                )
            finally:
                sys.path.pop(0)


# Proto message classes - lazy loaded on first use to avoid import-time compilation
_proto_classes = None


def _get_proto_classes():
    """
    Get proto message classes, loading them lazily on first use.

    This avoids requiring protoc at import time - only needed when serialization
    is actually used.
    """
    global _proto_classes
    if _proto_classes is None:
        _proto_classes = _load_proto_definitions()
    return _proto_classes


def _get_proto_class(name: str):
    """Get a specific proto class by name."""
    classes = _get_proto_classes()
    index = {
        'ProcessorState': 0,
        'Edge': 1,
        'Minicolumn': 2,
        'HierarchicalLayer': 3,
        'FloatList': 4,
        'SemanticRelation': 5,
        'AnyValue': 6,
        'AnyDict': 7,
        'AnyList': 8
    }
    return classes[index[name]]


def _python_value_to_any_value(value: Any) -> Any:
    """
    Convert a Python value to AnyValue protobuf message.

    Args:
        value: Python value (str, int, float, bool, dict, list)

    Returns:
        AnyValue protobuf message
    """
    AnyValueProto = _get_proto_class('AnyValue')
    AnyDictProto = _get_proto_class('AnyDict')
    AnyListProto = _get_proto_class('AnyList')

    any_val = AnyValueProto()

    if isinstance(value, str):
        any_val.string_value = value
    elif isinstance(value, bool):  # Check bool before int (bool is subclass of int)
        any_val.bool_value = value
    elif isinstance(value, int):
        any_val.int_value = value
    elif isinstance(value, float):
        any_val.double_value = value
    elif isinstance(value, dict):
        any_dict = AnyDictProto()
        for k, v in value.items():
            any_dict.values[k].CopyFrom(_python_value_to_any_value(v))
        any_val.dict_value.CopyFrom(any_dict)
    elif isinstance(value, (list, tuple)):
        any_list = AnyListProto()
        for item in value:
            any_list.values.append(_python_value_to_any_value(item))
        any_val.list_value.CopyFrom(any_list)
    else:
        # Fallback: convert to string
        any_val.string_value = str(value)

    return any_val


def _any_value_to_python(any_val: Any) -> Any:
    """
    Convert AnyValue protobuf message to Python value.

    Args:
        any_val: AnyValue protobuf message

    Returns:
        Python value
    """
    which = any_val.WhichOneof('value')

    if which == 'string_value':
        return any_val.string_value
    elif which == 'int_value':
        return any_val.int_value
    elif which == 'double_value':
        return any_val.double_value
    elif which == 'bool_value':
        return any_val.bool_value
    elif which == 'dict_value':
        return {
            k: _any_value_to_python(v)
            for k, v in any_val.dict_value.values.items()
        }
    elif which == 'list_value':
        return [_any_value_to_python(v) for v in any_val.list_value.values]
    else:
        return None


def edge_to_proto(edge: Edge) -> Any:
    """
    Convert an Edge to protobuf message.

    Args:
        edge: Edge object

    Returns:
        Edge protobuf message
    """
    EdgeProto = _get_proto_class('Edge')
    proto = EdgeProto()
    proto.target_id = edge.target_id
    proto.weight = edge.weight
    proto.relation_type = edge.relation_type
    proto.confidence = edge.confidence
    proto.source = edge.source
    return proto


def edge_from_proto(proto: Any) -> Edge:
    """
    Convert Edge protobuf message to Edge object.

    Args:
        proto: Edge protobuf message

    Returns:
        Edge object
    """
    return Edge(
        target_id=proto.target_id,
        weight=proto.weight,
        relation_type=proto.relation_type,
        confidence=proto.confidence,
        source=proto.source
    )


def minicolumn_to_proto(col: Minicolumn) -> Any:
    """
    Convert a Minicolumn to protobuf message.

    Args:
        col: Minicolumn object

    Returns:
        Minicolumn protobuf message
    """
    MinicolumnProto = _get_proto_class('Minicolumn')
    proto = MinicolumnProto()
    proto.id = col.id
    proto.content = col.content
    proto.layer = col.layer
    proto.activation = col.activation
    proto.occurrence_count = col.occurrence_count
    proto.document_ids.extend(list(col.document_ids))

    # Lateral connections
    for target_id, weight in col.lateral_connections.items():
        proto.lateral_connections[target_id] = weight

    # Typed connections
    for target_id, edge in col.typed_connections.items():
        proto.typed_connections[target_id].CopyFrom(edge_to_proto(edge))

    # Feedforward
    proto.feedforward_sources.extend(list(col.feedforward_sources))
    for target_id, weight in col.feedforward_connections.items():
        proto.feedforward_connections[target_id] = weight

    # Feedback
    for target_id, weight in col.feedback_connections.items():
        proto.feedback_connections[target_id] = weight

    # TF-IDF
    proto.tfidf = col.tfidf
    for doc_id, score in col.tfidf_per_doc.items():
        proto.tfidf_per_doc[doc_id] = score

    # PageRank
    proto.pagerank = col.pagerank

    # Cluster ID (optional)
    if col.cluster_id is not None:
        proto.cluster_id = col.cluster_id

    # Doc occurrence counts
    for doc_id, count in col.doc_occurrence_counts.items():
        proto.doc_occurrence_counts[doc_id] = count

    # Name tokens (optional)
    if col.name_tokens is not None:
        proto.name_tokens.extend(list(col.name_tokens))

    return proto


def minicolumn_from_proto(proto: Any) -> Minicolumn:
    """
    Convert Minicolumn protobuf message to Minicolumn object.

    Args:
        proto: Minicolumn protobuf message

    Returns:
        Minicolumn object
    """
    col = Minicolumn(proto.id, proto.content, proto.layer)
    col.activation = proto.activation
    col.occurrence_count = proto.occurrence_count
    col.document_ids = set(proto.document_ids)

    # Typed connections (primary)
    for target_id, edge_proto in proto.typed_connections.items():
        col.typed_connections[target_id] = edge_from_proto(edge_proto)

    # If no typed connections but lateral_connections exists (old format), convert
    if not col.typed_connections and proto.lateral_connections:
        for target_id, weight in proto.lateral_connections.items():
            col.typed_connections[target_id] = Edge(
                target_id=target_id,
                weight=weight,
                relation_type='co_occurrence',
                confidence=1.0,
                source='corpus'
            )

    # Invalidate cache to rebuild from typed_connections
    col._lateral_cache_valid = False

    # Feedforward
    col.feedforward_sources = set(proto.feedforward_sources)
    col.feedforward_connections = dict(proto.feedforward_connections)

    # Feedback
    col.feedback_connections = dict(proto.feedback_connections)

    # TF-IDF
    col.tfidf = proto.tfidf
    col.tfidf_per_doc = dict(proto.tfidf_per_doc)

    # PageRank
    col.pagerank = proto.pagerank

    # Cluster ID (optional)
    if proto.HasField('cluster_id'):
        col.cluster_id = proto.cluster_id

    # Doc occurrence counts
    col.doc_occurrence_counts = dict(proto.doc_occurrence_counts)

    # Name tokens (optional)
    if proto.name_tokens:
        col.name_tokens = set(proto.name_tokens)

    return col


def layer_to_proto(layer: HierarchicalLayer) -> Any:
    """
    Convert a HierarchicalLayer to protobuf message.

    Args:
        layer: HierarchicalLayer object

    Returns:
        HierarchicalLayer protobuf message
    """
    HierarchicalLayerProto = _get_proto_class('HierarchicalLayer')
    proto = HierarchicalLayerProto()
    proto.level = layer.level

    for content, col in layer.minicolumns.items():
        proto.minicolumns[content].CopyFrom(minicolumn_to_proto(col))

    return proto


def layer_from_proto(proto: Any) -> HierarchicalLayer:
    """
    Convert HierarchicalLayer protobuf message to HierarchicalLayer object.

    Args:
        proto: HierarchicalLayer protobuf message

    Returns:
        HierarchicalLayer object
    """
    layer = HierarchicalLayer(CorticalLayer(proto.level))

    for content, col_proto in proto.minicolumns.items():
        col = minicolumn_from_proto(col_proto)
        layer.minicolumns[content] = col
        layer._id_index[col.id] = content  # Rebuild ID index

    return layer


def to_proto(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str],
    document_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    embeddings: Optional[Dict[str, list]] = None,
    semantic_relations: Optional[list] = None,
    metadata: Optional[Dict] = None
) -> Any:
    """
    Convert processor state to Protocol Buffer message.

    Args:
        layers: Dictionary of all layers
        documents: Document collection
        document_metadata: Per-document metadata
        embeddings: Graph embeddings for terms
        semantic_relations: Extracted semantic relations
        metadata: Processor metadata

    Returns:
        ProcessorState protobuf message
    """
    ProcessorStateProto = _get_proto_class('ProcessorState')
    AnyDictProto = _get_proto_class('AnyDict')
    FloatListProto = _get_proto_class('FloatList')
    SemanticRelationProto = _get_proto_class('SemanticRelation')

    proto = ProcessorStateProto()
    proto.version = '2.2'

    # Serialize layers
    for layer_enum, layer in layers.items():
        proto.layers[layer_enum.value].CopyFrom(layer_to_proto(layer))

    # Documents
    for doc_id, text in documents.items():
        proto.documents[doc_id] = text

    # Document metadata
    if document_metadata:
        for doc_id, meta in document_metadata.items():
            any_dict = AnyDictProto()
            for k, v in meta.items():
                any_dict.values[k].CopyFrom(_python_value_to_any_value(v))
            proto.document_metadata[doc_id].CopyFrom(any_dict)

    # Embeddings
    if embeddings:
        for term, embedding in embeddings.items():
            float_list = FloatListProto()
            float_list.values.extend(embedding)
            proto.embeddings[term].CopyFrom(float_list)

    # Semantic relations
    if semantic_relations:
        for rel in semantic_relations:
            if len(rel) >= 4:
                rel_proto = SemanticRelationProto()
                rel_proto.term1 = rel[0]
                rel_proto.relation_type = rel[1]
                rel_proto.term2 = rel[2]
                rel_proto.weight = float(rel[3])
                proto.semantic_relations.append(rel_proto)

    # Metadata
    if metadata:
        any_dict = AnyDictProto()
        for k, v in metadata.items():
            any_dict.values[k].CopyFrom(_python_value_to_any_value(v))
        proto.metadata.CopyFrom(any_dict)

    return proto


def from_proto(proto: Any) -> Tuple:
    """
    Convert Protocol Buffer message to processor state.

    Args:
        proto: ProcessorState protobuf message

    Returns:
        Tuple of (layers, documents, document_metadata, embeddings, semantic_relations, metadata)
    """
    # Reconstruct layers
    layers = {}
    for level_value, layer_proto in proto.layers.items():
        layer = layer_from_proto(layer_proto)
        layers[CorticalLayer(level_value)] = layer

    # Documents
    documents = dict(proto.documents)

    # Document metadata
    document_metadata = {}
    for doc_id, any_dict_proto in proto.document_metadata.items():
        document_metadata[doc_id] = {
            k: _any_value_to_python(v)
            for k, v in any_dict_proto.values.items()
        }

    # Embeddings
    embeddings = {}
    for term, float_list_proto in proto.embeddings.items():
        embeddings[term] = list(float_list_proto.values)

    # Semantic relations
    semantic_relations = []
    for rel_proto in proto.semantic_relations:
        semantic_relations.append((
            rel_proto.term1,
            rel_proto.relation_type,
            rel_proto.term2,
            rel_proto.weight
        ))

    # Metadata
    metadata = {}
    if proto.HasField('metadata'):
        metadata = {
            k: _any_value_to_python(v)
            for k, v in proto.metadata.values.items()
        }

    return layers, documents, document_metadata, embeddings, semantic_relations, metadata
