"""
Persistence Module
==================

Save and load functionality for the cortical processor.

Supports:
- Pickle serialization for full state
- JSON export for graph visualization
- Incremental updates
"""

import pickle
import json
import os
import logging
from typing import Dict, Optional, Any

from .layers import CorticalLayer, HierarchicalLayer
from .minicolumn import Minicolumn

logger = logging.getLogger(__name__)


def save_processor(
    filepath: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str],
    document_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    embeddings: Optional[Dict[str, list]] = None,
    semantic_relations: Optional[list] = None,
    metadata: Optional[Dict] = None,
    verbose: bool = True,
    format: str = 'pickle'
) -> None:
    """
    Save processor state to a file.

    Args:
        filepath: Path to save file
        layers: Dictionary of all layers
        documents: Document collection
        document_metadata: Per-document metadata (source, timestamp, etc.)
        embeddings: Graph embeddings for terms (optional)
        semantic_relations: Extracted semantic relations (optional)
        metadata: Optional processor metadata (version, settings, etc.)
        verbose: Print progress
        format: Serialization format ('pickle' or 'protobuf'). Default: 'pickle'

    Raises:
        ValueError: If format is not 'pickle' or 'protobuf'
        ImportError: If format='protobuf' but protobuf package is not installed
    """
    if format not in ['pickle', 'protobuf']:
        raise ValueError(f"Invalid format '{format}'. Must be 'pickle' or 'protobuf'.")

    if format == 'pickle':
        # Original pickle serialization
        state = {
            'version': '2.2',
            'layers': {},
            'documents': documents,
            'document_metadata': document_metadata or {},
            'embeddings': embeddings or {},
            'semantic_relations': semantic_relations or [],
            'metadata': metadata or {}
        }

        # Serialize layers
        for layer_enum, layer in layers.items():
            state['layers'][layer_enum.value] = layer.to_dict()

        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    elif format == 'protobuf':
        # Protocol Buffers serialization (text format for git-friendliness)
        try:
            from .proto.serialization import to_proto
            from google.protobuf import text_format
        except ImportError as e:
            raise ImportError(
                "protobuf package is required for Protocol Buffers serialization. "
                "Install it with: pip install protobuf"
            ) from e

        proto_state = to_proto(
            layers, documents, document_metadata,
            embeddings, semantic_relations, metadata
        )

        # Use text format for human-readable, git-friendly output
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text_format.MessageToString(proto_state))

    if verbose:
        total_cols = sum(len(layer.minicolumns) for layer in layers.values())
        total_conns = sum(layer.total_connections() for layer in layers.values())
        logger.info(f"✓ Saved processor to {filepath} (format: {format})")
        logger.info(f"  - {len(documents)} documents")
        logger.info(f"  - {total_cols} minicolumns")
        logger.info(f"  - {total_conns} connections")
        if embeddings:
            logger.info(f"  - {len(embeddings)} embeddings")
        if semantic_relations:
            logger.info(f"  - {len(semantic_relations)} semantic relations")


def load_processor(
    filepath: str,
    verbose: bool = True,
    format: Optional[str] = None
) -> tuple:
    """
    Load processor state from a file.

    Args:
        filepath: Path to saved file
        verbose: Print progress
        format: Serialization format ('pickle' or 'protobuf'). If None, auto-detect.

    Returns:
        Tuple of (layers, documents, document_metadata, embeddings, semantic_relations, metadata)

    Raises:
        ValueError: If layer values are invalid (must be 0-3) or format is invalid
        ImportError: If format='protobuf' but protobuf package is not installed
    """
    # Auto-detect format if not specified
    if format is None:
        # Try to detect based on file content
        with open(filepath, 'rb') as f:
            # Read first few bytes
            header = f.read(64)

            # Pickle files start with 0x80 (protocol marker)
            if header[0:1] == b'\x80':
                format = 'pickle'
            else:
                # Protobuf text format starts with readable text like "version:" or "layers {"
                # Check if it looks like text (ASCII/UTF-8)
                try:
                    header_text = header.decode('utf-8')
                    if 'version' in header_text or 'layers' in header_text:
                        format = 'protobuf'
                    else:
                        # Default to pickle for unknown formats
                        format = 'pickle'
                except UnicodeDecodeError:
                    # Binary content - assume pickle
                    format = 'pickle'

    if format not in ['pickle', 'protobuf']:
        raise ValueError(f"Invalid format '{format}'. Must be 'pickle' or 'protobuf'.")

    if format == 'pickle':
        # Original pickle deserialization
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Reconstruct layers
        layers = {}
        for level_value, layer_data in state.get('layers', {}).items():
            # Validate layer value before creating enum
            level_int = int(level_value)
            if level_int not in [0, 1, 2, 3]:
                raise ValueError(
                    f"Invalid layer value {level_int} in saved state. "
                    f"Layer values must be 0-3 (TOKENS=0, BIGRAMS=1, CONCEPTS=2, DOCUMENTS=3)."
                )
            layer = HierarchicalLayer.from_dict(layer_data)
            layers[CorticalLayer(level_int)] = layer

        documents = state.get('documents', {})
        document_metadata = state.get('document_metadata', {})
        embeddings = state.get('embeddings', {})
        semantic_relations = state.get('semantic_relations', [])
        metadata = state.get('metadata', {})

    elif format == 'protobuf':
        # Protocol Buffers deserialization (text format)
        try:
            from .proto.serialization import from_proto
            from .proto import schema_pb2
            from google.protobuf import text_format
        except ImportError as e:
            raise ImportError(
                "protobuf package is required for Protocol Buffers deserialization. "
                "Install it with: pip install protobuf"
            ) from e

        with open(filepath, 'r', encoding='utf-8') as f:
            proto_text = f.read()

        proto_state = schema_pb2.ProcessorState()
        text_format.Parse(proto_text, proto_state)

        layers, documents, document_metadata, embeddings, semantic_relations, metadata = from_proto(proto_state)

    if verbose:
        total_cols = sum(len(layer.minicolumns) for layer in layers.values())
        total_conns = sum(layer.total_connections() for layer in layers.values())
        logger.info(f"✓ Loaded processor from {filepath} (format: {format})")
        logger.info(f"  - {len(documents)} documents")
        logger.info(f"  - {total_cols} minicolumns")
        logger.info(f"  - {total_conns} connections")
        if embeddings:
            logger.info(f"  - {len(embeddings)} embeddings")
        if semantic_relations:
            logger.info(f"  - {len(semantic_relations)} semantic relations")

    return layers, documents, document_metadata, embeddings, semantic_relations, metadata


def export_graph_json(
    filepath: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    layer_filter: Optional[CorticalLayer] = None,
    min_weight: float = 0.0,
    max_nodes: int = 500,
    verbose: bool = True
) -> Dict:
    """
    Export graph structure as JSON for visualization.

    Creates a format compatible with D3.js, vis.js, etc.

    Args:
        filepath: Output file path
        layers: Dictionary of layers
        layer_filter: Only export specific layer (None = all)
        min_weight: Minimum edge weight to include
        max_nodes: Maximum nodes to export
        verbose: Print progress messages

    Returns:
        The exported graph data
    """
    nodes = []
    edges = []
    node_ids = set()
    
    # Determine which layers to export
    if layer_filter is not None:
        layer_list = [layers.get(layer_filter)]
    else:
        layer_list = list(layers.values())
    
    # Collect nodes (sorted by PageRank)
    all_columns = []
    for layer in layer_list:
        if layer:
            all_columns.extend(layer.minicolumns.values())
    
    all_columns.sort(key=lambda c: c.pagerank, reverse=True)
    
    # Take top nodes
    for col in all_columns[:max_nodes]:
        nodes.append({
            'id': col.id,
            'label': col.content,
            'layer': col.layer,
            'pagerank': col.pagerank,
            'tfidf': col.tfidf,
            'activation': col.activation,
            'documents': len(col.document_ids)
        })
        node_ids.add(col.id)
    
    # Collect edges
    for col in all_columns[:max_nodes]:
        for target_id, weight in col.lateral_connections.items():
            if weight >= min_weight and target_id in node_ids:
                edges.append({
                    'source': col.id,
                    'target': target_id,
                    'weight': weight
                })
    
    graph = {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'node_count': len(nodes),
            'edge_count': len(edges),
            'layers': [l.value for l in layers.keys() if l is not None]
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(graph, f, indent=2)

    if verbose:
        logger.info(f"Graph exported to {filepath}")
        logger.info(f"  - {len(nodes)} nodes, {len(edges)} edges")

    return graph


def export_embeddings_json(
    filepath: str,
    embeddings: Dict[str, list],
    metadata: Optional[Dict] = None
) -> None:
    """
    Export embeddings as JSON.
    
    Args:
        filepath: Output file path
        embeddings: Dictionary of term -> embedding vector
        metadata: Optional metadata
    """
    data = {
        'embeddings': embeddings,
        'dimensions': len(next(iter(embeddings.values()))) if embeddings else 0,
        'terms': len(embeddings),
        'metadata': metadata or {}
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f)

    logger.info(f"Embeddings exported to {filepath}")
    logger.info(f"  - {len(embeddings)} terms, {data['dimensions']} dimensions")


def load_embeddings_json(filepath: str) -> Dict[str, list]:
    """
    Load embeddings from JSON.
    
    Args:
        filepath: Input file path
        
    Returns:
        Dictionary of term -> embedding vector
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data.get('embeddings', {})


def export_semantic_relations_json(
    filepath: str,
    relations: list
) -> None:
    """
    Export semantic relations as JSON.
    
    Args:
        filepath: Output file path
        relations: List of relation dictionaries
    """
    with open(filepath, 'w') as f:
        json.dump({
            'relations': relations,
            'count': len(relations)
        }, f, indent=2)

    logger.info(f"Relations exported to {filepath}")
    logger.info(f"  - {len(relations)} relations")


def load_semantic_relations_json(filepath: str) -> list:
    """
    Load semantic relations from JSON.
    
    Args:
        filepath: Input file path
        
    Returns:
        List of relation dictionaries
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data.get('relations', [])


def get_state_summary(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str]
) -> Dict:
    """
    Get a summary of the current processor state.

    Args:
        layers: Dictionary of layers
        documents: Document collection

    Returns:
        Summary statistics
    """
    summary = {
        'documents': len(documents),
        'layers': {}
    }

    for layer_enum, layer in layers.items():
        summary['layers'][layer_enum.name] = {
            'columns': len(layer.minicolumns),
            'connections': layer.total_connections(),
            'avg_activation': layer.average_activation(),
            'sparsity': layer.sparsity()
        }

    summary['total_columns'] = sum(
        len(layer.minicolumns) for layer in layers.values()
    )
    summary['total_connections'] = sum(
        layer.total_connections() for layer in layers.values()
    )

    return summary


# Layer colors for visualization
LAYER_COLORS = {
    CorticalLayer.TOKENS: '#4169E1',     # Royal Blue
    CorticalLayer.BIGRAMS: '#228B22',    # Forest Green
    CorticalLayer.CONCEPTS: '#FF8C00',   # Dark Orange
    CorticalLayer.DOCUMENTS: '#DC143C',  # Crimson
}

# Layer display names
LAYER_NAMES = {
    CorticalLayer.TOKENS: 'Tokens',
    CorticalLayer.BIGRAMS: 'Bigrams',
    CorticalLayer.CONCEPTS: 'Concepts',
    CorticalLayer.DOCUMENTS: 'Documents',
}


def export_conceptnet_json(
    filepath: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    semantic_relations: Optional[list] = None,
    include_cross_layer: bool = True,
    include_typed_edges: bool = True,
    min_weight: float = 0.0,
    min_confidence: float = 0.0,
    max_nodes_per_layer: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Export ConceptNet-style graph for visualization.

    Creates a rich graph format with:
    - Color-coded nodes by layer
    - Typed edges with relation types and confidence
    - Cross-layer connections (feedforward/feedback)
    - D3.js/Cytoscape-compatible output

    Args:
        filepath: Output file path (JSON)
        layers: Dictionary of layers
        semantic_relations: Optional list of (t1, rel, t2, weight) tuples
        include_cross_layer: Include feedforward/feedback edges
        include_typed_edges: Include typed_connections with relation types
        min_weight: Minimum edge weight to include
        min_confidence: Minimum confidence for typed edges
        max_nodes_per_layer: Maximum nodes per layer (by PageRank)
        verbose: Print progress messages

    Returns:
        The exported graph data

    Example:
        >>> export_conceptnet_json(
        ...     "graph.json", processor.layers,
        ...     semantic_relations=processor.semantic_relations
        ... )
    """
    nodes = []
    edges = []
    node_ids = set()
    edge_set = set()  # Track unique edges

    # Collect nodes from each layer
    for layer_enum, layer in layers.items():
        if layer is None or layer.column_count() == 0:
            continue

        color = LAYER_COLORS.get(layer_enum, '#808080')
        layer_name = LAYER_NAMES.get(layer_enum, f'Layer {layer_enum.value}')

        # Sort by PageRank and take top nodes
        sorted_cols = sorted(
            layer.minicolumns.values(),
            key=lambda c: c.pagerank,
            reverse=True
        )[:max_nodes_per_layer]

        for col in sorted_cols:
            node = {
                'id': col.id,
                'label': col.content,
                'layer': layer_enum.value,
                'layer_name': layer_name,
                'color': color,
                'pagerank': round(col.pagerank, 6),
                'tfidf': round(col.tfidf, 6),
                'activation': round(col.activation, 6),
                'occurrence_count': col.occurrence_count,
                'document_count': len(col.document_ids),
                'cluster_id': col.cluster_id
            }
            nodes.append(node)
            node_ids.add(col.id)

    # Collect lateral edges (same-layer connections)
    for layer_enum, layer in layers.items():
        if layer is None:
            continue

        for col in layer.minicolumns.values():
            if col.id not in node_ids:
                continue

            # Add typed edges with relation information
            if include_typed_edges:
                for target_id, edge_obj in col.typed_connections.items():
                    if target_id in node_ids and edge_obj.weight >= min_weight:
                        if edge_obj.confidence >= min_confidence:
                            edge_key = (col.id, target_id, edge_obj.relation_type)
                            if edge_key not in edge_set:
                                edge_set.add(edge_key)
                                edges.append({
                                    'source': col.id,
                                    'target': target_id,
                                    'weight': round(edge_obj.weight, 4),
                                    'relation_type': edge_obj.relation_type,
                                    'confidence': round(edge_obj.confidence, 4),
                                    'source_type': edge_obj.source,
                                    'edge_type': 'lateral',
                                    'color': _get_relation_color(edge_obj.relation_type)
                                })

            # Add regular lateral connections (without typed info)
            for target_id, weight in col.lateral_connections.items():
                if target_id in node_ids and weight >= min_weight:
                    # Skip if already added as typed edge
                    if include_typed_edges and target_id in col.typed_connections:
                        continue
                    edge_key = (col.id, target_id, 'co_occurrence')
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        edges.append({
                            'source': col.id,
                            'target': target_id,
                            'weight': round(weight, 4),
                            'relation_type': 'co_occurrence',
                            'confidence': 1.0,
                            'source_type': 'corpus',
                            'edge_type': 'lateral',
                            'color': '#999999'
                        })

    # Add cross-layer edges (feedforward/feedback)
    if include_cross_layer:
        for layer_enum, layer in layers.items():
            if layer is None:
                continue

            for col in layer.minicolumns.values():
                if col.id not in node_ids:
                    continue

                # Feedforward connections (to lower layers)
                for target_id, weight in col.feedforward_connections.items():
                    if target_id in node_ids and weight >= min_weight:
                        edge_key = (col.id, target_id, 'feedforward')
                        if edge_key not in edge_set:
                            edge_set.add(edge_key)
                            edges.append({
                                'source': col.id,
                                'target': target_id,
                                'weight': round(weight, 4),
                                'relation_type': 'feedforward',
                                'confidence': 1.0,
                                'source_type': 'structure',
                                'edge_type': 'cross_layer',
                                'color': '#4CAF50'  # Green
                            })

                # Feedback connections (to higher layers)
                for target_id, weight in col.feedback_connections.items():
                    if target_id in node_ids and weight >= min_weight:
                        edge_key = (col.id, target_id, 'feedback')
                        if edge_key not in edge_set:
                            edge_set.add(edge_key)
                            edges.append({
                                'source': col.id,
                                'target': target_id,
                                'weight': round(weight, 4),
                                'relation_type': 'feedback',
                                'confidence': 1.0,
                                'source_type': 'structure',
                                'edge_type': 'cross_layer',
                                'color': '#9C27B0'  # Purple
                            })

    # Add edges from semantic relations if provided
    if semantic_relations:
        for rel in semantic_relations:
            if len(rel) >= 4:
                t1, rel_type, t2, weight = rel[:4]
                # Find node IDs
                source_id = f"L0_{t1}"
                target_id = f"L0_{t2}"
                if source_id in node_ids and target_id in node_ids:
                    if weight >= min_weight:
                        edge_key = (source_id, target_id, rel_type)
                        if edge_key not in edge_set:
                            edge_set.add(edge_key)
                            edges.append({
                                'source': source_id,
                                'target': target_id,
                                'weight': round(weight, 4),
                                'relation_type': rel_type,
                                'confidence': 1.0,
                                'source_type': 'semantic',
                                'edge_type': 'semantic',
                                'color': _get_relation_color(rel_type)
                            })

    # Build graph structure
    graph = {
        'nodes': nodes,
        'edges': edges,
        'metadata': {
            'node_count': len(nodes),
            'edge_count': len(edges),
            'layers': {
                layer_enum.value: {
                    'name': LAYER_NAMES.get(layer_enum, f'Layer {layer_enum.value}'),
                    'color': LAYER_COLORS.get(layer_enum, '#808080'),
                    'node_count': sum(1 for n in nodes if n['layer'] == layer_enum.value)
                }
                for layer_enum in layers.keys()
            },
            'edge_types': _count_edge_types(edges),
            'relation_types': _count_relation_types(edges),
            'format_version': '1.0',
            'compatible_with': ['D3.js', 'Cytoscape.js', 'vis.js', 'Gephi']
        }
    }

    # Write to file
    with open(filepath, 'w') as f:
        json.dump(graph, f, indent=2)

    if verbose:
        logger.info(f"ConceptNet-style graph exported to {filepath}")
        logger.info(f"  Nodes: {len(nodes)}")
        logger.info(f"  Edges: {len(edges)}")
        logger.info(f"  Layers: {list(graph['metadata']['layers'].keys())}")
        logger.info(f"  Edge types: {graph['metadata']['edge_types']}")

    return graph


def _get_relation_color(relation_type: str) -> str:
    """Get color for a relation type."""
    relation_colors = {
        'IsA': '#E91E63',         # Pink
        'PartOf': '#9C27B0',      # Purple
        'HasA': '#673AB7',        # Deep Purple
        'UsedFor': '#3F51B5',     # Indigo
        'Causes': '#F44336',      # Red
        'HasProperty': '#FF9800', # Orange
        'AtLocation': '#4CAF50',  # Green
        'CapableOf': '#00BCD4',   # Cyan
        'SimilarTo': '#2196F3',   # Blue
        'Antonym': '#795548',     # Brown
        'RelatedTo': '#607D8B',   # Blue Grey
        'CoOccurs': '#9E9E9E',    # Grey
        'DerivedFrom': '#8BC34A', # Light Green
        'DefinedBy': '#FFEB3B',   # Yellow
        'feedforward': '#4CAF50', # Green
        'feedback': '#9C27B0',    # Purple
        'co_occurrence': '#999999',  # Grey
    }
    return relation_colors.get(relation_type, '#808080')


def _count_edge_types(edges: list) -> Dict[str, int]:
    """Count edges by edge_type."""
    counts: Dict[str, int] = {}
    for edge in edges:
        edge_type = edge.get('edge_type', 'unknown')
        counts[edge_type] = counts.get(edge_type, 0) + 1
    return counts


def _count_relation_types(edges: list) -> Dict[str, int]:
    """Count edges by relation_type."""
    counts: Dict[str, int] = {}
    for edge in edges:
        rel_type = edge.get('relation_type', 'unknown')
        counts[rel_type] = counts.get(rel_type, 0) + 1
    return counts
