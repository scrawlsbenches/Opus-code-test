"""
Persistence Module
==================

Save and load functionality for the cortical processor.

Supports:
- JSON serialization for full state (git-friendly, secure)
- JSON export for graph visualization
- Incremental updates
"""

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
    verbose: bool = True
) -> None:
    """
    Save processor state to a JSON directory.

    Args:
        filepath: Path to save directory
        layers: Dictionary of all layers
        documents: Document collection
        document_metadata: Per-document metadata (source, timestamp, etc.)
        embeddings: Graph embeddings for terms (optional)
        semantic_relations: Extracted semantic relations (optional)
        metadata: Optional processor metadata (version, settings, etc.)
        verbose: Print progress
    """
    from . import state_storage
    writer = state_storage.StateWriter(filepath)

    # Save config metadata
    if metadata:
        writer.save_config(
            metadata.get('config', {}),
            metadata.get('doc_lengths', {}),
            metadata.get('avg_doc_length', 0.0)
        )

    # Get staleness info from metadata if available
    stale = set(metadata.get('stale_computations', [])) if metadata else set()

    # Save all components
    writer.save_all(
        layers=layers,
        documents=documents,
        document_metadata=document_metadata or {},
        embeddings=embeddings or {},
        semantic_relations=semantic_relations or [],
        stale_computations=stale,
        force=False,
        verbose=verbose
    )


def load_processor(
    filepath: str,
    verbose: bool = True
) -> tuple:
    """
    Load processor state from a JSON directory.

    Args:
        filepath: Path to saved directory
        verbose: Print progress

    Returns:
        Tuple of (layers, documents, document_metadata, embeddings, semantic_relations, metadata)

    Raises:
        FileNotFoundError: If filepath doesn't exist
        ValueError: If state format is invalid
    """
    from . import state_storage
    loader = state_storage.StateLoader(filepath)

    layers, documents, document_metadata, embeddings, semantic_relations, manifest_data = loader.load_all(
        validate=True,
        verbose=verbose
    )

    # Load config and BM25 metadata
    config_dict, doc_lengths, avg_doc_length = loader.load_config()

    # Combine metadata
    metadata = {
        'config': config_dict,
        'doc_lengths': doc_lengths,
        'avg_doc_length': avg_doc_length,
        'stale_computations': manifest_data.get('stale_computations', [])
    }

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
