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
from typing import Dict, Optional, Any

from .layers import CorticalLayer, HierarchicalLayer
from .minicolumn import Minicolumn


def save_processor(
    filepath: str,
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str],
    metadata: Optional[Dict] = None,
    verbose: bool = True
) -> None:
    """
    Save processor state to a file.
    
    Args:
        filepath: Path to save file
        layers: Dictionary of all layers
        documents: Document collection
        metadata: Optional metadata (version, settings, etc.)
        verbose: Print progress
    """
    state = {
        'version': '2.0',
        'layers': {},
        'documents': documents,
        'metadata': metadata or {}
    }
    
    # Serialize layers
    for layer_enum, layer in layers.items():
        state['layers'][layer_enum.value] = layer.to_dict()
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    if verbose:
        total_cols = sum(len(layer.minicolumns) for layer in layers.values())
        total_conns = sum(layer.total_connections() for layer in layers.values())
        print(f"✓ Saved processor to {filepath}")
        print(f"  - {len(documents)} documents")
        print(f"  - {total_cols} minicolumns")
        print(f"  - {total_conns} connections")


def load_processor(
    filepath: str,
    verbose: bool = True
) -> tuple:
    """
    Load processor state from a file.
    
    Args:
        filepath: Path to saved file
        verbose: Print progress
        
    Returns:
        Tuple of (layers, documents, metadata)
    """
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    
    # Reconstruct layers
    layers = {}
    for level_value, layer_data in state.get('layers', {}).items():
        layer = HierarchicalLayer.from_dict(layer_data)
        layers[CorticalLayer(int(level_value))] = layer
    
    documents = state.get('documents', {})
    metadata = state.get('metadata', {})
    
    if verbose:
        total_cols = sum(len(layer.minicolumns) for layer in layers.values())
        total_conns = sum(layer.total_connections() for layer in layers.values())
        print(f"✓ Loaded processor from {filepath}")
        print(f"  - {len(documents)} documents")
        print(f"  - {total_cols} minicolumns")
        print(f"  - {total_conns} connections")
    
    return layers, documents, metadata


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
        print(f"Graph exported to {filepath}")
        print(f"  - {len(nodes)} nodes, {len(edges)} edges")

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
    
    print(f"Embeddings exported to {filepath}")
    print(f"  - {len(embeddings)} terms, {data['dimensions']} dimensions")


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
    
    print(f"Relations exported to {filepath}")
    print(f"  - {len(relations)} relations")


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
