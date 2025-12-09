"""
Analysis Module
===============

Graph analysis algorithms for the cortical network.

Contains implementations of:
- PageRank for importance scoring
- TF-IDF for term weighting
- Label propagation for clustering
- Activation propagation for information flow
"""

import math
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

from .layers import CorticalLayer, HierarchicalLayer
from .minicolumn import Minicolumn


def compute_pagerank(
    layer: HierarchicalLayer,
    damping: float = 0.85,
    iterations: int = 20,
    tolerance: float = 1e-6
) -> Dict[str, float]:
    """
    Compute PageRank scores for minicolumns in a layer.
    
    PageRank measures importance based on connection structure.
    Highly connected columns that are connected to other important
    columns receive higher scores.
    
    Args:
        layer: The layer to compute PageRank for
        damping: Damping factor (probability of following links)
        iterations: Maximum number of iterations
        tolerance: Convergence threshold
        
    Returns:
        Dictionary mapping column IDs to PageRank scores
    """
    n = len(layer.minicolumns)
    if n == 0:
        return {}
    
    # Initialize PageRank uniformly
    pagerank = {col.id: 1.0 / n for col in layer.minicolumns.values()}
    
    # Build incoming links map
    incoming: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    outgoing_sum: Dict[str, float] = defaultdict(float)
    
    for col in layer.minicolumns.values():
        for target_id, weight in col.lateral_connections.items():
            # Use O(1) lookup via get_by_id instead of O(n) linear search
            if layer.get_by_id(target_id) is not None:
                incoming[target_id].append((col.id, weight))
                outgoing_sum[col.id] += weight
    
    # Iterate until convergence
    for iteration in range(iterations):
        new_pagerank = {}
        max_diff = 0.0
        
        for col in layer.minicolumns.values():
            # Sum of weighted incoming PageRank
            incoming_sum = 0.0
            for source_id, weight in incoming[col.id]:
                if source_id in pagerank and outgoing_sum[source_id] > 0:
                    incoming_sum += pagerank[source_id] * weight / outgoing_sum[source_id]
            
            # Apply damping
            new_rank = (1 - damping) / n + damping * incoming_sum
            new_pagerank[col.id] = new_rank
            
            max_diff = max(max_diff, abs(new_rank - pagerank.get(col.id, 0)))
        
        pagerank = new_pagerank
        
        if max_diff < tolerance:
            break
    
    # Update minicolumn pagerank values
    for col in layer.minicolumns.values():
        col.pagerank = pagerank.get(col.id, 1.0 / n)
    
    return pagerank


def compute_tfidf(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str]
) -> None:
    """
    Compute TF-IDF scores for tokens.
    
    TF-IDF (Term Frequency - Inverse Document Frequency) measures
    how distinctive a term is to the corpus. High TF-IDF terms are
    both frequent in their documents and rare across the corpus.
    
    Args:
        layers: Dictionary of layers (needs TOKENS layer)
        documents: Dictionary mapping doc_id to content
    """
    layer0 = layers[CorticalLayer.TOKENS]
    num_docs = len(documents)
    
    if num_docs == 0:
        return
    
    for col in layer0.minicolumns.values():
        # Document frequency
        df = len(col.document_ids)
        
        if df > 0:
            # Inverse document frequency
            idf = math.log(num_docs / df)
            
            # Term frequency (normalized by occurrence count)
            tf = math.log1p(col.occurrence_count)
            
            # TF-IDF
            col.tfidf = tf * idf
            
            # Per-document TF-IDF using actual occurrence counts
            for doc_id in col.document_ids:
                # Get actual term frequency in this document
                doc_tf = col.doc_occurrence_counts.get(doc_id, 1)
                col.tfidf_per_doc[doc_id] = math.log1p(doc_tf) * idf


def propagate_activation(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    iterations: int = 3,
    decay: float = 0.8,
    lateral_weight: float = 0.3
) -> None:
    """
    Propagate activation through the network.
    
    This simulates how information flows through cortical layers:
    - Activation spreads to connected columns (lateral)
    - Activation flows up the hierarchy (feedforward)
    - Activation decays over time
    
    Args:
        layers: Dictionary of all layers
        iterations: Number of propagation iterations
        decay: How much activation decays per iteration
        lateral_weight: Weight for lateral spreading
    """
    for _ in range(iterations):
        # Store new activations
        new_activations: Dict[str, float] = {}
        
        # Process each layer
        for layer_enum in CorticalLayer:
            if layer_enum not in layers:
                continue
            layer = layers[layer_enum]
            
            for col in layer.minicolumns.values():
                # Start with decayed current activation
                new_act = col.activation * decay
                
                # Add lateral input using O(1) ID lookup
                for neighbor_id, weight in col.lateral_connections.items():
                    neighbor = layer.get_by_id(neighbor_id)
                    if neighbor:
                        new_act += neighbor.activation * weight * lateral_weight
                
                # Add feedforward input using O(1) ID lookup
                for source_id in col.feedforward_sources:
                    # Find source in lower layers
                    for lower_enum in CorticalLayer:
                        if lower_enum >= layer_enum:
                            break
                        if lower_enum not in layers:
                            continue
                        lower_layer = layers[lower_enum]
                        source = lower_layer.get_by_id(source_id)
                        if source:
                            new_act += source.activation * 0.5
                            break
                
                new_activations[col.id] = new_act
        
        # Apply new activations
        for layer_enum in CorticalLayer:
            if layer_enum not in layers:
                continue
            layer = layers[layer_enum]
            for col in layer.minicolumns.values():
                if col.id in new_activations:
                    col.activation = new_activations[col.id]


def cluster_by_label_propagation(
    layer: HierarchicalLayer,
    min_cluster_size: int = 3,
    max_iterations: int = 20
) -> Dict[int, List[str]]:
    """
    Cluster minicolumns using label propagation.
    
    Label propagation is a semi-supervised community detection
    algorithm. Each node adopts the most common label among its
    neighbors, causing labels to propagate through densely
    connected regions.
    
    Args:
        layer: Layer to cluster
        min_cluster_size: Minimum nodes per cluster
        max_iterations: Maximum iterations
        
    Returns:
        Dictionary mapping cluster_id to list of column contents
    """
    # Initialize each node with unique label
    labels = {col.content: i for i, col in enumerate(layer.minicolumns.values())}
    
    # Get column list for shuffling
    columns = list(layer.minicolumns.keys())
    
    for iteration in range(max_iterations):
        changed = False
        
        # Process in order (could shuffle for better results)
        for content in columns:
            col = layer.minicolumns[content]
            
            # Count neighbor labels weighted by connection strength
            label_weights: Dict[int, float] = defaultdict(float)
            
            for neighbor_id, weight in col.lateral_connections.items():
                # Use O(1) ID lookup instead of linear search
                neighbor = layer.get_by_id(neighbor_id)
                if neighbor and neighbor.content in labels:
                    label_weights[labels[neighbor.content]] += weight
            
            # Adopt most common label
            if label_weights:
                best_label = max(label_weights.items(), key=lambda x: x[1])[0]
                if labels[content] != best_label:
                    labels[content] = best_label
                    changed = True
        
        if not changed:
            break
    
    # Build clusters
    clusters: Dict[int, List[str]] = defaultdict(list)
    for content, label in labels.items():
        clusters[label].append(content)
    
    # Filter by minimum size
    filtered = {
        label: members 
        for label, members in clusters.items() 
        if len(members) >= min_cluster_size
    }
    
    # Update cluster_id on minicolumns
    for label, members in filtered.items():
        for content in members:
            if content in layer.minicolumns:
                layer.minicolumns[content].cluster_id = label
    
    return filtered


def build_concept_clusters(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    clusters: Dict[int, List[str]]
) -> None:
    """
    Build concept layer from token clusters.
    
    Creates Layer 2 (Concepts) minicolumns from clustered tokens.
    Each concept is named after its most important members.
    
    Args:
        layers: Dictionary of all layers
        clusters: Cluster dictionary from label propagation
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer2 = layers[CorticalLayer.CONCEPTS]
    
    for cluster_id, members in clusters.items():
        if len(members) < 2:
            continue
        
        # Get member columns and sort by PageRank
        member_cols = []
        for m in members:
            col = layer0.get_minicolumn(m)
            if col:
                member_cols.append(col)
        
        if not member_cols:
            continue
        
        member_cols.sort(key=lambda c: c.pagerank, reverse=True)
        
        # Name concept after top members
        top_names = [c.content for c in member_cols[:3]]
        concept_name = '/'.join(top_names)
        
        # Create concept minicolumn
        concept = layer2.get_or_create_minicolumn(concept_name)
        concept.cluster_id = cluster_id
        
        # Aggregate properties from members with weighted connections
        max_pagerank = max(c.pagerank for c in member_cols) if member_cols else 1.0
        for col in member_cols:
            concept.feedforward_sources.add(col.id)
            concept.document_ids.update(col.document_ids)
            concept.activation += col.activation * 0.5
            concept.occurrence_count += col.occurrence_count
            # Weighted feedforward: concept → token (weight by normalized PageRank)
            weight = col.pagerank / max_pagerank if max_pagerank > 0 else 1.0
            concept.add_feedforward_connection(col.id, weight)
            # Weighted feedback: token → concept (weight by normalized PageRank)
            col.add_feedback_connection(concept.id, weight)

        # Set PageRank as average of members
        concept.pagerank = sum(c.pagerank for c in member_cols) / len(member_cols)


def compute_document_connections(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str],
    min_shared_terms: int = 3
) -> None:
    """
    Build lateral connections between documents.
    
    Documents are connected based on shared vocabulary,
    weighted by TF-IDF scores of shared terms.
    
    Args:
        layers: Dictionary of all layers
        documents: Dictionary of documents
        min_shared_terms: Minimum shared terms for connection
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer3 = layers[CorticalLayer.DOCUMENTS]
    
    doc_ids = list(documents.keys())
    
    for i, doc1 in enumerate(doc_ids):
        col1 = layer3.get_minicolumn(doc1)
        if not col1:
            col1 = layer3.get_or_create_minicolumn(doc1)
        
        for doc2 in doc_ids[i+1:]:
            col2 = layer3.get_minicolumn(doc2)
            if not col2:
                col2 = layer3.get_or_create_minicolumn(doc2)
            
            # Find shared terms
            shared_weight = 0.0
            shared_count = 0
            
            for token_col in layer0.minicolumns.values():
                if doc1 in token_col.document_ids and doc2 in token_col.document_ids:
                    # Weight by TF-IDF
                    weight = token_col.tfidf
                    shared_weight += weight
                    shared_count += 1
            
            if shared_count >= min_shared_terms:
                col1.add_lateral_connection(col2.id, shared_weight)
                col2.add_lateral_connection(col1.id, shared_weight)


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two sparse vectors.
    
    Args:
        vec1: First vector as dict of term -> weight
        vec2: Second vector as dict of term -> weight
        
    Returns:
        Cosine similarity between 0 and 1
    """
    # Find common keys
    common = set(vec1.keys()) & set(vec2.keys())
    
    if not common:
        return 0.0
    
    # Compute dot product
    dot = sum(vec1[k] * vec2[k] for k in common)
    
    # Compute magnitudes
    mag1 = math.sqrt(sum(v * v for v in vec1.values()))
    mag2 = math.sqrt(sum(v * v for v in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot / (mag1 * mag2)
