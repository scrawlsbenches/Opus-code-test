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
from typing import Dict, List, Tuple, Set, Optional, Any
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


# Default relation weights for semantic PageRank
RELATION_WEIGHTS = {
    'IsA': 1.5,           # Hypernym relationships are strong
    'PartOf': 1.3,        # Meronym relationships
    'HasProperty': 1.2,   # Property associations
    'RelatedTo': 1.0,     # Default co-occurrence
    'SimilarTo': 1.4,     # Similarity relationships
    'Causes': 1.1,        # Causal relationships
    'UsedFor': 1.0,       # Functional relationships
    'CoOccurs': 0.8,      # Basic co-occurrence
    'Antonym': 0.3,       # Opposing concepts (lower weight)
    'DerivedFrom': 1.2,   # Morphological derivation
}


def compute_semantic_pagerank(
    layer: HierarchicalLayer,
    semantic_relations: List[Tuple[str, str, str, float]],
    relation_weights: Optional[Dict[str, float]] = None,
    damping: float = 0.85,
    iterations: int = 20,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Compute PageRank with semantic relation type weighting.

    This ConceptNet-style PageRank applies different multipliers based on
    the semantic relation type between nodes. For example, IsA relationships
    are weighted more heavily than simple co-occurrence.

    Args:
        layer: The layer to compute PageRank for
        semantic_relations: List of (term1, relation, term2, weight) tuples
        relation_weights: Optional custom relation weights dict. If None, uses defaults.
        damping: Damping factor (probability of following links)
        iterations: Maximum number of iterations
        tolerance: Convergence threshold

    Returns:
        Dict containing:
        - pagerank: Dict mapping column IDs to PageRank scores
        - iterations_run: Number of iterations until convergence
        - edges_with_relations: Number of edges that had semantic relation info

    Example:
        >>> relations = [("neural", "RelatedTo", "networks", 0.8)]
        >>> result = compute_semantic_pagerank(layer, relations)
        >>> print(f"PageRank converged in {result['iterations_run']} iterations")
    """
    n = len(layer.minicolumns)
    if n == 0:
        return {'pagerank': {}, 'iterations_run': 0, 'edges_with_relations': 0}

    # Use default weights if not provided
    weights = relation_weights or RELATION_WEIGHTS

    # Build semantic relation lookup: (term1, term2) -> (relation_type, weight)
    semantic_lookup: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for t1, relation, t2, rel_weight in semantic_relations:
        # Store in both directions for undirected lookup
        semantic_lookup[(t1, t2)] = (relation, rel_weight)
        semantic_lookup[(t2, t1)] = (relation, rel_weight)

    # Initialize PageRank uniformly
    pagerank = {col.id: 1.0 / n for col in layer.minicolumns.values()}

    # Build incoming links map with relation-weighted edges
    incoming: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    outgoing_sum: Dict[str, float] = defaultdict(float)
    edges_with_relations = 0

    # Build content -> id mapping for semantic lookup
    content_to_id: Dict[str, str] = {}
    for col in layer.minicolumns.values():
        content_to_id[col.content] = col.id

    for col in layer.minicolumns.values():
        for target_id, base_weight in col.lateral_connections.items():
            target = layer.get_by_id(target_id)
            if target is None:
                continue

            # Check if there's a semantic relation between these terms
            lookup_key = (col.content, target.content)
            if lookup_key in semantic_lookup:
                relation_type, rel_weight = semantic_lookup[lookup_key]
                # Apply relation type multiplier
                type_multiplier = weights.get(relation_type, 1.0)
                # Combined weight: base_weight * relation_weight * type_multiplier
                adjusted_weight = base_weight * rel_weight * type_multiplier
                edges_with_relations += 1
            else:
                # No semantic relation, use base weight
                adjusted_weight = base_weight

            incoming[target_id].append((col.id, adjusted_weight))
            outgoing_sum[col.id] += adjusted_weight

    # Iterate until convergence
    iterations_run = 0
    for iteration in range(iterations):
        iterations_run = iteration + 1
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

    return {
        'pagerank': pagerank,
        'iterations_run': iterations_run,
        'edges_with_relations': edges_with_relations
    }


def compute_hierarchical_pagerank(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    layer_iterations: int = 10,
    global_iterations: int = 5,
    damping: float = 0.85,
    cross_layer_damping: float = 0.7,
    tolerance: float = 1e-4
) -> Dict[str, Any]:
    """
    Compute PageRank with cross-layer propagation.

    This hierarchical PageRank allows importance to flow between layers:
    - Upward: tokens → bigrams → concepts → documents
    - Downward: documents → concepts → bigrams → tokens

    The algorithm alternates between:
    1. Computing local PageRank within each layer
    2. Propagating scores up the hierarchy (via feedback_connections)
    3. Propagating scores down the hierarchy (via feedforward_connections)

    Args:
        layers: Dictionary of all layers
        layer_iterations: Max iterations for intra-layer PageRank
        global_iterations: Max iterations for cross-layer propagation
        damping: Damping factor for intra-layer PageRank
        cross_layer_damping: Damping factor for cross-layer propagation (default 0.7)
        tolerance: Convergence threshold for global iterations

    Returns:
        Dict containing:
        - iterations_run: Number of global iterations
        - converged: Whether the algorithm converged
        - layer_stats: Per-layer statistics

    Example:
        >>> result = compute_hierarchical_pagerank(layers)
        >>> print(f"Converged in {result['iterations_run']} iterations")
    """
    # Define layer order for propagation
    layer_order = [
        CorticalLayer.TOKENS,
        CorticalLayer.BIGRAMS,
        CorticalLayer.CONCEPTS,
        CorticalLayer.DOCUMENTS
    ]

    # Filter to only existing layers with minicolumns
    active_layers = [l for l in layer_order if l in layers and layers[l].column_count() > 0]

    if not active_layers:
        return {'iterations_run': 0, 'converged': True, 'layer_stats': {}}

    # Store previous PageRank values for convergence check
    prev_pageranks: Dict[CorticalLayer, Dict[str, float]] = {}

    iterations_run = 0
    converged = False

    for global_iter in range(global_iterations):
        iterations_run = global_iter + 1
        max_global_diff = 0.0

        # Step 1: Compute local PageRank for each layer
        for layer_enum in active_layers:
            layer = layers[layer_enum]
            compute_pagerank(layer, damping=damping, iterations=layer_iterations, tolerance=1e-6)

        # Step 2: Propagate up (tokens → bigrams → concepts → documents)
        for i in range(len(active_layers) - 1):
            lower_layer_enum = active_layers[i]
            upper_layer_enum = active_layers[i + 1]
            lower_layer = layers[lower_layer_enum]
            upper_layer = layers[upper_layer_enum]

            # Propagate from lower to upper via feedback connections
            for col in lower_layer.minicolumns.values():
                if not col.feedback_connections:
                    continue

                for target_id, weight in col.feedback_connections.items():
                    target = upper_layer.get_by_id(target_id)
                    if target:
                        # Boost upper layer node based on lower layer importance
                        boost = col.pagerank * weight * cross_layer_damping
                        target.pagerank += boost

        # Step 3: Propagate down (documents → concepts → bigrams → tokens)
        for i in range(len(active_layers) - 1, 0, -1):
            upper_layer_enum = active_layers[i]
            lower_layer_enum = active_layers[i - 1]
            upper_layer = layers[upper_layer_enum]
            lower_layer = layers[lower_layer_enum]

            # Propagate from upper to lower via feedforward connections
            for col in upper_layer.minicolumns.values():
                if not col.feedforward_connections:
                    continue

                for target_id, weight in col.feedforward_connections.items():
                    target = lower_layer.get_by_id(target_id)
                    if target:
                        # Boost lower layer node based on upper layer importance
                        boost = col.pagerank * weight * cross_layer_damping
                        target.pagerank += boost

        # Normalize PageRank within each layer
        for layer_enum in active_layers:
            layer = layers[layer_enum]
            total = sum(col.pagerank for col in layer.minicolumns.values())
            if total > 0:
                for col in layer.minicolumns.values():
                    col.pagerank /= total

        # Check convergence
        for layer_enum in active_layers:
            layer = layers[layer_enum]
            current_pr = {col.id: col.pagerank for col in layer.minicolumns.values()}

            if layer_enum in prev_pageranks:
                for col_id, pr in current_pr.items():
                    prev_pr = prev_pageranks[layer_enum].get(col_id, 0)
                    max_global_diff = max(max_global_diff, abs(pr - prev_pr))

            prev_pageranks[layer_enum] = current_pr

        if max_global_diff < tolerance and global_iter > 0:
            converged = True
            break

    # Collect layer statistics
    layer_stats = {}
    for layer_enum in active_layers:
        layer = layers[layer_enum]
        pageranks = [col.pagerank for col in layer.minicolumns.values()]
        layer_stats[layer_enum.name] = {
            'nodes': len(pageranks),
            'max_pagerank': max(pageranks) if pageranks else 0,
            'min_pagerank': min(pageranks) if pageranks else 0,
            'avg_pagerank': sum(pageranks) / len(pageranks) if pageranks else 0
        }

    return {
        'iterations_run': iterations_run,
        'converged': converged,
        'layer_stats': layer_stats
    }


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
    max_iterations: int = 20,
    cluster_strictness: float = 1.0,
    bridge_weight: float = 0.0
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
        cluster_strictness: Controls clustering aggressiveness (0.0-1.0).
            - 1.0 (default): Strict clustering, topics stay separate
            - 0.5: Moderate mixing, allows some cross-topic clustering
            - 0.0: Minimal clustering, most tokens group together
            Lower values create fewer, larger clusters.
        bridge_weight: Weight for synthetic inter-document connections (0.0-1.0).
            When > 0, adds weak connections between tokens that appear in
            different documents, helping bridge topic-isolated clusters.
            - 0.0 (default): No bridging
            - 0.3: Light bridging
            - 0.7: Strong bridging

    Returns:
        Dictionary mapping cluster_id to list of column contents
    """
    # Clamp parameters to valid range
    cluster_strictness = max(0.0, min(1.0, cluster_strictness))
    bridge_weight = max(0.0, min(1.0, bridge_weight))

    # Initialize each node with unique label
    labels = {col.content: i for i, col in enumerate(layer.minicolumns.values())}

    # Get column list for shuffling
    columns = list(layer.minicolumns.keys())

    # Build augmented connection weights (includes optional bridging)
    augmented_connections: Dict[str, Dict[str, float]] = defaultdict(dict)

    for content in columns:
        col = layer.minicolumns[content]
        for neighbor_id, weight in col.lateral_connections.items():
            neighbor = layer.get_by_id(neighbor_id)
            if neighbor:
                augmented_connections[content][neighbor.content] = weight

    # Add synthetic bridge connections between documents if requested
    if bridge_weight > 0:
        # Group tokens by document
        doc_tokens: Dict[str, List[str]] = defaultdict(list)
        for content in columns:
            col = layer.minicolumns[content]
            for doc_id in col.document_ids:
                doc_tokens[doc_id].append(content)

        # Create weak connections between tokens from different documents
        doc_ids = list(doc_tokens.keys())
        for i, doc1 in enumerate(doc_ids):
            for doc2 in doc_ids[i+1:]:
                tokens1 = doc_tokens[doc1]
                tokens2 = doc_tokens[doc2]
                # Connect a sample of tokens to avoid O(n²) explosion
                sample_size = min(5, len(tokens1), len(tokens2))
                for t1 in tokens1[:sample_size]:
                    for t2 in tokens2[:sample_size]:
                        if t1 != t2:
                            # Add weak bidirectional bridge
                            current = augmented_connections[t1].get(t2, 0)
                            augmented_connections[t1][t2] = current + bridge_weight * 0.5
                            current = augmented_connections[t2].get(t1, 0)
                            augmented_connections[t2][t1] = current + bridge_weight * 0.5

    # Calculate label change threshold based on strictness
    # Higher strictness = requires stronger evidence to change label
    change_threshold = (1.0 - cluster_strictness) * 0.3

    for iteration in range(max_iterations):
        changed = False

        # Process in order (could shuffle for better results)
        for content in columns:
            # Count neighbor labels weighted by connection strength
            label_weights: Dict[int, float] = defaultdict(float)

            for neighbor_content, weight in augmented_connections[content].items():
                if neighbor_content in labels:
                    label_weights[labels[neighbor_content]] += weight

            # Apply strictness: current label gets a bonus based on strictness
            current_label = labels[content]
            if current_label in label_weights and cluster_strictness < 1.0:
                # Lower strictness = stronger bias toward current label
                label_weights[current_label] *= (1 + (1 - cluster_strictness) * 2)

            # Adopt most common label
            if label_weights:
                best_label, best_weight = max(label_weights.items(), key=lambda x: x[1])
                current_weight = label_weights.get(current_label, 0)

                # Only change if the improvement exceeds threshold
                if best_label != current_label:
                    if current_weight == 0 or (best_weight / max(current_weight, 0.001) - 1) > change_threshold:
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


def compute_concept_connections(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    semantic_relations: List[Tuple[str, str, str, float]] = None,
    min_shared_docs: int = 1,
    min_jaccard: float = 0.1,
    use_member_semantics: bool = False,
    use_embedding_similarity: bool = False,
    embedding_threshold: float = 0.3,
    embeddings: Dict[str, List[float]] = None
) -> Dict[str, Any]:
    """
    Build lateral connections between concepts in Layer 2.

    Concepts are connected based on:
    1. Shared documents (Jaccard similarity of document sets)
    2. Semantic relations between member tokens (if provided)
    3. Semantic relations between members independent of docs (use_member_semantics)
    4. Embedding similarity of concept centroids (use_embedding_similarity)

    Args:
        layers: Dictionary of all layers
        semantic_relations: Optional list of (term1, relation, term2, weight) tuples
        min_shared_docs: Minimum shared documents for connection (0 to disable filter)
        min_jaccard: Minimum Jaccard similarity threshold (0.0 to disable filter)
        use_member_semantics: Connect concepts via semantic relations between members,
                              even without document overlap
        use_embedding_similarity: Connect concepts via embedding similarity of centroids
        embedding_threshold: Minimum cosine similarity for embedding-based connections
        embeddings: Term embeddings dict (required if use_embedding_similarity=True)

    Returns:
        Statistics about connections created
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer2 = layers[CorticalLayer.CONCEPTS]

    if layer2.column_count() == 0:
        return {
            'connections_created': 0,
            'concepts': 0,
            'doc_overlap_connections': 0,
            'semantic_connections': 0,
            'embedding_connections': 0
        }

    concepts = list(layer2.minicolumns.values())
    connections_created = 0
    doc_overlap_connections = 0
    semantic_connections = 0
    embedding_connections = 0

    # Build semantic relation lookup for faster access
    semantic_lookup: Dict[str, Dict[str, Tuple[str, float]]] = defaultdict(dict)
    if semantic_relations:
        for t1, relation, t2, weight in semantic_relations:
            # Store relation in both directions
            semantic_lookup[t1][t2] = (relation, weight)
            semantic_lookup[t2][t1] = (relation, weight)

    # Relation type weights for scoring
    relation_weights = {
        'IsA': 1.5,
        'PartOf': 1.3,
        'HasProperty': 1.2,
        'RelatedTo': 1.0,
        'Antonym': 0.3,
    }

    # Pre-compute member tokens for each concept (used by multiple strategies)
    concept_members: Dict[str, Set[str]] = {}
    for concept in concepts:
        members = set()
        for token_id in concept.feedforward_connections:
            token = layer0.get_by_id(token_id)
            if token:
                members.add(token.content)
        concept_members[concept.id] = members

    # Pre-compute concept centroids if using embedding similarity
    concept_centroids: Dict[str, List[float]] = {}
    if use_embedding_similarity and embeddings:
        for concept in concepts:
            members = concept_members[concept.id]
            member_embeddings = [embeddings[m] for m in members if m in embeddings]
            if member_embeddings:
                dim = len(member_embeddings[0])
                centroid = [0.0] * dim
                for emb in member_embeddings:
                    for j, v in enumerate(emb):
                        centroid[j] += v
                for j in range(dim):
                    centroid[j] /= len(member_embeddings)
                concept_centroids[concept.id] = centroid

    # Track which pairs have been connected to avoid duplicates
    connected_pairs: Set[Tuple[str, str]] = set()

    def add_connection(c1: Minicolumn, c2: Minicolumn, weight: float) -> bool:
        """Add bidirectional connection if not already connected."""
        pair = tuple(sorted([c1.id, c2.id]))
        if pair in connected_pairs:
            # Already connected, strengthen existing connection
            c1.add_lateral_connection(c2.id, weight)
            c2.add_lateral_connection(c1.id, weight)
            return False
        connected_pairs.add(pair)
        c1.add_lateral_connection(c2.id, weight)
        c2.add_lateral_connection(c1.id, weight)
        return True

    # Compare all concept pairs
    for i, concept1 in enumerate(concepts):
        docs1 = concept1.document_ids
        members1 = concept_members[concept1.id]

        for concept2 in concepts[i+1:]:
            docs2 = concept2.document_ids
            members2 = concept_members[concept2.id]

            # Strategy 1: Document overlap (traditional method)
            shared_docs = docs1 & docs2
            union_docs = docs1 | docs2
            jaccard = len(shared_docs) / len(union_docs) if union_docs else 0

            passes_doc_filter = (
                len(shared_docs) >= min_shared_docs and jaccard >= min_jaccard
            )

            if passes_doc_filter:
                # Base weight from document overlap
                weight = jaccard

                # Add semantic relation bonus if available
                if semantic_relations:
                    semantic_bonus = 0.0
                    relation_count = 0
                    for m1 in members1:
                        if m1 in semantic_lookup:
                            for m2 in members2:
                                if m2 in semantic_lookup[m1]:
                                    relation, rel_weight = semantic_lookup[m1][m2]
                                    rel_multiplier = relation_weights.get(relation, 1.0)
                                    semantic_bonus += rel_weight * rel_multiplier
                                    relation_count += 1

                    # Normalize and add semantic bonus (max 50% boost)
                    if relation_count > 0:
                        avg_semantic = semantic_bonus / relation_count
                        weight *= (1 + min(avg_semantic, 0.5))

                if add_connection(concept1, concept2, weight):
                    connections_created += 1
                    doc_overlap_connections += 1

            # Strategy 2: Member semantic relations (independent of document overlap)
            if use_member_semantics and semantic_relations and not passes_doc_filter:
                semantic_score = 0.0
                relation_count = 0
                for m1 in members1:
                    if m1 in semantic_lookup:
                        for m2 in members2:
                            if m2 in semantic_lookup[m1]:
                                relation, rel_weight = semantic_lookup[m1][m2]
                                rel_multiplier = relation_weights.get(relation, 1.0)
                                semantic_score += rel_weight * rel_multiplier
                                relation_count += 1

                if relation_count > 0:
                    # Normalize by number of relations found
                    avg_score = semantic_score / relation_count
                    # Scale to reasonable weight range (0.1 - 0.8)
                    weight = min(0.1 + avg_score * 0.3, 0.8)
                    if add_connection(concept1, concept2, weight):
                        connections_created += 1
                        semantic_connections += 1

            # Strategy 3: Embedding similarity (independent of document overlap)
            if use_embedding_similarity and embeddings and not passes_doc_filter:
                if concept1.id in concept_centroids and concept2.id in concept_centroids:
                    centroid1 = concept_centroids[concept1.id]
                    centroid2 = concept_centroids[concept2.id]
                    sim = cosine_similarity(
                        {str(i): v for i, v in enumerate(centroid1)},
                        {str(i): v for i, v in enumerate(centroid2)}
                    )
                    if sim >= embedding_threshold:
                        # Scale similarity to connection weight
                        weight = sim * 0.7  # Scale down slightly
                        if add_connection(concept1, concept2, weight):
                            connections_created += 1
                            embedding_connections += 1

    return {
        'connections_created': connections_created,
        'concepts': len(concepts),
        'doc_overlap_connections': doc_overlap_connections,
        'semantic_connections': semantic_connections,
        'embedding_connections': embedding_connections
    }


def compute_bigram_connections(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    min_shared_docs: int = 1,
    component_weight: float = 0.5,
    chain_weight: float = 0.7,
    cooccurrence_weight: float = 0.3
) -> Dict[str, Any]:
    """
    Build lateral connections between bigrams in Layer 1.

    Bigrams are connected based on:
    1. Shared component terms ("neural_networks" ↔ "neural_processing")
    2. Document co-occurrence (appear in same documents)
    3. Chains ("machine_learning" ↔ "learning_algorithms" where right=left)

    Args:
        layers: Dictionary of all layers
        min_shared_docs: Minimum shared documents for co-occurrence connection
        component_weight: Weight for shared component connections (default 0.5)
        chain_weight: Weight for chain connections (default 0.7)
        cooccurrence_weight: Weight for document co-occurrence (default 0.3)

    Returns:
        Statistics about connections created:
        - connections_created: Total bidirectional connections
        - component_connections: Connections from shared components
        - chain_connections: Connections from chains
        - cooccurrence_connections: Connections from document co-occurrence
    """
    layer1 = layers[CorticalLayer.BIGRAMS]

    if layer1.column_count() == 0:
        return {
            'connections_created': 0,
            'bigrams': 0,
            'component_connections': 0,
            'chain_connections': 0,
            'cooccurrence_connections': 0
        }

    bigrams = list(layer1.minicolumns.values())

    # Build indexes for efficient lookup
    # left_component_index: {"neural": [bigram1, bigram2, ...]}
    # right_component_index: {"networks": [bigram1, bigram3, ...]}
    left_index: Dict[str, List[Minicolumn]] = defaultdict(list)
    right_index: Dict[str, List[Minicolumn]] = defaultdict(list)

    for bigram in bigrams:
        parts = bigram.content.split('_')
        if len(parts) == 2:
            left_index[parts[0]].append(bigram)
            right_index[parts[1]].append(bigram)

    # Track connection types for statistics
    component_connections = 0
    chain_connections = 0
    cooccurrence_connections = 0

    # Track which pairs we've already connected (avoid duplicates)
    connected_pairs: Set[Tuple[str, str]] = set()

    def add_connection(b1: Minicolumn, b2: Minicolumn, weight: float, conn_type: str) -> bool:
        """Add bidirectional connection if not already connected."""
        nonlocal component_connections, chain_connections, cooccurrence_connections

        pair = tuple(sorted([b1.id, b2.id]))
        if pair in connected_pairs:
            # Already connected, just strengthen the connection
            b1.add_lateral_connection(b2.id, weight)
            b2.add_lateral_connection(b1.id, weight)
            return False

        connected_pairs.add(pair)
        b1.add_lateral_connection(b2.id, weight)
        b2.add_lateral_connection(b1.id, weight)

        if conn_type == 'component':
            component_connections += 1
        elif conn_type == 'chain':
            chain_connections += 1
        elif conn_type == 'cooccurrence':
            cooccurrence_connections += 1

        return True

    # 1. Connect bigrams sharing a component
    # Left component matches: "neural_networks" ↔ "neural_processing"
    for component, bigram_list in left_index.items():
        for i, b1 in enumerate(bigram_list):
            for b2 in bigram_list[i+1:]:
                # Weight by component's PageRank importance (if available)
                weight = component_weight
                add_connection(b1, b2, weight, 'component')

    # Right component matches: "deep_learning" ↔ "machine_learning"
    for component, bigram_list in right_index.items():
        for i, b1 in enumerate(bigram_list):
            for b2 in bigram_list[i+1:]:
                weight = component_weight
                add_connection(b1, b2, weight, 'component')

    # 2. Connect chain bigrams (right of one = left of other)
    # "machine_learning" ↔ "learning_algorithms"
    for term in left_index:
        if term in right_index:
            # term appears as right component in some bigrams and left in others
            for b_left in right_index[term]:  # ends with term
                for b_right in left_index[term]:  # starts with term
                    if b_left.id != b_right.id:
                        add_connection(b_left, b_right, chain_weight, 'chain')

    # 3. Connect bigrams that co-occur in the same documents
    for i, b1 in enumerate(bigrams):
        docs1 = b1.document_ids
        if not docs1:
            continue

        for b2 in bigrams[i+1:]:
            docs2 = b2.document_ids
            if not docs2:
                continue

            shared_docs = docs1 & docs2
            if len(shared_docs) >= min_shared_docs:
                # Weight by Jaccard similarity of document sets
                jaccard = len(shared_docs) / len(docs1 | docs2)
                weight = cooccurrence_weight * jaccard
                add_connection(b1, b2, weight, 'cooccurrence')

    return {
        'connections_created': len(connected_pairs),
        'bigrams': len(bigrams),
        'component_connections': component_connections,
        'chain_connections': chain_connections,
        'cooccurrence_connections': cooccurrence_connections
    }


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
