"""
Connection building algorithms for network layers.

Contains:
- compute_document_connections: Build document-to-document similarity connections
- compute_bigram_connections: Build lateral connections between bigrams
- compute_concept_connections: Build lateral connections between concepts
"""

from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict

from ..layers import CorticalLayer, HierarchicalLayer
from ..minicolumn import Minicolumn
from .utils import cosine_similarity


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
    cooccurrence_weight: float = 0.3,
    max_bigrams_per_term: int = 100,
    max_bigrams_per_doc: int = 500,
    max_connections_per_bigram: int = 50
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
        max_bigrams_per_term: Skip terms appearing in more than this many bigrams
            to avoid O(n²) explosion from common terms like "self", "return" (default 100)
        max_bigrams_per_doc: Skip documents with more than this many bigrams for
            co-occurrence connections to avoid O(n²) explosion (default 500)
        max_connections_per_bigram: Maximum lateral connections per bigram minicolumn
            to keep graph sparse and focused on strongest connections (default 50)

    Returns:
        Statistics about connections created:
        - connections_created: Total bidirectional connections
        - component_connections: Connections from shared components
        - chain_connections: Connections from chains
        - cooccurrence_connections: Connections from document co-occurrence
        - skipped_common_terms: Number of terms skipped due to max_bigrams_per_term
        - skipped_large_docs: Number of docs skipped due to max_bigrams_per_doc
        - skipped_max_connections: Number of connections skipped due to per-bigram limit
    """
    # Without limits: O(n_bigrams²) worst case from common terms creating all-to-all connections
    # With limits: O(n_terms * max_bigrams_per_term² + n_docs * max_bigrams_per_doc²)
    # Typical with defaults (100, 500): O(n_terms * 10000 + n_docs * 250000) ≈ O(n_bigrams) linear
    layer1 = layers[CorticalLayer.BIGRAMS]

    if layer1.column_count() == 0:
        return {
            'connections_created': 0,
            'bigrams': 0,
            'component_connections': 0,
            'chain_connections': 0,
            'cooccurrence_connections': 0,
            'skipped_common_terms': 0,
            'skipped_large_docs': 0,
            'skipped_max_connections': 0
        }

    bigrams = list(layer1.minicolumns.values())

    # Build indexes for efficient lookup
    # left_component_index: {"neural": [bigram1, bigram2, ...]}
    # right_component_index: {"networks": [bigram1, bigram3, ...]}
    # Note: Bigrams use space separators (e.g., "neural networks")
    left_index: Dict[str, List[Minicolumn]] = defaultdict(list)
    right_index: Dict[str, List[Minicolumn]] = defaultdict(list)

    for bigram in bigrams:
        parts = bigram.content.split(' ')
        if len(parts) == 2:
            left_index[parts[0]].append(bigram)
            right_index[parts[1]].append(bigram)

    # Track connection types for statistics
    component_connections = 0
    chain_connections = 0
    cooccurrence_connections = 0
    skipped_max_connections = 0

    # Track which pairs we've already connected (avoid duplicates)
    connected_pairs: Set[Tuple[str, str]] = set()

    # Track connection count per bigram to enforce max_connections_per_bigram
    connection_counts: Dict[str, int] = defaultdict(int)

    # OPTIMIZATION: Accumulate all connections in memory first, then batch apply
    # This reduces ~4.7M individual add_lateral_connection calls to ~138K batch calls
    # Each batch call invalidates cache only once instead of per-connection
    pending_connections: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def add_connection(b1: Minicolumn, b2: Minicolumn, weight: float, conn_type: str) -> bool:
        """Queue bidirectional connection if not already connected and under limit."""
        nonlocal component_connections, chain_connections, cooccurrence_connections, skipped_max_connections

        pair = tuple(sorted([b1.id, b2.id]))
        if pair in connected_pairs:
            # Already connected, just strengthen the connection (accumulate weight)
            pending_connections[b1.id][b2.id] += weight
            pending_connections[b2.id][b1.id] += weight
            return False

        # Check if either bigram has reached its connection limit
        if (connection_counts[b1.id] >= max_connections_per_bigram or
            connection_counts[b2.id] >= max_connections_per_bigram):
            skipped_max_connections += 1
            return False

        connected_pairs.add(pair)
        pending_connections[b1.id][b2.id] += weight
        pending_connections[b2.id][b1.id] += weight
        connection_counts[b1.id] += 1
        connection_counts[b2.id] += 1

        if conn_type == 'component':
            component_connections += 1
        elif conn_type == 'chain':
            chain_connections += 1
        elif conn_type == 'cooccurrence':
            cooccurrence_connections += 1

        return True

    # Track skipped common terms for statistics
    skipped_common_terms = 0

    # 1. Connect bigrams sharing a component
    # Left component matches: "neural_networks" ↔ "neural_processing"
    for component, bigram_list in left_index.items():
        # Skip overly common terms to avoid O(n²) explosion
        if len(bigram_list) > max_bigrams_per_term:
            skipped_common_terms += 1
            continue
        for i, b1 in enumerate(bigram_list):
            for b2 in bigram_list[i+1:]:
                # Weight by component's PageRank importance (if available)
                weight = component_weight
                add_connection(b1, b2, weight, 'component')

    # Right component matches: "deep_learning" ↔ "machine_learning"
    for component, bigram_list in right_index.items():
        # Skip overly common terms to avoid O(n²) explosion
        if len(bigram_list) > max_bigrams_per_term:
            skipped_common_terms += 1
            continue
        for i, b1 in enumerate(bigram_list):
            for b2 in bigram_list[i+1:]:
                weight = component_weight
                add_connection(b1, b2, weight, 'component')

    # 2. Connect chain bigrams (right of one = left of other)
    # "machine_learning" ↔ "learning_algorithms"
    for term in left_index:
        if term in right_index:
            # Skip overly common terms
            if len(left_index[term]) > max_bigrams_per_term or len(right_index[term]) > max_bigrams_per_term:
                continue
            # term appears as right component in some bigrams and left in others
            for b_left in right_index[term]:  # ends with term
                for b_right in left_index[term]:  # starts with term
                    if b_left.id != b_right.id:
                        add_connection(b_left, b_right, chain_weight, 'chain')

    # 3. Connect bigrams that co-occur in the same documents
    # OPTIMIZED: Use inverted index approach instead of O(n²) matrix multiplication
    # Additional optimization: importance-based filtering and early termination
    skipped_large_docs = 0
    skipped_low_importance = 0

    # Build inverted index: doc_id -> list of bigram minicolumns
    # Sort by TF-IDF importance within each document for priority processing
    doc_to_bigrams: Dict[str, List[Minicolumn]] = defaultdict(list)
    for bigram in bigrams:
        for doc_id in bigram.document_ids:
            doc_to_bigrams[doc_id].append(bigram)

    # Compute importance threshold (median TF-IDF) for filtering
    tfidf_values = [b.tfidf for b in bigrams if b.tfidf > 0]
    importance_threshold = sorted(tfidf_values)[len(tfidf_values) // 4] if tfidf_values else 0

    # Process each document's bigram pairs
    for doc_id, doc_bigrams in doc_to_bigrams.items():
        # Skip large documents to avoid O(n²) explosion
        if len(doc_bigrams) > max_bigrams_per_doc:
            skipped_large_docs += 1
            continue

        # Filter to important bigrams only (reduces pairs quadratically)
        important_bigrams = [b for b in doc_bigrams if b.tfidf >= importance_threshold]
        if len(important_bigrams) < 2:
            continue

        # Sort by importance for priority connections
        important_bigrams.sort(key=lambda b: b.tfidf, reverse=True)

        # Connect pairs of important bigrams in this document
        # Limit to top connections per bigram to avoid explosion
        for i, b1 in enumerate(important_bigrams):
            # Early termination if this bigram is at connection limit
            if connection_counts[b1.id] >= max_connections_per_bigram:
                continue
            for b2 in important_bigrams[i+1:]:
                if connection_counts[b2.id] >= max_connections_per_bigram:
                    continue
                # Fast path: they share at least this document
                docs1 = b1.document_ids
                docs2 = b2.document_ids
                shared_docs = docs1 & docs2
                if len(shared_docs) < min_shared_docs:
                    continue
                jaccard = len(shared_docs) / len(docs1 | docs2)
                weight = cooccurrence_weight * jaccard
                add_connection(b1, b2, weight, 'cooccurrence')

    # OPTIMIZATION: Apply all accumulated connections in batch
    # This is ~34x faster than individual calls (one cache invalidation per minicolumn
    # instead of one per connection)
    for bigram_id, connections in pending_connections.items():
        bigram = layer1.get_by_id(bigram_id)
        if bigram:
            bigram.add_lateral_connections_batch(dict(connections))

    return {
        'connections_created': len(connected_pairs),
        'bigrams': len(bigrams),
        'component_connections': component_connections,
        'chain_connections': chain_connections,
        'cooccurrence_connections': cooccurrence_connections,
        'skipped_common_terms': skipped_common_terms,
        'skipped_large_docs': skipped_large_docs,
        'skipped_max_connections': skipped_max_connections
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
