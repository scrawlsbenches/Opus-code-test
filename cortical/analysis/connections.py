"""
Connection building algorithms for network layers.

Contains:
- compute_document_connections: Build document-to-document similarity connections
- compute_bigram_connections: Build lateral connections between bigrams
- compute_concept_connections: Build lateral connections between concepts
"""

from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import shared_memory
import struct
import os

from ..layers import CorticalLayer, HierarchicalLayer
from ..minicolumn import Minicolumn
from .utils import cosine_similarity


# =============================================================================
# SHARED MEMORY PARALLEL INFRASTRUCTURE
# =============================================================================

def _popcount64(x: int) -> int:
    """Count set bits in a 64-bit integer (Hamming weight)."""
    # Brian Kernighan's algorithm - O(set bits)
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


def _jaccard_from_bits(bits1: List[int], bits2: List[int]) -> float:
    """Compute Jaccard similarity from bit vectors using bitwise ops."""
    intersection = 0
    union = 0
    for b1, b2 in zip(bits1, bits2):
        intersection += _popcount64(b1 & b2)
        union += _popcount64(b1 | b2)
    return intersection / union if union > 0 else 0.0


@dataclass
class CompactBigramData:
    """
    Compact representation of bigram data for shared memory parallelism.

    All string IDs are converted to integer indices for minimal memory footprint.
    Document sets are encoded as bit vectors for fast Jaccard computation.
    """
    n_bigrams: int
    n_docs: int
    n_terms: int
    bits_per_bigram: int  # Number of uint64s needed per bigram

    # Index mappings (kept in main process)
    bigram_id_to_idx: Dict[str, int] = field(default_factory=dict)
    idx_to_bigram_id: List[str] = field(default_factory=list)
    term_to_idx: Dict[str, int] = field(default_factory=dict)
    doc_to_bit_pos: Dict[str, int] = field(default_factory=dict)

    # Flat arrays for shared memory (as bytes)
    left_term_data: bytes = b''      # int32[n_bigrams]
    right_term_data: bytes = b''     # int32[n_bigrams]
    tfidf_data: bytes = b''          # float32[n_bigrams]
    doc_bits_data: bytes = b''       # uint64[n_bigrams * bits_per_bigram]

    # Term groupings (serialized as length-prefixed arrays)
    left_groups_data: bytes = b''    # For each term: [n_bigrams, idx1, idx2, ...]
    right_groups_data: bytes = b''


def _build_compact_data(bigrams: List['Minicolumn']) -> CompactBigramData:
    """
    Build compact data structures from bigram Minicolumns.

    Converts all data to integer indices and bit vectors.
    """
    # Collect all unique terms and documents
    all_terms: Set[str] = set()
    all_docs: Set[str] = set()

    for bigram in bigrams:
        parts = bigram.content.split(' ')
        if len(parts) == 2:
            all_terms.add(parts[0])
            all_terms.add(parts[1])
        all_docs.update(bigram.document_ids)

    # Create mappings
    term_to_idx = {term: i for i, term in enumerate(sorted(all_terms))}
    doc_to_bit_pos = {doc: i for i, doc in enumerate(sorted(all_docs))}

    n_bigrams = len(bigrams)
    n_docs = len(all_docs)
    n_terms = len(all_terms)
    bits_per_bigram = (n_docs + 63) // 64  # Ceiling division

    # Pre-allocate arrays
    left_terms = []
    right_terms = []
    tfidfs = []
    doc_bits = []

    bigram_id_to_idx = {}
    idx_to_bigram_id = []

    # Group bigrams by left/right terms
    left_groups: Dict[int, List[int]] = defaultdict(list)
    right_groups: Dict[int, List[int]] = defaultdict(list)

    for idx, bigram in enumerate(bigrams):
        bigram_id_to_idx[bigram.id] = idx
        idx_to_bigram_id.append(bigram.id)

        parts = bigram.content.split(' ')
        if len(parts) == 2:
            left_idx = term_to_idx[parts[0]]
            right_idx = term_to_idx[parts[1]]
            left_terms.append(left_idx)
            right_terms.append(right_idx)
            left_groups[left_idx].append(idx)
            right_groups[right_idx].append(idx)
        else:
            left_terms.append(-1)
            right_terms.append(-1)

        tfidfs.append(bigram.tfidf)

        # Encode document_ids as bit vector
        bits = [0] * bits_per_bigram
        for doc_id in bigram.document_ids:
            bit_pos = doc_to_bit_pos[doc_id]
            word_idx = bit_pos // 64
            bit_idx = bit_pos % 64
            bits[word_idx] |= (1 << bit_idx)
        doc_bits.extend(bits)

    # Pack arrays into bytes
    left_term_data = struct.pack(f'{n_bigrams}i', *left_terms)
    right_term_data = struct.pack(f'{n_bigrams}i', *right_terms)
    tfidf_data = struct.pack(f'{n_bigrams}f', *tfidfs)
    doc_bits_data = struct.pack(f'{n_bigrams * bits_per_bigram}Q', *doc_bits)

    # Pack term groups (length-prefixed)
    left_groups_parts = []
    for term_idx in range(n_terms):
        group = left_groups.get(term_idx, [])
        left_groups_parts.append(struct.pack('I', len(group)))
        if group:
            left_groups_parts.append(struct.pack(f'{len(group)}I', *group))
    left_groups_data = b''.join(left_groups_parts)

    right_groups_parts = []
    for term_idx in range(n_terms):
        group = right_groups.get(term_idx, [])
        right_groups_parts.append(struct.pack('I', len(group)))
        if group:
            right_groups_parts.append(struct.pack(f'{len(group)}I', *group))
    right_groups_data = b''.join(right_groups_parts)

    return CompactBigramData(
        n_bigrams=n_bigrams,
        n_docs=n_docs,
        n_terms=n_terms,
        bits_per_bigram=bits_per_bigram,
        bigram_id_to_idx=bigram_id_to_idx,
        idx_to_bigram_id=idx_to_bigram_id,
        term_to_idx=term_to_idx,
        doc_to_bit_pos=doc_to_bit_pos,
        left_term_data=left_term_data,
        right_term_data=right_term_data,
        tfidf_data=tfidf_data,
        doc_bits_data=doc_bits_data,
        left_groups_data=left_groups_data,
        right_groups_data=right_groups_data,
    )


def _worker_process_components(
    shm_name: str,
    n_bigrams: int,
    n_terms: int,
    bits_per_bigram: int,
    term_range: Tuple[int, int],  # (start_term_idx, end_term_idx)
    is_left: bool,
    component_weight: float,
    max_bigrams_per_term: int,
    max_connections_per_bigram: int,
    groups_offset: int,
    groups_size: int,
) -> List[Tuple[int, int, float]]:
    """
    Worker function for component connections using shared memory.

    Returns list of (bigram_idx1, bigram_idx2, weight) tuples.
    """
    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        # Parse groups data from shared memory
        # Groups are at the end: left_groups then right_groups
        groups_start = groups_offset

        connections = []
        local_counts: Dict[int, int] = {}
        seen_pairs: Set[Tuple[int, int]] = set()

        # Navigate to correct term range
        offset = groups_start
        for term_idx in range(term_range[0]):
            # Skip this term's group
            group_len = struct.unpack_from('I', shm.buf, offset)[0]
            offset += 4 + group_len * 4

        # Process terms in range
        for term_idx in range(term_range[0], term_range[1]):
            group_len = struct.unpack_from('I', shm.buf, offset)[0]
            offset += 4

            if group_len == 0:
                continue

            if group_len > max_bigrams_per_term:
                continue  # Skip overly common terms

            # Read bigram indices for this term
            bigram_indices = list(struct.unpack_from(f'{group_len}I', shm.buf, offset))
            offset += group_len * 4

            # Connect all pairs
            for i, idx1 in enumerate(bigram_indices):
                if local_counts.get(idx1, 0) >= max_connections_per_bigram:
                    continue
                for idx2 in bigram_indices[i+1:]:
                    if local_counts.get(idx2, 0) >= max_connections_per_bigram:
                        continue

                    pair = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
                    if pair in seen_pairs:
                        continue

                    seen_pairs.add(pair)
                    connections.append((idx1, idx2, component_weight))
                    local_counts[idx1] = local_counts.get(idx1, 0) + 1
                    local_counts[idx2] = local_counts.get(idx2, 0) + 1

        return connections
    finally:
        shm.close()


def _worker_process_cooccurrence(
    shm_name: str,
    n_bigrams: int,
    bits_per_bigram: int,
    bigram_range: Tuple[int, int],  # (start_idx, end_idx)
    cooccurrence_weight: float,
    min_shared_docs: int,
    max_connections_per_bigram: int,
    tfidf_offset: int,
    doc_bits_offset: int,
    importance_threshold: float,
) -> List[Tuple[int, int, float]]:
    """
    Worker function for co-occurrence connections using shared memory bit vectors.

    Uses bitwise AND/OR for fast Jaccard computation.
    """
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        connections = []
        local_counts: Dict[int, int] = {}
        seen_pairs: Set[Tuple[int, int]] = set()

        start_idx, end_idx = bigram_range

        # Process bigram pairs in range
        for idx1 in range(start_idx, end_idx):
            if local_counts.get(idx1, 0) >= max_connections_per_bigram:
                continue

            # Get tfidf for idx1
            tfidf1 = struct.unpack_from('f', shm.buf, tfidf_offset + idx1 * 4)[0]
            if tfidf1 < importance_threshold:
                continue

            # Get doc bits for idx1
            bits1_offset = doc_bits_offset + idx1 * bits_per_bigram * 8
            bits1 = list(struct.unpack_from(f'{bits_per_bigram}Q', shm.buf, bits1_offset))

            # Compare with subsequent bigrams
            for idx2 in range(idx1 + 1, min(idx1 + 1000, n_bigrams)):  # Limit comparisons
                if local_counts.get(idx2, 0) >= max_connections_per_bigram:
                    continue

                tfidf2 = struct.unpack_from('f', shm.buf, tfidf_offset + idx2 * 4)[0]
                if tfidf2 < importance_threshold:
                    continue

                # Fast Jaccard via bit operations
                bits2_offset = doc_bits_offset + idx2 * bits_per_bigram * 8
                bits2 = list(struct.unpack_from(f'{bits_per_bigram}Q', shm.buf, bits2_offset))

                intersection = 0
                union = 0
                for b1, b2 in zip(bits1, bits2):
                    intersection += _popcount64(b1 & b2)
                    union += _popcount64(b1 | b2)

                if intersection < min_shared_docs:
                    continue

                jaccard = intersection / union if union > 0 else 0.0
                weight = cooccurrence_weight * jaccard

                pair = (idx1, idx2)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    connections.append((idx1, idx2, weight))
                    local_counts[idx1] = local_counts.get(idx1, 0) + 1
                    local_counts[idx2] = local_counts.get(idx2, 0) + 1

        return connections
    finally:
        shm.close()


def _compute_bigram_connections_shared_memory(
    layer1: HierarchicalLayer,
    bigrams: List[Minicolumn],
    min_shared_docs: int,
    component_weight: float,
    chain_weight: float,
    cooccurrence_weight: float,
    max_bigrams_per_term: int,
    max_bigrams_per_doc: int,
    max_connections_per_bigram: int,
    n_workers: int
) -> Dict[str, Any]:
    """
    Shared memory parallel implementation of bigram connection building.

    Uses compact integer indices and bit vectors for minimal memory/communication.
    """
    # Build compact data structures
    compact = _build_compact_data(bigrams)

    # Calculate memory layout
    left_term_offset = 0
    right_term_offset = left_term_offset + len(compact.left_term_data)
    tfidf_offset = right_term_offset + len(compact.right_term_data)
    doc_bits_offset = tfidf_offset + len(compact.tfidf_data)
    left_groups_offset = doc_bits_offset + len(compact.doc_bits_data)
    right_groups_offset = left_groups_offset + len(compact.left_groups_data)
    total_size = right_groups_offset + len(compact.right_groups_data)

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=total_size)

    try:
        # Copy data to shared memory
        shm.buf[left_term_offset:right_term_offset] = compact.left_term_data
        shm.buf[right_term_offset:tfidf_offset] = compact.right_term_data
        shm.buf[tfidf_offset:doc_bits_offset] = compact.tfidf_data
        shm.buf[doc_bits_offset:left_groups_offset] = compact.doc_bits_data
        shm.buf[left_groups_offset:right_groups_offset] = compact.left_groups_data
        shm.buf[right_groups_offset:total_size] = compact.right_groups_data

        all_connections: List[Tuple[int, int, float]] = []

        # Compute importance threshold
        tfidf_values = [b.tfidf for b in bigrams if b.tfidf > 0]
        importance_threshold = sorted(tfidf_values)[len(tfidf_values) // 4] if tfidf_values else 0

        # Partition terms for component workers
        terms_per_worker = (compact.n_terms + n_workers - 1) // n_workers

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []

            # Submit left component jobs
            for i in range(n_workers):
                start_term = i * terms_per_worker
                end_term = min((i + 1) * terms_per_worker, compact.n_terms)
                if start_term < end_term:
                    futures.append(executor.submit(
                        _worker_process_components,
                        shm.name,
                        compact.n_bigrams,
                        compact.n_terms,
                        compact.bits_per_bigram,
                        (start_term, end_term),
                        True,  # is_left
                        component_weight,
                        max_bigrams_per_term,
                        max_connections_per_bigram,
                        left_groups_offset,
                        len(compact.left_groups_data),
                    ))

            # Submit right component jobs
            for i in range(n_workers):
                start_term = i * terms_per_worker
                end_term = min((i + 1) * terms_per_worker, compact.n_terms)
                if start_term < end_term:
                    futures.append(executor.submit(
                        _worker_process_components,
                        shm.name,
                        compact.n_bigrams,
                        compact.n_terms,
                        compact.bits_per_bigram,
                        (start_term, end_term),
                        False,  # is_left (right)
                        component_weight,
                        max_bigrams_per_term,
                        max_connections_per_bigram,
                        right_groups_offset,
                        len(compact.right_groups_data),
                    ))

            # Submit cooccurrence jobs
            bigrams_per_worker = (compact.n_bigrams + n_workers - 1) // n_workers
            for i in range(n_workers):
                start_idx = i * bigrams_per_worker
                end_idx = min((i + 1) * bigrams_per_worker, compact.n_bigrams)
                if start_idx < end_idx:
                    futures.append(executor.submit(
                        _worker_process_cooccurrence,
                        shm.name,
                        compact.n_bigrams,
                        compact.bits_per_bigram,
                        (start_idx, end_idx),
                        cooccurrence_weight,
                        min_shared_docs,
                        max_connections_per_bigram,
                        tfidf_offset,
                        doc_bits_offset,
                        importance_threshold,
                    ))

            # Collect results
            for future in as_completed(futures):
                try:
                    connections = future.result()
                    all_connections.extend(connections)
                except Exception as e:
                    import sys
                    print(f"Worker error: {e}", file=sys.stderr)

        # Merge and deduplicate connections
        seen_pairs: Set[Tuple[int, int]] = set()
        connection_counts: Dict[int, int] = defaultdict(int)
        final_connections: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for idx1, idx2, weight in all_connections:
            pair = (idx1, idx2) if idx1 < idx2 else (idx2, idx1)
            if pair in seen_pairs:
                # Accumulate weight
                id1 = compact.idx_to_bigram_id[idx1]
                id2 = compact.idx_to_bigram_id[idx2]
                final_connections[id1][id2] += weight
                final_connections[id2][id1] += weight
            else:
                if (connection_counts[idx1] >= max_connections_per_bigram or
                    connection_counts[idx2] >= max_connections_per_bigram):
                    continue
                seen_pairs.add(pair)
                id1 = compact.idx_to_bigram_id[idx1]
                id2 = compact.idx_to_bigram_id[idx2]
                final_connections[id1][id2] += weight
                final_connections[id2][id1] += weight
                connection_counts[idx1] += 1
                connection_counts[idx2] += 1

        # Apply connections to bigrams
        for bigram_id, connections in final_connections.items():
            bigram = layer1.get_by_id(bigram_id)
            if bigram:
                bigram.add_lateral_connections_batch(dict(connections))

        return {
            'connections_created': len(seen_pairs),
            'bigrams': compact.n_bigrams,
            'parallel': True,
            'parallel_mode': 'shared_memory',
            'n_workers': n_workers,
            'shared_memory_size_mb': total_size / (1024 * 1024),
        }

    finally:
        shm.close()
        shm.unlink()


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


# =============================================================================
# PARALLEL BIGRAM CONNECTION HELPERS
# =============================================================================

@dataclass
class BigramData:
    """Serializable bigram data for process-based parallelism."""
    id: str
    left_term: str
    right_term: str
    document_ids: frozenset  # frozenset is hashable and picklable
    tfidf: float


@dataclass
class SimpleWorkerResult:
    """
    Simple result for process-based workers.

    Uses only basic types (tuples, dicts, ints) for easy pickling.
    """
    # List of (id1, id2, weight) tuples
    connections: List[Tuple[str, str, float]] = field(default_factory=list)
    # Statistics
    component_connections: int = 0
    chain_connections: int = 0
    cooccurrence_connections: int = 0
    skipped_common_terms: int = 0
    skipped_large_docs: int = 0


def _process_component_batch_simple(
    component_items: List[Tuple[str, List[str]]],  # (component, [bigram_ids])
    bigram_id_to_data: Dict[str, Tuple[str, str]],  # id -> (left, right)
    component_weight: float,
    max_bigrams_per_term: int,
    max_connections_per_bigram: int
) -> SimpleWorkerResult:
    """
    Process component batch using only simple serializable data.

    This can run in a separate process (no GIL contention).
    """
    result = SimpleWorkerResult()
    local_counts: Dict[str, int] = {}
    seen_pairs: Set[Tuple[str, str]] = set()

    for component, bigram_ids in component_items:
        if len(bigram_ids) > max_bigrams_per_term:
            result.skipped_common_terms += 1
            continue

        for i, id1 in enumerate(bigram_ids):
            if local_counts.get(id1, 0) >= max_connections_per_bigram:
                continue
            for id2 in bigram_ids[i+1:]:
                if local_counts.get(id2, 0) >= max_connections_per_bigram:
                    continue

                # Canonical pair
                pair = (id1, id2) if id1 < id2 else (id2, id1)
                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)
                result.connections.append((id1, id2, component_weight))
                local_counts[id1] = local_counts.get(id1, 0) + 1
                local_counts[id2] = local_counts.get(id2, 0) + 1
                result.component_connections += 1

    return result


def _process_chain_batch_simple(
    terms: List[str],
    left_index: Dict[str, List[str]],  # term -> [bigram_ids]
    right_index: Dict[str, List[str]],  # term -> [bigram_ids]
    chain_weight: float,
    max_bigrams_per_term: int,
    max_connections_per_bigram: int
) -> SimpleWorkerResult:
    """Process chain connections using simple data."""
    result = SimpleWorkerResult()
    local_counts: Dict[str, int] = {}
    seen_pairs: Set[Tuple[str, str]] = set()

    for term in terms:
        if term not in right_index:
            continue
        left_list = left_index.get(term, [])
        right_list = right_index.get(term, [])

        if len(left_list) > max_bigrams_per_term or len(right_list) > max_bigrams_per_term:
            continue

        for id_left in right_list:  # ends with term
            if local_counts.get(id_left, 0) >= max_connections_per_bigram:
                continue
            for id_right in left_list:  # starts with term
                if id_left == id_right:
                    continue
                if local_counts.get(id_right, 0) >= max_connections_per_bigram:
                    continue

                pair = (id_left, id_right) if id_left < id_right else (id_right, id_left)
                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)
                result.connections.append((id_left, id_right, chain_weight))
                local_counts[id_left] = local_counts.get(id_left, 0) + 1
                local_counts[id_right] = local_counts.get(id_right, 0) + 1
                result.chain_connections += 1

    return result


def _process_cooccurrence_batch_simple(
    doc_items: List[Tuple[str, List[Tuple[str, frozenset, float]]]],  # (doc_id, [(bigram_id, doc_ids, tfidf)])
    cooccurrence_weight: float,
    min_shared_docs: int,
    max_bigrams_per_doc: int,
    max_connections_per_bigram: int,
    importance_threshold: float
) -> SimpleWorkerResult:
    """Process co-occurrence connections using simple data."""
    result = SimpleWorkerResult()
    local_counts: Dict[str, int] = {}
    seen_pairs: Set[Tuple[str, str]] = set()

    for doc_id, bigram_data_list in doc_items:
        if len(bigram_data_list) > max_bigrams_per_doc:
            result.skipped_large_docs += 1
            continue

        # Filter to important bigrams
        important = [(bid, docs, tfidf) for bid, docs, tfidf in bigram_data_list
                     if tfidf >= importance_threshold]
        if len(important) < 2:
            continue

        # Sort by importance
        important.sort(key=lambda x: x[2], reverse=True)

        for i, (id1, docs1, _) in enumerate(important):
            if local_counts.get(id1, 0) >= max_connections_per_bigram:
                continue
            for id2, docs2, _ in important[i+1:]:
                if local_counts.get(id2, 0) >= max_connections_per_bigram:
                    continue

                shared = docs1 & docs2
                if len(shared) < min_shared_docs:
                    continue

                jaccard = len(shared) / len(docs1 | docs2)
                weight = cooccurrence_weight * jaccard

                pair = (id1, id2) if id1 < id2 else (id2, id1)
                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)
                result.connections.append((id1, id2, weight))
                local_counts[id1] = local_counts.get(id1, 0) + 1
                local_counts[id2] = local_counts.get(id2, 0) + 1
                result.cooccurrence_connections += 1

    return result


@dataclass
class WorkerResult:
    """Accumulator for parallel worker results (lock-free pattern)."""
    # Pairs as (min_id, max_id) tuples for deduplication
    pairs: Set[Tuple[str, str]] = field(default_factory=set)
    # Pending connections: bigram_id -> {target_id: weight}
    pending: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    # Statistics
    component_connections: int = 0
    chain_connections: int = 0
    cooccurrence_connections: int = 0
    skipped_common_terms: int = 0
    skipped_large_docs: int = 0
    skipped_max_connections: int = 0


def _process_component_batch(
    component_items: List[Tuple[str, List[Minicolumn]]],
    component_weight: float,
    max_bigrams_per_term: int,
    max_connections_per_bigram: int,
    connection_type: str  # 'left' or 'right'
) -> WorkerResult:
    """
    Process a batch of component groups in parallel.

    Thread-local accumulation - no shared mutable state.
    """
    result = WorkerResult()
    # Local connection counts (will be merged later)
    local_counts: Dict[str, int] = defaultdict(int)

    for component, bigram_list in component_items:
        # Skip overly common terms
        if len(bigram_list) > max_bigrams_per_term:
            result.skipped_common_terms += 1
            continue

        for i, b1 in enumerate(bigram_list):
            if local_counts[b1.id] >= max_connections_per_bigram:
                continue
            for b2 in bigram_list[i+1:]:
                if local_counts[b2.id] >= max_connections_per_bigram:
                    continue

                # Canonical pair for deduplication
                pair = (b1.id, b2.id) if b1.id < b2.id else (b2.id, b1.id)

                if pair in result.pairs:
                    # Already connected in this batch, strengthen
                    result.pending[b1.id][b2.id] += component_weight
                    result.pending[b2.id][b1.id] += component_weight
                    continue

                result.pairs.add(pair)
                result.pending[b1.id][b2.id] += component_weight
                result.pending[b2.id][b1.id] += component_weight
                local_counts[b1.id] += 1
                local_counts[b2.id] += 1
                result.component_connections += 1

    return result


def _process_chain_batch(
    terms: List[str],
    left_index: Dict[str, List[Minicolumn]],
    right_index: Dict[str, List[Minicolumn]],
    chain_weight: float,
    max_bigrams_per_term: int,
    max_connections_per_bigram: int
) -> WorkerResult:
    """Process a batch of chain connection terms in parallel."""
    result = WorkerResult()
    local_counts: Dict[str, int] = defaultdict(int)

    for term in terms:
        if term not in right_index:
            continue
        if len(left_index[term]) > max_bigrams_per_term or len(right_index[term]) > max_bigrams_per_term:
            continue

        for b_left in right_index[term]:  # ends with term
            if local_counts[b_left.id] >= max_connections_per_bigram:
                continue
            for b_right in left_index[term]:  # starts with term
                if b_left.id == b_right.id:
                    continue
                if local_counts[b_right.id] >= max_connections_per_bigram:
                    continue

                pair = (b_left.id, b_right.id) if b_left.id < b_right.id else (b_right.id, b_left.id)

                if pair in result.pairs:
                    result.pending[b_left.id][b_right.id] += chain_weight
                    result.pending[b_right.id][b_left.id] += chain_weight
                    continue

                result.pairs.add(pair)
                result.pending[b_left.id][b_right.id] += chain_weight
                result.pending[b_right.id][b_left.id] += chain_weight
                local_counts[b_left.id] += 1
                local_counts[b_right.id] += 1
                result.chain_connections += 1

    return result


def _process_cooccurrence_batch(
    doc_items: List[Tuple[str, List[Minicolumn]]],
    cooccurrence_weight: float,
    min_shared_docs: int,
    max_bigrams_per_doc: int,
    max_connections_per_bigram: int,
    importance_threshold: float
) -> WorkerResult:
    """Process a batch of documents for co-occurrence connections in parallel."""
    result = WorkerResult()
    local_counts: Dict[str, int] = defaultdict(int)

    for doc_id, doc_bigrams in doc_items:
        if len(doc_bigrams) > max_bigrams_per_doc:
            result.skipped_large_docs += 1
            continue

        # Filter to important bigrams
        important_bigrams = [b for b in doc_bigrams if b.tfidf >= importance_threshold]
        if len(important_bigrams) < 2:
            continue

        # Sort by importance
        important_bigrams.sort(key=lambda b: b.tfidf, reverse=True)

        for i, b1 in enumerate(important_bigrams):
            if local_counts[b1.id] >= max_connections_per_bigram:
                continue
            for b2 in important_bigrams[i+1:]:
                if local_counts[b2.id] >= max_connections_per_bigram:
                    continue

                # Check shared documents
                shared_docs = b1.document_ids & b2.document_ids
                if len(shared_docs) < min_shared_docs:
                    continue

                jaccard = len(shared_docs) / len(b1.document_ids | b2.document_ids)
                weight = cooccurrence_weight * jaccard

                pair = (b1.id, b2.id) if b1.id < b2.id else (b2.id, b1.id)

                if pair in result.pairs:
                    result.pending[b1.id][b2.id] += weight
                    result.pending[b2.id][b1.id] += weight
                    continue

                result.pairs.add(pair)
                result.pending[b1.id][b2.id] += weight
                result.pending[b2.id][b1.id] += weight
                local_counts[b1.id] += 1
                local_counts[b2.id] += 1
                result.cooccurrence_connections += 1

    return result


def _merge_worker_results(
    results: List[WorkerResult],
    max_connections_per_bigram: int
) -> Tuple[Set[Tuple[str, str]], Dict[str, Dict[str, float]], Dict[str, int]]:
    """
    Merge results from parallel workers with deduplication.

    Returns:
        - merged_pairs: Deduplicated set of connection pairs
        - merged_pending: Combined pending connections
        - stats: Combined statistics dict
    """
    merged_pairs: Set[Tuple[str, str]] = set()
    merged_pending: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    connection_counts: Dict[str, int] = defaultdict(int)

    stats = {
        'component_connections': 0,
        'chain_connections': 0,
        'cooccurrence_connections': 0,
        'skipped_common_terms': 0,
        'skipped_large_docs': 0,
        'skipped_max_connections': 0
    }

    for result in results:
        # Accumulate statistics (some may be overcounted, but it's informational)
        stats['component_connections'] += result.component_connections
        stats['chain_connections'] += result.chain_connections
        stats['cooccurrence_connections'] += result.cooccurrence_connections
        stats['skipped_common_terms'] += result.skipped_common_terms
        stats['skipped_large_docs'] += result.skipped_large_docs

        # Merge pairs with deduplication and connection limit enforcement
        for pair in result.pairs:
            if pair in merged_pairs:
                # Already merged, just accumulate weights
                b1_id, b2_id = pair
                merged_pending[b1_id][b2_id] += result.pending[b1_id].get(b2_id, 0)
                merged_pending[b2_id][b1_id] += result.pending[b2_id].get(b1_id, 0)
            else:
                b1_id, b2_id = pair
                # Check connection limits
                if (connection_counts[b1_id] >= max_connections_per_bigram or
                    connection_counts[b2_id] >= max_connections_per_bigram):
                    stats['skipped_max_connections'] += 1
                    continue

                merged_pairs.add(pair)
                merged_pending[b1_id][b2_id] += result.pending[b1_id].get(b2_id, 0)
                merged_pending[b2_id][b1_id] += result.pending[b2_id].get(b1_id, 0)
                connection_counts[b1_id] += 1
                connection_counts[b2_id] += 1

    return merged_pairs, dict(merged_pending), stats


def _merge_simple_results(
    results: List[SimpleWorkerResult],
    max_connections_per_bigram: int
) -> Tuple[Dict[str, Dict[str, float]], int, Dict[str, int]]:
    """
    Merge results from process-based workers.

    Returns:
        - merged_pending: Combined pending connections {id1: {id2: weight}}
        - connections_created: Count of unique connection pairs
        - stats: Combined statistics dict
    """
    merged_pending: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    connection_counts: Dict[str, int] = defaultdict(int)
    seen_pairs: Set[Tuple[str, str]] = set()

    stats = {
        'component_connections': 0,
        'chain_connections': 0,
        'cooccurrence_connections': 0,
        'skipped_common_terms': 0,
        'skipped_large_docs': 0,
        'skipped_max_connections': 0
    }

    for result in results:
        # Accumulate statistics
        stats['component_connections'] += result.component_connections
        stats['chain_connections'] += result.chain_connections
        stats['cooccurrence_connections'] += result.cooccurrence_connections
        stats['skipped_common_terms'] += result.skipped_common_terms
        stats['skipped_large_docs'] += result.skipped_large_docs

        # Merge connections with deduplication
        for id1, id2, weight in result.connections:
            pair = (id1, id2) if id1 < id2 else (id2, id1)

            if pair in seen_pairs:
                # Already merged, accumulate weight
                merged_pending[id1][id2] += weight
                merged_pending[id2][id1] += weight
            else:
                # Check connection limits
                if (connection_counts[id1] >= max_connections_per_bigram or
                    connection_counts[id2] >= max_connections_per_bigram):
                    stats['skipped_max_connections'] += 1
                    continue

                seen_pairs.add(pair)
                merged_pending[id1][id2] += weight
                merged_pending[id2][id1] += weight
                connection_counts[id1] += 1
                connection_counts[id2] += 1

    return dict(merged_pending), len(seen_pairs), stats


def _compute_bigram_connections_parallel_process(
    layer1: HierarchicalLayer,
    bigrams: List[Minicolumn],
    left_index: Dict[str, List[Minicolumn]],
    right_index: Dict[str, List[Minicolumn]],
    min_shared_docs: int,
    component_weight: float,
    chain_weight: float,
    cooccurrence_weight: float,
    max_bigrams_per_term: int,
    max_bigrams_per_doc: int,
    max_connections_per_bigram: int,
    n_workers: int
) -> Dict[str, Any]:
    """
    Process-based parallel implementation of bigram connection building.

    Uses ProcessPoolExecutor to bypass Python's GIL for true CPU parallelism.
    All data is serialized (pickled) before passing to worker processes.
    """
    all_results: List[SimpleWorkerResult] = []

    def partition_list(items: List, n_parts: int) -> List[List]:
        """Split list into n approximately equal parts."""
        if not items:
            return []
        k, m = divmod(len(items), n_parts)
        return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
                for i in range(n_parts)]

    # =========================================================================
    # STEP 1: Extract serializable data from Minicolumn objects
    # =========================================================================

    # Build ID -> (left_term, right_term) mapping for component processing
    bigram_id_to_data: Dict[str, Tuple[str, str]] = {}
    for bigram in bigrams:
        parts = bigram.content.split(' ')
        if len(parts) == 2:
            bigram_id_to_data[bigram.id] = (parts[0], parts[1])

    # Convert left_index: term -> [Minicolumn] to term -> [bigram_id]
    left_index_simple: Dict[str, List[str]] = {
        term: [b.id for b in blist] for term, blist in left_index.items()
    }
    right_index_simple: Dict[str, List[str]] = {
        term: [b.id for b in blist] for term, blist in right_index.items()
    }

    # Build document index with serializable data
    doc_to_bigrams_simple: Dict[str, List[Tuple[str, frozenset, float]]] = defaultdict(list)
    for bigram in bigrams:
        doc_ids_frozen = frozenset(bigram.document_ids)
        for doc_id in bigram.document_ids:
            doc_to_bigrams_simple[doc_id].append((bigram.id, doc_ids_frozen, bigram.tfidf))

    # Compute importance threshold
    tfidf_values = [b.tfidf for b in bigrams if b.tfidf > 0]
    importance_threshold = sorted(tfidf_values)[len(tfidf_values) // 4] if tfidf_values else 0

    # Prepare batch items
    left_items = [(term, ids) for term, ids in left_index_simple.items()]
    right_items = [(term, ids) for term, ids in right_index_simple.items()]
    chain_terms = list(left_index_simple.keys())
    doc_items = list(doc_to_bigrams_simple.items())

    # =========================================================================
    # STEP 2: Submit work to process pool
    # =========================================================================

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []

        # Submit left component batch jobs
        for batch in partition_list(left_items, n_workers):
            if batch:
                futures.append(executor.submit(
                    _process_component_batch_simple,
                    batch,
                    bigram_id_to_data,
                    component_weight,
                    max_bigrams_per_term,
                    max_connections_per_bigram
                ))

        # Submit right component batch jobs
        for batch in partition_list(right_items, n_workers):
            if batch:
                futures.append(executor.submit(
                    _process_component_batch_simple,
                    batch,
                    bigram_id_to_data,
                    component_weight,
                    max_bigrams_per_term,
                    max_connections_per_bigram
                ))

        # Submit chain batch jobs
        for batch in partition_list(chain_terms, n_workers):
            if batch:
                futures.append(executor.submit(
                    _process_chain_batch_simple,
                    batch,
                    left_index_simple,
                    right_index_simple,
                    chain_weight,
                    max_bigrams_per_term,
                    max_connections_per_bigram
                ))

        # Submit co-occurrence batch jobs
        for batch in partition_list(doc_items, n_workers):
            if batch:
                futures.append(executor.submit(
                    _process_cooccurrence_batch_simple,
                    batch,
                    cooccurrence_weight,
                    min_shared_docs,
                    max_bigrams_per_doc,
                    max_connections_per_bigram,
                    importance_threshold
                ))

        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                import sys
                print(f"Worker error: {e}", file=sys.stderr)

    # =========================================================================
    # STEP 3: Merge results and apply to Minicolumn objects
    # =========================================================================

    merged_pending, connections_created, stats = _merge_simple_results(
        all_results, max_connections_per_bigram
    )

    # Apply all accumulated connections in batch
    for bigram_id, connections in merged_pending.items():
        bigram = layer1.get_by_id(bigram_id)
        if bigram:
            bigram.add_lateral_connections_batch(connections)

    return {
        'connections_created': connections_created,
        'bigrams': len(bigrams),
        'component_connections': stats['component_connections'],
        'chain_connections': stats['chain_connections'],
        'cooccurrence_connections': stats['cooccurrence_connections'],
        'skipped_common_terms': stats['skipped_common_terms'],
        'skipped_large_docs': stats['skipped_large_docs'],
        'skipped_max_connections': stats['skipped_max_connections'],
        'parallel': True,
        'parallel_mode': 'process',
        'n_workers': n_workers
    }


def _compute_bigram_connections_parallel(
    layer1: HierarchicalLayer,
    bigrams: List[Minicolumn],
    left_index: Dict[str, List[Minicolumn]],
    right_index: Dict[str, List[Minicolumn]],
    min_shared_docs: int,
    component_weight: float,
    chain_weight: float,
    cooccurrence_weight: float,
    max_bigrams_per_term: int,
    max_bigrams_per_doc: int,
    max_connections_per_bigram: int,
    n_workers: int
) -> Dict[str, Any]:
    """
    Parallel implementation of bigram connection building.

    DEPRECATED: This uses ThreadPoolExecutor which doesn't help CPU-bound work
    due to Python's GIL. Use _compute_bigram_connections_parallel_process instead.

    Kept for reference and potential I/O-bound variants.
    """
    all_results: List[WorkerResult] = []

    # Partition work into batches for workers
    def partition_list(items: List, n_parts: int) -> List[List]:
        """Split list into n approximately equal parts."""
        if not items:
            return []
        k, m = divmod(len(items), n_parts)
        return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
                for i in range(n_parts)]

    # Build document index for co-occurrence (needed by workers)
    doc_to_bigrams: Dict[str, List[Minicolumn]] = defaultdict(list)
    for bigram in bigrams:
        for doc_id in bigram.document_ids:
            doc_to_bigrams[doc_id].append(bigram)

    # Compute importance threshold
    tfidf_values = [b.tfidf for b in bigrams if b.tfidf > 0]
    importance_threshold = sorted(tfidf_values)[len(tfidf_values) // 4] if tfidf_values else 0

    # Convert indexes to list of items for partitioning
    left_items = list(left_index.items())
    right_items = list(right_index.items())
    chain_terms = list(left_index.keys())
    doc_items = list(doc_to_bigrams.items())

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []

        # Submit left component batch jobs
        for batch in partition_list(left_items, n_workers):
            if batch:
                futures.append(executor.submit(
                    _process_component_batch,
                    batch,
                    component_weight,
                    max_bigrams_per_term,
                    max_connections_per_bigram,
                    'left'
                ))

        # Submit right component batch jobs
        for batch in partition_list(right_items, n_workers):
            if batch:
                futures.append(executor.submit(
                    _process_component_batch,
                    batch,
                    component_weight,
                    max_bigrams_per_term,
                    max_connections_per_bigram,
                    'right'
                ))

        # Submit chain batch jobs
        for batch in partition_list(chain_terms, n_workers):
            if batch:
                futures.append(executor.submit(
                    _process_chain_batch,
                    batch,
                    left_index,
                    right_index,
                    chain_weight,
                    max_bigrams_per_term,
                    max_connections_per_bigram
                ))

        # Submit co-occurrence batch jobs
        for batch in partition_list(doc_items, n_workers):
            if batch:
                futures.append(executor.submit(
                    _process_cooccurrence_batch,
                    batch,
                    cooccurrence_weight,
                    min_shared_docs,
                    max_bigrams_per_doc,
                    max_connections_per_bigram,
                    importance_threshold
                ))

        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                # Log error but continue processing
                import sys
                print(f"Worker error: {e}", file=sys.stderr)

    # Merge all worker results
    merged_pairs, merged_pending, stats = _merge_worker_results(
        all_results, max_connections_per_bigram
    )

    # Apply all accumulated connections in batch
    for bigram_id, connections in merged_pending.items():
        bigram = layer1.get_by_id(bigram_id)
        if bigram:
            bigram.add_lateral_connections_batch(connections)

    return {
        'connections_created': len(merged_pairs),
        'bigrams': len(bigrams),
        'component_connections': stats['component_connections'],
        'chain_connections': stats['chain_connections'],
        'cooccurrence_connections': stats['cooccurrence_connections'],
        'skipped_common_terms': stats['skipped_common_terms'],
        'skipped_large_docs': stats['skipped_large_docs'],
        'skipped_max_connections': stats['skipped_max_connections'],
        'parallel': True,
        'parallel_mode': 'thread',
        'n_workers': n_workers
    }


def compute_bigram_connections(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    min_shared_docs: int = 1,
    component_weight: float = 0.5,
    chain_weight: float = 0.7,
    cooccurrence_weight: float = 0.3,
    max_bigrams_per_term: int = 100,
    max_bigrams_per_doc: int = 500,
    max_connections_per_bigram: int = 50,
    n_workers: Optional[int] = None
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
        n_workers: Number of parallel workers. None or 1 for sequential execution,
            >1 for parallel execution using thread-local accumulation. Default is
            None (sequential).

            NOTE: Due to Python's GIL (Global Interpreter Lock), thread-based
            parallelism does not speed up CPU-bound operations like bigram
            connection building. The parallel option is preserved for:
            - Future use with ProcessPoolExecutor when data becomes serializable
            - I/O-bound variants of the algorithm
            - Documentation of the map-reduce pattern used

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

    # =========================================================================
    # PARALLEL EXECUTION PATH (Shared Memory for minimal overhead)
    # =========================================================================
    if n_workers is not None and n_workers > 1:
        return _compute_bigram_connections_shared_memory(
            layer1=layer1,
            bigrams=bigrams,
            min_shared_docs=min_shared_docs,
            component_weight=component_weight,
            chain_weight=chain_weight,
            cooccurrence_weight=cooccurrence_weight,
            max_bigrams_per_term=max_bigrams_per_term,
            max_bigrams_per_doc=max_bigrams_per_doc,
            max_connections_per_bigram=max_connections_per_bigram,
            n_workers=n_workers
        )

    # =========================================================================
    # SEQUENTIAL EXECUTION PATH (original implementation)
    # =========================================================================

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

        # OPTIMIZATION: Direct comparison is faster than sorted() for 2 items
        pair = (b1.id, b2.id) if b1.id < b2.id else (b2.id, b1.id)
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
            # OPTIMIZATION: Early bailout for bigrams at connection limit
            if connection_counts[b1.id] >= max_connections_per_bigram:
                continue
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
            # OPTIMIZATION: Early bailout for bigrams at connection limit
            if connection_counts[b1.id] >= max_connections_per_bigram:
                continue
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
                # OPTIMIZATION: Early bailout for bigrams at connection limit
                if connection_counts[b_left.id] >= max_connections_per_bigram:
                    continue
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

    OPTIMIZATION (Sprint 8): Instead of O(n²·m) nested loops checking every
    document pair against every token, we iterate tokens once and accumulate
    document pairs. This is O(m·d²) where d = avg docs per token, much faster.

    Args:
        layers: Dictionary of all layers
        documents: Dictionary of documents
        min_shared_terms: Minimum shared terms for connection
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer3 = layers[CorticalLayer.DOCUMENTS]

    # Accumulate shared weights and counts for each document pair
    # Key: (doc1, doc2) tuple where doc1 < doc2 lexicographically
    pair_weights: Dict[Tuple[str, str], float] = defaultdict(float)
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    # Single pass through all tokens - O(m·d²) instead of O(n²·m)
    for token_col in layer0.minicolumns.values():
        doc_list = list(token_col.document_ids)
        if len(doc_list) < 2:
            continue

        weight = token_col.tfidf

        # For each pair of documents sharing this token
        for i, doc1 in enumerate(doc_list):
            for doc2 in doc_list[i+1:]:
                # Canonical ordering for consistent keys
                key = (doc1, doc2) if doc1 < doc2 else (doc2, doc1)
                pair_weights[key] += weight
                pair_counts[key] += 1

    # Create connections for pairs meeting threshold
    for (doc1, doc2), count in pair_counts.items():
        if count >= min_shared_terms:
            weight = pair_weights[(doc1, doc2)]

            col1 = layer3.get_minicolumn(doc1)
            if not col1:
                col1 = layer3.get_or_create_minicolumn(doc1)

            col2 = layer3.get_minicolumn(doc2)
            if not col2:
                col2 = layer3.get_or_create_minicolumn(doc2)

            col1.add_lateral_connection(col2.id, weight)
            col2.add_lateral_connection(col1.id, weight)
