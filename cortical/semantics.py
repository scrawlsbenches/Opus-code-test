"""
Semantics Module
================

Corpus-derived semantic relations and retrofitting.

Extracts semantic relationships from co-occurrence patterns,
then uses them to adjust connection weights (retrofitting).
This is like building a "poor man's ConceptNet" from the corpus itself.
"""

import math
import re
from typing import Any, Dict, List, Tuple, Set, Optional
import copy
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .layers import CorticalLayer, HierarchicalLayer
from .minicolumn import Minicolumn
from .constants import RELATION_WEIGHTS
from .tokenizer import Tokenizer


# Commonsense relation patterns with confidence scores
# Format: (pattern_regex, relation_type, confidence, swap_order)
# swap_order: if True, the captured groups are in reverse order (t2, t1)
RELATION_PATTERNS = [
    # IsA patterns (hypernym/type relations)
    (r'(\w+)\s+(?:is|are)\s+(?:a|an)\s+(?:type\s+of\s+)?(\w+)', 'IsA', 0.9, False),
    (r'(\w+),?\s+(?:a|an)\s+(?:kind|type|form)\s+of\s+(\w+)', 'IsA', 0.95, False),
    (r'(\w+)\s+(?:is|are)\s+considered\s+(?:a|an)?\s*(\w+)', 'IsA', 0.8, False),
    (r'(?:a|an)\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)', 'IsA', 0.85, False),
    (r'(\w+)\s+(?:belongs?\s+to|falls?\s+under)\s+(?:the\s+)?(\w+)', 'IsA', 0.8, False),

    # HasA/Contains patterns (meronym relations)
    (r'(\w+)\s+(?:has|have|contains?|includes?)\s+(?:a|an|the)?\s*(\w+)', 'HasA', 0.85, False),
    (r'(\w+)\s+(?:consists?\s+of|comprises?|is\s+made\s+of)\s+(\w+)', 'HasA', 0.9, False),
    (r'(?:a|an|the)\s+(\w+)\s+(?:with|having)\s+(?:a|an|the)?\s*(\w+)', 'HasA', 0.75, False),

    # PartOf patterns (part-whole relations)
    (r'(\w+)\s+(?:is|are)\s+(?:a\s+)?part\s+of\s+(?:a|an|the)?\s*(\w+)', 'PartOf', 0.95, False),
    (r'(\w+)\s+(?:is|are)\s+(?:a\s+)?component\s+of\s+(\w+)', 'PartOf', 0.9, False),
    (r'(\w+)\s+(?:is|are)\s+(?:in|within|inside)\s+(?:a|an|the)?\s*(\w+)', 'PartOf', 0.7, False),

    # UsedFor patterns (functional relations)
    (r'(\w+)\s+(?:is|are)\s+used\s+(?:for|to|in)\s+(\w+)', 'UsedFor', 0.9, False),
    (r'(\w+)\s+(?:helps?|enables?|allows?)\s+(\w+)', 'UsedFor', 0.75, False),
    (r'(?:use|using)\s+(\w+)\s+(?:for|to)\s+(\w+)', 'UsedFor', 0.85, False),
    (r'(\w+)\s+(?:is|are)\s+(?:useful|helpful)\s+for\s+(\w+)', 'UsedFor', 0.8, False),

    # Causes patterns (causal relations)
    (r'(\w+)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(\w+)', 'Causes', 0.9, False),
    (r'(\w+)\s+(?:produces?|generates?|creates?)\s+(\w+)', 'Causes', 0.8, False),
    (r'(\w+)\s+(?:can\s+)?(?:cause|lead\s+to|result\s+in)\s+(\w+)', 'Causes', 0.85, False),
    (r'(?:because\s+of|due\s+to)\s+(\w+),?\s+(\w+)', 'Causes', 0.7, True),  # Reversed order

    # CapableOf patterns (ability relations)
    (r'(\w+)\s+(?:can|could|is\s+able\s+to)\s+(\w+)', 'CapableOf', 0.85, False),
    (r'(\w+)\s+(?:has\s+the\s+ability\s+to|is\s+capable\s+of)\s+(\w+)', 'CapableOf', 0.9, False),

    # AtLocation patterns (spatial relations)
    (r'(\w+)\s+(?:is|are)\s+(?:found|located|situated)\s+(?:in|at|on)\s+(\w+)', 'AtLocation', 0.9, False),
    (r'(\w+)\s+(?:lives?|exists?|occurs?)\s+(?:in|at|on)\s+(\w+)', 'AtLocation', 0.85, False),

    # HasProperty patterns (attribute relations)
    (r'(\w+)\s+(?:is|are)\s+(\w+)', 'HasProperty', 0.5, False),  # Very general, low confidence
    (r'(\w+)\s+(?:is|are)\s+(?:typically|usually|often|generally)\s+(\w+)', 'HasProperty', 0.7, False),
    (r'(?:a|an)\s+(\w+)\s+(\w+)\s+(?:is|are)', 'HasProperty', 0.6, True),  # "a big dog" → dog HasProperty big

    # Antonym patterns (opposite relations)
    (r'(\w+)\s+(?:is|are)\s+(?:the\s+)?opposite\s+of\s+(\w+)', 'Antonym', 0.95, False),
    (r'(\w+)\s+(?:vs\.?|versus|or)\s+(\w+)', 'Antonym', 0.5, False),  # Lower confidence
    (r'(\w+)\s+(?:not|isn\'t|aren\'t)\s+(\w+)', 'Antonym', 0.6, False),

    # DerivedFrom patterns (morphological/etymological relations)
    (r'(\w+)\s+(?:comes?\s+from|is\s+derived\s+from|originates?\s+from)\s+(\w+)', 'DerivedFrom', 0.9, False),
    (r'(\w+)\s+(?:is\s+based\s+on|stems?\s+from)\s+(\w+)', 'DerivedFrom', 0.85, False),

    # DefinedBy patterns (definitional relations)
    (r'(\w+)\s+(?:means?|refers?\s+to|denotes?)\s+(\w+)', 'DefinedBy', 0.85, False),
    (r'(\w+)\s+(?:is\s+defined\s+as|is\s+known\s+as)\s+(?:a|an|the)?\s*(\w+)', 'DefinedBy', 0.9, False),
]


def extract_pattern_relations(
    documents: Dict[str, str],
    valid_terms: Set[str],
    min_confidence: float = 0.5
) -> List[Tuple[str, str, str, float]]:
    """
    Extract semantic relations using pattern matching on document text.

    Uses regex patterns to identify commonsense relations like IsA, HasA,
    UsedFor, Causes, etc. from natural language expressions.

    Args:
        documents: Dictionary mapping doc_id to document content
        valid_terms: Set of terms that exist in the corpus (from layer0)
        min_confidence: Minimum confidence threshold for extracted relations

    Returns:
        List of (term1, relation_type, term2, confidence) tuples

    Example:
        >>> relations = extract_pattern_relations(docs, {"dog", "animal", "pet"})
        >>> # Finds relations like ("dog", "IsA", "animal", 0.9)
    """
    relations: List[Tuple[str, str, str, float]] = []
    seen_relations: Set[Tuple[str, str, str]] = set()

    for doc_id, content in documents.items():
        content_lower = content.lower()

        for pattern, relation_type, confidence, swap_order in RELATION_PATTERNS:
            if confidence < min_confidence:
                continue

            for match in re.finditer(pattern, content_lower):
                groups = match.groups()
                if len(groups) >= 2:
                    t1, t2 = groups[0], groups[1]

                    if swap_order:
                        t1, t2 = t2, t1

                    # Clean terms (remove leading/trailing non-alphanumeric)
                    t1 = t1.strip().lower()
                    t2 = t2.strip().lower()

                    # Skip if terms are the same
                    if t1 == t2:
                        continue

                    # Skip if terms don't exist in corpus
                    if t1 not in valid_terms or t2 not in valid_terms:
                        continue

                    # Skip common stopwords that might slip through patterns
                    if t1 in Tokenizer.DEFAULT_STOP_WORDS or t2 in Tokenizer.DEFAULT_STOP_WORDS:
                        continue

                    # Create relation key to avoid duplicates
                    rel_key = (t1, relation_type, t2)

                    # For symmetric relations, also check reverse
                    if relation_type in {'SimilarTo', 'Antonym', 'RelatedTo'}:
                        rev_key = (t2, relation_type, t1)
                        if rev_key in seen_relations:
                            continue

                    if rel_key not in seen_relations:
                        seen_relations.add(rel_key)
                        relations.append((t1, relation_type, t2, confidence))

    return relations


def get_pattern_statistics(relations: List[Tuple[str, str, str, float]]) -> Dict[str, Any]:
    """
    Get statistics about extracted pattern-based relations.

    Args:
        relations: List of (term1, relation_type, term2, confidence) tuples

    Returns:
        Dictionary with statistics about relation types and counts
    """
    type_counts: Dict[str, int] = defaultdict(int)
    type_confidences: Dict[str, List[float]] = defaultdict(list)

    for t1, rel_type, t2, conf in relations:
        type_counts[rel_type] += 1
        type_confidences[rel_type].append(conf)

    # Compute average confidence per type
    avg_confidences = {
        rel_type: sum(confs) / len(confs)
        for rel_type, confs in type_confidences.items()
    }

    return {
        'total_relations': len(relations),
        'relation_type_counts': dict(type_counts),
        'average_confidence_by_type': avg_confidences,
        'unique_types': len(type_counts)
    }


def extract_corpus_semantics(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str],
    tokenizer,
    window_size: int = 5,
    min_cooccurrence: int = 2,
    use_pattern_extraction: bool = True,
    min_pattern_confidence: float = 0.6,
    max_similarity_pairs: int = 100000,
    min_context_keys: int = 3
) -> List[Tuple[str, str, str, float]]:
    """
    Extract semantic relations from corpus co-occurrence patterns.

    Analyzes word co-occurrences to infer semantic relationships:
    - Words appearing together frequently → CoOccurs
    - Words appearing in similar contexts → SimilarTo
    - Pattern-based extraction → IsA, HasA, UsedFor, Causes, etc.

    Args:
        layers: Dictionary of layers (needs TOKENS)
        documents: Dictionary of documents
        tokenizer: Tokenizer instance for processing text
        window_size: Co-occurrence window size
        min_cooccurrence: Minimum co-occurrences to form relation
        use_pattern_extraction: Whether to extract relations from text patterns
        min_pattern_confidence: Minimum confidence for pattern-based extraction
        max_similarity_pairs: Maximum pairs to check for SimilarTo relations.
            Set to 0 for unlimited (may be slow for large corpora). Default 100000.
        min_context_keys: Minimum context keys for a term to be considered for
            SimilarTo relations. Terms with fewer keys are skipped. Default 3.

    Returns:
        List of (term1, relation, term2, weight) tuples
    """
    layer0 = layers[CorticalLayer.TOKENS]
    relations: List[Tuple[str, str, str, float]] = []
    
    # Track co-occurrences within window
    cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
    
    # Track context vectors for similarity
    context_vectors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    for doc_id, content in documents.items():
        tokens = tokenizer.tokenize(content)
        
        # Window-based co-occurrence
        for i, token in enumerate(tokens):
            window_start = max(0, i - window_size)
            window_end = min(len(tokens), i + window_size + 1)
            
            for j in range(window_start, window_end):
                if i != j:
                    other = tokens[j]
                    if token < other:  # Avoid duplicates
                        cooccurrence[(token, other)] += 1
                    else:
                        cooccurrence[(other, token)] += 1
                    
                    # Build context vector
                    context_vectors[token][other] += 1
    
    # Extract RelatedTo from co-occurrence
    # Compute total once outside the loop (was being computed per iteration!)
    total = sum(cooccurrence.values())

    for (t1, t2), count in cooccurrence.items():
        if count >= min_cooccurrence:
            # Normalize by frequency
            col1 = layer0.get_minicolumn(t1)
            col2 = layer0.get_minicolumn(t2)

            if col1 and col2:
                # PMI-like score
                expected = (col1.occurrence_count * col2.occurrence_count) / (total + 1)
                pmi = math.log((count + 1) / (expected + 1))

                if pmi > 0:
                    relations.append((t1, 'CoOccurs', t2, min(pmi, 3.0)))
    
    # Extract SimilarTo from context similarity
    terms = list(context_vectors.keys())
    n_terms = len(terms)

    if n_terms > 1 and HAS_NUMPY:
        # Fast path: use numpy vectorization
        # Build vocabulary of all context keys
        all_keys: Set[str] = set()
        for vec in context_vectors.values():
            all_keys.update(vec.keys())
        vocab = sorted(all_keys)
        key_to_idx = {k: i for i, k in enumerate(vocab)}
        n_vocab = len(vocab)

        # Convert sparse vectors to dense numpy matrix
        matrix = np.zeros((n_terms, n_vocab), dtype=np.float32)
        for i, term in enumerate(terms):
            vec = context_vectors[term]
            for k, v in vec.items():
                matrix[i, key_to_idx[k]] = v

        # Normalize rows for cosine similarity (dot product of normalized = cosine)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        matrix_norm = matrix / norms

        # Compute all pairwise cosine similarities via matrix multiplication
        similarities = matrix_norm @ matrix_norm.T

        # Count non-zero elements per row (for min common keys filter)
        nonzero_counts = (matrix > 0).astype(np.int32)

        # Extract pairs with similarity > 0.3 and at least 3 common keys
        for i in range(n_terms):
            row_i = nonzero_counts[i]
            for j in range(i + 1, n_terms):
                if similarities[i, j] > 0.3:
                    common_count = np.sum(row_i & nonzero_counts[j])
                    if common_count >= 3:
                        relations.append((terms[i], 'SimilarTo', terms[j], float(similarities[i, j])))

    elif n_terms > 1:
        # Fallback: pure Python implementation with optimizations
        # Pre-filter terms by minimum context keys
        key_sets: Dict[str, set] = {}
        magnitudes: Dict[str, float] = {}

        for term in terms:
            vec = context_vectors[term]
            keys = set(vec.keys())
            # Skip terms with too few context keys (can't meet min_context_keys threshold)
            if len(keys) < min_context_keys:
                continue
            key_sets[term] = keys
            mag = math.sqrt(sum(v * v for v in vec.values()))
            magnitudes[term] = mag

        # Get filtered terms with enough context
        filtered_terms = [t for t in terms if t in key_sets and magnitudes.get(t, 0) > 0]

        # Track pairs checked for early termination
        pairs_checked = 0

        for i, t1 in enumerate(filtered_terms):
            vec1 = context_vectors[t1]
            mag1 = magnitudes[t1]
            keys1 = key_sets[t1]

            for t2 in filtered_terms[i+1:]:
                # Check pair limit
                if max_similarity_pairs > 0 and pairs_checked >= max_similarity_pairs:
                    break

                pairs_checked += 1
                mag2 = magnitudes[t2]

                common = keys1 & key_sets[t2]
                if len(common) >= min_context_keys:
                    vec2 = context_vectors[t2]
                    dot = sum(vec1[k] * vec2[k] for k in common)
                    sim = dot / (mag1 * mag2)
                    if sim > 0.3:
                        relations.append((t1, 'SimilarTo', t2, sim))

            # Also check outer loop for pair limit
            if max_similarity_pairs > 0 and pairs_checked >= max_similarity_pairs:
                break

    # Extract commonsense relations from text patterns
    if use_pattern_extraction:
        valid_terms = set(layer0.minicolumns.keys())
        pattern_relations = extract_pattern_relations(
            documents,
            valid_terms,
            min_confidence=min_pattern_confidence
        )
        relations.extend(pattern_relations)

    return relations


def retrofit_connections(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    semantic_relations: List[Tuple[str, str, str, float]],
    iterations: int = 10,
    alpha: float = 0.3
) -> Dict[str, Any]:
    """
    Retrofit lateral connections using semantic relations.
    
    Adjusts connection weights by blending co-occurrence patterns
    with semantic relations. This is inspired by Faruqui et al.'s
    retrofitting algorithm for word vectors.
    
    Args:
        layers: Dictionary of layers
        semantic_relations: List of (term1, relation, term2, weight) tuples
        iterations: Number of retrofitting iterations
        alpha: Blend factor (0=all semantic, 1=all original)

    Returns:
        Dictionary with retrofitting statistics

    Raises:
        ValueError: If alpha is not in range [0, 1]
    """
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    layer0 = layers[CorticalLayer.TOKENS]

    # Store original weights
    original_weights: Dict[str, Dict[str, float]] = {}
    for col in layer0.minicolumns.values():
        original_weights[col.content] = dict(col.lateral_connections)
    
    # Build semantic neighbor lookup
    semantic_neighbors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    
    for t1, relation, t2, weight in semantic_relations:
        relation_weight = RELATION_WEIGHTS.get(relation, 0.5)
        combined_weight = weight * relation_weight
        
        # Bidirectional
        semantic_neighbors[t1].append((t2, combined_weight))
        semantic_neighbors[t2].append((t1, combined_weight))
    
    # Iterative retrofitting
    tokens_affected = set()
    total_adjustment = 0.0
    
    for iteration in range(iterations):
        iteration_adjustment = 0.0
        
        for col in layer0.minicolumns.values():
            term = col.content
            
            if term not in semantic_neighbors:
                continue
            
            tokens_affected.add(term)
            
            # Get semantic target weights
            semantic_targets: Dict[str, float] = {}
            for neighbor, weight in semantic_neighbors[term]:
                neighbor_col = layer0.get_minicolumn(neighbor)
                if neighbor_col:
                    semantic_targets[neighbor_col.id] = weight
            
            if not semantic_targets:
                continue
            
            # Adjust each connection
            for target_id in list(col.lateral_connections.keys()):
                original = original_weights[term].get(target_id, 0)
                semantic = semantic_targets.get(target_id, 0)

                # Blend original and semantic
                new_weight = alpha * original + (1 - alpha) * semantic

                if new_weight > 0:
                    adjustment = abs(col.lateral_connections[target_id] - new_weight)
                    iteration_adjustment += adjustment
                    col.set_lateral_connection_weight(target_id, new_weight)

            # Add new semantic connections
            for target_id, semantic_weight in semantic_targets.items():
                if target_id not in col.lateral_connections:
                    col.set_lateral_connection_weight(target_id, (1 - alpha) * semantic_weight)
                    iteration_adjustment += (1 - alpha) * semantic_weight
        
        total_adjustment += iteration_adjustment
    
    return {
        'iterations': iterations,
        'alpha': alpha,
        'tokens_affected': len(tokens_affected),
        'total_adjustment': total_adjustment,
        'relations_used': len(semantic_relations)
    }


def retrofit_embeddings(
    embeddings: Dict[str, List[float]],
    semantic_relations: List[Tuple[str, str, str, float]],
    iterations: int = 10,
    alpha: float = 0.4
) -> Dict[str, Any]:
    """
    Retrofit embeddings using semantic relations.
    
    Like Faruqui et al.'s retrofitting, but for graph embeddings.
    Pulls semantically related terms closer in embedding space.
    
    Args:
        embeddings: Dictionary mapping terms to embedding vectors
        semantic_relations: List of (term1, relation, term2, weight) tuples
        iterations: Number of iterations
        alpha: Blend factor (higher = more original embedding)

    Returns:
        Dictionary with retrofitting statistics

    Raises:
        ValueError: If alpha is not in range [0, 1]
    """
    if not (0 <= alpha <= 1):
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

    # Store original embeddings
    original = copy.deepcopy(embeddings)
    
    # Build neighbor lookup
    neighbors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    
    for t1, relation, t2, weight in semantic_relations:
        if t1 in embeddings and t2 in embeddings:
            relation_weight = RELATION_WEIGHTS.get(relation, 0.5)
            combined = weight * relation_weight
            neighbors[t1].append((t2, combined))
            neighbors[t2].append((t1, combined))
    
    # Iterative retrofitting
    total_movement = 0.0
    terms_moved = set()
    
    for iteration in range(iterations):
        for term in list(embeddings.keys()):
            if term not in neighbors or not neighbors[term]:
                continue
            
            terms_moved.add(term)
            vec = embeddings[term]
            orig = original[term]
            
            # Compute semantic center (weighted average of neighbors)
            semantic_center = [0.0] * len(vec)
            total_weight = 0.0
            
            for neighbor, weight in neighbors[term]:
                if neighbor in embeddings:
                    neighbor_vec = embeddings[neighbor]
                    for i in range(len(vec)):
                        semantic_center[i] += neighbor_vec[i] * weight
                    total_weight += weight
            
            if total_weight > 0:
                for i in range(len(semantic_center)):
                    semantic_center[i] /= total_weight
                
                # Blend original with semantic center
                new_vec = []
                movement = 0.0
                
                for i in range(len(vec)):
                    new_val = alpha * orig[i] + (1 - alpha) * semantic_center[i]
                    movement += abs(new_val - vec[i])
                    new_vec.append(new_val)
                
                embeddings[term] = new_vec
                total_movement += movement
    
    return {
        'iterations': iterations,
        'alpha': alpha,
        'terms_retrofitted': len(terms_moved),
        'total_movement': total_movement
    }


def get_relation_type_weight(relation_type: str) -> float:
    """
    Get the weight for a relation type.

    Args:
        relation_type: Type of semantic relation

    Returns:
        Weight multiplier for this relation type
    """
    return RELATION_WEIGHTS.get(relation_type, 0.5)


def build_isa_hierarchy(
    semantic_relations: List[Tuple[str, str, str, float]]
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Build IsA parent-child hierarchy from semantic relations.

    Extracts all IsA relations and builds bidirectional parent-child mappings.
    For example, if "dog IsA animal", then:
    - parents["dog"] = {"animal"}
    - children["animal"] = {"dog"}

    Args:
        semantic_relations: List of (term1, relation, term2, weight) tuples

    Returns:
        Tuple of (parents, children) dicts:
        - parents: Maps term to set of parent terms (hypernyms)
        - children: Maps term to set of child terms (hyponyms)

    Example:
        >>> relations = [("dog", "IsA", "animal", 1.0), ("cat", "IsA", "animal", 1.0)]
        >>> parents, children = build_isa_hierarchy(relations)
        >>> print(parents["dog"])  # {"animal"}
        >>> print(children["animal"])  # {"dog", "cat"}
    """
    parents: Dict[str, Set[str]] = defaultdict(set)
    children: Dict[str, Set[str]] = defaultdict(set)

    for t1, relation, t2, weight in semantic_relations:
        if relation == 'IsA':
            # t1 IsA t2 means t2 is a parent (hypernym) of t1
            parents[t1].add(t2)
            children[t2].add(t1)

    return dict(parents), dict(children)


def get_ancestors(
    term: str,
    parents: Dict[str, Set[str]],
    max_depth: int = 10
) -> Dict[str, int]:
    """
    Get all ancestors of a term with their depth in the hierarchy.

    Performs BFS traversal up the IsA hierarchy to find all ancestors.

    Args:
        term: Starting term
        parents: Parent mapping from build_isa_hierarchy()
        max_depth: Maximum depth to traverse (prevents infinite loops)

    Returns:
        Dict mapping ancestor terms to their depth (1 = direct parent, 2 = grandparent, etc.)

    Example:
        >>> # If dog IsA canine IsA animal
        >>> ancestors = get_ancestors("dog", parents)
        >>> # ancestors = {"canine": 1, "animal": 2}
    """
    ancestors: Dict[str, int] = {}
    frontier = [(p, 1) for p in parents.get(term, set())]
    visited = {term}

    while frontier:
        current, depth = frontier.pop(0)
        if current in visited or depth > max_depth:
            continue
        visited.add(current)
        ancestors[current] = depth

        # Add parents of current term
        for parent in parents.get(current, set()):
            if parent not in visited:
                frontier.append((parent, depth + 1))

    return ancestors


def get_descendants(
    term: str,
    children: Dict[str, Set[str]],
    max_depth: int = 10
) -> Dict[str, int]:
    """
    Get all descendants of a term with their depth in the hierarchy.

    Performs BFS traversal down the IsA hierarchy to find all descendants.

    Args:
        term: Starting term
        children: Children mapping from build_isa_hierarchy()
        max_depth: Maximum depth to traverse (prevents infinite loops)

    Returns:
        Dict mapping descendant terms to their depth (1 = direct child, 2 = grandchild, etc.)
    """
    descendants: Dict[str, int] = {}
    frontier = [(c, 1) for c in children.get(term, set())]
    visited = {term}

    while frontier:
        current, depth = frontier.pop(0)
        if current in visited or depth > max_depth:
            continue
        visited.add(current)
        descendants[current] = depth

        # Add children of current term
        for child in children.get(current, set()):
            if child not in visited:
                frontier.append((child, depth + 1))

    return descendants


def inherit_properties(
    semantic_relations: List[Tuple[str, str, str, float]],
    decay_factor: float = 0.7,
    max_depth: int = 5
) -> Dict[str, Dict[str, Tuple[float, str, int]]]:
    """
    Compute inherited properties for all terms based on IsA hierarchy.

    If "dog IsA animal" and "animal HasProperty living", then "dog" inherits
    "living" with a decayed weight. Properties propagate down the IsA hierarchy
    with weight decaying at each level.

    Args:
        semantic_relations: List of (term1, relation, term2, weight) tuples
        decay_factor: Weight multiplier per inheritance level (default 0.7)
        max_depth: Maximum inheritance depth (default 5)

    Returns:
        Dict mapping terms to their inherited properties:
        {
            term: {
                property: (weight, source_ancestor, depth)
            }
        }

    Example:
        >>> relations = [
        ...     ("dog", "IsA", "animal", 1.0),
        ...     ("animal", "HasProperty", "living", 0.9),
        ...     ("animal", "HasProperty", "mortal", 0.8),
        ... ]
        >>> inherited = inherit_properties(relations)
        >>> print(inherited["dog"])
        >>> # {"living": (0.63, "animal", 1), "mortal": (0.56, "animal", 1)}
    """
    # Build hierarchy
    parents, children = build_isa_hierarchy(semantic_relations)

    # Extract direct properties for each term
    # Properties come from HasProperty, HasA, CapableOf, etc.
    property_relations = {'HasProperty', 'HasA', 'CapableOf', 'AtLocation', 'UsedFor'}
    direct_properties: Dict[str, Dict[str, float]] = defaultdict(dict)

    for t1, relation, t2, weight in semantic_relations:
        if relation in property_relations:
            # t1 HasProperty t2 means t2 is a property of t1
            direct_properties[t1][t2] = max(direct_properties[t1].get(t2, 0), weight)

    # Compute inherited properties for each term
    inherited: Dict[str, Dict[str, Tuple[float, str, int]]] = {}

    # Get all terms that have parents (i.e., can inherit)
    all_terms = set(parents.keys())
    # Also include terms with direct properties (they might be ancestors)
    all_terms.update(direct_properties.keys())

    for term in all_terms:
        term_inherited: Dict[str, Tuple[float, str, int]] = {}

        # Get all ancestors and their depths
        ancestors = get_ancestors(term, parents, max_depth=max_depth)

        # For each ancestor, inherit their properties
        for ancestor, depth in ancestors.items():
            if ancestor in direct_properties:
                # Compute decayed weight
                decay = decay_factor ** depth
                for prop, prop_weight in direct_properties[ancestor].items():
                    inherited_weight = prop_weight * decay

                    # Keep the strongest inheritance path
                    if prop not in term_inherited or term_inherited[prop][0] < inherited_weight:
                        term_inherited[prop] = (inherited_weight, ancestor, depth)

        if term_inherited:
            inherited[term] = term_inherited

    return inherited


def compute_property_similarity(
    term1: str,
    term2: str,
    inherited_properties: Dict[str, Dict[str, Tuple[float, str, int]]],
    direct_properties: Optional[Dict[str, Dict[str, float]]] = None
) -> float:
    """
    Compute similarity between terms based on shared properties (direct + inherited).

    Args:
        term1: First term
        term2: Second term
        inherited_properties: Output from inherit_properties()
        direct_properties: Optional dict of direct properties {term: {prop: weight}}

    Returns:
        Similarity score based on Jaccard-like overlap of properties

    Example:
        >>> sim = compute_property_similarity("dog", "cat", inherited, direct)
        >>> # Both inherit "living" from "animal", so similarity > 0
    """
    # Get all properties for each term
    props1: Dict[str, float] = {}
    props2: Dict[str, float] = {}

    # Add inherited properties
    if term1 in inherited_properties:
        for prop, (weight, _, _) in inherited_properties[term1].items():
            props1[prop] = max(props1.get(prop, 0), weight)

    if term2 in inherited_properties:
        for prop, (weight, _, _) in inherited_properties[term2].items():
            props2[prop] = max(props2.get(prop, 0), weight)

    # Add direct properties if provided
    if direct_properties:
        if term1 in direct_properties:
            for prop, weight in direct_properties[term1].items():
                props1[prop] = max(props1.get(prop, 0), weight)
        if term2 in direct_properties:
            for prop, weight in direct_properties[term2].items():
                props2[prop] = max(props2.get(prop, 0), weight)

    if not props1 or not props2:
        return 0.0

    # Compute weighted Jaccard similarity
    common_props = set(props1.keys()) & set(props2.keys())
    all_props = set(props1.keys()) | set(props2.keys())

    if not all_props:
        return 0.0

    # Sum of minimum weights for common properties
    intersection_weight = sum(
        min(props1[p], props2[p]) for p in common_props
    )

    # Sum of maximum weights for all properties
    union_weight = sum(
        max(props1.get(p, 0), props2.get(p, 0)) for p in all_props
    )

    return intersection_weight / union_weight if union_weight > 0 else 0.0


def apply_inheritance_to_connections(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    inherited_properties: Dict[str, Dict[str, Tuple[float, str, int]]],
    boost_factor: float = 0.3
) -> Dict[str, Any]:
    """
    Boost lateral connections between terms that share inherited properties.

    Terms that share properties through inheritance should have stronger
    connections, even if they don't directly co-occur.

    Args:
        layers: Dictionary of layers
        inherited_properties: Output from inherit_properties()
        boost_factor: Weight boost for shared properties (default 0.3)

    Returns:
        Statistics about connections boosted

    Example:
        >>> # "dog" and "cat" both inherit "living" from "animal"
        >>> # Their lateral connection gets boosted
        >>> stats = apply_inheritance_to_connections(layers, inherited)
    """
    layer0 = layers[CorticalLayer.TOKENS]
    connections_boosted = 0
    total_boost = 0.0

    # Get terms that have inherited properties
    terms_with_inheritance = set(inherited_properties.keys())

    # For each pair of terms with inherited properties
    terms_list = list(terms_with_inheritance)

    for i, term1 in enumerate(terms_list):
        col1 = layer0.get_minicolumn(term1)
        if not col1:
            continue

        props1 = inherited_properties[term1]

        for term2 in terms_list[i + 1:]:
            col2 = layer0.get_minicolumn(term2)
            if not col2:
                continue

            props2 = inherited_properties[term2]

            # Find shared inherited properties
            shared_props = set(props1.keys()) & set(props2.keys())
            if not shared_props:
                continue

            # Compute boost based on shared properties
            boost = 0.0
            for prop in shared_props:
                w1, _, _ = props1[prop]
                w2, _, _ = props2[prop]
                # Average of the two inheritance weights
                boost += (w1 + w2) / 2 * boost_factor

            if boost > 0:
                # Add boost to lateral connections
                col1.add_lateral_connection(col2.id, boost)
                col2.add_lateral_connection(col1.id, boost)
                connections_boosted += 1
                total_boost += boost

    return {
        'connections_boosted': connections_boosted,
        'total_boost': total_boost,
        'terms_with_inheritance': len(terms_with_inheritance)
    }
