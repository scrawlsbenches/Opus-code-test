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
from collections import defaultdict

from .layers import CorticalLayer, HierarchicalLayer
from .minicolumn import Minicolumn


# Relation type weights for retrofitting
RELATION_WEIGHTS = {
    'IsA': 1.5,
    'PartOf': 1.2,
    'HasA': 1.0,
    'UsedFor': 0.8,
    'CapableOf': 0.7,
    'AtLocation': 0.6,
    'Causes': 0.9,
    'HasProperty': 0.8,
    'SameAs': 2.0,
    'RelatedTo': 0.5,
    'Antonym': -0.5,
    'DerivedFrom': 1.0,
    'SimilarTo': 1.5,
    'CoOccurs': 0.6,
    'DefinedBy': 1.0,
}


def extract_corpus_semantics(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str],
    tokenizer,
    window_size: int = 5,
    min_cooccurrence: int = 2
) -> List[Tuple[str, str, str, float]]:
    """
    Extract semantic relations from corpus co-occurrence patterns.
    
    Analyzes word co-occurrences to infer semantic relationships:
    - Words appearing together frequently → RelatedTo
    - Words appearing in similar contexts → SimilarTo
    - Words in definitional patterns → IsA, DefinedBy
    
    Args:
        layers: Dictionary of layers (needs TOKENS)
        documents: Dictionary of documents
        tokenizer: Tokenizer instance for processing text
        window_size: Co-occurrence window size
        min_cooccurrence: Minimum co-occurrences to form relation
        
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
    for (t1, t2), count in cooccurrence.items():
        if count >= min_cooccurrence:
            # Normalize by frequency
            col1 = layer0.get_minicolumn(t1)
            col2 = layer0.get_minicolumn(t2)
            
            if col1 and col2:
                # PMI-like score
                total = sum(cooccurrence.values())
                expected = (col1.occurrence_count * col2.occurrence_count) / (total + 1)
                pmi = math.log((count + 1) / (expected + 1))
                
                if pmi > 0:
                    relations.append((t1, 'CoOccurs', t2, min(pmi, 3.0)))
    
    # Extract SimilarTo from context similarity
    terms = list(context_vectors.keys())
    for i, t1 in enumerate(terms):
        vec1 = context_vectors[t1]
        
        for t2 in terms[i+1:]:
            vec2 = context_vectors[t2]
            
            # Cosine similarity of context vectors
            common = set(vec1.keys()) & set(vec2.keys())
            if len(common) >= 3:
                dot = sum(vec1[k] * vec2[k] for k in common)
                mag1 = math.sqrt(sum(v*v for v in vec1.values()))
                mag2 = math.sqrt(sum(v*v for v in vec2.values()))
                
                if mag1 > 0 and mag2 > 0:
                    sim = dot / (mag1 * mag2)
                    if sim > 0.3:
                        relations.append((t1, 'SimilarTo', t2, sim))
    
    # Extract IsA from definitional patterns
    isa_patterns = [
        r'(\w+)\s+(?:is|are)\s+(?:a|an)\s+(?:type\s+of\s+)?(\w+)',
        r'(\w+),?\s+(?:a|an)\s+(?:kind|type)\s+of\s+(\w+)',
        r'(\w+)\s+(?:such\s+as|like)\s+(\w+)',
    ]
    
    for doc_id, content in documents.items():
        content_lower = content.lower()
        for pattern in isa_patterns:
            for match in re.finditer(pattern, content_lower):
                t1, t2 = match.groups()
                if t1 in layer0.minicolumns and t2 in layer0.minicolumns:
                    relations.append((t1, 'IsA', t2, 1.0))
    
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
    """
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
                    col.lateral_connections[target_id] = new_weight
            
            # Add new semantic connections
            for target_id, semantic_weight in semantic_targets.items():
                if target_id not in col.lateral_connections:
                    col.lateral_connections[target_id] = (1 - alpha) * semantic_weight
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
    """
    import copy
    
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
