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
from typing import Dict, List, Tuple, Set, Optional
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
) -> Dict[str, any]:
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
) -> Dict[str, any]:
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
