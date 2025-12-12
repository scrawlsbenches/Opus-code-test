"""
Centralized constants for the Cortical Text Processor.

This module provides a single source of truth for constants used across
multiple modules, preventing drift and inconsistencies.

Task #96: Centralize duplicate constants
"""

from typing import Dict, FrozenSet

# =============================================================================
# RELATION TYPE WEIGHTS
# =============================================================================

# Weights for semantic relation types used in:
# - PageRank computation (analysis.py)
# - Semantic retrofitting (semantics.py)
# - Query expansion (query/expansion.py)
#
# Higher values = stronger connections in the knowledge graph.
# These are tuned based on ConceptNet-style relation semantics.

RELATION_WEIGHTS: Dict[str, float] = {
    # Strong semantic relationships
    'SameAs': 2.0,          # Synonymy - strongest connection
    'IsA': 1.5,             # Hypernym/type relationships
    'SimilarTo': 1.5,       # High similarity

    # Structural relationships
    'PartOf': 1.3,          # Meronym relationships
    'HasA': 1.2,            # Possession relationships
    'HasProperty': 1.2,     # Property associations
    'DerivedFrom': 1.2,     # Morphological derivation

    # Causal and functional
    'Causes': 1.1,          # Causal relationships
    'UsedFor': 1.0,         # Functional relationships
    'CapableOf': 0.9,       # Capability relationships
    'DefinedBy': 1.0,       # Definition relationships

    # Co-occurrence and spatial
    'RelatedTo': 0.8,       # General relatedness
    'CoOccurs': 0.7,        # Basic co-occurrence
    'AtLocation': 0.6,      # Spatial relationships

    # Negative/opposing
    'Antonym': -0.5,        # Opposing concepts (negative weight)
}


# =============================================================================
# DOCUMENT TYPE BOOSTS
# =============================================================================

# Boost factors for ranking documents by type in search results.
# Used in query/ranking.py for multi_stage_rank().
# Higher values = ranked higher in results.

DOC_TYPE_BOOSTS: Dict[str, float] = {
    'docs': 1.5,            # docs/ folder documentation
    'root_docs': 1.3,       # Root-level markdown (CLAUDE.md, README.md)
    'code': 1.0,            # Regular code files
    'test': 0.8,            # Test files (often less relevant for conceptual queries)
}


# =============================================================================
# QUERY TYPE KEYWORDS
# =============================================================================

# Keywords that suggest a conceptual query (should boost documentation)
CONCEPTUAL_KEYWORDS: FrozenSet[str] = frozenset([
    'what', 'explain', 'describe', 'overview', 'introduction', 'concept',
    'architecture', 'design', 'pattern', 'algorithm', 'approach', 'method',
    'how does', 'why does', 'purpose', 'goal', 'rationale', 'theory',
    'understand', 'learn', 'documentation', 'guide', 'tutorial', 'example',
])

# Keywords that suggest an implementation query (should prefer code)
IMPLEMENTATION_KEYWORDS: FrozenSet[str] = frozenset([
    'where', 'implement', 'code', 'function', 'class', 'method', 'variable',
    'line', 'file', 'bug', 'fix', 'error', 'exception', 'call', 'invoke',
    'compute', 'calculate', 'return', 'parameter', 'argument',
])


# NOTE: LAYER_COLORS and LAYER_NAMES are defined in persistence.py
# because they use CorticalLayer enum keys for type safety in exports.
