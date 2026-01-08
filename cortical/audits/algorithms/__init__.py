"""
Algorithm implementations for codebase auditing.

These algorithms were implemented from first principles by sub-agents
as part of the algorithm implementation challenge (2026-01-07).

Available algorithms:
- inverted_index: Fast text search for audit findings
- decision_tree: Comment classification (misleading/accurate)
- trie: Comment marker lookup (FUTURE:, TODO:, etc.)
- naive_bayes: Probabilistic comment classification
- union_find: Clustering related audit findings
- bloom_filter: Fast suspicious pattern pre-screening
- suffix_array: Finding repeated patterns in comments
- dag: Task dependency management for GoT
- markov_chain: Comment pattern detection
- lsh: Similar comment detection via MinHash
- count_min_sketch: Streaming pattern frequency

All implementations:
- Zero external dependencies (stdlib only)
- Handle edge cases properly
- Production-ready with documentation
"""

from .inverted_index import AuditInvertedIndex
from .decision_tree import CommentDecisionTree
from .trie import CommentMarkerTrie
from .naive_bayes import CommentClassifier
from .union_find import FindingCluster
from .bloom_filter import SuspiciousCommentFilter
from .suffix_array import CommentPatternFinder
from .dag import TaskDAG
from .markov_chain import CommentMarkovChain
from .lsh import SimilarCommentFinder
from .count_min_sketch import PatternFrequencySketch

__all__ = [
    'AuditInvertedIndex',
    'CommentDecisionTree',
    'CommentMarkerTrie',
    'CommentClassifier',
    'FindingCluster',
    'SuspiciousCommentFilter',
    'CommentPatternFinder',
    'TaskDAG',
    'CommentMarkovChain',
    'SimilarCommentFinder',
    'PatternFrequencySketch',
]
