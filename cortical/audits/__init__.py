"""
Cortical Audits - Codebase Quality Analysis.

This package provides tools for analyzing and maintaining codebase quality:
- Comment classification (misleading vs accurate)
- Pattern detection (repeated code, copy-paste)
- Similarity search (LSH-based)
- Training data generation
- PLN-based reasoning with WovenMind integration
- Codebase health analysis

Components:
    algorithms/     - Core algorithm implementations (Bloom filter, Naive Bayes, etc.)
    patterns.py     - Pattern definitions for comment classification
    classifier.py   - Comment classification logic
    training.py     - Training data generation
    scanner.py      - Codebase scanning utilities
    persistence.py  - State persistence for audit reasoning
    health.py       - Codebase health analysis
    reasoning.py    - PLN-based audit reasoning

Usage:
    from cortical.audits import CommentClassifier, MISLEADING_PATTERNS
    from cortical.audits.algorithms import SuspiciousCommentFilter
    from cortical.audits import AuditReasoner, CodebaseAnalyzer

CLI:
    python scripts/audit_tool.py scan cortical/
    python scripts/audit_tool.py generate cortical/
    python scripts/audit_reasoning.py cortical/
"""

# Re-export key classes from algorithms
from .algorithms import (
    AuditInvertedIndex,
    CommentDecisionTree,
    CommentMarkerTrie,
    CommentClassifier,
    FindingCluster,
    SuspiciousCommentFilter,
    CommentPatternFinder,
    TaskDAG,
    CommentMarkovChain,
    SimilarCommentFinder,
    PatternFrequencySketch,
)

# Re-export pattern definitions
from .patterns import (
    MISLEADING_PATTERNS,
    ACCURATE_PATTERNS,
    EXCLUDE_PATTERNS,
    SUSPICIOUS_PATTERNS,
    COMMENT_MARKERS,
)

# Re-export scanner utilities
from .scanner import (
    find_python_files,
    extract_comments_from_file,
    extract_comments,
    scan_directory,
    iter_comments,
    Comment,
    ScanResult,
)

# Re-export classifier utilities
from .classifier import (
    classify_comment,
    classify_with_model,
    should_exclude,
    ClassificationResult,
)

# Re-export training utilities
from .training import (
    generate_training_data,
    write_training_files,
    load_training_files,
    TrainingData,
    TrainingStats,
)

# Re-export persistence utilities
from .persistence import (
    PersistenceBackend,
    FilePersistenceBackend,
    NullPersistenceBackend,
    InMemoryPersistenceBackend,
    AuditPersistenceState,
    FileImportanceRecord,
    create_default_persistence,
)

# Re-export health analysis
from .health import (
    CodebaseAnalyzer,
    HealthAnalysisResult,
    analyze_directory,
    get_file_churn,
    get_git_blame_for_line,
    analyze_import_dependencies,
    DEFAULT_SUSPICIOUS_PATTERNS,
)

# Re-export reasoning
from .reasoning import (
    AuditReasoner,
    AuditQuery,
    translate_audit_query,
    is_natural_language_query,
    abstraction_to_rule,
    load_woven_mind_abstractions,
    DEFAULT_SIMPLE_RULES,
    DEFAULT_COMPOUND_RULES,
)

__all__ = [
    # Algorithms
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
    # Patterns
    'MISLEADING_PATTERNS',
    'ACCURATE_PATTERNS',
    'EXCLUDE_PATTERNS',
    'SUSPICIOUS_PATTERNS',
    'COMMENT_MARKERS',
    # Scanner
    'find_python_files',
    'extract_comments_from_file',
    'extract_comments',
    'scan_directory',
    'iter_comments',
    'Comment',
    'ScanResult',
    # Classifier
    'classify_comment',
    'classify_with_model',
    'should_exclude',
    'ClassificationResult',
    # Training
    'generate_training_data',
    'write_training_files',
    'load_training_files',
    'TrainingData',
    'TrainingStats',
]
