"""
SparkSLM - Statistical First-Blitz Language Model
=================================================

"The spark that ignites before the fire fully forms"

This package provides fast, lightweight language understanding to prime
deeper search and analysis. It's System 1 thinking for the Cortical processor.

Components:
- NGramModel: Statistical word prediction based on context
- AlignmentIndex: User definitions, patterns, and preferences
- SparkPredictor: Unified facade for first-blitz predictions
- AnomalyDetector: Statistical and pattern-based anomaly detection
- CodeTokenizer: Code-aware tokenizer preserving punctuation and operators
- DiffTokenizer: Git diff tokenizer for code evolution training
- ASTIndex: AST-based code indexing for structural analysis
- SparkCodeIntelligence: Hybrid AST + N-gram code intelligence engine
- IntentParser: Commit message intent parsing for code evolution model
- CoChangeModel: Learn file co-change patterns from git history

Usage:
    from cortical.spark import SparkPredictor

    # Initialize with processor
    spark = SparkPredictor()
    spark.train(processor)

    # Get first-blitz suggestions
    primed = spark.prime("authentication handler")
    # Returns: keywords, topics, completions, safety check

    # Predict next words
    completions = spark.complete("neural net")
    # Returns: [("network", 0.7), ("networks", 0.2), ...]

    # Code intelligence
    from cortical.spark import SparkCodeIntelligence
    engine = SparkCodeIntelligence()
    engine.train()
    completions = engine.complete("self.", top_n=10)

Philosophy:
    SparkSLM is NOT a neural language model. It's statistical pattern
    matching that provides useful "sparks" to guide deeper analysis.
    Fast, interpretable, zero dependencies beyond the cortical core.
"""

from .ngram import NGramModel
from .alignment import AlignmentIndex
from .predictor import SparkPredictor
from .anomaly import AnomalyDetector, AnomalyResult
from .tokenizer import CodeTokenizer
from .diff_tokenizer import DiffTokenizer, DiffToken, DiffHunk, DiffFile, SPECIAL_TOKENS
from .ast_index import ASTIndex, FunctionInfo, ClassInfo, ImportInfo
from .intelligence import SparkCodeIntelligence
from .intent_parser import IntentParser, IntentResult
from .co_change import CoChangeModel, CoChangeEdge, Commit
from .suggester import (
    SampleSuggester,
    DefinitionSuggestion,
    PatternSuggestion,
    PreferenceSuggestion,
    Observation
)
from .transfer import (
    VocabularyAnalyzer,
    VocabularyAnalysis,
    PortableModel,
    TransferAdapter,
    TransferMetrics,
    PROGRAMMING_VOCABULARY,
    create_portable_model,
    transfer_knowledge,
)
from .quality import (
    QualityEvaluator,
    SearchQualityEvaluator,
    AlignmentEvaluator,
    PredictionMetrics,
    SearchMetrics,
    SearchComparison,
    AlignmentMetrics,
    generate_test_queries,
    generate_relevance_judgments,
)

__all__ = [
    # Core
    'NGramModel',
    'AlignmentIndex',
    'SparkPredictor',
    # Anomaly Detection
    'AnomalyDetector',
    'AnomalyResult',
    # Code Intelligence
    'CodeTokenizer',
    'DiffTokenizer',
    'DiffToken',
    'DiffHunk',
    'DiffFile',
    'SPECIAL_TOKENS',
    'ASTIndex',
    'FunctionInfo',
    'ClassInfo',
    'ImportInfo',
    'SparkCodeIntelligence',
    # Intent Parsing
    'IntentParser',
    'IntentResult',
    # Co-Change Model
    'CoChangeModel',
    'CoChangeEdge',
    'Commit',
    # Sample Suggestion
    'SampleSuggester',
    'DefinitionSuggestion',
    'PatternSuggestion',
    'PreferenceSuggestion',
    'Observation',
    # Transfer Learning
    'VocabularyAnalyzer',
    'VocabularyAnalysis',
    'PortableModel',
    'TransferAdapter',
    'TransferMetrics',
    'PROGRAMMING_VOCABULARY',
    'create_portable_model',
    'transfer_knowledge',
    # Quality Evaluation
    'QualityEvaluator',
    'SearchQualityEvaluator',
    'AlignmentEvaluator',
    'PredictionMetrics',
    'SearchMetrics',
    'SearchComparison',
    'AlignmentMetrics',
    'generate_test_queries',
    'generate_relevance_judgments',
]
