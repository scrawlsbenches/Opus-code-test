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

Philosophy:
    SparkSLM is NOT a neural language model. It's statistical pattern
    matching that provides useful "sparks" to guide deeper analysis.
    Fast, interpretable, zero dependencies beyond the cortical core.
"""

from .ngram import NGramModel
from .alignment import AlignmentIndex
from .predictor import SparkPredictor
from .anomaly import AnomalyDetector, AnomalyResult

__all__ = ['NGramModel', 'AlignmentIndex', 'SparkPredictor', 'AnomalyDetector', 'AnomalyResult']
