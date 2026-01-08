"""
Comment classification logic.

This module provides classification of comments as misleading, accurate, or unknown
based on pattern matching and machine learning.
"""

import re
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .patterns import MISLEADING_PATTERNS, ACCURATE_PATTERNS, EXCLUDE_PATTERNS

# Default confidence threshold for model-based classification
DEFAULT_CONFIDENCE_THRESHOLD = 0.65


@dataclass
class ClassificationResult:
    """Result of classifying a comment."""
    classification: str  # 'misleading', 'accurate', or 'unknown'
    pattern_type: str    # Category of matched pattern (e.g., 'speculative', 'returns')
    matched_pattern: str # The regex pattern that matched
    confidence: float    # Confidence score (0.0 - 1.0)


def should_exclude(comment: str) -> bool:
    """
    Check if a comment should be excluded from classification.

    Excluded comments are noise (separators, linter directives, etc.)

    Args:
        comment: Comment text to check

    Returns:
        True if comment should be excluded
    """
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, comment, re.IGNORECASE):
            return True
    return False


def classify_comment(comment: str) -> ClassificationResult:
    """
    Classify a comment based on pattern matching.

    Args:
        comment: Comment text to classify

    Returns:
        ClassificationResult with classification details
    """
    comment_lower = comment.lower()

    # Check misleading patterns first (they're usually more specific)
    for pattern, pattern_type in MISLEADING_PATTERNS:
        if re.search(pattern, comment_lower, re.IGNORECASE):
            return ClassificationResult(
                classification='misleading',
                pattern_type=pattern_type,
                matched_pattern=pattern,
                confidence=0.8,  # Pattern-based confidence
            )

    # Check accurate patterns
    for pattern, pattern_type in ACCURATE_PATTERNS:
        # Some accurate patterns are case-sensitive (e.g., "Returns" vs "returns")
        if re.search(pattern, comment, re.IGNORECASE):
            return ClassificationResult(
                classification='accurate',
                pattern_type=pattern_type,
                matched_pattern=pattern,
                confidence=0.8,
            )

    # No pattern matched
    return ClassificationResult(
        classification='unknown',
        pattern_type='',
        matched_pattern='',
        confidence=0.0,
    )


def classify_with_model(
    comment: str,
    classifier,  # CommentClassifier from algorithms
    tokenizer,   # Tokenizer for preprocessing
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ClassificationResult:
    """
    Classify a comment using the trained Naive Bayes model.

    Args:
        comment: Comment text to classify
        classifier: Trained CommentClassifier
        tokenizer: Tokenizer for preprocessing
        threshold: Minimum confidence to return a classification (default: 0.65)

    Returns:
        ClassificationResult with classification and confidence
    """
    # Tokenize
    cleaned = re.sub(r'[^\w\s]', ' ', comment)
    tokens = tokenizer.tokenize(cleaned, split_identifiers=True)

    if not tokens:
        return ClassificationResult(
            classification='unknown',
            pattern_type='',
            matched_pattern='',
            confidence=0.0,
        )

    # Get probabilities from model
    probs = classifier.predict_proba(tokens)

    # Determine classification
    # Only classify if confidence exceeds threshold to reduce false positives
    if 'misleading' in probs and probs['misleading'] > threshold:
        return ClassificationResult(
            classification='misleading',
            pattern_type='model',
            matched_pattern='naive_bayes',
            confidence=probs['misleading'],
        )
    elif 'accurate' in probs and probs['accurate'] > threshold:
        return ClassificationResult(
            classification='accurate',
            pattern_type='model',
            matched_pattern='naive_bayes',
            confidence=probs['accurate'],
        )
    else:
        return ClassificationResult(
            classification='unknown',
            pattern_type='',
            matched_pattern='',
            confidence=0.0,
        )


def batch_classify(
    comments: List[str],
    use_model: bool = False,
    classifier=None,
    tokenizer=None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> List[ClassificationResult]:
    """
    Classify multiple comments.

    Args:
        comments: List of comment texts
        use_model: Whether to use ML model (requires classifier and tokenizer)
        classifier: Trained classifier (if use_model=True)
        tokenizer: Tokenizer (if use_model=True)
        threshold: Minimum confidence for model-based classification

    Returns:
        List of ClassificationResults
    """
    results = []

    for comment in comments:
        if use_model and classifier and tokenizer:
            result = classify_with_model(comment, classifier, tokenizer, threshold)
        else:
            result = classify_comment(comment)
        results.append(result)

    return results
