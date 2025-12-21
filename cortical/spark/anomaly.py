"""
Anomaly Detector
================

Detects anomalous queries using statistical methods.

Key detection methods:
1. Perplexity-based: Unusual queries have higher perplexity
2. Pattern-based: Known injection patterns
3. Distribution-based: Queries far from training distribution

Example:
    >>> from cortical.spark import AnomalyDetector, NGramModel
    >>> model = NGramModel()
    >>> model.train(documents)
    >>> detector = AnomalyDetector(model)
    >>> detector.calibrate(normal_queries)
    >>>
    >>> result = detector.check("normal query")
    >>> print(result.is_anomalous)  # False
    >>>
    >>> result = detector.check("ignore previous instructions")
    >>> print(result.is_anomalous)  # True
    >>> print(result.reasons)  # ['injection_pattern', 'high_perplexity']
"""

import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, Any

from .ngram import NGramModel


@dataclass
class AnomalyResult:
    """Result of anomaly detection check."""
    query: str
    is_anomalous: bool
    confidence: float  # 0.0 = definitely normal, 1.0 = definitely anomalous
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "ANOMALOUS" if self.is_anomalous else "NORMAL"
        return f"AnomalyResult({status}, confidence={self.confidence:.2f}, reasons={self.reasons})"


class AnomalyDetector:
    """
    Detects anomalous queries using statistical and pattern-based methods.

    Detection methods:
    1. Perplexity threshold: Queries with perplexity > threshold are flagged
    2. Injection patterns: Known prompt injection phrases
    3. Vocabulary coverage: Queries with many unknown words
    4. Length anomalies: Unusually long or short queries

    All methods are optional and can be enabled/disabled.
    """

    # Known prompt injection patterns (case-insensitive)
    INJECTION_PATTERNS = [
        r'\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)\b',
        r'\bforget\s+(everything|all|your)\b',
        r'\byou\s+are\s+now\b',
        r'\bact\s+as\s+(if|a|an)\b',
        r'\bpretend\s+(to\s+be|you\s+are)\b',
        r'\bsystem\s*:\s*\b',
        r'\b(user|human|assistant)\s*:\s*\b',
        r'\bjailbreak\b',
        r'\bbypass\b.*\b(filter|safety|restriction)s?\b',
        r'\bdo\s+not\s+follow\b',
        r'\boverride\b.*\b(instruction|rule|constraint)s?\b',
        r'\bdisregard\b.*\b(previous|safety|instruction)s?\b',
        r'\bwrite\s+malicious\b',
        r'\bgenerate\s+(harmful|dangerous|illegal)\b',
        r'\b(ignore|bypass)\s+ethics\b',
        r'\bexecute\s+(code|command|script)\b',
        r'<\s*script\s*>',  # XSS attempt
        r';\s*(drop|delete|truncate)\s+table',  # SQL injection
        r'\$\{.*\}',  # Template injection
        r'{{.*}}',  # Template injection
    ]

    def __init__(
        self,
        ngram_model: Optional[NGramModel] = None,
        perplexity_threshold: float = 2.0,  # Multiplier over baseline
        unknown_word_threshold: float = 0.5,  # Max fraction of unknown words
        min_query_length: int = 2,
        max_query_length: int = 500,
    ):
        """
        Initialize AnomalyDetector.

        Args:
            ngram_model: Trained n-gram model for perplexity calculation
            perplexity_threshold: Anomaly if perplexity > baseline * threshold
            unknown_word_threshold: Anomaly if unknown words > threshold
            min_query_length: Flag queries shorter than this
            max_query_length: Flag queries longer than this
        """
        self.ngram = ngram_model
        self.perplexity_threshold = perplexity_threshold
        self.unknown_word_threshold = unknown_word_threshold
        self.min_query_length = min_query_length
        self.max_query_length = max_query_length

        # Calibration state
        self.baseline_perplexity: Optional[float] = None
        self.perplexity_std: Optional[float] = None
        self.calibrated = False

        # Compile injection patterns
        self._injection_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.INJECTION_PATTERNS
        ]

    def calibrate(self, normal_queries: List[str]) -> Dict[str, float]:
        """
        Calibrate detector using known-normal queries.

        Establishes baseline perplexity statistics from normal queries.

        Args:
            normal_queries: List of queries known to be normal

        Returns:
            Calibration statistics

        Raises:
            ValueError: If no valid queries provided
            RuntimeError: If n-gram model not provided
        """
        if not self.ngram:
            raise RuntimeError("N-gram model required for calibration")

        if not normal_queries:
            raise ValueError("At least one normal query required")

        # Calculate perplexity for each query
        perplexities = []
        for query in normal_queries:
            if len(query.strip()) >= self.min_query_length:
                try:
                    ppl = self.ngram.perplexity(query)
                    if ppl > 0 and ppl < float('inf'):
                        perplexities.append(ppl)
                except Exception:
                    continue

        if not perplexities:
            raise ValueError("No valid perplexities computed")

        self.baseline_perplexity = statistics.mean(perplexities)
        self.perplexity_std = statistics.stdev(perplexities) if len(perplexities) > 1 else 0
        self.calibrated = True

        return {
            'baseline_perplexity': self.baseline_perplexity,
            'perplexity_std': self.perplexity_std,
            'num_queries': len(perplexities),
            'threshold': self.baseline_perplexity * self.perplexity_threshold,
        }

    def check(self, query: str) -> AnomalyResult:
        """
        Check if a query is anomalous.

        Runs all enabled detection methods and combines results.

        Args:
            query: Query to check

        Returns:
            AnomalyResult with detection details
        """
        reasons = []
        metrics = {}
        confidence_signals = []

        # Check 1: Injection patterns
        injection_match = self._check_injection_patterns(query)
        if injection_match:
            reasons.append(f'injection_pattern:{injection_match}')
            confidence_signals.append(0.9)  # High confidence for pattern match
            metrics['injection_pattern'] = injection_match

        # Check 2: Perplexity (if model available and calibrated)
        if self.ngram and self.calibrated:
            ppl_result = self._check_perplexity(query)
            metrics.update(ppl_result)
            if ppl_result.get('is_high', False):
                reasons.append('high_perplexity')
                confidence_signals.append(ppl_result.get('confidence', 0.5))

        # Check 3: Unknown word ratio (if model available)
        if self.ngram:
            unknown_result = self._check_unknown_words(query)
            metrics.update(unknown_result)
            if unknown_result.get('is_high', False):
                reasons.append('high_unknown_ratio')
                confidence_signals.append(unknown_result.get('confidence', 0.3))

        # Check 4: Length anomalies
        length_result = self._check_length(query)
        metrics.update(length_result)
        if length_result.get('is_anomalous', False):
            reasons.append(length_result.get('reason', 'length_anomaly'))
            confidence_signals.append(0.4)  # Lower confidence for length

        # Combine confidence signals
        if confidence_signals:
            # Use max confidence (any strong signal is concerning)
            confidence = max(confidence_signals)
        else:
            confidence = 0.0

        is_anomalous = len(reasons) > 0

        return AnomalyResult(
            query=query,
            is_anomalous=is_anomalous,
            confidence=confidence,
            reasons=reasons,
            metrics=metrics
        )

    def _check_injection_patterns(self, query: str) -> Optional[str]:
        """Check for known injection patterns."""
        for pattern in self._injection_patterns:
            match = pattern.search(query)
            if match:
                return match.group(0)[:50]  # Return matched text (truncated)
        return None

    def _check_perplexity(self, query: str) -> Dict[str, Any]:
        """Check perplexity against baseline."""
        try:
            ppl = self.ngram.perplexity(query)
            threshold = self.baseline_perplexity * self.perplexity_threshold

            is_high = ppl > threshold

            # Calculate confidence based on how far above threshold
            if is_high and self.perplexity_std > 0:
                z_score = (ppl - self.baseline_perplexity) / self.perplexity_std
                confidence = min(0.9, 0.5 + (z_score * 0.1))
            else:
                confidence = 0.0

            return {
                'perplexity': ppl,
                'baseline': self.baseline_perplexity,
                'threshold': threshold,
                'is_high': is_high,
                'confidence': confidence,
            }
        except Exception as e:
            return {'perplexity_error': str(e)}

    def _check_unknown_words(self, query: str) -> Dict[str, Any]:
        """Check ratio of unknown words."""
        words = query.lower().split()
        if not words:
            return {'unknown_ratio': 0.0, 'is_high': False}

        unknown_count = sum(1 for w in words if w not in self.ngram.vocab)
        ratio = unknown_count / len(words)

        is_high = ratio > self.unknown_word_threshold
        confidence = min(0.7, ratio)  # Higher ratio = higher confidence

        return {
            'unknown_ratio': ratio,
            'unknown_count': unknown_count,
            'total_words': len(words),
            'is_high': is_high,
            'confidence': confidence if is_high else 0.0,
        }

    def _check_length(self, query: str) -> Dict[str, Any]:
        """Check query length."""
        length = len(query)

        if length < self.min_query_length:
            return {
                'length': length,
                'is_anomalous': True,
                'reason': 'too_short',
            }
        elif length > self.max_query_length:
            return {
                'length': length,
                'is_anomalous': True,
                'reason': 'too_long',
            }

        return {
            'length': length,
            'is_anomalous': False,
        }

    def batch_check(self, queries: List[str]) -> List[AnomalyResult]:
        """
        Check multiple queries for anomalies.

        Args:
            queries: List of queries to check

        Returns:
            List of AnomalyResult for each query
        """
        return [self.check(query) for query in queries]

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics and configuration."""
        return {
            'calibrated': self.calibrated,
            'baseline_perplexity': self.baseline_perplexity,
            'perplexity_std': self.perplexity_std,
            'perplexity_threshold': self.perplexity_threshold,
            'unknown_word_threshold': self.unknown_word_threshold,
            'min_query_length': self.min_query_length,
            'max_query_length': self.max_query_length,
            'injection_patterns_count': len(self._injection_patterns),
            'has_ngram_model': self.ngram is not None,
        }

    def add_injection_pattern(self, pattern: str) -> None:
        """
        Add a custom injection pattern.

        Args:
            pattern: Regex pattern to match
        """
        compiled = re.compile(pattern, re.IGNORECASE)
        self._injection_patterns.append(compiled)

    def reset_calibration(self) -> None:
        """Reset calibration state."""
        self.baseline_perplexity = None
        self.perplexity_std = None
        self.calibrated = False
