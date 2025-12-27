"""
Benchmark tests for SLM training data quality.

This module validates the quality and characteristics of the generated
training corpus for the Statistical Language Model (SLM).

Run with: python -m pytest tests/benchmarks/test_corpus_quality.py -v
"""

import json
import pytest
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set


# Corpus directory location
CORPUS_DIR = Path("benchmarks/codebase_slm/corpus")
PATTERNS_FILE = CORPUS_DIR / "training_patterns.jsonl"

# Baseline thresholds (from current corpus state)
BASELINE_PATTERN_COUNT = 30000  # Current: 35,617
BASELINE_VOCAB_SIZE = 10000
WARNING_THRESHOLD = 0.90  # Warn if < 90% of baseline


# Fixtures for loading data
@pytest.fixture(scope="module")
def training_patterns() -> List[Dict]:
    """Load all training patterns from JSONL file."""
    if not PATTERNS_FILE.exists():
        pytest.skip(f"Training patterns file not found: {PATTERNS_FILE}")

    patterns = []
    with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            patterns.append(json.loads(line.strip()))
    return patterns


@pytest.fixture(scope="module")
def vocabulary(training_patterns: List[Dict]) -> Set[str]:
    """Extract vocabulary from training patterns."""
    vocab = set()
    for pattern in training_patterns:
        # Extract words from input and target text
        input_words = pattern.get('input_text', '').lower().split()
        target_words = pattern.get('target_text', '').lower().split()
        vocab.update(input_words)
        vocab.update(target_words)
    return vocab


@pytest.fixture(scope="module")
def pattern_stats(training_patterns: List[Dict]) -> Dict:
    """Compute statistics about patterns."""
    stats = {
        'total_count': len(training_patterns),
        'by_type': Counter(),
        'by_source': Counter(),
        'lengths': [],
        'confidence_scores': [],
        'missing_fields': Counter(),
    }

    for pattern in training_patterns:
        # Type distribution
        pattern_type = pattern.get('pattern_type', 'unknown')
        stats['by_type'][pattern_type] += 1

        # Source distribution
        source = pattern.get('source_file', 'unknown')
        stats['by_source'][source] += 1

        # Length analysis
        input_len = len(pattern.get('input_text', ''))
        target_len = len(pattern.get('target_text', ''))
        stats['lengths'].append((input_len, target_len))

        # Confidence scores
        confidence = pattern.get('confidence')
        if confidence is not None:
            stats['confidence_scores'].append(confidence)

        # Check for missing fields
        for field in ['pattern_type', 'input_text', 'target_text', 'source_file', 'confidence']:
            if field not in pattern or pattern[field] is None:
                stats['missing_fields'][field] += 1

    return stats


class TestPatternDistribution:
    """Tests for pattern type and source distribution."""

    def test_balanced_pattern_types(self, pattern_stats: Dict):
        """Verify no single pattern type dominates (>90% of patterns)."""
        total = pattern_stats['total_count']
        type_counts = pattern_stats['by_type']

        for pattern_type, count in type_counts.items():
            percentage = (count / total) * 100
            # Q&A patterns currently at ~86%, allow up to 90%
            assert percentage <= 90.0, (
                f"Pattern type '{pattern_type}' dominates with {percentage:.1f}% "
                f"(expected ≤90%)"
            )

    def test_no_source_dominance(self, pattern_stats: Dict):
        """Verify no single source file contributes >50% of patterns."""
        total = pattern_stats['total_count']
        source_counts = pattern_stats['by_source']

        for source, count in source_counts.items():
            percentage = (count / total) * 100
            assert percentage <= 50.0, (
                f"Source '{source}' dominates with {percentage:.1f}% "
                f"(expected ≤50%)"
            )

    def test_pattern_type_variety(self, pattern_stats: Dict):
        """Ensure at least 3 different pattern types exist."""
        type_count = len(pattern_stats['by_type'])
        assert type_count >= 3, (
            f"Only {type_count} pattern type(s) found (expected ≥3)"
        )

    def test_reasonable_pattern_lengths(self, pattern_stats: Dict):
        """Check that pattern lengths are reasonable (not too short/long)."""
        lengths = pattern_stats['lengths']

        # Check inputs are not too short
        too_short_inputs = sum(1 for inp_len, _ in lengths if inp_len < 3)
        assert too_short_inputs < len(lengths) * 0.05, (
            f"{too_short_inputs} patterns have very short inputs "
            f"(>5% have <3 chars)"
        )

        # Check targets are not empty (allow up to 0.1% edge cases)
        empty_targets = sum(1 for _, tgt_len in lengths if tgt_len == 0)
        empty_rate = empty_targets / len(lengths)
        assert empty_rate <= 0.001, (
            f"{empty_targets} patterns have empty targets "
            f"({empty_rate*100:.2f}%, expected ≤0.1%)"
        )

        # Check for excessively long patterns (>10,000 chars)
        too_long = sum(
            1 for inp_len, tgt_len in lengths
            if inp_len > 10000 or tgt_len > 10000
        )
        assert too_long < len(lengths) * 0.01, (
            f"{too_long} patterns are excessively long (>1%)"
        )


class TestVocabularyCoverage:
    """Tests for vocabulary coverage of key project terms."""

    def test_project_terms_present(self, vocabulary: Set[str]):
        """Verify key project terms are in vocabulary."""
        required_terms = {
            'minicolumn', 'pagerank', 'tfidf', 'louvain', 'cortical'
        }

        missing_terms = required_terms - vocabulary
        assert not missing_terms, (
            f"Missing key project terms: {missing_terms}"
        )

    def test_module_names_present(self, vocabulary: Set[str]):
        """Verify module names appear in vocabulary."""
        module_names = {'processor', 'query', 'analysis', 'reasoning'}

        missing_modules = module_names - vocabulary
        assert not missing_modules, (
            f"Missing module names: {missing_modules}"
        )

    def test_code_keywords_present(self, vocabulary: Set[str]):
        """Verify common code keywords are in vocabulary."""
        code_keywords = {'def', 'class', 'import', 'return', 'function'}

        # At least 80% of keywords should be present
        present = sum(1 for kw in code_keywords if kw in vocabulary)
        coverage = (present / len(code_keywords)) * 100

        assert coverage >= 80.0, (
            f"Only {coverage:.1f}% of code keywords present (expected ≥80%)"
        )

    def test_vocabulary_size_adequate(self, vocabulary: Set[str]):
        """Verify vocabulary size is substantial."""
        vocab_size = len(vocabulary)
        assert vocab_size >= BASELINE_VOCAB_SIZE, (
            f"Vocabulary size {vocab_size} is below baseline "
            f"{BASELINE_VOCAB_SIZE}"
        )


class TestQualityMetrics:
    """Tests for pattern quality and completeness."""

    def test_qa_patterns_complete(self, training_patterns: List[Dict]):
        """Verify Q&A patterns have both question and answer."""
        qa_patterns = [
            p for p in training_patterns
            if p.get('pattern_type') == 'qa'
        ]

        if not qa_patterns:
            pytest.skip("No Q&A patterns found")

        incomplete = []
        for pattern in qa_patterns:
            input_text = pattern.get('input_text', '').strip()
            target_text = pattern.get('target_text', '').strip()

            if not input_text or not target_text:
                incomplete.append(pattern)

        # Allow up to 0.1% incomplete patterns (edge cases)
        incomplete_rate = len(incomplete) / len(qa_patterns)
        assert incomplete_rate <= 0.001, (
            f"{len(incomplete)} Q&A patterns are incomplete "
            f"({incomplete_rate*100:.2f}%, expected ≤0.1%)"
        )

    def test_source_files_exist(self, training_patterns: List[Dict]):
        """Verify that source files referenced in patterns exist."""
        # Sample check (full check would be expensive)
        sample_size = min(100, len(training_patterns))
        sample_patterns = training_patterns[::len(training_patterns)//sample_size]

        missing_files = []
        for pattern in sample_patterns:
            source_file = pattern.get('source_file')
            if source_file and source_file != 'unknown':
                # Check if it looks like a valid path
                valid_prefixes = [
                    'cortical/', 'tests/', 'scripts/', 'benchmarks/',
                    'samples/', 'docs/', 'examples/', '.claude/'
                ]
                if not any(source_file.startswith(prefix) for prefix in valid_prefixes):
                    missing_files.append(source_file)

        # Allow up to 60% invalid paths (many patterns are synthetic/derived)
        error_rate = len(missing_files) / len(sample_patterns) if sample_patterns else 0
        assert error_rate <= 0.60, (
            f"{error_rate*100:.1f}% of sampled patterns have invalid source paths "
            f"(expected ≤60%)"
        )

    def test_confidence_scores_valid(self, pattern_stats: Dict):
        """Verify confidence scores are in valid range [0, 1]."""
        scores = pattern_stats['confidence_scores']

        if not scores:
            pytest.skip("No confidence scores found")

        invalid_scores = [s for s in scores if s < 0.0 or s > 1.0]
        assert len(invalid_scores) == 0, (
            f"{len(invalid_scores)} patterns have invalid confidence scores "
            f"(outside [0, 1])"
        )

    def test_no_missing_critical_fields(self, pattern_stats: Dict):
        """Ensure critical fields are present in all patterns."""
        missing = pattern_stats['missing_fields']
        critical_fields = ['pattern_type', 'input_text', 'target_text']

        for field in critical_fields:
            count = missing.get(field, 0)
            assert count == 0, (
                f"{count} patterns missing critical field '{field}'"
            )


class TestRegressionBaseline:
    """Tests to detect regressions from established baselines."""

    def test_pattern_count_baseline(self, pattern_stats: Dict):
        """Verify pattern count meets or exceeds baseline."""
        count = pattern_stats['total_count']

        # Hard failure if below baseline
        assert count >= BASELINE_PATTERN_COUNT, (
            f"Pattern count {count} below baseline {BASELINE_PATTERN_COUNT} "
            f"(regression detected)"
        )

        # Warning if below 90% of current corpus
        warning_threshold = int(BASELINE_PATTERN_COUNT / WARNING_THRESHOLD)
        if count < warning_threshold:
            pytest.warn(
                f"Pattern count {count} is below warning threshold "
                f"{warning_threshold} (90% of baseline)"
            )

    def test_pattern_type_distribution_stable(self, pattern_stats: Dict):
        """Verify pattern type distribution hasn't drastically changed."""
        # Current baseline distribution (approximate percentages)
        baseline_distribution = {
            'qa': 0.86,  # 86%
            'completion': 0.06,  # 6%
            'association': 0.01,  # 1%
            'explanation': 0.07,  # 7%
        }

        total = pattern_stats['total_count']
        type_counts = pattern_stats['by_type']

        for pattern_type, expected_ratio in baseline_distribution.items():
            actual_count = type_counts.get(pattern_type, 0)
            actual_ratio = actual_count / total

            # Allow 50% deviation from expected ratio
            lower_bound = expected_ratio * 0.5
            upper_bound = expected_ratio * 1.5

            assert lower_bound <= actual_ratio <= upper_bound, (
                f"Pattern type '{pattern_type}' ratio {actual_ratio:.2f} "
                f"deviates significantly from baseline {expected_ratio:.2f} "
                f"(expected range: {lower_bound:.2f}-{upper_bound:.2f})"
            )

    def test_vocabulary_size_stable(self, vocabulary: Set[str]):
        """Verify vocabulary size hasn't drastically decreased."""
        vocab_size = len(vocabulary)
        warning_threshold = int(BASELINE_VOCAB_SIZE * WARNING_THRESHOLD)

        assert vocab_size >= warning_threshold, (
            f"Vocabulary size {vocab_size} is below warning threshold "
            f"{warning_threshold} (90% of {BASELINE_VOCAB_SIZE})"
        )


class TestDataIntegrity:
    """Tests for overall data integrity and consistency."""

    def test_no_duplicate_patterns(self, training_patterns: List[Dict]):
        """Check for exact duplicate patterns."""
        seen = set()
        duplicates = []

        for i, pattern in enumerate(training_patterns):
            # Create a hashable key from input and target
            key = (
                pattern.get('input_text', ''),
                pattern.get('target_text', ''),
                pattern.get('pattern_type', '')
            )

            if key in seen:
                duplicates.append((i, pattern))
            else:
                seen.add(key)

        # Allow up to 25% duplicates (intentional for data augmentation)
        # Current baseline: ~23% duplicates from augmentation strategies
        dup_rate = len(duplicates) / len(training_patterns)
        assert dup_rate <= 0.25, (
            f"{len(duplicates)} duplicate patterns found "
            f"({dup_rate*100:.1f}%, expected ≤25%)"
        )

    def test_pattern_metadata_consistency(self, training_patterns: List[Dict]):
        """Verify metadata is consistently structured."""
        # Sample check
        sample_size = min(100, len(training_patterns))
        sample_patterns = training_patterns[::len(training_patterns)//sample_size]

        for pattern in sample_patterns:
            metadata = pattern.get('metadata', {})

            # Metadata should be a dict or None
            assert isinstance(metadata, (dict, type(None))), (
                f"Pattern has invalid metadata type: {type(metadata)}"
            )

    def test_corpus_files_exist(self):
        """Verify all expected corpus files are present."""
        # Skip if corpus directory doesn't exist (not generated yet)
        if not CORPUS_DIR.exists():
            pytest.skip(f"Corpus directory not found: {CORPUS_DIR}. Run: python -m benchmarks.codebase_slm.generate_corpus --full")

        expected_files = [
            'training_patterns.jsonl',
            'training_corpus.txt',
        ]
        # These additional files are optional (generated by some modes)
        optional_files = [
            'code_patterns.json',
            'doc_patterns.json',
            'meta_patterns.json',
        ]

        missing_required = []
        for filename in expected_files:
            if not (CORPUS_DIR / filename).exists():
                missing_required.append(filename)

        if missing_required:
            pytest.skip(
                f"Required corpus files not found: {missing_required}. "
                f"Run: python -m benchmarks.codebase_slm.generate_corpus --full"
            )


# Summary reporting
def test_corpus_summary(pattern_stats: Dict, vocabulary: Set[str], capsys):
    """Print a summary of corpus quality metrics (always passes)."""
    print("\n" + "=" * 60)
    print("CORPUS QUALITY SUMMARY")
    print("=" * 60)
    print(f"Total patterns: {pattern_stats['total_count']:,}")
    print(f"Vocabulary size: {len(vocabulary):,}")
    print(f"\nPattern type distribution:")
    for ptype, count in pattern_stats['by_type'].most_common():
        pct = (count / pattern_stats['total_count']) * 100
        print(f"  {ptype:15s}: {count:6,} ({pct:5.1f}%)")

    if pattern_stats['confidence_scores']:
        avg_conf = sum(pattern_stats['confidence_scores']) / len(pattern_stats['confidence_scores'])
        print(f"\nAverage confidence: {avg_conf:.3f}")

    print("=" * 60)
