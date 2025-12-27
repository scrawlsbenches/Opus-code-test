# Corpus Quality Benchmark Tests

This directory contains benchmark tests for validating SLM training data quality.

## Test File: `test_corpus_quality.py`

Validates the quality and characteristics of training data in `benchmarks/codebase_slm/corpus/`.

### Test Categories

#### 1. Pattern Distribution Tests (`TestPatternDistribution`)
- ✅ Balanced pattern types (no single type >90%)
- ✅ No source file dominance (no single source >50%)
- ✅ Pattern type variety (≥3 types)
- ✅ Reasonable pattern lengths (not too short/long)

#### 2. Vocabulary Coverage Tests (`TestVocabularyCoverage`)
- ✅ Project terms present (minicolumn, pagerank, tfidf, louvain, cortical)
- ✅ Module names present (processor, query, analysis, reasoning)
- ✅ Code keywords present (def, class, import, return, function)
- ✅ Vocabulary size adequate (≥10,000 tokens)

#### 3. Quality Metrics Tests (`TestQualityMetrics`)
- ✅ Q&A patterns complete (input + target present)
- ✅ Source files exist and are valid paths
- ✅ Confidence scores in range [0, 1]
- ✅ No missing critical fields

#### 4. Regression Baseline Tests (`TestRegressionBaseline`)
- ✅ Pattern count ≥30,000 (current: 35,617)
- ✅ Pattern type distribution stable
- ✅ Vocabulary size stable (≥10,000)

#### 5. Data Integrity Tests (`TestDataIntegrity`)
- ✅ Limited duplicate patterns (≤25%, intentional for augmentation)
- ✅ Metadata consistency
- ✅ All corpus files present

### Running Tests

```bash
# Run all benchmark tests
python -m pytest tests/benchmarks/test_corpus_quality.py -v

# View corpus summary
python -m pytest tests/benchmarks/test_corpus_quality.py::test_corpus_summary -v -s

# Run specific test category
python -m pytest tests/benchmarks/test_corpus_quality.py::TestPatternDistribution -v
```

### Current Baseline (as of 2025-12-27)

- **Total patterns**: 35,617
- **Vocabulary size**: 51,372
- **Pattern distribution**:
  - Q&A: 30,788 (86.4%)
  - Explanation: 2,349 (6.6%)
  - Completion: 2,068 (5.8%)
  - Association: 412 (1.2%)
- **Average confidence**: 0.934

### Thresholds

| Metric | Baseline | Warning | Hard Limit |
|--------|----------|---------|------------|
| Pattern count | 30,000 | 27,000 (90%) | 30,000 |
| Vocabulary size | 10,000 | 9,000 (90%) | 10,000 |
| Q&A dominance | - | - | 90% |
| Source dominance | - | - | 50% |
| Empty targets | - | - | 0.1% |
| Duplicates | - | - | 25% |

### Adding New Tests

1. Add test method to appropriate class
2. Use fixtures: `training_patterns`, `vocabulary`, `pattern_stats`
3. Follow naming: `test_<what_is_tested>`
4. Include clear assertion messages

### Fixtures

- `training_patterns`: List of all pattern dicts from JSONL
- `vocabulary`: Set of all unique tokens
- `pattern_stats`: Dict with counts, distributions, lengths, scores
