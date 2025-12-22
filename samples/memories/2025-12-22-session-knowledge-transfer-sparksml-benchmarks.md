# Knowledge Transfer: SparkSLM Benchmarks and Demo

**Date:** 2025-12-22
**Session Focus:** Setting up pytest/coverage, creating SparkSLM benchmarks and demo
**Branch:** `claude/setup-pytest-sparksml-71W05`

---

## Summary

This session established comprehensive benchmarking and demonstration infrastructure for SparkSLM, the Statistical First-Blitz Language Model component of the Cortical Text Processor.

## Work Completed

### 1. Environment Setup
- Installed `pytest` (9.0.2) and `coverage` (7.13.0)
- Verified all 54 existing SparkSLM integration tests pass

### 2. Created Benchmark Suite

#### `tests/performance/test_spark_benchmarks.py`
17 pytest benchmark tests covering:

| Test Class | Purpose |
|------------|---------|
| `TestNGramTrainingBenchmark` | Training time and scaling |
| `TestNGramPredictionBenchmark` | Prediction latency, batch throughput |
| `TestSparkPredictorBenchmark` | Prime query, complete query, expansion |
| `TestAnomalyDetectorBenchmark` | Normal/suspicious text checking |
| `TestSparkMemoryBenchmark` | Memory footprint analysis |
| `TestSparkQualityBenchmark` | Prediction accuracy, perplexity |

#### `scripts/benchmark_spark.py`
CLI tool for running benchmarks:

```bash
python scripts/benchmark_spark.py                    # Full suite
python scripts/benchmark_spark.py --benchmark training
python scripts/benchmark_spark.py --output results.json
python scripts/benchmark_spark.py --compare before.json after.json
```

### 3. Created Interactive Demo

#### `examples/spark_demo.py`
6-section demo with interactive mode:

1. **NGramModel** - Word prediction, sequence completion
2. **SparkPredictor** - Query priming facade
3. **AnomalyDetector** - Prompt injection detection
4. **Processor Integration** - Spark-enhanced search
5. **Quality Evaluation** - Accuracy and perplexity
6. **Real Corpus Demo** - Training on samples/ directory

```bash
python examples/spark_demo.py              # Full demo
python examples/spark_demo.py --quick      # Less verbose
python examples/spark_demo.py -i           # Interactive mode
python examples/spark_demo.py --section corpus
```

---

## Key Findings: SparkSLM Scaling Limits

### Training Performance

| Documents | Tokens | Training Time | Use Case |
|-----------|--------|---------------|----------|
| 500 | 100K | 0.2s | Instant |
| 1,000 | 200K | 0.5s | Instant |
| 2,000 | 400K | 1.1s | Instant |
| 5,000 | 1M | 3.1s | Fast startup |
| 10,000 | 2M | 6.3s | Background |

### Real Corpus (samples/)
- **528 documents, 387K tokens**
- **Training: ~1.2 seconds**
- **Vocabulary: 23,233 terms**
- **Contexts: 255,293 unique**
- **Throughput: ~330K tokens/sec**

### Prediction Latency
- Known context: **1-30 μs** (microseconds)
- Unknown context fallback: Slower due to vocabulary scan
- Suitable for real-time autocomplete

### Practical Limits

| Use Case | Max Documents | Training Time |
|----------|---------------|---------------|
| Interactive (<1s) | ~2,000 | <1s |
| App startup (<3s) | ~5,000 | 2-3s |
| Background (<10s) | ~10,000-15,000 | 5-10s |
| One-time index | 50,000+ | 30s+ |

---

## Technical Notes

### AnomalyDetector API
The `AnomalyDetector` requires an `NGramModel` and uses `calibrate()`, not `train()`:

```python
model = NGramModel(n=3)
model.train(documents)
detector = AnomalyDetector(ngram_model=model)
detector.calibrate(sample_queries)
result = detector.check(text)
if result.is_anomalous:  # Note: is_anomalous, not is_anomaly
    print(result.reasons)
```

### Interesting Demo Output
Sequence completions from real corpus show domain knowledge:
```
'neural networks' -> 'neural networks to approximate value functions'
'machine learning' -> 'machine learning models predict next tick'
'the cortical' -> 'the cortical text processor is a'
```

---

## Files Changed

| File | Type | Lines |
|------|------|-------|
| `tests/performance/test_spark_benchmarks.py` | New | ~400 |
| `scripts/benchmark_spark.py` | New | ~650 |
| `examples/spark_demo.py` | Modified | +104 |

## Commits

1. `feat(spark): Add comprehensive SparkSLM benchmarks`
2. `feat(spark): Add interactive SparkSLM demo`
3. `feat(spark): Add real corpus demo section to SparkSLM demo`

---

## Recommendations

1. **Current samples/ corpus is optimal** - 528 docs trains in ~1s
2. **Could scale to 5,000+ docs** while maintaining acceptable startup
3. **Consider pre-training** for massive corpora (100K+ docs)
4. **Potential optimization**: Cache `_most_frequent_words()` result to improve unknown context fallback

---

## Related Files

- `cortical/spark/__init__.py` - SparkSLM package exports
- `cortical/spark/ngram.py` - NGramModel implementation
- `cortical/spark/predictor.py` - SparkPredictor facade
- `cortical/spark/anomaly.py` - AnomalyDetector
- `tests/unit/test_spark_integration.py` - 54 integration tests

---

**Tags:** `sparksml`, `benchmarks`, `performance`, `demo`, `ngram`, `anomaly-detection`
