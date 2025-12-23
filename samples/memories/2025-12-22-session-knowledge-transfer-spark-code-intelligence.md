# Knowledge Transfer: SparkCodeIntelligence Module

**Date:** 2025-12-22
**Session:** SparkSLM Code Intelligence Implementation
**Tags:** `spark`, `code-intelligence`, `ast`, `refactoring`, `testing`

---

## Summary

This session created SparkCodeIntelligence, a hybrid AST + N-gram code intelligence engine. The work involved:
1. Training SparkSLM on the repository
2. Creating a comprehensive code assistant script
3. Fixing performance issues (500ms fallback → O(1) cached lookup)
4. Converting model storage from pickle to JSON (git-friendly)
5. Refactoring into modular packages with full test coverage

---

## What Was Built

### New Modules in `cortical/spark/`

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `tokenizer.py` | Code-aware tokenization | `CodeTokenizer` |
| `ast_index.py` | Structural code analysis | `ASTIndex`, `FunctionInfo`, `ClassInfo`, `ImportInfo` |
| `intelligence.py` | Hybrid AST + N-gram engine | `SparkCodeIntelligence` |

### CLI Script
`scripts/spark_code_intelligence.py` - Command-line interface with:
- `train` - Train on codebase
- `complete` - Code completion
- `find-callers` / `find-class` / `inheritance` / `imports` - Structural queries
- `query` - Natural language queries
- `coverage` - Test coverage estimation
- `benchmark` - Performance benchmarks

### Test Coverage (118 tests)
- `tests/unit/test_code_tokenizer.py` - 32 tests
- `tests/unit/test_ast_index.py` - 29 tests
- `tests/unit/test_spark_intelligence.py` - 31 tests
- `tests/integration/test_spark_intelligence_integration.py` - 17 tests
- `tests/performance/test_spark_intelligence_perf.py` - 14 tests

---

## Key Technical Decisions

### 1. JSON Over Pickle
**Decision:** Store models as JSON instead of pickle
**Rationale:** Git-friendly (diffable), secure (no RCE risk), human-readable
**Trade-off:** Larger file size (~55MB), slightly slower save/load

### 2. Separate N-gram Fallback Cache
**Decision:** Added `_cached_frequent_words` to NGramModel
**Rationale:** Original fallback was O(n) over 662k contexts = 500ms
**Fix:** Cache most frequent words after training, O(1) lookup

### 3. Case-Insensitive Query Matching
**Decision:** Natural language queries match class names case-insensitively
**Rationale:** "class that inherits processor" should find "Processor"

---

## Integration Points (For Future Work)

### High Value, Low Effort
1. **Enhance `/codebase-search` skill**
   - Add structural queries (callers, inheritance) alongside semantic search
   - AST queries are instant vs text search

2. **Pre-commit refactoring check**
   - When renaming functions, warn about callers
   - `engine.find_callers("old_name")` before commit

### High Value, Medium Effort
3. **Boost ML file prediction**
   - Use call graph to suggest related files
   - If changing `foo()`, suggest files that call `foo()`
   - If changing `BaseClass`, suggest child classes

4. **CI coverage estimation**
   - `CoverageEstimator` already maps tests to sources
   - Estimate impact of PRs without running full suite

### Medium Value
5. **Documentation generation**
   - ASTIndex extracts docstrings, signatures, class structures
   - Auto-generate API docs

6. **New `/code-intelligence` slash command**
   - Wrap CLI for quick structural queries during sessions

---

## Performance Characteristics

| Operation | Latency | Notes |
|-----------|---------|-------|
| Training (50 files) | ~2s | AST parse + N-gram training |
| Code completion | <5ms | Uses AST for context, N-gram fallback |
| Find callers | <2ms | Dict lookup in reverse call graph |
| Find class | <0.5ms | Direct dict lookup |
| Natural language query | <5ms | Regex pattern matching + lookup |
| Save (JSON) | ~1s | 55MB model file |
| Load (JSON) | ~1s | Parsing and reconstruction |

---

## Files Changed

### Created
- `cortical/spark/tokenizer.py`
- `cortical/spark/ast_index.py`
- `cortical/spark/intelligence.py`
- `tests/unit/test_code_tokenizer.py`
- `tests/unit/test_ast_index.py`
- `tests/unit/test_spark_intelligence.py`
- `tests/integration/test_spark_intelligence_integration.py`
- `tests/performance/test_spark_intelligence_perf.py`

### Modified
- `cortical/spark/__init__.py` - Added new exports
- `cortical/spark/ngram.py` - Added `_cached_frequent_words` and `finalize()`
- `scripts/spark_code_intelligence.py` - Refactored to use modules
- `.gitignore` - Added `.spark_intelligence_model.json`

---

## SparkSLM Components (Preserved)

The original SparkSLM is untouched and fully available:
- `NGramModel` - Statistical word prediction
- `SparkPredictor` - Unified facade
- `AnomalyDetector` - Prompt injection detection
- `AlignmentIndex` - User definitions/patterns
- `QualityEvaluator` - Quality metrics

SparkCodeIntelligence is a **separate** system that happens to use NGramModel internally.

---

## Commands for Next Session

```bash
# Train on codebase
python scripts/spark_code_intelligence.py train -v

# Query examples
python scripts/spark_code_intelligence.py find-callers compute_pagerank
python scripts/spark_code_intelligence.py find-class NGramModel
python scripts/spark_code_intelligence.py query "what calls process_document"

# Run tests
python -m pytest tests/unit/test_code_tokenizer.py tests/unit/test_ast_index.py tests/unit/test_spark_intelligence.py -v

# Benchmarks
python scripts/spark_code_intelligence.py benchmark
```

---

## Commit History

1. `feat(spark): Add benchmarks and coverage estimator to SparkCodeIntelligence`
2. `refactor(spark): Convert SparkCodeIntelligence to JSON format`
3. `refactor(spark): Split SparkCodeIntelligence into modular packages` (118 tests)

---

## Open Questions for Future Sessions

1. Should SparkCodeIntelligence integrate with CorticalTextProcessor or remain separate?
2. Should we add incremental indexing (only re-index changed files)?
3. Would a language server protocol (LSP) wrapper be useful?
4. Should call graph include external library calls?

---

*This session built a powerful code intelligence tool. The modular design makes it easy to integrate with existing skills and workflows.*
