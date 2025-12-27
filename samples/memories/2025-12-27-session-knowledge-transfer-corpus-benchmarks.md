# Knowledge Transfer: Corpus Generation, Benchmarks, and ML Hooks

**Date:** 2025-12-27
**Session:** claude/continue-prism-incident-Zs1oM
**Previous Session:** claude/accept-handoff-ctrSI (PRISM model incident recovery)
**Tags:** `corpus`, `benchmarks`, `slm-training`, `ml-hooks`, `vocabulary`, `data-quality`

---

## Executive Summary

This session continued from the PRISM model incident recovery. We generated a fresh training corpus (35,617 patterns), created comprehensive benchmark tests (35 tests), trained a v2 model (not promoted due to regression), analyzed vocabulary coverage (99.8%), and added a PostToolUse hook for richer ML data collection.

---

## Session Context

### Starting State
- **Previous incident:** PRISM model was accidentally overwritten (15,814 → 329 vocab)
- **Recovery:** Model restored from git, safeguards added (`--dry-run`, backups)
- **ML data:** 2,421 commits, 186 sessions, but only 19 chats (gap identified)
- **Task completion:** 82.5% (137/166 tasks)

### Ending State
- **Corpus:** 35,617 patterns generated from codebase
- **Tests:** 35 benchmark tests created and passing
- **v2 Model:** Trained but not promoted (file_location regressed)
- **ML hooks:** PostToolUse added for tool capture
- **Task completion:** 84.3% (140/166 tasks)

---

## 1. Corpus Generation

### Command
```bash
python -m benchmarks.codebase_slm.generate_corpus --full
```

### Output Statistics
| Metric | Count |
|--------|-------|
| Python files processed | 149 |
| Markdown files processed | 166 + 107 (samples) |
| Training patterns | 35,617 |
| Functions extracted | 2,553 |
| Classes extracted | 403 |
| Imports extracted | 985 |
| With docstrings | 2,418 |

### Files Generated
```
benchmarks/codebase_slm/corpus/
├── code_patterns.json      (2.2 MB) - Extracted code structures
├── doc_patterns.json       (10.9 MB) - Extracted documentation
├── meta_patterns.json      (352 KB) - GoT/task metadata
├── training_corpus.txt     (4.3 MB) - Raw training text
└── training_patterns.jsonl (10.8 MB) - Structured patterns
```

**Note:** The `corpus/` directory is gitignored because it's regeneratable from the codebase. Always run `generate_corpus.py --full` before training.

---

## 2. SparkSLM Training

### Training Command
```bash
# Safe: dry-run first
python -m benchmarks.codebase_slm.train_augmented --dry-run

# Training to new file (don't overwrite production model)
python -m benchmarks.codebase_slm.train_augmented --output models/prism_augmented_v2.json
```

### v2 Model Statistics
| Metric | Current Model | v2 Model | Delta |
|--------|---------------|----------|-------|
| Vocabulary | 15,814 | 15,910 | +96 |
| Documents | 37,318 | 37,711 | +393 |
| Tokens | 649,107 | 654,560 | +5,453 |

### Benchmark Scores
| Category | Current | v2 | Change |
|----------|---------|-----|--------|
| concept | 0% | 17% | +17% |
| file_location | 88% | 50% | **-38%** |
| hierarchical | 0% | 0% | 0% |
| **Overall** | 29% | 25% | -4% |

### Decision: D-20251227-033410
**Keep current PRISM model as production.**

**Rationale:** The v2 model regressed on file_location (88% → 50%), which is the primary use case for the SLM. The concept improvement (0% → 17%) doesn't justify the file_location regression.

**Files:**
- Production: `benchmarks/codebase_slm/models/prism_augmented.json`
- Reference: `benchmarks/codebase_slm/models/prism_augmented_v2.json`

---

## 3. Benchmark Tests Created

### Test Files
| File | Tests | Purpose |
|------|-------|---------|
| `tests/benchmarks/test_corpus_quality.py` | 19 | Validate training data quality |
| `tests/benchmarks/test_slm_regression.py` | 16 | Detect model regressions |
| `tests/benchmarks/README.md` | - | Documentation |

### Test Categories

#### Corpus Quality Tests (`test_corpus_quality.py`)
```
TestPatternDistribution (4 tests)
├── test_balanced_pattern_types      # No type > 90%
├── test_no_source_dominance         # No source > 50%
├── test_pattern_type_variety        # All 4 types present
└── test_reasonable_pattern_lengths  # Not too short/long

TestVocabularyCoverage (4 tests)
├── test_project_terms_present       # minicolumn, pagerank, etc.
├── test_module_names_present        # processor, query, analysis
├── test_code_keywords_present       # def, class, import
└── test_vocabulary_size_adequate    # >= 10,000 terms

TestQualityMetrics (4 tests)
├── test_qa_patterns_complete        # Has question + answer
├── test_source_files_exist          # Paths are valid
├── test_confidence_scores_valid     # 0.0 <= score <= 1.0
└── test_no_missing_critical_fields  # Required fields present

TestRegressionBaseline (3 tests)
├── test_pattern_count_baseline      # >= 30,000 patterns
├── test_pattern_type_distribution_stable
└── test_vocabulary_size_stable      # >= 10,000

TestDataIntegrity (3 tests)
├── test_no_duplicate_patterns       # <= 25% duplicates
├── test_pattern_metadata_consistency
└── test_corpus_files_exist
```

#### SLM Regression Tests (`test_slm_regression.py`)
```
TestModelSize (3 tests)
├── test_prism_vocab_size            # >= 15,000
├── test_prism_document_count        # >= 35,000
└── test_prism_token_count_reasonable # >= 500,000

TestTrainingSafeguards (4 tests)
├── test_dry_run_available           # --dry-run flag exists
├── test_output_flag_available       # --output flag exists
├── test_backup_mechanism_documented
└── test_corpus_check_exists         # Warns on missing corpus

TestCorpusRegeneration (3 tests)
├── test_generate_script_exists
├── test_generate_script_has_full_mode
└── test_corpus_patterns_file_exists

TestBenchmarkStability (4 tests)
├── test_benchmark_suite_exists
├── test_benchmark_suite_importable
├── test_benchmark_has_categories
└── test_benchmark_has_help_flag

TestFullPipeline (2 tests)
├── test_pipeline_documentation_exists
└── test_model_provenance_structure
```

### Running Tests
```bash
# All benchmark tests
python -m pytest tests/benchmarks/ -v

# Quick run
python -m pytest tests/benchmarks/ -q

# With coverage
python -m coverage run -m pytest tests/benchmarks/ && python -m coverage report
```

### Current Baselines (captured in tests)
- Pattern count: >= 30,000 (actual: 35,617)
- Vocabulary size: >= 10,000 (actual: 51,372 tokens in corpus)
- Model vocab: >= 15,000 (actual: 15,814)
- Model docs: >= 35,000 (actual: 37,318)

---

## 4. Vocabulary Coverage Analysis

### Summary
| Metric | Value |
|--------|-------|
| Model vocabulary | 15,814 terms |
| Corpus vocabulary | 15,932 terms |
| Coverage | **99.8%** |
| Terms in both | 15,789 |
| Model-only terms | 25 (mostly git hashes) |
| Corpus-only terms | 143 (candidates for expansion) |

### Critical Terms Check (All Present ✅)
| Category | Terms |
|----------|-------|
| **Core** | minicolumn, pagerank, tfidf, louvain, cortical, layer |
| **Reasoning** | thought, graph, cognitive, loop, woven, mind, prism, loom |
| **GoT** | task, sprint, decision, edge, handoff |
| **Query** | expansion, search, passage, intent, retrieval |

### Top Vocabulary Expansion Candidates
1. `n²` (72 occurrences) - Mathematical notation
2. `train_augmented` (14) - Training function
3. `σ` (12) - Greek letter (statistics)
4. `prism_augmented` (9) - Model variant
5. `data_augmentation` (8) - Training technique

### Recommendation
Vocabulary is healthy. Consider adding mathematical notation (n², σ, α, etc.) if they appear in real queries.

---

## 5. ML Hook Configuration

### Before
```json
{
  "hooks": {
    "SessionStart": [...],
    "Stop": [...]
  }
}
```

### After
```json
{
  "hooks": {
    "SessionStart": [...],
    "PostToolUse": [
      {
        "type": "command",
        "command": "bash scripts/ml-tool-capture-hook.sh"
      }
    ],
    "Stop": [...]
  }
}
```

### New Hook: `ml-tool-capture-hook.sh`
**Purpose:** Capture individual tool invocations during sessions for ML training.

**Features:**
- Runs after each tool execution
- Logs tool name, input, and output
- Async write to avoid blocking Claude Code (<100ms)
- Truncates outputs >10KB to prevent bloat
- Batches to per-session files: `.git-ml/tool_uses/{session_id}.jsonl`

**Data Flow:**
```
Tool invocation
    ↓
PostToolUse hook
    ↓
.git-ml/tool_uses/{session}.jsonl (per-session)
    ↓
Session end (Stop hook)
    ↓
.git-ml/actions/tool_uses.jsonl (merged)
```

### Why This Matters
Before: 2,421 commits, 186 sessions, but only 19 chats
After: Will capture hundreds of tool uses per session

This addresses the training data gap - tool interactions are rich data for:
- Learning tool selection patterns
- Understanding input/output relationships
- Training code generation models

---

## 6. Tasks Completed

| Task ID | Title | Notes |
|---------|-------|-------|
| T-20251226-194614-9a6d95fe | Sprint completion: S-021 and S-022 | Verified status |
| T-20251226-223329-dd4adee2 | S20.7: Corpus builder | Generated 35,617 patterns |
| T-20251226-223332-4c5b2c73 | S20.8: Validation and sampling | Created benchmark tests |

---

## 7. Commits Made

| Hash | Message |
|------|---------|
| `fefa9e05` | feat(benchmarks): Add corpus quality and SLM regression tests |
| `90ba2570` | chore(slm): Add v2 model trained on fresh corpus |
| `6a696417` | feat(ml): Add PostToolUse hook for comprehensive tool capture |

---

## 8. Key Commands Reference

### Corpus Management
```bash
# Generate corpus (required before training)
python -m benchmarks.codebase_slm.generate_corpus --full

# Check corpus size
wc -l benchmarks/codebase_slm/corpus/training_patterns.jsonl
```

### Model Training
```bash
# ALWAYS dry-run first
python -m benchmarks.codebase_slm.train_augmented --dry-run

# Train to new file (safe)
python -m benchmarks.codebase_slm.train_augmented --output models/experimental.json

# Check current model
python -c "import json; m=json.load(open('benchmarks/codebase_slm/models/prism_augmented.json')); print(f'Vocab: {len(m[\"vocab\"])}, Docs: {m[\"total_documents\"]}')"
```

### Benchmark Tests
```bash
# Run all benchmarks
python -m pytest tests/benchmarks/ -v

# Run specific category
python -m pytest tests/benchmarks/test_corpus_quality.py::TestVocabularyCoverage -v

# View quality summary
python -m pytest tests/benchmarks/test_corpus_quality.py::test_corpus_summary -v -s
```

### ML Data Collection
```bash
# Check stats
python scripts/ml_data_collector.py stats

# Estimate training readiness
python scripts/ml_data_collector.py estimate

# File prediction model
python scripts/ml_file_prediction.py train
python scripts/ml_file_prediction.py evaluate --split 0.2
```

---

## 9. Remaining Work

### High Priority Pending Tasks
| Task ID | Title | Priority |
|---------|-------|----------|
| T-20251226-144757-f13543d6 | Investigate baseline_drift benchmark | high |
| T-20251226-223323-3a111f16 | S20.4: Q&A pattern generator | high |
| T-20251226-223319-fe8ac9b9 | S20.2: Doc extractor improvements | high |

### Sprint Status
- **S-021 (Training Pipeline):** 95% complete - corpus generation done
- **S-022 (Benchmarks & Evaluation):** 86% complete - tests added

### Future Improvements
1. **Corpus balancing:** Current corpus is 86.4% Q&A patterns - may need more variety
2. **File location training:** Need patterns that improve file_location accuracy
3. **Greek letter handling:** Add mathematical notation to vocabulary
4. **UserPromptSubmit hook:** Could add for query context capture

---

## 10. Files Modified/Created This Session

### Created
- `tests/benchmarks/test_corpus_quality.py` (418 lines)
- `tests/benchmarks/test_slm_regression.py` (500+ lines)
- `tests/benchmarks/README.md`
- `tests/benchmarks/__init__.py`
- `scripts/ml-tool-capture-hook.sh`
- `benchmarks/codebase_slm/models/prism_augmented_v2.json` (13 MB)
- `benchmarks/codebase_slm/corpus/*` (gitignored, regeneratable)

### Modified
- `.claude/settings.local.json` - Added PostToolUse hook
- `scripts/ml-session-capture-hook.sh` - Added tool log processing

### GoT State
- Decision: D-20251227-033410 (keep current model)
- Tasks completed: 3

---

## 11. Lessons Learned

### 1. Benchmark Scores Can Be Misleading
The v2 model showed "17% improvement" on concept but "38% regression" on file_location. Always evaluate the **primary use case** (file_location for code search), not just overall scores.

### 2. Corpus Quality Matters More Than Size
35,617 patterns with 86% Q&A concentration produces different results than a balanced corpus. Pattern type distribution affects model behavior.

### 3. Safeguards Prevent Incidents
The `--dry-run` flag added after the model incident caught the file_location regression before we would have overwritten production.

### 4. Tool Capture Fills Data Gap
2,421 commits with only 19 chats was a red flag. Adding PostToolUse hook will capture orders of magnitude more interaction data.

---

## 12. Quick Start for Next Session

```bash
# 1. Check GoT state
python scripts/got_utils.py dashboard

# 2. Check ML data
python scripts/ml_data_collector.py stats

# 3. Run benchmark tests
python -m pytest tests/benchmarks/ -q

# 4. Check model status
wc -l benchmarks/codebase_slm/models/prism_augmented.json

# 5. If corpus needed, regenerate
python -m benchmarks.codebase_slm.generate_corpus --full
```

---

## Related Documents

- Previous session: `samples/memories/2025-12-27-knowledge-transfer-prism-model-incident.md`
- PRISM training docs: `docs/ml-training-best-practices.md`
- Benchmark thresholds: `docs/ml-milestone-thresholds.md`
- Task knowledge base: `docs/task-knowledge-base-woven-prism.md`
