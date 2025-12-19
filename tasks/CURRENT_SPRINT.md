# Parallel Sprint Tracker

> **Multi-Thread Mode Enabled**
>
> Sprints 6, 7, and 8 are designed for parallel execution across different threads.
> Each sprint works on independent areas to minimize merge conflicts.

---

# Active Sprints

## Sprint 6: TestExpert Activation ✅
**Sprint ID:** sprint-006-test-expert
**Epic:** Hubris MoE System (efba)
**Status:** Complete ✅
**Session:** 0c9WR
**Isolation:** `scripts/hubris/experts/test_expert.py`, `tests/hubris/`

### Goals
- [x] Wire TestExpert to actual test outcomes (pass/fail signals)
- [x] Add `suggest-tests` command to CLI
- [x] Integrate with CI results from `.git-ml/commits/` (test_passed field)
- [x] Create test selection accuracy metrics
- [x] Add TestExpert to feedback loop (post-test-run hook)

### Key Files
- `scripts/hubris/experts/test_expert.py` - Core expert logic
- `scripts/hubris_cli.py` - Add suggest-tests command
- `scripts/hubris/feedback_collector.py` - Add test outcome signals

### Success Criteria
- TestExpert predicts relevant tests for code changes
- Accuracy tracked via calibration system
- CLI command shows test suggestions with confidence

### Notes
- TestExpert exists but isn't wired to real outcomes
- CI already captures test results in commit data
- Low risk of conflicts with Sprint 7 or 8

---

## Sprint 7: RefactorExpert (New Expert Type)
**Sprint ID:** sprint-007-refactor-expert
**Epic:** Hubris MoE System (efba)
**Status:** Available 🟢
**Isolation:** `scripts/hubris/experts/refactor_expert.py` (new file)

### Goals
- [ ] Create RefactorExpert class inheriting from MicroExpert
- [ ] Define refactoring signal types (extract, inline, rename, move)
- [ ] Train on commit history patterns (commits with "refactor:" prefix)
- [ ] Add refactoring detection heuristics (duplicate code, long methods)
- [ ] Register in ExpertConsolidator
- [ ] Add `suggest-refactor` CLI command

### Key Files (New)
- `scripts/hubris/experts/refactor_expert.py` - New expert
- `scripts/hubris/refactor_patterns.py` - Pattern definitions (optional)

### Key Files (Modify)
- `scripts/hubris/expert_consolidator.py` - Register new expert
- `scripts/hubris_cli.py` - Add suggest-refactor command

### Success Criteria
- RefactorExpert can predict files needing refactoring
- Integrates with existing credit system
- Documented in README

### Notes
- Self-contained new expert - minimal conflicts
- Can reuse FileExpert's TF-IDF infrastructure
- Training data: commits with "refactor:" in message

---

## Sprint 8: Core Library Performance
**Sprint ID:** sprint-008-core-performance
**Epic:** Cortical Text Processor (core)
**Status:** Available 🟢
**Isolation:** `cortical/` directory

### Goals
- [ ] Profile `compute_all()` phases with real corpus
- [ ] Optimize slowest phase identified by profiling
- [ ] Address coverage debt: `cortical/query/analogy.py` (3%)
- [ ] Address coverage debt: `cortical/gaps.py` (9%)
- [ ] Add performance regression tests
- [ ] Update performance benchmarks in docs

### Key Files
- `cortical/processor/compute.py` - Optimization target
- `cortical/query/analogy.py` - Coverage improvement
- `cortical/gaps.py` - Coverage improvement
- `tests/performance/` - Add benchmarks
- `scripts/profile_full_analysis.py` - Profiling tool

### Success Criteria
- No performance regression (verify via benchmarks)
- Coverage improved on target modules
- Profiling data documented

### Notes
- Independent of Hubris work (different directory)
- Profile before optimizing - follow CLAUDE.md principles
- Small, focused improvements over large rewrites

---

## Sprint 9: Projects Architecture ✅
**Sprint ID:** sprint-009-projects-arch
**Epic:** Cortical Text Processor (core)
**Status:** Complete ✅
**Isolation:** `cortical/projects/` (new), `tests/`, `.github/workflows/`
**Session:** ac6d

### Goals
- [x] Create `cortical/projects/` directory structure
- [x] Move MCP server to `cortical/projects/mcp/`
- [x] Move proto to `cortical/projects/proto/` (already removed, placeholder added)
- [x] Update pyproject.toml with per-project dependencies
- [x] Update CI to isolate project tests
- [x] Document Projects architecture pattern
- [x] Verify all tests pass (4,769 core + 38 MCP = 4,807 total)
- [x] Create knowledge transfer document

### Task Dependency Graph (for Sub-Agents)

```
                    ┌─────────────────────┐
                    │ T-001: Create       │
                    │ directory structure │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ T-002: Move MCP         │     │ T-003: Move proto       │
│ (can run in parallel)   │     │ (can run in parallel)   │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │                               │
            └───────────────┬───────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
┌─────────────────────────┐ ┌─────────────────────────┐
│ T-004: Update           │ │ T-006: Document         │
│ pyproject.toml          │ │ architecture            │
└───────────┬─────────────┘ └─────────────────────────┘
            │
            ▼
┌─────────────────────────┐
│ T-005: Update CI        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ T-007: Verify tests     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ T-008: Knowledge        │
│ transfer document       │
└─────────────────────────┘
```

### Parallelization Opportunities

| Phase | Tasks | Sub-Agents |
|-------|-------|------------|
| 1. Setup | T-001 | 1 agent |
| 2. Move | T-002, T-003 | 2 parallel agents |
| 3. Config | T-004, T-006 | 2 parallel agents |
| 4. CI | T-005 | 1 agent |
| 5. Verify | T-007 | 1 agent |
| 6. Document | T-008 | 1 agent |

### Key Files

**New Files:**
- `cortical/projects/__init__.py`
- `cortical/projects/mcp/__init__.py`
- `cortical/projects/mcp/server.py`
- `cortical/projects/mcp/tests/`
- `cortical/projects/proto/__init__.py`
- `docs/projects-architecture.md`

**Modified Files:**
- `pyproject.toml` - Add project dependency groups
- `.github/workflows/ci.yml` - Isolate project tests
- `CLAUDE.md` - Add Projects section

**Removed Files:**
- `cortical/mcp_server.py` (moved)
- `tests/test_mcp_server.py` (moved)

### Success Criteria
- Core tests run independently of project tests
- MCP tests only run when `[mcp]` dependencies installed
- CI shows separate status for core vs projects
- Zero-dependency core library maintained
- All existing tests pass

### Notes
- This sprint addresses the recurring MCP test failures
- Creates pattern for future optional features
- Maintains backward compatibility via `cortical.projects.mcp`
- Sub-agents should coordinate via task file updates

---

# Parallel Work Guidelines

## Thread Assignment

| Thread | Sprint | Primary Focus |
|--------|--------|---------------|
| Thread A | Sprint 6 | TestExpert wiring |
| Thread B | Sprint 7 | RefactorExpert creation |
| Thread C | Sprint 8 | Core library performance |

## Conflict Avoidance

Each sprint is designed with isolated file sets:

```
Sprint 6 (TestExpert):
  └── scripts/hubris/experts/test_expert.py
  └── Tests for test selection

Sprint 7 (RefactorExpert):
  └── scripts/hubris/experts/refactor_expert.py (NEW)
  └── scripts/hubris/refactor_patterns.py (NEW)

Sprint 8 (Core Performance):
  └── cortical/**
  └── tests/performance/
```

## Shared Files (Coordinate Changes)

These files may be touched by multiple sprints - coordinate:

- `scripts/hubris_cli.py` - Sprint 6 and 7 both add commands
- `scripts/hubris/expert_consolidator.py` - Sprint 7 registers new expert
- `scripts/hubris/README.md` - All sprints update docs

**Strategy:** Add new functions/commands at end of file to minimize merge conflicts.

## Branch Naming

```
Sprint 6: claude/sprint-006-test-expert-{session-id}
Sprint 7: claude/sprint-007-refactor-expert-{session-id}
Sprint 8: claude/sprint-008-core-performance-{session-id}
```

## Documentation Maintenance

Each sprint should update:
1. **README** for the component being modified
2. **CURRENT_SPRINT.md** - Mark goals complete
3. **Memory document** if significant learnings

---

# Previous Sprints

## Sprint 5: UX & Documentation (Complete ✅)
**Dates:** 2025-12-18
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ Cold-start UX fix with ML fallback (ce81d6d)
- ✅ Calibration command for CLI (8ac951c)
- ✅ Hubris documentation update (e2fc096)

## Sprint 4: Meta-Learning (Complete ✅)
**Dates:** 2025-12-18
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ Wire feedback_collector to live git hooks
- ✅ Add EXPERIMENTAL banner to predictions
- ✅ Create sprint tracking persistence
- ✅ Add git lock detection
- ✅ Exception handling audit (9 fixes)
- ✅ Calibration tracking Phase 1+2

## Sprint 3: MoE Integration (Complete ✅)
**Dates:** 2025-12-15 to 2025-12-17
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ CLI interface (hubris_cli.py)
- ✅ Feedback collector
- ✅ README documentation
- ✅ Integration tests

## Sprint 2: Credit System (Complete ✅)
**Dates:** 2025-12-13 to 2025-12-14
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ CreditAccount, CreditLedger
- ✅ ValueSignal, ValueAttributor
- ✅ CreditRouter, Staking

## Sprint 1: Expert Foundation (Complete ✅)
**Dates:** 2025-12-10 to 2025-12-12
**Epic:** Hubris MoE System (efba)

Completed:
- ✅ MicroExpert base class
- ✅ FileExpert, TestExpert, ErrorDiagnosisExpert, EpisodeExpert
- ✅ ExpertConsolidator and routing

---

# Epics

## Active: Hubris MoE (efba)
**Started:** 2025-12-10
**Status:** Phase 5 - Expert Expansion

### Phases:
- **Phase 1:** Expert foundation ✅ (Sprint 1)
- **Phase 2:** Credit system ✅ (Sprint 2)
- **Phase 3:** Integration ✅ (Sprint 3)
- **Phase 4:** Meta-learning ✅ (Sprint 4-5)
- **Phase 5:** Expert expansion 🔄 (Sprint 6-7)
  - TestExpert activation
  - RefactorExpert creation
  - Additional expert types

## Active: Cortical Core (core)
**Status:** Maintenance

### Focus Areas:
- Performance optimization (Sprint 8)
- Coverage improvement
- Search quality

---

# Sprint Selection Guide

**For New Thread:**

1. Read this file to see available sprints
2. Pick an **Available 🟢** sprint
3. Create branch: `claude/{sprint-id}-{session-id}`
4. Update sprint status to **In Progress 🟡**
5. Work on goals
6. Mark complete, update status to **Complete ✅**

**Sprint Status Key:**
- 🟢 Available - Ready to start
- 🟡 In Progress - Being worked on
- ✅ Complete - All goals done
- 🔴 Blocked - Waiting on dependency

---

# Future Sprints (Backlog)

## Sprint 10: DocumentationExpert
Create expert for documentation suggestions.

## Sprint 11: SecurityExpert
Create expert for security vulnerability detection.

## Sprint 12: PerformanceExpert
Create expert for performance optimization suggestions.

## Sprint 13: DependencyExpert
Create expert for dependency update recommendations.

## Sprint 14: CLI Project Migration
Move CLI wrapper to projects/cli if it becomes problematic.

---

# Stats

| Sprint | Duration | Status | Epic |
|--------|----------|--------|------|
| Sprint 1 | 3 days | ✅ | Hubris MoE |
| Sprint 2 | 2 days | ✅ | Hubris MoE |
| Sprint 3 | 3 days | ✅ | Hubris MoE |
| Sprint 4 | 1 day | ✅ | Hubris MoE |
| Sprint 5 | 1 day | ✅ | Hubris MoE |
| Sprint 6 | - | 🟢 | Hubris MoE |
| Sprint 7 | - | 🟢 | Hubris MoE |
| Sprint 8 | - | 🟢 | Core |
| Sprint 9 | 1 day | ✅ | Core (Projects) |

---

## Sprint 15: Search Quality Fundamentals
**Sprint ID:** sprint-015-search-quality
**Epic:** NLU Enhancement (nlu)
**Status:** Available 🟢
**Isolation:** `cortical/query/`, `cortical/code_concepts.py`

### Context
Investigation in `samples/memories/2025-12-14-search-relevance-investigation.md` identified root causes of poor search results. This sprint implements the fixes.

### Goals
- [ ] Enable code stop word filtering by default in `find_documents_for_query()`
- [ ] Weight lateral expansion by TF-IDF, not raw co-occurrence count
- [ ] Apply test file penalty (0.8) by default in basic search
- [ ] Add security concept group to `code_concepts.py`
- [ ] Add domain-specific concept groups (ML, database, frontend)
- [ ] Update tests for new default behaviors

### Key Files
- `cortical/query/search.py:54-59` - Add `filter_code_stop_words=True` default
- `cortical/query/expansion.py:164` - Incorporate TF-IDF: `score = weight * neighbor.pagerank * neighbor.tfidf * 0.6`
- `cortical/query/search.py` - Integrate test file penalty
- `cortical/code_concepts.py` - Add new concept groups

### Success Criteria
- "security test fuzzing" returns security code, not staleness tests
- Ubiquitous terms (def, self, return) don't dominate expansions
- Test files rank lower unless explicitly searching for tests
- Dog-food: Improved results for real queries

### Tasks (Detailed)
```
T-NLU-001: Enable code stop word filtering by default
T-NLU-002: Weight lateral expansion by TF-IDF
T-NLU-003: Apply test file penalty in basic search
T-NLU-004: Add security concept group (fuzzing, injection, xss, csrf, sanitize)
T-NLU-005: Add ML concept group (training, inference, model, epoch, batch)
T-NLU-006: Add database concept group (query, schema, migration, index)
T-NLU-007: Add frontend concept group (component, render, state, props)
T-NLU-008: Update unit tests for new defaults
T-NLU-009: Dog-food validation with real queries
```

---

## Sprint 16: Enhanced NLU Queries
**Sprint ID:** sprint-016-enhanced-nlu
**Epic:** NLU Enhancement (nlu)
**Status:** Available 🟢
**Isolation:** `cortical/query/intent.py`, `cortical/query/nlu/` (new)

### Goals
- [ ] Implement negation parsing ("find X not in tests")
- [ ] Implement scope parsing ("config in core module")
- [ ] Implement temporal parsing ("recent changes to auth")
- [ ] Add explainable search results (show WHY matched)
- [ ] Add query reformulation suggestions
- [ ] Integrate enhanced NLU into main search API

### Key Files (New)
- `cortical/query/nlu/__init__.py` - NLU package
- `cortical/query/nlu/parser.py` - Enhanced query parser
- `cortical/query/nlu/explainer.py` - Result explanation generator

### Key Files (Modify)
- `cortical/query/intent.py` - Extend with negation/scope/temporal
- `cortical/processor/query_api.py` - Add `search_explained()` method
- `nlu_showcase.py` - Update with production implementation

### Success Criteria
- "authentication not tests" excludes test files
- "config in core" scopes to core module
- Each result shows match explanation
- Interactive mode works with enhanced queries

### Tasks (Detailed)
```
T-NLU-010: Implement negation parser (not, without, except, exclude)
T-NLU-011: Implement scope parser (in, within, from module/file)
T-NLU-012: Implement temporal parser (recent, old, today, yesterday)
T-NLU-013: Create ExplainedResult dataclass with match reasons
T-NLU-014: Add search_explained() to CorticalTextProcessor
T-NLU-015: Add query reformulation suggestions
T-NLU-016: Integrate into nlu_showcase.py
T-NLU-017: Add unit tests for enhanced parsing
T-NLU-018: Documentation and examples
```

---

## Sprint 17: SparkSLM - Statistical First-Blitz Predictor
**Sprint ID:** sprint-017-spark-slm
**Epic:** NLU Enhancement (nlu)
**Status:** Available 🟢
**Isolation:** `cortical/spark/` (new package)

### Concept
SparkSLM is NOT a neural language model. It's a fast statistical predictor that provides "first blitz thoughts" - quick, rough predictions to prime thinking before more sophisticated analysis.

**Inspired by:** Andrej Karpathy's llm.c (pure C LLM training without tensor libraries). While llm.c still needs CUDA for real training, it proves you can build from scratch. SparkSLM takes this further - pure Python, zero dependencies, statistical only.

### Architecture
```
SparkSLM = Statistical Language Model for Spark Ideas
├── N-gram predictor (bigram/trigram transition probabilities)
├── Topic classifier (TF-IDF based fast categorization)
├── Keyword extractor (rapid salient term identification)
└── Prompt primer (initial expansion before full search)
```

### Use Cases
1. **Prompt Injection Detection**: Statistical anomaly detection on input patterns
2. **First-Blitz Ideas**: Quick topic classification before deeper search
3. **Auto-Complete**: N-gram based next-word suggestions
4. **Query Priming**: Rapid keyword extraction to seed query expansion

### Goals
- [ ] Create `cortical/spark/` package structure
- [ ] Implement N-gram model (bigram/trigram) with Laplace smoothing
- [ ] Implement fast topic classifier using TF-IDF centroids
- [ ] Implement keyword extractor using TextRank variant
- [ ] Create SparkPredictor class with `prime()` method
- [ ] Add statistical anomaly detection for prompt injection
- [ ] Integrate with query expansion as optional primer

### Key Files (New)
- `cortical/spark/__init__.py`
- `cortical/spark/ngram.py` - N-gram language model
- `cortical/spark/classifier.py` - Fast topic classification
- `cortical/spark/keywords.py` - Keyword extraction
- `cortical/spark/predictor.py` - SparkPredictor main class
- `cortical/spark/anomaly.py` - Prompt injection detection

### Success Criteria
- N-gram model trained on corpus vocabulary
- Topic classification in <10ms for typical queries
- Prompt injection detection with reasonable precision
- Query priming shows measurable improvement in result relevance

### Honest Limitations
- NOT a true language model - no semantic understanding
- Cannot generate coherent text (just statistical completions)
- Anomaly detection is pattern-based, not semantic
- Useful as "spark" primer, not replacement for real search

### Tasks (Detailed)
```
T-SPARK-001: Create cortical/spark/ package structure
T-SPARK-002: Implement NGramModel class with train/predict
T-SPARK-003: Implement FastTopicClassifier using TF-IDF centroids
T-SPARK-004: Implement KeywordExtractor using TextRank-lite
T-SPARK-005: Create SparkPredictor facade class
T-SPARK-006: Implement AnomalyDetector for prompt injection
T-SPARK-007: Integrate SparkPredictor with query expansion
T-SPARK-008: Add training script for SparkSLM
T-SPARK-009: Benchmark speed and accuracy
T-SPARK-010: Documentation and examples
```

---

## Sprint 18: Procedural + Learned Reasoning
**Sprint ID:** sprint-018-reasoning
**Epic:** NLU Enhancement (nlu)
**Status:** Available 🟢
**Isolation:** `cortical/reasoning/` (new package)

### Concept
Two-track reasoning system:
1. **Procedural**: Hard-coded rules and patterns that always work
2. **Learned**: Statistical patterns that improve with usage

### Architecture
```
Reasoning System
├── Procedural (Rule-Based)
│   ├── Inference rules (if A→B and B→C, then A→C)
│   ├── Type constraints (class X extends Y → X is-a Y)
│   ├── Negation handling (not A and A→B → not B)
│   └── Scope resolution (in module X → constrain to X/*)
│
└── Learned (Statistical)
    ├── Co-occurrence patterns (A often with B → A related B)
    ├── Session patterns (user searched X then Y → X→Y chain)
    ├── Feedback integration (user clicked result → boost pattern)
    └── Temporal patterns (recently modified → relevance boost)
```

### Goals
- [ ] Create `cortical/reasoning/` package
- [ ] Implement procedural inference engine
- [ ] Implement learned pattern store
- [ ] Create reasoning API: `why(A, B)` → explanation chain
- [ ] Implement feedback loop for learning
- [ ] Add graph traversal for "path from A to B"
- [ ] Integrate with search for reasoning-enhanced results

### Key Files (New)
- `cortical/reasoning/__init__.py`
- `cortical/reasoning/procedural.py` - Rule-based inference
- `cortical/reasoning/learned.py` - Pattern learning
- `cortical/reasoning/explainer.py` - Chain explanations
- `cortical/reasoning/feedback.py` - Learning from usage

### Success Criteria
- `why("authentication", "security")` → explanation chain
- `path("login", "session")` → concept path
- Session patterns captured and used
- Reasoning integrates with enhanced search

### Tasks (Detailed)
```
T-REASON-001: Create cortical/reasoning/ package
T-REASON-002: Implement InferenceRule base class
T-REASON-003: Implement procedural rules (transitivity, type, negation)
T-REASON-004: Implement PatternStore for learned patterns
T-REASON-005: Implement why() explanation generator
T-REASON-006: Implement path() graph traversal
T-REASON-007: Create feedback integration hooks
T-REASON-008: Session pattern capture
T-REASON-009: Integrate with search_explained()
T-REASON-010: Documentation and examples
```

---

## Sprint 19: Sample Generation for Knowledge Base
**Sprint ID:** sprint-019-samples
**Epic:** NLU Enhancement (nlu)
**Status:** Available 🟢
**Isolation:** `samples/`, `scripts/generate_samples.py` (new)

### Context
The showcase identified underrepresented domains:
- API: 2 docs
- Security: 11 docs  
- Database: 11 docs
- Frontend: 16 docs
- Testing: 4 docs

### Goals
- [ ] Create sample generation script
- [ ] Generate API documentation samples
- [ ] Generate security documentation samples
- [ ] Generate database documentation samples
- [ ] Generate architecture decision records (ADRs)
- [ ] Generate "how does X work?" tutorial documents
- [ ] Generate glossary/definition documents

### Sample Types Needed
```
samples/
├── tutorials/          # "How does X work?" docs
│   ├── how-search-works.md
│   ├── how-indexing-works.md
│   └── how-query-expansion-works.md
├── architecture/       # "Where is X?" location docs
│   ├── module-map.md
│   ├── data-flow.md
│   └── file-responsibilities.md
├── glossary/           # "What is X?" definition docs
│   ├── ir-terms.md
│   ├── graph-terms.md
│   └── nlp-terms.md
├── decisions/          # "Why X?" rationale docs (ADRs)
│   ├── adr-001-zero-dependencies.md
│   ├── adr-002-cortical-metaphor.md
│   └── adr-003-layered-architecture.md
├── security/           # Security-focused docs
├── api/                # API documentation
├── database/           # Database patterns
└── frontend/           # Frontend patterns
```

### Key Files (New)
- `scripts/generate_samples.py` - Sample generation script
- Various sample documents

### Success Criteria
- Domain coverage balanced (no domain <10% representation)
- All query types answerable (where/what/how/why)
- Showcase shows improved knowledge gaps

### Tasks (Detailed)
```
T-SAMPLE-001: Create scripts/generate_samples.py scaffold
T-SAMPLE-002: Generate 5 tutorial documents (how does X work?)
T-SAMPLE-003: Generate 5 architecture documents (where is X?)
T-SAMPLE-004: Generate 5 glossary documents (what is X?)
T-SAMPLE-005: Generate 5 ADR documents (why X?)
T-SAMPLE-006: Generate 10 security domain documents
T-SAMPLE-007: Generate 10 API domain documents
T-SAMPLE-008: Generate 10 database domain documents
T-SAMPLE-009: Re-run showcase to verify improvement
T-SAMPLE-010: Document sample generation process
```

---

# NLU Enhancement Epic

## Epic: NLU Enhancement (nlu)
**Started:** 2025-12-19
**Status:** Planning

### Vision
Transform the Cortical Text Processor into a system that truly understands user queries, explains its reasoning, and improves over time.

### Phases
- **Phase 1:** Search Quality ← Sprint 15
- **Phase 2:** Enhanced NLU ← Sprint 16
- **Phase 3:** SparkSLM ← Sprint 17
- **Phase 4:** Reasoning ← Sprint 18
- **Phase 5:** Knowledge Base ← Sprint 19

### Dependencies
```
Sprint 15 (Search Quality) ──┐
                             ├──→ Sprint 16 (Enhanced NLU)
Sprint 19 (Samples) ─────────┘
                                        │
                                        ▼
                             Sprint 17 (SparkSLM)
                                        │
                                        ▼
                             Sprint 18 (Reasoning)
```

Sprint 15 and 19 can run in parallel. Sprint 16 depends on both. Sprint 17 and 18 are sequential.

