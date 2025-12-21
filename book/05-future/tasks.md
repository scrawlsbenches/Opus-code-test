---
title: "Task Backlog"
generated: "2025-12-20T19:52:22.715247Z"
generator: "future"
source_files:
  - ".got/snapshots/"
tags:
  - tasks
  - backlog
  - planning
  - got
---

# Task Backlog

*Total: 214 tasks*

## In Progress (2)

### Complete corpus balancing for MoE training - generate ~2,500 documents across 80+ domains

- **ID:** ``
- **Priority:** high
- **Category:** arch

## Context
Building a balanced corpus capable of training ~100 specialized micro-models for a Mixture of Experts (MoE) system.

## Current State (as of 2025-12-18)
- Total samples: 301 documents
- Domains covered: 101/101
- Balance score: 0.357 (target >0.7)
- Gini coefficient: 0.643 (lower is better)
- Gap: ~2,342 documents needed to reach target

## Progress (2025-12-18)

### Phase 1: Document Creation (44 docs across 17 domains)
**Critical Gap Domains (16 domains that had 0 docs):**
- chemica

### CommandExpert data collection milestone

- **ID:** ``
- **Priority:** low
- **Category:** tracking

Track data collection for CommandExpert to become useful. This is a long-running tracking task.

## Pending (49)

### Test GoT CLI functionality

- **ID:** ``
- **Priority:** high
- **Category:** feature

### Implement NestedLoopExecutor with step-based cooperative multitasking

- **ID:** ``
- **Priority:** high
- **Category:** arch

Sprint 4: Implement hierarchical loop execution with step-based approach (not coroutines). Key features: 1) Priority queue execution with adjustable time budgets 2) Parent control over child loops (pause/resume/terminate) 3) Hierarchical time allocation (children get fraction of parent budget) 4) State machine for yielding/resuming at checkpoints 5) Proper propagation of results up the hierarchy. See discussion in session about avoiding coroutine complexity.

### Implement ClaudeCodeSpawner for production parallel agents

- **ID:** ``
- **Priority:** high
- **Category:** arch

Create AgentSpawner implementation that uses the Task tool to spawn real Claude Code agents. Features: 1) Serialize boundary to agent prompt 2) Parse agent output to extract files modified 3) Handle timeout via Task tool timeout parameter 4) Collect results from agent's final output 5) Map Task tool status to AgentStatus enum. Consider: Environment detection for when running inside Claude Code vs testing.

### Create projects directory structure

- **ID:** ``
- **Priority:** high
- **Category:** arch

Create cortical/projects/ with mcp/, proto/, cli/ subdirectories. Each project gets __init__.py, tests/, requirements.txt

### Move MCP server to projects/mcp

- **ID:** ``
- **Priority:** high
- **Category:** arch

Move cortical/mcp_server.py to cortical/projects/mcp/server.py. Update imports. Move tests/test_mcp_server.py to cortical/projects/mcp/tests/

### Update pyproject.toml with project dependencies

- **ID:** ``
- **Priority:** high
- **Category:** arch

Create optional dependency groups: [mcp], [proto], [cli]. Update [dev] to reference projects. Ensure core has zero deps.

### Update CI for projects isolation

- **ID:** ``
- **Priority:** high
- **Category:** arch

Update .github/workflows/ci.yml to run core tests always, project tests as separate jobs that can fail independently.

### Document Projects architecture

- **ID:** ``
- **Priority:** high
- **Category:** docs

Create docs/projects-architecture.md explaining the pattern. Update CLAUDE.md with Projects section.

### Verify all tests pass after restructure

- **ID:** ``
- **Priority:** high
- **Category:** test

Run full test suite. Fix any import errors. Ensure coverage is maintained.

### Continue Hubris model integration: add book chapter, update generate_book.py, create README for scripts/hubris/

- **ID:** ``
- **Priority:** high
- **Category:** feature

Continue the Hubris micro-model integration started in naming session

### Establish Refactoring Commit Convention (CLAUDE.md Update)

- **ID:** ``
- **Priority:** high
- **Category:** process

### Chunked ML Tracking - Eliminate JSONL Merge Conflicts

- **ID:** ``
- **Priority:** high
- **Category:** arch

### Add async API for batch operations (add_documents_batch, find_passages_batch, compute_all with run_in_executor)

- **ID:** ``
- **Priority:** high
- **Category:** feature

### Fix 7 bare exception handlers with specific exception types

- **ID:** ``
- **Priority:** high
- **Category:** bugfix

### Phase 2: Add inter-agent communication to ParallelCoordinator

- **ID:** ``
- **Priority:** medium
- **Category:** arch

Add optional agent communication for complex workflows where boundary isolation isn't sufficient. Consider: 1) File-based message passing (simple, git-friendly) 2) Shared state file that agents can read/update 3) Dependency signals (agent B waits for signal from agent A) 4) Progress updates visible to coordinator 5) Emergency stop propagation. IMPORTANT: Only implement if proven necessary - current boundary-based isolation worked for 8 parallel agents in Sprints 1-3.

### Add reasoning loop metrics and observability

- **ID:** ``
- **Priority:** medium
- **Category:** feature

Integrate CognitiveLoop with the observability system (cortical/observability.py). Track: 1) Time spent in each QAPV phase 2) Phase transition counts and patterns 3) Questions raised vs answered ratio 4) Decision counts per loop 5) Child loop spawn rate 6) Loop completion vs abandonment rates. Use for ML training data and loop optimization.

### Implement QAPV cycle behavioral validation

- **ID:** ``
- **Priority:** medium
- **Category:** feature

Add validation that enforces proper QAPV cycle behavior: 1) Cannot skip Question phase (must understand before acting) 2) Answer phase requires evidence/justification 3) Produce phase tracks artifacts created 4) Verify phase runs checks before transitioning. Add LoopValidator class that can audit a loop's history and flag violations. Useful for training data quality.

### Add crisis recovery integration with CognitiveLoop

- **ID:** ``
- **Priority:** medium
- **Category:** feature

Connect RecoveryProcedures with CognitiveLoop for automatic recovery: 1) When loop enters BLOCKED state, suggest recovery procedures 2) When verification fails repeatedly, trigger crisis mode 3) Preserve loop state during git stash/checkout operations 4) Generate memory document on abandoned loops 5) Auto-save loop state before risky operations. Reference: crisis_manager.py RecoveryProcedures class.

### Add intelligent question batching with dependency analysis

- **ID:** ``
- **Priority:** medium
- **Category:** feature

Enhance QuestionBatcher with dependency-aware batching: 1) Detect when answer to Q1 affects Q2 (don't batch) 2) Group questions by topic for cognitive efficiency 3) Predict which questions human can answer vs needs research 4) Prioritize blocking questions across multiple loops 5) Suggest default answers based on codebase patterns. Integration with ProductionState.blocking_questions.

### ML training data extraction from reasoning loops

- **ID:** ``
- **Priority:** medium
- **Category:** feature

Extract training data from completed CognitiveLoops: 1) Input: goal + context → Output: successful phase sequence 2) Input: question raised → Output: how it was answered 3) Input: verification failure → Output: recovery action taken 4) Input: decision context → Output: decision + rationale 5) Track which loop patterns lead to success vs abandonment. Store in .git-ml/reasoning/ alongside commit data.

### Book generation system improvements and documentation

- **ID:** ``
- **Priority:** medium
- **Category:** docs

## BOOK GENERATION SYSTEM OVERVIEW

The Living Book is auto-generated from codebase artifacts by `scripts/generate_book.py`.

### Generators (11 implemented + 1 placeholder):

| Generator | Output Dir | Source |
|-----------|-----------|--------|
| AlgorithmChapterGenerator | 01-foundations | docs/VISION.md algorithm sections |
| ModuleDocGenerator | 02-architecture | .ai_meta files from cortical/ |
| DecisionStoryGenerator | 03-decisions | ADR files in samples/decisions/ |
| CommitNarrativeGene

### Move proto to projects/proto

- **ID:** ``
- **Priority:** medium
- **Category:** arch

Move cortical/proto/ to cortical/projects/proto/. Update imports in any referencing files.

### Create knowledge transfer document

- **ID:** ``
- **Priority:** medium
- **Category:** docs

Create memory document summarizing the Projects architecture decision and implementation.

### Implement ErrorDiagnosisExpert for debugging assistance

- **ID:** ``
- **Priority:** medium
- **Category:** feat

Expert that maps error messages and stack traces to likely source files. Train from debugging session patterns.

### Create episode expert training from session transcripts

- **ID:** ``
- **Priority:** medium
- **Category:** feat

Train micro-experts from each session transcript. Capture file co-occurrence, query patterns, and error-fix associations per episode.

### RefactorExpert Model Training Milestone Review

- **ID:** ``
- **Priority:** medium
- **Category:** ml

### CI Results Capture for TestExpert Training

- **ID:** ``
- **Priority:** medium
- **Category:** ml

### Action Sequence Collection for EpisodeExpert

- **ID:** ``
- **Priority:** medium
- **Category:** ml

### Add 'commits behind origin' indicator to session dashboard

- **ID:** ``
- **Priority:** medium
- **Category:** arch

Show how many commits behind origin/main the current branch is during session start and periodically. Helps track when sync is needed. Could show: 'Branch is 5 commits behind origin/main - consider rebasing'. Discuss placement: dashboard, status line, or periodic reminder.

### Add auto-push before context compaction to session hooks

- **ID:** ``
- **Priority:** medium
- **Category:** enhancement

Add git push to session hooks before context window compaction to create recovery checkpoints. If compaction causes issues, we can recover from the last pushed state.

### Add sprint/epic persistence to tasks/ directory (sprints/, epics/ subdirs or CURRENT_SPRINT.md)

- **ID:** ``
- **Priority:** medium
- **Category:** arch

### Sprint 6 TestExpert Follow-up: Calibration Data Collection and Benchmark Creation

- **ID:** ``
- **Priority:** medium
- **Category:** feature

## Context
Sprint 6 (TestExpert Activation) is complete. The model is trained and operational but needs production usage to collect calibration data.

## Current State
- **Branch**: `claude/testexpert-activation-sprint6-0c9WR` (ready for PR to main)
- **Tests**: 4,775 passing, 89% coverage
- **Model**: Trained on 868 commits with 138,196 source→test mappings

## What's Done
1. ✅ `suggest-tests` CLI command working
2. ✅ TestExpert trained with source-to-test mappings
3. ✅ Post-test feedback hook 

### Make test files resilient to missing pytest - use conditional imports or unittest.skipIf patterns

- **ID:** ``
- **Priority:** medium
- **Category:** arch

### Add streaming API for large documents (>1MB) with generator-based processing

- **ID:** ``
- **Priority:** medium
- **Category:** feature

### Batch ML tracking commits per session instead of per-operation

- **ID:** ``
- **Priority:** medium
- **Category:** optimization

### Implement ThoughtGraph visualization for reasoning traces

- **ID:** ``
- **Priority:** low
- **Category:** feature

Add visualization capabilities for ThoughtGraph: 1) Export to Mermaid diagram format 2) Export to DOT (Graphviz) format 3) Generate ASCII tree representation for terminal 4) Highlight paths from evidence to conclusion 5) Color-code nodes by type (question, hypothesis, evidence, decision) 6) Show edge weights for reasoning strength. Useful for debugging and explaining reasoning.

### Refactor large reasoning module files into focused packages

- **ID:** ``
- **Priority:** low
- **Category:** refactor

Split large files (800+ lines) into focused modules: 1) cognitive_loop.py → cognitive_loop/ package (core.py, serialization.py, manager.py) 2) verification.py → verification/ package (checks.py, suite.py, analyzer.py, regression.py) 3) production_state.py → production/ package (state.py, metrics.py, planner.py, cleaner.py). Follow the processor/ package refactor pattern. Keep backward-compatible imports.

### Proactively refactor scripts/index_codebase.py before it exceeds token limit

- **ID:** ``
- **Priority:** low
- **Category:** arch

scripts/index_codebase.py is at ~20662 tokens (2263 lines), approaching 25000 token limit. Consider extracting:
- Indexing logic into cortical/indexing.py (reusable library code)
- Keep CLI wrapper in scripts/index_codebase.py
- Extract git-related helpers if substantial
Lower priority as it's further from the limit.

### Add configurable thresholds to replace hard-coded magic numbers

- **ID:** ``
- **Priority:** low
- **Category:** feature

Various functions have hard-coded thresholds (e.g., max_bigrams_per_term=100, max_bigrams_per_doc=500, candidate_multiplier=3). Consider making these configurable via CorticalConfig or function parameters for better tunability.

**PRIMARY REASON:** Developer reliability. Magic numbers scattered throughout the codebase make it difficult for developers to understand, tune, and maintain the system. When thresholds need to change, developers must search the entire codebase to find all instances. Cen

### Consider async support for large corpus processing

- **ID:** ``
- **Priority:** low
- **Category:** feature

For processing very large corpora, async/await support could improve throughput. Functions like add_documents_batch and compute_all could benefit from async patterns for parallel document processing.

**IMPLEMENTATION NOTES:**
- This should only be performed on a DEDICATED THREAD
- Do NOT process documents in batches within async (defeats the purpose)
- Use threading.Thread or concurrent.futures.ThreadPoolExecutor
- Keep the main processing loop synchronous; only offload I/O-bound operations
- C

### Add logging level configuration guidance

- **ID:** ``
- **Priority:** low
- **Category:** docs

Some info-level log messages (e.g., in compute_all, save_processor) might be better as debug-level for production use. Add documentation about recommended logging configuration for different environments.

### Pickle removal complete - verify CI passes and consider follow-up cleanup

- **ID:** ``
- **Priority:** low
- **Category:** maintenance

## Session Summary (2025-12-17)

Pickle serialization has been completely removed from the codebase (T-017).

### Completed Work:
1. Removed pickle serialization code from `cortical/persistence.py`
2. Removed format/signing_key parameters from `cortical/processor/persistence_api.py`
3. Removed SignatureVerificationError from `cortical/__init__.py` exports
4. Updated 14 test files to use JSON directories instead of .pkl files:
   - tests/unit/test_persistence.py
   - tests/unit/test_repl.py
   - 

### Implement expert consolidation pipeline

- **ID:** ``
- **Priority:** low
- **Category:** feat

Periodically merge episode experts into domain experts using weighted merge, union, or intersection strategies.

### Update README.md with context compaction recovery practices

- **ID:** ``
- **Priority:** low
- **Category:** docs

Document the practice of pushing changes before context window compaction as a recovery strategy. Add to development workflow section.

### Add 'why' tracking to task completions (decision_context field)

- **ID:** ``
- **Priority:** low
- **Category:** arch

### Create SprintPlanningExpert after collecting 10+ sprints of data

- **ID:** ``
- **Priority:** low
- **Category:** moe

### Add user preference tracking for cross-session style learning

- **ID:** ``
- **Priority:** low
- **Category:** moe

### Improve Hubris MoE cold-start UX - show clearer message when experts are untrained, consider falling back to ML file prediction model

- **ID:** ``
- **Priority:** low
- **Category:** moe

### Evaluate and optionally add gzip compression for chunk storage exports

- **ID:** ``
- **Priority:** low
- **Category:** optimization

## Deferred (4)

### Add tests

- **ID:** ``
- **Priority:** high
- **Category:** test

CLOSED: Task lacks specificity - 'test.py' is not a real file. Auto-generated without proper context.

### Phase 2: Context-aware patterns and calibration

- **ID:** ``
- **Priority:** low
- **Category:** feature

Wire feedback loop for command success/failure. Learn context→command patterns. Add confidence calibration. Target: >65% accuracy, ECE <0.15.

### Phase 3: CLI suggest-command integration

- **ID:** ``
- **Priority:** low
- **Category:** feature

Add suggest-command to hubris_cli.py. User-facing command suggestions with explanations. Track adoption rate.

### Add docs

- **ID:** ``
- **Priority:** low
- **Category:** docs

CLOSED: Task lacks specificity - 'module.py' is not a real file. Auto-generated without proper context.

## Completed (159)

### Implement OrchestrationPlan and Batch classes

- **ID:** ``
- **Priority:** high
- **Category:** feature

Create core orchestration classes in scripts/orchestration_utils.py:

1. OrchestrationPlan class:
   - plan_id generation (OP-YYYYMMDD-HHMMSS-XXXX format)
   - title, goal, batches attributes
   - save() with atomic writes (temp -> rename)
   - load() class method

2. Batch class:
   - batch_id, name, batch_type (parallel|sequential)
   - agents list, depends_on list
   - status tracking

3. Agent class:
   - agent_id, task_type, description
   - scope dict (files_read, files_write, constraints)

### Implement ExecutionTracker class

- **ID:** ``
- **Priority:** high
- **Category:** feature

Add ExecutionTracker to scripts/orchestration_utils.py:

1. ExecutionTracker class:
   - Links to an OrchestrationPlan
   - execution_id generation
   - start_batch(batch_id) method
   - record_agent_result(agent_id, result) method
   - complete_batch(batch_id, verification) method
   - record_replan(trigger, old_plan, new_plan, reason) method

2. AgentResult dataclass:
   - status, started_at, completed_at
   - duration_ms, output_summary
   - files_modified, errors

3. Persistence to .claude/o

### Write unit tests for orchestration_utils.py

- **ID:** ``
- **Priority:** high
- **Category:** test

Create tests/unit/test_orchestration_utils.py:

Test coverage requirements:
1. OrchestrationPlan tests:
   - ID generation format validation
   - Add/remove batches
   - Save/load round-trip
   - Atomic write behavior

2. Batch tests:
   - Agent management
   - Dependency tracking
   - Status transitions

3. ExecutionTracker tests:
   - Batch lifecycle (start -> record -> complete)
   - Agent result recording
   - Replan event recording

4. Edge cases:
   - Empty plan
   - Plan with no batches
 

### Implement verify_batch.py script

- **ID:** ``
- **Priority:** high
- **Category:** feature

Create scripts/verify_batch.py for automated batch verification:

1. verify_tests(quick=False) -> VerificationResult:
   - Run smoke tests (quick=True) or full tests
   - Capture pass/fail and test counts
   - Return structured result

2. verify_no_conflicts(modified_files) -> VerificationResult:
   - Check no file modified by multiple agents
   - Report conflicts if found

3. verify_git_status() -> VerificationResult:
   - Check for uncommitted .py files
   - Report status summary

4. run_verif

### Investigate performance bottleneck in search

- **ID:** ``
- **Priority:** high
- **Category:** perf

### Fix non-atomic file writes (data loss risk)

- **ID:** ``
- **Priority:** high
- **Category:** bugfix

TaskSession.save() should write to .tmp then atomic rename

### Fix path traversal vulnerability in archive

- **ID:** ``
- **Priority:** high
- **Category:** security

archive_old_session_files() must validate paths stay within tasks/

### Fix task counter overflow at 100 tasks

- **ID:** ``
- **Priority:** high
- **Category:** bugfix

Expand counter format from 02d to 03d or use base36

### Investigate ML data collection for ephemeral Claude Code Web environment

- **ID:** ``
- **Priority:** high
- **Category:** arch

Current ML collection assumes persistent environment. In Claude Code Web: 1) Post-commit/pre-push hooks create files but don't persist across sessions, 2) Session tracking needs Stop hook integration, 3) Consider tracking ALL .git-ml/ data in git since user has no privacy concerns. Need to design approach that works for ephemeral environments.

### Design session capture strategy for Claude Code Web

- **ID:** ``
- **Priority:** high
- **Category:** arch

BLOCKING: Have 0/100 sessions and 0/200 chats - this is the real bottleneck for ML training. Stop hook is configured (.claude/settings.local.json) but data not persisting in ephemeral environment. Options: 1) Ensure Stop hook commits data before session ends, 2) Manual /ml-log after significant work, 3) Different architecture for ephemeral environments.

### Unit tests for: Workflow templates

- **ID:** ``
- **Priority:** high
- **Category:** test

Write comprehensive unit tests.

Coverage requirements:
- Happy path scenarios
- Edge cases
- Error conditions
- Target: 90%+ coverage for new code


### Integration tests for: Workflow templates

- **ID:** ``
- **Priority:** high
- **Category:** test

Write integration tests verifying feature works with existing system.

Test scenarios:
- End-to-end workflows
- Interaction with other components
- Performance characteristics


### Refactor scripts/ml_data_collector.py into a package structure

- **ID:** ``
- **Priority:** high
- **Category:** arch

scripts/ml_data_collector.py exceeds 25000 tokens (~37759 tokens, 4153 lines). Split into scripts/ml_collector/ package with:
- core.py: Configuration, exceptions, base classes
- data_classes.py: TranscriptExchange, DiffHunk, CommitContext, ChatEntry, ActionEntry
- commit_collector.py: Git commit collection and parsing
- chat_collector.py: Chat session logging and retrieval
- session_manager.py: Session start/end, handoffs
- github_collector.py: GitHub PR/Issue collection
- export.py: Training d

### Refactor tests/unit/test_processor_core.py into multiple test files

- **ID:** ``
- **Priority:** high
- **Category:** testing

tests/unit/test_processor_core.py exceeds 25000 tokens (~34468 tokens, 3534 lines with 27 test classes). Split into:
- test_processor_init.py: TestProcessorInitialization
- test_processor_documents.py: TestDocumentManagement, TestIncrementalDocumentAddition, TestBatchDocumentOperations
- test_processor_metadata.py: TestMetadataManagement
- test_processor_staleness.py: TestStalenessTracking
- test_processor_layers.py: TestLayerAccess
- test_processor_config.py: TestConfiguration, TestBasicValidat

### Add session context generator for agent handoff

- **ID:** ``
- **Priority:** high
- **Category:** agent-dx

When a new agent session starts, generate a concise "catch-up" summary:
- What work was done in the last N sessions
- Current pending tasks sorted by priority
- Recent file changes with semantic summaries
- Key decisions made (from commit messages)

This reduces the 'cold start' problem where agents spend time re-understanding context.

Implementation:
- scripts/session_context.py with SessionContextGenerator class
- Read from tasks/*.json for pending/completed work
- Use git log for recent chan

### Implement state_storage.py (Phase 1 of Task #206)

- **ID:** ``
- **Priority:** high
- **Category:** arch

Create cortical/state_storage.py with StateWriter and StateLoader classes.
This is Phase 1 of Task #206 (git-friendly pkl replacement).

Key classes:
- StateWriter: serialize layers, connections, computed values to JSON
- StateLoader: load state with hash validation  
- StateManifest: track versions, checksums, staleness

Design follows chunk_index.py pattern:
- Append-only where possible
- Content hashing for change detection
- Split large state into multiple files

Files to create:
- cortical/

### Replace broad except Exception blocks with specific exception handling

- **ID:** ``
- **Priority:** high
- **Category:** bugfix

Found multiple 'except Exception:' blocks in wal.py, cli_wrapper.py, state_storage.py, mcp_server.py, and ml_experiments modules that could hide bugs and make debugging difficult. Replace these with specific exception types that are actually expected.

### Implement LRU eviction for query expansion cache

- **ID:** ``
- **Priority:** high
- **Category:** feature

The _query_expansion_cache has _query_cache_max_size=100 defined but there's no actual LRU eviction logic. The cache can grow unbounded. Implement proper LRU cache behavior using collections.OrderedDict or functools.lru_cache.

### Bound MetricsCollector timing history to prevent memory growth

- **ID:** ``
- **Priority:** high
- **Category:** bugfix

MetricsCollector.operations stores all timings in an unbounded list (op_data['timings']). For long-running processes, this can cause memory growth. Should implement circular buffer or rolling window for timing history.

### Phase out pickle persistence in favor of JSON format (SEC-003)

- **ID:** ``
- **Priority:** high
- **Category:** security

## PKL SECURITY ISSUES SUMMARY

**Critical Security Vulnerability (RCE):**
Python's `pickle.load()` can execute arbitrary code during deserialization. If an attacker controls a .pkl file, they achieve Remote Code Execution.

**Current Mitigations (Partial):**
1. DeprecationWarning in persistence.py (lines 133-140, 216-224)
2. HMAC signature verification option (SEC-003) - but optional
3. StateWriter/StateLoader JSON alternative exists (state_storage.py)

**Affected Files:**
- cortical/persistenc

### Add regression tests for edge cases identified in code review

- **ID:** ``
- **Priority:** high
- **Category:** test

Create regression tests for the following edge cases found during code review:

## Cache/Memory Bounds
- Empty cache behavior for query expansion (T-003)
- MetricsCollector with empty/single operation timing history (T-005)

## Division by Zero
- Scoring calculations when max_score is 0 (T-015)
- TF-IDF normalization with empty corpus

## Validation Edge Cases
- graph_boosted_search with pagerank_weight + proximity_weight > 1.0 (T-004)
- Empty query strings
- top_n = 0 or negative

## Persistenc

### Enable code stop word filtering by default in search

- **ID:** ``
- **Priority:** high
- **Category:** bugfix

Set filter_code_stop_words=True by default in find_documents_for_query() at cortical/query/search.py:54-59. This filters ubiquitous code tokens (def, self, return) from expansion, reducing noise in code search results. Low risk - filtering logic already exists and is tested.

### Apply test file penalty by default in search

- **ID:** ``
- **Priority:** high
- **Category:** bugfix

Detect test files and apply 0.8 penalty in find_documents_for_query() at cortical/query/search.py. Files matching tests/ or test_ pattern should be penalized. Penalty already defined in constants.py. Low risk - easy implementation.

### Document dog-fooding workflow

- **ID:** ``
- **Priority:** high
- **Category:** docs

Create a practical guide for using the task system in daily work

### Update README.md with comprehensive project overview

- **ID:** ``
- **Priority:** high
- **Category:** docs

The README.md needs to be updated to reflect current project state and be more engaging.

Current issues:
- May be outdated with recent features
- Missing text-as-memories system
- Missing security features (HMAC signing, Bandit CI)
- Missing Claude skills and commands

README should include:
1. Compelling intro with the 'visual cortex for text' metaphor
2. Quick start (5-line example)
3. Key features table with badges
4. Architecture diagram (ASCII or link to docs)
5. Installation instructions


### Design and implement MicroExpert base class with serialization

- **ID:** ``
- **Priority:** high
- **Category:** arch

Create the foundational MicroExpert dataclass that all experts inherit from. Include JSON serialization, versioning, metrics tracking, and calibration support. See docs/moe-thousand-brains-architecture.md for full spec.

### Implement ExpertRouter with intent classification

- **ID:** ``
- **Priority:** high
- **Category:** arch

Create router that maps queries to appropriate experts based on intent. Use keyword/pattern matching initially, with hooks for learned routing weights later.

### Create VotingAggregator for expert prediction fusion

- **ID:** ``
- **Priority:** high
- **Category:** arch

Implement confidence-weighted voting to combine predictions from multiple experts. Include disagreement detection and consensus calculation.

### Implement CreditAccount and CreditTransaction data classes

- **ID:** ``
- **Priority:** high
- **Category:** arch

Foundation for the intelligence exchange. Credit accounts track balances, ROI, and win rates. Transactions are append-only ledger entries.

### Implement ValueSignal types and attribution algorithm

- **ID:** ``
- **Priority:** high
- **Category:** arch

Value signals capture outcomes (CI pass, user feedback, etc). Attribution algorithm distributes value across contributing experts using Shapley-value-inspired approach.

### SEC-001: Add pickle security warning to README

- **ID:** ``
- **Priority:** high
- **Category:** security

Add a security warning to README.md about pickle deserialization risks.

Location: cortical/persistence.py:156 uses pickle.load() which can execute arbitrary code.

Warning should include:
- Risk explanation (RCE via malicious pickle files)
- Recommendation to use JSON format (StateLoader) for untrusted sources
- Link to Python pickle documentation security warning

Effort: 30 minutes

### SEC-002: Add Bandit SAST to CI pipeline

- **ID:** ``
- **Priority:** high
- **Category:** security

Add Bandit (Python static analysis security tool) to CI workflow.

Add to .github/workflows/ci.yml:
```yaml
security-scan:
  name: "Security Scan"
  runs-on: ubuntu-latest
  steps:
  - uses: actions/checkout@v4
  - name: Set up Python
    uses: actions/setup-python@v5
    with:
      python-version: '3.11'
  - name: Run Bandit
    run: |
      pip install bandit
      bandit -r cortical/ -ll -f txt
```

Effort: 1 hour

### Implement test suite at session start/end hooks

- **ID:** ``
- **Priority:** high
- **Category:** arch

Run full test suite on SessionStart (validate baseline, postpone if failing) and SessionEnd (validate no regressions before commit/merge/push). Part of Continuous Consciousness Sprint 2.

### Implement checkpoint commit system for crash protection

- **ID:** ``
- **Priority:** high
- **Category:** arch

Auto-commit to WIP branch every 15 minutes with 'wip: Checkpoint [timestamp]' messages. Squash on session end. Protects against session termination/crash. Part of Continuous Consciousness Sprint 2.

### Add EXPERIMENTAL warning banner to all Hubris predictions

- **ID:** ``
- **Priority:** high
- **Category:** moe

### Add tests for query/search.py to improve coverage from 26% to >80%

- **ID:** ``
- **Priority:** high
- **Category:** testing

Added 20 new tests achieving 96% coverage

### Add tests for query/ranking.py to improve coverage from 25% to >80%

- **ID:** ``
- **Priority:** high
- **Category:** testing

Added 27 new tests achieving 100% coverage

### Add tests for analysis.py to improve coverage from 60% to >80%

- **ID:** ``
- **Priority:** high
- **Category:** testing

Added 31 new tests achieving 94% coverage

### ML-P0: Build training data export command

- **ID:** ``
- **Priority:** high
- **Category:** feature

Add 'export' command to ml_data_collector.py with formats: jsonl, huggingface, csv

### ML-TEST: Unit tests for ML data collector

- **ID:** ``
- **Priority:** high
- **Category:** test

Add unit tests for export, feedback, quality-report commands

### Optimize doc_connections O(n²) bottleneck - 139s at 539 docs

- **ID:** ``
- **Priority:** high
- **Category:** performance

### Install dev dependencies (pytest, coverage) - tests failing due to missing pytest module

- **ID:** ``
- **Priority:** high
- **Category:** bugfix

### Mark completed task: Remove legacy TASK_LIST.md

- **ID:** ``
- **Priority:** high
- **Category:** cleanup

Task T-20251215-203333-4e1b-001 was completed in this session. TASK_LIST.md and TASK_ARCHIVE.md were deleted. Mark as complete with retrospective.

### Filter deleted files from ML training data

- **ID:** ``
- **Priority:** high
- **Category:** bugfix

MRR dropped from 0.38 to 0.15 because training data includes deleted files (TASK_LIST.md, cortical/processor.py). Need to filter commits to only include files that currently exist, or weight recent commits higher.

### Investigate removing unused protobuf serialization

- **ID:** ``
- **Priority:** high
- **Category:** cleanup

The protobuf serialization feature was added during performance tuning exploration but was never actually used. 

**Context:**
- Added for cross-language corpus sharing (theoretical use case)
- Library already has working pickle + JSON serialization
- Requires `protoc` compiler for runtime compilation (or pre-compiled schema_pb2.py)
- Just caused CI smoke test failures (fixed with lazy loading in commit a93518f)
- Adds ~600 lines across 3 files for unused functionality
- Adds protobuf as a dev d

### Write unit tests for verify_batch.py

- **ID:** ``
- **Priority:** medium
- **Category:** test

Create tests for verify_batch.py:

1. Test verify_no_conflicts():
   - No conflicts case
   - Single conflict case
   - Multiple conflicts case

2. Test verify_git_status():
   - Clean status
   - Modified files
   - Untracked files

3. Test run_verification():
   - All pass case
   - Mixed results case
   - Integration with real git status

4. Test CLI parsing:
   - --quick flag
   - --check option
   - --json output

Note: Use mocking for subprocess calls to avoid running actual tests

### Implement OrchestrationMetrics class

- **ID:** ``
- **Priority:** medium
- **Category:** feature

Add metrics collection to scripts/orchestration_utils.py:

1. OrchestrationMetrics class:
   - metrics_file path (.claude/orchestration/metrics.jsonl)
   - record(event_type, **kwargs): append to JSONL
   - get_summary(): aggregate statistics
   - get_failure_patterns(): analyze common failures

2. Event types to support:
   - batch_start, batch_complete
   - agent_complete
   - verification
   - replan

3. Metrics schema per event:
   - timestamp, plan_id, event_type
   - batch_id, duration_ms


### Add task system integration

- **ID:** ``
- **Priority:** medium
- **Category:** feature

Add orchestration integration to scripts/task_utils.py:

1. create_orchestration_tasks(plan, session) -> List[Task]:
   - Create parent task for the plan
   - Create child tasks for each batch
   - Set depends_on relationships
   - Link task IDs back to plan

2. link_plan_to_task(plan_id, task_id) -> None:
   - Update plan with task_links
   - Update task with plan reference

3. capture_batch_retrospective(plan_id, batch_id, metrics):
   - Extract metrics for completed batch
   - Format as retro

### Update CI to output pending tasks intelligently

- **ID:** ``
- **Priority:** medium
- **Category:** automation

Add a CI step that shows pending tasks in a smart way: grouped by priority, showing blockers first, with context about what's ready to work on next. Could integrate with the workflow system to show task chains and dependencies.

### Add workflow templates (bugfix, feature, refactor)

- **ID:** ``
- **Priority:** medium
- **Category:** feature

YAML templates in .claude/workflows/ that spawn task chains

*... and 109 more completed tasks*

