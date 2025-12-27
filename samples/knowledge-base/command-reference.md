# Command Reference

This document provides a comprehensive reference of CLI commands used in the Cortical Text Processor project. Commands are organized by functional area for easy navigation.

## Table of Contents

- [GoT Task Management](#got-task-management)
- [GoT Sprint Management](#got-sprint-management)
- [GoT Edge Management](#got-edge-management)
- [GoT Query Language](#got-query-language)
- [GoT Handoff Management](#got-handoff-management)
- [GoT Decision Logging](#got-decision-logging)
- [Testing Commands](#testing-commands)
- [Code Coverage](#code-coverage)
- [Corpus Generation (SLM Training)](#corpus-generation-slm-training)
- [Codebase Indexing](#codebase-indexing)
- [ML Data Collection](#ml-data-collection)
- [File Prediction Model](#file-prediction-model)
- [Session and Memory Management](#session-and-memory-management)
- [Validation and Health Checks](#validation-and-health-checks)
- [Reasoning Framework](#reasoning-framework)
- [Woven Mind (Dual-Process)](#woven-mind-dual-process)
- [Benchmarking](#benchmarking)
- [Orchestration and Director](#orchestration-and-director)

---

## GoT Task Management

Graph of Thought (GoT) task management commands for creating and tracking work items.

### Create Task
```bash
# Create a task with title
python scripts/got_utils.py task create "Task title"

# Create task with priority
python scripts/got_utils.py task create "Task title" --priority high
python scripts/got_utils.py task create "Task title" --priority medium
python scripts/got_utils.py task create "Task title" --priority low

# Create task with category
python scripts/got_utils.py task create "Fix bug" --priority high --category bugfix
python scripts/got_utils.py task create "Add feature" --category feature
```

### List Tasks
```bash
# List all tasks
python scripts/got_utils.py task list

# Filter by status
python scripts/got_utils.py task list --status pending
python scripts/got_utils.py task list --status in_progress
python scripts/got_utils.py task list --status completed

# Filter by priority
python scripts/got_utils.py task list --priority high
```

### Task Operations
```bash
# Show task details
python scripts/got_utils.py task show T-20251226-123456-a1b2c3d4

# Start working on a task
python scripts/got_utils.py task start T-20251226-123456-a1b2c3d4

# Complete a task
python scripts/got_utils.py task complete T-20251226-123456-a1b2c3d4

# Delete a task
python scripts/got_utils.py task delete T-20251226-123456-a1b2c3d4
python scripts/got_utils.py task delete T-20251226-123456-a1b2c3d4 --force
```

### Dashboard View
```bash
# View comprehensive task dashboard
python scripts/got_utils.py dashboard
```

---

## GoT Sprint Management

Sprint tracking and management commands.

### List Sprints
```bash
# List all sprints with status
python scripts/got_utils.py sprint list
```

### Sprint Status
```bash
# Show current/active sprint
python scripts/got_utils.py sprint status
```

### Create Sprint
```bash
# Create a new sprint
python scripts/got_utils.py sprint create "Sprint Title" --number 17
python scripts/got_utils.py sprint create "Q1 Performance" --number 18
```

---

## GoT Edge Management

Manage relationships between entities in the GoT graph.

### Add Edge
```bash
# Add edge between entities
python scripts/got_utils.py edge add SOURCE_ID TARGET_ID EDGE_TYPE

# Example: Task depends on another task
python scripts/got_utils.py edge add T-20251226-111111 T-20251226-222222 DEPENDS_ON

# Example: Task blocks another task
python scripts/got_utils.py edge add T-20251226-111111 T-20251226-333333 BLOCKS

# Example: Sprint contains task
python scripts/got_utils.py edge add S-sprint-017 T-20251226-444444 CONTAINS
```

### List Edges
```bash
# List all edges
python scripts/got_utils.py edge list

# Filter by edge type
python scripts/got_utils.py edge list --type DEPENDS_ON
python scripts/got_utils.py edge list --type BLOCKS
python scripts/got_utils.py edge list --type CONTAINS

# Filter by source entity
python scripts/got_utils.py edge list --source T-20251226-123456

# Filter by target entity
python scripts/got_utils.py edge list --target T-20251226-123456
```

### Show Edge Types
```bash
# Display available edge types
python scripts/got_utils.py edge types
```

### Edges for Entity
```bash
# Show all edges connected to an entity
python scripts/got_utils.py edge for T-20251226-123456
python scripts/got_utils.py edge for S-sprint-017
```

---

## GoT Query Language

Natural language queries for the GoT graph.

### Dependency Queries
```bash
# What blocks a task
python scripts/got_utils.py query "what blocks T-20251226-123456"

# What depends on a task
python scripts/got_utils.py query "what depends on T-20251226-123456"

# Find path between entities
python scripts/got_utils.py query "path from T-20251226-111111 to T-20251226-222222"

# All relationships for an entity
python scripts/got_utils.py query "relationships T-20251226-123456"
```

### Task Status Queries
```bash
# Active tasks
python scripts/got_utils.py query "active tasks"

# Pending tasks
python scripts/got_utils.py query "pending tasks"

# Blocked tasks
python scripts/got_utils.py query "blocked tasks"
```

---

## GoT Handoff Management

Agent-to-agent work handoff primitives.

### Initiate Handoff
```bash
# Initiate a handoff to another agent
python scripts/got_utils.py handoff initiate T-20251226-123456 \
  --target implementation-agent \
  --instructions "Implement feature X with tests"
```

### Accept Handoff
```bash
# Accept a handoff
python scripts/got_utils.py handoff accept H-20251226-123456 \
  --agent implementation-agent
```

### Complete Handoff
```bash
# Complete a handoff with results
python scripts/got_utils.py handoff complete H-20251226-123456 \
  --agent implementation-agent \
  --result '{"status": "success", "files": ["file1.py", "file2.py"]}'
```

### Reject Handoff
```bash
# Reject a handoff with reason
python scripts/got_utils.py handoff reject H-20251226-123456 \
  --agent implementation-agent \
  --reason "Missing required context"
```

### List Handoffs
```bash
# List all handoffs
python scripts/got_utils.py handoff list

# Filter by status
python scripts/got_utils.py handoff list --status initiated
python scripts/got_utils.py handoff list --status accepted
python scripts/got_utils.py handoff list --status completed
python scripts/got_utils.py handoff list --status rejected
```

---

## GoT Decision Logging

Log architectural and design decisions.

### Log Decision
```bash
# Log a decision with rationale
python scripts/got_utils.py decision log "Use JSON format for persistence" \
  --rationale "JSON is secure, git-friendly, and debuggable"

# Log decision with tags
python scripts/got_utils.py decision log "Implement incremental indexing" \
  --rationale "Faster updates for large corpora" \
  --tags architecture,performance
```

---

## Testing Commands

Run tests with various configurations and filters.

### Quick Test Shortcuts (Makefile)
```bash
# Smoke tests (~1s) - fastest sanity check
make test-smoke

# Fast tests (~5s) - no slow tests
make test-fast

# Quick tests (~30s) - smoke + unit, before commit
make test-quick

# Parallel tests (~23s) - 4 workers
make test-parallel
```

### Test Runner Script
```bash
# Smoke tests
python scripts/run_tests.py smoke

# Unit tests
python scripts/run_tests.py unit
python scripts/run_tests.py quick  # smoke + unit

# Parallel execution (3x faster)
python scripts/run_tests.py unit -j 4

# Pre-commit checks
python scripts/run_tests.py precommit  # smoke + unit + integration

# All tests
python scripts/run_tests.py all
```

### Category-Specific Tests
```bash
# Smoke tests (quick sanity checks)
python -m pytest tests/smoke/ -v

# Unit tests (fast isolated tests)
python -m pytest tests/unit/ -v

# Integration tests (component interaction)
python -m pytest tests/integration/ -v

# Performance tests (timing tests)
python -m pytest tests/performance/ -v
python scripts/run_tests.py performance  # no coverage

# Regression tests (bug-specific)
python -m pytest tests/regression/ -v

# Behavioral tests (user workflow quality)
python -m pytest tests/behavioral/ -v
```

### Optional Dependency Tests
```bash
# Development (default) - excludes optional tests
pytest tests/

# Include optional tests (like CI)
pytest tests/ -m ""

# Using run_tests.py with optional tests
python scripts/run_tests.py unit --include-optional

# Run only specific optional tests
pytest tests/ -m "fuzz"      # Property-based tests
pytest tests/ -m "protobuf"  # Serialization tests
```

### Test with Timeout
```bash
# Add timeout for hanging tests
python -m pytest tests/ --timeout=60
```

---

## Code Coverage

Measure and report test coverage.

### Run with Coverage
```bash
# Run tests with coverage collection
python -m coverage run -m pytest tests/ -q

# Run tests and show report
python -m coverage run -m pytest tests/ && python -m coverage report --include="cortical/*"

# Check baseline coverage (should be ~98%+)
python -m coverage run -m pytest tests/ -q && python -m coverage report --include="cortical/*" | tail -1
```

### Coverage Report Options
```bash
# Text report
python -m coverage report --include="cortical/*"

# HTML report
python -m coverage html --include="cortical/*"
# Open htmlcov/index.html in browser

# Show missing lines
python -m coverage report --include="cortical/*" --show-missing
```

---

## Corpus Generation (SLM Training)

Generate training data for repository-native Statistical Language Model.

### Generate Corpus
```bash
# Quick generation (sample data)
python -m benchmarks.codebase_slm.generate_corpus --quick

# Full corpus generation
python -m benchmarks.codebase_slm.generate_corpus --full

# Custom output directory
python -m benchmarks.codebase_slm.generate_corpus --full --output /path/to/corpus
```

### Train SLM
```bash
# Quick training (for testing)
python -m benchmarks.codebase_slm.train_slm --quick

# Full training
python -m benchmarks.codebase_slm.train_slm --full

# Interactive training with progress
python -m benchmarks.codebase_slm.train_slm --full --interactive

# Custom model output
python -m benchmarks.codebase_slm.train_slm --full --output /path/to/model
```

---

## Codebase Indexing

Index and search the codebase using semantic search.

### Index Codebase
```bash
# Full index (creates corpus_dev.json/)
python scripts/index_codebase.py

# Incremental update (only changed files, fastest)
python scripts/index_codebase.py --incremental
python scripts/index_codebase.py -i

# Check status without indexing
python scripts/index_codebase.py --status
python scripts/index_codebase.py -s

# Force full rebuild
python scripts/index_codebase.py --force
python scripts/index_codebase.py -f

# Verbose output (per-file progress)
python scripts/index_codebase.py --incremental --verbose
python scripts/index_codebase.py -i -v

# Write detailed log
python scripts/index_codebase.py --incremental --log indexing.log
```

### Chunk-Based Indexing (Git-Friendly)
```bash
# Index with chunk storage
python scripts/index_codebase.py --incremental --use-chunks

# Check chunk status
python scripts/index_codebase.py --status --use-chunks

# Compact old chunks (like git gc)
python scripts/index_codebase.py --compact --use-chunks
python scripts/index_codebase.py --compact --before 2025-12-01 --use-chunks
```

### Generate AI Metadata
```bash
# Generate metadata for all modules
python scripts/generate_ai_metadata.py

# Incremental generation (only changed files)
python scripts/generate_ai_metadata.py --incremental

# Combined indexing and metadata generation
python scripts/index_codebase.py --incremental && python scripts/generate_ai_metadata.py --incremental
```

### Search Codebase
```bash
# Basic search
python scripts/search_codebase.py "query here"

# Search with more results
python scripts/search_codebase.py "PageRank algorithm" --top 10

# Verbose output (show full passages)
python scripts/search_codebase.py "bigram separator" --verbose

# Show query expansion
python scripts/search_codebase.py "compute pagerank" --expand

# Interactive search mode
python scripts/search_codebase.py --interactive
```

### Interactive Search Commands
```
# Inside interactive mode:
/expand <query>    # Show query expansion terms
/concepts          # List concept clusters
/stats             # Show corpus statistics
/quit              # Exit interactive mode
```

---

## ML Data Collection

Automatic collection of training data for project-specific micro-model.

### Session Management
```bash
# Check current session status
python scripts/ml_data_collector.py session status

# Start new session manually
python scripts/ml_data_collector.py session start

# End session with summary
python scripts/ml_data_collector.py session end --summary "Implemented feature X"
```

### Stats and Validation
```bash
# View collection statistics
python scripts/ml_data_collector.py stats

# Estimate training viability
python scripts/ml_data_collector.py estimate

# Validate collected data
python scripts/ml_data_collector.py validate
```

### CI Integration
```bash
# Manually record CI results
python scripts/ml_data_collector.py ci set \
  --commit abc123def456 \
  --result pass \
  --coverage 89.5

# Auto-capture from GitHub Actions environment
python scripts/ml_data_collector.py ci-autocapture
```

### Backfill Historical Data
```bash
# Backfill last 100 commits
python scripts/ml_data_collector.py backfill -n 100

# Backfill all commits
python scripts/ml_data_collector.py backfill --all
```

### GitHub Data Collection
```bash
# Collect recent PRs and issues (requires gh CLI)
python scripts/ml_data_collector.py github collect

# Show GitHub data statistics
python scripts/ml_data_collector.py github stats

# Fetch specific PR
python scripts/ml_data_collector.py github fetch-pr --number 42
```

### Process Transcripts
```bash
# Process specific transcript file
python scripts/ml_data_collector.py transcript --file /path/to/transcript.jsonl

# Dry run (preview without saving)
python scripts/ml_data_collector.py transcript --file /path/to/transcript.jsonl --dry-run --verbose
```

### Session Handoff
```bash
# Generate session handoff document
python scripts/ml_data_collector.py handoff
```

---

## File Prediction Model

ML model for predicting which files to modify based on task description.

### Train Model
```bash
# Train file prediction model
python scripts/ml_file_prediction.py train

# Train with custom data split
python scripts/ml_file_prediction.py train --split 0.2
```

### Make Predictions
```bash
# Predict files for a task
python scripts/ml_file_prediction.py predict "Add authentication feature"

# Predict with seed files (boost co-occurring files)
python scripts/ml_file_prediction.py predict "Fix related bug" \
  --seed auth.py login.py
```

### Evaluate Model
```bash
# Evaluate performance (80/20 train/test split)
python scripts/ml_file_prediction.py evaluate --split 0.2

# View detailed metrics
python scripts/ml_file_prediction.py evaluate --split 0.2 --verbose
```

### Model Statistics
```bash
# View model stats
python scripts/ml_file_prediction.py stats
```

### Pre-Commit Suggestions
```bash
# Test pre-commit hook without committing
bash scripts/test-ml-precommit-hook.sh
```

**Environment variables for pre-commit suggestions:**
```bash
export ML_SUGGEST_ENABLED=0        # Disable suggestions (default: 1)
export ML_SUGGEST_THRESHOLD=0.7    # Confidence threshold (default: 0.5)
export ML_SUGGEST_BLOCKING=1       # Block commit if missing (default: 0)
export ML_SUGGEST_TOP_N=10         # Predictions to check (default: 5)
```

---

## Session and Memory Management

Knowledge management and session handoff tools.

### Session Memory Generation
```bash
# Generate draft memory from current session
python scripts/session_memory_generator.py --session-id abc123

# Generate from recent commits (no session ID)
python scripts/session_memory_generator.py --commits 10

# Dry run (preview without saving)
python scripts/session_memory_generator.py --session-id abc123 --dry-run
```

### Memory Creation
```bash
# Create new memory entry
python scripts/new_memory.py "Topic or learning"

# Create decision record
python scripts/new_memory.py "Decision title" --decision
```

### Session Handoff
```bash
# Generate session handoff document
python scripts/session_handoff.py
```

### Wiki Link Resolution
```bash
# Check wiki-style links in a file
python scripts/resolve_wiki_links.py samples/memories/2025-12-26-topic.md

# Find backlinks to a file
python scripts/resolve_wiki_links.py --backlinks samples/memories/2025-12-26-topic.md
```

---

## Validation and Health Checks

Validate system state and run health checks.

### GoT Validation
```bash
# Validate GoT graph integrity
python scripts/got_utils.py validate

# Check for broken edges
python scripts/got_utils.py validate --edges

# Verify checksums
python scripts/got_utils.py validate --checksums
```

### Compact Events
```bash
# Compact event log (default: preserve 30 days)
python scripts/got_utils.py compact

# Compact with custom retention
python scripts/got_utils.py compact --preserve-days 60
```

### Showcase Demo
```bash
# Run interactive showcase
python showcase.py

# Non-interactive mode
python showcase.py --non-interactive
```

### Batch Verification
```bash
# Quick verification
python scripts/verify_batch.py --quick

# Full verification with detailed output
python scripts/verify_batch.py --verbose
```

---

## Reasoning Framework

Graph of Thought reasoning framework with QAPV cycle.

### Reasoning Demos
```bash
# Quick reasoning demo
python scripts/reasoning_demo.py --quick

# Reasoning with persistence (WAL, snapshots)
python scripts/reasoning_demo.py --quick --persist

# Full reasoning demo
python scripts/reasoning_demo.py
```

### Graph Persistence
```bash
# Graph persistence demo (WAL, recovery)
python examples/graph_persistence_demo.py

# Validate persistence integration
python scripts/validate_reasoning_persistence.py
```

### Performance Tests
```bash
# Graph persistence performance tests
python -m pytest tests/performance/test_graph_persistence_perf.py -v

# End-to-end reasoning tests
python -m pytest tests/integration/test_reasoning_persistence_e2e.py -v
```

---

## Woven Mind (Dual-Process)

Dual-process cognitive architecture (System 1/System 2).

### Woven Mind Demo
```bash
# Run all demo sections
python examples/woven_mind_demo.py --section all

# Run specific section
python examples/woven_mind_demo.py --section routing
python examples/woven_mind_demo.py --section consolidation
python examples/woven_mind_demo.py --section adaptation
```

### Woven Mind Tests
```bash
# Run all Woven Mind tests
python -m pytest tests/unit/test_woven_mind*.py -v

# Specific test files
python -m pytest tests/unit/test_woven_mind.py -v
python -m pytest tests/unit/test_loom.py -v
python -m pytest tests/unit/test_consolidation.py -v
```

### Woven Mind Benchmarks
```bash
# List available benchmarks
python -m benchmarks.woven_mind.runner --list

# Run all benchmarks
python -m benchmarks.woven_mind.runner --all

# Run specific category
python -m benchmarks.woven_mind.runner --category stability
python -m benchmarks.woven_mind.runner --category performance
python -m benchmarks.woven_mind.runner --category consolidation

# Quick mode (faster, smaller dataset)
python -m benchmarks.woven_mind.runner --all --quick

# Save results for comparison
python -m benchmarks.woven_mind.runner --all --output results/baseline.json

# Compare with baseline
python -m benchmarks.woven_mind.runner --all --compare results/baseline.json
```

---

## Benchmarking

Performance benchmarking and profiling.

### Profile Analysis
```bash
# Profile full analysis phases
python scripts/profile_full_analysis.py

# Detect timeouts and bottlenecks
python scripts/profile_full_analysis.py --timeout 30
```

### GoT Query Profiling
```bash
# Profile GoT query API performance
python scripts/profile_got_query.py
```

---

## Orchestration and Director

Director orchestration for complex multi-agent workflows.

### Orchestration Plans
```bash
# Generate orchestration plan
python scripts/orchestration_utils.py generate --type plan

# List orchestration plans
python scripts/orchestration_utils.py list

# Show plan details
python scripts/orchestration_utils.py show PLAN_ID
```

### Director Commands (Slash Commands)
```bash
# Note: These are slash commands used within Claude Code CLI
# Not direct shell commands

# General orchestration
/director

# Woven Mind + PRISM integration
/woven-mind-director

# Delegate tasks to sub-agents
/delegate <task description>

# Pre-merge sanity check
/sanity-check <branch-name>
```

### Context Recovery
```bash
# Restore cognitive state after context loss
/context-recovery
```

---

## Common Workflows

### Before Every Commit
```bash
# 1. Run quick tests
make test-quick

# 2. Check coverage
python -m coverage run -m pytest tests/ -q && \
  python -m coverage report --include="cortical/*" | tail -1

# 3. Validate GoT (if used)
python scripts/got_utils.py validate
```

### After Major Changes
```bash
# 1. Run smoke tests
make test-smoke

# 2. Run full test suite
python -m pytest tests/ -v

# 3. Re-index codebase
python scripts/index_codebase.py --incremental

# 4. Update AI metadata
python scripts/generate_ai_metadata.py --incremental
```

### Daily Development Session Start
```bash
# 1. Check GoT state
python scripts/got_utils.py validate
python scripts/got_utils.py task list --status in_progress

# 2. Check current sprint
python scripts/got_utils.py sprint status

# 3. View ML collection stats
python scripts/ml_data_collector.py stats

# 4. Start session (auto via SessionStart hook)
# python scripts/ml_data_collector.py session start
```

### End of Session
```bash
# 1. Complete active tasks
python scripts/got_utils.py task complete T-XXXXX

# 2. Generate session memory (auto via Stop hook)
# python scripts/session_memory_generator.py --session-id abc123

# 3. Commit and push
git add -A && git commit -m "feat: description"
git push
```

---

## Environment Variables

### ML Data Collection
```bash
ML_COLLECTION_ENABLED=0    # Disable ML collection (default: 1)
```

### Pre-Commit Suggestions
```bash
ML_SUGGEST_ENABLED=0       # Disable file suggestions (default: 1)
ML_SUGGEST_THRESHOLD=0.7   # Confidence threshold (default: 0.5)
ML_SUGGEST_BLOCKING=1      # Block commit if missing files (default: 0)
ML_SUGGEST_TOP_N=10        # Number of predictions to check (default: 5)
```

### GoT Auto-Commit & Auto-Push
```bash
GOT_AUTO_COMMIT=0          # Disable auto-commit (default: ON)
GOT_AUTO_PUSH=0            # Disable auto-push (default: ON)
```

---

## Quick Command Lookup

| Task | Command |
|------|---------|
| Smoke test | `make test-smoke` |
| Quick test | `make test-quick` |
| Full tests | `python -m pytest tests/ -v` |
| Coverage | `python -m coverage run -m pytest tests/ && python -m coverage report --include="cortical/*"` |
| Index codebase | `python scripts/index_codebase.py --incremental` |
| Search code | `python scripts/search_codebase.py "query"` |
| Create task | `python scripts/got_utils.py task create "Title" --priority high` |
| List tasks | `python scripts/got_utils.py task list` |
| Sprint status | `python scripts/got_utils.py sprint status` |
| ML stats | `python scripts/ml_data_collector.py stats` |
| Train SLM | `python -m benchmarks.codebase_slm.train_slm --quick` |
| Generate corpus | `python -m benchmarks.codebase_slm.generate_corpus --quick` |
| Validate GoT | `python scripts/got_utils.py validate` |
| Run showcase | `python showcase.py` |

---

**Last Updated:** 2025-12-26
**For comprehensive documentation, see:** `/home/user/Opus-code-test/CLAUDE.md`
