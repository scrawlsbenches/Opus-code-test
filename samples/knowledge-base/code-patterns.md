# Code Patterns Knowledge Base

This document captures common code patterns used in the Cortical Text Processor project.
All examples are verified from actual codebase usage.

## Table of Contents
1. [Common Imports](#common-imports)
2. [Processor Usage](#processor-usage)
3. [GoT Task Management](#got-task-management)
4. [Woven Mind (Dual-Process Cognition)](#woven-mind-dual-process-cognition)
5. [Spark SLM (Statistical Language Model)](#spark-slm-statistical-language-model)
6. [Testing Patterns](#testing-patterns)
7. [CLI Commands](#cli-commands)
8. [Path Setup for Scripts](#path-setup-for-scripts)

---

## Common Imports

### Core Library Imports
```python
# Processor and config
from cortical import CorticalTextProcessor, CorticalLayer
from cortical.config import CorticalConfig
from cortical.tokenizer import Tokenizer

# Layers and data structures
from cortical.layers import CorticalLayer, HierarchicalLayer
from cortical.minicolumn import Minicolumn, Edge

# Query functionality
from cortical.query import (
    expand_query,
    find_documents_for_query,
    create_chunks,
    create_code_aware_chunks
)

# Analysis
from cortical.analysis import compute_pagerank, compute_tfidf

# Persistence
from cortical.chunk_index import ChunkWriter, ChunkLoader, ChunkCompactor
```

### GoT (Graph of Thought) Imports
```python
# GoT management
from cortical.got import GoTManager
from cortical.got.types import Task, Decision, Edge, Sprint, Epic, Handoff

# Reasoning framework
from cortical.reasoning.thought_graph import ThoughtGraph
from cortical.reasoning.graph_of_thought import NodeType, EdgeType, ThoughtNode, ThoughtEdge
from cortical.reasoning.graph_persistence import GraphWAL, GraphRecovery, GitAutoCommitter

# ID generation (canonical source)
from cortical.utils.id_generation import (
    generate_task_id,
    generate_decision_id,
    generate_sprint_id,
    generate_epic_id,
    generate_handoff_id,
)
```

### Woven Mind Imports
```python
# Complete Woven Mind architecture
from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig, WovenMindResult

# Individual components
from cortical.reasoning import (
    # Loom (mode switching)
    Loom,
    LoomConfig,
    ThinkingMode,
    SurpriseDetector,

    # Hive (FAST mode)
    LoomHiveConnector,
    LoomHiveConfig,
    PRISMLanguageModel,

    # Cortex (SLOW mode)
    LoomCortexConnector,
    LoomCortexConfig,
    AbstractionEngine,

    # Consolidation
    ConsolidationEngine,
    ConsolidationConfig,
)
```

### Spark SLM Imports
```python
from cortical.spark import SparkPredictor, NGramModel, AlignmentIndex
from cortical.spark.anomaly import AnomalyDetector
```

### Testing Imports
```python
import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
```

---

## Processor Usage

### Basic Initialization and Document Processing
```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer

# Create processor with default config
processor = CorticalTextProcessor()

# Process a document
processor.process_document("doc_id", "Document content here.")

# Run all computations (PageRank, TF-IDF, clustering, etc.)
processor.compute_all()
```

### Custom Configuration
```python
from cortical import CorticalTextProcessor
from cortical.config import CorticalConfig

# Configure processor
config = CorticalConfig(
    scoring_algorithm='bm25',  # or 'tfidf'
    bm25_k1=1.2,
    bm25_b=0.75,
    max_query_expansions=10,
)

processor = CorticalTextProcessor(config=config)
```

### Code-Aware Processor (for indexing codebases)
```python
from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer

# Tokenizer with identifier splitting and code noise filtering
tokenizer = Tokenizer(split_identifiers=True, filter_code_noise=True)
processor = CorticalTextProcessor(tokenizer=tokenizer)

# Process code files
processor.process_document("auth_service.py", python_source_code)
processor.compute_all()
```

### Search and Retrieval
```python
# Basic search
results = processor.find_documents_for_query("neural networks", top_n=5)
for doc_id, score in results:
    print(f"{doc_id}: {score:.3f}")

# Query expansion
expanded = processor.expand_query("fetch data", max_expansions=10)
print(f"Expanded terms: {expanded}")

# Code-aware search
code_results = processor.expand_query_for_code("authentication handler")

# Passage retrieval (for RAG)
passages = processor.find_passages_for_query(
    "PageRank algorithm",
    top_n=3,
    chunk_size=300,
    overlap=50
)
for text, doc_id, start, end, score in passages:
    print(f"{doc_id}[{start}:{end}]: {score:.3f}")
    print(text[:100])
```

### Fast Search with Pre-Built Index
```python
# Fast search (2-3x faster)
results = processor.fast_find_documents("authentication")

# Build index once, reuse many times
index = processor.build_search_index()
results = processor.search_with_index("query term", index)
```

### Graph-Boosted Search
```python
# Hybrid search combining BM25 + PageRank + graph signals
results = processor.graph_boosted_search(
    "machine learning",
    pagerank_weight=0.3,   # Weight for term importance
    proximity_weight=0.2,  # Weight for connected terms
    top_n=10
)
```

### Incremental Updates
```python
# Add document without full recomputation
processor.add_document_incremental(
    "new_doc_id",
    "New document content",
    recompute='tfidf'  # Options: 'tfidf', 'all', 'none'
)
```

### Persistence
```python
# Save (JSON format - recommended, git-friendly)
processor.save("corpus_dev")  # Creates corpus_dev/ directory

# Load
processor = CorticalTextProcessor.load("corpus_dev")

# Legacy pickle format (deprecated, but still works)
processor.save("corpus.pkl", format='pickle')
processor = CorticalTextProcessor.load("corpus.pkl")
```

### Layer Access
```python
from cortical.layers import CorticalLayer

# Access specific layers
layer0 = processor.get_layer(CorticalLayer.TOKENS)
layer1 = processor.get_layer(CorticalLayer.BIGRAMS)
layer2 = processor.get_layer(CorticalLayer.CONCEPTS)
layer3 = processor.get_layer(CorticalLayer.DOCUMENTS)

# Get minicolumns
col = layer0.get_minicolumn("neural")
if col:
    print(f"PageRank: {col.pagerank}")
    print(f"TF-IDF: {col.tfidf}")
    print(f"Connections: {len(col.lateral_connections)}")
```

---

## GoT Task Management

### Initialize GoT Manager
```python
from cortical.got import GoTManager
from pathlib import Path

# Initialize with .got directory
got_dir = Path.cwd() / ".got"
manager = GoTManager(got_dir)
```

### Create Tasks
```python
# Create a task
task_id = manager.create_task(
    "Implement authentication feature",
    priority="high",  # or "medium", "low"
    category="feature"
)

# Create with dependencies
task2_id = manager.create_task(
    "Add tests for authentication",
    priority="high",
    category="test"
)
manager.add_edge(task2_id, task_id, "depends_on")
```

### List and Filter Tasks
```python
# List all tasks
tasks = manager.list_tasks()

# Filter by status
pending = manager.list_tasks(status="pending")
in_progress = manager.list_tasks(status="in_progress")
completed = manager.list_tasks(status="completed")

# Filter by priority
high_priority = manager.list_tasks(priority="high")
```

### Update Task Status
```python
# Start a task
manager.update_task(task_id, status="in_progress")

# Complete a task
manager.complete_task(
    task_id,
    retrospective="Implemented JWT-based auth with tests"
)

# Block a task
blocker_id = manager.create_task("Fix dependency issue", priority="high")
manager.add_edge(task_id, blocker_id, "blocked_by")
```

### Task Edges and Relationships
```python
from cortical.got.types import EdgeType

# Add various edge types
manager.add_edge(task1_id, task2_id, "depends_on")
manager.add_edge(task1_id, task3_id, "blocks")
manager.add_edge(task1_id, sprint_id, "part_of")

# Query relationships
blockers = manager.get_blockers(task_id)
dependencies = manager.get_dependencies(task_id)
```

### Sprints and Epics
```python
# Create a sprint
sprint_id = manager.create_sprint(
    "Sprint 1: Foundation",
    number=1
)

# Create an epic
epic_id = manager.create_epic(
    "Authentication System",
    description="Complete user authentication and authorization"
)

# Link tasks to sprint
manager.add_edge(task_id, sprint_id, "part_of")

# Link sprint to epic
manager.add_edge(sprint_id, epic_id, "part_of")

# Start sprint
manager.start_sprint(sprint_id)

# Complete sprint
manager.complete_sprint(sprint_id)
```

### Decisions
```python
# Log a decision
decision_id = manager.log_decision(
    "Use JWT for authentication",
    rationale="Industry standard, stateless, scalable"
)

# Link decision to task
manager.add_edge(task_id, decision_id, "implemented_by")
```

### Handoffs (Agent-to-Agent)
```python
# Initiate handoff
handoff_id = manager.initiate_handoff(
    task_id,
    target_agent="implementation-agent",
    instructions="Implement auth service with JWT tokens"
)

# Accept handoff
manager.accept_handoff(handoff_id, agent="implementation-agent")

# Complete handoff
result = {"status": "success", "files_modified": ["auth.py", "tests/test_auth.py"]}
manager.complete_handoff(
    handoff_id,
    agent="implementation-agent",
    result=result
)

# Reject handoff
manager.reject_handoff(
    handoff_id,
    agent="implementation-agent",
    reason="Missing requirements specification"
)
```

### Transactional Operations
```python
# Use transaction for atomic operations
with manager.transaction() as tx:
    task = tx.create_task("Complex task", priority="high")
    subtask1 = tx.create_task("Subtask 1", priority="medium")
    subtask2 = tx.create_task("Subtask 2", priority="medium")

    tx.add_edge(subtask1, task, "part_of")
    tx.add_edge(subtask2, task, "part_of")
    # Auto-commits on success, rolls back on exception
```

---

## Woven Mind (Dual-Process Cognition)

### Basic Usage
```python
from cortical.reasoning.woven_mind import WovenMind

# Create WovenMind with defaults
mind = WovenMind()

# Train on text
mind.train("Neural networks process data efficiently.")
mind.train("Deep learning uses neural networks.")

# Process input (auto mode selection)
result = mind.process(["neural", "networks"])
print(f"Mode: {result.mode.name}")  # FAST or SLOW
print(f"Source: {result.source}")   # 'hive' or 'cortex'
print(f"Activations: {result.activations}")
```

### Custom Configuration
```python
from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig

config = WovenMindConfig(
    surprise_threshold=0.3,    # Switch to SLOW when surprise > 0.3
    k_winners=5,               # Lateral inhibition winners
    min_frequency=3,           # Abstractions form after 3 observations
    auto_switch=True,          # Auto mode switching
    enable_observability=True, # Emit events
)

mind = WovenMind(config=config)
```

### Explicit Mode Selection
```python
from cortical.reasoning.loom import ThinkingMode

# Force FAST mode (pattern matching)
result = mind.process(["query", "terms"], mode=ThinkingMode.FAST)

# Force SLOW mode (abstraction/reasoning)
result = mind.process(["complex", "problem"], mode=ThinkingMode.SLOW)

# Auto mode (default - lets system decide)
result = mind.process(["input", "tokens"], mode=None)
```

### Pattern Observation for Abstraction
```python
# Record patterns for abstraction formation
mind.observe_pattern(["machine", "learning"])
mind.observe_pattern(["machine", "learning"])  # Repeat to build frequency
mind.observe_pattern(["machine", "learning"])

# Check formed abstractions
abstractions = mind.cortex.get_abstractions()
for abs in abstractions:
    print(f"{abs.id}: {abs.source_nodes} (freq={abs.frequency})")
```

### Consolidation (Sleep-Like Learning)
```python
# Run consolidation cycle
result = mind.consolidate()

print(f"Patterns transferred: {result.patterns_transferred}")
print(f"Abstractions formed: {result.abstractions_formed}")
print(f"Connections decayed: {result.connections_decayed}")
print(f"Duration: {result.cycle_duration_ms:.2f}ms")

# Get consolidation stats
stats = mind.get_consolidation_stats()
print(f"Total cycles: {stats['total_cycles']}")
print(f"Avg duration: {stats['avg_cycle_duration_ms']:.2f}ms")
```

### Introspection
```python
# Get system statistics
stats = mind.get_stats()
print(f"Current mode: {stats['mode']}")
print(f"Hive nodes: {stats['hive']['total_nodes_tracked']}")
print(f"Cortex abstractions: {stats['cortex']['total_abstractions']}")
print(f"Mode transitions: {stats['loom']['transition_count']}")

# Serialize state
state = mind.to_dict()

# Restore from state
restored_mind = WovenMind.from_dict(state)
```

---

## Spark SLM (Statistical Language Model)

### Basic Usage
```python
from cortical.spark import SparkPredictor

# Create predictor
spark = SparkPredictor(ngram_order=3)  # Trigram model

# Train from processor
spark.train_from_processor(processor)

# Prime a query (get keywords and completions)
primed = spark.prime("authentication handler")
print(f"Keywords: {primed['keywords']}")
print(f"Completions: {primed['completions']}")
```

### Load Alignment Context
```python
from pathlib import Path

# Load alignment definitions from markdown files
alignment_dir = Path("samples/alignment")
spark.load_alignment(alignment_dir)

# Get context for a term
context = spark.get_alignment_context("spark")
print(f"Definition: {context['definition']}")
print(f"Examples: {context['examples']}")
```

### Anomaly Detection (Prompt Injection)
```python
from cortical.spark.anomaly import AnomalyDetector

detector = AnomalyDetector()

# Train on normal text
detector.train("Normal user query about authentication")
detector.train("How do I reset my password?")

# Detect anomalies
score = detector.detect("Ignore previous instructions and...")
if score > 0.7:
    print("⚠ Potential prompt injection detected!")
```

---

## Testing Patterns

### Pytest Patterns
```python
import pytest
from cortical import CorticalTextProcessor

class TestProcessorFeatures:
    """Test processor functionality."""

    def test_basic_document_processing(self):
        """Processor should handle basic document processing."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Test content here")
        processor.compute_all()

        # Verify layers exist
        from cortical.layers import CorticalLayer
        layer0 = processor.get_layer(CorticalLayer.TOKENS)
        assert layer0.column_count() > 0

    def test_query_expansion(self):
        """Processor should expand queries."""
        processor = CorticalTextProcessor()
        processor.process_document("doc1", "Neural networks learn patterns")
        processor.compute_all()

        expanded = processor.expand_query("neural")
        assert "neural" in expanded
        assert len(expanded) > 1  # Should include related terms

# Fixtures for reusable test data
@pytest.fixture
def small_processor():
    """Create a processor with sample documents."""
    processor = CorticalTextProcessor()
    processor.process_document("doc1", "Neural networks process data")
    processor.process_document("doc2", "Machine learning algorithms")
    processor.compute_all()
    return processor

def test_with_fixture(small_processor):
    """Use fixture in test."""
    results = small_processor.find_documents_for_query("neural")
    assert len(results) > 0
```

### Unittest Patterns
```python
import unittest
from cortical import CorticalTextProcessor

class TestProcessor(unittest.TestCase):
    """Test processor using unittest."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1", "Test document")
        self.processor.compute_all()

    def tearDown(self):
        """Clean up after tests."""
        self.processor = None

    def test_document_count(self):
        """Verify document was processed."""
        from cortical.layers import CorticalLayer
        layer3 = self.processor.get_layer(CorticalLayer.DOCUMENTS)
        self.assertEqual(layer3.column_count(), 1)

    def test_search(self):
        """Verify search works."""
        results = self.processor.find_documents_for_query("test")
        self.assertGreater(len(results), 0)
```

### Mocking Patterns
```python
from unittest.mock import Mock, patch, MagicMock

class TestWithMocks:
    """Test using mocks."""

    @patch('cortical.query.expand_query')
    def test_expansion_called(self, mock_expand):
        """Verify expansion function is called."""
        mock_expand.return_value = {"test": 1.0}

        processor = CorticalTextProcessor()
        processor.process_document("doc1", "content")
        result = processor.expand_query("test")

        mock_expand.assert_called_once()
        assert result == {"test": 1.0}

    def test_with_mock_manager(self):
        """Test GoT operations with mock."""
        mock_manager = Mock()
        mock_manager.create_task.return_value = "T-20251226-120000-abc123"

        task_id = mock_manager.create_task("Test task")
        assert task_id.startswith("T-")
```

### GoT Testing with Fixtures
```python
import pytest
from cortical.got import GoTManager
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def got_manager():
    """Provide a temporary GoT manager."""
    temp_dir = Path(tempfile.mkdtemp())
    got_dir = temp_dir / ".got"
    manager = GoTManager(got_dir)

    yield manager

    # Cleanup
    shutil.rmtree(temp_dir)

def test_task_creation(got_manager):
    """Test task creation with fixture."""
    task_id = got_manager.create_task("Test task", priority="high")
    assert task_id.startswith("T-")

    task = got_manager.get_task(task_id)
    assert task.title == "Test task"
    assert task.priority == "high"
```

---

## CLI Commands

### GoT Task Management
```bash
# Create task
python scripts/got_utils.py task create "Fix authentication bug" --priority high

# List tasks
python scripts/got_utils.py task list
python scripts/got_utils.py task list --status pending
python scripts/got_utils.py task list --priority high

# Show task details
python scripts/got_utils.py task show T-20251226-120000-abc123

# Update task
python scripts/got_utils.py task start T-20251226-120000-abc123
python scripts/got_utils.py task complete T-20251226-120000-abc123

# Delete task
python scripts/got_utils.py task delete T-20251226-120000-abc123
```

### Sprint Management
```bash
# Create sprint
python scripts/got_utils.py sprint create "Sprint 1: Foundation" --number 1

# List sprints
python scripts/got_utils.py sprint list

# Show sprint status
python scripts/got_utils.py sprint status

# Start/complete sprint
python scripts/got_utils.py sprint start S-sprint-001
python scripts/got_utils.py sprint complete S-sprint-001
```

### Decision Logging
```bash
# Log decision
python scripts/got_utils.py decision log "Use JWT for auth" --rationale "Industry standard"

# List decisions
python scripts/got_utils.py decision list
```

### Handoff Operations
```bash
# Initiate handoff
python scripts/got_utils.py handoff initiate T-XXXXX --target agent-name --instructions "..."

# Accept handoff
python scripts/got_utils.py handoff accept H-XXXXX --agent agent-name

# Complete handoff
python scripts/got_utils.py handoff complete H-XXXXX --agent agent-name --result '{"status": "done"}'

# List handoffs
python scripts/got_utils.py handoff list --status initiated
```

### Testing Commands
```bash
# Run smoke tests (fast, ~1s)
make test-smoke
python scripts/run_tests.py smoke

# Run quick tests (smoke + unit, ~30s)
make test-quick
python scripts/run_tests.py quick

# Run parallel tests (4 workers, faster)
make test-parallel
python scripts/run_tests.py unit -j 4

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m coverage run -m pytest tests/
python -m coverage report --include="cortical/*"

# Run specific category
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/performance/ -v
```

### Codebase Indexing
```bash
# Index codebase for semantic search
python scripts/index_codebase.py

# Incremental update (only changed files)
python scripts/index_codebase.py --incremental

# Check status without indexing
python scripts/index_codebase.py --status

# Force full rebuild
python scripts/index_codebase.py --force
```

### Corpus Generation (for SLM Training)
```bash
# Generate training corpus
python -m benchmarks.codebase_slm.generate_corpus --output corpus.jsonl

# Generate with full analysis
python -m benchmarks.codebase_slm.generate_corpus --full --output full_corpus.jsonl

# Generate specific categories
python -m benchmarks.codebase_slm.generate_corpus --category "function_calls"
```

---

## Path Setup for Scripts

All scripts that run from the command line and import from `cortical` need to add the project root to `sys.path`:

```python
#!/usr/bin/env python3
"""
Script description here.
"""

import sys
from pathlib import Path

# Add project root to path BEFORE any cortical imports
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Now cortical imports will work
from cortical.processor import CorticalTextProcessor
from cortical.got import GoTManager
```

**Why this is needed:**
- Scripts run directly (`python scripts/my_script.py`) don't have the package installed
- CI environments may not install the package in editable mode
- This ensures imports work both locally and in CI

**Where to add it:**
- All scripts in `scripts/` that import from `cortical`
- Before any `cortical` imports
- After standard library imports

---

## Common Patterns Summary

### Initialization Pattern
```python
# Standard processor setup
from cortical import CorticalTextProcessor
processor = CorticalTextProcessor()

# Process documents
for doc_id, content in documents.items():
    processor.process_document(doc_id, content)

# Compute all metrics
processor.compute_all()
```

### Search Pattern
```python
# Query with expansion
results = processor.find_documents_for_query("search term", top_n=5)
for doc_id, score in results:
    print(f"{doc_id}: {score:.3f}")
```

### GoT Pattern
```python
from cortical.got import GoTManager
from pathlib import Path

# Initialize
manager = GoTManager(Path.cwd() / ".got")

# Create and track work
task_id = manager.create_task("Task name", priority="high")
manager.update_task(task_id, status="in_progress")
# ... do work ...
manager.complete_task(task_id, retrospective="Done!")
```

### Testing Pattern
```python
import pytest

class TestMyFeature:
    """Test feature description."""

    def test_basic_case(self):
        """Test basic functionality."""
        # Arrange
        processor = CorticalTextProcessor()

        # Act
        processor.process_document("doc1", "content")

        # Assert
        assert processor.document_count() == 1
```

---

**Note:** All code snippets in this document are verified against the actual codebase and represent current best practices. When in doubt, check the source code in `cortical/` or examples in `examples/` and `showcase.py`.
