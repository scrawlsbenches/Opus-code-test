# Q&A Knowledge Base

This document contains question-answer pairs for training a repository-native SLM.

---

## File Location Questions

Q: Where is PageRank implemented?
A: cortical/analysis/pagerank.py

Q: Where is the GoT manager?
A: cortical/got/api.py

Q: Where is TF-IDF implemented?
A: cortical/analysis/tfidf.py

Q: Where is the clustering algorithm?
A: cortical/analysis/clustering.py (Louvain community detection)

Q: Where is the tokenizer?
A: cortical/tokenizer.py

Q: Where is query expansion implemented?
A: cortical/query/expansion.py

Q: Where is document search implemented?
A: cortical/query/search.py

Q: Where is the main processor class?
A: cortical/processor/__init__.py (CorticalTextProcessor composed from mixins)

Q: Where is the Minicolumn data structure defined?
A: cortical/minicolumn.py

Q: Where is the configuration class?
A: cortical/config.py (CorticalConfig dataclass)

Q: Where is the persistence layer?
A: cortical/persistence.py for save/load, cortical/processor/persistence_api.py for API methods

Q: Where is semantic relation extraction?
A: cortical/semantics.py

Q: Where is the WovenMind cognitive architecture?
A: cortical/reasoning/woven_mind.py

Q: Where is the Loom (mode switching)?
A: cortical/reasoning/loom.py

Q: Where is the consolidation engine?
A: cortical/reasoning/consolidation.py

Q: Where is the Graph of Thought implementation?
A: cortical/reasoning/graph_of_thought.py for core data structures, cortical/reasoning/thought_graph.py for graph representation

Q: Where is graph persistence (WAL)?
A: cortical/reasoning/graph_persistence.py

Q: Where is the GoT query language?
A: cortical/got/query_builder.py for fluent API, cortical/got/path_finder.py for path finding

Q: Where is the SparkSLM predictor?
A: cortical/spark/predictor.py

Q: Where is anomaly detection?
A: cortical/spark/anomaly.py (AnomalyDetector for prompt injection detection)

Q: Where are N-gram models?
A: cortical/spark/ngram.py

Q: Where is the fingerprinting code?
A: cortical/fingerprint.py

Q: Where is the HierarchicalLayer class?
A: cortical/layers.py

Q: Where is the observability layer?
A: cortical/observability.py (timing, metrics collection, trace context)

Q: Where is chunk-based storage?
A: cortical/chunk_index.py (git-friendly chunk-based indexing)

Q: Where is the WAL base class?
A: cortical/wal.py (BaseWALEntry, TransactionWALEntry)

Q: Where are canonical ID generation functions?
A: cortical/utils/id_generation.py (generate_task_id, generate_plan_id, etc.)

Q: Where is checksum computation?
A: cortical/utils/checksums.py (compute_checksum)

Q: Where is the passage retrieval code?
A: cortical/query/passages.py

Q: Where is intent-based search?
A: cortical/query/intent.py

Q: Where is the graph walker?
A: cortical/got/graph_walker.py (visitor pattern for graph traversal)

---

## Concept Questions

Q: What is the Cortical Text Processor?
A: A zero-dependency Python library for hierarchical text analysis that organizes text through 4 layers inspired by visual cortex organization.

Q: What are the 4 layers in the processor?
A: Layer 0 (TOKENS) - individual words, Layer 1 (BIGRAMS) - word pairs, Layer 2 (CONCEPTS) - semantic clusters, Layer 3 (DOCUMENTS) - full documents.

Q: What is a Minicolumn?
A: The core data structure with lateral_connections, typed_connections, feedforward_connections, and feedback_connections.

Q: What is GoT?
A: Graph of Thought - a task, sprint, and decision tracking system with IDs like T-20251221-014654-d4b7 for tasks, S-sprint-017 for sprints, and EPIC-nlu for epics.

Q: What is the Hebbian Hive?
A: The System 1 (fast) component of WovenMind that performs fast pattern matching, automatic responses, and spreading activation.

Q: What is PRISM-SLM?
A: A system combining PRISM (attention routing) with a Statistical Language Model for intelligent text processing.

Q: What is the Loom?
A: The routing mechanism in WovenMind that switches between FAST (Hive) and SLOW (Cortex) modes based on surprise detection.

Q: What is WovenMind?
A: A dual-process cognitive system inspired by Kahneman's System 1/System 2 theory, combining fast automatic processing (Hive) with slow deliberate reasoning (Cortex).

Q: What is the ConsolidationEngine?
A: A sleep-like memory consolidation system that transfers patterns from the Hive to the Cortex.

Q: What is a ThoughtGraph?
A: Graph-based thought representation with typed edges for representing reasoning chains in the Graph of Thought framework.

Q: What is the QAPV cycle?
A: Question → Answer → Produce → Verify - the phases of the CognitiveLoop in the reasoning framework.

Q: What is GraphWAL?
A: Write-Ahead Log for durable graph operations in the graph persistence system.

Q: What is BM25?
A: Best Match 25 - the default scoring algorithm for code search, optimized for term frequency saturation and length normalization.

Q: What is staleness tracking?
A: A system that tracks which computations are up-to-date vs needing recalculation, preventing unnecessary recomputation.

Q: What are the core algorithms?
A: PageRank for term importance, TF-IDF for document relevance, Louvain community detection for concept clustering, co-occurrence counting for lateral connections, and pattern-based relation extraction for semantic relations.

---

## How-To Questions

Q: How do I create a task in GoT?
A: python scripts/got_utils.py task create "Title" --priority high

Q: How do I run smoke tests?
A: make test-smoke or python scripts/run_tests.py smoke

Q: How do I run all tests?
A: python scripts/run_tests.py all

Q: How do I check code coverage?
A: python -m coverage run -m pytest tests/ && python -m coverage report --include="cortical/*"

Q: How do I index the codebase?
A: python scripts/index_codebase.py --incremental

Q: How do I search the codebase?
A: python scripts/search_codebase.py "your query here"

Q: How do I validate GoT state?
A: python scripts/got_utils.py validate

Q: How do I create a sprint?
A: python scripts/got_utils.py sprint create "Title" --number N

Q: How do I view sprint status?
A: python scripts/got_utils.py sprint status

Q: How do I list all tasks?
A: python scripts/got_utils.py task list

Q: How do I start working on a task?
A: python scripts/got_utils.py task start T-XXX

Q: How do I complete a task?
A: python scripts/got_utils.py task complete T-XXX

Q: How do I run the reasoning demo?
A: python scripts/reasoning_demo.py --quick

Q: How do I run the WovenMind demo?
A: python examples/woven_mind_demo.py --section all

Q: How do I process a document?
A: processor.process_document(id, text)

Q: How do I build the network?
A: processor.compute_all()

Q: How do I search for documents?
A: processor.find_documents_for_query(query)

Q: How do I save processor state?
A: processor.save("corpus_state") for JSON (recommended) or processor.save("corpus.pkl", format='pickle') for pickle (deprecated)

Q: How do I load processor state?
A: processor = CorticalTextProcessor.load("corpus_state") - auto-detects format

Q: How do I enable metrics collection?
A: processor = CorticalTextProcessor(enable_metrics=True)

Q: How do I log a decision?
A: python scripts/got_utils.py decision log "Decision" --rationale "Why"

Q: How do I create a memory entry?
A: python scripts/new_memory.py "topic"

Q: How do I find paths in GoT graph?
A: PathFinder(manager).shortest_path(from_id, to_id)

Q: How do I query tasks fluently?
A: Query(manager).tasks().where(status="pending").execute()

---

## Architecture Questions

Q: What are the 4 layers in the processor?
A: Layer 0 (TOKENS) for individual words, Layer 1 (BIGRAMS) for word pairs, Layer 2 (CONCEPTS) for semantic clusters, and Layer 3 (DOCUMENTS) for full documents.

Q: What are the main packages in cortical/?
A: processor/ (main API), query/ (search), analysis/ (graph algorithms), reasoning/ (cognitive architecture), got/ (task tracking), spark/ (SLM), and utils/ (shared utilities).

Q: How is the processor organized?
A: The CorticalTextProcessor is composed from mixins: core.py (initialization), documents.py (document processing), compute.py (PageRank/TF-IDF/clustering), query_api.py (search), introspection.py (state inspection), and persistence_api.py (save/load).

Q: What is the query package structure?
A: Split into 8 modules: expansion.py (query expansion), search.py (document search), passages.py (passage retrieval), chunking.py (text chunking), intent.py (intent queries), definitions.py (definition search), ranking.py (multi-stage ranking), analogy.py (analogy completion), and utils.py (shared utilities).

Q: What are the reasoning framework components?
A: CognitiveLoop (QAPV cycle), ThoughtGraph (graph representation), ParallelCoordinator (parallel agents), VerificationManager (testing protocols), CrisisManager (failure recovery), and graph persistence (WAL, snapshots, git integration).

Q: What is the GoT architecture?
A: GoTManager (CRUD operations), WAL (transaction log), query_builder.py (fluent API), graph_walker.py (visitor pattern), path_finder.py (BFS/DFS), and pattern_matcher.py (subgraph matching).

Q: How does WovenMind work?
A: It orchestrates dual-process cognition with the Loom routing between Hive (System 1 fast processing) and Cortex (System 2 slow reasoning), plus ConsolidationEngine for memory transfer.

Q: What is the test organization?
A: tests/smoke/ (quick checks), tests/unit/ (isolated tests), tests/integration/ (component interaction), tests/performance/ (timing tests), tests/regression/ (bug tests), tests/behavioral/ (workflow quality).

Q: What are the core data structures?
A: Minicolumn (core unit with connections), Edge (typed connection with weight/confidence), and HierarchicalLayer (container with O(1) ID lookups via _id_index).

Q: How does staleness tracking work?
A: The processor tracks computation types (COMP_TFIDF, COMP_PAGERANK, etc.) and marks them stale when documents are added, then recomputes only stale computations when compute_all() is called.

Q: What is the persistence format?
A: JSON is the default and recommended format (git-friendly, secure, cross-platform). Pickle format is deprecated due to security concerns (RCE vulnerability).

Q: What is chunk-based indexing?
A: Git-friendly storage that saves document changes as append-only JSON files in corpus_chunks/ with unique timestamp+session filenames to avoid merge conflicts.

---

## Testing Questions

Q: What is the test coverage target?
A: 95%+ for core logic, 90%+ for error handling, current baseline is 98% (as of 2025-12-24).

Q: What is the TDD workflow?
A: RED (write failing tests) → GREEN (implement minimal code to pass) → REFACTOR (clean up while tests pass).

Q: When should I write tests?
A: BEFORE writing any implementation code - tests define the contract, implementation fulfills it.

Q: How do I run tests in parallel?
A: make test-parallel or python scripts/run_tests.py unit -j 4

Q: What are the test categories?
A: smoke (quick sanity), unit (isolated), integration (component interaction), performance (timing), regression (bug-specific), behavioral (workflow quality).

Q: How do I run pre-commit tests?
A: python scripts/run_tests.py precommit (smoke + unit + integration)

Q: What fixtures are available?
A: small_processor (25-doc corpus, session scope), shared_processor (full samples/ corpus, session scope), fresh_processor (empty, function scope), small_corpus_docs (raw documents).

---

## Command Reference Questions

Q: How do I add an edge in GoT?
A: python scripts/got_utils.py edge add SOURCE_ID TARGET_ID EDGE_TYPE

Q: How do I list edges in GoT?
A: python scripts/got_utils.py edge list [--type TYPE] [--source ID] [--target ID]

Q: How do I query what blocks a task?
A: python scripts/got_utils.py query "what blocks TASK_ID"

Q: How do I find dependencies?
A: python scripts/got_utils.py query "what depends on TASK_ID"

Q: How do I check for active tasks?
A: python scripts/got_utils.py query "active tasks"

Q: How do I initiate a handoff?
A: python scripts/got_utils.py handoff initiate TASK_ID --target AGENT --instructions "..."

Q: How do I accept a handoff?
A: python scripts/got_utils.py handoff accept HANDOFF_ID --agent AGENT

Q: How do I complete a handoff?
A: python scripts/got_utils.py handoff complete HANDOFF_ID --agent AGENT --result JSON

Q: How do I generate session memory?
A: python scripts/session_memory_generator.py --session-id ID

Q: How do I check wiki-links?
A: python scripts/resolve_wiki_links.py FILE

---

## Configuration Questions

Q: What is the default scoring algorithm?
A: BM25 (Best Match 25) optimized for code search.

Q: How do I change the scoring algorithm to TF-IDF?
A: config = CorticalConfig(scoring_algorithm='tfidf')

Q: What are the BM25 parameters?
A: bm25_k1 (term frequency saturation, 0.0-3.0, default 1.2) and bm25_b (length normalization, 0.0-1.0, default 0.75).

Q: Is GoT auto-commit enabled by default?
A: Yes, GOT_AUTO_COMMIT is ON by default. Disable with export GOT_AUTO_COMMIT=0.

Q: Is GoT auto-push enabled by default?
A: Yes, GOT_AUTO_PUSH is ON by default for claude/* branches. Disable with export GOT_AUTO_PUSH=0.

Q: What platforms are supported?
A: Linux and macOS only. Windows is not supported (uses POSIX-specific fcntl.flock() for process-safe locking).

---

## Work Priority Questions

Q: What is the work priority order?
A: 1. Security → 2. Bugs → 3. Features → 4. Documentation

Q: What should I do if a tool fails?
A: Follow the Tool Reliability Policy: STOP, ASSESS, FIX the tool, USE the fixed tool, DOCUMENT.

Q: Should I work around broken tools?
A: No, never work around broken tools with direct file manipulation. Fix the tool first, then use it.

Q: Can I edit GoT files directly?
A: Never! GoT data is transactional and event-sourced with checksum integrity. Always use GoT CLI commands.

Q: What happens if I edit GoT files directly?
A: Breaks checksum validation (auto-deleted as corrupted), breaks event log, corrupts dependency tracking, leaves orphaned edges.

---

## Performance Questions

Q: What is O(1) lookup in layers?
A: Always use layer.get_by_id(col_id) instead of iterating layer.minicolumns - the _id_index provides O(1) access.

Q: How do I avoid O(n²) patterns?
A: Use limits like max_bigrams_per_term and max_bigrams_per_doc to prevent common terms from creating millions of pairs.

Q: What was the bigram separator bug?
A: Bigrams use space separators throughout ("neural networks", not "neural_networks").

Q: How do I profile performance?
A: python scripts/profile_full_analysis.py

Q: What is incremental indexing?
A: python scripts/index_codebase.py --incremental - only re-indexes changed files for fastest updates.

---

## Integration Questions

Q: How do I use the codebase search?
A: Index with python scripts/index_codebase.py --incremental, then search with python scripts/search_codebase.py "query"

Q: How do I use interactive search?
A: python scripts/search_codebase.py --interactive

Q: What are the interactive mode commands?
A: /expand <query> (show expansion), /concepts (list clusters), /stats (corpus statistics), /quit (exit).

Q: How do I enable chunk-based indexing?
A: python scripts/index_codebase.py --incremental --use-chunks

Q: How do I compact chunks?
A: python scripts/index_codebase.py --compact --use-chunks

---

## ML Data Collection Questions

Q: Is ML data collection automatic?
A: Yes, fully automatic via SessionStart hook. Zero configuration required.

Q: Where is ML data stored?
A: .git-ml/ directory (gitignored and regeneratable via backfill).

Q: What ML data is collected?
A: Commits (with diffs), chats (query/response pairs), sessions (linking chats to commits), and actions (tool uses).

Q: How do I check ML collection stats?
A: python scripts/ml_data_collector.py stats

Q: How do I train the file prediction model?
A: python scripts/ml_file_prediction.py train

Q: How do I predict files for a task?
A: python scripts/ml_file_prediction.py predict "Add authentication feature"

---

## Debugging Questions

Q: How do I inspect layer state?
A: Iterate through processor.layers.items() to check column counts and examine specific minicolumns.

Q: How do I trace query expansion?
A: Use processor.expand_query() with max_expansions and inspect the returned term weights.

Q: How do I check semantic relations?
A: Run processor.extract_corpus_semantics() and examine processor.semantic_relations.

Q: How do I get metrics summary?
A: processor.get_metrics_summary() when metrics are enabled.

Q: How do I reset metrics?
A: processor.reset_metrics()

---

## Sub-Agent Questions

Q: When should I use sub-agents?
A: For well-defined mechanical tasks with clear specifications while keeping context-heavy decisions in main agent.

Q: What agent type should I use for exploration?
A: Explore agent (quick or thorough) for finding implementation patterns or understanding system flow.

Q: What agent type should I use for implementation?
A: general-purpose agent for writing tests and implementing features.

Q: What should I verify after sub-agent completion?
A: Always run git status and git diff to verify file changes actually persisted.

Q: What is the parallel execution pattern?
A: Spawn multiple sub-agents for independent mechanical tasks while main agent handles context-heavy decisions.

---

## Recovery Questions

Q: How do I restore cognitive state?
A: Use /context-recovery slash command or read recent memories with ls -t samples/memories/*.md | head -5

Q: What are signs of cognitive breakdown?
A: Repeating failed approaches, contradicting earlier statements, making changes without reading, asking answered questions.

Q: What is the recovery protocol?
A: DETECT breakdown type, STOP immediately, DIAGNOSE state, INFORM user clearly, RECOVER from files/memories, VERIFY consistency.

Q: How do I check branch state?
A: git branch -vv && cat .branch-state/active/*.json 2>/dev/null

Q: How do I verify previous agents' claims?
A: Trust but verify - identify claim, find evidence, verify empirically with commands, document verification before proceeding.
