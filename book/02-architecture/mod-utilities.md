---
title: "Utilities"
generated: "2025-12-16T17:02:01.489198Z"
generator: "architecture"
source_files:
  - "/home/user/Opus-code-test/cortical/patterns.py"
  - "/home/user/Opus-code-test/cortical/code_concepts.py"
  - "/home/user/Opus-code-test/cortical/diff.py"
  - "/home/user/Opus-code-test/cortical/gaps.py"
  - "/home/user/Opus-code-test/cortical/fingerprint.py"
  - "/home/user/Opus-code-test/cortical/mcp_server.py"
  - "/home/user/Opus-code-test/cortical/fluent.py"
  - "/home/user/Opus-code-test/cortical/cli_wrapper.py"
tags:
  - architecture
  - modules
  - utilities
---

# Utilities

Utility modules supporting various features.

## Modules

- **cli_wrapper.py**: CLI wrapper framework for collecting context and triggering actions.
- **code_concepts.py**: Code Concepts Module
- **diff.py**: Semantic Diff Module
- **fingerprint.py**: Fingerprint Module
- **fluent.py**: Fluent API for CorticalTextProcessor - chainable method interface.
- **gaps.py**: Gaps Module
- **mcp_server.py**: MCP (Model Context Protocol) Server for Cortical Text Processor.
- **patterns.py**: Code Pattern Detection Module


## cli_wrapper.py

CLI wrapper framework for collecting context and triggering actions.

Design philosophy: QUIET BY DEFAULT, POWERFUL WHEN NEEDED.

Most of the time you just want to run a command and check if it worked...


### Classes

#### GitContext

Git repository context information.

**Methods:**

- `collect`
- `to_dict`

#### ExecutionContext

Complete context for a CLI command execution.

**Methods:**

- `to_dict`
- `to_json`
- `summary`

#### HookType

Types of hooks that can be registered.

#### HookRegistry

Registry for CLI execution hooks.

**Methods:**

- `register`
- `register_pre`
- `register_post`
- `register_success`
- `register_error`
- `get_hooks`
- `trigger`

#### CLIWrapper

Wrapper for CLI command execution with context collection and hooks.

**Methods:**

- `run`
- `on_success`
- `on_error`
- `on_complete`

#### TaskCompletionManager

Manager for task completion triggers and context window management.

**Methods:**

- `on_task_complete`
- `on_any_complete`
- `handle_completion`
- `get_session_summary`
- `should_trigger_reindex`

#### ContextWindowManager

Manages context window state based on CLI execution history.

**Methods:**

- `add_execution`
- `add_file_read`
- `get_recent_files`
- `get_context_summary`
- `suggest_pruning`

#### Session

Track a sequence of commands as a session.

**Methods:**

- `run`
- `should_reindex`
- `summary`
- `results`
- `success_rate`
- `all_passed`
- `modified_files`

#### TaskCheckpoint

Save/restore context state when switching between tasks.

**Methods:**

- `save`
- `load`
- `list_tasks`
- `delete`
- `summarize`

### Functions

#### create_wrapper_with_completion_manager

```python
create_wrapper_with_completion_manager() -> Tuple[CLIWrapper, TaskCompletionManager]
```

Create a CLIWrapper with an attached TaskCompletionManager.

#### run_with_context

```python
run_with_context(command: Union[str, List[str]], **kwargs) -> ExecutionContext
```

Convenience function to run a command with full context collection.

#### run

```python
run(command: Union[str, List[str]], git: bool = False, timeout: Optional[float] = None, cwd: Optional[str] = None) -> ExecutionContext
```

Run a command. That's it.

#### test_then_commit

```python
test_then_commit(test_cmd: Union[str, List[str]] = 'python -m unittest discover -s tests', message: str = 'Update', add_all: bool = True) -> Tuple[bool, List[ExecutionContext]]
```

Run tests, commit only if they pass.

#### commit_and_push

```python
commit_and_push(message: str, add_all: bool = True, branch: Optional[str] = None) -> Tuple[bool, List[ExecutionContext]]
```

Add, commit, and push in one go.

#### sync_with_main

```python
sync_with_main(main_branch: str = 'main') -> Tuple[bool, List[ExecutionContext]]
```

Fetch and rebase current branch on main.

#### GitContext.collect

```python
GitContext.collect(cls, cwd: Optional[str] = None) -> 'GitContext'
```

Collect git context from current directory.

#### GitContext.to_dict

```python
GitContext.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary.

#### ExecutionContext.to_dict

```python
ExecutionContext.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for serialization.

#### ExecutionContext.to_json

```python
ExecutionContext.to_json(self, indent: int = 2) -> str
```

Convert to JSON string.

#### ExecutionContext.summary

```python
ExecutionContext.summary(self) -> str
```

Return a concise summary string.

#### HookRegistry.register

```python
HookRegistry.register(self, hook_type: HookType, callback: HookCallback, pattern: Optional[str] = None) -> None
```

Register a hook callback.

#### HookRegistry.register_pre

```python
HookRegistry.register_pre(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for pre-execution hooks.

#### HookRegistry.register_post

```python
HookRegistry.register_post(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for post-execution hooks.

#### HookRegistry.register_success

```python
HookRegistry.register_success(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for success hooks.

#### HookRegistry.register_error

```python
HookRegistry.register_error(self, pattern: Optional[str], callback: HookCallback) -> None
```

Convenience method for error hooks.

#### HookRegistry.get_hooks

```python
HookRegistry.get_hooks(self, hook_type: HookType, command: List[str]) -> List[HookCallback]
```

Get all hooks that should be triggered for a command.

#### HookRegistry.trigger

```python
HookRegistry.trigger(self, hook_type: HookType, context: ExecutionContext) -> None
```

Trigger all matching hooks.

#### CLIWrapper.run

```python
CLIWrapper.run(self, command: Union[str, List[str]], cwd: Optional[str] = None, timeout: Optional[float] = None, env: Optional[Dict[str, str]] = None, **kwargs) -> ExecutionContext
```

Execute a command with context collection and hooks.

#### CLIWrapper.on_success

```python
CLIWrapper.on_success(self, pattern: Optional[str] = None)
```

Decorator to register a success hook.

#### CLIWrapper.on_error

```python
CLIWrapper.on_error(self, pattern: Optional[str] = None)
```

Decorator to register an error hook.

#### CLIWrapper.on_complete

```python
CLIWrapper.on_complete(self, pattern: Optional[str] = None)
```

Decorator to register a completion hook (success or failure).

#### TaskCompletionManager.on_task_complete

```python
TaskCompletionManager.on_task_complete(self, task_type: str, callback: HookCallback) -> None
```

Register a callback for when a specific task type completes.

#### TaskCompletionManager.on_any_complete

```python
TaskCompletionManager.on_any_complete(self, callback: HookCallback) -> None
```

Register a callback for any task completion.

#### TaskCompletionManager.handle_completion

```python
TaskCompletionManager.handle_completion(self, context: ExecutionContext) -> None
```

Handle task completion and trigger appropriate callbacks.

#### TaskCompletionManager.get_session_summary

```python
TaskCompletionManager.get_session_summary(self) -> Dict[str, Any]
```

Get summary of all tasks completed in this session.

#### TaskCompletionManager.should_trigger_reindex

```python
TaskCompletionManager.should_trigger_reindex(self) -> bool
```

Determine if corpus should be re-indexed based on session activity.

#### ContextWindowManager.add_execution

```python
ContextWindowManager.add_execution(self, context: ExecutionContext) -> None
```

Add an execution to the context window.

#### ContextWindowManager.add_file_read

```python
ContextWindowManager.add_file_read(self, filepath: str) -> None
```

Track that a file was read.

#### ContextWindowManager.get_recent_files

```python
ContextWindowManager.get_recent_files(self, limit: int = 10) -> List[str]
```

Get most recently accessed files.

#### ContextWindowManager.get_context_summary

```python
ContextWindowManager.get_context_summary(self) -> Dict[str, Any]
```

Get a summary of current context window state.

#### ContextWindowManager.suggest_pruning

```python
ContextWindowManager.suggest_pruning(self) -> List[str]
```

Suggest files that could be pruned from context.

#### Session.run

```python
Session.run(self, command: Union[str, List[str]], **kwargs) -> ExecutionContext
```

Run a command within this session.

#### Session.should_reindex

```python
Session.should_reindex(self) -> bool
```

Check if corpus re-indexing is recommended based on session activity.

#### Session.summary

```python
Session.summary(self) -> Dict[str, Any]
```

Get a summary of this session's activity.

#### Session.results

```python
Session.results(self) -> List[ExecutionContext]
```

All command results from this session.

#### Session.success_rate

```python
Session.success_rate(self) -> float
```

Fraction of commands that succeeded (0.0 to 1.0).

#### Session.all_passed

```python
Session.all_passed(self) -> bool
```

True if all commands in this session succeeded.

#### Session.modified_files

```python
Session.modified_files(self) -> List[str]
```

List of files modified during this session (from git context).

#### TaskCheckpoint.save

```python
TaskCheckpoint.save(self, task_name: str, context: Dict[str, Any]) -> None
```

Save context for a task.

#### TaskCheckpoint.load

```python
TaskCheckpoint.load(self, task_name: str) -> Optional[Dict[str, Any]]
```

Load context for a task. Returns None if not found.

#### TaskCheckpoint.list_tasks

```python
TaskCheckpoint.list_tasks(self) -> List[str]
```

List all saved task checkpoints.

#### TaskCheckpoint.delete

```python
TaskCheckpoint.delete(self, task_name: str) -> bool
```

Delete a checkpoint. Returns True if deleted.

#### TaskCheckpoint.summarize

```python
TaskCheckpoint.summarize(self, task_name: str) -> Optional[str]
```

Get a one-line summary of a task checkpoint.

### Dependencies

**Standard Library:**

- `dataclasses.asdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `datetime.datetime`
- `enum.Enum`
- ... and 15 more



## code_concepts.py

Code Concepts Module
====================

Programming concept groups for semantic code search.

Maps common programming synonyms and related terms to enable
intent-based code retrieval. When a develo...


### Functions

#### get_related_terms

```python
get_related_terms(term: str, max_terms: int = 5) -> List[str]
```

Get programming terms related to the given term.

#### expand_code_concepts

```python
expand_code_concepts(terms: List[str], max_expansions_per_term: int = 3, weight: float = 0.6) -> Dict[str, float]
```

Expand a list of terms using code concept groups.

#### get_concept_group

```python
get_concept_group(term: str) -> List[str]
```

Get the concept group names a term belongs to.

#### list_concept_groups

```python
list_concept_groups() -> List[str]
```

List all available concept group names.

#### get_group_terms

```python
get_group_terms(group_name: str) -> List[str]
```

Get all terms in a concept group.

### Dependencies

**Standard Library:**

- `typing.Dict`
- `typing.FrozenSet`
- `typing.List`
- `typing.Set`



## diff.py

Semantic Diff Module
====================

Provides "What Changed?" functionality for comparing:
- Two versions of a document
- Two processor states
- Before/after states of a corpus

This goes beyond...


### Classes

#### TermChange

Represents a change to a term/concept.

**Methods:**

- `pagerank_delta`
- `tfidf_delta`
- `documents_added`
- `documents_removed`

#### RelationChange

Represents a change to a semantic relation.

#### ClusterChange

Represents a change to concept clustering.

#### SemanticDiff

Complete semantic diff between two states.

**Methods:**

- `summary`
- `to_dict`

### Functions

#### compare_processors

```python
compare_processors(old_processor: 'CorticalTextProcessor', new_processor: 'CorticalTextProcessor', top_movers: int = 20, min_pagerank_delta: float = 0.0001) -> SemanticDiff
```

Compare two processor states to find semantic differences.

#### compare_documents

```python
compare_documents(processor: 'CorticalTextProcessor', doc_id_old: str, doc_id_new: str) -> Dict[str, Any]
```

Compare two documents within the same corpus.

#### what_changed

```python
what_changed(processor: 'CorticalTextProcessor', old_content: str, new_content: str, temp_doc_prefix: str = '_diff_temp_') -> Dict[str, Any]
```

Compare two text contents to show what changed semantically.

#### TermChange.pagerank_delta

```python
TermChange.pagerank_delta(self) -> Optional[float]
```

Change in PageRank importance.

#### TermChange.tfidf_delta

```python
TermChange.tfidf_delta(self) -> Optional[float]
```

Change in TF-IDF score.

#### TermChange.documents_added

```python
TermChange.documents_added(self) -> Set[str]
```

Documents where this term newly appears.

#### TermChange.documents_removed

```python
TermChange.documents_removed(self) -> Set[str]
```

Documents where this term no longer appears.

#### SemanticDiff.summary

```python
SemanticDiff.summary(self) -> str
```

Generate a human-readable summary of changes.

#### SemanticDiff.to_dict

```python
SemanticDiff.to_dict(self) -> Dict[str, Any]
```

Convert to dictionary for serialization.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `dataclasses.dataclass`
- `dataclasses.field`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- ... and 7 more



## fingerprint.py

Fingerprint Module
==================

Semantic fingerprinting for code comparison and similarity analysis.

A fingerprint is an interpretable representation of a text's semantic
content, including te...


### Classes

#### SemanticFingerprint

Structured representation of a text's semantic fingerprint.

### Functions

#### compute_fingerprint

```python
compute_fingerprint(text: str, tokenizer: Tokenizer, layers: Optional[Dict[CorticalLayer, HierarchicalLayer]] = None, top_n: int = 20) -> SemanticFingerprint
```

Compute the semantic fingerprint of a text.

#### compare_fingerprints

```python
compare_fingerprints(fp1: SemanticFingerprint, fp2: SemanticFingerprint) -> Dict[str, Any]
```

Compare two fingerprints and compute similarity metrics.

#### explain_fingerprint

```python
explain_fingerprint(fp: SemanticFingerprint, top_n: int = 10) -> Dict[str, Any]
```

Generate a human-readable explanation of a fingerprint.

#### explain_similarity

```python
explain_similarity(fp1: SemanticFingerprint, fp2: SemanticFingerprint, comparison: Optional[Dict[str, Any]] = None) -> str
```

Generate a human-readable explanation of why two fingerprints are similar.

### Dependencies

**Standard Library:**

- `code_concepts.get_concept_group`
- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `math`
- ... and 7 more



## fluent.py

Fluent API for CorticalTextProcessor - chainable method interface.

Example:
    from cortical import FluentProcessor

    # Simple usage
    results = (FluentProcessor()
        .add_document("doc1",...


### Classes

#### FluentProcessor

Fluent/chainable API wrapper for CorticalTextProcessor.

**Methods:**

- `from_existing`
- `from_files`
- `from_directory`
- `load`
- `add_document`
- `add_documents`
- `with_config`
- `with_tokenizer`
- `build`
- `save`
- `search`
- `fast_search`
- `search_passages`
- `expand`
- `processor`
- `is_built`

### Functions

#### FluentProcessor.from_existing

```python
FluentProcessor.from_existing(cls, processor: CorticalTextProcessor) -> 'FluentProcessor'
```

Create a FluentProcessor from an existing CorticalTextProcessor.

#### FluentProcessor.from_files

```python
FluentProcessor.from_files(cls, file_paths: List[Union[str, Path]], tokenizer: Optional[Tokenizer] = None, config: Optional[CorticalConfig] = None) -> 'FluentProcessor'
```

Create a processor from a list of files.

#### FluentProcessor.from_directory

```python
FluentProcessor.from_directory(cls, directory: Union[str, Path], pattern: str = '*.txt', recursive: bool = False, tokenizer: Optional[Tokenizer] = None, config: Optional[CorticalConfig] = None) -> 'FluentProcessor'
```

Create a processor from all files in a directory.

#### FluentProcessor.load

```python
FluentProcessor.load(cls, path: Union[str, Path]) -> 'FluentProcessor'
```

Load a processor from a saved file.

#### FluentProcessor.add_document

```python
FluentProcessor.add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> 'FluentProcessor'
```

Add a document to the processor (chainable).

#### FluentProcessor.add_documents

```python
FluentProcessor.add_documents(self, documents: Union[Dict[str, str], List[Tuple[str, str]], List[Tuple[str, str, Dict]]]) -> 'FluentProcessor'
```

Add multiple documents at once (chainable).

#### FluentProcessor.with_config

```python
FluentProcessor.with_config(self, config: CorticalConfig) -> 'FluentProcessor'
```

Set configuration (chainable).

#### FluentProcessor.with_tokenizer

```python
FluentProcessor.with_tokenizer(self, tokenizer: Tokenizer) -> 'FluentProcessor'
```

Set custom tokenizer (chainable).

#### FluentProcessor.build

```python
FluentProcessor.build(self, verbose: bool = True, build_concepts: bool = True, pagerank_method: str = 'standard', connection_strategy: str = 'document_overlap', cluster_strictness: float = 1.0, bridge_weight: float = 0.0, show_progress: bool = False) -> 'FluentProcessor'
```

Build the processor by computing all analysis phases (chainable).

#### FluentProcessor.save

```python
FluentProcessor.save(self, path: Union[str, Path]) -> 'FluentProcessor'
```

Save the processor to disk (chainable).

#### FluentProcessor.search

```python
FluentProcessor.search(self, query: str, top_n: int = 5, use_expansion: bool = True, use_semantic: bool = True) -> List[Tuple[str, float]]
```

Search for documents matching the query.

#### FluentProcessor.fast_search

```python
FluentProcessor.fast_search(self, query: str, top_n: int = 5, candidate_multiplier: int = 3, use_code_concepts: bool = True) -> List[Tuple[str, float]]
```

Fast document search with pre-filtering.

#### FluentProcessor.search_passages

```python
FluentProcessor.search_passages(self, query: str, top_n: int = 5, chunk_size: Optional[int] = None, overlap: Optional[int] = None, use_expansion: bool = True) -> List[Tuple[str, str, int, int, float]]
```

Search for passage chunks matching the query.

#### FluentProcessor.expand

```python
FluentProcessor.expand(self, query: str, max_expansions: Optional[int] = None, use_variants: bool = True, use_code_concepts: bool = False) -> Dict[str, float]
```

Expand a query with related terms.

#### FluentProcessor.processor

```python
FluentProcessor.processor(self) -> CorticalTextProcessor
```

Access the underlying CorticalTextProcessor instance.

#### FluentProcessor.is_built

```python
FluentProcessor.is_built(self) -> bool
```

Check if the processor has been built.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `os`
- `pathlib.Path`
- `processor.CorticalTextProcessor`
- `tokenizer.Tokenizer`
- ... and 6 more



## gaps.py

Gaps Module
===========

Knowledge gap detection and anomaly analysis.

Identifies:
- Isolated documents that don't connect well to the corpus
- Weakly covered topics (few documents)
- Bridge opportun...


### Functions

#### analyze_knowledge_gaps

```python
analyze_knowledge_gaps(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str]) -> Dict
```

Analyze the corpus to identify potential knowledge gaps.

#### detect_anomalies

```python
detect_anomalies(layers: Dict[CorticalLayer, HierarchicalLayer], documents: Dict[str, str], threshold: float = 0.3) -> List[Dict]
```

Detect documents that don't fit well with the rest of the corpus.

### Dependencies

**Standard Library:**

- `analysis.cosine_similarity`
- `collections.defaultdict`
- `layers.CorticalLayer`
- `layers.HierarchicalLayer`
- `math`
- ... and 5 more



## mcp_server.py

MCP (Model Context Protocol) Server for Cortical Text Processor.

Provides an MCP server interface for AI agents to integrate with the
Cortical Text Processor, enabling semantic search, query expansio...


### Classes

#### CorticalMCPServer

MCP Server wrapper for CorticalTextProcessor.

**Methods:**

- `run`

### Functions

#### create_mcp_server

```python
create_mcp_server(corpus_path: Optional[str] = None, config: Optional[CorticalConfig] = None) -> CorticalMCPServer
```

Create a Cortical MCP Server instance.

#### main

```python
main()
```

Main entry point for running the MCP server from command line.

#### CorticalMCPServer.run

```python
CorticalMCPServer.run(self, transport: str = 'stdio')
```

Run the MCP server.

### Dependencies

**Standard Library:**

- `config.CorticalConfig`
- `logging`
- `mcp.server.FastMCP`
- `os`
- `pathlib.Path`
- ... and 5 more



## patterns.py

Code Pattern Detection Module
==============================

Detects common programming patterns in indexed code.

Identifies design patterns, idioms, and code structures including:
- Singleton patte...


### Functions

#### detect_patterns_in_text

```python
detect_patterns_in_text(text: str, patterns: Optional[List[str]] = None) -> Dict[str, List[int]]
```

Detect programming patterns in a text string.

#### detect_patterns_in_documents

```python
detect_patterns_in_documents(documents: Dict[str, str], patterns: Optional[List[str]] = None) -> Dict[str, Dict[str, List[int]]]
```

Detect patterns across multiple documents.

#### get_pattern_summary

```python
get_pattern_summary(pattern_results: Dict[str, List[int]]) -> Dict[str, int]
```

Summarize pattern detection results by counting occurrences.

#### get_patterns_by_category

```python
get_patterns_by_category(pattern_results: Dict[str, List[int]]) -> Dict[str, Dict[str, int]]
```

Group pattern results by category.

#### get_pattern_description

```python
get_pattern_description(pattern_name: str) -> Optional[str]
```

Get the description for a pattern.

#### get_pattern_category

```python
get_pattern_category(pattern_name: str) -> Optional[str]
```

Get the category for a pattern.

#### list_all_patterns

```python
list_all_patterns() -> List[str]
```

List all available pattern names.

#### list_patterns_by_category

```python
list_patterns_by_category(category: str) -> List[str]
```

List all patterns in a specific category.

#### list_all_categories

```python
list_all_categories() -> List[str]
```

List all pattern categories.

#### format_pattern_report

```python
format_pattern_report(pattern_results: Dict[str, List[int]], show_lines: bool = False) -> str
```

Format pattern detection results as a human-readable report.

#### get_corpus_pattern_statistics

```python
get_corpus_pattern_statistics(doc_patterns: Dict[str, Dict[str, List[int]]]) -> Dict[str, any]
```

Compute statistics across all documents.

### Dependencies

**Standard Library:**

- `collections.defaultdict`
- `re`
- `typing.Dict`
- `typing.List`
- `typing.Optional`
- ... and 2 more



---

*This chapter is part of [The Cortical Chronicles](../README.md), a self-documenting book generated by the Cortical Text Processor.*
