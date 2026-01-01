# Codebase Test Coverage Inventory

**Purpose:** Tag every file with required test coverage
**Created:** 2025-12-31
**Status:** CLASSIFIED - Sub-agent analysis complete

## Tags Legend

| Tag | Meaning |
|-----|---------|
| `[API]` | Public API - Needs Behavioral Test |
| `[CORE]` | Core Implementation - Needs Contract Test |
| `[INTERNAL]` | Internal Helper - Unit Tests Sufficient |
| `[DEMO]` | Demo/Example - Convert to Behavioral Test |
| `[✓]` | Already covered |
| `[-]` | Not applicable (init, config, etc.) |

---

## Executive Summary

### Behavioral Test Priority (31 API Files)

These files define the public interface users interact with:

| Module | API Files | Priority |
|--------|-----------|----------|
| **cortical/processor/** | core.py, documents.py, compute.py, query_api.py, persistence_api.py, introspection.py, spark_api.py | **HIGH** |
| **cortical/query/** | search.py, passages.py, expansion.py, ranking.py, definitions.py | **HIGH** |
| **cortical/reasoning/** | woven_mind.py, loom.py, cognitive_loop.py, graph_of_thought.py, thought_graph.py, workflow.py, pubsub.py, claude_code_spawner.py, prism_got.py | **MEDIUM** |
| **cortical/got/** | api.py, query_builder.py, pattern_matcher.py, path_finder.py, graph_walker.py, orphan.py, query_api.py | **MEDIUM** |
| **cortical/spark/** | intelligence.py, predictor.py, anomaly.py, suggester.py | **MEDIUM** |
| **cortical/cel/** | container.py | **MEDIUM** |
| **cortical/ root** | fluent.py, async_api.py, diff.py, results.py | **HIGH** |

### Contract Test Priority (35 CORE Files)

These files implement performance-critical algorithms:

| Module | Core Files |
|--------|------------|
| **cortical/analysis/** | pagerank.py, tfidf.py, clustering.py, connections.py, activation.py, parallel.py, quality.py |
| **cortical/reasoning/** | loom_hive.py, loom_cortex.py, attention_router.py, context_pool.py, goal_stack.py, nested_loop.py, thought_patterns.py, homeostasis.py, loop_validator.py, metrics.py |
| **cortical/spark/** | ngram.py, transfer.py, ast_index.py, alignment.py, co_change.py, diff_tokenizer.py, intent_parser.py, git_trainer.py, quality.py |
| **cortical/got/** | versioned_store.py, schema.py, recovery.py, indexer.py, wal.py, tx_manager.py |
| **cortical/cel/** | protocols.py, events.py, dag.py, materializer.py, health.py |
| **cortical/ root** | tokenizer.py, minicolumn.py, layers.py, semantics.py, patterns.py, embeddings.py, persistence.py, wal.py |

### Demo Conversion Priority (28 Demo Files)

All `examples/` files should be converted to behavioral tests with Given-When-Then structure.

---

## Singularity Analysis: NO VIOLATIONS FOUND

**CRITICAL FINDING:** All 5 "duplicate" file pairs analyzed by sub-agents are **DISTINCT** and serve different purposes. No singularity violations exist.

| Capability | File 1 | File 2 | Relationship | Status |
|------------|--------|--------|--------------|--------|
| WAL | `cortical/wal.py` | `cortical/got/wal.py` | **Hierarchical**: Base infrastructure → GoT-specific manager | ✓ DISTINCT |
| Validation | `cortical/validation.py` | `cortical/got/validation.py` | **Layer**: Utilities vs Domain-specific rules | ✓ DISTINCT |
| Tokenizer | `cortical/tokenizer.py` | `cortical/spark/tokenizer.py` | **Domain**: Text tokenization vs Code tokenization | ✓ DISTINCT |
| Quality | `cortical/analysis/quality.py` | `cortical/spark/quality.py` | **Domain**: Clustering metrics vs Prediction metrics | ✓ DISTINCT |
| Config | `cortical/config.py` | `cortical/got/config.py` | **Subsystem**: Main config vs GoT config | ✓ DISTINCT |

---

## Root Showcases

| Lines | File | Tag |
|------:|------|-----|
| 1,067 | `showcase.py` | [DEMO] |
| 970 | `nlu_showcase.py` | [DEMO] |
| 634 | `repo_showcase.py` | [DEMO] |
| 522 | `secureshowcase.py` | [DEMO] |

---

## benchmarks/

### benchmarks/cel/

| Lines | File | Tag |
|------:|------|-----|
| 1,042 | `benchmarks.py` | [-] |
| 916 | `sanity_benchmarks.py` | [-] |
| 815 | `performance_benchmarks.py` | [-] |
| 371 | `runner.py` | [-] |
| 53 | `__init__.py` | [-] |

### benchmarks/codebase_slm/

| Lines | File | Tag |
|------:|------|-----|
| 637 | `data_augmentation.py` | [-] |
| 455 | `train_augmented.py` | [-] |
| 422 | `pln_generator.py` | [-] |
| 409 | `hybrid_pipeline.py` | [-] |
| 380 | `dialogue_generator.py` | [-] |
| 376 | `benchmark_suite.py` | [-] |
| 375 | `train_slm.py` | [-] |
| 350 | `explore_data_generators.py` | [-] |
| 238 | `explore_woven_mind.py` | [-] |
| 205 | `generate_corpus.py` | [-] |

### benchmarks/codebase_slm/generators/

| Lines | File | Tag |
|------:|------|-----|
| 548 | `pattern_generator.py` | [-] |
| 504 | `doc_extractor.py` | [-] |
| 423 | `code_extractor.py` | [-] |
| 333 | `meta_extractor.py` | [-] |
| 24 | `__init__.py` | [-] |

### benchmarks/corpus/

| Lines | File | Tag |
|------:|------|-----|
| 2,524 | `runner.py` | [-] |
| 517 | `base.py` | [-] |
| 31 | `__init__.py` | [-] |

### benchmarks/prism_slm/

| Lines | File | Tag |
|------:|------|-----|
| 733 | `generation.py` | [-] |
| 420 | `learning.py` | [-] |
| 374 | `integration.py` | [-] |
| 314 | `runner.py` | [-] |
| 50 | `__init__.py` | [-] |

### benchmarks/woven_mind/

| Lines | File | Tag |
|------:|------|-----|
| 523 | `cognitive.py` | [-] |
| 519 | `stability.py` | [-] |
| 489 | `scale.py` | [-] |
| 481 | `quality.py` | [-] |
| 379 | `base.py` | [-] |
| 266 | `runner.py` | [-] |
| 34 | `__init__.py` | [-] |

---

## cortical/

### cortical/ (root)

| Lines | File | Tag |
|------:|------|-----|
| 1,379 | `ml_storage.py` | [CORE] |
| 1,168 | `cli_wrapper.py` | [INTERNAL] |
| 927 | `semantics.py` | [CORE] |
| 818 | `wal.py` | [CORE] |
| 763 | `state_storage.py` | [CORE] |
| 624 | `diff.py` | [API] |
| 601 | `persistence.py` | [CORE] |
| 571 | `chunk_index.py` | [CORE] |
| 554 | `async_api.py` | [API] |
| 542 | `patterns.py` | [CORE] |
| 524 | `minicolumn.py` | [CORE] |
| 510 | `fluent.py` | [API] |
| 500 | `results.py` | [API] |
| 445 | `embeddings.py` | [CORE] |
| 437 | `observability.py` | [INTERNAL] |
| 426 | `tokenizer.py` | [CORE] |
| 400 | `config.py` | [-] |
| 349 | `progress.py` | [INTERNAL] |
| 315 | `fingerprint.py` | [CORE] |
| 314 | `layers.py` | [CORE] |
| 276 | `code_concepts.py` | [CORE] |
| 248 | `validation.py` | [-] |
| 245 | `gaps.py` | [CORE] |
| 210 | `top_words.py` | [CORE] |
| 161 | `types.py` | [-] |
| 114 | `constants.py` | [-] |
| 81 | `__init__.py` | [-] |

### cortical/analysis/

| Lines | File | Tag |
|------:|------|-----|
| 671 | `clustering.py` | [CORE] |
| 520 | `connections.py` | [CORE] |
| 495 | `quality.py` | [CORE] |
| 471 | `pagerank.py` | [CORE] |
| 251 | `tfidf.py` | [CORE] |
| 223 | `parallel.py` | [CORE] |
| 194 | `utils.py` | [INTERNAL] |
| 123 | `__init__.py` | [-] |
| 76 | `activation.py` | [CORE] |

### cortical/cel/

| Lines | File | Tag |
|------:|------|-----|
| 713 | `container.py` | [API] |
| 687 | `tracing.py` | [CORE] |
| 481 | `tracing_integration.py` | [CORE] |
| 445 | `config.py` | [INTERNAL] |
| 101 | `__init__.py` | [-] |

### cortical/cel/adapters/

| Lines | File | Tag |
|------:|------|-----|
| 729 | `got.py` | [INTERNAL] |
| 37 | `__init__.py` | [-] |

### cortical/cel/core/

| Lines | File | Tag |
|------:|------|-----|
| 827 | `protocols.py` | [CORE] |
| 539 | `events.py` | [CORE] |
| 481 | `references.py` | [CORE] |
| 71 | `__init__.py` | [-] |

### cortical/cel/performance/

| Lines | File | Tag |
|------:|------|-----|
| 715 | `streaming_store.py` | [CORE] |
| 523 | `optimized_dag.py` | [CORE] |
| 509 | `snapshots.py` | [CORE] |
| 508 | `entity_index.py` | [CORE] |
| 49 | `__init__.py` | [-] |

### cortical/cel/sanity/

| Lines | File | Tag |
|------:|------|-----|
| 675 | `compaction.py` | [CORE] |
| 522 | `migration.py` | [CORE] |
| 487 | `health.py` | [CORE] |
| 57 | `__init__.py` | [-] |

### cortical/cel/wisdom/

| Lines | File | Tag |
|------:|------|-----|
| 515 | `dag.py` | [CORE] |
| 488 | `materializer.py` | [CORE] |
| 421 | `semantic.py` | [CORE] |
| 37 | `__init__.py` | [-] |

### cortical/got/

| Lines | File | Tag |
|------:|------|-----|
| 2,918 | `api.py` | [API] |
| 1,449 | `query_builder.py` | [API] |
| 1,368 | `types.py` | [CORE] |
| 946 | `validation.py` | [CORE] |
| 739 | `pattern_matcher.py` | [API] |
| 650 | `versioned_store.py` | [CORE] |
| 646 | `path_finder.py` | [API] |
| 642 | `graph_walker.py` | [API] |
| 632 | `claudemd.py` | [CORE] |
| 632 | `recovery.py` | [CORE] |
| 621 | `entity_schemas.py` | [CORE] |
| 590 | `query_api.py` | [API] |
| 574 | `schema.py` | [CORE] |
| 523 | `orphan.py` | [API] |
| 466 | `wal.py` | [CORE] |
| 414 | `indexer.py` | [CORE] |
| 353 | `protocol.py` | [CORE] |
| 338 | `sync.py` | [CORE] |
| 327 | `tx_manager.py` | [CORE] |
| 327 | `__init__.py` | [-] |
| 247 | `conflict.py` | [CORE] |
| 166 | `transaction.py` | [CORE] |
| 62 | `errors.py` | [-] |
| 50 | `config.py` | [-] |

### cortical/got/cli/

| Lines | File | Tag |
|------:|------|-----|
| 711 | `doc.py` | [INTERNAL] |
| 634 | `batch.py` | [INTERNAL] |
| 587 | `task.py` | [INTERNAL] |
| 585 | `sprint.py` | [INTERNAL] |
| 512 | `analyze.py` | [INTERNAL] |
| 427 | `backup.py` | [INTERNAL] |
| 400 | `orphan.py` | [INTERNAL] |
| 368 | `query.py` | [INTERNAL] |
| 353 | `decision.py` | [INTERNAL] |
| 350 | `handoff.py` | [INTERNAL] |
| 341 | `backlog.py` | [INTERNAL] |
| 326 | `edge.py` | [INTERNAL] |
| 231 | `__init__.py` | [-] |
| 205 | `shared.py` | [INTERNAL] |

### cortical/ml_experiments/

| Lines | File | Tag |
|------:|------|-----|
| 631 | `file_prediction_adapter.py` | [CORE] |
| 511 | `dataset.py` | [CORE] |
| 430 | `experiment.py` | [CORE] |
| 389 | `metrics.py` | [CORE] |
| 362 | `utils.py` | [INTERNAL] |
| 87 | `__init__.py` | [-] |

### cortical/processor/

| Lines | File | Tag |
|------:|------|-----|
| 1,313 | `spark_api.py` | [API] |
| 1,276 | `compute.py` | [API] |
| 784 | `query_api.py` | [API] |
| 462 | `documents.py` | [API] |
| 357 | `introspection.py` | [API] |
| 260 | `persistence_api.py` | [API] |
| 224 | `core.py` | [API] |
| 74 | `__init__.py` | [-] |

### cortical/projects/

| Lines | File | Tag |
|------:|------|-----|
| 13 | `__init__.py` | [-] |
| 24 | `cli/__init__.py` | [-] |

### cortical/query/

| Lines | File | Tag |
|------:|------|-----|
| 704 | `search.py` | [API] |
| 491 | `expansion.py` | [API] |
| 469 | `ranking.py` | [API] |
| 408 | `passages.py` | [API] |
| 346 | `definitions.py` | [API] |
| 336 | `chunking.py` | [CORE] |
| 330 | `analogy.py` | [CORE] |
| 220 | `intent.py` | [CORE] |
| 185 | `__init__.py` | [-] |
| 95 | `utils.py` | [CORE] |

### cortical/reasoning/

| Lines | File | Tag |
|------:|------|-----|
| 2,044 | `graph_persistence.py` | [-] |
| 1,367 | `collaboration.py` | [-] |
| 1,340 | `production_state.py` | [INTERNAL] |
| 1,255 | `crisis_manager.py` | [-] |
| 1,224 | `thought_graph.py` | [API] |
| 1,223 | `claude_code_spawner.py` | [API] |
| 1,197 | `verification.py` | [-] |
| 1,147 | `prism_got.py` | [API] |
| 1,120 | `loom.py` | [API] |
| 970 | `cognitive_loop.py` | [API] |
| 929 | `prism_slm.py` | [-] |
| 828 | `workflow.py` | [API] |
| 808 | `pubsub.py` | [API] |
| 743 | `__init__.py` | [-] |
| 719 | `prism_pln.py` | [-] |
| 634 | `consolidation.py` | [INTERNAL] |
| 615 | `prism_attention.py` | [-] |
| 599 | `abstraction.py` | [INTERNAL] |
| 583 | `homeostasis.py` | [CORE] |
| 570 | `rejection_protocol.py` | [-] |
| 562 | `metrics.py` | [CORE] |
| 505 | `qapv_verification.py` | [-] |
| 486 | `nested_loop.py` | [CORE] |
| 484 | `thought_patterns.py` | [CORE] |
| 472 | `goal_stack.py` | [CORE] |
| 430 | `abstraction_pln.py` | [INTERNAL] |
| 416 | `loom_cortex.py` | [CORE] |
| 404 | `woven_mind.py` | [API] |
| 400 | `loom_hive.py` | [CORE] |
| 396 | `context_pool.py` | [CORE] |
| 390 | `loop_validator.py` | [CORE] |
| 378 | `graph_of_thought.py` | [API] |
| 349 | `attention_router.py` | [CORE] |

### cortical/spark/

| Lines | File | Tag |
|------:|------|-----|
| 716 | `quality.py` | [CORE] |
| 626 | `transfer.py` | [CORE] |
| 606 | `intelligence.py` | [API] |
| 567 | `diff_tokenizer.py` | [CORE] |
| 523 | `ast_index.py` | [CORE] |
| 490 | `suggester.py` | [API] |
| 471 | `co_change.py` | [CORE] |
| 468 | `intent_parser.py` | [CORE] |
| 452 | `alignment.py` | [CORE] |
| 437 | `git_trainer.py` | [CORE] |
| 434 | `ngram.py` | [CORE] |
| 373 | `predictor.py` | [API] |
| 346 | `anomaly.py` | [API] |
| 193 | `tokenizer.py` | [INTERNAL] |
| 145 | `__init__.py` | [-] |

### cortical/utils/

| Lines | File | Tag |
|------:|------|-----|
| 524 | `id_generation.py` | [INTERNAL] |
| 295 | `locking.py` | [INTERNAL] |
| 117 | `checksums.py` | [INTERNAL] |
| 100 | `persistence.py` | [INTERNAL] |
| 49 | `__init__.py` | [-] |
| 32 | `text.py` | [INTERNAL] |

---

## examples/

| Lines | File | Tag |
|------:|------|-----|
| 1,027 | `cel_demo.py` | [DEMO] |
| 860 | `woven_mind_demo.py` | [DEMO] |
| 639 | `spark_demo.py` | [DEMO] |
| 572 | `prism_got_comprehensive_demo.py` | [DEMO] |
| 550 | `got_demo.py` | [DEMO] |
| 483 | `rejection_protocol_demo.py` | [DEMO] |
| 377 | `code_evolution_demo.py` | [DEMO] |
| 339 | `reasoning_metrics_demo.py` | [DEMO] |
| 329 | `prism_got_demo.py` | [DEMO] |
| 321 | `context_pool_demo.py` | [DEMO] |
| 314 | `prism_got_demo_corpus.py` | [DEMO] |
| 312 | `subprocess_spawner_demo.py` | [DEMO] |
| 309 | `prism_got_nlu_demo.py` | [DEMO] |
| 290 | `graph_persistence_demo.py` | [DEMO] |
| 283 | `async_api_demo.py` | [DEMO] |
| 264 | `pubsub_demo.py` | [DEMO] |
| 261 | `qapv_verification_demo.py` | [DEMO] |
| 242 | `examples_results_usage.py` | [DEMO] |
| 236 | `nested_loop_demo.py` | [DEMO] |
| 204 | `git_auto_committer_demo.py` | [DEMO] |
| 197 | `parallel_demo.py` | [DEMO] |
| 194 | `demo_ci_integration.py` | [DEMO] |
| 183 | `prism_slm_demo.py` | [DEMO] |
| 183 | `repl_demo.py` | [DEMO] |
| 168 | `demo_pattern_detection.py` | [DEMO] |
| 152 | `got_dashboard_origin_demo.py` | [DEMO] |
| 132 | `observability_demo.py` | [DEMO] |
| 120 | `demo_progress.py` | [DEMO] |

---

## llm_orchestration/

| Lines | File | Tag |
|------:|------|-----|
| 1,139 | `recovery.py` | [CORE] |
| 1,131 | `learning.py` | [CORE] |
| 1,018 | `thought_patterns.py` | [CORE] |
| 977 | `evolution.py` | [CORE] |
| 971 | `protocols.py` | [CORE] |
| 918 | `agents.py` | [CORE] |
| 912 | `cognitive_state.py` | [CORE] |
| 836 | `tools.py` | [CORE] |
| 715 | `orchestration.py` | [API] |
| 540 | `types.py` | [-] |
| 528 | `agile.py` | [CORE] |
| 445 | `metrics.py` | [CORE] |
| 220 | `__init__.py` | [-] |

### llm_orchestration/examples/

| Lines | File | Tag |
|------:|------|-----|
| 344 | `learning_demo.py` | [DEMO] |
| 288 | `basic_workflow.py` | [DEMO] |
| 281 | `recovery_demo.py` | [DEMO] |
| 267 | `multi_session.py` | [DEMO] |
| 14 | `__init__.py` | [-] |

---

## scripts/

### scripts/ (root)

| Lines | File | Tag |
|------:|------|-----|
| 5,496 | `generate_book.py` | [-] |
| 4,599 | `ml_data_collector.py` | [-] |
| 3,016 | `got_utils.py` | [-] |
| 2,273 | `index_codebase.py` | [-] |
| 2,028 | `ml_file_prediction.py` | [-] |
| 1,327 | `hubris_cli.py` | [-] |
| 1,237 | `orchestration_utils.py` | [-] |
| 1,211 | `claudemd_generation_demo.py` | [-] |
| 1,050 | `repl.py` | [-] |
| 1,039 | `got_dashboard.py` | [-] |
| 930 | `benchmark_scoring.py` | [-] |
| 928 | `ascii_visualizer_animated.py` | [-] |
| 908 | `benchmark_spark.py` | [-] |
| 820 | `world_model_analysis.py` | [-] |
| 807 | `run_sprint_reasoning.py` | [-] |
| 765 | `generate_ai_metadata.py` | [-] |
| 759 | `knowledge_bridge.py` | [-] |
| 741 | `knowledge_analysis.py` | [-] |
| 739 | `ascii_codebase_art.py` | [-] |
| 728 | `spark_code_assistant.py` | [-] |
| 707 | `cognitive_pipeline.py` | [-] |
| 704 | `cognitive_integration_demo.py` | [-] |
| 690 | `migrate_got.py` | [-] |
| 666 | `verify_batch.py` | [-] |
| 642 | `generate_ascii_gifs.py` | [-] |
| 624 | `session_memory_generator.py` | [-] |
| 612 | `spark_code_intelligence.py` | [-] |
| 605 | `suggest_consolidation.py` | [-] |
| 588 | `profile_got_query.py` | [-] |
| 586 | `analyze_louvain_resolution.py` | [-] |
| 570 | `evaluate_cluster.py` | [-] |
| 558 | `question_connection.py` | [-] |
| 542 | `thought_chain.py` | [-] |
| 531 | `reasoning_demo.py` | [-] |
| 521 | `run_tests.py` | [-] |
| 516 | `llm_generate_response.py` | [-] |
| 512 | `analyze_cross_domain_bridges.py` | [-] |
| 506 | `migrate_sprints_to_got.py` | [-] |
| 504 | `search_codebase.py` | [-] |
| 463 | `verify_milestone.py` | [-] |
| 445 | `corpus_health.py` | [-] |
| 441 | `validate_reasoning_persistence.py` | [-] |
| 440 | `cognitive_demo_refined.py` | [-] |
| 439 | `branch_manifest.py` | [-] |
| 434 | `cognitive_demo.py` | [-] |
| 431 | `cognitive_demo_ast.py` | [-] |
| 427 | `session_handoff.py` | [-] |
| 420 | `suggest_related.py` | [-] |
| 417 | `generate_sample_docs.py` | [-] |
| 395 | `new_memory.py` | [-] |
| 379 | `got_priority_executor.py` | [-] |
| 345 | `profile_full_analysis.py` | [-] |
| 345 | `analyze_corpus_balance.py` | [-] |
| 344 | `migrate_ml_to_cali.py` | [-] |
| 321 | `explain_code.py` | [-] |
| 304 | `find_similar.py` | [-] |
| 290 | `ci_task_report.py` | [-] |
| 283 | `ask_codebase.py` | [-] |
| 255 | `hubris-feedback-hook.py` | [-] |
| 245 | `backfill_chat_history.py` | [-] |
| 230 | `task_diff.py` | [-] |
| 215 | `benchmark_sparsity.py` | [-] |
| 194 | `train_spark_from_git.py` | [-] |
| 189 | `cli_wrappers.py` | [-] |
| 150 | `resolve_wiki_links.py` | [-] |
| 51 | `demo_utils.py` | [-] |
| 48 | `ascii_effects.py` | [-] |
| 19 | `doc_utils.py` | [-] |

### scripts/hubris/

| Lines | File | Tag |
|------:|------|-----|
| 1,989 | `ml_file_prediction_v1.py` | [-] |
| 682 | `feedback_collector.py` | [CORE] |
| 612 | `test_calibration_tracker.py` | [-] |
| 518 | `expert_consolidator.py` | [CORE] |
| 509 | `staking.py` | [CORE] |
| 500 | `test_feedback.py` | [-] |
| 469 | `credit_account.py` | [CORE] |
| 446 | `calibration_tracker.py` | [CORE] |
| 391 | `value_signal.py` | [CORE] |
| 355 | `credit_router.py` | [CORE] |
| 304 | `voting_aggregator.py` | [CORE] |
| 304 | `micro_expert.py` | [CORE] |
| 288 | `estimate_command_data.py` | [-] |
| 279 | `expert_router.py` | [CORE] |
| 234 | `train_command_expert.py` | [-] |
| 213 | `test_calibration_demo.py` | [-] |
| 20 | `__init__.py` | [-] |

**NOTE:** `micro_expert.py` and `expert_router.py` are [CORE] components that should be moved to `cortical/` per Sovereignty Principle.

### scripts/hubris/experts/

| Lines | File | Tag |
|------:|------|-----|
| 739 | `refactor_expert.py` | [CORE] |
| 523 | `episode_expert.py` | [CORE] |
| 467 | `error_expert.py` | [CORE] |
| 411 | `test_expert.py` | [CORE] |
| 395 | `command_expert.py` | [CORE] |
| 353 | `file_expert.py` | [CORE] |
| 13 | `__init__.py` | [-] |

### scripts/ml_collector/

| Lines | File | Tag |
|------:|------|-----|
| 583 | `orchestration.py` | [CORE] |
| 557 | `persistence.py` | [CORE] |
| 515 | `chunked_storage.py` | [CORE] |
| 447 | `task_linker.py` | [CORE] |
| 347 | `session.py` | [CORE] |
| 323 | `commit.py` | [CORE] |
| 285 | `transcript.py` | [CORE] |
| 274 | `quality.py` | [CORE] |
| 219 | `stats.py` | [CORE] |
| 199 | `export.py` | [CORE] |
| 168 | `config.py` | [-] |
| 117 | `data_classes.py` | [-] |
| 106 | `ci.py` | [CORE] |
| 89 | `__init__.py` | [-] |
| 79 | `hooks.py` | [CORE] |
| 21 | `core.py` | [CORE] |

---

## Summary Statistics

| Category | Count | Description |
|----------|------:|-------------|
| **[API]** | 31 | Public APIs needing behavioral tests |
| **[CORE]** | 76 | Core implementations needing contract tests |
| **[INTERNAL]** | 31 | Internal helpers (unit tests sufficient) |
| **[DEMO]** | 32 | Demos to convert to behavioral tests |
| **[-]** | 504 | Not applicable (init, config, scripts, benchmarks, tests) |
| **TOTAL** | 674 | All Python files |

| Area | Files | Lines |
|------|------:|------:|
| cortical/ | 164 | 93,185 |
| scripts/ | 108 | 66,887 |
| tests/ | 321 | 190,868 |
| benchmarks/ | 35 | 16,530 |
| examples/ | 28 | 9,541 |
| llm_orchestration/ | 18 | 10,350 |
| **TOTAL** | **674** | **387,361** |

---

## Recommended Next Steps

### Phase 1: High Priority API Behavioral Tests

1. **cortical/processor/** (7 files) - Main user interface
2. **cortical/query/** (5 API files) - Search functionality
3. **cortical/ root** (4 API files) - fluent.py, async_api.py, diff.py, results.py

### Phase 2: Medium Priority API Behavioral Tests

4. **cortical/reasoning/** (9 API files) - Cognitive systems
5. **cortical/got/** (7 API files) - Graph of Thought
6. **cortical/spark/** (4 API files) - Code intelligence

### Phase 3: Contract Tests for Core Algorithms

7. **cortical/analysis/** (7 files) - PageRank, TF-IDF, clustering
8. **cortical/cel/performance/** (4 files) - Performance layer
9. **cortical/spark/** (9 CORE files) - Statistical models

### Phase 4: Demo Conversion

10. Convert 32 demo files to Given-When-Then behavioral tests
