# Codebase Test Coverage Inventory

**Purpose:** Tag every file with required test coverage
**Created:** 2025-12-31
**Status:** DRAFT - Needs tagging

## Tags Legend

| Tag | Meaning |
|-----|---------|
| `[B]` | Needs Behavioral Test |
| `[C]` | Needs Contract Test |
| `[S]` | Needs Singularity Check (no duplicates) |
| `[D]` | Is a Demo (convert to behavioral test) |
| `[✓]` | Already covered |
| `[-]` | Not applicable (init, config, etc.) |
| `[?]` | Needs review |

---

## Root Showcases

| Lines | File | Tag |
|------:|------|-----|
| 1,067 | `showcase.py` | [ ] |
| 970 | `nlu_showcase.py` | [ ] |
| 634 | `repo_showcase.py` | [ ] |
| 522 | `secureshowcase.py` | [ ] |

---

## benchmarks/

### benchmarks/cel/

| Lines | File | Tag |
|------:|------|-----|
| 1,042 | `benchmarks.py` | [ ] |
| 916 | `sanity_benchmarks.py` | [ ] |
| 815 | `performance_benchmarks.py` | [ ] |
| 371 | `runner.py` | [ ] |
| 53 | `__init__.py` | [ ] |

### benchmarks/codebase_slm/

| Lines | File | Tag |
|------:|------|-----|
| 637 | `data_augmentation.py` | [ ] |
| 455 | `train_augmented.py` | [ ] |
| 422 | `pln_generator.py` | [ ] |
| 409 | `hybrid_pipeline.py` | [ ] |
| 380 | `dialogue_generator.py` | [ ] |
| 376 | `benchmark_suite.py` | [ ] |
| 375 | `train_slm.py` | [ ] |
| 350 | `explore_data_generators.py` | [ ] |
| 238 | `explore_woven_mind.py` | [ ] |
| 205 | `generate_corpus.py` | [ ] |

### benchmarks/codebase_slm/generators/

| Lines | File | Tag |
|------:|------|-----|
| 548 | `pattern_generator.py` | [ ] |
| 504 | `doc_extractor.py` | [ ] |
| 423 | `code_extractor.py` | [ ] |
| 333 | `meta_extractor.py` | [ ] |
| 24 | `__init__.py` | [ ] |

### benchmarks/corpus/

| Lines | File | Tag |
|------:|------|-----|
| 2,524 | `runner.py` | [ ] |
| 517 | `base.py` | [ ] |
| 31 | `__init__.py` | [ ] |

### benchmarks/prism_slm/

| Lines | File | Tag |
|------:|------|-----|
| 733 | `generation.py` | [ ] |
| 420 | `learning.py` | [ ] |
| 374 | `integration.py` | [ ] |
| 314 | `runner.py` | [ ] |
| 50 | `__init__.py` | [ ] |

### benchmarks/woven_mind/

| Lines | File | Tag |
|------:|------|-----|
| 523 | `cognitive.py` | [ ] |
| 519 | `stability.py` | [ ] |
| 489 | `scale.py` | [ ] |
| 481 | `quality.py` | [ ] |
| 379 | `base.py` | [ ] |
| 266 | `runner.py` | [ ] |
| 34 | `__init__.py` | [ ] |

---

## cortical/

### cortical/ (root)

| Lines | File | Tag |
|------:|------|-----|
| 1,379 | `ml_storage.py` | [ ] |
| 1,168 | `cli_wrapper.py` | [ ] |
| 927 | `semantics.py` | [ ] |
| 818 | `wal.py` | [ ] |
| 763 | `state_storage.py` | [ ] |
| 624 | `diff.py` | [ ] |
| 601 | `persistence.py` | [ ] |
| 571 | `chunk_index.py` | [ ] |
| 554 | `async_api.py` | [ ] |
| 542 | `patterns.py` | [ ] |
| 524 | `minicolumn.py` | [ ] |
| 510 | `fluent.py` | [ ] |
| 500 | `results.py` | [ ] |
| 445 | `embeddings.py` | [ ] |
| 437 | `observability.py` | [ ] |
| 426 | `tokenizer.py` | [ ] |
| 400 | `config.py` | [ ] |
| 349 | `progress.py` | [ ] |
| 315 | `fingerprint.py` | [ ] |
| 314 | `layers.py` | [ ] |
| 276 | `code_concepts.py` | [ ] |
| 248 | `validation.py` | [ ] |
| 245 | `gaps.py` | [ ] |
| 210 | `top_words.py` | [ ] |
| 161 | `types.py` | [ ] |
| 114 | `constants.py` | [ ] |
| 81 | `__init__.py` | [ ] |

### cortical/analysis/

| Lines | File | Tag |
|------:|------|-----|
| 671 | `clustering.py` | [ ] |
| 520 | `connections.py` | [ ] |
| 495 | `quality.py` | [ ] |
| 471 | `pagerank.py` | [ ] |
| 251 | `tfidf.py` | [ ] |
| 223 | `parallel.py` | [ ] |
| 194 | `utils.py` | [ ] |
| 123 | `__init__.py` | [ ] |
| 76 | `activation.py` | [ ] |

### cortical/cel/

| Lines | File | Tag |
|------:|------|-----|
| 713 | `container.py` | [ ] |
| 687 | `tracing.py` | [ ] |
| 481 | `tracing_integration.py` | [ ] |
| 445 | `config.py` | [ ] |
| 101 | `__init__.py` | [ ] |

### cortical/cel/adapters/

| Lines | File | Tag |
|------:|------|-----|
| 729 | `got.py` | [ ] |
| 37 | `__init__.py` | [ ] |

### cortical/cel/core/

| Lines | File | Tag |
|------:|------|-----|
| 827 | `protocols.py` | [ ] |
| 539 | `events.py` | [ ] |
| 481 | `references.py` | [ ] |
| 71 | `__init__.py` | [ ] |

### cortical/cel/performance/

| Lines | File | Tag |
|------:|------|-----|
| 715 | `streaming_store.py` | [ ] |
| 523 | `optimized_dag.py` | [ ] |
| 509 | `snapshots.py` | [ ] |
| 508 | `entity_index.py` | [ ] |
| 49 | `__init__.py` | [ ] |

### cortical/cel/sanity/

| Lines | File | Tag |
|------:|------|-----|
| 675 | `compaction.py` | [ ] |
| 522 | `migration.py` | [ ] |
| 487 | `health.py` | [ ] |
| 57 | `__init__.py` | [ ] |

### cortical/cel/wisdom/

| Lines | File | Tag |
|------:|------|-----|
| 515 | `dag.py` | [ ] |
| 488 | `materializer.py` | [ ] |
| 421 | `semantic.py` | [ ] |
| 37 | `__init__.py` | [ ] |

### cortical/got/

| Lines | File | Tag |
|------:|------|-----|
| 2,918 | `api.py` | [ ] |
| 1,449 | `query_builder.py` | [ ] |
| 1,368 | `types.py` | [ ] |
| 946 | `validation.py` | [ ] |
| 739 | `pattern_matcher.py` | [ ] |
| 650 | `versioned_store.py` | [ ] |
| 646 | `path_finder.py` | [ ] |
| 642 | `graph_walker.py` | [ ] |
| 632 | `claudemd.py` | [ ] |
| 632 | `recovery.py` | [ ] |
| 621 | `entity_schemas.py` | [ ] |
| 590 | `query_api.py` | [ ] |
| 574 | `schema.py` | [ ] |
| 523 | `orphan.py` | [ ] |
| 466 | `wal.py` | [ ] |
| 414 | `indexer.py` | [ ] |
| 353 | `protocol.py` | [ ] |
| 338 | `sync.py` | [ ] |
| 327 | `tx_manager.py` | [ ] |
| 327 | `__init__.py` | [ ] |
| 247 | `conflict.py` | [ ] |
| 166 | `transaction.py` | [ ] |
| 62 | `errors.py` | [ ] |
| 50 | `config.py` | [ ] |

### cortical/got/cli/

| Lines | File | Tag |
|------:|------|-----|
| 711 | `doc.py` | [ ] |
| 634 | `batch.py` | [ ] |
| 587 | `task.py` | [ ] |
| 585 | `sprint.py` | [ ] |
| 512 | `analyze.py` | [ ] |
| 427 | `backup.py` | [ ] |
| 400 | `orphan.py` | [ ] |
| 368 | `query.py` | [ ] |
| 353 | `decision.py` | [ ] |
| 350 | `handoff.py` | [ ] |
| 341 | `backlog.py` | [ ] |
| 326 | `edge.py` | [ ] |
| 231 | `__init__.py` | [ ] |
| 205 | `shared.py` | [ ] |

### cortical/ml_experiments/

| Lines | File | Tag |
|------:|------|-----|
| 631 | `file_prediction_adapter.py` | [ ] |
| 511 | `dataset.py` | [ ] |
| 430 | `experiment.py` | [ ] |
| 389 | `metrics.py` | [ ] |
| 362 | `utils.py` | [ ] |
| 87 | `__init__.py` | [ ] |

### cortical/processor/

| Lines | File | Tag |
|------:|------|-----|
| 1,313 | `spark_api.py` | [ ] |
| 1,276 | `compute.py` | [ ] |
| 784 | `query_api.py` | [ ] |
| 462 | `documents.py` | [ ] |
| 357 | `introspection.py` | [ ] |
| 260 | `persistence_api.py` | [ ] |
| 224 | `core.py` | [ ] |
| 74 | `__init__.py` | [ ] |

### cortical/projects/

| Lines | File | Tag |
|------:|------|-----|
| 13 | `__init__.py` | [ ] |
| 24 | `cli/__init__.py` | [ ] |

### cortical/query/

| Lines | File | Tag |
|------:|------|-----|
| 704 | `search.py` | [ ] |
| 491 | `expansion.py` | [ ] |
| 469 | `ranking.py` | [ ] |
| 408 | `passages.py` | [ ] |
| 346 | `definitions.py` | [ ] |
| 336 | `chunking.py` | [ ] |
| 330 | `analogy.py` | [ ] |
| 220 | `intent.py` | [ ] |
| 185 | `__init__.py` | [ ] |
| 95 | `utils.py` | [ ] |

### cortical/reasoning/

| Lines | File | Tag |
|------:|------|-----|
| 2,044 | `graph_persistence.py` | [ ] |
| 1,367 | `collaboration.py` | [ ] |
| 1,340 | `production_state.py` | [ ] |
| 1,255 | `crisis_manager.py` | [ ] |
| 1,224 | `thought_graph.py` | [ ] |
| 1,223 | `claude_code_spawner.py` | [ ] |
| 1,197 | `verification.py` | [ ] |
| 1,147 | `prism_got.py` | [ ] |
| 1,120 | `loom.py` | [ ] |
| 970 | `cognitive_loop.py` | [ ] |
| 929 | `prism_slm.py` | [ ] |
| 828 | `workflow.py` | [ ] |
| 808 | `pubsub.py` | [ ] |
| 743 | `__init__.py` | [ ] |
| 719 | `prism_pln.py` | [ ] |
| 634 | `consolidation.py` | [ ] |
| 615 | `prism_attention.py` | [ ] |
| 599 | `abstraction.py` | [ ] |
| 583 | `homeostasis.py` | [ ] |
| 570 | `rejection_protocol.py` | [ ] |
| 562 | `metrics.py` | [ ] |
| 505 | `qapv_verification.py` | [ ] |
| 486 | `nested_loop.py` | [ ] |
| 484 | `thought_patterns.py` | [ ] |
| 472 | `goal_stack.py` | [ ] |
| 430 | `abstraction_pln.py` | [ ] |
| 416 | `loom_cortex.py` | [ ] |
| 404 | `woven_mind.py` | [ ] |
| 400 | `loom_hive.py` | [ ] |
| 396 | `context_pool.py` | [ ] |
| 390 | `loop_validator.py` | [ ] |
| 378 | `graph_of_thought.py` | [ ] |
| 349 | `attention_router.py` | [ ] |

### cortical/spark/

| Lines | File | Tag |
|------:|------|-----|
| 716 | `quality.py` | [ ] |
| 626 | `transfer.py` | [ ] |
| 606 | `intelligence.py` | [ ] |
| 567 | `diff_tokenizer.py` | [ ] |
| 523 | `ast_index.py` | [ ] |
| 490 | `suggester.py` | [ ] |
| 471 | `co_change.py` | [ ] |
| 468 | `intent_parser.py` | [ ] |
| 452 | `alignment.py` | [ ] |
| 437 | `git_trainer.py` | [ ] |
| 434 | `ngram.py` | [ ] |
| 373 | `predictor.py` | [ ] |
| 346 | `anomaly.py` | [ ] |
| 193 | `tokenizer.py` | [ ] |
| 145 | `__init__.py` | [ ] |

### cortical/utils/

| Lines | File | Tag |
|------:|------|-----|
| 524 | `id_generation.py` | [ ] |
| 295 | `locking.py` | [ ] |
| 117 | `checksums.py` | [ ] |
| 100 | `persistence.py` | [ ] |
| 49 | `__init__.py` | [ ] |
| 32 | `text.py` | [ ] |

---

## examples/

| Lines | File | Tag |
|------:|------|-----|
| 1,027 | `cel_demo.py` | [ ] |
| 860 | `woven_mind_demo.py` | [ ] |
| 639 | `spark_demo.py` | [ ] |
| 572 | `prism_got_comprehensive_demo.py` | [ ] |
| 550 | `got_demo.py` | [ ] |
| 483 | `rejection_protocol_demo.py` | [ ] |
| 377 | `code_evolution_demo.py` | [ ] |
| 339 | `reasoning_metrics_demo.py` | [ ] |
| 329 | `prism_got_demo.py` | [ ] |
| 321 | `context_pool_demo.py` | [ ] |
| 314 | `prism_got_demo_corpus.py` | [ ] |
| 312 | `subprocess_spawner_demo.py` | [ ] |
| 309 | `prism_got_nlu_demo.py` | [ ] |
| 290 | `graph_persistence_demo.py` | [ ] |
| 283 | `async_api_demo.py` | [ ] |
| 264 | `pubsub_demo.py` | [ ] |
| 261 | `qapv_verification_demo.py` | [ ] |
| 242 | `examples_results_usage.py` | [ ] |
| 236 | `nested_loop_demo.py` | [ ] |
| 204 | `git_auto_committer_demo.py` | [ ] |
| 197 | `parallel_demo.py` | [ ] |
| 194 | `demo_ci_integration.py` | [ ] |
| 183 | `prism_slm_demo.py` | [ ] |
| 183 | `repl_demo.py` | [ ] |
| 168 | `demo_pattern_detection.py` | [ ] |
| 152 | `got_dashboard_origin_demo.py` | [ ] |
| 132 | `observability_demo.py` | [ ] |
| 120 | `demo_progress.py` | [ ] |

---

## llm_orchestration/

| Lines | File | Tag |
|------:|------|-----|
| 1,139 | `recovery.py` | [ ] |
| 1,131 | `learning.py` | [ ] |
| 1,018 | `thought_patterns.py` | [ ] |
| 977 | `evolution.py` | [ ] |
| 971 | `protocols.py` | [ ] |
| 918 | `agents.py` | [ ] |
| 912 | `cognitive_state.py` | [ ] |
| 836 | `tools.py` | [ ] |
| 715 | `orchestration.py` | [ ] |
| 540 | `types.py` | [ ] |
| 528 | `agile.py` | [ ] |
| 445 | `metrics.py` | [ ] |
| 220 | `__init__.py` | [ ] |

### llm_orchestration/examples/

| Lines | File | Tag |
|------:|------|-----|
| 344 | `learning_demo.py` | [ ] |
| 288 | `basic_workflow.py` | [ ] |
| 281 | `recovery_demo.py` | [ ] |
| 267 | `multi_session.py` | [ ] |
| 14 | `__init__.py` | [ ] |

---

## scripts/

### scripts/ (root)

| Lines | File | Tag |
|------:|------|-----|
| 5,496 | `generate_book.py` | [ ] |
| 4,599 | `ml_data_collector.py` | [ ] |
| 3,016 | `got_utils.py` | [ ] |
| 2,273 | `index_codebase.py` | [ ] |
| 2,028 | `ml_file_prediction.py` | [ ] |
| 1,327 | `hubris_cli.py` | [ ] |
| 1,237 | `orchestration_utils.py` | [ ] |
| 1,211 | `claudemd_generation_demo.py` | [ ] |
| 1,050 | `repl.py` | [ ] |
| 1,039 | `got_dashboard.py` | [ ] |
| 930 | `benchmark_scoring.py` | [ ] |
| 928 | `ascii_visualizer_animated.py` | [ ] |
| 908 | `benchmark_spark.py` | [ ] |
| 820 | `world_model_analysis.py` | [ ] |
| 807 | `run_sprint_reasoning.py` | [ ] |
| 765 | `generate_ai_metadata.py` | [ ] |
| 759 | `knowledge_bridge.py` | [ ] |
| 741 | `knowledge_analysis.py` | [ ] |
| 739 | `ascii_codebase_art.py` | [ ] |
| 728 | `spark_code_assistant.py` | [ ] |
| 707 | `cognitive_pipeline.py` | [ ] |
| 704 | `cognitive_integration_demo.py` | [ ] |
| 690 | `migrate_got.py` | [ ] |
| 666 | `verify_batch.py` | [ ] |
| 642 | `generate_ascii_gifs.py` | [ ] |
| 624 | `session_memory_generator.py` | [ ] |
| 612 | `spark_code_intelligence.py` | [ ] |
| 605 | `suggest_consolidation.py` | [ ] |
| 588 | `profile_got_query.py` | [ ] |
| 586 | `analyze_louvain_resolution.py` | [ ] |
| 570 | `evaluate_cluster.py` | [ ] |
| 558 | `question_connection.py` | [ ] |
| 542 | `thought_chain.py` | [ ] |
| 531 | `reasoning_demo.py` | [ ] |
| 521 | `run_tests.py` | [ ] |
| 516 | `llm_generate_response.py` | [ ] |
| 512 | `analyze_cross_domain_bridges.py` | [ ] |
| 506 | `migrate_sprints_to_got.py` | [ ] |
| 504 | `search_codebase.py` | [ ] |
| 463 | `verify_milestone.py` | [ ] |
| 445 | `corpus_health.py` | [ ] |
| 441 | `validate_reasoning_persistence.py` | [ ] |
| 440 | `cognitive_demo_refined.py` | [ ] |
| 439 | `branch_manifest.py` | [ ] |
| 434 | `cognitive_demo.py` | [ ] |
| 431 | `cognitive_demo_ast.py` | [ ] |
| 427 | `session_handoff.py` | [ ] |
| 420 | `suggest_related.py` | [ ] |
| 417 | `generate_sample_docs.py` | [ ] |
| 395 | `new_memory.py` | [ ] |
| 379 | `got_priority_executor.py` | [ ] |
| 345 | `profile_full_analysis.py` | [ ] |
| 345 | `analyze_corpus_balance.py` | [ ] |
| 344 | `migrate_ml_to_cali.py` | [ ] |
| 321 | `explain_code.py` | [ ] |
| 304 | `find_similar.py` | [ ] |
| 290 | `ci_task_report.py` | [ ] |
| 283 | `ask_codebase.py` | [ ] |
| 255 | `hubris-feedback-hook.py` | [ ] |
| 245 | `backfill_chat_history.py` | [ ] |
| 230 | `task_diff.py` | [ ] |
| 215 | `benchmark_sparsity.py` | [ ] |
| 194 | `train_spark_from_git.py` | [ ] |
| 189 | `cli_wrappers.py` | [ ] |
| 150 | `resolve_wiki_links.py` | [ ] |
| 51 | `demo_utils.py` | [ ] |
| 48 | `ascii_effects.py` | [ ] |
| 19 | `doc_utils.py` | [ ] |

### scripts/hubris/

| Lines | File | Tag |
|------:|------|-----|
| 1,989 | `ml_file_prediction_v1.py` | [ ] |
| 682 | `feedback_collector.py` | [ ] |
| 612 | `test_calibration_tracker.py` | [ ] |
| 518 | `expert_consolidator.py` | [ ] |
| 509 | `staking.py` | [ ] |
| 500 | `test_feedback.py` | [ ] |
| 469 | `credit_account.py` | [ ] |
| 446 | `calibration_tracker.py` | [ ] |
| 391 | `value_signal.py` | [ ] |
| 355 | `credit_router.py` | [ ] |
| 304 | `voting_aggregator.py` | [ ] |
| 304 | `micro_expert.py` | [ ] |
| 288 | `estimate_command_data.py` | [ ] |
| 279 | `expert_router.py` | [ ] |
| 234 | `train_command_expert.py` | [ ] |
| 213 | `test_calibration_demo.py` | [ ] |
| 20 | `__init__.py` | [ ] |

### scripts/hubris/experts/

| Lines | File | Tag |
|------:|------|-----|
| 739 | `refactor_expert.py` | [ ] |
| 523 | `episode_expert.py` | [ ] |
| 467 | `error_expert.py` | [ ] |
| 411 | `test_expert.py` | [ ] |
| 395 | `command_expert.py` | [ ] |
| 353 | `file_expert.py` | [ ] |
| 13 | `__init__.py` | [ ] |

### scripts/ml_collector/

| Lines | File | Tag |
|------:|------|-----|
| 583 | `orchestration.py` | [ ] |
| 557 | `persistence.py` | [ ] |
| 515 | `chunked_storage.py` | [ ] |
| 447 | `task_linker.py` | [ ] |
| 347 | `session.py` | [ ] |
| 323 | `commit.py` | [ ] |
| 285 | `transcript.py` | [ ] |
| 274 | `quality.py` | [ ] |
| 219 | `stats.py` | [ ] |
| 199 | `export.py` | [ ] |
| 168 | `config.py` | [ ] |
| 117 | `data_classes.py` | [ ] |
| 106 | `ci.py` | [ ] |
| 89 | `__init__.py` | [ ] |
| 79 | `hooks.py` | [ ] |
| 21 | `core.py` | [ ] |

---

## Known Duplicates (Singularity Check Required)

| Capability | File 1 | File 2 | Action |
|------------|--------|--------|--------|
| WAL | `cortical/wal.py` (818) | `cortical/got/wal.py` (466) | [S] |
| Validation | `cortical/validation.py` (248) | `cortical/got/validation.py` (946) | [S] |
| Types | `cortical/types.py` (161) | `cortical/got/types.py` (1,368) | [S] |
| Config | `cortical/config.py` (400) | `cortical/got/config.py` (50) | [S] |
| Config | `cortical/config.py` (400) | `cortical/cel/config.py` (445) | [S] |
| Tokenizer | `cortical/tokenizer.py` (426) | `cortical/spark/tokenizer.py` (193) | [S] |
| Quality | `cortical/analysis/quality.py` (495) | `cortical/spark/quality.py` (716) | [S] |
| Persistence | `cortical/persistence.py` (601) | `cortical/utils/persistence.py` (100) | [S] |
| Orphan | `cortical/got/orphan.py` (523) | `cortical/got/cli/orphan.py` (400) | [S] |

---

## Summary Statistics

| Area | Files | Lines |
|------|------:|------:|
| cortical/ | 164 | 93,185 |
| scripts/ | 108 | 66,887 |
| tests/ | 321 | 190,868 |
| benchmarks/ | 35 | 16,530 |
| examples/ | 28 | 9,541 |
| llm_orchestration/ | 18 | 10,350 |
| **TOTAL** | **674** | **387,361** |
