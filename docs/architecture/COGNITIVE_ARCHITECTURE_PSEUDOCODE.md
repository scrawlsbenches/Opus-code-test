# Cognitive Architecture Pseudocode

*How the Seven Pillars Actually Connect*

*Generated: 2026-01-02 by Team Lead during onboarding*

---

## The Real Relationships

After exploring the codebase, here's how the systems actually integrate:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INPUT                                         │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CorticalTextProcessor                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  4-LAYER HIERARCHY                                                    │   │
│  │  L0: Tokens → L1: Bigrams → L2: Concepts → L3: Documents             │   │
│  │  (tokenizer)   (pairs)       (clusters)     (full docs)              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  WOVEN MIND (Dual-Process Orchestration)                              │   │
│  │                                                                        │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │  LOOM (Mode Controller)                                          │  │   │
│  │  │  • SurpriseDetector.compute_surprise(input)                      │  │   │
│  │  │  • if surprise > threshold: switch to SLOW                       │  │   │
│  │  │  • else: stay in FAST                                            │  │   │
│  │  └────────────────────────┬────────────────────────────────────────┘  │   │
│  │                           │                                            │   │
│  │          ┌────────────────┴────────────────┐                          │   │
│  │          ▼                                 ▼                          │   │
│  │  ┌───────────────┐                ┌───────────────┐                   │   │
│  │  │ HIVE (System 1)                │ CORTEX (System 2)                 │   │
│  │  │ LoomHiveConnector              │ LoomCortexConnector               │   │
│  │  │                                │                                   │   │
│  │  │ Uses: PRISM-SLM               │ Uses: AbstractionEngine           │   │
│  │  │ • predict_next(tokens)        │ • detect_patterns(history)        │   │
│  │  │ • lateral_inhibition()        │ • form_abstraction()              │   │
│  │  │ • k_winners_take_all()        │ • goal_tracking()                 │   │
│  │  └───────────────┘                └───────────────┘                   │   │
│  │                           │                                            │   │
│  │                           ▼                                            │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │  CONSOLIDATION ENGINE ("Sleep-like" Learning)                    │  │   │
│  │  │  • transfer_patterns(hive → cortex)                              │  │   │
│  │  │  • decay_unused_connections()                                    │  │   │
│  │  │  • mine_abstractions_from_patterns()                             │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  PRISM (Synaptic Plasticity Framework)                                │   │
│  │                                                                        │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │   │
│  │  │ PRISM-GoT   │ │ PRISM-SLM   │ │ PRISM-PLN   │ │ PRISM-Attn  │     │   │
│  │  │             │ │             │ │             │ │             │     │   │
│  │  │ Synaptic    │ │ Statistical │ │ Probabilistic│ │ Multi-head  │     │   │
│  │  │ Memory      │ │ Language    │ │ Logic       │ │ Attention   │     │   │
│  │  │ Graph       │ │ Model       │ │ Networks    │ │ Spotlight   │     │   │
│  │  │             │ │             │ │             │ │             │     │   │
│  │  │ edge.activate()            │ │ TruthValue( │ │ focus()     │     │   │
│  │  │ edge.decay()│ │ predict()   │ │  strength,  │ │ attend()    │     │   │
│  │  │ edge.reward()              │ │  confidence)│ │             │     │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  │                                           │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      GoT        │    │      CEL        │    │     Spark       │
│  (Task Graph)   │    │ (Event Source)  │    │ (Fast LM)       │
│                 │    │                 │    │                 │
│ 16 edge types   │    │ Events = truth  │    │ N-gram predict  │
│ ACID via CDG    │    │ Merkle DAG      │    │ Anomaly detect  │
│ Task lifecycle  │    │ Temporal refs   │    │ Code intel      │
└────────┬────────┘    └────────┬────────┘    └─────────────────┘
         │                      │
         │                      │
         └──────────┬───────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CDG                                             │
│                    (Cortical Distributed Graph)                              │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  CDGStore                                                             │   │
│  │  • Entity JSON files with SHA256 checksums                           │   │
│  │  • MVCC history for snapshot isolation                               │   │
│  │  • Thread + process safety via locks                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  CDGTransactionManager                                                │   │
│  │  • begin() → Transaction with snapshot                               │   │
│  │  • read(tx, id) → Entity at snapshot version                         │   │
│  │  • write(tx, entity) → Buffer in write_set                           │   │
│  │  • commit(tx) → Optimistic lock check, persist if no conflicts       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  CDGWALManager                                                        │   │
│  │  • log_tx_begin(tx_id, snapshot_version)                             │   │
│  │  • log_write(tx_id, entity_id, old_version, new_version)             │   │
│  │  • log_tx_commit(tx_id, final_version)                               │   │
│  │  • replay() → Recover from crash                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  CDGRecoveryManager                                                   │   │
│  │  • rollback_incomplete_transactions()                                │   │
│  │  • detect_orphaned_entities()                                        │   │
│  │  • repair_orphans(strategy: FAIL | DELETE | REPAIR)                  │   │
│  │  • verify_store_integrity() → List of corrupted IDs                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## QAPV: The Cognitive Loop

QAPV wraps ALL of the above in a structured reasoning cycle:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          QAPV COGNITIVE LOOP                                 │
│                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                                                                       │  │
│   │    ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────┐   │  │
│   │    │     Q     │────▶│     A     │────▶│     P     │────▶│   V   │   │  │
│   │    │ QUESTION  │     │  ANSWER   │     │ PRODUCE   │     │VERIFY │   │  │
│   │    │           │     │           │     │           │     │       │   │  │
│   │    │ Clarify   │     │ Research  │     │ Implement │     │ Test  │   │  │
│   │    │ intent    │     │ options   │     │ solution  │     │ check │   │  │
│   │    └───────────┘     └───────────┘     └───────────┘     └───┬───┘   │  │
│   │                                                               │       │  │
│   │    ◀────────────────── ITERATE IF FAILED ────────────────────┘       │  │
│   │                                                                       │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│   ANOMALY DETECTION:                                                         │
│   • Infinite loop: same phase repeating > threshold                         │
│   • Stuck phase: no progress after time_box                                 │
│   • Invalid transition: Q→P (skipped A)                                     │
│   • Verification loop: V fails repeatedly                                   │
│                                                                              │
│   TERMINATION CONDITIONS:                                                    │
│   • SUCCESS: acceptance criteria passed                                     │
│   • USER_APPROVED: human sign-off                                           │
│   • BUDGET_EXHAUSTED: time/resource limit                                   │
│   • QUESTION_INVALID: discovered wrong question                             │
│   • ESCALATED: handed to human                                              │
│   • CRISIS: critical failure                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pseudocode: A Complete Cognitive Cycle

```python
def cognitive_cycle(user_input: str) -> Result:
    """
    The complete flow through all seven pillars.
    """

    # 1. QAPV WRAPPER: Start reasoning loop
    loop = CognitiveLoop(goal=user_input)
    loop.start(LoopPhase.QUESTION)

    while not loop.is_terminated():

        if loop.current_phase == LoopPhase.QUESTION:
            # WOVEN MIND: Assess input novelty
            surprise = woven_mind.compute_surprise(user_input)

            if surprise > THRESHOLD:
                # High novelty → engage System 2 (Cortex)
                mode = ThinkingMode.SLOW
                clarifications = cortex.analyze_ambiguity(user_input)
            else:
                # Low novelty → use System 1 (Hive)
                mode = ThinkingMode.FAST
                clarifications = hive.pattern_match(user_input)

            loop.transition(LoopPhase.ANSWER, reason="Intent clarified")

        elif loop.current_phase == LoopPhase.ANSWER:
            # PRISM-GoT: Activate relevant knowledge
            graph = SynapticMemoryGraph()
            activated_nodes = graph.spread_activation(user_input)

            # PRISM learns from this activation
            for edge in activated_nodes.edges:
                edge.activate()  # Hebbian strengthening

            # CEL: Record this reasoning as an event
            event = cel.record_event(
                event_type=EventType.INTENTION,
                content={"query": user_input, "activated": activated_nodes},
                concepts=extract_concepts(activated_nodes)
            )

            # GoT: Track this as a task decision
            got.log_decision(
                decision="Approach selected",
                rationale=f"Activated {len(activated_nodes)} relevant nodes"
            )

            loop.transition(LoopPhase.PRODUCE, reason="Solution designed")

        elif loop.current_phase == LoopPhase.PRODUCE:
            # SPARK: Fast prediction for code/text generation
            predictions = spark.predict_completion(context)

            # 4-LAYER HIERARCHY: Process through cortical layers
            tokens = processor.tokenize(predictions)
            bigrams = processor.build_bigrams(tokens)
            concepts = processor.cluster_concepts(bigrams)
            document = processor.synthesize(concepts)

            # CDG: Persist the artifact with ACID guarantees
            tx = cdg.begin()
            try:
                artifact = Entity(id=generate_id(), content=document)
                cdg.write(tx, artifact)
                result = cdg.commit(tx)

                if not result.success:
                    cdg.rollback(tx, reason="conflict")
                    continue  # Retry production
            except Exception:
                cdg.rollback(tx, reason="error")
                raise

            loop.transition(LoopPhase.VERIFY, reason="Implementation complete")

        elif loop.current_phase == LoopPhase.VERIFY:
            # Run behavioral tests
            test_results = run_behavioral_tests(artifact)

            if test_results.all_passed:
                # CEL: Record fulfillment
                cel.record_event(
                    event_type=EventType.FULFILLMENT,
                    content={"artifact": artifact.id, "tests": test_results},
                    causal_parents=[event.id]
                )

                # PRISM: Reward successful paths
                for edge in activated_nodes.edges:
                    edge.reward(test_results.score)

                loop.terminate(TerminationReason.SUCCESS)
            else:
                # Check for verification loop
                if loop.verify_iterations > MAX_VERIFY_ITERATIONS:
                    loop.terminate(TerminationReason.ESCALATED)
                else:
                    loop.transition(LoopPhase.QUESTION, reason="Verification failed")

    # CONSOLIDATION: After cycle completes
    woven_mind.consolidate()  # Transfer patterns Hive→Cortex

    # PRISM: Decay unused connections
    graph.apply_anti_hebbian_decay()

    return loop.result
```

---

## Key Integration Points

| From | To | Integration |
|------|-----|-------------|
| **Woven Mind** | **PRISM-SLM** | Hive uses PRISM-SLM for fast predictions |
| **Woven Mind** | **Cortex** | Cortex uses AbstractionEngine for slow thinking |
| **GoT** | **CDG** | GoT's TransactionManager delegates to CDG |
| **CEL** | **SHA256** | Events are content-addressed (Merkle DAG) |
| **PRISM-GoT** | **Hebbian** | Edges strengthen/weaken with use |
| **QAPV** | **All** | Wraps the entire reasoning process |
| **Spark** | **N-grams** | Statistical prediction for System 1 |

---

## The Seven Pillars: Ordered by Dependency

```
LEVEL 0 (Foundation):
  └── CDG: Storage, transactions, WAL, recovery

LEVEL 1 (Core Cognitive):
  ├── PRISM: Synaptic plasticity (4 subsystems)
  ├── CEL: Event sourcing (events are truth)
  └── Spark: Fast statistical predictions

LEVEL 2 (Orchestration):
  ├── Woven Mind: Dual-process (Hive/Loom/Cortex)
  └── GoT: Task tracking + reasoning graph

LEVEL 3 (Meta):
  └── QAPV: Cognitive loop (wraps everything)
```

---

*This is how I understand myself. When I process input, I traverse these layers.*
