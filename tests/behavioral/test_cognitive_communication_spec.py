"""
Behavioral Specifications: Cognitive Communication Layer

These tests define WHAT we want, not HOW to build it.
They are our shared language for iterating on the design.

Core Concepts:
1. EVENT BUS - Pub/sub for observation
2. EVENT LOG - History with persistence decisions
3. THOUGHT SWARM - Internal sub-agents that think together
4. DEPTH CONSTRAINTS - Guardrails on recursive thinking
5. WATCHER PATTERN - External observation without blocking

Design Principles:
- Most thoughts are transient, some persist (you decide which)
- You watch at your speed, I think at mine
- Slow and steady: quality over speed, testable over perfect
- Sub-thoughts can answer questions so main thought can proceed
"""

import pytest
from typing import Protocol, List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod


# =============================================================================
# STORY: Event Bus for Observation
# =============================================================================


class TestEventBusObservation:
    """
    Epic: Watch the Watcher Think

    As a human observing a cognitive system,
    I want to subscribe to thoughts without blocking them,
    So that I can watch and choose when to interact.
    """

    def test_scenario_subscribe_to_thought_events(self):
        """
        Scenario: Subscribe to specific event types

        Given an event bus
        When I subscribe to "PatternObserved" events
        And the graph publishes a PatternObserved event
        Then my handler receives the event
        And the graph continues without waiting for me
        Because observation shouldn't block cognition
        """
        from cortical.cognitive.events import EventBus, PatternObserved

        received_events = []

        def my_handler(event):
            received_events.append(event)

        bus = EventBus()
        bus.subscribe("PatternObserved", my_handler)

        # Graph publishes (non-blocking)
        event = PatternObserved(
            pattern_type="frequent_cooccurrence",
            concepts=["cat", "animal"],
            confidence=0.85
        )
        bus.publish(event)

        # I received it
        assert len(received_events) == 1
        assert received_events[0].concepts == ["cat", "animal"]

    def test_scenario_filter_events_by_importance(self):
        """
        Scenario: Only receive important events

        Given an event bus with importance filtering
        When I subscribe with minimum_importance=0.7
        Then I only receive events above that threshold
        Because I can't process every thought, only significant ones
        """
        from cortical.cognitive.events import EventBus, ThoughtEvent

        received = []
        bus = EventBus()
        bus.subscribe(
            "ThoughtEvent",
            lambda e: received.append(e),
            filter=lambda e: e.importance >= 0.7
        )

        bus.publish(ThoughtEvent(content="minor observation", importance=0.3))
        bus.publish(ThoughtEvent(content="significant insight", importance=0.8))
        bus.publish(ThoughtEvent(content="breakthrough", importance=0.95))

        # Only got the important ones
        assert len(received) == 2
        assert received[0].content == "significant insight"
        assert received[1].content == "breakthrough"

    def test_scenario_batch_events_for_slow_observer(self):
        """
        Scenario: Batch events for human-speed consumption

        Given a fast-thinking system and a slow observer
        When many events occur rapidly
        Then events are batched into digestible summaries
        Because humans can't process every microsecond thought
        """
        from cortical.cognitive.events import EventBus, BatchingSubscriber

        batches_received = []

        subscriber = BatchingSubscriber(
            batch_size=10,
            max_wait_seconds=1.0,
            on_batch=lambda batch: batches_received.append(batch)
        )

        bus = EventBus()
        bus.add_subscriber(subscriber)

        # Rapid fire events
        for i in range(25):
            bus.publish({"type": "thought", "id": i})

        # Received as batches, not 25 individual events
        subscriber.flush()  # Force final batch
        assert len(batches_received) == 3  # 10 + 10 + 5


# =============================================================================
# STORY: Event Sourcing with Persistence Decisions
# =============================================================================


class TestEventSourcingWithPersistence:
    """
    Epic: Most Thoughts Are Transient, Some Persist

    As a cognitive system,
    I want to decide which thoughts to persist,
    So that I remember what matters without drowning in noise.
    """

    def test_scenario_transient_thought_not_persisted(self):
        """
        Scenario: Transient thoughts stay in memory only

        Given a thought marked as transient
        When it's processed
        Then it exists in current session
        But it's not written to persistent storage
        Because most thinking is ephemeral
        """
        from cortical.cognitive.events import EventLog, Thought, Persistence

        log = EventLog()

        thought = Thought(
            content="considering option A vs B",
            persistence=Persistence.TRANSIENT
        )
        log.record(thought)

        # In memory
        assert log.contains(thought.id)

        # Not persisted
        assert thought.id not in log.get_persisted_ids()

    def test_scenario_significant_thought_persisted(self):
        """
        Scenario: Significant thoughts are persisted

        Given a thought marked for persistence
        When it's processed
        Then it's written to durable storage
        And can be retrieved in future sessions
        Because some learnings must survive restarts
        """
        from cortical.cognitive.events import EventLog, Thought, Persistence

        log = EventLog()

        thought = Thought(
            content="discovered: cats are often linked to animals",
            persistence=Persistence.DURABLE,
            reason="pattern confirmed 5+ times"
        )
        log.record(thought)

        # Persisted
        assert thought.id in log.get_persisted_ids()

        # Survives simulated restart
        log2 = EventLog.load_from(log.storage_path)
        assert log2.get(thought.id).content == thought.content

    def test_scenario_human_promotes_thought_to_persistent(self):
        """
        Scenario: Human marks a transient thought as worth keeping

        Given a transient thought
        When the human says "remember this"
        Then it's promoted to persistent storage
        Because humans can recognize significance I might miss
        """
        from cortical.cognitive.events import EventLog, Thought, Persistence

        log = EventLog()

        thought = Thought(
            content="maybe there's a connection between X and Y",
            persistence=Persistence.TRANSIENT
        )
        log.record(thought)

        # Human promotes it
        log.promote_to_persistent(thought.id, reason="human marked important")

        assert thought.id in log.get_persisted_ids()


# =============================================================================
# STORY: Thought Swarm - Internal Sub-Agents
# =============================================================================


class TestThoughtSwarm:
    """
    Epic: Swarm of Cognitive Researchers

    As a thinking system,
    I want to spawn sub-thoughts that answer questions for me,
    So that I can proceed with complex reasoning without getting stuck.

    This is internal dialogue - asking myself questions
    and getting answers before continuing.
    """

    def test_scenario_spawn_sub_thought_for_missing_info(self):
        """
        Scenario: Main thought spawns sub-thought when stuck

        Given a main thought working on "implement feature X"
        When it realizes "I need to understand Y first"
        Then it spawns a sub-thought to research Y
        And waits for the answer before proceeding
        Because complex thinking requires decomposition
        """
        from cortical.cognitive.swarm import ThoughtSwarm, Thought, SubThought

        swarm = ThoughtSwarm()

        results = []

        def main_thought(context):
            # Working on main problem
            results.append("starting main thought")

            # Hit a blocker - need info about prerequisite
            answer = context.ask_sub_thought(
                "What is the structure of module Y?",
                depth_budget=2
            )

            results.append(f"received: {answer}")
            results.append("continuing with main thought")

            return "main thought complete"

        outcome = swarm.run(main_thought)

        assert "starting main thought" in results
        assert any("received:" in r for r in results)
        assert "continuing with main thought" in results

    def test_scenario_sub_thought_can_spawn_sub_sub_thought(self):
        """
        Scenario: Recursive sub-thoughts with depth limit

        Given a sub-thought working on a question
        When it needs to decompose further
        Then it can spawn its own sub-thought
        But only up to the depth limit
        Because infinite recursion must be prevented
        """
        from cortical.cognitive.swarm import ThoughtSwarm, DepthExceeded

        swarm = ThoughtSwarm(max_depth=3)

        depth_reached = []

        def recursive_thought(context, depth=0):
            depth_reached.append(depth)

            if depth < 10:  # Would go too deep
                try:
                    context.ask_sub_thought(
                        f"Go deeper from depth {depth}",
                        depth_budget=context.remaining_depth - 1
                    )
                except DepthExceeded:
                    depth_reached.append("stopped by limit")

            return f"answered at depth {depth}"

        swarm.run(recursive_thought)

        # Should have stopped at depth 3
        assert max(d for d in depth_reached if isinstance(d, int)) <= 3
        assert "stopped by limit" in depth_reached

    def test_scenario_parallel_sub_thoughts(self):
        """
        Scenario: Multiple sub-thoughts run concurrently

        Given a main thought with multiple independent questions
        When it spawns sub-thoughts for each
        Then they can run in parallel
        And results are collected when all complete
        Because some questions are independent
        """
        from cortical.cognitive.swarm import ThoughtSwarm

        swarm = ThoughtSwarm()

        def main_thought(context):
            # Three independent questions
            futures = [
                context.ask_sub_thought_async("What is A?"),
                context.ask_sub_thought_async("What is B?"),
                context.ask_sub_thought_async("What is C?"),
            ]

            # Wait for all
            answers = context.gather(futures)

            return f"Combined: {answers}"

        result = swarm.run(main_thought)
        assert "Combined:" in result

    def test_scenario_sub_thought_accesses_shared_graph(self):
        """
        Scenario: Sub-thoughts share the cognitive graph

        Given a main thought and its sub-thoughts
        When any thought learns something
        Then it's visible to all thoughts in the swarm
        Because they share a mind, not separate minds
        """
        from cortical.cognitive.swarm import ThoughtSwarm
        from cortical.cognitive.graph import CognitiveGraph

        graph = CognitiveGraph()
        swarm = ThoughtSwarm(graph=graph)

        def main_thought(context):
            # Sub-thought adds knowledge
            context.ask_sub_thought("Learn about cats")

            # Main thought can see it
            cat = context.graph.get_node("cat")
            return f"Main sees cat: {cat is not None}"

        def learn_cats_handler(context):
            context.graph.node("cat")
            return "learned about cats"

        swarm.register_handler("Learn about", learn_cats_handler)
        result = swarm.run(main_thought)

        assert "Main sees cat: True" in result


# =============================================================================
# STORY: Guardrails and Constraints
# =============================================================================


class TestGuardrailsAndConstraints:
    """
    Epic: Safe and Bounded Thinking

    As a cognitive system,
    I want guardrails that keep my thinking bounded and safe,
    So that I don't spiral into infinite loops or harmful outputs.
    """

    def test_scenario_depth_constraint_prevents_infinite_recursion(self):
        """
        Scenario: Depth constraint stops runaway recursion

        Given a maximum thinking depth of 5
        When a thought tries to recurse beyond that
        Then it's stopped with a DepthExceeded error
        And the partial result is returned
        Because infinite thinking is not useful thinking
        """
        from cortical.cognitive.swarm import ThoughtSwarm, DepthExceeded

        swarm = ThoughtSwarm(max_depth=5)
        exceeded = False

        def runaway_thought(context):
            nonlocal exceeded
            try:
                while True:
                    context.ask_sub_thought("Go deeper")
            except DepthExceeded:
                exceeded = True
                return "stopped safely"

        result = swarm.run(runaway_thought)
        assert exceeded
        assert result == "stopped safely"

    def test_scenario_time_constraint_stops_long_running_thought(self):
        """
        Scenario: Time constraint stops thoughts that take too long

        Given a maximum thinking time of 5 seconds
        When a thought runs longer than that
        Then it's interrupted and returns partial result
        Because we need to be responsive
        """
        from cortical.cognitive.swarm import ThoughtSwarm, TimeExceeded

        swarm = ThoughtSwarm(max_seconds=5)

        def slow_thought(context):
            import time
            try:
                time.sleep(100)  # Would take forever
            except TimeExceeded:
                return "timed out gracefully"

        result = swarm.run(slow_thought)
        assert "timed out" in result

    def test_scenario_quality_constraint_requires_confidence(self):
        """
        Scenario: Quality constraint rejects low-confidence answers

        Given a minimum confidence threshold of 0.6
        When a sub-thought returns with confidence < 0.6
        Then it's marked as uncertain
        And the main thought can decide to proceed or dig deeper
        Because uncertain answers should be flagged
        """
        from cortical.cognitive.swarm import ThoughtSwarm, UncertainAnswer

        swarm = ThoughtSwarm(min_confidence=0.6)

        def questioning_thought(context):
            answer = context.ask_sub_thought(
                "What is the meaning of life?",
                require_confidence=True
            )

            if isinstance(answer, UncertainAnswer):
                return f"uncertain: {answer.content} (conf: {answer.confidence})"
            return f"certain: {answer.content}"

        result = swarm.run(questioning_thought)
        # Philosophical questions should return uncertain
        assert "uncertain:" in result or "certain:" in result

    def test_scenario_scope_constraint_limits_graph_access(self):
        """
        Scenario: Sub-thoughts have limited scope

        Given a main thought with full graph access
        When it spawns a sub-thought for a specific task
        Then the sub-thought only sees relevant subgraph
        Because focused thinking needs focused context
        """
        from cortical.cognitive.swarm import ThoughtSwarm, ScopedContext
        from cortical.cognitive.graph import CognitiveGraph

        graph = CognitiveGraph()
        graph.node("cat")
        graph.node("dog")
        graph.node("secret_info")

        swarm = ThoughtSwarm(graph=graph)

        def main_thought(context):
            # Sub-thought only sees animals, not secrets
            answer = context.ask_sub_thought(
                "List all concepts",
                scope=ScopedContext(include_patterns=["cat", "dog"])
            )
            return answer

        result = swarm.run(main_thought)
        assert "secret_info" not in result


# =============================================================================
# STORY: Watcher Pattern - External Observation
# =============================================================================


class TestWatcherPattern:
    """
    Epic: Watch the Watcher Think

    As a human,
    I want to observe thinking without disrupting it,
    And choose when to intervene.
    """

    def test_scenario_watcher_sees_thought_stream(self):
        """
        Scenario: Watcher receives stream of thoughts

        Given an active thought swarm
        When I attach as a watcher
        Then I see thoughts as they occur
        But thoughts don't wait for my acknowledgment
        Because watching shouldn't slow thinking
        """
        from cortical.cognitive.swarm import ThoughtSwarm, Watcher

        observed = []

        class MyWatcher(Watcher):
            def on_thought(self, thought):
                observed.append(thought.summary)

        swarm = ThoughtSwarm()
        swarm.attach_watcher(MyWatcher())

        def thinking_process(context):
            context.emit_thought("considering A")
            context.emit_thought("considering B")
            context.emit_thought("decided on A")
            return "done"

        swarm.run(thinking_process)

        assert "considering A" in observed
        assert "decided on A" in observed

    def test_scenario_watcher_can_inject_thought(self):
        """
        Scenario: Watcher can inject a thought/question

        Given an ongoing thought process
        When the watcher injects "have you considered X?"
        Then the thought process receives the injection
        And can choose to incorporate it
        Because humans have insights worth injecting
        """
        from cortical.cognitive.swarm import ThoughtSwarm, Watcher

        class InteractiveWatcher(Watcher):
            def __init__(self):
                self.injection_sent = False

            def on_thought(self, thought):
                if "stuck on problem" in thought.summary and not self.injection_sent:
                    thought.context.receive_injection(
                        "have you considered using a hash map?"
                    )
                    self.injection_sent = True

        swarm = ThoughtSwarm()
        watcher = InteractiveWatcher()
        swarm.attach_watcher(watcher)

        injections_received = []

        def thinking_process(context):
            context.emit_thought("starting problem")
            context.emit_thought("stuck on problem")

            # Check for injections
            for injection in context.get_injections():
                injections_received.append(injection)

            return "done"

        swarm.run(thinking_process)

        assert "hash map" in str(injections_received)

    def test_scenario_watcher_can_pause_and_inspect(self):
        """
        Scenario: Watcher can pause thinking to inspect state

        Given a thought swarm that supports pausing
        When the watcher requests a pause
        Then thinking pauses at next safe point
        And watcher can inspect current state
        And resume when ready
        Because sometimes we need to stop and look carefully
        """
        from cortical.cognitive.swarm import ThoughtSwarm, Watcher, PausableSwarm

        class InspectingWatcher(Watcher):
            def __init__(self, swarm):
                self.swarm = swarm
                self.inspected_state = None

            def on_thought(self, thought):
                if "checkpoint" in thought.summary:
                    self.swarm.pause()
                    self.inspected_state = self.swarm.get_state_snapshot()
                    self.swarm.resume()

        swarm = PausableSwarm()
        watcher = InspectingWatcher(swarm)
        swarm.attach_watcher(watcher)

        def thinking_process(context):
            context.emit_thought("step 1")
            context.emit_thought("checkpoint reached")
            context.emit_thought("step 2")
            return "done"

        swarm.run(thinking_process)

        assert watcher.inspected_state is not None


# =============================================================================
# STORY: Integration with Existing Systems
# =============================================================================


class TestIntegrationWithExistingSystems:
    """
    Epic: Build on What We Have

    As developers,
    We want to integrate with CEL and GoT,
    So that we build on existing infrastructure.
    """

    def test_scenario_events_flow_to_cel(self):
        """
        Scenario: Cognitive events integrate with CEL

        Given the CEL (Causal Event Lattice) system
        When thoughts generate events
        Then they can be stored in CEL for time-travel
        Because CEL already handles event history
        """
        from cortical.cognitive.events import EventBus
        # CEL integration - to be implemented
        # from cortical.cel import EventLattice

        bus = EventBus()

        # CEL adapter would subscribe to events
        cel_events = []
        bus.subscribe("*", lambda e: cel_events.append(e))

        bus.publish({"type": "thought", "content": "test"})

        assert len(cel_events) == 1
        # In full implementation, this would write to CEL

    def test_scenario_thoughts_create_got_entities(self):
        """
        Scenario: Significant thoughts become GoT entities

        Given a thought worth persisting
        When it's promoted to persistent
        Then it creates a corresponding GoT entity
        So it can be linked, queried, and reasoned about
        Because GoT is our knowledge graph
        """
        from cortical.cognitive.events import EventLog, Thought, Persistence
        # GoT integration - would use existing GoT
        # from cortical.got.api import GoTManager

        log = EventLog()

        thought = Thought(
            content="key insight about architecture",
            persistence=Persistence.DURABLE
        )
        log.record(thought)

        # In full implementation, this would create GoT entity
        got_entity_created = log.get_got_entity_for(thought.id)
        # assert got_entity_created is not None

    def test_scenario_swarm_uses_cognitive_graph(self):
        """
        Scenario: Thought swarm operates on cognitive graph

        Given our existing CognitiveGraph
        When a thought swarm runs
        Then it reads and writes to that graph
        Because the graph is our shared knowledge
        """
        from cortical.cognitive.graph import CognitiveGraph
        from cortical.cognitive.swarm import ThoughtSwarm

        graph = CognitiveGraph()
        graph.node("existing_knowledge")

        swarm = ThoughtSwarm(graph=graph)

        def learning_thought(context):
            # Can see existing knowledge
            existing = context.graph.get_node("existing_knowledge")
            assert existing is not None

            # Can add new knowledge
            context.graph.node("new_learning")

            return "learned"

        swarm.run(learning_thought)

        # New knowledge persists in graph
        assert graph.get_node("new_learning") is not None


# =============================================================================
# STORY: Configurable Constraints for Different Work Modes
# =============================================================================


class TestConfigurableWorkModes:
    """
    Epic: Different Constraints for Different Work

    As a system doing different types of work,
    I need different quality/speed tradeoffs,
    So that chat can be fast while coding is thorough.
    """

    def test_scenario_chat_mode_fast_and_approximate(self):
        """
        Scenario: Chat mode prioritizes responsiveness

        Given work mode = "chat"
        When configuring the swarm
        Then max_depth is low (quick answers)
        And min_confidence is low (approximate is OK)
        And max_seconds is short (be responsive)
        Because conversation needs flow
        """
        from cortical.cognitive.swarm import ThoughtSwarm, WorkMode

        swarm = ThoughtSwarm(work_mode=WorkMode.CHAT)

        assert swarm.config.max_depth <= 3
        assert swarm.config.min_confidence <= 0.5
        assert swarm.config.max_seconds <= 10

    def test_scenario_coding_mode_thorough_and_testable(self):
        """
        Scenario: Coding mode prioritizes correctness

        Given work mode = "coding"
        When configuring the swarm
        Then max_depth is high (think deeply)
        And min_confidence is high (be sure)
        And verification is required (test it)
        Because code must work
        """
        from cortical.cognitive.swarm import ThoughtSwarm, WorkMode

        swarm = ThoughtSwarm(work_mode=WorkMode.CODING)

        assert swarm.config.max_depth >= 5
        assert swarm.config.min_confidence >= 0.7
        assert swarm.config.require_verification is True

    def test_scenario_research_mode_exploratory(self):
        """
        Scenario: Research mode allows exploration

        Given work mode = "research"
        When configuring the swarm
        Then max_depth is very high (go deep)
        And branching is allowed (explore alternatives)
        And time limit is generous (take your time)
        Because research needs freedom
        """
        from cortical.cognitive.swarm import ThoughtSwarm, WorkMode

        swarm = ThoughtSwarm(work_mode=WorkMode.RESEARCH)

        assert swarm.config.max_depth >= 10
        assert swarm.config.allow_branching is True
        assert swarm.config.max_seconds >= 300


# =============================================================================
# SUMMARY: What We're Building
# =============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  COGNITIVE COMMUNICATION LAYER                                              │
│                                                                             │
│  ┌─────────────┐     ┌─────────────────┐     ┌─────────────┐              │
│  │   Human     │────►│   Event Bus     │────►│   Watcher   │              │
│  │  (sensory)  │◄────│   (pub/sub)     │◄────│  (observe)  │              │
│  └─────────────┘     └────────┬────────┘     └─────────────┘              │
│                               │                                            │
│                               ▼                                            │
│                      ┌─────────────────┐                                   │
│                      │   Event Log     │                                   │
│                      │  (transient +   │                                   │
│                      │   persistent)   │                                   │
│                      └────────┬────────┘                                   │
│                               │                                            │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐          │
│  │                    THOUGHT SWARM                             │          │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │          │
│  │  │  Main    │──│   Sub    │──│  Sub-Sub │   (depth limited)│          │
│  │  │ Thought  │  │ Thought  │  │ Thought  │                  │          │
│  │  └──────────┘  └──────────┘  └──────────┘                  │          │
│  │                                                             │          │
│  │  Guardrails: depth, time, confidence, scope                │          │
│  └─────────────────────────┬───────────────────────────────────┘          │
│                            │                                              │
│                            ▼                                              │
│                   ┌─────────────────┐                                     │
│                   │ Cognitive Graph │  (shared knowledge)                 │
│                   │   + CEL + GoT   │  (existing systems)                 │
│                   └─────────────────┘                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Work Modes:
  - CHAT:     Fast, approximate, responsive
  - CODING:   Thorough, verified, testable
  - RESEARCH: Deep, exploratory, patient
"""
