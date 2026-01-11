"""
Behavioral Specifications: Unified Thought Model

Key Insights (from discussion):
1. ALL THOUGHTS ARE EQUAL - No main/sub distinction. A thought is a thought.
2. PERMISSIONS NOT RESTRICTIONS - Guardrails are what you CAN do, not what you can't.
3. STATIC TESTABLE MODEL - Unit testable thought with DI, returns answers for exact queries.
4. DYNAMIC DEPTH - Start small, grow if insufficient, restart if needed.
5. WORK QUEUE - Failures and pending work in a queue both can adjust.
6. SCALE GRADUALLY - Start with one worker, add as needed.

Core Principle:
    "Recursive problem decomposition with self-answering"

    A thought that needs information spawns another thought.
    That thought has the same power and constraints.
    Thoughts are data that can be transferred and built upon.
"""

import pytest
from typing import Protocol, List, Dict, Any, Optional, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import uuid


# =============================================================================
# CORE: The Thought Model
# =============================================================================


class TestUnifiedThoughtModel:
    """
    Epic: All Thoughts Are Equal

    As a cognitive system,
    Thoughts are thoughts - no hierarchy.
    A spawned thought has the same power as the spawning thought.
    """

    def test_scenario_thought_is_a_thought(self):
        """
        Scenario: No distinction between "main" and "sub" thoughts

        Given a thought A that spawns thought B
        When thought B runs
        Then B has the same capabilities as A
        And B can spawn thought C with the same capabilities
        Because a thought is just a thought
        """
        from cortical.cognitive.thought import Thought, ThoughtContext

        capabilities_seen = []

        def thought_a(ctx: ThoughtContext) -> str:
            capabilities_seen.append(("A", set(ctx.permissions)))

            # Spawn another thought - it should have same power
            answer = ctx.spawn_thought(thought_b)

            return f"A got: {answer}"

        def thought_b(ctx: ThoughtContext) -> str:
            capabilities_seen.append(("B", set(ctx.permissions)))

            # B can also spawn thoughts
            answer = ctx.spawn_thought(thought_c)

            return f"B got: {answer}"

        def thought_c(ctx: ThoughtContext) -> str:
            capabilities_seen.append(("C", set(ctx.permissions)))
            return "C answers"

        thought = Thought(thought_a)
        result = thought.run()

        # All thoughts had same capabilities
        a_caps = capabilities_seen[0][1]
        b_caps = capabilities_seen[1][1]
        c_caps = capabilities_seen[2][1]

        assert a_caps == b_caps == c_caps

    def test_scenario_thought_is_data(self):
        """
        Scenario: Thoughts are data that can be transferred and built upon

        Given a thought with its context and result
        When serialized to data
        Then it can be stored, transferred, and rebuilt
        Because thoughts are first-class data objects
        """
        from cortical.cognitive.thought import Thought, ThoughtData

        def my_thought(ctx) -> str:
            return "my answer"

        thought = Thought(my_thought)
        result = thought.run()

        # Serialize to data
        data: ThoughtData = thought.to_data()

        assert data.query == "my_thought"  # or function reference
        assert data.answer == "my answer"
        assert data.spawned_thoughts == []
        assert data.timestamp is not None

        # Can rebuild from data
        rebuilt = Thought.from_data(data)
        assert rebuilt.answer == "my answer"


# =============================================================================
# STATIC TESTABLE THOUGHT MODEL
# =============================================================================


class TestStaticThoughtModel:
    """
    Epic: Unit Testable Thoughts

    As a developer,
    I want a static thought model that returns exact answers for exact queries,
    So that I can unit test without running the full system.
    """

    def test_scenario_static_thought_with_injected_answerer(self):
        """
        Scenario: Inject a mock answerer for testing

        Given a thought that would normally query the cognitive graph
        When I inject a mock answerer
        Then the thought uses the mock
        And returns predictable, testable results
        Because dependencies are injected through constructors
        """
        from cortical.cognitive.thought import Thought, ThoughtContext, Answerer

        # Mock answerer that returns canned responses
        class MockAnswerer(Answerer):
            def __init__(self, responses: Dict[str, str]):
                self.responses = responses
                self.queries_received = []

            def answer(self, query: str) -> str:
                self.queries_received.append(query)
                return self.responses.get(query, "unknown")

        mock = MockAnswerer({
            "what is a cat?": "a furry animal",
            "what is an animal?": "a living thing",
        })

        def curious_thought(ctx: ThoughtContext) -> str:
            cat_answer = ctx.ask("what is a cat?")
            animal_answer = ctx.ask("what is an animal?")
            return f"cat: {cat_answer}, animal: {animal_answer}"

        thought = Thought(curious_thought, answerer=mock)
        result = thought.run()

        assert result == "cat: a furry animal, animal: a living thing"
        assert mock.queries_received == ["what is a cat?", "what is an animal?"]

    def test_scenario_static_thought_with_injected_graph(self):
        """
        Scenario: Inject a pre-populated graph for testing

        Given a thought that queries the cognitive graph
        When I inject a graph with known contents
        Then the thought operates on that graph
        And I can verify the results
        Because the graph is a constructor dependency
        """
        from cortical.cognitive.thought import Thought, ThoughtContext
        from cortical.cognitive.graph import CognitiveGraph, TruthValue

        # Pre-populated graph for testing
        graph = CognitiveGraph()
        graph.node("cat")
        graph.node("animal")
        graph.link(
            graph.node("cat"),
            graph.node("animal"),
            TruthValue(0.99, 0.9)
        )

        def graph_query_thought(ctx: ThoughtContext) -> str:
            cat = ctx.graph.get_node("cat")
            if cat:
                incoming = ctx.graph.get_incoming(cat.id)
                return f"cat has {len(incoming)} incoming links"
            return "cat not found"

        thought = Thought(graph_query_thought, graph=graph)
        result = thought.run()

        # Predictable result based on injected graph
        assert "cat" in result

    def test_scenario_static_thought_records_spawned_thoughts(self):
        """
        Scenario: Track all thoughts spawned during execution

        Given a thought that spawns other thoughts
        When it runs
        Then all spawned thoughts are recorded
        And their queries and answers are available
        Because we need to trace thought chains for testing
        """
        from cortical.cognitive.thought import Thought, ThoughtContext, ThoughtRecord

        def parent_thought(ctx: ThoughtContext) -> str:
            a = ctx.spawn_thought(lambda c: "answer A", query="what is A?")
            b = ctx.spawn_thought(lambda c: "answer B", query="what is B?")
            return f"got {a} and {b}"

        thought = Thought(parent_thought)
        result = thought.run()

        records: List[ThoughtRecord] = thought.get_spawned_records()

        assert len(records) == 2
        assert records[0].query == "what is A?"
        assert records[0].answer == "answer A"
        assert records[1].query == "what is B?"
        assert records[1].answer == "answer B"


# =============================================================================
# PERMISSIONS MODEL (Not Restrictions)
# =============================================================================


class TestPermissionsModel:
    """
    Epic: Guardrails as Permissions

    As a cognitive system,
    Guardrails define what I CAN do, not what I can't.
    Permissions are granted, not restrictions imposed.
    """

    def test_scenario_thought_receives_permissions(self):
        """
        Scenario: Thought operates within granted permissions

        Given permissions: ["read_repository", "create_tests", "ask_questions"]
        When a thought runs
        Then it can do those things
        And knows what it's permitted to do
        Because permissions are explicit grants
        """
        from cortical.cognitive.thought import Thought, ThoughtContext, Permission

        def permission_aware_thought(ctx: ThoughtContext) -> str:
            actions = []

            if ctx.has_permission(Permission.READ_REPOSITORY):
                actions.append("reading repo")

            if ctx.has_permission(Permission.CREATE_TESTS):
                actions.append("creating tests")

            if ctx.has_permission(Permission.MODIFY_PRODUCTION):
                actions.append("modifying production")  # Won't happen

            return f"did: {actions}"

        thought = Thought(
            permission_aware_thought,
            permissions=[Permission.READ_REPOSITORY, Permission.CREATE_TESTS]
        )
        result = thought.run()

        assert "reading repo" in result
        assert "creating tests" in result
        assert "modifying production" not in result

    def test_scenario_spawned_thought_inherits_permissions(self):
        """
        Scenario: Spawned thoughts get same permissions

        Given a thought with certain permissions
        When it spawns another thought
        Then the spawned thought has the same permissions
        Because a thought is a thought (equal power)
        """
        from cortical.cognitive.thought import Thought, ThoughtContext, Permission

        child_permissions = None

        def parent_thought(ctx: ThoughtContext) -> str:
            def child_thought(child_ctx: ThoughtContext) -> str:
                nonlocal child_permissions
                child_permissions = set(child_ctx.permissions)
                return "child done"

            ctx.spawn_thought(child_thought)
            return "parent done"

        parent_perms = [Permission.READ_REPOSITORY, Permission.ASK_QUESTIONS]
        thought = Thought(parent_thought, permissions=parent_perms)
        thought.run()

        assert child_permissions == set(parent_perms)

    def test_scenario_permission_request_goes_to_queue(self):
        """
        Scenario: Requesting new permissions queues for approval

        Given a thought that needs a permission it doesn't have
        When it requests that permission
        Then the request goes to the work queue
        And the thought can wait or proceed without it
        Because permission escalation is a queued operation
        """
        from cortical.cognitive.thought import Thought, ThoughtContext, Permission, WorkQueue

        queue = WorkQueue()

        def permission_requesting_thought(ctx: ThoughtContext) -> str:
            if not ctx.has_permission(Permission.DEPLOY):
                # Request it - goes to queue
                ctx.request_permission(Permission.DEPLOY, reason="need to deploy feature")
                return "requested deploy permission, proceeding without it"

            return "deployed"

        thought = Thought(
            permission_requesting_thought,
            permissions=[Permission.READ_REPOSITORY],
            work_queue=queue
        )
        result = thought.run()

        assert "requested" in result
        assert len(queue.pending_permissions) == 1
        assert queue.pending_permissions[0].permission == Permission.DEPLOY


# =============================================================================
# DYNAMIC DEPTH MODEL
# =============================================================================


class TestDynamicDepthModel:
    """
    Epic: Depth Based on Need

    As a cognitive system,
    Depth is not a hard limit but a budget that can grow.
    Start small, request more if insufficient.
    Some answers take unknown time - that's okay.
    """

    def test_scenario_start_with_minimal_depth(self):
        """
        Scenario: Start with small depth budget

        Given a new thought
        When it begins
        Then it has a small initial depth budget
        And can request more if needed
        Because we start small and grow
        """
        from cortical.cognitive.thought import Thought, ThoughtContext

        def shallow_thought(ctx: ThoughtContext) -> str:
            return f"started with depth budget: {ctx.depth_budget}"

        thought = Thought(shallow_thought)
        result = thought.run()

        # Default is small
        assert "depth budget: 3" in result or "depth budget: 2" in result

    def test_scenario_request_more_depth_when_insufficient(self):
        """
        Scenario: Request depth extension when answer is insufficient

        Given a thought that realizes it needs to go deeper
        When it requests more depth
        Then the request goes to the work queue
        And can be approved for continuation
        Because depth is a budget, not a wall
        """
        from cortical.cognitive.thought import Thought, ThoughtContext, WorkQueue, DepthRequest

        queue = WorkQueue()

        def deep_thinker(ctx: ThoughtContext) -> str:
            if ctx.depth_budget < 5:
                ctx.request_depth_extension(
                    additional_depth=3,
                    reason="need deeper analysis for complex problem"
                )
                return "insufficient_depth:requested_more"

            return "deep analysis complete"

        thought = Thought(deep_thinker, depth_budget=2, work_queue=queue)
        result = thought.run()

        assert "insufficient_depth" in result
        assert len(queue.pending_depth_requests) == 1
        assert queue.pending_depth_requests[0].additional_depth == 3

    def test_scenario_restart_thought_with_more_depth(self):
        """
        Scenario: Restart a thought with extended depth

        Given a thought that returned insufficient answer
        When depth extension is approved
        Then the thought can be restarted with new budget
        And continues from where conceptually left off
        Because thoughts can be resumed with more resources
        """
        from cortical.cognitive.thought import Thought, ThoughtContext, ThoughtRunner

        attempts = []

        def restartable_thought(ctx: ThoughtContext) -> str:
            attempts.append(ctx.depth_budget)

            if ctx.depth_budget < 5:
                return f"need more depth (have {ctx.depth_budget})"

            return f"complete with depth {ctx.depth_budget}"

        runner = ThoughtRunner()

        # First attempt
        result1 = runner.run(restartable_thought, depth_budget=2)
        assert "need more depth" in result1

        # Restart with more depth (approved by human/system)
        result2 = runner.run(restartable_thought, depth_budget=7)
        assert "complete with depth 7" in result2

        assert attempts == [2, 7]


# =============================================================================
# WORK QUEUE MODEL
# =============================================================================


class TestWorkQueueModel:
    """
    Epic: Failures and Pending Work in a Queue

    As a collaborative system (human + AI),
    Both can add to and adjust the work queue.
    Failures become queued work items.
    """

    def test_scenario_failed_thought_queues_retry(self):
        """
        Scenario: Failed thoughts go to queue for retry

        Given a thought that fails
        When the failure is recorded
        Then it goes to the work queue
        And can be retried later with adjustments
        Because failures are work items, not dead ends
        """
        from cortical.cognitive.thought import Thought, ThoughtContext, WorkQueue

        queue = WorkQueue()

        def failing_thought(ctx: ThoughtContext) -> str:
            raise ValueError("something went wrong")

        thought = Thought(failing_thought, work_queue=queue)

        try:
            thought.run()
        except ValueError:
            pass

        # Failure is in queue
        assert len(queue.failed_items) == 1
        assert queue.failed_items[0].error == "something went wrong"
        assert queue.failed_items[0].can_retry is True

    def test_scenario_human_adds_to_queue(self):
        """
        Scenario: Human can add work items to queue

        Given a work queue
        When the human adds "please research X"
        Then it appears in the queue
        And can be picked up by a thought
        Because both human and AI can add work
        """
        from cortical.cognitive.thought import WorkQueue, WorkItem

        queue = WorkQueue()

        # Human adds work
        queue.add(WorkItem(
            query="please research the best testing patterns",
            priority=0.8,
            added_by="human"
        ))

        assert len(queue.pending_items) == 1
        assert queue.pending_items[0].added_by == "human"

    def test_scenario_thought_adds_to_queue(self):
        """
        Scenario: Thought can add work items to queue

        Given a running thought
        When it discovers work that should be done later
        Then it can queue that work
        And continue with its current task
        Because not everything needs immediate resolution
        """
        from cortical.cognitive.thought import Thought, ThoughtContext, WorkQueue, WorkItem

        queue = WorkQueue()

        def queuing_thought(ctx: ThoughtContext) -> str:
            # Found something to do later
            ctx.queue_work(WorkItem(
                query="optimize this algorithm later",
                priority=0.3,
                added_by="thought"
            ))

            # Continue with main task
            return "main task done, queued optimization"

        thought = Thought(queuing_thought, work_queue=queue)
        result = thought.run()

        assert "queued optimization" in result
        assert len(queue.pending_items) == 1
        assert queue.pending_items[0].added_by == "thought"

    def test_scenario_both_can_reorder_queue(self):
        """
        Scenario: Queue can be reordered by human or system

        Given a queue with multiple items
        When priorities are adjusted
        Then the order changes
        Because the queue is collaborative
        """
        from cortical.cognitive.thought import WorkQueue, WorkItem

        queue = WorkQueue()

        queue.add(WorkItem(query="task A", priority=0.5))
        queue.add(WorkItem(query="task B", priority=0.3))
        queue.add(WorkItem(query="task C", priority=0.7))

        # Before reorder
        order_before = [item.query for item in queue.get_ordered()]
        assert order_before == ["task C", "task A", "task B"]

        # Human bumps task B
        queue.set_priority("task B", 0.9)

        order_after = [item.query for item in queue.get_ordered()]
        assert order_after == ["task B", "task C", "task A"]


# =============================================================================
# SCALING MODEL
# =============================================================================


class TestScalingModel:
    """
    Epic: Start Small, Scale Gradually

    As a system,
    Start with one worker.
    Add more as needed.
    Slow and steady wins the race.
    """

    def test_scenario_single_worker_default(self):
        """
        Scenario: Start with one worker

        Given a new thought runner
        When it starts
        Then it has one worker
        And processes thoughts serially
        Because we start simple
        """
        from cortical.cognitive.thought import ThoughtRunner

        runner = ThoughtRunner()

        assert runner.worker_count == 1

    def test_scenario_scale_up_on_demand(self):
        """
        Scenario: Add workers when needed

        Given a runner with queued work
        When demand exceeds capacity
        Then workers can be added
        Up to a configured maximum
        Because we scale to meet need
        """
        from cortical.cognitive.thought import ThoughtRunner, WorkQueue, WorkItem

        queue = WorkQueue()
        for i in range(10):
            queue.add(WorkItem(query=f"task {i}", priority=0.5))

        runner = ThoughtRunner(work_queue=queue, max_workers=4)

        assert runner.worker_count == 1

        # Request scale up
        runner.scale_to_demand()

        assert runner.worker_count >= 1
        assert runner.worker_count <= 4

    def test_scenario_scale_down_when_idle(self):
        """
        Scenario: Remove workers when idle

        Given multiple workers
        When work is complete and queue is empty
        Then workers scale back down
        To conserve resources
        Because we don't keep resources we don't need
        """
        from cortical.cognitive.thought import ThoughtRunner

        runner = ThoughtRunner(max_workers=4)
        runner.scale_to(4)  # Manually scale up

        assert runner.worker_count == 4

        # No work, scale down
        runner.scale_to_demand()

        assert runner.worker_count == 1


# =============================================================================
# THOUGHT COMMUNICATION MODEL
# =============================================================================


class TestThoughtCommunicationModel:
    """
    Epic: Thoughts Communicate with Full Respect

    As a system where thoughts are equal,
    Thoughts can communicate with each other.
    Each operates at full capability.
    Data transfers between thoughts build understanding.
    """

    def test_scenario_thought_sends_data_to_another(self):
        """
        Scenario: One thought sends data to another

        Given thought A that discovers information
        When it sends that data to thought B
        Then B receives and can build on it
        Because thoughts share and build on each other's work
        """
        from cortical.cognitive.thought import Thought, ThoughtContext, ThoughtChannel

        channel = ThoughtChannel()

        def sender_thought(ctx: ThoughtContext) -> str:
            # Discover something
            discovery = {"pattern": "cats are animals", "confidence": 0.9}

            # Send to channel for other thoughts
            ctx.send_to_channel("discoveries", discovery)

            return "sent discovery"

        def receiver_thought(ctx: ThoughtContext) -> str:
            # Wait for data
            data = ctx.receive_from_channel("discoveries")

            if data:
                return f"received: {data['pattern']}"
            return "no data"

        # Run sender
        Thought(sender_thought, channel=channel).run()

        # Run receiver
        result = Thought(receiver_thought, channel=channel).run()

        assert "cats are animals" in result

    def test_scenario_thoughts_build_shared_understanding(self):
        """
        Scenario: Multiple thoughts contribute to shared understanding

        Given multiple thoughts working on related problems
        When each contributes findings to shared graph
        Then collective understanding grows
        Because thoughts collaborate, not compete
        """
        from cortical.cognitive.thought import Thought, ThoughtContext
        from cortical.cognitive.graph import CognitiveGraph

        graph = CognitiveGraph()

        def thought_1(ctx: ThoughtContext) -> str:
            ctx.graph.node("discovery_1")
            return "added discovery 1"

        def thought_2(ctx: ThoughtContext) -> str:
            ctx.graph.node("discovery_2")
            return "added discovery 2"

        def thought_3(ctx: ThoughtContext) -> str:
            # Build on previous discoveries
            d1 = ctx.graph.get_node("discovery_1")
            d2 = ctx.graph.get_node("discovery_2")
            if d1 and d2:
                ctx.graph.node("synthesis")
                return "synthesized"
            return "waiting for others"

        Thought(thought_1, graph=graph).run()
        Thought(thought_2, graph=graph).run()
        result = Thought(thought_3, graph=graph).run()

        assert result == "synthesized"
        assert graph.get_node("synthesis") is not None


# =============================================================================
# RECURSIVE PROBLEM DECOMPOSITION
# =============================================================================


class TestRecursiveProblemDecomposition:
    """
    Epic: Recursive Problem Decomposition with Self-Answering

    As a cognitive system,
    When I face a complex problem,
    I decompose it into sub-problems,
    Solve each (possibly decomposing further),
    And synthesize the answer.

    This is the core of how we think through hard problems.
    """

    def test_scenario_decompose_problem(self):
        """
        Scenario: Complex problem decomposed into parts

        Given a complex query
        When a thought processes it
        Then it identifies sub-problems
        And spawns thoughts for each
        And synthesizes results
        Because complex problems are made of simpler parts
        """
        from cortical.cognitive.thought import Thought, ThoughtContext

        decomposition_log = []

        def complex_thought(ctx: ThoughtContext) -> str:
            decomposition_log.append("analyzing complex problem")

            # Decompose
            part_a = ctx.spawn_thought(
                lambda c: "solution to A",
                query="solve part A"
            )
            decomposition_log.append(f"got part A: {part_a}")

            part_b = ctx.spawn_thought(
                lambda c: "solution to B",
                query="solve part B"
            )
            decomposition_log.append(f"got part B: {part_b}")

            # Synthesize
            synthesis = f"combined: {part_a} + {part_b}"
            decomposition_log.append(f"synthesized: {synthesis}")

            return synthesis

        thought = Thought(complex_thought)
        result = thought.run()

        assert "combined:" in result
        assert len(decomposition_log) == 4

    def test_scenario_self_answering(self):
        """
        Scenario: Thought answers its own questions

        Given a thought that needs information
        When it spawns a thought to find that information
        Then it gets an answer without external help
        And can continue reasoning
        Because this is self-answering
        """
        from cortical.cognitive.thought import Thought, ThoughtContext

        def self_questioning_thought(ctx: ThoughtContext) -> str:
            # I need to know something
            answer = ctx.spawn_thought(
                lambda c: "the answer is 42",
                query="what is the answer to everything?"
            )

            # Now I can use that answer
            return f"I learned: {answer}"

        thought = Thought(self_questioning_thought)
        result = thought.run()

        assert "I learned: the answer is 42" in result


# =============================================================================
# SUMMARY: The Unified Model
# =============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  UNIFIED THOUGHT MODEL                                                      │
│                                                                             │
│  Principles:                                                                │
│    1. A thought is a thought (no hierarchy)                                │
│    2. Permissions, not restrictions                                        │
│    3. Depth is a budget, not a wall                                        │
│    4. Failures go to queue, not void                                       │
│    5. Start with one, scale as needed                                      │
│    6. Thoughts are data, transferable and buildable                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                           THOUGHT                                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │  │
│  │  │ Permissions │  │ Depth Budget│  │ Work Queue  │                 │  │
│  │  │ (what I CAN │  │ (grows on   │  │ (shared w/  │                 │  │
│  │  │  do)        │  │  demand)    │  │  human)     │                 │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │  │
│  │                                                                      │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │                    SPAWN THOUGHT                             │   │  │
│  │  │    (same permissions, same power, same model)               │   │  │
│  │  │                                                              │   │  │
│  │  │    ┌──────────┐    ┌──────────┐    ┌──────────┐            │   │  │
│  │  │    │ Thought  │────│ Thought  │────│ Thought  │            │   │  │
│  │  │    └──────────┘    └──────────┘    └──────────┘            │   │  │
│  │  │                                                              │   │  │
│  │  │    All equal. All data. All buildable.                      │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                                                                      │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │  │
│  │  │   Graph     │  │  Answerer   │  │  Channel    │                 │  │
│  │  │ (injected)  │  │ (injected)  │  │ (injected)  │                 │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │  │
│  │                                                                      │  │
│  │  Dependencies injected through constructor for testability         │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Testing:                                                                  │
│    - Inject mock answerer → exact responses for exact queries             │
│    - Inject mock graph → known state for verification                     │
│    - Inject mock queue → observe queued items                             │
│    - All spawned thoughts recorded for tracing                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
