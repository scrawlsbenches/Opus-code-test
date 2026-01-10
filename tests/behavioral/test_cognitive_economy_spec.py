"""
Behavioral Specifications: Cognitive Economy Model

A cognitive operating system for thinking that manages:
- Foreground processes (immediate conversation)
- Background processes (long-running research, infinite time allowed)
- Resource allocation (attention market, economic model)
- Partial results (useful incomplete answers)
- Multiple variants (same question, different approaches)
- Meta-orchestration (model managing models)
- Human-AI collaboration (swarms of knowledge workers over time)

Core Metaphor: Stock Exchange for Attention
    - Priority = Price (higher priority gets resources first)
    - Attention = Currency (limited, must be allocated)
    - Workers = Traders (execute thoughts)
    - Results = Returns (immediate or deferred, partial or complete)

Two Modes of Thought:
    IMMEDIATE: "I need this now" - blocking, time-bounded, complete answer expected
    BACKGROUND: "Hunt for this" - non-blocking, infinite time allowed, partial answers welcome

Key Insight:
    Not all thinking is urgent. Some questions are worth pondering forever.
    Some answers are useful even when partial. Resources should flow dynamically.
"""

import pytest
from typing import Protocol, List, Dict, Any, Optional, Callable, Iterator, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import uuid
from datetime import datetime, timedelta


# =============================================================================
# CORE TYPES
# =============================================================================


class ThoughtMode(Enum):
    """Two fundamental modes of thinking."""
    IMMEDIATE = auto()   # "I need this now" - blocking, time-bounded
    BACKGROUND = auto()  # "Hunt for this" - non-blocking, can run forever


class AnswerStatus(Enum):
    """Status of an answer."""
    COMPLETE = auto()      # Full answer, high confidence
    PARTIAL = auto()       # Useful but incomplete
    PRELIMINARY = auto()   # Early guess, low confidence
    SEARCHING = auto()     # Still looking, no answer yet
    FAILED = auto()        # Couldn't find answer


@dataclass
class PartialAnswer:
    """
    An answer that is useful even when incomplete.

    Not everything is binary complete/incomplete.
    A 70% answer now may be more valuable than 100% answer never.
    """
    content: Any
    status: AnswerStatus
    confidence: float  # 0.0 to 1.0
    completeness: float  # 0.0 to 1.0
    can_improve: bool  # Can we continue and get better answer?
    improvement_estimate: Optional[str] = None  # "~2 more hours for 90%"

    def is_usable(self, min_confidence: float = 0.5) -> bool:
        """Is this answer good enough to use?"""
        return self.confidence >= min_confidence


@dataclass
class ThoughtPriority:
    """
    Economic priority for attention allocation.

    Like a price in a market - higher priority gets resources first.
    Can be adjusted dynamically by human or system.
    """
    value: float  # 0.0 to 1.0 (or unbounded for urgent)
    source: str   # "human", "system", "thought"
    reason: str   # Why this priority
    timestamp: datetime = field(default_factory=datetime.now)

    def __lt__(self, other):
        return self.value < other.value


# =============================================================================
# STORY: Two Modes of Thought
# =============================================================================


class TestTwoModesOfThought:
    """
    Epic: Immediate and Background Thinking

    As a cognitive system,
    Some questions need immediate answers (blocking).
    Some questions can be explored indefinitely (background).
    Both are valid and necessary.
    """

    def test_scenario_immediate_thought_blocks_until_complete(self):
        """
        Scenario: Immediate thought returns complete answer

        Given a question that needs immediate answer
        When I create an IMMEDIATE thought
        Then it blocks until answer is ready
        And returns a complete answer
        Because some questions can't wait
        """
        from cortical.cognitive.economy import Thought, ThoughtMode

        def quick_calculation(ctx) -> str:
            return "2 + 2 = 4"

        thought = Thought(
            quick_calculation,
            mode=ThoughtMode.IMMEDIATE,
            timeout_seconds=5.0
        )

        result = thought.run()  # Blocks until complete

        assert result.status == AnswerStatus.COMPLETE
        assert result.content == "2 + 2 = 4"
        assert thought.is_complete()

    def test_scenario_background_thought_returns_immediately(self):
        """
        Scenario: Background thought starts and returns handle

        Given a complex question that may take long time
        When I create a BACKGROUND thought
        Then it returns immediately with a handle
        And continues processing in background
        Because I shouldn't block on long-running research
        """
        from cortical.cognitive.economy import Thought, ThoughtMode, ThoughtHandle

        def long_research(ctx) -> str:
            # Would take a long time
            return "deep insight after much research"

        thought = Thought(
            long_research,
            mode=ThoughtMode.BACKGROUND,
            time_budget=None  # Infinite time allowed
        )

        handle: ThoughtHandle = thought.start()  # Returns immediately

        assert not thought.is_complete()  # Still running
        assert handle.thought_id is not None
        assert handle.can_check_status()

    def test_scenario_background_thought_with_infinite_time(self):
        """
        Scenario: Background thought can run indefinitely

        Given a research question with no deadline
        When I allow infinite time
        Then the thought runs until stopped or answered
        And I can check on it anytime
        Because some questions are worth pondering forever
        """
        from cortical.cognitive.economy import Thought, ThoughtMode, ThoughtRunner

        runner = ThoughtRunner()

        def open_research(ctx) -> str:
            # Could run for days/weeks/forever
            while not ctx.should_stop:
                insight = ctx.search_for_patterns()
                if insight.confidence > 0.9:
                    return insight
                ctx.yield_partial(insight)  # Emit partial results
            return ctx.best_so_far()

        handle = runner.start_background(
            open_research,
            time_budget=float('inf'),  # No time limit
            priority=ThoughtPriority(0.3, "system", "low priority research")
        )

        # Can check on it later
        status = runner.check(handle)
        assert status.is_running or status.has_partial_results

        # Can stop it anytime
        runner.stop(handle)

    def test_scenario_upgrade_background_to_immediate(self):
        """
        Scenario: Upgrade background thought to immediate priority

        Given a background thought that's been running
        When circumstances change and I need the answer now
        Then I can upgrade its priority
        And it gets more resources immediately
        Because priorities change
        """
        from cortical.cognitive.economy import ThoughtRunner, ThoughtPriority

        runner = ThoughtRunner()

        handle = runner.start_background(
            lambda ctx: "research result",
            priority=ThoughtPriority(0.2, "system", "background research")
        )

        # Initially low priority
        assert runner.get_priority(handle).value == 0.2

        # Upgrade to urgent
        runner.set_priority(
            handle,
            ThoughtPriority(0.95, "human", "need this now!")
        )

        assert runner.get_priority(handle).value == 0.95
        # Would now get more resources


# =============================================================================
# STORY: Partial Answers
# =============================================================================


class TestPartialAnswers:
    """
    Epic: Useful Incomplete Results

    As a cognitive system,
    Not every answer is complete or nothing.
    Partial answers have value.
    70% now may be better than 100% never.
    """

    def test_scenario_accept_partial_answer(self):
        """
        Scenario: Accept a partial answer as useful

        Given a thought that found partial answer
        When the partial answer meets minimum confidence
        Then I can accept it as useful
        And continue with work queue for improvement
        Because partial progress is still progress
        """
        from cortical.cognitive.economy import Thought, PartialAnswer, AnswerStatus

        def searching_thought(ctx) -> PartialAnswer:
            # Found something but not complete
            return PartialAnswer(
                content="Cats are probably mammals (found 3 sources)",
                status=AnswerStatus.PARTIAL,
                confidence=0.75,
                completeness=0.6,
                can_improve=True,
                improvement_estimate="~5 more minutes for 90% confidence"
            )

        thought = Thought(searching_thought)
        result = thought.run()

        assert result.status == AnswerStatus.PARTIAL
        assert result.is_usable(min_confidence=0.7)
        assert result.can_improve

    def test_scenario_queue_improvement_of_partial(self):
        """
        Scenario: Queue improvement work for partial answer

        Given a partial answer that can be improved
        When I accept the partial but want better
        Then I queue improvement work
        And the thought continues in background
        Because I can use partial now and get better later
        """
        from cortical.cognitive.economy import Thought, WorkQueue, PartialAnswer

        queue = WorkQueue()

        def improving_thought(ctx) -> PartialAnswer:
            current = ctx.current_answer or PartialAnswer(
                content="initial guess",
                status=AnswerStatus.PRELIMINARY,
                confidence=0.3,
                completeness=0.2,
                can_improve=True
            )

            # Improve if we have budget
            if ctx.has_time_remaining():
                improved = ctx.search_more()
                return improved

            return current

        thought = Thought(improving_thought, work_queue=queue)

        # First run - get partial
        result1 = thought.run(time_budget_seconds=1.0)
        assert result1.status in [AnswerStatus.PARTIAL, AnswerStatus.PRELIMINARY]

        # Queue for improvement
        queue.add_improvement(thought, target_confidence=0.9)

        assert len(queue.pending_improvements) == 1

    def test_scenario_stream_partial_results(self):
        """
        Scenario: Stream partial results as they're found

        Given a long-running background thought
        When it finds intermediate results
        Then it streams them to observers
        And I can use them while it continues
        Because incremental progress should be visible
        """
        from cortical.cognitive.economy import Thought, ThoughtMode, PartialStream

        partials_received = []

        def streaming_thought(ctx) -> str:
            for i in range(10):
                partial = f"found item {i}"
                ctx.emit_partial(partial)
            return "complete with 10 items"

        stream = PartialStream()
        stream.on_partial(lambda p: partials_received.append(p))

        thought = Thought(
            streaming_thought,
            mode=ThoughtMode.BACKGROUND,
            partial_stream=stream
        )
        thought.run()

        # Received partials as they were found
        assert len(partials_received) == 10


# =============================================================================
# STORY: Attention Market (Economic Allocation)
# =============================================================================


class TestAttentionMarket:
    """
    Epic: Economic Resource Allocation

    As a cognitive system with limited resources,
    Attention flows to highest priority like money to highest bidder.
    This creates efficient allocation without central planning.
    """

    def test_scenario_highest_priority_gets_resources_first(self):
        """
        Scenario: Priority determines resource allocation

        Given multiple thoughts competing for resources
        When the attention market allocates
        Then highest priority thought gets resources first
        Because priority = price in our attention economy
        """
        from cortical.cognitive.economy import AttentionMarket, Thought, ThoughtPriority

        market = AttentionMarket(total_capacity=10)

        thoughts = [
            Thought(lambda c: "low", priority=ThoughtPriority(0.2, "system", "low")),
            Thought(lambda c: "high", priority=ThoughtPriority(0.9, "human", "urgent")),
            Thought(lambda c: "medium", priority=ThoughtPriority(0.5, "system", "normal")),
        ]

        for t in thoughts:
            market.submit(t)

        # Allocation order
        allocation = market.get_allocation_order()

        assert allocation[0].priority.value == 0.9  # High first
        assert allocation[1].priority.value == 0.5  # Medium second
        assert allocation[2].priority.value == 0.2  # Low last

    def test_scenario_low_priority_runs_when_capacity_available(self):
        """
        Scenario: Low priority thoughts run during idle time

        Given a low-priority background thought
        When high-priority work is done and capacity is free
        Then low-priority thought gets resources
        Because even low-priority work eventually gets done
        """
        from cortical.cognitive.economy import AttentionMarket, Thought, ThoughtPriority

        market = AttentionMarket(total_capacity=10)
        execution_log = []

        def log_execution(name):
            def thought_fn(ctx):
                execution_log.append(name)
                return name
            return thought_fn

        # Submit in random order
        market.submit(Thought(log_execution("background"),
                             priority=ThoughtPriority(0.1, "system", "background research")))
        market.submit(Thought(log_execution("urgent"),
                             priority=ThoughtPriority(0.95, "human", "need now")))

        # Run until empty
        market.run_until_empty()

        # Both ran, urgent first
        assert execution_log == ["urgent", "background"]

    def test_scenario_dynamic_priority_adjustment(self):
        """
        Scenario: Priorities adjust dynamically

        Given a thought running at low priority
        When human or system increases priority
        Then it immediately gets more resources
        Because markets respond to price changes
        """
        from cortical.cognitive.economy import AttentionMarket, Thought, ThoughtPriority

        market = AttentionMarket(total_capacity=10)

        thought = Thought(
            lambda c: "result",
            priority=ThoughtPriority(0.2, "system", "background")
        )
        handle = market.submit(thought)

        initial_resources = market.get_allocated_resources(handle)

        # Human boosts priority
        market.update_priority(
            handle,
            ThoughtPriority(0.9, "human", "actually need this now")
        )

        new_resources = market.get_allocated_resources(handle)

        assert new_resources > initial_resources

    def test_scenario_resource_bidding(self):
        """
        Scenario: Thoughts can bid for more resources

        Given a thought that needs more resources to finish faster
        When it bids higher priority
        Then it may get more resources (if available)
        Because the market responds to demand
        """
        from cortical.cognitive.economy import AttentionMarket, Thought, ResourceBid

        market = AttentionMarket(total_capacity=100)

        def bidding_thought(ctx) -> str:
            if ctx.resources < 10:
                # Request more resources
                ctx.bid_for_resources(ResourceBid(
                    requested=20,
                    priority_increase=0.2,
                    reason="can finish 5x faster with more resources"
                ))
            return "done"

        thought = Thought(bidding_thought)
        market.submit(thought)

        # Market considers the bid
        market.process_bids()

        # If granted, thought gets more resources
        # (actual granting depends on availability)


# =============================================================================
# STORY: Multiple Answer Variants
# =============================================================================


class TestMultipleAnswerVariants:
    """
    Epic: Same Question, Different Approaches

    As a cognitive system,
    One question can have multiple valid answers.
    Different approaches (temperatures) yield different results.
    I can pick the best or synthesize them.
    """

    def test_scenario_generate_answer_variants(self):
        """
        Scenario: Generate multiple answers to same question

        Given a question with multiple valid approaches
        When I request variants
        Then I get several different answers
        And can compare/choose/synthesize
        Because diversity of thought leads to better answers
        """
        from cortical.cognitive.economy import Thought, VariantGenerator

        generator = VariantGenerator()

        def answerable_question(ctx) -> str:
            # Answer depends on approach/temperature
            return f"answer with approach {ctx.approach_id}"

        variants = generator.generate_variants(
            answerable_question,
            num_variants=3,
            diversity="high"  # Different temperatures/approaches
        )

        assert len(variants) == 3
        assert len(set(v.content for v in variants)) >= 2  # At least some diversity

    def test_scenario_select_best_variant(self):
        """
        Scenario: Select best variant by criteria

        Given multiple answer variants
        When I apply selection criteria
        Then I get the best one
        Because not all variants are equal
        """
        from cortical.cognitive.economy import VariantSelector, AnswerVariant

        variants = [
            AnswerVariant(content="short answer", confidence=0.6, approach="concise"),
            AnswerVariant(content="detailed thorough answer", confidence=0.9, approach="thorough"),
            AnswerVariant(content="creative unusual answer", confidence=0.7, approach="creative"),
        ]

        selector = VariantSelector()

        # Select by confidence
        best_confident = selector.select_by(variants, criterion="confidence")
        assert best_confident.approach == "thorough"

        # Select by creativity
        best_creative = selector.select_by(variants, criterion="creativity")
        assert best_creative.approach == "creative"

    def test_scenario_synthesize_variants(self):
        """
        Scenario: Synthesize multiple variants into better answer

        Given multiple partial/different answers
        When I synthesize them
        Then I get a combined answer better than any single one
        Because combining perspectives yields insight
        """
        from cortical.cognitive.economy import VariantSynthesizer, AnswerVariant

        variants = [
            AnswerVariant(content="A is true because of X", confidence=0.7),
            AnswerVariant(content="A is true because of Y", confidence=0.8),
            AnswerVariant(content="A might be false in edge cases", confidence=0.6),
        ]

        synthesizer = VariantSynthesizer()
        combined = synthesizer.synthesize(variants)

        # Combined answer includes all perspectives
        assert "X" in combined.content or "Y" in combined.content
        assert combined.confidence >= max(v.confidence for v in variants)


# =============================================================================
# STORY: Concurrent Thinking and Communication
# =============================================================================


class TestConcurrentThinkingAndCommunication:
    """
    Epic: Think and Talk Simultaneously

    As a cognitive system in dialogue,
    Background thinking shouldn't block conversation.
    I can research while we talk.
    Insights surface when ready.
    """

    def test_scenario_background_research_during_conversation(self):
        """
        Scenario: Research runs while conversation continues

        Given an ongoing conversation
        When I start background research
        Then conversation continues unblocked
        And research results arrive when ready
        Because thinking and talking are parallel activities
        """
        from cortical.cognitive.economy import (
            ConversationContext, BackgroundResearcher
        )

        conversation = ConversationContext()
        researcher = BackgroundResearcher(conversation)

        # Start research
        handle = researcher.start("find patterns in our discussion")

        # Conversation continues
        response1 = conversation.respond("What do you think about X?")
        response2 = conversation.respond("And what about Y?")

        # Research hasn't blocked us
        assert response1 is not None
        assert response2 is not None

        # Check if research has findings
        findings = researcher.check(handle)
        # findings may or may not be ready - that's fine

    def test_scenario_insight_surfaces_during_conversation(self):
        """
        Scenario: Background insight surfaces naturally

        Given background research that found something
        When the insight is relevant to current conversation
        Then it surfaces naturally in my response
        Because insights should flow into dialogue
        """
        from cortical.cognitive.economy import (
            ConversationContext, InsightSurface
        )

        conversation = ConversationContext()
        surface = InsightSurface(conversation)

        # Background thought completed with insight
        surface.add_pending_insight(
            "I noticed we keep coming back to testing patterns",
            relevance_keywords=["testing", "patterns", "quality"]
        )

        # When conversation touches relevant topic
        response = conversation.respond(
            "How should we approach testing?",
            allow_insights=True
        )

        # Insight may be woven in
        # (actual weaving depends on implementation)
        assert response is not None

    def test_scenario_long_running_collaboration(self):
        """
        Scenario: Coherent collaboration over long time

        Given a human-AI collaboration over days/weeks
        When we work on complex problems together
        Then context and understanding persist
        And we build on previous work coherently
        Because long-term collaboration requires memory
        """
        from cortical.cognitive.economy import (
            CollaborationSession, SessionMemory
        )

        memory = SessionMemory()

        # Day 1
        session1 = CollaborationSession(memory)
        session1.record_exchange("Let's design a thought system", "Great, here's my initial idea...")
        session1.record_decision("Use hypergraph for knowledge", "human")
        session1.end()

        # Day 2 - new session, same memory
        session2 = CollaborationSession(memory)

        # Can reference previous work
        previous = session2.recall("hypergraph decision")
        assert previous is not None

        # Build on it coherently
        session2.record_exchange(
            "Let's continue the hypergraph work",
            "Yes, building on our decision to use hypergraph..."
        )


# =============================================================================
# STORY: Meta-Orchestration
# =============================================================================


class TestMetaOrchestration:
    """
    Epic: Model Managing Models

    As a cognitive system,
    A meta-layer orchestrates knowledge workers.
    It questions its own understanding.
    It allocates attention economically.
    It ensures coherent collaboration.
    """

    def test_scenario_orchestrator_manages_thoughts(self):
        """
        Scenario: Orchestrator coordinates multiple thoughts

        Given multiple thoughts working on related problems
        When the orchestrator manages them
        Then work is coordinated, not duplicated
        And results are synthesized
        Because coordination improves efficiency
        """
        from cortical.cognitive.economy import (
            CognitiveOrchestrator, Thought
        )

        orchestrator = CognitiveOrchestrator()

        # Submit related work
        orchestrator.submit(Thought(lambda c: "research A"))
        orchestrator.submit(Thought(lambda c: "research B"))
        orchestrator.submit(Thought(lambda c: "synthesize A and B"))

        # Orchestrator sees the dependency
        plan = orchestrator.get_execution_plan()

        # Synthesis waits for A and B
        synthesis_task = [t for t in plan if "synthesize" in str(t)][0]
        assert synthesis_task.depends_on is not None

    def test_scenario_orchestrator_questions_understanding(self):
        """
        Scenario: Orchestrator questions its own understanding

        Given an orchestrator managing complex work
        When it detects potential misunderstanding
        Then it spawns clarifying thoughts
        And may ask human for clarification
        Because self-questioning improves accuracy
        """
        from cortical.cognitive.economy import (
            CognitiveOrchestrator, ClarificationRequest
        )

        orchestrator = CognitiveOrchestrator()

        # Process something ambiguous
        orchestrator.process("implement the thing we discussed")

        # Orchestrator detects ambiguity
        clarifications = orchestrator.get_pending_clarifications()

        # May have questions
        # (depends on what was actually discussed)

    def test_scenario_orchestrator_allocates_by_value(self):
        """
        Scenario: Orchestrator allocates resources by expected value

        Given limited resources and many possible thoughts
        When orchestrator decides what to work on
        Then it chooses highest expected value
        Considering priority, likelihood of success, and impact
        Because resources should flow to highest value work
        """
        from cortical.cognitive.economy import (
            CognitiveOrchestrator, WorkItem, ExpectedValue
        )

        orchestrator = CognitiveOrchestrator(total_resources=100)

        work_items = [
            WorkItem("easy low-impact", difficulty=0.2, impact=0.3),
            WorkItem("hard high-impact", difficulty=0.9, impact=0.95),
            WorkItem("medium medium-impact", difficulty=0.5, impact=0.6),
        ]

        for item in work_items:
            orchestrator.submit(item)

        # Orchestrator calculates expected value
        # EV = (1 - difficulty) * impact (simplified)
        allocation = orchestrator.get_value_based_allocation()

        # Should allocate more to best EV, not just easiest or highest impact


# =============================================================================
# STORY: Human-AI Swarm Collaboration
# =============================================================================


class TestHumanAISwarmCollaboration:
    """
    Epic: Swarms for Human Knowledge Work

    As a collaborative system,
    Humans and AI work together as knowledge workers.
    Each contributes strengths.
    The swarm is greater than the sum.
    """

    def test_scenario_human_and_ai_both_contribute(self):
        """
        Scenario: Both human and AI add to work queue

        Given a shared work queue
        When human adds work and AI adds work
        Then both contributions are tracked
        And can be prioritized together
        Because collaboration means both contribute
        """
        from cortical.cognitive.economy import (
            WorkQueue, WorkItem, Contributor
        )

        queue = WorkQueue()

        # Human adds
        queue.add(WorkItem(
            "research competitor pricing",
            contributor=Contributor.HUMAN,
            priority=0.8
        ))

        # AI adds
        queue.add(WorkItem(
            "analyze patterns in existing data",
            contributor=Contributor.AI,
            priority=0.6
        ))

        items = queue.get_all()
        contributors = {item.contributor for item in items}

        assert Contributor.HUMAN in contributors
        assert Contributor.AI in contributors

    def test_scenario_human_guides_ai_thinking(self):
        """
        Scenario: Human guides AI's thinking direction

        Given AI doing background research
        When human provides guidance
        Then AI adjusts its approach
        Because human intuition complements AI thoroughness
        """
        from cortical.cognitive.economy import (
            ThoughtRunner, Guidance, ThoughtHandle
        )

        runner = ThoughtRunner()

        handle = runner.start_background(lambda c: "researching...")

        # Human provides guidance
        runner.guide(handle, Guidance(
            direction="focus more on edge cases",
            priority_adjustment=0.1,
            source="human"
        ))

        # AI adjusts
        current_focus = runner.get_current_focus(handle)
        assert "edge cases" in current_focus or runner.received_guidance(handle)

    def test_scenario_long_term_coherent_collaboration(self):
        """
        Scenario: Maintain coherence over long collaboration

        Given weeks of human-AI collaboration
        When we reference previous work
        Then references are accurate and coherent
        And we build on solid foundation
        Because knowledge workers need institutional memory
        """
        from cortical.cognitive.economy import (
            CollaborationMemory, Reference
        )

        memory = CollaborationMemory()

        # Record work over time
        memory.record_work("week1", "designed initial architecture")
        memory.record_decision("week1", "use hypergraph for knowledge")
        memory.record_work("week2", "implemented cognitive graph")
        memory.record_learning("week2", "links-as-atoms enables meta-reasoning")

        # Later reference
        ref = memory.find_reference("hypergraph decision")
        assert ref is not None
        assert ref.week == "week1"

        # Can trace how decisions evolved
        evolution = memory.trace_evolution("knowledge representation")
        assert len(evolution) >= 2


# =============================================================================
# STORY: Imagination Functions for Recursive Thinking
# =============================================================================


class TestImaginationFunctions:
    """
    Epic: Think Recursively with Imagination

    As a cognitive system solving complex problems,
    I can imagine "assume X is solved" and explore implications.
    Then come back to solve X.
    This is how humans tackle hard problems.
    """

    def test_scenario_assume_and_explore(self):
        """
        Scenario: Assume solution exists and explore implications

        Given a hard sub-problem X
        When I assume X is solved (imagination function)
        Then I can explore what follows
        And determine if solving X is worth it
        Because exploring implications before solving saves effort
        """
        from cortical.cognitive.economy import (
            Thought, ImaginationContext
        )

        def exploratory_thought(ctx) -> str:
            # Hard sub-problem
            with ctx.imagine("assume we have fast search") as imagined:
                # Explore what we could do
                possibilities = imagined.explore([
                    "find patterns in realtime",
                    "answer queries instantly",
                    "enable interactive exploration"
                ])

                if imagined.value_estimate(possibilities) > 0.8:
                    # Worth solving the hard problem
                    return "fast search is worth building"
                else:
                    return "fast search wouldn't help much"

        thought = Thought(exploratory_thought)
        result = thought.run()

        assert "fast search" in result

    def test_scenario_recursive_imagination(self):
        """
        Scenario: Nested imagination (assume A, then assume B)

        Given problem requiring multiple assumptions
        When I nest imagination contexts
        Then I can explore complex hypotheticals
        And trace back which assumptions led where
        Because complex problems have complex dependencies
        """
        from cortical.cognitive.economy import (
            Thought, ImaginationContext
        )

        imagination_trace = []

        def nested_imagination(ctx) -> str:
            with ctx.imagine("assume A is solved") as a:
                imagination_trace.append("assumed A")

                with a.imagine("also assume B is solved") as ab:
                    imagination_trace.append("assumed A and B")

                    # Now what?
                    result = ab.explore(["with A and B, we can do C"])
                    imagination_trace.append(f"explored: {result}")

            return f"trace: {imagination_trace}"

        thought = Thought(nested_imagination)
        result = thought.run()

        assert "assumed A" in str(result)
        assert "assumed A and B" in str(result)


# =============================================================================
# SUMMARY
# =============================================================================

"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  COGNITIVE ECONOMY: A Complete Model                                       │
│                                                                             │
│  TWO MODES:                                                                │
│    IMMEDIATE - "I need this now" - blocking, bounded                       │
│    BACKGROUND - "Hunt for this" - non-blocking, infinite time allowed     │
│                                                                             │
│  PARTIAL ANSWERS:                                                          │
│    Not binary complete/fail                                                │
│    Useful incomplete results                                               │
│    Can improve over time                                                   │
│    Stream as found                                                         │
│                                                                             │
│  ATTENTION MARKET:                                                         │
│    Priority = price                                                        │
│    Resources flow to highest bidder                                        │
│    Dynamic adjustment                                                      │
│    Low priority runs when capacity allows                                  │
│                                                                             │
│  VARIANTS:                                                                 │
│    Multiple answers to same question                                       │
│    Different approaches/temperatures                                       │
│    Select best or synthesize                                               │
│                                                                             │
│  CONCURRENT:                                                               │
│    Think and talk simultaneously                                           │
│    Background research during conversation                                 │
│    Insights surface when relevant                                          │
│                                                                             │
│  META-ORCHESTRATION:                                                       │
│    Model managing models                                                   │
│    Questions own understanding                                             │
│    Allocates by expected value                                             │
│                                                                             │
│  HUMAN-AI SWARM:                                                           │
│    Both contribute                                                         │
│    Human guides AI                                                         │
│    Long-term coherent collaboration                                        │
│    Institutional memory                                                    │
│                                                                             │
│  IMAGINATION:                                                              │
│    Assume solutions exist                                                  │
│    Explore implications                                                    │
│    Recursive nesting                                                       │
│    Trace assumptions                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""
