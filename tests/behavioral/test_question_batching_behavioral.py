"""
Behavioral tests for question batching.

These tests verify realistic scenarios for human-AI question batching,
inspired by the complex-reasoning-workflow.md document Part 4.3
(Before asking the human questions: batch them, explain why, respect their time).

Test scenarios:
- Batch questions by category
- Parse various response formats
- Handle blocking vs non-blocking questions
- Generate markdown batches for async communication
"""

import pytest

from cortical.reasoning.collaboration import (
    QuestionBatcher,
    BatchedQuestion,
)


class TestQuestionCategorization:
    """Test question categorization and batching."""

    def test_categorize_mixed_questions(self):
        """
        Scenario: Add questions of various categories.
        Expected: Questions grouped by category in output.
        """
        batcher = QuestionBatcher()

        # Add questions of different types
        batcher.add_question(
            "Which authentication method should we use?",
            category="design",
            urgency="high"
        )
        batcher.add_question(
            "What's the API endpoint format?",
            category="technical",
            urgency="medium"
        )
        batcher.add_question(
            "Can I proceed with this approach?",
            category="approval",
            urgency="critical",
            blocking=True
        )
        batcher.add_question(
            "What do you mean by 'fast'?",
            category="clarification",
            urgency="high"
        )

        categorized = batcher.categorize_questions()

        assert "design" in categorized
        assert "technical" in categorized
        assert "approval" in categorized
        assert "clarification" in categorized

        # Each category has correct questions
        assert len(categorized["design"]) == 1
        assert len(categorized["approval"]) == 1

    def test_blocking_questions_first(self):
        """
        Scenario: Mix of blocking and non-blocking questions.
        Expected: Blocking questions appear first within category.
        """
        batcher = QuestionBatcher()

        batcher.add_question("Non-blocking question 1", blocking=False)
        batcher.add_question("Blocking question", blocking=True, urgency="critical")
        batcher.add_question("Non-blocking question 2", blocking=False)

        categorized = batcher.categorize_questions()
        general_questions = categorized.get("general", [])

        # Blocking question should be first
        if len(general_questions) > 0:
            assert general_questions[0].blocking is True

    def test_urgency_ordering(self):
        """
        Scenario: Questions with different urgencies.
        Expected: Critical before high before medium before low.
        """
        batcher = QuestionBatcher()

        batcher.add_question("Low urgency", urgency="low")
        batcher.add_question("Critical urgency", urgency="critical")
        batcher.add_question("Medium urgency", urgency="medium")
        batcher.add_question("High urgency", urgency="high")

        categorized = batcher.categorize_questions()
        general_questions = categorized.get("general", [])

        # Check ordering (critical < high < medium < low)
        urgencies = [q.urgency for q in general_questions]
        expected_order = ["critical", "high", "medium", "low"]
        assert urgencies == expected_order


class TestMarkdownBatchGeneration:
    """Test markdown batch generation."""

    def test_generate_empty_batch(self):
        """
        Scenario: No questions to batch.
        Expected: Appropriate message.
        """
        batcher = QuestionBatcher()
        md = batcher.generate_batch()

        assert "No pending questions" in md

    def test_generate_single_question_batch(self):
        """
        Scenario: Single question with context.
        Expected: Well-formatted markdown with context.
        """
        batcher = QuestionBatcher()
        batcher.add_question(
            "What database should we use?",
            context="Need to support high write throughput",
            default="PostgreSQL",
            urgency="high"
        )

        md = batcher.generate_batch()

        assert "Question Request" in md
        assert "What database should we use?" in md
        assert "high write throughput" in md
        assert "PostgreSQL" in md
        assert "Q-" in md  # Question ID present

    def test_generate_multi_category_batch(self):
        """
        Scenario: Questions across categories.
        Expected: Organized sections by category.
        """
        batcher = QuestionBatcher()

        batcher.add_question("Technical Q1?", category="technical")
        batcher.add_question("Design Q1?", category="design")
        batcher.add_question("Approval Q1?", category="approval")

        md = batcher.generate_batch()

        assert "Technical Questions" in md
        assert "Design Decisions" in md
        assert "Approval Required" in md

    def test_blocking_questions_highlighted(self):
        """
        Scenario: Batch with blocking questions.
        Expected: BLOCKING marker and urgency note.
        """
        batcher = QuestionBatcher()

        batcher.add_question(
            "Need immediate decision on X",
            blocking=True,
            urgency="critical"
        )

        md = batcher.generate_batch()

        assert "BLOCKING" in md
        assert "URGENT" in md or "blocking" in md.lower()

    def test_related_questions_linked(self):
        """
        Scenario: Questions that relate to each other.
        Expected: Related questions shown in batch.
        """
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Should we use REST or GraphQL?")
        batcher.add_question(
            "If REST, which version of OpenAPI?",
            related_ids=[q1]
        )

        md = batcher.generate_batch()

        assert "Related to:" in md
        assert q1 in md


class TestResponseParsing:
    """Test parsing of human responses."""

    def test_parse_standard_format(self):
        """
        Scenario: Human responds with Q-NNN: Answer format.
        Expected: All answers correctly matched.
        """
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Question 1?")
        q2 = batcher.add_question("Question 2?")
        q3 = batcher.add_question("Question 3?")

        response = """
        Q-000: Answer to question 1
        Q-001: Answer to question 2
        Q-002: Answer to question 3
        """

        result = batcher.process_responses(response)

        assert len(result['matched']) == 3
        assert result['matched'][q1] == "Answer to question 1"
        assert result['matched'][q2] == "Answer to question 2"
        assert result['matched'][q3] == "Answer to question 3"
        assert len(result['unanswered_questions']) == 0

    def test_parse_numeric_format(self):
        """
        Scenario: Human uses 1:, 2: instead of Q-NNN.
        Expected: Still parsed correctly (1 = Q-000, 2 = Q-001).
        """
        batcher = QuestionBatcher()

        q1 = batcher.add_question("First question?")
        q2 = batcher.add_question("Second question?")

        response = """
        1: First answer
        2: Second answer
        """

        result = batcher.process_responses(response)

        assert len(result['matched']) == 2
        assert result['matched'][q1] == "First answer"
        assert result['matched'][q2] == "Second answer"

    def test_parse_multiline_answer(self):
        """
        Scenario: Question ID on one line, answer on next.
        Expected: Answer captured from next line.
        """
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Complex question?")

        response = """
        Q-000
        This is a detailed multi-word answer
        """

        result = batcher.process_responses(response)

        assert q1 in result['matched']
        assert "detailed multi-word" in result['matched'][q1]

    def test_parse_partial_response(self):
        """
        Scenario: Human only answers some questions.
        Expected: Answered questions matched, others tracked as unanswered.
        """
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Question 1?")
        q2 = batcher.add_question("Question 2?")
        q3 = batcher.add_question("Question 3?")

        response = """
        Q-000: Answered question 1
        Q-002: Skipped 2, answered 3
        """

        result = batcher.process_responses(response)

        assert len(result['matched']) == 2
        assert q1 in result['matched']
        assert q3 in result['matched']
        assert q2 in result['unanswered_questions']

    def test_parse_with_markdown_artifacts(self):
        """
        Scenario: Response includes markdown formatting.
        Expected: Formatting stripped, answers extracted.
        """
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Question?")

        response = """
        # Answers

        ```
        Q-000: Use approach A
        ```
        """

        result = batcher.process_responses(response)

        assert q1 in result['matched']
        assert "approach A" in result['matched'][q1]

    def test_unparsed_lines_tracked(self):
        """
        Scenario: Response includes unparseable content.
        Expected: Unparseable lines tracked for review.
        """
        batcher = QuestionBatcher()

        batcher.add_question("Question?")

        response = """
        Q-000: Valid answer
        This line doesn't match any format
        Random commentary here
        """

        result = batcher.process_responses(response)

        assert len(result['unparsed_lines']) >= 1


class TestQuestionLifecycle:
    """Test question lifecycle management."""

    def test_mark_answered_manually(self):
        """
        Scenario: Mark question as answered programmatically.
        Expected: Question tracked as answered.
        """
        batcher = QuestionBatcher()

        q_id = batcher.add_question("To be answered later")

        # Initially unanswered
        assert q_id in [q.id for q in batcher.get_unanswered_questions()]

        # Mark as answered
        success = batcher.mark_answered(q_id, "The answer is 42")

        assert success
        assert q_id not in [q.id for q in batcher.get_unanswered_questions()]

        # Retrieve the question
        question = batcher.get_question(q_id)
        assert question.answered
        assert question.response == "The answer is 42"

    def test_get_pending_blockers(self):
        """
        Scenario: Mix of blocking and non-blocking questions.
        Expected: Only unanswered blocking questions returned.
        """
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Blocker 1", blocking=True)
        q2 = batcher.add_question("Non-blocker", blocking=False)
        q3 = batcher.add_question("Blocker 2", blocking=True)

        # All blockers initially pending
        pending = batcher.get_pending_blockers()
        assert len(pending) == 2

        # Answer one blocker
        batcher.mark_answered(q1, "Resolved")

        pending = batcher.get_pending_blockers()
        assert len(pending) == 1
        assert pending[0].id == q3

    def test_answered_questions_excluded_from_batch(self):
        """
        Scenario: Some questions already answered.
        Expected: Only unanswered questions in generated batch.
        """
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Already answered")
        q2 = batcher.add_question("Still pending")

        batcher.mark_answered(q1, "Done")

        md = batcher.generate_batch()

        assert "Still pending" in md
        assert "Already answered" not in md


class TestQuestionWithDefaults:
    """Test questions with default values."""

    def test_default_shown_in_batch(self):
        """
        Scenario: Question has a suggested default.
        Expected: Default shown in batch for human reference.
        """
        batcher = QuestionBatcher()

        batcher.add_question(
            "What timeout value?",
            default="30 seconds",
            context="For API requests"
        )

        md = batcher.generate_batch()

        assert "30 seconds" in md
        assert "Default" in md or "default" in md

    def test_multiple_questions_with_defaults(self):
        """
        Scenario: Multiple questions with different defaults.
        Expected: Each default shown correctly.
        """
        batcher = QuestionBatcher()

        batcher.add_question("Cache TTL?", default="3600")
        batcher.add_question("Max connections?", default="100")
        batcher.add_question("Retry count?", default="3")

        md = batcher.generate_batch()

        assert "3600" in md
        assert "100" in md
        assert "3" in md


class TestRealisticScenarios:
    """Test realistic question batching scenarios."""

    def test_api_design_questions(self):
        """
        Scenario: Designing an API requires multiple decisions.
        Expected: All questions batched coherently.
        """
        batcher = QuestionBatcher()

        q1 = batcher.add_question(
            "REST or GraphQL?",
            category="design",
            urgency="high",
            blocking=True,
            context="Affects all client integrations"
        )

        q2 = batcher.add_question(
            "What authentication method?",
            category="design",
            urgency="high",
            related_ids=[q1],
            context="Depends on API style choice"
        )

        q3 = batcher.add_question(
            "Preferred HTTP library?",
            category="technical",
            urgency="medium",
            default="requests"
        )

        q4 = batcher.add_question(
            "Do you approve starting implementation?",
            category="approval",
            urgency="critical",
            blocking=True
        )

        md = batcher.generate_batch()

        # All questions present
        assert "REST or GraphQL" in md
        assert "authentication" in md
        assert "HTTP library" in md
        assert "approve" in md

        # Blocking highlighted
        assert "BLOCKING" in md

        # Response instructions included
        assert "How to Respond" in md

    def test_bug_investigation_questions(self):
        """
        Scenario: Investigating a bug requires clarifying questions.
        Expected: Questions organized for efficient investigation.
        """
        batcher = QuestionBatcher()

        batcher.add_question(
            "When did the bug first appear?",
            category="clarification",
            urgency="high"
        )
        batcher.add_question(
            "Is it reproducible?",
            category="clarification",
            urgency="high"
        )
        batcher.add_question(
            "What environment: dev, staging, prod?",
            category="clarification",
            urgency="high"
        )
        batcher.add_question(
            "Any recent deployments?",
            category="clarification",
            urgency="medium"
        )

        md = batcher.generate_batch()

        # All clarification questions grouped
        assert "Clarification" in md
        # Count actual question entries (not examples in "How to Respond")
        assert "Q-000" in md
        assert "Q-003" in md  # Fourth question

    def test_human_response_workflow(self):
        """
        Scenario: Full workflow from batch to parsed responses.
        Expected: Complete round-trip works.
        """
        batcher = QuestionBatcher()

        # AI batches questions
        batcher.add_question("Use TypeScript?", default="yes")
        batcher.add_question("Add ESLint?", default="yes")
        batcher.add_question("Preferred test framework?")

        # Generate batch for human
        batch = batcher.generate_batch()
        assert "TypeScript" in batch
        assert "ESLint" in batch
        assert "test framework" in batch

        # Human responds
        response = """
        Q-000: Yes, TypeScript please
        Q-001: Yes, with Airbnb config
        Q-002: Jest with React Testing Library
        """

        result = batcher.process_responses(response)

        # All answered
        assert len(result['matched']) == 3
        assert len(result['unanswered_questions']) == 0

        # No more questions to batch
        remaining = batcher.generate_batch()
        assert "No pending questions" in remaining
