"""
Comprehensive unit tests for QuestionBatcher class.

Tests all features:
- Enhanced add_question() with category, priority, blocking, related_ids
- categorize_questions() grouping and sorting
- generate_batch() formatted markdown output
- process_responses() parsing structured responses
- get_pending_blockers() identifying blocking questions
"""

import pytest
from cortical.reasoning.collaboration import QuestionBatcher, BatchedQuestion


class TestBatchedQuestion:
    """Tests for BatchedQuestion dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal fields."""
        q = BatchedQuestion(id="Q-001", question="What is this?")

        assert q.id == "Q-001"
        assert q.question == "What is this?"
        assert q.context == ""
        assert q.default is None
        assert q.urgency == "medium"
        assert q.category == "general"
        assert q.blocking is False
        assert q.related_ids == []
        assert q.answered is False
        assert q.response is None

    def test_create_full(self):
        """Test creating with all fields."""
        q = BatchedQuestion(
            id="Q-001",
            question="What API version?",
            context="Building integration",
            default="v2",
            urgency="critical",
            category="technical",
            blocking=True,
            related_ids=["Q-002", "Q-003"],
            answered=True,
            response="Use v3"
        )

        assert q.id == "Q-001"
        assert q.question == "What API version?"
        assert q.context == "Building integration"
        assert q.default == "v2"
        assert q.urgency == "critical"
        assert q.category == "technical"
        assert q.blocking is True
        assert q.related_ids == ["Q-002", "Q-003"]
        assert q.answered is True
        assert q.response == "Use v3"


class TestQuestionBatcherBasics:
    """Basic tests for QuestionBatcher initialization and simple operations."""

    def test_create(self):
        """Test creating batcher."""
        batcher = QuestionBatcher()
        assert batcher is not None
        assert len(batcher.get_all_questions()) == 0

    def test_add_question_minimal(self):
        """Test adding question with minimal parameters."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question("What is the API key?")

        assert q_id == "Q-000"
        assert len(batcher.get_all_questions()) == 1

        question = batcher.get_question(q_id)
        assert question is not None
        assert question.question == "What is the API key?"
        assert question.category == "general"
        assert question.urgency == "medium"
        assert not question.blocking

    def test_add_question_with_all_parameters(self):
        """Test adding question with all parameters."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question(
            question="Should we use Redis?",
            context="Deciding on caching strategy",
            default="Yes",
            urgency="high",
            category="design",
            blocking=True,
            related_ids=["Q-001"]
        )

        question = batcher.get_question(q_id)
        assert question.question == "Should we use Redis?"
        assert question.context == "Deciding on caching strategy"
        assert question.default == "Yes"
        assert question.urgency == "high"
        assert question.category == "design"
        assert question.blocking is True
        assert question.related_ids == ["Q-001"]

    def test_add_multiple_questions(self):
        """Test adding multiple questions generates unique IDs."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("Question 1")
        q2 = batcher.add_question("Question 2")
        q3 = batcher.add_question("Question 3")

        assert q1 == "Q-000"
        assert q2 == "Q-001"
        assert q3 == "Q-002"
        assert len(batcher.get_all_questions()) == 3

    def test_get_question_exists(self):
        """Test getting an existing question."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question("Test question")

        question = batcher.get_question(q_id)
        assert question is not None
        assert question.id == q_id

    def test_get_question_nonexistent(self):
        """Test getting non-existent question returns None."""
        batcher = QuestionBatcher()
        question = batcher.get_question("Q-999")
        assert question is None

    def test_get_all_questions(self):
        """Test getting all questions."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1")
        batcher.add_question("Q2")
        batcher.add_question("Q3")

        all_questions = batcher.get_all_questions()
        assert len(all_questions) == 3

    def test_get_unanswered_questions(self):
        """Test getting only unanswered questions."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("Q1")
        q2 = batcher.add_question("Q2")
        q3 = batcher.add_question("Q3")

        # Mark one as answered
        batcher.mark_answered(q2, "Answer 2")

        unanswered = batcher.get_unanswered_questions()
        assert len(unanswered) == 2
        assert all(q.id in [q1, q3] for q in unanswered)


class TestQuestionCategorizaton:
    """Tests for categorize_questions() method."""

    def test_categorize_empty(self):
        """Test categorizing with no questions."""
        batcher = QuestionBatcher()
        categorized = batcher.categorize_questions()
        assert categorized == {}

    def test_categorize_single_category(self):
        """Test categorizing questions in single category."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1", category="technical")
        batcher.add_question("Q2", category="technical")

        categorized = batcher.categorize_questions()
        assert "technical" in categorized
        assert len(categorized["technical"]) == 2

    def test_categorize_multiple_categories(self):
        """Test categorizing questions across categories."""
        batcher = QuestionBatcher()
        batcher.add_question("Tech Q", category="technical")
        batcher.add_question("Design Q", category="design")
        batcher.add_question("Approval Q", category="approval")

        categorized = batcher.categorize_questions()
        assert len(categorized) == 3
        assert "technical" in categorized
        assert "design" in categorized
        assert "approval" in categorized

    def test_categorize_sort_by_urgency(self):
        """Test questions sorted by urgency within category."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("Low", category="tech", urgency="low")
        q2 = batcher.add_question("Critical", category="tech", urgency="critical")
        q3 = batcher.add_question("Medium", category="tech", urgency="medium")
        q4 = batcher.add_question("High", category="tech", urgency="high")

        categorized = batcher.categorize_questions()
        tech_questions = categorized["tech"]

        # Should be sorted: critical, high, medium, low
        assert tech_questions[0].urgency == "critical"
        assert tech_questions[1].urgency == "high"
        assert tech_questions[2].urgency == "medium"
        assert tech_questions[3].urgency == "low"

    def test_categorize_blocking_first(self):
        """Test blocking questions appear first regardless of urgency."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("High", urgency="high", blocking=False)
        q2 = batcher.add_question("Low Blocking", urgency="low", blocking=True)
        q3 = batcher.add_question("Critical", urgency="critical", blocking=False)

        categorized = batcher.categorize_questions()
        general = categorized["general"]

        # Blocking question should be first, even with low urgency
        assert general[0].blocking is True
        assert general[0].urgency == "low"

    def test_categorize_ignores_answered(self):
        """Test categorize only includes unanswered questions."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("Q1", category="tech")
        q2 = batcher.add_question("Q2", category="tech")

        batcher.mark_answered(q1, "Answer")

        categorized = batcher.categorize_questions()
        assert len(categorized["tech"]) == 1
        assert categorized["tech"][0].id == q2


class TestGenerateBatch:
    """Tests for generate_batch() markdown generation."""

    def test_generate_empty(self):
        """Test generating batch with no questions."""
        batcher = QuestionBatcher()
        md = batcher.generate_batch()
        assert "No pending questions" in md

    def test_generate_single_question(self):
        """Test generating batch with single question."""
        batcher = QuestionBatcher()
        batcher.add_question("What is the config file?")

        md = batcher.generate_batch()
        assert "## Question Request" in md
        assert "1 question(s)" in md
        assert "What is the config file?" in md

    def test_generate_with_category_headers(self):
        """Test batch includes category headers."""
        batcher = QuestionBatcher()
        batcher.add_question("Tech Q", category="technical")
        batcher.add_question("Design Q", category="design")

        md = batcher.generate_batch()
        assert "### Technical Questions" in md
        assert "### Design Decisions" in md

    def test_generate_with_context(self):
        """Test batch includes context when provided."""
        batcher = QuestionBatcher()
        batcher.add_question(
            "What API version?",
            context="Building integration with external service"
        )

        md = batcher.generate_batch()
        assert "*Context:* Building integration with external service" in md

    def test_generate_with_default(self):
        """Test batch includes default when provided."""
        batcher = QuestionBatcher()
        batcher.add_question("Use caching?", default="Yes")

        md = batcher.generate_batch()
        assert "*Default if no response:* `Yes`" in md

    def test_generate_with_related_ids(self):
        """Test batch includes related question IDs."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("Q1")
        q2 = batcher.add_question("Q2", related_ids=[q1])

        md = batcher.generate_batch()
        assert "*Related to:* Q-000" in md

    def test_generate_blocking_marker(self):
        """Test batch marks blocking questions."""
        batcher = QuestionBatcher()
        batcher.add_question("Blocking Q", blocking=True)
        batcher.add_question("Normal Q", blocking=False)

        md = batcher.generate_batch()
        assert "🔴 **[BLOCKING]**" in md

    def test_generate_blocking_warning(self):
        """Test batch includes urgent warning for blocking questions."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1", blocking=True)
        batcher.add_question("Q2", blocking=True)

        md = batcher.generate_batch()
        assert "**URGENT:** 2 blocking question(s)" in md
        assert "work cannot proceed until answered" in md

    def test_generate_urgency_markers(self):
        """Test batch includes urgency markers."""
        batcher = QuestionBatcher()
        batcher.add_question("Critical", urgency="critical")
        batcher.add_question("High", urgency="high")
        batcher.add_question("Medium", urgency="medium")

        md = batcher.generate_batch()
        # Critical and high get markers
        assert "⚠️" in md  # critical marker
        assert "⬆️" in md  # high marker

    def test_generate_includes_response_instructions(self):
        """Test batch includes instructions for responding."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1")

        md = batcher.generate_batch()
        assert "### How to Respond" in md
        assert "Q-001: Your answer here" in md

    def test_generate_ignores_answered_questions(self):
        """Test batch only shows unanswered questions."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("First question text here")
        q2 = batcher.add_question("Second question text here")

        batcher.mark_answered(q1, "Answer 1")

        md = batcher.generate_batch()
        # Should show 1 question, not 2
        assert "1 question(s)" in md
        # The answered question's text shouldn't appear (Q-000)
        assert "First question text here" not in md
        # But the unanswered question should appear
        assert "Second question text here" in md


class TestProcessResponses:
    """Tests for process_responses() parsing."""

    def test_process_empty_response(self):
        """Test processing empty response."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1")

        result = batcher.process_responses("")
        assert result['matched'] == {}
        assert len(result['unanswered_questions']) == 1

    def test_process_qid_colon_format(self):
        """Test parsing 'Q-001: Answer' format."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question("What is X?")

        response = "Q-000: X is Y"
        result = batcher.process_responses(response)

        assert q_id in result['matched']
        assert result['matched'][q_id] == "X is Y"
        assert len(result['unanswered_questions']) == 0

    def test_process_numeric_colon_format(self):
        """Test parsing '1: Answer' format (maps to Q-000, first question)."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("Q1")
        q2 = batcher.add_question("Q2")

        response = """
        1: Answer to first
        2: Answer to second
        """
        result = batcher.process_responses(response)

        # Numeric format is 1-indexed: 1->Q-000, 2->Q-001
        assert "Q-000" in result['matched']
        assert "Q-001" in result['matched']
        assert result['matched']['Q-000'] == "Answer to first"
        assert result['matched']['Q-001'] == "Answer to second"

    def test_process_multiline_response(self):
        """Test parsing responses across multiple lines."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1")
        batcher.add_question("Q2")

        response = """
        Q-000: First answer
        Q-001: Second answer
        """
        result = batcher.process_responses(response)

        assert len(result['matched']) == 2
        assert result['matched']['Q-000'] == "First answer"
        assert result['matched']['Q-001'] == "Second answer"

    def test_process_with_markdown_artifacts(self):
        """Test parsing ignores markdown artifacts."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1")

        response = """
        ```
        Q-000: The answer
        ```
        ### Some header
        """
        result = batcher.process_responses(response)

        assert "Q-000" in result['matched']
        assert result['matched']['Q-000'] == "The answer"

    def test_process_partial_responses(self):
        """Test handling partial responses (some questions unanswered)."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("Q1")
        q2 = batcher.add_question("Q2")
        q3 = batcher.add_question("Q3")

        response = """
        Q-000: Answer 1
        Q-002: Answer 3
        """
        result = batcher.process_responses(response)

        assert len(result['matched']) == 2
        assert "Q-001" in result['unanswered_questions']
        assert "Q-001" not in result['matched']

    def test_process_invalid_question_id(self):
        """Test responses to non-existent questions are unparsed."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1")

        response = "Q-999: This ID doesn't exist"
        result = batcher.process_responses(response)

        assert len(result['matched']) == 0
        assert "Q-999: This ID doesn't exist" in result['unparsed_lines']

    def test_process_updates_question_state(self):
        """Test processing updates question answered state."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question("Q1")

        response = "Q-000: The answer"
        batcher.process_responses(response)

        question = batcher.get_question(q_id)
        assert question.answered is True
        assert question.response == "The answer"

    def test_process_unparsed_lines(self):
        """Test unparsed lines are tracked."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1")

        response = """
        Q-000: Valid answer
        This is random text that can't be parsed
        Another unparsable line
        """
        result = batcher.process_responses(response)

        assert len(result['unparsed_lines']) >= 1
        # Should include the unparsable text (excluding empty lines)


class TestPendingBlockers:
    """Tests for get_pending_blockers() method."""

    def test_no_blockers(self):
        """Test when no blocking questions exist."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1", blocking=False)
        batcher.add_question("Q2", blocking=False)

        blockers = batcher.get_pending_blockers()
        assert len(blockers) == 0

    def test_with_blockers(self):
        """Test identifying blocking questions."""
        batcher = QuestionBatcher()
        batcher.add_question("Q1", blocking=True)
        batcher.add_question("Q2", blocking=False)
        batcher.add_question("Q3", blocking=True)

        blockers = batcher.get_pending_blockers()
        assert len(blockers) == 2
        assert all(b.blocking for b in blockers)

    def test_blockers_excludes_answered(self):
        """Test pending blockers excludes answered questions."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("Q1", blocking=True)
        q2 = batcher.add_question("Q2", blocking=True)

        batcher.mark_answered(q1, "Answer")

        blockers = batcher.get_pending_blockers()
        assert len(blockers) == 1
        assert blockers[0].id == q2


class TestMarkAnswered:
    """Tests for mark_answered() method."""

    def test_mark_existing_question(self):
        """Test marking existing question as answered."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question("Q1")

        result = batcher.mark_answered(q_id, "The answer")

        assert result is True
        question = batcher.get_question(q_id)
        assert question.answered is True
        assert question.response == "The answer"

    def test_mark_nonexistent_question(self):
        """Test marking non-existent question returns False."""
        batcher = QuestionBatcher()
        result = batcher.mark_answered("Q-999", "Answer")
        assert result is False

    def test_mark_updates_unanswered_list(self):
        """Test marking answered removes from unanswered list."""
        batcher = QuestionBatcher()
        q_id = batcher.add_question("Q1")

        assert len(batcher.get_unanswered_questions()) == 1

        batcher.mark_answered(q_id, "Answer")

        assert len(batcher.get_unanswered_questions()) == 0


class TestQuestionBatcherIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow_single_question(self):
        """Test complete workflow: add -> batch -> respond -> verify."""
        batcher = QuestionBatcher()

        # Add question
        q_id = batcher.add_question(
            "Should we use Redis for caching?",
            context="Designing cache layer",
            default="Yes",
            urgency="high",
            category="design"
        )

        # Generate batch
        md = batcher.generate_batch()
        assert "Should we use Redis" in md
        assert "Design Decisions" in md

        # Process response
        response = "Q-000: Yes, use Redis with 1 hour TTL"
        result = batcher.process_responses(response)

        # Verify
        assert q_id in result['matched']
        assert len(result['unanswered_questions']) == 0

        question = batcher.get_question(q_id)
        assert question.answered is True
        assert "Redis with 1 hour TTL" in question.response

    def test_full_workflow_multiple_categories(self):
        """Test workflow with multiple question categories."""
        batcher = QuestionBatcher()

        # Add questions in different categories
        q1 = batcher.add_question("What API key?", category="technical", urgency="critical")
        q2 = batcher.add_question("Approve deployment?", category="approval", blocking=True)
        q3 = batcher.add_question("Use microservices?", category="design", urgency="high")

        # Generate batch
        md = batcher.generate_batch()
        assert "Technical Questions" in md
        assert "Approval Required" in md
        assert "Design Decisions" in md
        assert "**URGENT:** 1 blocking question" in md

        # Process response
        response = """
        Q-000: Use the production key from AWS Secrets Manager
        Q-001: Approved for staging, hold on production
        Q-002: Yes, start with 3 services
        """
        result = batcher.process_responses(response)

        # Verify all answered
        assert len(result['matched']) == 3
        assert len(result['unanswered_questions']) == 0
        assert len(batcher.get_pending_blockers()) == 0

    def test_workflow_with_partial_response(self):
        """Test workflow when human only answers some questions."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Q1", blocking=True)
        q2 = batcher.add_question("Q2", blocking=False)
        q3 = batcher.add_question("Q3", blocking=False)

        # Human only answers the blocker
        response = "Q-000: Unblock with this answer"
        result = batcher.process_responses(response)

        # Verify partial completion
        assert len(result['matched']) == 1
        assert len(result['unanswered_questions']) == 2
        assert len(batcher.get_pending_blockers()) == 0  # Blocker is answered

    def test_workflow_with_related_questions(self):
        """Test workflow with question dependencies."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Use authentication?", category="design")
        q2 = batcher.add_question(
            "Which auth method?",
            category="technical",
            related_ids=[q1]
        )
        q3 = batcher.add_question(
            "Store tokens where?",
            category="technical",
            related_ids=[q1, q2]
        )

        # Generate batch shows relationships
        md = batcher.generate_batch()
        assert "Related to:* Q-000" in md
        assert "Related to:* Q-000, Q-001" in md

        # Answer all
        response = """
        Q-000: Yes, use OAuth2
        Q-001: OAuth2 with Google and GitHub providers
        Q-002: Store in Redis with 24h expiry
        """
        result = batcher.process_responses(response)

        assert len(result['matched']) == 3
        assert all(batcher.get_question(qid).answered for qid in [q1, q2, q3])


class TestQuestionBatcherBehavioral:
    """Behavioral tests for real-world usage patterns."""

    def test_blocking_questions_appear_first_in_batch(self):
        """Test blocking questions always appear before non-blocking."""
        batcher = QuestionBatcher()

        # Add in mixed order
        batcher.add_question("Low priority", urgency="low", blocking=False)
        batcher.add_question("Critical blocker", urgency="critical", blocking=True)
        batcher.add_question("Medium normal", urgency="medium", blocking=False)
        batcher.add_question("Low blocker", urgency="low", blocking=True)

        categorized = batcher.categorize_questions()
        general = categorized["general"]

        # First two should be blockers (regardless of urgency)
        assert general[0].blocking is True
        assert general[1].blocking is True
        assert general[2].blocking is False
        assert general[3].blocking is False

        # Within blockers, should be sorted by urgency
        assert general[0].urgency == "critical"
        assert general[1].urgency == "low"

    def test_batch_remains_useful_after_partial_answers(self):
        """Test regenerating batch after partial responses."""
        batcher = QuestionBatcher()

        q1 = batcher.add_question("Q1")
        q2 = batcher.add_question("Q2")
        q3 = batcher.add_question("Q3")

        # First batch
        md1 = batcher.generate_batch()
        assert "3 question(s)" in md1

        # Answer one
        batcher.mark_answered(q1, "Answer 1")

        # Second batch should only show unanswered
        md2 = batcher.generate_batch()
        assert "2 question(s)" in md2
        assert q1 not in md2 or "Answer 1" in md2  # Either excluded or shows answer

    def test_complex_response_parsing(self):
        """Test parsing various real-world response formats."""
        batcher = QuestionBatcher()
        q1 = batcher.add_question("Q1")
        q2 = batcher.add_question("Q2")
        q3 = batcher.add_question("Q3")

        # Messy but valid response
        response = """
        Here are my answers:

        Q-000: Yes, proceed with this approach

        For the second question:
        Q-001: Use PostgreSQL instead of MySQL

        3: The third answer goes here
        """
        result = batcher.process_responses(response)

        # Should successfully parse all three despite messiness
        assert len(result['matched']) == 3
        assert "Yes, proceed" in result['matched']['Q-000']
        assert "PostgreSQL" in result['matched']['Q-001']
        assert "third answer" in result['matched']['Q-002']
