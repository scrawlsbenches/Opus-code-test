"""
Behavioral test: Semantic Intent Matching

HYPOTHESIS:
Experiences should be findable by semantic similarity of their INTENT,
not just by categorical Context fields.

Current Problem (as identified in agent survey):
- "Implement JWT authentication" should match "Add JWT token verification"
- The system only matches on goal_type="feature_implementation"
- This loses the valuable semantic signal in intent

This test proves:
1. Keywords can be extracted from intent strings
2. Experiences can be found by keyword overlap
3. Semantic matching finds relevant experiences that categorical matching misses
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from llm_orchestration.learning import (
    LearningCycle,
    Context,
    Action,
    Outcome,
    OutcomeType,
    ExperienceType,
)


class TestSemanticIntentMatching:
    """
    Prove that: Intent-based matching finds semantically similar experiences

    This is the foundation for useful learning.
    If this test passes, agents can find "what worked for JWT auth"
    not just "what worked for features".
    """

    @pytest.fixture
    def temp_storage(self):
        """Provide temporary storage for learning system."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_keyword_extraction_from_intent(self, temp_storage):
        """
        HYPOTHESIS: Keywords can be extracted from natural language intent.

        Given: An intent string like "Implement JWT authentication for API"
        When: Keywords are extracted
        Then: We get meaningful terms: ["jwt", "authentication", "api", "implement"]

        This is the foundation - if we can't extract keywords, nothing else works.
        """
        cycle = LearningCycle(temp_storage)

        # Test keyword extraction
        intent = "Implement JWT authentication for the user API"
        keywords = cycle.extract_keywords(intent)

        # Should extract meaningful terms
        assert "jwt" in keywords, "Should extract 'jwt' as keyword"
        assert "authentication" in keywords, "Should extract 'authentication'"
        assert "api" in keywords, "Should extract 'api'"

        # Should NOT include stop words
        assert "for" not in keywords, "Should not include stop word 'for'"
        assert "the" not in keywords, "Should not include stop word 'the'"

        print(f"\n=== KEYWORD EXTRACTION ===")
        print(f"Intent: {intent}")
        print(f"Keywords: {keywords}")

    def test_intent_similarity_calculation(self, temp_storage):
        """
        HYPOTHESIS: Similar intents have measurable keyword overlap.

        Jaccard similarity = |intersection| / |union|

        Given: Two related intents sharing keywords
        When: Similarity is calculated
        Then: Score is positive (> 0.1)

        Given: Two unrelated intents with no shared keywords
        When: Similarity is calculated
        Then: Score is zero or very low
        """
        cycle = LearningCycle(temp_storage)

        # Related intents (share "jwt")
        intent1 = "Implement JWT authentication"
        intent2 = "Add JWT token verification"
        # Keywords: {implement, jwt, authentication} vs {add, jwt, token, verification}
        # Intersection: {jwt}, Union: 6 words → 1/6 = 0.167

        similarity = cycle.intent_similarity(intent1, intent2)
        assert similarity > 0.1, (
            f"Related intents should have similarity > 0.1, got {similarity}"
        )

        # Highly similar intents (share multiple keywords)
        intent_a = "Implement JWT token authentication handler"
        intent_b = "Add JWT authentication token support"
        # Both share: jwt, authentication, token → higher overlap

        similarity_high = cycle.intent_similarity(intent_a, intent_b)
        assert similarity_high > similarity, (
            f"More related intents should have higher similarity. "
            f"Got {similarity_high} vs {similarity}"
        )

        # Completely unrelated intents (no shared keywords)
        intent3 = "Fix database connection pooling"
        similarity_unrelated = cycle.intent_similarity(intent1, intent3)
        assert similarity_unrelated < 0.1, (
            f"Unrelated intents should have similarity < 0.1, got {similarity_unrelated}"
        )

        print(f"\n=== INTENT SIMILARITY (Jaccard on keywords) ===")
        print(f"'{intent1}' vs '{intent2}': {similarity:.3f}")
        print(f"'{intent_a}' vs '{intent_b}': {similarity_high:.3f}")
        print(f"'{intent1}' vs '{intent3}': {similarity_unrelated:.3f}")

    def test_find_experiences_by_intent(self, temp_storage):
        """
        HYPOTHESIS: Experiences can be found by intent similarity.

        Given: Multiple experiences with different contexts but related intents
        When: find_by_intent() is called with a similar intent
        Then: Semantically related experiences are returned

        This is the critical test - can we find "JWT auth" experiences
        regardless of their Context categories?
        """
        cycle = LearningCycle(temp_storage)

        # Create experiences with DIFFERENT contexts but SIMILAR intents
        # Experience 1: JWT auth (context: API, feature)
        context1 = Context(
            goal_type="feature_implementation",
            goal_complexity="moderate",
            domain="api",
        )
        exp1 = cycle.start_experience(
            context=context1,
            intent="Implement JWT authentication for REST API",
            experience_type=ExperienceType.TASK_EXECUTION,
        )
        exp1.add_action(Action("write_code", "Create JWT validator", "src/auth/jwt.py"))
        cycle.complete_experience(exp1, Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="JWT auth working",
        ))

        # Experience 2: JWT token bug (context: Security, bugfix - DIFFERENT!)
        context2 = Context(
            goal_type="bugfix",  # Different!
            goal_complexity="complex",  # Different!
            domain="security",  # Different!
        )
        exp2 = cycle.start_experience(
            context=context2,
            intent="Fix JWT token authentication expiry bug",  # Shares jwt, token, authentication
            experience_type=ExperienceType.TASK_EXECUTION,
        )
        exp2.add_action(Action("write_test", "Test token edge cases", "tests/test_tokens.py"))
        cycle.complete_experience(exp2, Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="Token bug fixed",
        ))

        # Experience 3: Database work (completely unrelated)
        context3 = Context(
            goal_type="feature_implementation",  # Same as exp1!
            goal_complexity="moderate",  # Same as exp1!
            domain="database",
        )
        exp3 = cycle.start_experience(
            context=context3,
            intent="Add connection pooling to database layer",
            experience_type=ExperienceType.TASK_EXECUTION,
        )
        exp3.add_action(Action("write_code", "Pool connections", "src/db/pool.py"))
        cycle.complete_experience(exp3, Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="Pool added",
        ))

        # Now search by intent: "Add JWT token authentication"
        search_intent = "Add JWT token authentication"

        # Using intent-based search
        found = cycle.find_by_intent(search_intent, limit=5)

        # Should find exp1 and exp2 (JWT/token related)
        found_ids = [exp.id for exp in found]
        assert exp1.id in found_ids, (
            "Should find 'Implement JWT authentication' experience"
        )
        assert exp2.id in found_ids, (
            "Should find 'Fix token expiry' experience (semantically related)"
        )

        # Should NOT prioritize exp3 even though Context matches exp1 better
        # (This proves semantic matching works)
        if len(found) > 2:
            # If exp3 appears, it should be last
            assert found[-1].id == exp3.id or exp3.id not in found_ids, (
                "Database experience should not be prioritized over token experiences"
            )

        print(f"\n=== INTENT-BASED SEARCH ===")
        print(f"Search: '{search_intent}'")
        print(f"Found {len(found)} experiences:")
        for exp in found:
            print(f"  - {exp.id}: {exp.intent}")

    def test_semantic_vs_categorical_matching(self, temp_storage):
        """
        HYPOTHESIS: Semantic matching outperforms categorical matching
        for finding relevant experiences.

        This test demonstrates WHY we need semantic matching:
        - Categorical matching finds exp3 (same goal_type/complexity)
        - Semantic matching finds exp1 and exp2 (related intent)

        For an agent working on authentication, exp1/exp2 are more useful
        than exp3 despite exp3 having identical Context categories.
        """
        cycle = LearningCycle(temp_storage)

        # Create same experiences as above
        # Exp1: JWT implementation (feature, moderate, api)
        exp1 = self._create_jwt_experience(cycle, "api")

        # Exp2: Token bug (bugfix, complex, security)
        exp2 = self._create_token_experience(cycle)

        # Exp3: Database (feature, moderate, api) - SAME context as exp1
        exp3 = self._create_db_experience(cycle)

        # Query context matches exp1 and exp3 perfectly
        query_context = Context(
            goal_type="feature_implementation",
            goal_complexity="moderate",
            domain="api",
        )

        # Categorical matching (current behavior)
        categorical_results = cycle.find_similar_context(query_context, limit=5)
        categorical_ids = [exp.id for exp, _ in categorical_results]

        # Semantic matching (new behavior)
        query_intent = "Implement authentication tokens"
        semantic_results = cycle.find_by_intent(query_intent, limit=5)
        semantic_ids = [exp.id for exp in semantic_results]

        print(f"\n=== CATEGORICAL vs SEMANTIC ===")
        print(f"Query context: {query_context.goal_type}, {query_context.domain}")
        print(f"Query intent: '{query_intent}'")
        print(f"\nCategorical results (by context):")
        for exp_id in categorical_ids:
            exp = cycle.get_experience(exp_id)
            print(f"  - {exp.intent}")
        print(f"\nSemantic results (by intent):")
        for exp in semantic_results:
            print(f"  - {exp.intent}")

        # Key assertion: semantic should prioritize auth-related over db
        if exp2.id in semantic_ids:
            exp2_position = semantic_ids.index(exp2.id)
            if exp3.id in semantic_ids:
                exp3_position = semantic_ids.index(exp3.id)
                assert exp2_position < exp3_position, (
                    "Token experience should rank higher than database experience "
                    "when searching for authentication-related tasks"
                )

    def _create_jwt_experience(self, cycle, domain: str):
        """Helper to create a JWT experience."""
        context = Context(
            goal_type="feature_implementation",
            goal_complexity="moderate",
            domain=domain,
        )
        exp = cycle.start_experience(
            context=context,
            intent="Implement JWT authentication",
            experience_type=ExperienceType.TASK_EXECUTION,
        )
        exp.add_action(Action("write_code", "JWT logic", "src/auth.py"))
        cycle.complete_experience(exp, Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="JWT working",
        ))
        return exp

    def _create_token_experience(self, cycle):
        """Helper to create a JWT token bug experience."""
        context = Context(
            goal_type="bugfix",
            goal_complexity="complex",
            domain="security",
        )
        exp = cycle.start_experience(
            context=context,
            intent="Fix JWT token authentication expiry bug",  # Shares jwt, token, authentication
            experience_type=ExperienceType.TASK_EXECUTION,
        )
        exp.add_action(Action("write_test", "Token tests", "tests/auth.py"))
        cycle.complete_experience(exp, Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="Token bug fixed",
        ))
        return exp

    def _create_db_experience(self, cycle):
        """Helper to create a database experience."""
        context = Context(
            goal_type="feature_implementation",
            goal_complexity="moderate",
            domain="api",  # Same domain as JWT!
        )
        exp = cycle.start_experience(
            context=context,
            intent="Add database connection pooling",
            experience_type=ExperienceType.TASK_EXECUTION,
        )
        exp.add_action(Action("write_code", "Pool logic", "src/db.py"))
        cycle.complete_experience(exp, Outcome(
            outcome_type=OutcomeType.SUCCESS,
            description="Pool added",
        ))
        return exp

    def test_combined_context_and_intent_matching(self, temp_storage):
        """
        HYPOTHESIS: Best results come from combining context AND intent.

        The system should use BOTH signals:
        - Context: "Is this a similar situation?"
        - Intent: "Is this a similar task?"

        Combining them gives the best of both worlds.
        """
        cycle = LearningCycle(temp_storage)

        # Create experiences
        exp_jwt_api = self._create_jwt_experience(cycle, "api")
        exp_token_security = self._create_token_experience(cycle)
        exp_db = self._create_db_experience(cycle)

        # Search with both context and intent
        search_context = Context(
            goal_type="feature_implementation",
            goal_complexity="moderate",
            domain="api",
        )
        search_intent = "Add token-based authentication"

        # Combined search
        results = cycle.find_by_context_and_intent(
            context=search_context,
            intent=search_intent,
            context_weight=0.3,
            intent_weight=0.7,
            limit=5,
        )

        # JWT API experience should rank highest (matches both)
        assert len(results) > 0, "Should find at least one experience"
        top_result = results[0]
        assert top_result[0].id == exp_jwt_api.id, (
            "JWT API experience should rank highest - matches both context and intent"
        )

        print(f"\n=== COMBINED MATCHING ===")
        print(f"Context: {search_context.goal_type}, {search_context.domain}")
        print(f"Intent: '{search_intent}'")
        print(f"Results (context_weight={0.3}, intent_weight={0.7}):")
        for exp, score in results:
            print(f"  - {exp.intent} (score: {score:.3f})")
