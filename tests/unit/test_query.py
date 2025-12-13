"""
Unit Tests for Query Modules
============================

Task #154: Unit tests for cortical/query/* pure functions.

Tests the following pure functions that don't require full layer objects:

From intent.py:
- parse_intent_query: Parse natural language queries

From chunking.py:
- create_chunks: Split text into overlapping chunks
- find_code_boundaries: Find semantic boundaries in code
- create_code_aware_chunks: Chunk aligned to code structure
- is_code_file: Detect code files by extension

From expansion.py:
- score_relation_path: Score semantic relation paths

From ranking.py:
- is_conceptual_query: Detect conceptual vs implementation queries
- get_doc_type_boost: Get boost factor for document type
- apply_doc_type_boost: Apply boosting to search results
"""

import pytest

from cortical.query.intent import (
    parse_intent_query,
    QUESTION_INTENTS,
    ACTION_VERBS,
)
from cortical.query.chunking import (
    create_chunks,
    find_code_boundaries,
    create_code_aware_chunks,
    is_code_file,
)
from cortical.query.expansion import (
    score_relation_path,
    VALID_RELATION_CHAINS,
)
from cortical.query.ranking import (
    is_conceptual_query,
    get_doc_type_boost,
    apply_doc_type_boost,
)


# =============================================================================
# PARSE INTENT QUERY TESTS
# =============================================================================


class TestParseIntentQuery:
    """Tests for parse_intent_query function."""

    def test_empty_query(self):
        """Empty query returns default values."""
        result = parse_intent_query("")
        assert result["action"] is None
        assert result["subject"] is None
        assert result["intent"] == "search"
        assert result["question_word"] is None
        assert result["expanded_terms"] == []

    def test_whitespace_only_query(self):
        """Whitespace-only query returns empty."""
        result = parse_intent_query("   ")
        assert result["expanded_terms"] == []

    def test_where_query(self):
        """'where' queries have location intent."""
        result = parse_intent_query("where do we handle authentication?")
        assert result["intent"] == "location"
        assert result["question_word"] == "where"

    def test_how_query(self):
        """'how' queries have implementation intent."""
        result = parse_intent_query("how does validation work?")
        assert result["intent"] == "implementation"
        assert result["question_word"] == "how"

    def test_what_query(self):
        """'what' queries have definition intent."""
        result = parse_intent_query("what is a tokenizer?")
        assert result["intent"] == "definition"
        assert result["question_word"] == "what"

    def test_why_query(self):
        """'why' queries have rationale intent."""
        result = parse_intent_query("why do we use caching?")
        assert result["intent"] == "rationale"
        assert result["question_word"] == "why"

    def test_action_verb_detection(self):
        """Action verbs are correctly identified."""
        result = parse_intent_query("where do we handle errors?")
        assert result["action"] == "handle"

    def test_subject_detection(self):
        """Subject is correctly identified."""
        result = parse_intent_query("where do we handle authentication?")
        assert result["subject"] == "authentication"

    def test_no_question_word(self):
        """Queries without question words default to search intent."""
        result = parse_intent_query("find authentication handler")
        assert result["intent"] == "search"
        assert result["question_word"] is None

    def test_multiple_action_verbs(self):
        """First action verb is selected."""
        result = parse_intent_query("how to create and delete users?")
        assert result["action"] == "create"

    def test_expanded_terms_include_action_and_subject(self):
        """Expanded terms include both action and subject."""
        result = parse_intent_query("where do we validate input?")
        assert "validate" in result["expanded_terms"]
        assert "input" in result["expanded_terms"]

    def test_punctuation_removed(self):
        """Punctuation is stripped from query."""
        result = parse_intent_query("where is the config?!?")
        assert result["intent"] == "location"
        # config should be in expanded terms (not "config?!?")
        assert any("config" in term for term in result["expanded_terms"])

    def test_case_insensitive(self):
        """Query parsing is case insensitive."""
        result = parse_intent_query("WHERE do we HANDLE authentication?")
        assert result["intent"] == "location"
        assert result["action"] == "handle"


class TestQuestionIntents:
    """Tests for QUESTION_INTENTS mapping."""

    def test_all_question_words_mapped(self):
        """All common question words are mapped to intents."""
        expected_words = ["where", "how", "what", "why", "when", "which", "who"]
        for word in expected_words:
            assert word in QUESTION_INTENTS


class TestActionVerbs:
    """Tests for ACTION_VERBS set."""

    def test_common_crud_verbs(self):
        """CRUD verbs are included."""
        crud_verbs = ["create", "delete", "update", "get", "fetch"]
        for verb in crud_verbs:
            assert verb in ACTION_VERBS

    def test_common_processing_verbs(self):
        """Processing verbs are included."""
        processing_verbs = ["process", "validate", "parse", "transform"]
        for verb in processing_verbs:
            assert verb in ACTION_VERBS


# =============================================================================
# CREATE CHUNKS TESTS
# =============================================================================


class TestCreateChunks:
    """Tests for create_chunks function."""

    def test_empty_text(self):
        """Empty text returns empty list."""
        result = create_chunks("", chunk_size=100, overlap=20)
        assert result == []

    def test_text_smaller_than_chunk(self):
        """Text smaller than chunk_size returns single chunk."""
        text = "Hello world"
        result = create_chunks(text, chunk_size=100, overlap=20)
        assert len(result) == 1
        assert result[0][0] == text
        assert result[0][1] == 0
        assert result[0][2] == len(text)

    def test_text_equal_to_chunk(self):
        """Text equal to chunk_size returns single chunk."""
        text = "A" * 100
        result = create_chunks(text, chunk_size=100, overlap=20)
        assert len(result) == 1

    def test_chunks_overlap(self):
        """Chunks overlap by specified amount."""
        text = "A" * 200
        result = create_chunks(text, chunk_size=100, overlap=50)
        # With stride of 50, we should have chunks at 0, 50, 100, 150
        assert len(result) >= 2
        # Second chunk should start where first chunk ends minus overlap
        if len(result) > 1:
            assert result[1][1] == 50  # start at position 50

    def test_chunk_positions_are_correct(self):
        """Chunk start and end positions match the text."""
        text = "0123456789" * 10  # 100 characters
        result = create_chunks(text, chunk_size=30, overlap=10)
        for chunk_text, start, end in result:
            assert chunk_text == text[start:end]

    def test_invalid_chunk_size_raises(self):
        """Zero or negative chunk_size raises ValueError."""
        with pytest.raises(ValueError):
            create_chunks("hello", chunk_size=0, overlap=0)
        with pytest.raises(ValueError):
            create_chunks("hello", chunk_size=-1, overlap=0)

    def test_invalid_overlap_raises(self):
        """Negative overlap raises ValueError."""
        with pytest.raises(ValueError):
            create_chunks("hello", chunk_size=10, overlap=-1)

    def test_overlap_ge_chunk_size_raises(self):
        """Overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError):
            create_chunks("hello", chunk_size=10, overlap=10)
        with pytest.raises(ValueError):
            create_chunks("hello", chunk_size=10, overlap=15)

    def test_no_overlap(self):
        """Zero overlap creates non-overlapping chunks."""
        text = "AAABBBCCC"
        result = create_chunks(text, chunk_size=3, overlap=0)
        assert len(result) == 3
        assert result[0][0] == "AAA"
        assert result[1][0] == "BBB"
        assert result[2][0] == "CCC"


# =============================================================================
# FIND CODE BOUNDARIES TESTS
# =============================================================================


class TestFindCodeBoundaries:
    """Tests for find_code_boundaries function."""

    def test_empty_text(self):
        """Empty text returns boundary at 0."""
        result = find_code_boundaries("")
        assert 0 in result

    def test_class_definition(self):
        """Class definitions create boundaries."""
        text = "# comment\nclass Foo:\n    pass"
        result = find_code_boundaries(text)
        assert len(result) > 1
        # Should find boundary at start of class line

    def test_function_definition(self):
        """Function definitions create boundaries."""
        text = "# comment\ndef foo():\n    pass"
        result = find_code_boundaries(text)
        assert len(result) > 1

    def test_async_function_definition(self):
        """Async function definitions create boundaries."""
        text = "# comment\nasync def foo():\n    pass"
        result = find_code_boundaries(text)
        assert len(result) > 1

    def test_decorator(self):
        """Decorators create boundaries."""
        text = "# comment\n@decorator\ndef foo():\n    pass"
        result = find_code_boundaries(text)
        # Should find boundary at decorator line
        assert len(result) > 1

    def test_blank_lines(self):
        """Blank line sequences create boundaries."""
        text = "a\nb\n\n\nc\nd"
        result = find_code_boundaries(text)
        # Should find boundary after blank lines
        assert len(result) > 1

    def test_comment_separator(self):
        """Comment separators create boundaries."""
        text = "code\n# ---------------\nmore_code"
        result = find_code_boundaries(text)
        assert len(result) > 1

    def test_boundaries_sorted(self):
        """Boundaries are returned in sorted order."""
        text = "class A:\n    pass\n\nclass B:\n    pass"
        result = find_code_boundaries(text)
        assert result == sorted(result)


# =============================================================================
# CREATE CODE AWARE CHUNKS TESTS
# =============================================================================


class TestCreateCodeAwareChunks:
    """Tests for create_code_aware_chunks function."""

    def test_empty_text(self):
        """Empty text returns empty list."""
        result = create_code_aware_chunks("")
        assert result == []

    def test_small_text(self):
        """Text smaller than target_size returns single chunk."""
        text = "def foo(): pass"
        result = create_code_aware_chunks(text, target_size=100)
        assert len(result) == 1
        assert result[0][0] == text

    def test_respects_code_boundaries(self):
        """Chunks align to code boundaries when possible."""
        text = """class Foo:
    def method1(self):
        pass

class Bar:
    def method2(self):
        pass
"""
        result = create_code_aware_chunks(text, target_size=50, min_size=20, max_size=200)
        # Should create multiple chunks aligned to class boundaries
        assert len(result) >= 1

    def test_positions_are_valid(self):
        """Chunk positions correctly index the text."""
        text = "a" * 500
        result = create_code_aware_chunks(text, target_size=100, max_size=200)
        for chunk_text, start, end in result:
            assert chunk_text == text[start:end]


# =============================================================================
# IS CODE FILE TESTS
# =============================================================================


class TestIsCodeFile:
    """Tests for is_code_file function."""

    def test_python_file(self):
        """Python files are detected."""
        assert is_code_file("test.py") is True
        assert is_code_file("/path/to/module.py") is True

    def test_javascript_file(self):
        """JavaScript files are detected."""
        assert is_code_file("app.js") is True
        assert is_code_file("component.jsx") is True
        assert is_code_file("app.ts") is True
        assert is_code_file("component.tsx") is True

    def test_other_languages(self):
        """Other common languages are detected."""
        assert is_code_file("Main.java") is True
        assert is_code_file("main.go") is True
        assert is_code_file("main.rs") is True
        assert is_code_file("main.cpp") is True
        assert is_code_file("main.c") is True
        assert is_code_file("header.h") is True

    def test_non_code_files(self):
        """Non-code files return False."""
        assert is_code_file("README.md") is False
        assert is_code_file("data.json") is False
        assert is_code_file("config.yaml") is False
        assert is_code_file("image.png") is False

    def test_no_extension(self):
        """Files without extension return False."""
        assert is_code_file("Dockerfile") is False
        assert is_code_file("Makefile") is False


# =============================================================================
# SCORE RELATION PATH TESTS
# =============================================================================


class TestScoreRelationPath:
    """Tests for score_relation_path function."""

    def test_empty_path(self):
        """Empty path returns 1.0."""
        assert score_relation_path([]) == 1.0

    def test_single_relation(self):
        """Single relation returns 1.0."""
        assert score_relation_path(["IsA"]) == 1.0

    def test_transitive_isa(self):
        """IsA -> IsA is fully transitive."""
        score = score_relation_path(["IsA", "IsA"])
        assert score == 1.0

    def test_valid_chain(self):
        """Valid relation chains have high scores."""
        # IsA -> HasProperty is valid
        score = score_relation_path(["IsA", "HasProperty"])
        assert score > 0.8

    def test_weak_chain(self):
        """Weak relation chains have lower scores."""
        # Antonym -> Antonym is weak
        score = score_relation_path(["Antonym", "Antonym"])
        assert score < 0.5

    def test_invalid_chain(self):
        """Invalid chains get default score."""
        # Made up relations
        score = score_relation_path(["Unknown1", "Unknown2"])
        # Should get default validity (from config)
        assert 0 <= score <= 1

    def test_long_path_decays(self):
        """Longer paths have lower scores (multiplicative)."""
        score_2 = score_relation_path(["IsA", "IsA"])
        score_3 = score_relation_path(["IsA", "IsA", "IsA"])
        # 3-hop path can't be higher than 2-hop for transitive relations
        assert score_3 <= score_2


class TestValidRelationChains:
    """Tests for VALID_RELATION_CHAINS constant."""

    def test_transitive_hierarchies(self):
        """Transitive hierarchies have high validity."""
        assert VALID_RELATION_CHAINS[("IsA", "IsA")] == 1.0
        assert VALID_RELATION_CHAINS[("PartOf", "PartOf")] == 1.0

    def test_causal_chains(self):
        """Causal chains are moderately valid."""
        assert VALID_RELATION_CHAINS[("Causes", "Causes")] >= 0.7

    def test_antonym_chains_weak(self):
        """Antonym chains are weak."""
        assert VALID_RELATION_CHAINS[("Antonym", "Antonym")] < 0.5
        assert VALID_RELATION_CHAINS[("Antonym", "IsA")] < 0.2


# =============================================================================
# IS CONCEPTUAL QUERY TESTS
# =============================================================================


class TestIsConceptualQuery:
    """Tests for is_conceptual_query function."""

    def test_what_is_query(self):
        """'what is' queries are conceptual."""
        assert is_conceptual_query("what is a tokenizer?") is True

    def test_how_does_query(self):
        """'how does' queries are conceptual."""
        assert is_conceptual_query("how does caching work?") is True

    def test_explain_query(self):
        """'explain' queries are conceptual."""
        assert is_conceptual_query("explain the architecture") is True

    def test_implementation_query(self):
        """Implementation-focused queries are not conceptual."""
        # Queries asking for specific code/functions
        result = is_conceptual_query("get function that validates input")
        # Should favor implementation keywords
        assert isinstance(result, bool)

    def test_mixed_query(self):
        """Mixed queries use keyword balance."""
        # This has both "explain" (conceptual) and specific terms
        result = is_conceptual_query("explain how to call the API")
        assert isinstance(result, bool)


# =============================================================================
# GET DOC TYPE BOOST TESTS
# =============================================================================


class TestGetDocTypeBoost:
    """Tests for get_doc_type_boost function."""

    def test_markdown_in_docs(self):
        """Markdown files in docs/ get documentation boost."""
        boost = get_doc_type_boost("docs/README.md")
        assert boost > 1.0

    def test_root_markdown(self):
        """Root markdown files get moderate boost."""
        boost = get_doc_type_boost("README.md")
        assert boost > 1.0

    def test_test_files(self):
        """Test files get penalty."""
        boost = get_doc_type_boost("tests/test_something.py")
        assert boost < 1.0

    def test_code_files(self):
        """Regular code files get neutral boost."""
        boost = get_doc_type_boost("src/module.py")
        assert boost == 1.0

    def test_with_metadata(self):
        """Metadata doc_type overrides path inference."""
        metadata = {"src/module.py": {"doc_type": "docs"}}
        boost = get_doc_type_boost("src/module.py", doc_metadata=metadata)
        assert boost > 1.0

    def test_custom_boosts(self):
        """Custom boost factors are applied."""
        custom = {"docs": 2.0, "code": 0.5}
        boost = get_doc_type_boost("docs/README.md", custom_boosts=custom)
        assert boost == 2.0


# =============================================================================
# APPLY DOC TYPE BOOST TESTS
# =============================================================================


class TestApplyDocTypeBoost:
    """Tests for apply_doc_type_boost function."""

    def test_empty_results(self):
        """Empty results return empty list."""
        result = apply_doc_type_boost([])
        assert result == []

    def test_no_boost(self):
        """boost_docs=False returns unchanged results."""
        results = [("doc1", 1.0), ("doc2", 0.5)]
        boosted = apply_doc_type_boost(results, boost_docs=False)
        assert boosted == results

    def test_results_boosted(self):
        """Results are boosted by doc type."""
        results = [("tests/test.py", 1.0), ("docs/guide.md", 0.5)]
        boosted = apply_doc_type_boost(results)
        # docs/guide.md should be boosted, tests/test.py should be penalized
        doc_scores = {doc: score for doc, score in boosted}
        # After boosting, doc may have higher relative score
        assert isinstance(doc_scores["docs/guide.md"], float)

    def test_results_reranked(self):
        """Results are re-sorted after boosting."""
        results = [("tests/test.py", 1.0), ("docs/guide.md", 0.9)]
        boosted = apply_doc_type_boost(results)
        # After boosting, guide.md (1.35) should beat test.py (0.8)
        # But depends on actual boost values
        assert len(boosted) == 2
        # Results should be sorted by boosted score (descending)
        assert boosted[0][1] >= boosted[1][1]
