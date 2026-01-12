"""
BDD Specification: Unified Knowledge Query System

Epic: Self-Referential Queryable Knowledge for Code Navigation

As a coding assistant (Claude),
I want to query my own knowledge graph about code structure and semantics,
So that I can find files intelligently instead of relying on grep.

Background:
- Current state: ASTIndex knows code structure, CognitiveAgent knows word associations
- Problem: These systems don't talk to each other
- Solution: Bridge them with CODE atoms and REFERS_TO links
- Goal: Answer "where is authentication handled?" with structured, confident responses

Design Principles:
1. UNIFIED GRAPH
   - CODE atoms (FILE, CLASS, FUNCTION) live alongside WORD atoms
   - REFERS_TO links connect semantic (WORD) to structural (CODE)
   - Single graph is the source of truth

2. HONEST UNCERTAINTY
   - Every step can say "I don't know"
   - Unknown word → "I don't recognize this term"
   - No REFERS_TO → "I don't know what code this relates to"
   - Low confidence → "I found possibilities but I'm uncertain"

3. STRUCTURED RESPONSES
   - Not just file paths, but WHY those files
   - Confidence scores at every level
   - Suggestions when uncertain

4. WORKS WITH SMALL CORPUS
   - Sparse data → honest uncertainty
   - Few code files → still useful
   - Graceful degradation, not hallucination

Acceptance Criteria:
[ ] CODE atoms created from ASTIndex parsing
[ ] REFERS_TO links connect words to code entities
[ ] Query parsing extracts intent and subject
[ ] Semantic expansion via SIMILARITY links
[ ] Code entity lookup via REFERS_TO links
[ ] Structured response with confidence
[ ] "I don't know" when appropriate
"""

import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
from pathlib import Path


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class QueryIntent(Enum):
    """Types of queries we can handle."""
    LOCATION = auto()       # "where is X?" → find file/location
    IMPLEMENTATION = auto() # "how does X work?" → find implementation details
    DEFINITION = auto()     # "what is X?" → find definition
    RELATIONSHIP = auto()   # "what uses X?" / "what does X use?"
    UNKNOWN = auto()        # Can't determine intent


class ConfidenceLevel(Enum):
    """Confidence levels for responses."""
    HIGH = auto()      # >0.7 - confident answer
    MEDIUM = auto()    # 0.4-0.7 - probable answer
    LOW = auto()       # 0.1-0.4 - uncertain, showing possibilities
    NONE = auto()      # <0.1 - effectively "I don't know"


@dataclass
class CodeEntity:
    """A code entity (file, class, function) from the graph."""
    name: str
    entity_type: str  # FILE, CLASS, FUNCTION
    file_path: str
    line_number: Optional[int] = None

    def __str__(self) -> str:
        if self.line_number:
            return f"{self.file_path}:{self.line_number}"
        return self.file_path


@dataclass
class QueryResult:
    """
    Structured response to a knowledge query.

    This is what we return - not just answers, but context about
    what we understood, what we found, and how confident we are.
    """
    # What we understood from the query
    understood_terms: List[str]
    intent: QueryIntent

    # What we found
    entities: List[Tuple[CodeEntity, float, str]]  # (entity, confidence, reason)

    # Our confidence
    overall_confidence: ConfidenceLevel

    # What we're uncertain about
    uncertain_about: List[str] = field(default_factory=list)

    # Suggestions if we're uncertain
    suggestions: List[str] = field(default_factory=list)

    # Did we hit "I don't know" at any step?
    unknown_terms: List[str] = field(default_factory=list)
    no_code_mapping: List[str] = field(default_factory=list)

    @property
    def is_confident(self) -> bool:
        """Are we confident enough to give a direct answer?"""
        return self.overall_confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)

    @property
    def found_something(self) -> bool:
        """Did we find any relevant code?"""
        return len(self.entities) > 0

    def top_entity(self) -> Optional[CodeEntity]:
        """Get the highest confidence entity."""
        if not self.entities:
            return None
        return self.entities[0][0]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def knowledge_graph(memory_cognitive_container):
    """
    Unified knowledge graph with CODE and WORD atoms.
    Uses DI container for test isolation.
    """
    from cortical.cognitive.training import IncrementalTrainer

    trainer = memory_cognitive_container.resolve(IncrementalTrainer)
    return UnifiedKnowledgeGraph(trainer)


class UnifiedKnowledgeGraph:
    """
    Test wrapper providing unified query capabilities.

    Bridges:
    - CognitiveAgent (WORD atoms, SIMILARITY links)
    - ASTIndex (code structure)
    - REFERS_TO links (semantic → structural mapping)
    """

    def __init__(self, trainer: 'IncrementalTrainer'):
        self._trainer = trainer
        self._agent = trainer.agent
        self._code_entities: Dict[str, CodeEntity] = {}

    def index_code_file(self, file_path: str, content: str) -> None:
        """
        Index a code file, creating CODE atoms and REFERS_TO links.

        This bridges ASTIndex parsing into the cognitive graph.
        """
        # Would call ASTIndex.index_file() and create atoms
        pass

    def train_on_text(self, texts: List[str]) -> None:
        """Train WORD atoms and SIMILARITY links."""
        self._trainer.bridge.learn_vocabulary(texts)
        for i, text in enumerate(texts):
            self._trainer.bridge.feed_text(text, doc_id=f"doc_{i}")

    def query(self, query_text: str) -> QueryResult:
        """
        Answer a natural language query about the codebase.

        This is the main entry point for knowledge queries.
        """
        return self._agent.query_knowledge(query_text)

    def add_code_entity(self, entity: CodeEntity) -> None:
        """Add a code entity to the graph."""
        self._code_entities[entity.name] = entity

    def add_refers_to(self, word: str, entity_name: str, weight: float) -> None:
        """Create a REFERS_TO link from word to code entity."""
        pass


# =============================================================================
# STORY 1: CODE ENTITY ATOMS
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.knowledge
class TestCodeEntityAtoms:
    """
    Story 1: Code entities become atoms in the graph

    As a knowledge system,
    I want code structure (files, classes, functions) as atoms,
    So that I can reason about code the same way I reason about words.
    """

    def test_file_becomes_atom(self, knowledge_graph):
        """
        Scenario: Python file creates FILE atom

        Given a Python source file
        When indexing the file
        Then a FILE atom is created in the graph
        And the atom contains file path and metadata

        Because files are first-class entities we need to find.
        """
        # Given
        file_content = '''
        """Authentication handler module."""

        class AuthHandler:
            def login(self, username, password):
                pass
        '''

        # When
        knowledge_graph.index_code_file("auth/handler.py", file_content)

        # Then
        result = knowledge_graph.query("where is AuthHandler?")
        assert result.found_something
        assert any("auth/handler.py" in str(e[0]) for e in result.entities)

    def test_class_becomes_atom(self, knowledge_graph):
        """
        Scenario: Class definition creates CLASS atom

        Given a Python file with class definitions
        When indexing the file
        Then CLASS atoms are created for each class
        And DEFINES links connect FILE → CLASS

        Because classes are key navigation targets.
        """
        # Given
        file_content = '''
        class UserManager:
            """Manages user accounts."""
            pass

        class SessionManager:
            """Manages user sessions."""
            pass
        '''

        # When
        knowledge_graph.index_code_file("managers.py", file_content)

        # Then
        result = knowledge_graph.query("what is UserManager?")
        assert result.intent == QueryIntent.DEFINITION
        assert result.found_something

    def test_function_becomes_atom(self, knowledge_graph):
        """
        Scenario: Function definition creates FUNCTION atom

        Given a Python file with functions
        When indexing the file
        Then FUNCTION atoms are created
        And CONTAINS links connect CLASS → FUNCTION for methods

        Because functions are where the action happens.
        """
        # Given
        file_content = '''
        def authenticate(username: str, password: str) -> bool:
            """Verify user credentials."""
            return check_password(username, password)
        '''

        # When
        knowledge_graph.index_code_file("auth.py", file_content)

        # Then
        result = knowledge_graph.query("where is authenticate defined?")
        assert result.found_something
        top = result.top_entity()
        assert top is not None
        assert "auth.py" in top.file_path


# =============================================================================
# STORY 2: REFERS_TO LINKS
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.knowledge
class TestRefersToLinks:
    """
    Story 2: Words connect to code via REFERS_TO links

    As a query system,
    I want words linked to relevant code entities,
    So that natural language queries find structural results.
    """

    def test_word_refers_to_file(self, knowledge_graph):
        """
        Scenario: Word in filename creates REFERS_TO link

        Given file "authentication.py"
        When indexing
        Then "authentication" REFERS_TO authentication.py (high weight)
        And "auth" REFERS_TO authentication.py (medium weight, substring)

        Because filenames are strong signals.
        """
        # Given
        knowledge_graph.index_code_file("authentication.py", "# Auth module")

        # When
        result = knowledge_graph.query("where is authentication?")

        # Then
        assert result.found_something
        assert result.overall_confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)

    def test_word_refers_to_class(self, knowledge_graph):
        """
        Scenario: Word in class name creates REFERS_TO link

        Given class "AuthenticationHandler" in some_file.py
        When indexing
        Then "authentication" REFERS_TO AuthenticationHandler
        And "handler" REFERS_TO AuthenticationHandler

        Because class names indicate purpose.
        """
        # Given
        file_content = '''
        class AuthenticationHandler:
            """Handles user authentication."""
            pass
        '''
        knowledge_graph.index_code_file("handlers.py", file_content)

        # When
        result = knowledge_graph.query("where is authentication handled?")

        # Then
        assert result.found_something
        # Should find AuthenticationHandler
        entities = [e[0].name for e in result.entities]
        assert any("Auth" in name for name in entities)

    def test_docstring_creates_refers_to(self, knowledge_graph):
        """
        Scenario: Words in docstrings create REFERS_TO links

        Given a function with docstring mentioning "validate credentials"
        When indexing
        Then "validate" and "credentials" REFER_TO that function

        Because docstrings describe intent.
        """
        # Given
        file_content = '''
        def check_login(user, pwd):
            """Validate user credentials against the database."""
            pass
        '''
        knowledge_graph.index_code_file("login.py", file_content)

        # When
        result = knowledge_graph.query("where do we validate credentials?")

        # Then
        assert result.found_something
        assert any("login.py" in str(e[0]) for e in result.entities)

    def test_refers_to_weight_hierarchy(self, knowledge_graph):
        """
        Scenario: REFERS_TO weights reflect relevance

        Given "auth" appears in:
          - filename: auth.py (weight: 1.0)
          - class name: AuthHandler (weight: 0.9)
          - function name: do_auth (weight: 0.8)
          - docstring: "handles auth" (weight: 0.5)
          - code body: auth_token = ... (weight: 0.3)
        When querying "auth"
        Then results are ranked by weight

        Because not all occurrences are equally meaningful.
        """
        # Given
        knowledge_graph.index_code_file("auth.py", "class AuthHandler: pass")
        knowledge_graph.index_code_file("utils.py", "def helper(): auth_token = 1")

        # When
        result = knowledge_graph.query("where is auth?")

        # Then
        assert result.found_something
        # auth.py should rank higher than utils.py
        if len(result.entities) >= 2:
            assert result.entities[0][1] >= result.entities[1][1]


# =============================================================================
# STORY 3: QUERY UNDERSTANDING
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.knowledge
class TestQueryUnderstanding:
    """
    Story 3: Natural language queries are parsed into structured intents

    As a user asking questions,
    I want my queries understood semantically,
    So that I get contextually appropriate responses.
    """

    def test_where_query_location_intent(self, knowledge_graph):
        """
        Scenario: "where" questions have LOCATION intent

        Given query "where is the login handler?"
        When parsing intent
        Then intent is LOCATION
        And subject is ["login", "handler"]

        Because "where" means find the location.
        """
        # Given
        knowledge_graph.index_code_file("login.py", "class LoginHandler: pass")

        # When
        result = knowledge_graph.query("where is the login handler?")

        # Then
        assert result.intent == QueryIntent.LOCATION
        assert "login" in result.understood_terms or "handler" in result.understood_terms

    def test_how_query_implementation_intent(self, knowledge_graph):
        """
        Scenario: "how" questions have IMPLEMENTATION intent

        Given query "how does authentication work?"
        When parsing intent
        Then intent is IMPLEMENTATION
        And response includes implementation details, not just location

        Because "how" means explain the mechanism.
        """
        # Given
        knowledge_graph.index_code_file("auth.py", '''
        def authenticate(user, pwd):
            """Check password hash and return token."""
            hash = compute_hash(pwd)
            if verify(user, hash):
                return generate_token(user)
            return None
        ''')

        # When
        result = knowledge_graph.query("how does authentication work?")

        # Then
        assert result.intent == QueryIntent.IMPLEMENTATION

    def test_what_query_definition_intent(self, knowledge_graph):
        """
        Scenario: "what" questions have DEFINITION intent

        Given query "what is TransactionManager?"
        When parsing intent
        Then intent is DEFINITION
        And response points to class definition

        Because "what" means define/describe.
        """
        # Given
        knowledge_graph.index_code_file("tx.py", '''
        class TransactionManager:
            """Manages database transactions with ACID guarantees."""
            pass
        ''')

        # When
        result = knowledge_graph.query("what is TransactionManager?")

        # Then
        assert result.intent == QueryIntent.DEFINITION


# =============================================================================
# STORY 4: SEMANTIC EXPANSION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.knowledge
class TestSemanticExpansion:
    """
    Story 4: Queries expand via SIMILARITY links

    As a query system,
    I want to expand query terms semantically,
    So that "auth" also finds "login", "session", "credentials".
    """

    def test_query_expands_to_similar_terms(self, knowledge_graph):
        """
        Scenario: Query terms expand via associations

        Given trained associations: auth ~ login ~ session ~ credentials
        And file "session_manager.py"
        When querying "where is authentication?"
        Then "session_manager.py" is found
        Because "authentication" expands to include "session"
        """
        # Given: Train word associations
        knowledge_graph.train_on_text([
            "authentication login session credentials token",
            "user authentication requires valid session",
            "login creates session with auth token",
        ])
        knowledge_graph.index_code_file("session_manager.py", "class SessionManager: pass")

        # When
        result = knowledge_graph.query("where is authentication?")

        # Then
        assert "session" in result.understood_terms or result.found_something

    def test_expansion_does_not_hallucinate(self, knowledge_graph):
        """
        Scenario: Expansion stays grounded in learned associations

        Given NO training connecting "auth" to "database"
        When querying "where is authentication?"
        Then database files are NOT found via expansion

        Because we only expand based on actual learned associations.
        """
        # Given: Train specific associations (no database)
        knowledge_graph.train_on_text([
            "authentication login session",
        ])
        knowledge_graph.index_code_file("database.py", "class Database: pass")
        knowledge_graph.index_code_file("auth.py", "class Auth: pass")

        # When
        result = knowledge_graph.query("where is authentication?")

        # Then: auth.py found, database.py not found via expansion
        file_paths = [str(e[0]) for e in result.entities]
        # database.py should not appear, or should rank much lower
        if "database.py" in str(file_paths):
            # If it appears, it should be lower confidence
            auth_confidence = next((e[1] for e in result.entities if "auth" in str(e[0])), 0)
            db_confidence = next((e[1] for e in result.entities if "database" in str(e[0])), 0)
            assert auth_confidence > db_confidence


# =============================================================================
# STORY 5: HONEST UNCERTAINTY
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.knowledge
class TestHonestUncertainty:
    """
    Story 5: System honestly reports what it doesn't know

    As a user,
    I want honest "I don't know" responses,
    So that I don't waste time on hallucinated answers.
    """

    def test_unknown_word_acknowledged(self, knowledge_graph):
        """
        Scenario: Unknown word is explicitly flagged

        Given query with word not in vocabulary
        When querying
        Then unknown_terms includes that word
        And overall_confidence is LOW or NONE

        Because honesty about limits builds trust.
        """
        # Given: Minimal training (doesn't include "quantum")
        knowledge_graph.train_on_text(["hello world"])

        # When
        result = knowledge_graph.query("where is quantum entanglement?")

        # Then
        assert "quantum" in result.unknown_terms or result.overall_confidence == ConfidenceLevel.NONE

    def test_no_code_mapping_acknowledged(self, knowledge_graph):
        """
        Scenario: Known word with no code mapping is acknowledged

        Given word "philosophy" in vocabulary (from training)
        But no code files relate to philosophy
        When querying "where is philosophy handled?"
        Then no_code_mapping includes "philosophy"
        And response suggests this is outside code scope

        Because we know the word but can't map it to code.
        """
        # Given: Word in vocab but no code
        knowledge_graph.train_on_text(["philosophy epistemology metaphysics"])
        # No code files indexed

        # When
        result = knowledge_graph.query("where is philosophy implemented?")

        # Then
        assert not result.found_something
        assert "philosophy" in result.no_code_mapping or result.overall_confidence == ConfidenceLevel.NONE

    def test_low_confidence_shows_possibilities(self, knowledge_graph):
        """
        Scenario: Uncertain results shown as possibilities

        Given ambiguous query matching multiple files equally
        When querying
        Then overall_confidence is LOW or MEDIUM
        And uncertain_about explains the ambiguity
        And suggestions offer clarification options

        Because uncertainty should guide, not block.
        """
        # Given: Ambiguous setup
        knowledge_graph.index_code_file("auth/login.py", "def login(): pass")
        knowledge_graph.index_code_file("auth/oauth.py", "def login_oauth(): pass")
        knowledge_graph.index_code_file("api/login.py", "def api_login(): pass")

        # When
        result = knowledge_graph.query("where is login?")

        # Then: Multiple results with uncertainty
        assert len(result.entities) > 1
        if result.overall_confidence == ConfidenceLevel.LOW:
            assert len(result.suggestions) > 0 or len(result.uncertain_about) > 0

    def test_confident_when_clear_match(self, knowledge_graph):
        """
        Scenario: High confidence when match is clear

        Given single obvious match
        When querying
        Then overall_confidence is HIGH
        And no uncertainty flags

        Because clear matches deserve confidence.
        """
        # Given: Clear, unique match
        knowledge_graph.index_code_file("transaction_manager.py", '''
        class TransactionManager:
            """The one and only transaction manager."""
            pass
        ''')

        # When
        result = knowledge_graph.query("where is TransactionManager?")

        # Then
        assert result.overall_confidence == ConfidenceLevel.HIGH
        assert len(result.uncertain_about) == 0
        assert result.top_entity() is not None


# =============================================================================
# STORY 6: NON-CODE QUESTIONS
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.knowledge
class TestNonCodeQuestions:
    """
    Story 6: Handle non-code questions appropriately

    As a knowledge system,
    I want to answer general questions when I can,
    And clearly say "I don't know" when I can't.
    """

    def test_answer_from_trained_knowledge(self, knowledge_graph):
        """
        Scenario: Answer questions from trained text

        Given training on documentation about the project
        When asking about project concepts
        Then answer from learned associations

        Because knowledge includes documentation, not just code.
        """
        # Given: Train on documentation
        knowledge_graph.train_on_text([
            "The cognitive graph uses atoms as the universal unit.",
            "Atoms can be nodes or links. Links connect atoms.",
            "Truth values have strength and confidence.",
        ])

        # When
        result = knowledge_graph.query("what are atoms?")

        # Then: Should find relevant concepts
        assert "atoms" in result.understood_terms
        # May not find code entities, but should understand the term

    def test_clearly_state_unknown(self, knowledge_graph):
        """
        Scenario: Clearly state when topic is unknown

        Given no training or code about a topic
        When asking about that topic
        Then response clearly indicates "I don't know"

        Because silence is worse than honest ignorance.
        """
        # Given: Minimal training
        knowledge_graph.train_on_text(["hello world"])

        # When
        result = knowledge_graph.query("what is the meaning of life?")

        # Then
        assert result.overall_confidence == ConfidenceLevel.NONE
        assert not result.found_something


# =============================================================================
# STORY 7: COMPLETE QUERY FLOW
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.knowledge
class TestCompleteQueryFlow:
    """
    Story 7: End-to-end query flow

    As Claude working on a coding task,
    I want the full query flow to work seamlessly,
    So that I can find code without manual grep.
    """

    def test_realistic_query_flow(self, knowledge_graph):
        """
        Scenario: Complete realistic query

        Given a codebase with auth-related files
        And trained word associations
        When asking "where is authentication handled?"
        Then get structured response with:
          - understood terms
          - code entities with confidence
          - reasons for each match

        Because this is the actual use case.
        """
        # Given: Realistic setup
        knowledge_graph.train_on_text([
            "authentication login session jwt token",
            "user authentication validates credentials",
            "session management handles auth tokens",
        ])

        knowledge_graph.index_code_file("auth/handler.py", '''
        """Authentication handler - main entry point for auth."""

        class AuthHandler:
            def authenticate(self, username: str, password: str) -> str:
                """Authenticate user and return JWT token."""
                pass
        ''')

        knowledge_graph.index_code_file("auth/session.py", '''
        """Session management for authenticated users."""

        class SessionManager:
            def create_session(self, user_id: str) -> Session:
                pass
        ''')

        knowledge_graph.index_code_file("utils/helpers.py", '''
        """General utilities."""

        def format_date(d): pass
        ''')

        # When
        result = knowledge_graph.query("where is authentication handled?")

        # Then
        assert result.intent == QueryIntent.LOCATION
        assert "authentication" in result.understood_terms or "auth" in result.understood_terms
        assert result.found_something

        # auth/handler.py should rank highest
        top = result.top_entity()
        assert top is not None
        assert "auth" in top.file_path.lower()

        # Should explain why
        top_reason = result.entities[0][2]
        assert len(top_reason) > 0

        # utils/helpers.py should NOT be in results
        all_paths = [str(e[0]) for e in result.entities]
        assert not any("helpers" in p for p in all_paths)

    def test_response_format_matches_spec(self, knowledge_graph):
        """
        Scenario: Response matches expected format

        Given any valid query
        When getting response
        Then format matches:
          {
            understood: ["authentication", "auth", "login", "session"],
            code_entities: [
              {file: "auth/handler.py", confidence: 0.9, why: "..."},
            ],
            uncertain_about: [...],
            suggestion: "..."
          }

        Because this is what we promised.
        """
        # Given
        knowledge_graph.index_code_file("example.py", "class Example: pass")

        # When
        result = knowledge_graph.query("where is Example?")

        # Then: All expected fields present
        assert hasattr(result, 'understood_terms')
        assert hasattr(result, 'entities')
        assert hasattr(result, 'overall_confidence')
        assert hasattr(result, 'uncertain_about')
        assert hasattr(result, 'suggestions')
        assert hasattr(result, 'unknown_terms')
        assert hasattr(result, 'no_code_mapping')

        # Entities have expected structure
        if result.entities:
            entity, confidence, reason = result.entities[0]
            assert isinstance(entity, CodeEntity)
            assert isinstance(confidence, float)
            assert isinstance(reason, str)


# =============================================================================
# IMPLEMENTATION NOTES
# =============================================================================

"""
Implementation Plan:

1. NEW ATOM TYPES (add to cortical/cognitive/graph.py):
   - FILE = auto()
   - CLASS = auto()
   - FUNCTION = auto()
   - MODULE = auto()

   New link types:
   - DEFINES = auto()      # FILE DEFINES CLASS/FUNCTION
   - REFERS_TO = auto()    # WORD REFERS_TO CODE_ENTITY
   - CONTAINS = auto()     # CLASS CONTAINS METHOD

2. CODE ENTITY BRIDGE (new file: cortical/cognitive/code_bridge.py):

   class CodeEntityBridge:
       def __init__(self, graph: CognitiveGraph, ast_index: ASTIndex):
           self.graph = graph
           self.ast_index = ast_index

       def index_file(self, file_path: str, content: str) -> None:
           # Parse with AST
           self.ast_index.index_file(file_path)

           # Create FILE atom
           file_atom = self.graph.node(file_path, AtomType.FILE)

           # Create CLASS atoms
           for cls in self.ast_index.find_classes_in_file(file_path):
               class_atom = self.graph.node(cls.name, AtomType.CLASS)
               self.graph.link(AtomType.DEFINES, [file_atom, class_atom])

               # Create REFERS_TO from class name words
               for word in split_camel_case(cls.name):
                   word_atom = self.graph.node(word.lower(), AtomType.WORD)
                   self._create_refers_to(word_atom, class_atom, weight=0.9)

           # Similar for functions, docstrings, etc.

3. QUERY HANDLER (new file: cortical/cognitive/query_handler.py):

   class QueryHandler:
       def __init__(self, agent: CognitiveAgent, code_bridge: CodeEntityBridge):
           self.agent = agent
           self.code_bridge = code_bridge

       def query(self, query_text: str) -> QueryResult:
           # Step 1: Parse intent
           intent = parse_intent_query(query_text)

           # Step 2: Extract and expand terms
           terms = self._extract_terms(query_text)
           expanded = self._expand_terms(terms)  # Uses SIMILARITY

           # Step 3: Find code entities
           entities = self._find_code_entities(expanded)  # Uses REFERS_TO

           # Step 4: Rank and explain
           ranked = self._rank_entities(entities, intent)

           # Step 5: Assess confidence
           confidence = self._assess_confidence(ranked, terms)

           # Step 6: Build response
           return QueryResult(
               understood_terms=expanded,
               intent=self._to_query_intent(intent),
               entities=ranked,
               overall_confidence=confidence,
               ...
           )

4. INTEGRATION with existing systems:
   - Use existing parse_intent_query() from cortical/query/intent.py
   - Use existing expand_query() from cortical/query/expansion.py
   - Use existing ASTIndex from cortical/spark/ast_index.py
   - Add to CognitiveAgent as query_knowledge() method
"""
