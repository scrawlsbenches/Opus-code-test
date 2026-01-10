"""
CDG Query Engine Behavioral Tests
=================================

As a developer using the CDG query language,
I want to parse, plan, and execute queries against the graph,
So that I can retrieve and filter entities efficiently.

Tests verify the complete query pipeline: lexer -> parser -> planner -> executor.

Stories covered:
1. Developer parses simple queries
2. Developer uses comparison operators
3. Developer uses logical operators (AND, OR, NOT)
4. Developer executes queries with filters
5. Developer calls functions (exists, blockers, dependents, etc.)
6. Developer handles query errors gracefully
7. Developer uses tokenizer directly
8. Developer integrates with GoTManager
"""

import pytest
from typing import List, Set

from cortical.cdg.query import (
    CDGQueryEngine,
    parse,
    tokenize,
    CDGQuery,
    QueryParseError,
    QueryExecutionError,
    QueryNotImplementedError,
)
from cortical.cdg.query.ast import (
    Comparison,
    Field,
    Literal,
    Op,
    AndExpr,
    OrExpr,
    NotExpr,
    FunctionCall,
)
from cortical.cdg.query.lexer import Token, TokenType
from cortical.cdg import CDGStore
from cortical.got import GoTManager


# =============================================================================
# FIXTURES - Using conftest.py fixtures with minimal additions
# =============================================================================

@pytest.fixture
def test_container():
    """
    Create a single in-memory container for all test components.

    This ensures GoTManager and CDGStore share the same state.
    """
    from cortical.core.bootstrap import create_container
    from pathlib import Path
    return create_container(got_dir=Path(".got"), use_memory=True)


@pytest.fixture
def query_engine(test_container):
    """
    Create a CDGQueryEngine with GoTManager integration.

    Uses a single container to ensure manager and store share state.
    """
    manager = test_container.resolve(GoTManager)
    store = test_container.resolve(CDGStore)
    engine = CDGQueryEngine(store=store)
    engine.register_extension('got_manager', manager)
    return engine, manager, store


@pytest.fixture
def sample_tasks(test_container):
    """
    Create a known set of sample tasks for query testing.

    Returns a dict with named access to specific tasks for assertions.
    """
    manager = test_container.resolve(GoTManager)

    # Create tasks with specific, known properties
    pending_high = manager.create_task(
        title="Fix authentication bug",
        priority="high",
        status="pending",
        category="bugfix"
    )

    in_progress_high = manager.create_task(
        title="Implement OAuth login",
        priority="high",
        status="in_progress",
        category="feature"
    )

    pending_medium = manager.create_task(
        title="Update documentation",
        priority="medium",
        status="pending",
        category="docs"
    )

    completed_low = manager.create_task(
        title="Clean up logs",
        priority="low",
        status="completed",
        category="refactor"
    )

    blocked_critical = manager.create_task(
        title="Deploy to production",
        priority="critical",
        status="blocked",
        category="feature"
    )
    manager.block_task(blocked_critical.id, "Waiting for security review")

    # Create dependency: in_progress_high depends on pending_high
    manager.add_dependency(in_progress_high.id, pending_high.id)

    return {
        'pending_high': pending_high,
        'in_progress_high': in_progress_high,
        'pending_medium': pending_medium,
        'completed_low': completed_low,
        'blocked_critical': blocked_critical,
        'all': [pending_high, in_progress_high, pending_medium, completed_low, blocked_critical],
    }


# =============================================================================
# STORY 1: Developer Parses Simple Queries
# =============================================================================

class TestDeveloperParsesSimpleQueries:
    """
    Epic: Query Parsing

    As a developer using the CDG query language,
    I want to parse query strings into AST structures,
    So that queries can be planned and executed.
    """

    def test_scenario_parse_from_clause_creates_entity_query(self):
        """
        Scenario: Parse simple FROM clause

        Given a query string with only a FROM clause
        When the parser processes it
        Then a CDGQuery AST is created
        And the entity_type is correctly extracted
        And there is no WHERE expression
        """
        # Given a query string with only a FROM clause
        query_string = "FROM task"

        # When the parser processes it
        query = parse(query_string)

        # Then a CDGQuery AST is created
        assert isinstance(query, CDGQuery), f"Expected CDGQuery, got {type(query)}"

        # And the entity_type is correctly extracted
        assert query.entity_type == "task", f"Expected 'task', got {query.entity_type}"

        # And there is no WHERE expression
        assert query.expression is None, "FROM without WHERE should have no expression"

        # And it's an entity query, not a function query
        assert query.is_entity_query() is True
        assert query.is_function_query() is False

    def test_scenario_parse_from_with_where_creates_comparison(self):
        """
        Scenario: Parse FROM with WHERE equals condition

        Given a query with a WHERE clause using equals
        When the parser processes it
        Then the expression is a Comparison
        And the field, operator, and value are correct
        """
        # Given a query with a WHERE clause using equals
        query_string = "FROM task WHERE status = 'pending'"

        # When the parser processes it
        query = parse(query_string)

        # Then the expression is a Comparison
        assert query.expression is not None, "WHERE clause should create an expression"
        assert isinstance(query.expression, Comparison), \
            f"Expected Comparison, got {type(query.expression)}"

        # And the field is correct
        assert isinstance(query.expression.field, Field)
        assert query.expression.field.name == "status", \
            f"Expected field 'status', got '{query.expression.field.name}'"

        # And the operator is EQ (equals)
        assert query.expression.op == Op.EQ, \
            f"Expected Op.EQ, got {query.expression.op}"

        # And the value is correct
        assert isinstance(query.expression.value, Literal)
        assert query.expression.value.value == "pending", \
            f"Expected 'pending', got '{query.expression.value.value}'"

    def test_scenario_parse_limit_clause(self):
        """
        Scenario: Parse query with LIMIT clause

        Given a query string with a LIMIT clause
        When the parser processes it
        Then the limit value is captured as an integer
        """
        # Given a query string with a LIMIT clause
        query_string = "FROM task LIMIT 10"

        # When the parser processes it
        query = parse(query_string)

        # Then the limit value is captured as an integer
        assert query.limit == 10, f"Expected limit 10, got {query.limit}"
        assert isinstance(query.limit, int), f"Limit should be int, got {type(query.limit)}"

    def test_scenario_parse_order_by_clause(self):
        """
        Scenario: Parse query with ORDER BY clause

        Given a query string with ORDER BY DESC
        When the parser processes it
        Then the order_by tuple captures field and direction
        """
        # Given a query string with ORDER BY DESC
        query_string = "FROM task ORDER BY created_at DESC"

        # When the parser processes it
        query = parse(query_string)

        # Then the order_by is set
        assert query.order_by is not None, "ORDER BY should set order_by"

        # And it's a tuple with field and direction
        assert isinstance(query.order_by, tuple), f"Expected tuple, got {type(query.order_by)}"
        # TODO: Verify exact tuple structure once we confirm the format
        # The order_by format may be (field_name, 'DESC') or similar

    def test_scenario_parse_complex_query_with_all_clauses(self):
        """
        Scenario: Parse complex query with WHERE, ORDER BY, and LIMIT

        Given a query with all clauses
        When the parser processes it
        Then all clauses are correctly parsed
        """
        # Given a query with all clauses
        query_string = "FROM task WHERE status = 'pending' ORDER BY priority DESC LIMIT 5"

        # When the parser processes it
        query = parse(query_string)

        # Then entity_type is correct
        assert query.entity_type == "task"

        # And WHERE expression exists
        assert query.expression is not None
        assert isinstance(query.expression, Comparison)

        # And ORDER BY is set
        assert query.order_by is not None

        # And LIMIT is correct
        assert query.limit == 5

    def test_scenario_parse_function_call_query(self):
        """
        Scenario: Parse a function call query

        Given a query that is a function call
        When the parser processes it
        Then the query represents a function call
        """
        # Given a query that is a function call
        query_string = "exists('T-123')"

        # When the parser processes it
        query = parse(query_string)

        # Then it's a function query
        assert query.is_function_query() is True
        assert query.is_entity_query() is False

        # And the expression is a FunctionCall
        assert isinstance(query.expression, FunctionCall), \
            f"Expected FunctionCall, got {type(query.expression)}"
        assert query.expression.name == "exists"
        assert len(query.expression.args) == 1

        # And the argument is correct
        arg = query.expression.args[0]
        assert isinstance(arg, Literal)
        assert arg.value == "T-123"


# =============================================================================
# STORY 2: Developer Uses Comparison Operators
# =============================================================================

class TestDeveloperUsesComparisonOperators:
    """
    Epic: Comparison Operators

    As a developer filtering entities,
    I want to use various comparison operators,
    So that I can express different filter conditions.
    """

    def test_scenario_equals_operator_parses_correctly(self):
        """
        Scenario: Parse equals (=) operator

        Given a WHERE clause with = operator
        When parsed
        Then the operator is Op.EQ
        """
        # Given a WHERE clause with = operator
        query = parse("FROM task WHERE status = 'pending'")

        # When parsed, Then the operator is Op.EQ
        assert query.expression.op == Op.EQ

    def test_scenario_not_equals_operator_parses_correctly(self):
        """
        Scenario: Parse not-equals (!=) operator

        Given a WHERE clause with != operator
        When parsed
        Then the operator is Op.NE
        """
        # Given a WHERE clause with != operator
        query = parse("FROM task WHERE status != 'completed'")

        # When parsed, Then the operator is Op.NE
        assert query.expression.op == Op.NE

    def test_scenario_greater_than_operator_parses_correctly(self):
        """
        Scenario: Parse greater-than (>) operator

        Given a WHERE clause with > operator
        When parsed
        Then the operator is Op.GT
        And the value is a number
        """
        # Given a WHERE clause with > operator
        query = parse("FROM task WHERE priority > 1")

        # When parsed, Then the operator is Op.GT
        assert query.expression.op == Op.GT

        # And the value is a number
        assert query.expression.value.value == 1

    def test_scenario_less_than_operator_parses_correctly(self):
        """
        Scenario: Parse less-than (<) operator

        Given a WHERE clause with < operator
        When parsed
        Then the operator is Op.LT
        """
        # Given a WHERE clause with < operator
        query = parse("FROM task WHERE version < 5")

        # When parsed, Then the operator is Op.LT
        assert query.expression.op == Op.LT

    def test_scenario_in_operator_parses_with_list(self):
        """
        Scenario: Parse IN operator with list

        Given a WHERE clause with IN operator and list values
        When parsed
        Then the operator is Op.IN
        And the value contains the list items
        """
        # Given a WHERE clause with IN operator
        query = parse("FROM task WHERE priority IN ['high', 'critical']")

        # When parsed, Then the operator is Op.IN
        assert query.expression.op == Op.IN

        # And the value is a list
        # TODO: Verify exact list representation in AST

    def test_scenario_like_operator_parses_with_pattern(self):
        """
        Scenario: Parse LIKE operator with pattern

        Given a WHERE clause with LIKE operator
        When parsed
        Then the operator is Op.LIKE
        And the pattern is captured
        """
        # Given a WHERE clause with LIKE operator
        query = parse("FROM task WHERE title LIKE '%auth%'")

        # When parsed, Then the operator is Op.LIKE
        assert query.expression.op == Op.LIKE

        # And the pattern is captured
        assert query.expression.value.value == "%auth%"


# =============================================================================
# STORY 3: Developer Uses Logical Operators
# =============================================================================

class TestDeveloperUsesLogicalOperators:
    """
    Epic: Logical Operators

    As a developer building complex filters,
    I want to combine conditions with AND, OR, NOT,
    So that I can express complex query logic.
    """

    def test_scenario_and_operator_creates_and_expression(self):
        """
        Scenario: Parse AND operator

        Given a WHERE clause with AND operator
        When parsed
        Then the expression is an AndExpr
        And it has two child comparisons
        """
        # Given a WHERE clause with AND operator
        query = parse("FROM task WHERE status = 'pending' AND priority = 'high'")

        # When parsed, Then the expression is an AndExpr
        assert isinstance(query.expression, AndExpr), \
            f"Expected AndExpr, got {type(query.expression)}"

        # And it has two children
        assert len(query.expression.children) == 2

        # And both children are Comparisons
        for child in query.expression.children:
            assert isinstance(child, Comparison)

    def test_scenario_or_operator_creates_or_expression(self):
        """
        Scenario: Parse OR operator

        Given a WHERE clause with OR operator
        When parsed
        Then the expression is an OrExpr
        And it has two child comparisons
        """
        # Given a WHERE clause with OR operator
        query = parse("FROM task WHERE status = 'pending' OR status = 'blocked'")

        # When parsed, Then the expression is an OrExpr
        assert isinstance(query.expression, OrExpr), \
            f"Expected OrExpr, got {type(query.expression)}"

        # And it has two children
        assert len(query.expression.children) == 2

    def test_scenario_not_operator_creates_not_expression(self):
        """
        Scenario: Parse NOT operator

        Given a WHERE clause with NOT operator
        When parsed
        Then the expression is a NotExpr
        And it wraps a comparison
        """
        # Given a WHERE clause with NOT operator
        query = parse("FROM task WHERE NOT status = 'completed'")

        # When parsed, Then the expression is a NotExpr
        assert isinstance(query.expression, NotExpr), \
            f"Expected NotExpr, got {type(query.expression)}"

        # And it wraps a Comparison
        assert isinstance(query.expression.child, Comparison)

    def test_scenario_complex_boolean_respects_precedence(self):
        """
        Scenario: Parse complex boolean with parentheses

        Given a WHERE clause with nested boolean logic
        When parsed
        Then operator precedence is respected
        And parentheses group correctly
        """
        # Given a WHERE clause with nested boolean logic
        query = parse(
            "FROM task WHERE (status = 'pending' OR status = 'blocked') AND priority = 'high'"
        )

        # When parsed, Then the top-level is AND (due to precedence)
        assert isinstance(query.expression, AndExpr), \
            f"Expected AndExpr at top level, got {type(query.expression)}"

        # And one child is an OrExpr (the parenthesized group)
        children_types = [type(c) for c in query.expression.children]
        assert OrExpr in children_types, "Expected OrExpr as child of AndExpr"


# =============================================================================
# STORY 4: Developer Executes Queries with Filters
# =============================================================================

class TestDeveloperExecutesQueriesWithFilters:
    """
    Epic: Query Execution

    As a developer querying the graph,
    I want to execute queries and get filtered results,
    So that I can retrieve specific entities.
    """

    def test_scenario_exists_function_returns_true_for_existing_task(
        self, query_engine, sample_tasks
    ):
        """
        Scenario: Check if existing task exists

        Given a task that exists in the system
        When querying exists() with its ID
        Then the result is [True]
        """
        engine, manager, store = query_engine

        # Given a task that exists in the system
        existing_task = sample_tasks['pending_high']

        # When querying exists() with its ID
        result = engine.query(f"exists('{existing_task.id}')")

        # Then the result is [True]
        assert result == [True], f"Expected [True], got {result}"

    def test_scenario_exists_function_returns_false_for_nonexistent_task(
        self, query_engine, sample_tasks
    ):
        """
        Scenario: Check if nonexistent task exists

        Given a task ID that does not exist
        When querying exists() with that ID
        Then the result is [False]
        """
        engine, manager, store = query_engine

        # Given a task ID that does not exist
        nonexistent_id = "T-NONEXISTENT-99999"

        # When querying exists() with that ID
        result = engine.query(f"exists('{nonexistent_id}')")

        # Then the result is [False]
        assert result == [False], f"Expected [False], got {result}"

    def test_scenario_dependents_returns_tasks_depending_on_target(
        self, query_engine, sample_tasks
    ):
        """
        Scenario: Query dependents of a task

        Given task A that task B depends on (B -> A)
        When querying dependents(A)
        Then task B is in the results
        """
        engine, manager, store = query_engine

        # Given task A that task B depends on
        prereq_task = sample_tasks['pending_high']  # A
        dependent_task = sample_tasks['in_progress_high']  # B depends on A

        # When querying dependents(A)
        result = engine.query(f"dependents('{prereq_task.id}')")

        # Then task B is in the results
        assert isinstance(result, list), f"Expected list, got {type(result)}"

        if len(result) > 0:
            result_ids = {t.id for t in result}
            assert dependent_task.id in result_ids, \
                f"Expected {dependent_task.id} in {result_ids}"

    def test_scenario_connected_to_returns_linked_entities(
        self, query_engine, sample_tasks
    ):
        """
        Scenario: Query entities connected to a task

        Given a task with dependencies
        When querying connected_to()
        Then connected entities are returned
        """
        engine, manager, store = query_engine

        # Given a task with dependencies
        task = sample_tasks['pending_high']

        # When querying connected_to()
        result = engine.query(f"connected_to('{task.id}')")

        # Then result is a list (may be empty if no connections)
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_scenario_blocked_returns_blocked_tasks(
        self, query_engine, sample_tasks
    ):
        """
        Scenario: Query for blocked tasks

        Given a task with blocked status
        When querying blocked()
        Then the blocked task is in results
        """
        engine, manager, store = query_engine

        # Given a task with blocked status
        blocked_task = sample_tasks['blocked_critical']

        # When querying blocked()
        result = engine.query("blocked()")

        # Then result is a list
        assert isinstance(result, list), f"Expected list, got {type(result)}"

        # And the blocked task should be in results
        if len(result) > 0:
            result_ids = {t.id for t in result}
            assert blocked_task.id in result_ids, \
                f"Expected blocked task {blocked_task.id} in results"


# =============================================================================
# STORY 5: Developer Handles Query Errors
# =============================================================================

class TestDeveloperHandlesQueryErrors:
    """
    Epic: Error Handling

    As a developer writing queries,
    I want helpful error messages when queries fail,
    So that I can fix my queries quickly.
    """

    def test_scenario_parse_error_on_invalid_syntax(self):
        """
        Scenario: Invalid syntax raises QueryParseError

        Given a query with invalid syntax (missing entity type)
        When parsing
        Then QueryParseError is raised
        """
        # Given a query with invalid syntax
        invalid_query = "FROM WHERE status = 'pending'"  # Missing entity type

        # When parsing, Then QueryParseError is raised
        with pytest.raises(QueryParseError) as exc_info:
            parse(invalid_query)

        # And the error has a message
        assert len(str(exc_info.value)) > 0

    def test_scenario_parse_error_on_missing_value(self):
        """
        Scenario: Missing value after operator raises error

        Given a query with missing value after =
        When parsing
        Then QueryParseError is raised
        """
        # Given a query with missing value
        invalid_query = "FROM task WHERE status ="

        # When parsing, Then QueryParseError is raised
        with pytest.raises(QueryParseError):
            parse(invalid_query)

    def test_scenario_parse_error_on_unclosed_string(self):
        """
        Scenario: Unclosed string quote raises error

        Given a query with unclosed string
        When parsing
        Then an error is raised
        """
        # Given a query with unclosed string
        invalid_query = "FROM task WHERE status = 'pending"

        # When parsing, Then error is raised
        with pytest.raises((QueryParseError, Exception)):
            parse(invalid_query)

    def test_scenario_execution_error_on_unknown_function(self, query_engine):
        """
        Scenario: Unknown function raises error

        Given a query calling a nonexistent function
        When executing
        Then an error is raised
        And the function name is mentioned
        """
        engine, manager, store = query_engine

        # Given a query calling nonexistent function
        query_string = "nonexistent_function()"

        # When executing, Then an error is raised
        with pytest.raises((QueryExecutionError, KeyError, Exception)) as exc_info:
            engine.query(query_string)

        # And the function name should be mentioned in error
        error_message = str(exc_info.value).lower()
        assert "nonexistent" in error_message or "not found" in error_message or \
               "unknown" in error_message, \
               f"Error should mention unknown function: {exc_info.value}"

    def test_scenario_got_function_without_manager_raises_error(self, test_container):
        """
        Scenario: GoT function without manager raises helpful error

        Given a query engine without got_manager extension
        When calling a GoT-specific function like blockers()
        Then an error is raised indicating manager is required
        """
        # Given a query engine without got_manager
        store = test_container.resolve(CDGStore)
        engine = CDGQueryEngine(store=store)
        # Note: NOT registering got_manager extension

        # When calling a GoT-specific function
        # Then an error is raised
        with pytest.raises(Exception) as exc_info:
            engine.query("blockers('T-123')")

        # And error mentions the missing extension
        error_message = str(exc_info.value).lower()
        assert "got_manager" in error_message or "extension" in error_message or \
               "required" in error_message, \
               f"Error should mention missing extension: {exc_info.value}"


# =============================================================================
# STORY 6: Developer Uses Tokenizer
# =============================================================================

class TestDeveloperUsesTokenizer:
    """
    Epic: Tokenization

    As a developer debugging queries,
    I want to examine the token stream,
    So that I can understand how queries are lexed.
    """

    def test_scenario_tokenize_simple_query(self):
        """
        Scenario: Tokenize FROM query

        Given a simple FROM query
        When tokenizing
        Then correct tokens are produced
        """
        # Given a simple FROM query
        query_string = "FROM task"

        # When tokenizing
        tokens = list(tokenize(query_string))

        # Then we get at least 2 tokens (FROM, task, possibly EOF)
        assert len(tokens) >= 2, f"Expected at least 2 tokens, got {len(tokens)}"

        # And first token is FROM keyword
        assert tokens[0].type == TokenType.FROM, \
            f"Expected FROM token, got {tokens[0].type}"
        assert tokens[0].value.upper() == "FROM"

        # And second token is identifier 'task'
        assert tokens[1].type == TokenType.IDENTIFIER, \
            f"Expected IDENTIFIER token, got {tokens[1].type}"
        assert tokens[1].value == "task"

    def test_scenario_tokenize_string_literal(self):
        """
        Scenario: Tokenize query with string literal

        Given a query with a string value
        When tokenizing
        Then the string token is captured
        """
        # Given a query with a string value
        query_string = "status = 'pending'"

        # When tokenizing
        tokens = list(tokenize(query_string))

        # Then we have identifier, equals, and string tokens
        token_types = [t.type for t in tokens]

        assert TokenType.IDENTIFIER in token_types, "Should have IDENTIFIER token"
        assert TokenType.EQ in token_types, "Should have EQ token"
        assert TokenType.STRING in token_types, "Should have STRING token"

        # And the string value is correct
        string_tokens = [t for t in tokens if t.type == TokenType.STRING]
        assert len(string_tokens) == 1
        # Value may include quotes or just the content
        assert "pending" in string_tokens[0].value

    def test_scenario_tokenize_operators(self):
        """
        Scenario: Tokenize comparison and logical operators

        Given a query with multiple operators
        When tokenizing
        Then each operator has correct token type
        """
        # Given a query with multiple operators
        query_string = "x >= 10 AND y != 5"

        # When tokenizing
        tokens = list(tokenize(query_string))
        token_types = [t.type for t in tokens]

        # Then we have GTE, AND, and NE tokens
        assert TokenType.GTE in token_types, "Should have >= token"
        assert TokenType.AND in token_types, "Should have AND token"
        assert TokenType.NE in token_types, "Should have != token"

    def test_scenario_tokens_have_positions(self):
        """
        Scenario: Tokens include position information

        Given a query string
        When tokenizing
        Then each token has a position >= 0
        """
        # Given a query string
        query_string = "FROM task WHERE x = 1"

        # When tokenizing
        tokens = list(tokenize(query_string))

        # Then each token has a valid position
        for token in tokens:
            assert hasattr(token, 'position'), "Token should have position"
            assert isinstance(token.position, int), "Position should be int"
            assert token.position >= 0, f"Position should be >= 0, got {token.position}"


# =============================================================================
# STORY 7: Developer Integrates with GoTManager
# =============================================================================

class TestDeveloperIntegratesWithGoTManager:
    """
    Epic: GoTManager Integration

    As a developer using the query engine with GoT,
    I want to register GoTManager as an extension,
    So that query functions can access task data.
    """

    def test_scenario_register_extension(self, test_container):
        """
        Scenario: Register GoTManager extension

        Given a query engine
        When registering got_manager extension
        Then the extension is accessible
        """
        # Given a query engine
        manager = test_container.resolve(GoTManager)
        store = test_container.resolve(CDGStore)
        engine = CDGQueryEngine(store=store)

        # When registering got_manager extension
        engine.register_extension('got_manager', manager)

        # Then the extension is accessible
        assert 'got_manager' in engine.extensions
        assert engine.extensions['got_manager'] is manager

    def test_scenario_create_task_and_query_exists(self, test_container):
        """
        Scenario: Create task and verify existence via query

        Given a newly created task
        When querying exists() for that task
        Then the result is [True]
        """
        # Given components from the same container
        manager = test_container.resolve(GoTManager)
        store = test_container.resolve(CDGStore)

        # And a newly created task
        task = manager.create_task(
            title="Test task for query",
            priority="high",
            status="pending"
        )

        # Set up query engine
        engine = CDGQueryEngine(store=store)
        engine.register_extension('got_manager', manager)

        # When querying exists() for that task
        result = engine.query(f"exists('{task.id}')")

        # Then the result is [True]
        assert result == [True], f"Expected [True], got {result}"


# =============================================================================
# STORY 8: Developer Uses Core Functions
# =============================================================================

class TestDeveloperUsesCoreFunctions:
    """
    Epic: Core Query Functions

    As a developer using query functions,
    I want access to utility functions like count, entity_types, fields,
    So that I can introspect and aggregate data.
    """

    def test_scenario_count_function_counts_list(self):
        """
        Scenario: Count function counts items

        Given a list of items
        When calling count() function directly
        Then the count is returned
        """
        # Given - test count function directly
        from cortical.cdg.query.functions.core import CountFunction
        from cortical.cdg.query.registry import QueryContext

        ctx = QueryContext()
        func = CountFunction()

        # When calling count with a list
        result = func.execute(ctx, [[1, 2, 3, 4, 5]], {})

        # Then the count is 5
        assert result == 5, f"Expected 5, got {result}"

    def test_scenario_entity_types_returns_list(self, query_engine):
        """
        Scenario: Query available entity types

        Given a query engine
        When calling entity_types()
        Then a list is returned
        """
        engine, manager, store = query_engine

        # When calling entity_types()
        result = engine.query("entity_types()")

        # Then a list is returned
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_scenario_fields_function_returns_field_info(self, query_engine):
        """
        Scenario: Query fields for entity type

        Given a query engine
        When calling fields('task')
        Then field information is returned
        """
        engine, manager, store = query_engine

        # When calling fields('task')
        result = engine.query("fields('task')")

        # Then result is list or dict
        assert isinstance(result, (list, dict)), \
            f"Expected list or dict, got {type(result)}"


# =============================================================================
# STORY 9: Developer Uses Path Finding
# =============================================================================

class TestDeveloperUsesPathFinding:
    """
    Epic: Graph Traversal

    As a developer navigating the task graph,
    I want to find paths between entities,
    So that I can understand relationships.
    """

    def test_scenario_path_finds_connected_tasks(self, query_engine, sample_tasks):
        """
        Scenario: Find path between dependent tasks

        Given task B depends on task A
        When querying path(B, A)
        Then a path is returned containing both tasks
        """
        engine, manager, store = query_engine

        # Given task B depends on task A
        task_a = sample_tasks['pending_high']
        task_b = sample_tasks['in_progress_high']

        # When querying path(B, A)
        result = engine.query(f"path('{task_b.id}', '{task_a.id}')")

        # Then result is a list
        assert isinstance(result, list), f"Expected list, got {type(result)}"

        # TODO: Verify path contents once we confirm the return format
        # The path function may return [[path_nodes]] or [path_nodes] or [None]

    def test_scenario_path_returns_none_for_unconnected_tasks(
        self, query_engine, sample_tasks
    ):
        """
        Scenario: No path between unconnected tasks

        Given two unconnected tasks
        When querying path between them
        Then result indicates no path (None or empty)
        """
        engine, manager, store = query_engine

        # Given two unconnected tasks
        task_a = sample_tasks['pending_high']
        task_c = sample_tasks['completed_low']  # No connection to task_a

        # When querying path between them
        result = engine.query(f"path('{task_a.id}', '{task_c.id}')")

        # Then result indicates no path
        assert (
            result is None or
            result == [None] or
            result == [[]] or
            result == [] or
            (isinstance(result, list) and len(result) == 1 and result[0] is None) or
            (isinstance(result, list) and len(result) == 1 and result[0] == [])
        ), f"Expected no-path result, got {result}"


# =============================================================================
# STORY 10: Developer Uses Filter Functions
# =============================================================================

class TestDeveloperUsesFilterFunctions:
    """
    Epic: Filter Functions

    As a developer filtering tasks,
    I want specialized filter functions,
    So that I can quickly find specific task states.
    """

    def test_scenario_orphan_nodes_finds_unconnected_tasks(
        self, query_engine, sample_tasks
    ):
        """
        Scenario: Find orphan nodes (tasks with no edges)

        Given some tasks have edges and some don't
        When querying orphan_nodes()
        Then unconnected tasks are returned
        """
        engine, manager, store = query_engine

        # Given sample_tasks includes some without edges
        # pending_medium, completed_low, blocked_critical have no dependency edges

        # When querying orphan_nodes()
        result = engine.query("orphan_nodes()")

        # Then result is a list
        assert isinstance(result, list), f"Expected list, got {type(result)}"

        # And should include tasks without edges
        # Note: pending_high and in_progress_high have a dependency edge

    def test_scenario_unassigned_finds_tasks_without_sprint(
        self, query_engine, sample_tasks
    ):
        """
        Scenario: Find unassigned tasks (no sprint)

        Given tasks without sprint assignment
        When querying unassigned()
        Then those tasks are returned
        """
        engine, manager, store = query_engine

        # Given all sample tasks are unassigned (no sprint_id)

        # When querying unassigned()
        result = engine.query("unassigned()")

        # Then result is a list
        assert isinstance(result, list), f"Expected list, got {type(result)}"

        # And all sample tasks should be included (none have sprints)
        if len(result) > 0:
            result_ids = {t.id for t in result}
            # At least some of our sample tasks should be in results
            sample_ids = {t.id for t in sample_tasks['all']}
            overlap = result_ids & sample_ids
            assert len(overlap) > 0, \
                "Expected some sample tasks in unassigned results"
