"""
Behavioral tests for the expression module structure.

Feature: Expression Module Structure

These tests verify that the expression module is properly structured
and all components are importable.
"""

import pytest


class TestModuleImportable:
    """Scenario: Module is importable."""

    def test_import_expression_module(self):
        """Given the expression module exists, When I import it, Then no error."""
        from cortical.got import expression
        assert expression is not None

    def test_parse_function_available(self):
        """The parse function should be available from the module."""
        from cortical.got.expression import parse
        assert callable(parse)

    def test_execute_function_available(self):
        """The execute function should be available from the module."""
        from cortical.got.expression import execute
        assert callable(execute)


class TestSubmodulesImportable:
    """Scenario: Submodules are importable."""

    def test_import_lexer(self):
        """Lexer submodule should be importable."""
        from cortical.got.expression import lexer
        assert lexer is not None

    def test_import_parser(self):
        """Parser submodule should be importable."""
        from cortical.got.expression import parser
        assert parser is not None

    def test_import_ast(self):
        """AST submodule should be importable."""
        from cortical.got.expression import ast
        assert ast is not None

    def test_import_registry(self):
        """Registry submodule should be importable."""
        from cortical.got.expression import registry
        assert registry is not None

    def test_import_executor(self):
        """Executor submodule should be importable."""
        from cortical.got.expression import executor
        assert executor is not None

    def test_import_errors(self):
        """Errors submodule should be importable."""
        from cortical.got.expression import errors
        assert errors is not None


class TestASTNodesAvailable:
    """Scenario: AST node types are available."""

    def test_expression_base_class(self):
        """Expression base class should be importable."""
        from cortical.got.expression import Expression
        assert Expression is not None

    def test_literal_node(self):
        """Literal node should be importable."""
        from cortical.got.expression import Literal
        assert Literal is not None

    def test_comparison_node(self):
        """Comparison node should be importable."""
        from cortical.got.expression import Comparison
        assert Comparison is not None

    def test_and_expr_node(self):
        """AndExpr node should be importable."""
        from cortical.got.expression import AndExpr
        assert AndExpr is not None

    def test_or_expr_node(self):
        """OrExpr node should be importable."""
        from cortical.got.expression import OrExpr
        assert OrExpr is not None

    def test_function_call_node(self):
        """FunctionCall node should be importable."""
        from cortical.got.expression import FunctionCall
        assert FunctionCall is not None

    def test_query_node(self):
        """Query node should be importable."""
        from cortical.got.expression import Query
        assert Query is not None


class TestErrorTypesAvailable:
    """Scenario: Error types are available."""

    def test_query_error(self):
        """QueryError should be importable."""
        from cortical.got.expression import QueryError
        assert issubclass(QueryError, Exception)

    def test_lexer_error(self):
        """LexerError should be importable."""
        from cortical.got.expression import LexerError
        assert issubclass(LexerError, Exception)

    def test_parse_error(self):
        """ParseError should be importable."""
        from cortical.got.expression import ParseError
        assert issubclass(ParseError, Exception)

    def test_execution_error(self):
        """ExecutionError should be importable."""
        from cortical.got.expression import ExecutionError
        assert issubclass(ExecutionError, Exception)


class TestRegistryAvailable:
    """Scenario: Function registry is available."""

    def test_function_registry(self):
        """FunctionRegistry should be importable."""
        from cortical.got.expression import FunctionRegistry
        assert FunctionRegistry is not None

    def test_function_signature(self):
        """FunctionSignature should be importable."""
        from cortical.got.expression import FunctionSignature
        assert FunctionSignature is not None

    def test_query_function(self):
        """QueryFunction base class should be importable."""
        from cortical.got.expression import QueryFunction
        assert QueryFunction is not None


class TestLexerComponentsAvailable:
    """Scenario: Lexer components are available."""

    def test_lexer_class(self):
        """Lexer class should be importable."""
        from cortical.got.expression import Lexer
        assert Lexer is not None

    def test_token_class(self):
        """Token class should be importable."""
        from cortical.got.expression import Token
        assert Token is not None

    def test_token_type_enum(self):
        """TokenType enum should be importable."""
        from cortical.got.expression import TokenType
        assert TokenType is not None

    def test_tokenize_function(self):
        """tokenize function should be importable."""
        from cortical.got.expression import tokenize
        assert callable(tokenize)
