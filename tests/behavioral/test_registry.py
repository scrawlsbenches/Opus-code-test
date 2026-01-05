"""
Behavioral tests for function registry.

Feature: Function Registry for Extensible Query Functions

As a developer
I want to register custom query functions
So that I can extend the query system without modifying core code

Scenario: Register and retrieve functions
  Given I have a custom query function
  When I register it with @FunctionRegistry.register("my_func")
  Then I can retrieve it via FunctionRegistry.get("my_func")

Scenario: Case-insensitive function lookup
  Given I have a registered function "MyFunc"
  When I look it up with any case variation
  Then I get the same function

Scenario: List all registered functions
  Given I have multiple registered functions
  When I call list_functions()
  Then I get all function signatures

Scenario: Unknown function returns None
  Given I have a registry with some functions
  When I try to get a function that doesn't exist
  Then I get None
"""

import pytest
from typing import Any, Dict, List

from cortical.got.expression.registry import (
    FunctionRegistry,
    FunctionSignature,
    QueryFunction,
)


# Test fixture functions

class CountFunction(QueryFunction):
    """Example function that counts entities."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="count",
            description="Count entities matching criteria",
            required_args=["type"],
            optional_args={"status": None},
            returns="integer count"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> int:
        # Dummy implementation for testing
        return 42


class SearchFunction(QueryFunction):
    """Example function that searches entities."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="search",
            description="Search entities by text",
            required_args=["query"],
            optional_args={"limit": 10},
            returns="list of entities"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> List[str]:
        # Dummy implementation for testing
        return ["result1", "result2"]


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before and after each test, re-register after."""
    FunctionRegistry.clear()
    yield
    FunctionRegistry.clear()
    # Re-import to re-register functions for subsequent tests
    import importlib
    from cortical.got.expression import functions
    importlib.reload(functions.graph)
    importlib.reload(functions.filters)


class TestFunctionRegistration:
    """Scenario: Register and retrieve functions."""

    def test_register_function_with_decorator(self):
        """
        Given I have a custom query function
        When I register it with @FunctionRegistry.register("count")
        Then the function is stored in the registry
        """
        # When
        @FunctionRegistry.register("count")
        class MyCountFunction(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="count",
                    description="Count things",
                    required_args=[],
                    optional_args={},
                    returns="int"
                )

            def execute(self, manager, args, kwargs):
                return 5

        # Then
        result = FunctionRegistry.get("count")
        assert result is MyCountFunction
        assert result is not None

    def test_retrieve_registered_function(self):
        """
        Given I have registered a function
        When I retrieve it via get()
        Then I get the same function class
        """
        # Given
        @FunctionRegistry.register("my_func")
        class MyFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="my_func",
                    description="Test function",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "result"

        # When
        retrieved = FunctionRegistry.get("my_func")

        # Then
        assert retrieved is MyFunc

    def test_can_instantiate_and_execute_registered_function(self):
        """
        Given I have registered a function
        When I retrieve and instantiate it
        Then I can execute it successfully
        """
        # Given
        @FunctionRegistry.register("executable")
        class ExecutableFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="executable",
                    description="Executable test",
                    required_args=["x"],
                    optional_args={},
                    returns="int"
                )

            def execute(self, manager, args, kwargs):
                return args[0] * 2

        # When
        func_class = FunctionRegistry.get("executable")
        instance = func_class()
        result = instance.execute(None, [5], {})

        # Then
        assert result == 10


class TestCaseInsensitiveLookup:
    """Scenario: Case-insensitive function lookup."""

    def test_lookup_with_different_cases(self):
        """
        Given I have a registered function "MyFunc"
        When I look it up with different case variations
        Then I get the same function
        """
        # Given
        @FunctionRegistry.register("MyFunc")
        class MyFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="MyFunc",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "test"

        # When/Then
        assert FunctionRegistry.get("MyFunc") is MyFunc
        assert FunctionRegistry.get("myfunc") is MyFunc
        assert FunctionRegistry.get("MYFUNC") is MyFunc
        assert FunctionRegistry.get("myFunc") is MyFunc

    def test_register_with_uppercase_retrieve_with_lowercase(self):
        """
        Given I register a function with uppercase name
        When I retrieve with lowercase
        Then I get the function
        """
        # Given
        @FunctionRegistry.register("UPPER")
        class UpperFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="UPPER",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "upper"

        # When
        result = FunctionRegistry.get("upper")

        # Then
        assert result is UpperFunc


class TestListFunctions:
    """Scenario: List all registered functions."""

    def test_list_functions_returns_all_signatures(self):
        """
        Given I have multiple registered functions
        When I call list_functions()
        Then I get all function signatures
        """
        # Given
        @FunctionRegistry.register("func1")
        class Func1(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="func1",
                    description="First function",
                    required_args=["a"],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "1"

        @FunctionRegistry.register("func2")
        class Func2(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="func2",
                    description="Second function",
                    required_args=["b"],
                    optional_args={},
                    returns="int"
                )

            def execute(self, manager, args, kwargs):
                return 2

        # When
        signatures = FunctionRegistry.list_functions()

        # Then
        assert len(signatures) == 2
        names = [sig.name for sig in signatures]
        assert "func1" in names
        assert "func2" in names

    def test_list_functions_empty_when_no_functions(self):
        """
        Given I have an empty registry
        When I call list_functions()
        Then I get an empty list
        """
        # Given - registry is already cleared by fixture

        # When
        signatures = FunctionRegistry.list_functions()

        # Then
        assert signatures == []

    def test_signatures_contain_correct_metadata(self):
        """
        Given I have a registered function with detailed signature
        When I get the signatures list
        Then the signature contains all metadata
        """
        # Given
        @FunctionRegistry.register("detailed")
        class DetailedFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="detailed",
                    description="A detailed test function",
                    required_args=["arg1", "arg2"],
                    optional_args={"opt1": "default1", "opt2": 42},
                    returns="dict with results"
                )

            def execute(self, manager, args, kwargs):
                return {}

        # When
        signatures = FunctionRegistry.list_functions()
        sig = signatures[0]

        # Then
        assert sig.name == "detailed"
        assert sig.description == "A detailed test function"
        assert sig.required_args == ["arg1", "arg2"]
        assert sig.optional_args == {"opt1": "default1", "opt2": 42}
        assert sig.returns == "dict with results"


class TestUnknownFunction:
    """Scenario: Unknown function returns None."""

    def test_get_nonexistent_function_returns_none(self):
        """
        Given I have a registry
        When I try to get a function that doesn't exist
        Then I get None
        """
        # When
        result = FunctionRegistry.get("nonexistent")

        # Then
        assert result is None

    def test_get_unknown_function_with_existing_functions(self):
        """
        Given I have some registered functions
        When I try to get a different function that doesn't exist
        Then I get None
        """
        # Given
        @FunctionRegistry.register("exists")
        class ExistsFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="exists",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "test"

        # When
        result = FunctionRegistry.get("doesnotexist")

        # Then
        assert result is None


class TestRegistryClear:
    """Scenario: Clear registry for test isolation."""

    def test_clear_removes_all_functions(self):
        """
        Given I have registered functions
        When I call clear()
        Then all functions are removed
        """
        # Given
        @FunctionRegistry.register("func1")
        class Func1(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="func1",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "1"

        @FunctionRegistry.register("func2")
        class Func2(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="func2",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "2"

        # Verify they exist
        assert FunctionRegistry.get("func1") is not None
        assert FunctionRegistry.get("func2") is not None

        # When
        FunctionRegistry.clear()

        # Then
        assert FunctionRegistry.get("func1") is None
        assert FunctionRegistry.get("func2") is None
        assert FunctionRegistry.list_functions() == []

    def test_can_register_after_clear(self):
        """
        Given I have cleared the registry
        When I register a new function
        Then it works normally
        """
        # Given
        @FunctionRegistry.register("before_clear")
        class BeforeFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="before_clear",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "before"

        FunctionRegistry.clear()

        # When
        @FunctionRegistry.register("after_clear")
        class AfterFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="after_clear",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "after"

        # Then
        assert FunctionRegistry.get("before_clear") is None
        assert FunctionRegistry.get("after_clear") is AfterFunc
