"""
Unit tests for function registry.

Tests the FunctionRegistry class, FunctionSignature dataclass,
and QueryFunction ABC in isolation.
"""

import pytest
from typing import Any, Dict, List

from cortical.got.expression.registry import (
    FunctionRegistry,
    FunctionSignature,
    QueryFunction,
)


# Test helper classes

class MockFunction(QueryFunction):
    """Mock function for testing."""

    @classmethod
    def signature(cls) -> FunctionSignature:
        return FunctionSignature(
            name="mock",
            description="Mock function",
            required_args=["arg1"],
            optional_args={"opt1": "default"},
            returns="any"
        )

    def execute(
        self,
        manager: Any,
        args: List[Any],
        kwargs: Dict[str, Any]
    ) -> Any:
        return "mock_result"


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure clean registry for each test."""
    FunctionRegistry.clear()
    yield
    FunctionRegistry.clear()


class TestFunctionSignature:
    """Tests for FunctionSignature dataclass."""

    def test_create_signature_with_all_fields(self):
        """Test creating signature with all fields."""
        sig = FunctionSignature(
            name="test_func",
            description="A test function",
            required_args=["arg1", "arg2"],
            optional_args={"opt1": "default1", "opt2": 42},
            returns="str"
        )

        assert sig.name == "test_func"
        assert sig.description == "A test function"
        assert sig.required_args == ["arg1", "arg2"]
        assert sig.optional_args == {"opt1": "default1", "opt2": 42}
        assert sig.returns == "str"

    def test_create_signature_with_minimal_fields(self):
        """Test creating signature with minimal fields."""
        sig = FunctionSignature(
            name="simple",
            description="Simple function",
            required_args=[],
            optional_args={},
            returns="None"
        )

        assert sig.name == "simple"
        assert sig.required_args == []
        assert sig.optional_args == {}

    def test_signature_equality(self):
        """Test that signatures with same values are equal."""
        sig1 = FunctionSignature(
            name="func",
            description="desc",
            required_args=["a"],
            optional_args={},
            returns="str"
        )
        sig2 = FunctionSignature(
            name="func",
            description="desc",
            required_args=["a"],
            optional_args={},
            returns="str"
        )

        assert sig1 == sig2


class TestQueryFunctionABC:
    """Tests for QueryFunction abstract base class."""

    def test_cannot_instantiate_query_function_directly(self):
        """Test that QueryFunction cannot be instantiated directly."""
        with pytest.raises(TypeError):
            QueryFunction()  # Should fail - abstract class

    def test_subclass_must_implement_signature(self):
        """Test that subclass must implement signature method."""
        class IncompleteFunc(QueryFunction):
            def execute(self, manager, args, kwargs):
                return None

        with pytest.raises(TypeError):
            IncompleteFunc()  # Should fail - missing signature

    def test_subclass_must_implement_execute(self):
        """Test that subclass must implement execute method."""
        class IncompleteFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="incomplete",
                    description="Incomplete",
                    required_args=[],
                    optional_args={},
                    returns="None"
                )

        with pytest.raises(TypeError):
            IncompleteFunc()  # Should fail - missing execute

    def test_valid_query_function_subclass(self):
        """Test that properly implemented subclass can be instantiated."""
        class ValidFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="valid",
                    description="Valid function",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "valid"

        # Should not raise
        instance = ValidFunc()
        assert instance is not None
        assert instance.execute(None, [], {}) == "valid"


class TestFunctionRegistryRegister:
    """Tests for FunctionRegistry.register() method."""

    def test_register_function_decorator(self):
        """Test that register works as a decorator."""
        @FunctionRegistry.register("test_func")
        class TestFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="test_func",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "test"

        # Verify it's registered
        assert FunctionRegistry.get("test_func") is TestFunc

    def test_register_returns_original_class(self):
        """Test that register decorator returns the original class."""
        @FunctionRegistry.register("orig")
        class OriginalFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="orig",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "original"

        # The decorator should return the same class
        assert OriginalFunc is not None
        assert FunctionRegistry.get("orig") is OriginalFunc

    def test_register_normalizes_name_to_lowercase(self):
        """Test that register stores name in lowercase."""
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

        # Should be retrievable with lowercase
        assert FunctionRegistry.get("myfunc") is MyFunc

    def test_register_multiple_functions(self):
        """Test registering multiple functions."""
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

        assert FunctionRegistry.get("func1") is Func1
        assert FunctionRegistry.get("func2") is Func2

    def test_register_overwrites_existing_function(self):
        """Test that re-registering a name overwrites the previous function."""
        @FunctionRegistry.register("overwrite")
        class FirstFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="overwrite",
                    description="First",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "first"

        @FunctionRegistry.register("overwrite")
        class SecondFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="overwrite",
                    description="Second",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "second"

        # Should get the second function
        assert FunctionRegistry.get("overwrite") is SecondFunc


class TestFunctionRegistryGet:
    """Tests for FunctionRegistry.get() method."""

    def test_get_registered_function(self):
        """Test getting a registered function."""
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
                return "exists"

        result = FunctionRegistry.get("exists")
        assert result is ExistsFunc

    def test_get_unregistered_returns_none(self):
        """Test that getting an unregistered function returns None."""
        result = FunctionRegistry.get("nonexistent")
        assert result is None

    def test_get_case_insensitive(self):
        """Test that get is case-insensitive."""
        @FunctionRegistry.register("CamelCase")
        class CamelFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="CamelCase",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "camel"

        # All case variations should work
        assert FunctionRegistry.get("CamelCase") is CamelFunc
        assert FunctionRegistry.get("camelcase") is CamelFunc
        assert FunctionRegistry.get("CAMELCASE") is CamelFunc
        assert FunctionRegistry.get("camelCase") is CamelFunc

    def test_get_with_empty_string(self):
        """Test getting with empty string returns None."""
        result = FunctionRegistry.get("")
        assert result is None

    def test_get_with_whitespace(self):
        """Test that whitespace in name is preserved."""
        # Names with whitespace are probably invalid, but test the behavior
        @FunctionRegistry.register("func with spaces")
        class SpaceFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="func with spaces",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "spaces"

        # Should need exact spacing (after lowercase conversion)
        assert FunctionRegistry.get("func with spaces") is SpaceFunc
        assert FunctionRegistry.get("FUNC WITH SPACES") is SpaceFunc


class TestFunctionRegistryListFunctions:
    """Tests for FunctionRegistry.list_functions() method."""

    def test_list_functions_empty_registry(self):
        """Test list_functions on empty registry."""
        result = FunctionRegistry.list_functions()
        assert result == []
        assert isinstance(result, list)

    def test_list_functions_single_function(self):
        """Test list_functions with one registered function."""
        @FunctionRegistry.register("single")
        class SingleFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="single",
                    description="Single function",
                    required_args=["arg1"],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "single"

        result = FunctionRegistry.list_functions()
        assert len(result) == 1
        assert isinstance(result[0], FunctionSignature)
        assert result[0].name == "single"

    def test_list_functions_multiple_functions(self):
        """Test list_functions with multiple registered functions."""
        @FunctionRegistry.register("func_a")
        class FuncA(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="func_a",
                    description="Function A",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "a"

        @FunctionRegistry.register("func_b")
        class FuncB(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="func_b",
                    description="Function B",
                    required_args=["x"],
                    optional_args={},
                    returns="int"
                )

            def execute(self, manager, args, kwargs):
                return 1

        result = FunctionRegistry.list_functions()
        assert len(result) == 2

        # Extract names
        names = {sig.name for sig in result}
        assert names == {"func_a", "func_b"}

    def test_list_functions_returns_signatures_not_classes(self):
        """Test that list_functions returns FunctionSignature instances."""
        @FunctionRegistry.register("test")
        class TestFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="test",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "test"

        result = FunctionRegistry.list_functions()
        assert len(result) == 1
        sig = result[0]

        assert isinstance(sig, FunctionSignature)
        assert not isinstance(sig, type)  # Not a class

    def test_list_functions_includes_all_signature_fields(self):
        """Test that returned signatures have all fields."""
        @FunctionRegistry.register("detailed")
        class DetailedFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="detailed",
                    description="A detailed function",
                    required_args=["req1", "req2"],
                    optional_args={"opt1": "default1", "opt2": 42},
                    returns="complex result"
                )

            def execute(self, manager, args, kwargs):
                return {}

        result = FunctionRegistry.list_functions()
        sig = result[0]

        assert sig.name == "detailed"
        assert sig.description == "A detailed function"
        assert sig.required_args == ["req1", "req2"]
        assert sig.optional_args == {"opt1": "default1", "opt2": 42}
        assert sig.returns == "complex result"


class TestFunctionRegistryClear:
    """Tests for FunctionRegistry.clear() method."""

    def test_clear_empty_registry(self):
        """Test clearing an already empty registry."""
        FunctionRegistry.clear()  # Should not raise
        assert FunctionRegistry.list_functions() == []

    def test_clear_removes_all_functions(self):
        """Test that clear removes all registered functions."""
        # Register some functions
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

        # Verify they're registered
        assert len(FunctionRegistry.list_functions()) == 2

        # Clear
        FunctionRegistry.clear()

        # Verify they're gone
        assert FunctionRegistry.list_functions() == []
        assert FunctionRegistry.get("func1") is None
        assert FunctionRegistry.get("func2") is None

    def test_clear_allows_fresh_registration(self):
        """Test that after clear, can register new functions."""
        # Register and clear
        @FunctionRegistry.register("old")
        class OldFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="old",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "old"

        FunctionRegistry.clear()

        # Register new function
        @FunctionRegistry.register("new")
        class NewFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="new",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "new"

        # Verify state
        assert FunctionRegistry.get("old") is None
        assert FunctionRegistry.get("new") is NewFunc
        assert len(FunctionRegistry.list_functions()) == 1

    def test_multiple_clears(self):
        """Test that multiple clears are safe."""
        @FunctionRegistry.register("test")
        class TestFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="test",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "test"

        FunctionRegistry.clear()
        FunctionRegistry.clear()  # Should not raise
        FunctionRegistry.clear()  # Should not raise

        assert FunctionRegistry.list_functions() == []


class TestFunctionRegistrySingleton:
    """Tests for FunctionRegistry singleton behavior."""

    def test_instance_returns_same_instance(self):
        """Test that instance() returns the same singleton."""
        instance1 = FunctionRegistry.instance()
        instance2 = FunctionRegistry.instance()

        assert instance1 is instance2

    def test_class_methods_work_without_instance(self):
        """Test that class methods work without calling instance()."""
        # Should not need to call instance() explicitly
        @FunctionRegistry.register("test")
        class TestFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="test",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "test"

        result = FunctionRegistry.get("test")
        assert result is TestFunc

    def test_state_shared_across_calls(self):
        """Test that registry state is shared across all accesses."""
        @FunctionRegistry.register("shared")
        class SharedFunc(QueryFunction):
            @classmethod
            def signature(cls) -> FunctionSignature:
                return FunctionSignature(
                    name="shared",
                    description="Test",
                    required_args=[],
                    optional_args={},
                    returns="str"
                )

            def execute(self, manager, args, kwargs):
                return "shared"

        # Access via different methods
        via_get = FunctionRegistry.get("shared")
        via_list = FunctionRegistry.list_functions()[0]

        assert via_get is SharedFunc
        assert via_list.name == "shared"
