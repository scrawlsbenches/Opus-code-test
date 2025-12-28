"""
Unit tests for cortical/got/protocol.py

Tests the GoTBackend Protocol class:
- Protocol structure verification
- Method signature validation
- Protocol conformance checking
"""

import pytest
from typing import get_type_hints, Optional, List, Dict, Any, Tuple
from unittest.mock import MagicMock
import inspect

from cortical.got.protocol import GoTBackend
from cortical.reasoning.graph_of_thought import ThoughtNode


class TestGoTBackendProtocolStructure:
    """Tests for GoTBackend Protocol structure."""

    def test_protocol_is_protocol_class(self):
        """GoTBackend is a Protocol class."""
        from typing import Protocol
        assert issubclass(GoTBackend, Protocol)

    def test_protocol_has_task_crud_methods(self):
        """Protocol defines task CRUD methods."""
        assert hasattr(GoTBackend, 'create_task')
        assert hasattr(GoTBackend, 'get_task')
        assert hasattr(GoTBackend, 'list_tasks')
        assert hasattr(GoTBackend, 'update_task')
        assert hasattr(GoTBackend, 'delete_task')

    def test_protocol_has_state_transition_methods(self):
        """Protocol defines state transition methods."""
        assert hasattr(GoTBackend, 'start_task')
        assert hasattr(GoTBackend, 'complete_task')
        assert hasattr(GoTBackend, 'block_task')

    def test_protocol_has_relationship_methods(self):
        """Protocol defines relationship management methods."""
        assert hasattr(GoTBackend, 'add_dependency')
        assert hasattr(GoTBackend, 'add_blocks')
        assert hasattr(GoTBackend, 'get_blockers')
        assert hasattr(GoTBackend, 'get_dependents')
        assert hasattr(GoTBackend, 'get_task_dependencies')

    def test_protocol_has_query_methods(self):
        """Protocol defines query and analytics methods."""
        assert hasattr(GoTBackend, 'get_stats')
        assert hasattr(GoTBackend, 'validate')
        assert hasattr(GoTBackend, 'get_blocked_tasks')
        assert hasattr(GoTBackend, 'get_active_tasks')
        assert hasattr(GoTBackend, 'what_blocks')
        assert hasattr(GoTBackend, 'what_depends_on')
        assert hasattr(GoTBackend, 'get_all_relationships')

    def test_protocol_has_persistence_methods(self):
        """Protocol defines persistence methods."""
        assert hasattr(GoTBackend, 'sync_to_git')
        assert hasattr(GoTBackend, 'export_graph')

    def test_protocol_has_query_language_method(self):
        """Protocol defines query language method."""
        assert hasattr(GoTBackend, 'query')


class TestGoTBackendMethodSignatures:
    """Tests for method signatures in GoTBackend Protocol."""

    def test_create_task_signature(self):
        """create_task has correct signature."""
        sig = inspect.signature(GoTBackend.create_task)
        params = list(sig.parameters.keys())

        assert 'self' in params
        assert 'title' in params
        assert 'priority' in params
        assert 'category' in params
        assert 'description' in params
        assert 'sprint_id' in params
        assert 'depends_on' in params
        assert 'blocks' in params

        # Check defaults
        assert sig.parameters['priority'].default == 'medium'
        assert sig.parameters['category'].default == 'feature'
        assert sig.parameters['description'].default == ''

    def test_get_task_signature(self):
        """get_task has correct signature."""
        sig = inspect.signature(GoTBackend.get_task)
        params = list(sig.parameters.keys())

        assert 'self' in params
        assert 'task_id' in params

    def test_list_tasks_signature(self):
        """list_tasks has correct signature."""
        sig = inspect.signature(GoTBackend.list_tasks)
        params = list(sig.parameters.keys())

        assert 'status' in params
        assert 'priority' in params
        assert 'category' in params
        assert 'sprint_id' in params
        assert 'blocked_only' in params

        # Check defaults
        assert sig.parameters['blocked_only'].default is False

    def test_update_task_signature(self):
        """update_task has correct signature."""
        sig = inspect.signature(GoTBackend.update_task)
        params = list(sig.parameters.keys())

        assert 'self' in params
        assert 'task_id' in params
        # **updates should be captured

    def test_delete_task_signature(self):
        """delete_task has correct signature."""
        sig = inspect.signature(GoTBackend.delete_task)
        params = list(sig.parameters.keys())

        assert 'task_id' in params
        assert 'force' in params
        assert sig.parameters['force'].default is False

    def test_start_task_signature(self):
        """start_task has correct signature."""
        sig = inspect.signature(GoTBackend.start_task)
        params = list(sig.parameters.keys())

        assert 'task_id' in params

    def test_complete_task_signature(self):
        """complete_task has correct signature."""
        sig = inspect.signature(GoTBackend.complete_task)
        params = list(sig.parameters.keys())

        assert 'task_id' in params
        assert 'retrospective' in params
        assert sig.parameters['retrospective'].default == ''

    def test_block_task_signature(self):
        """block_task has correct signature."""
        sig = inspect.signature(GoTBackend.block_task)
        params = list(sig.parameters.keys())

        assert 'task_id' in params
        assert 'reason' in params
        assert 'blocked_by' in params

    def test_query_signature(self):
        """query has correct signature."""
        sig = inspect.signature(GoTBackend.query)
        params = list(sig.parameters.keys())

        assert 'query_str' in params


class TestGoTBackendProtocolConformance:
    """Tests for protocol conformance checking."""

    def test_mock_implementation_conforms(self):
        """A mock implementation can conform to the protocol."""
        # Create a mock that implements all required methods
        mock_backend = MagicMock(spec=GoTBackend)

        # Verify mock has all required methods
        assert hasattr(mock_backend, 'create_task')
        assert hasattr(mock_backend, 'get_task')
        assert hasattr(mock_backend, 'list_tasks')
        assert hasattr(mock_backend, 'update_task')
        assert hasattr(mock_backend, 'delete_task')
        assert hasattr(mock_backend, 'start_task')
        assert hasattr(mock_backend, 'complete_task')
        assert hasattr(mock_backend, 'block_task')
        assert hasattr(mock_backend, 'add_dependency')
        assert hasattr(mock_backend, 'add_blocks')
        assert hasattr(mock_backend, 'get_blockers')
        assert hasattr(mock_backend, 'get_dependents')
        assert hasattr(mock_backend, 'get_task_dependencies')
        assert hasattr(mock_backend, 'get_stats')
        assert hasattr(mock_backend, 'validate')
        assert hasattr(mock_backend, 'get_blocked_tasks')
        assert hasattr(mock_backend, 'get_active_tasks')
        assert hasattr(mock_backend, 'what_blocks')
        assert hasattr(mock_backend, 'what_depends_on')
        assert hasattr(mock_backend, 'get_all_relationships')
        assert hasattr(mock_backend, 'sync_to_git')
        assert hasattr(mock_backend, 'export_graph')
        assert hasattr(mock_backend, 'query')

    def test_protocol_can_be_used_as_type_hint(self):
        """GoTBackend can be used as a type hint."""
        def process_backend(backend: GoTBackend) -> str:
            return backend.create_task("Test")

        # This should not raise
        mock = MagicMock()
        mock.create_task.return_value = "T-001"

        result = process_backend(mock)
        assert result == "T-001"

    def test_minimal_conforming_class(self):
        """A minimal class can conform to the protocol."""
        class MinimalBackend:
            def create_task(self, title, priority="medium", category="feature",
                          description="", sprint_id=None, depends_on=None,
                          blocks=None) -> str:
                return "T-001"

            def get_task(self, task_id):
                return None

            def list_tasks(self, status=None, priority=None, category=None,
                          sprint_id=None, blocked_only=False):
                return []

            def update_task(self, task_id, **updates):
                return True

            def delete_task(self, task_id, force=False):
                return (True, "Deleted")

            def start_task(self, task_id):
                return True

            def complete_task(self, task_id, retrospective=""):
                return True

            def block_task(self, task_id, reason="", blocked_by=None):
                return True

            def add_dependency(self, task_id, depends_on_id):
                return True

            def add_blocks(self, blocker_id, blocked_id):
                return True

            def get_blockers(self, task_id):
                return []

            def get_dependents(self, task_id):
                return []

            def get_task_dependencies(self, task_id):
                return []

            def get_stats(self):
                return {}

            def validate(self):
                return []

            def get_blocked_tasks(self):
                return []

            def get_active_tasks(self):
                return []

            def what_blocks(self, task_id):
                return []

            def what_depends_on(self, task_id):
                return []

            def get_all_relationships(self, task_id):
                return {}

            def sync_to_git(self):
                return "synced"

            def export_graph(self, output_path=None):
                return {}

            def query(self, query_str):
                return []

        backend = MinimalBackend()

        # Verify it works
        assert backend.create_task("Test") == "T-001"
        assert backend.get_task("T-001") is None
        assert backend.list_tasks() == []
        assert backend.update_task("T-001", status="done") is True
        assert backend.delete_task("T-001") == (True, "Deleted")


class TestGoTBackendDocstrings:
    """Tests for method documentation."""

    def test_create_task_documented(self):
        """create_task has docstring."""
        assert GoTBackend.create_task.__doc__ is not None
        assert "Create a new task" in GoTBackend.create_task.__doc__

    def test_get_task_documented(self):
        """get_task has docstring."""
        assert GoTBackend.get_task.__doc__ is not None
        assert "Get a task by ID" in GoTBackend.get_task.__doc__

    def test_list_tasks_documented(self):
        """list_tasks has docstring."""
        assert GoTBackend.list_tasks.__doc__ is not None
        assert "List tasks" in GoTBackend.list_tasks.__doc__

    def test_query_documented(self):
        """query has docstring with examples."""
        assert GoTBackend.query.__doc__ is not None
        assert "what blocks" in GoTBackend.query.__doc__
        assert "what depends on" in GoTBackend.query.__doc__
        assert "path from" in GoTBackend.query.__doc__


class TestGoTBackendMethodCount:
    """Tests for protocol completeness."""

    def test_all_methods_defined(self):
        """Protocol defines expected number of methods."""
        # Get all public methods (excluding dunder methods)
        methods = [m for m in dir(GoTBackend)
                   if not m.startswith('_') and callable(getattr(GoTBackend, m))]

        # Should have at least 22 methods based on the protocol definition
        assert len(methods) >= 22, f"Expected at least 22 methods, got {len(methods)}: {methods}"

    def test_method_categories(self):
        """Protocol methods cover all expected categories."""
        # CRUD
        crud_methods = {'create_task', 'get_task', 'list_tasks', 'update_task', 'delete_task'}

        # State transitions
        state_methods = {'start_task', 'complete_task', 'block_task'}

        # Relationships
        relationship_methods = {'add_dependency', 'add_blocks', 'get_blockers',
                               'get_dependents', 'get_task_dependencies'}

        # Query/Analytics
        query_methods = {'get_stats', 'validate', 'get_blocked_tasks', 'get_active_tasks',
                        'what_blocks', 'what_depends_on', 'get_all_relationships'}

        # Persistence
        persistence_methods = {'sync_to_git', 'export_graph'}

        # Query language
        query_language_methods = {'query'}

        all_expected = (crud_methods | state_methods | relationship_methods |
                       query_methods | persistence_methods | query_language_methods)

        # Verify all expected methods exist
        for method in all_expected:
            assert hasattr(GoTBackend, method), f"Missing method: {method}"
