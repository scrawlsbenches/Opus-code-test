"""
Unit tests for cortical/got/cli/edge.py

Tests the edge CLI command handlers for:
- Adding edges between entities
- Listing edges with filters
- Showing edge types
- Getting edges for specific entities
"""

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from cortical.got.cli.edge import (
    VALID_EDGE_TYPES,
    cmd_edge_add,
    cmd_edge_list,
    cmd_edge_types,
    cmd_edge_for,
    setup_edge_parser,
    handle_edge_command,
)


class TestValidEdgeTypes:
    """Test the VALID_EDGE_TYPES constant."""

    def test_contains_semantic_edges(self):
        """Semantic edge types should be present."""
        semantic = ["REQUIRES", "ENABLES", "CONFLICTS", "SUPPORTS", "REFUTES",
                    "SIMILAR", "CONTAINS", "CONTRADICTS"]
        for edge_type in semantic:
            assert edge_type in VALID_EDGE_TYPES

    def test_contains_temporal_edges(self):
        """Temporal edge types should be present."""
        temporal = ["PRECEDES", "TRIGGERS", "BLOCKS"]
        for edge_type in temporal:
            assert edge_type in VALID_EDGE_TYPES

    def test_contains_epistemic_edges(self):
        """Epistemic edge types should be present."""
        epistemic = ["ANSWERS", "RAISES", "EXPLORES", "OBSERVES", "SUGGESTS"]
        for edge_type in epistemic:
            assert edge_type in VALID_EDGE_TYPES

    def test_contains_practical_edges(self):
        """Practical edge types should be present."""
        practical = ["IMPLEMENTS", "TESTS", "DEPENDS_ON", "REFINES",
                     "MOTIVATES", "JUSTIFIES"]
        for edge_type in practical:
            assert edge_type in VALID_EDGE_TYPES

    def test_contains_structural_edges(self):
        """Structural edge types should be present."""
        structural = ["HAS_OPTION", "HAS_ASPECT", "PART_OF"]
        for edge_type in structural:
            assert edge_type in VALID_EDGE_TYPES


class TestCmdEdgeAdd:
    """Tests for cmd_edge_add command handler."""

    def test_add_edge_success(self, capsys):
        """Successfully adding an edge returns 0."""
        args = SimpleNamespace(
            source_id="T-001",
            target_id="T-002",
            edge_type="DEPENDS_ON",
            weight=1.0,
        )
        manager = MagicMock()
        manager.add_edge.return_value = MagicMock()  # Truthy edge object

        result = cmd_edge_add(args, manager)

        assert result == 0
        manager.add_edge.assert_called_once_with(
            source_id="T-001",
            target_id="T-002",
            edge_type="DEPENDS_ON",
            weight=1.0,
            reason="",
        )
        manager.save.assert_called_once()
        output = capsys.readouterr().out
        assert "Created edge" in output
        assert "DEPENDS_ON" in output

    def test_add_edge_with_custom_weight(self, capsys):
        """Adding edge with custom weight works."""
        args = SimpleNamespace(
            source_id="T-001",
            target_id="T-002",
            edge_type="blocks",  # lowercase to test normalization
            weight=0.5,
        )
        manager = MagicMock()
        manager.add_edge.return_value = MagicMock()

        result = cmd_edge_add(args, manager)

        assert result == 0
        manager.add_edge.assert_called_once_with(
            source_id="T-001",
            target_id="T-002",
            edge_type="BLOCKS",  # Should be uppercased
            weight=0.5,
            reason="",
        )

    def test_add_edge_invalid_type(self, capsys):
        """Invalid edge type returns error code 1."""
        args = SimpleNamespace(
            source_id="T-001",
            target_id="T-002",
            edge_type="INVALID_TYPE",
        )
        manager = MagicMock()

        result = cmd_edge_add(args, manager)

        assert result == 1
        manager.add_edge.assert_not_called()
        output = capsys.readouterr().out
        assert "Invalid edge type" in output
        assert "Valid types" in output

    def test_add_edge_manager_returns_none(self, capsys):
        """When manager returns None, edge creation failed."""
        args = SimpleNamespace(
            source_id="T-001",
            target_id="T-002",
            edge_type="DEPENDS_ON",
            weight=1.0,
        )
        manager = MagicMock()
        manager.add_edge.return_value = None  # Failed

        result = cmd_edge_add(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Failed to create edge" in output

    def test_add_edge_manager_exception(self, capsys):
        """Exception during edge creation is handled."""
        args = SimpleNamespace(
            source_id="T-001",
            target_id="T-002",
            edge_type="DEPENDS_ON",
            weight=1.0,
        )
        manager = MagicMock()
        manager.add_edge.side_effect = Exception("Database error")

        result = cmd_edge_add(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Error creating edge" in output
        assert "Database error" in output


class TestCmdEdgeList:
    """Tests for cmd_edge_list command handler."""

    def test_list_edges_empty(self, capsys):
        """Listing edges when none exist."""
        args = SimpleNamespace()
        manager = MagicMock()
        manager.list_edges.return_value = []

        result = cmd_edge_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "No edges found" in output

    def test_list_edges_success(self, capsys):
        """Successfully listing edges."""
        edge1 = MagicMock()
        edge1.source_id = "T-001"
        edge1.target_id = "T-002"
        edge1.edge_type = "DEPENDS_ON"

        edge2 = MagicMock()
        edge2.source_id = "T-002"
        edge2.target_id = "T-003"
        edge2.edge_type = "BLOCKS"

        args = SimpleNamespace()
        manager = MagicMock()
        manager.list_edges.return_value = [edge1, edge2]

        result = cmd_edge_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Total: 2 edge(s)" in output
        assert "DEPENDS_ON" in output
        assert "BLOCKS" in output

    def test_list_edges_filter_by_type(self, capsys):
        """Filter edges by type."""
        edge1 = MagicMock()
        edge1.source_id = "T-001"
        edge1.target_id = "T-002"
        edge1.edge_type = "DEPENDS_ON"

        edge2 = MagicMock()
        edge2.source_id = "T-002"
        edge2.target_id = "T-003"
        edge2.edge_type = "BLOCKS"

        args = SimpleNamespace(type="depends_on")
        manager = MagicMock()
        manager.list_edges.return_value = [edge1, edge2]

        result = cmd_edge_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Total: 1 edge(s)" in output
        assert "DEPENDS_ON" in output

    def test_list_edges_filter_by_source(self, capsys):
        """Filter edges by source ID."""
        edge1 = MagicMock()
        edge1.source_id = "T-001"
        edge1.target_id = "T-002"
        edge1.edge_type = "DEPENDS_ON"

        edge2 = MagicMock()
        edge2.source_id = "T-002"
        edge2.target_id = "T-003"
        edge2.edge_type = "BLOCKS"

        args = SimpleNamespace(type=None, source="T-001", target=None)
        manager = MagicMock()
        manager.list_edges.return_value = [edge1, edge2]

        result = cmd_edge_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Total: 1 edge(s)" in output

    def test_list_edges_filter_by_target(self, capsys):
        """Filter edges by target ID."""
        edge1 = MagicMock()
        edge1.source_id = "T-001"
        edge1.target_id = "T-002"
        edge1.edge_type = "DEPENDS_ON"

        args = SimpleNamespace(type=None, source=None, target="T-002")
        manager = MagicMock()
        manager.list_edges.return_value = [edge1]

        result = cmd_edge_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Total: 1 edge(s)" in output

    def test_list_edges_no_matches_after_filter(self, capsys):
        """Filter returns no matches."""
        edge1 = MagicMock()
        edge1.source_id = "T-001"
        edge1.target_id = "T-002"
        edge1.edge_type = "DEPENDS_ON"

        args = SimpleNamespace(type="BLOCKS", source=None, target=None)
        manager = MagicMock()
        manager.list_edges.return_value = [edge1]

        result = cmd_edge_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "No edges match the filter criteria" in output

    def test_list_edges_with_from_to_attributes(self, capsys):
        """Handle edges with from_id/to_id instead of source_id/target_id."""
        edge1 = MagicMock(spec=[])  # No attributes by default
        edge1.from_id = "T-001"
        edge1.to_id = "T-002"
        edge1.edge_type = "DEPENDS_ON"
        # Remove source_id/target_id to force fallback
        del edge1.source_id
        del edge1.target_id

        args = SimpleNamespace()
        manager = MagicMock()
        manager.list_edges.return_value = [edge1]

        result = cmd_edge_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Total: 1 edge(s)" in output

    def test_list_edges_exception(self, capsys):
        """Exception during edge listing is handled."""
        args = SimpleNamespace()
        manager = MagicMock()
        manager.list_edges.side_effect = Exception("Connection error")

        result = cmd_edge_list(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Error listing edges" in output

    def test_list_edges_long_ids_truncated(self, capsys):
        """Long entity IDs are truncated for display."""
        long_id = "T-" + "a" * 50  # Very long ID
        edge1 = MagicMock()
        edge1.source_id = long_id
        edge1.target_id = "T-002"
        edge1.edge_type = "DEPENDS_ON"

        args = SimpleNamespace()
        manager = MagicMock()
        manager.list_edges.return_value = [edge1]

        result = cmd_edge_list(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "..." in output  # Truncation indicator


class TestCmdEdgeTypes:
    """Tests for cmd_edge_types command handler."""

    def test_edge_types_displays_all_categories(self, capsys):
        """All edge type categories are displayed."""
        args = SimpleNamespace()
        manager = MagicMock()

        result = cmd_edge_types(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Semantic" in output
        assert "Temporal" in output
        assert "Epistemic" in output
        assert "Practical" in output
        assert "Structural" in output

    def test_edge_types_displays_descriptions(self, capsys):
        """Edge type descriptions are displayed."""
        args = SimpleNamespace()
        manager = MagicMock()

        result = cmd_edge_types(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "DEPENDS_ON" in output
        assert "BLOCKS" in output
        assert "REQUIRES" in output


class TestCmdEdgeFor:
    """Tests for cmd_edge_for command handler."""

    def test_edge_for_no_edges(self, capsys):
        """Entity with no edges."""
        args = SimpleNamespace(entity_id="T-001")
        manager = MagicMock()
        manager.get_edges_for_task.return_value = ([], [])

        result = cmd_edge_for(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "No edges found for entity" in output

    def test_edge_for_outgoing_only(self, capsys):
        """Entity with only outgoing edges."""
        edge1 = MagicMock()
        edge1.target_id = "T-002"
        edge1.edge_type = "DEPENDS_ON"

        args = SimpleNamespace(entity_id="T-001")
        manager = MagicMock()
        manager.get_edges_for_task.return_value = ([edge1], [])

        result = cmd_edge_for(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Outgoing (1)" in output
        assert "DEPENDS_ON" in output
        assert "T-002" in output

    def test_edge_for_incoming_only(self, capsys):
        """Entity with only incoming edges."""
        edge1 = MagicMock()
        edge1.source_id = "T-000"
        edge1.edge_type = "BLOCKS"

        args = SimpleNamespace(entity_id="T-001")
        manager = MagicMock()
        manager.get_edges_for_task.return_value = ([], [edge1])

        result = cmd_edge_for(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Incoming (1)" in output
        assert "BLOCKS" in output
        assert "T-000" in output

    def test_edge_for_both_directions(self, capsys):
        """Entity with both incoming and outgoing edges."""
        out_edge = MagicMock()
        out_edge.target_id = "T-002"
        out_edge.edge_type = "DEPENDS_ON"

        in_edge = MagicMock()
        in_edge.source_id = "T-000"
        in_edge.edge_type = "BLOCKS"

        args = SimpleNamespace(entity_id="T-001")
        manager = MagicMock()
        manager.get_edges_for_task.return_value = ([out_edge], [in_edge])

        result = cmd_edge_for(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Outgoing (1)" in output
        assert "Incoming (1)" in output

    def test_edge_for_fallback_to_list_edges(self, capsys):
        """Fallback to filtering list_edges when get_edges_for_task fails."""
        edge1 = MagicMock()
        edge1.source_id = "T-001"
        edge1.target_id = "T-002"
        edge1.edge_type = "DEPENDS_ON"

        args = SimpleNamespace(entity_id="T-001")
        manager = MagicMock()
        manager.get_edges_for_task.side_effect = Exception("Not implemented")
        manager.list_edges.return_value = [edge1]

        result = cmd_edge_for(args, manager)

        assert result == 0
        output = capsys.readouterr().out
        assert "Outgoing (1)" in output

    def test_edge_for_exception(self, capsys):
        """Exception during edge retrieval is handled."""
        args = SimpleNamespace(entity_id="T-001")
        manager = MagicMock()
        manager.get_edges_for_task.side_effect = Exception("First error")
        manager.list_edges.side_effect = Exception("Second error")

        result = cmd_edge_for(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Error getting edges" in output


class TestSetupEdgeParser:
    """Tests for setup_edge_parser function."""

    def test_setup_creates_edge_subparser(self):
        """Edge subparser is created correctly."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_edge_parser(subparsers)

        # Parse 'edge add' command
        args = parser.parse_args(['edge', 'add', 'T-001', 'T-002', 'DEPENDS_ON'])
        assert args.source_id == 'T-001'
        assert args.target_id == 'T-002'
        assert args.edge_type == 'DEPENDS_ON'

    def test_setup_edge_add_with_weight(self):
        """Edge add parser handles weight argument."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_edge_parser(subparsers)

        args = parser.parse_args(['edge', 'add', 'T-001', 'T-002', 'BLOCKS', '--weight', '0.5'])
        assert args.weight == 0.5

    def test_setup_edge_list_with_filters(self):
        """Edge list parser handles filter arguments."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_edge_parser(subparsers)

        args = parser.parse_args(['edge', 'list', '--type', 'BLOCKS', '--source', 'T-001'])
        assert args.type == 'BLOCKS'
        assert args.source == 'T-001'

    def test_setup_edge_for(self):
        """Edge for parser handles entity_id."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_edge_parser(subparsers)

        args = parser.parse_args(['edge', 'for', 'T-001'])
        assert args.entity_id == 'T-001'


class TestHandleEdgeCommand:
    """Tests for handle_edge_command routing function."""

    def test_handle_add_command(self):
        """Routes to add handler."""
        args = SimpleNamespace(
            edge_command="add",
            source_id="T-001",
            target_id="T-002",
            edge_type="DEPENDS_ON",
            weight=1.0,
        )
        manager = MagicMock()
        manager.add_edge.return_value = MagicMock()

        result = handle_edge_command(args, manager)

        assert result == 0

    def test_handle_list_command(self):
        """Routes to list handler."""
        args = SimpleNamespace(edge_command="list")
        manager = MagicMock()
        manager.list_edges.return_value = []

        result = handle_edge_command(args, manager)

        assert result == 0

    def test_handle_types_command(self):
        """Routes to types handler."""
        args = SimpleNamespace(edge_command="types")
        manager = MagicMock()

        result = handle_edge_command(args, manager)

        assert result == 0

    def test_handle_for_command(self):
        """Routes to for handler."""
        args = SimpleNamespace(edge_command="for", entity_id="T-001")
        manager = MagicMock()
        manager.get_edges_for_task.return_value = ([], [])

        result = handle_edge_command(args, manager)

        assert result == 0

    def test_handle_no_subcommand(self, capsys):
        """No subcommand returns error."""
        args = SimpleNamespace()  # No edge_command
        manager = MagicMock()

        result = handle_edge_command(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "No edge subcommand specified" in output

    def test_handle_none_subcommand(self, capsys):
        """None subcommand returns error."""
        args = SimpleNamespace(edge_command=None)
        manager = MagicMock()

        result = handle_edge_command(args, manager)

        assert result == 1

    def test_handle_unknown_subcommand(self, capsys):
        """Unknown subcommand returns error."""
        args = SimpleNamespace(edge_command="unknown")
        manager = MagicMock()

        result = handle_edge_command(args, manager)

        assert result == 1
        output = capsys.readouterr().out
        assert "Unknown edge subcommand" in output
