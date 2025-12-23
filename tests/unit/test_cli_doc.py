"""
Unit tests for cortical.got.cli.doc module.

Tests use mocked GoTManager to avoid file system operations.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
from argparse import Namespace
from datetime import datetime, timezone

from cortical.got.cli.doc import (
    detect_doc_type,
    extract_title_from_content,
    extract_tags_from_content,
    cmd_doc_scan,
    cmd_doc_list,
    cmd_doc_show,
    cmd_doc_link,
    cmd_doc_tasks,
    cmd_doc_docs,
    setup_doc_parser,
    handle_doc_command,
)
from cortical.got.types import Document, Task


class TestDetectDocType(unittest.TestCase):
    """Test document type detection."""

    def test_architecture_detection(self):
        """Test detection of architecture documents."""
        self.assertEqual(
            detect_doc_type("docs/architecture.md"),
            "architecture"
        )
        self.assertEqual(
            detect_doc_type("path/to/design-doc.md"),
            "architecture"
        )

    def test_api_detection(self):
        """Test detection of API documents."""
        self.assertEqual(
            detect_doc_type("docs/api-reference.md"),
            "api"
        )
        self.assertEqual(
            detect_doc_type("path/to/interface.md"),
            "api"
        )

    def test_guide_detection(self):
        """Test detection of guide documents."""
        self.assertEqual(
            detect_doc_type("docs/user-guide.md"),
            "guide"
        )
        self.assertEqual(
            detect_doc_type("path/to/tutorial.md"),
            "guide"
        )
        self.assertEqual(
            detect_doc_type("path/to/how-to-setup.md"),
            "guide"
        )

    def test_memory_detection(self):
        """Test detection of memory documents."""
        self.assertEqual(
            detect_doc_type("samples/memories/session-log.md"),
            "memory"
        )
        # "knowledge-transfer" is in memory patterns (checked first)
        self.assertEqual(
            detect_doc_type("docs/knowledge-transfer-2025.md"),
            "memory"
        )

    def test_knowledge_transfer_with_handoff(self):
        """Test knowledge-transfer detection with handoff keyword."""
        # "handoff" is only in knowledge-transfer patterns
        self.assertEqual(
            detect_doc_type("docs/handoff-notes.md"),
            "knowledge-transfer"
        )

    def test_decision_detection(self):
        """Test detection of decision documents."""
        self.assertEqual(
            detect_doc_type("docs/adr-001-decision.md"),
            "decision"
        )
        self.assertEqual(
            detect_doc_type("path/to/rationale.md"),
            "decision"
        )

    def test_research_detection(self):
        """Test detection of research documents."""
        self.assertEqual(
            detect_doc_type("docs/research-notes.md"),
            "research"
        )
        self.assertEqual(
            detect_doc_type("path/to/investigation.md"),
            "research"
        )

    def test_title_based_detection(self):
        """Test detection using title when path is generic."""
        self.assertEqual(
            detect_doc_type("misc.md", "Architecture Overview"),
            "architecture"
        )
        self.assertEqual(
            detect_doc_type("doc.md", "API Reference"),
            "api"
        )

    def test_unknown_type_returns_general(self):
        """Test that unknown types return 'general'."""
        self.assertEqual(
            detect_doc_type("random/file.md"),
            "general"
        )


class TestExtractTitleFromContent(unittest.TestCase):
    """Test title extraction from markdown content."""

    def test_extract_h1_title(self):
        """Test extraction of H1 title."""
        content = """# Main Title

Some content here.
"""
        self.assertEqual(
            extract_title_from_content(content),
            "Main Title"
        )

    def test_extract_h2_title(self):
        """Test extraction of H2 title when no H1 exists."""
        content = """## Secondary Title

Some content here.
"""
        self.assertEqual(
            extract_title_from_content(content),
            "Secondary Title"
        )

    def test_h1_takes_precedence(self):
        """Test that H1 takes precedence over H2."""
        content = """# Main Title
## Secondary Title

Some content here.
"""
        self.assertEqual(
            extract_title_from_content(content),
            "Main Title"
        )

    def test_no_heading_returns_empty(self):
        """Test that content without headings returns empty string."""
        content = """Just some plain text without headings."""
        self.assertEqual(
            extract_title_from_content(content),
            ""
        )

    def test_whitespace_trimmed(self):
        """Test that whitespace is trimmed from extracted title."""
        content = """#   Spaced Title

Content.
"""
        self.assertEqual(
            extract_title_from_content(content),
            "Spaced Title"
        )


class TestExtractTagsFromContent(unittest.TestCase):
    """Test tag extraction from markdown content."""

    def test_extract_backtick_tags(self):
        """Test extraction of backtick-quoted tags."""
        content = """# Title

**Tags:** `tag1`, `tag2`, `tag3`

Content.
"""
        self.assertEqual(
            extract_tags_from_content(content),
            ["tag1", "tag2", "tag3"]
        )

    def test_extract_plain_tags(self):
        """Test extraction of plain comma-separated tags."""
        content = """# Title

Tags: alpha, beta, gamma

Content.
"""
        tags = extract_tags_from_content(content)
        # Trim whitespace from each tag
        tags = [t.strip() for t in tags]
        self.assertEqual(tags, ["alpha", "beta", "gamma"])

    def test_tags_case_insensitive(self):
        """Test that 'Tags:' is case-insensitive."""
        content = """# Title

tags: `one`, `two`

Content.
"""
        self.assertEqual(
            extract_tags_from_content(content),
            ["one", "two"]
        )

    def test_no_tags_returns_empty_list(self):
        """Test that content without tags returns empty list."""
        content = """# Title

No tags here.
"""
        self.assertEqual(
            extract_tags_from_content(content),
            []
        )

    def test_bold_tags_marker(self):
        """Test extraction with bold Tags marker."""
        content = """# Title

**Tags:** `important`, `urgent`
"""
        self.assertEqual(
            extract_tags_from_content(content),
            ["important", "urgent"]
        )


class TestCmdDocList(unittest.TestCase):
    """Test cmd_doc_list command handler."""

    def test_list_empty(self):
        """Test listing when no documents exist."""
        mock_manager = Mock()
        mock_manager.got_dir = "/fake/.got"

        args = Namespace(doc_type=None, tag=None, stale=False)

        with patch('cortical.got.cli.doc.list_documents', return_value=[]) as mock_list:
            with patch('builtins.print') as mock_print:
                result = cmd_doc_list(args, mock_manager)

        self.assertEqual(result, 0)
        mock_list.assert_called_once_with(
            mock_manager.got_dir,
            doc_type=None,
            tag=None,
            stale_only=False,
        )
        mock_print.assert_called_with("No documents found.")

    def test_list_with_documents(self):
        """Test listing with sample documents."""
        mock_manager = Mock()
        mock_manager.got_dir = "/fake/.got"

        # Create mock documents
        doc1 = Document(
            id="DOC-001",
            path="docs/arch.md",
            title="Architecture",
            doc_type="architecture",
            tags=["core"],
            is_stale=False,
        )
        doc2 = Document(
            id="DOC-002",
            path="docs/api.md",
            title="API Reference",
            doc_type="api",
            tags=["api", "reference"],
            is_stale=True,
        )

        args = Namespace(doc_type=None, tag=None, stale=False)

        with patch('cortical.got.cli.doc.list_documents', return_value=[doc1, doc2]):
            with patch('builtins.print') as mock_print:
                result = cmd_doc_list(args, mock_manager)

        self.assertEqual(result, 0)
        # Check that it printed document info
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("DOC-001" in str(call) for call in print_calls))
        self.assertTrue(any("DOC-002" in str(call) for call in print_calls))

    def test_list_with_filters(self):
        """Test listing with type and tag filters."""
        mock_manager = Mock()
        mock_manager.got_dir = "/fake/.got"

        args = Namespace(doc_type="architecture", tag="core", stale=True)

        with patch('cortical.got.cli.doc.list_documents', return_value=[]) as mock_list:
            with patch('builtins.print'):
                result = cmd_doc_list(args, mock_manager)

        self.assertEqual(result, 0)
        mock_list.assert_called_once_with(
            mock_manager.got_dir,
            doc_type="architecture",
            tag="core",
            stale_only=True,
        )


class TestCmdDocShow(unittest.TestCase):
    """Test cmd_doc_show command handler."""

    def test_show_document_not_found(self):
        """Test showing document that doesn't exist."""
        mock_manager = Mock()
        mock_manager.got_dir = "/fake/.got"

        # Mock show_document to return None
        with patch('cortical.got.cli.doc.show_document', return_value=None):
            args = Namespace(doc_id="DOC-999")

            with patch('builtins.print') as mock_print:
                result = cmd_doc_show(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_with("Document not found: DOC-999")

    def test_show_document_success(self):
        """Test showing existing document."""
        mock_manager = Mock()
        mock_manager.got_dir = "/fake/.got"

        # Create mock document
        doc = Document(
            id="DOC-001",
            path="docs/arch.md",
            title="Architecture",
            doc_type="architecture",
            category="docs",
            tags=["core", "design"],
            line_count=100,
            word_count=500,
            content_hash="abc123",
            last_file_modified="2025-01-01T00:00:00Z",
            last_verified="2025-01-01T00:00:00Z",
            is_stale=False,
            version=1,
        )

        with patch('cortical.got.cli.doc.show_document', return_value=doc):
            args = Namespace(doc_id="DOC-001")

            with patch('builtins.print') as mock_print:
                result = cmd_doc_show(args, mock_manager)

        self.assertEqual(result, 0)
        # Check that it printed document details
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("DOC-001" in str(call) for call in print_calls))
        self.assertTrue(any("Architecture" in str(call) for call in print_calls))


class TestCmdDocLink(unittest.TestCase):
    """Test cmd_doc_link command handler."""

    def test_link_success(self):
        """Test successful document-task linking."""
        mock_manager = Mock()
        mock_manager.got_dir = "/fake/.got"

        with patch('cortical.got.cli.doc.link_document_to_task', return_value=True):
            args = Namespace(
                doc_id="DOC-001",
                task_id="T-001",
                edge_type="DOCUMENTED_BY"
            )

            with patch('builtins.print') as mock_print:
                result = cmd_doc_link(args, mock_manager)

        self.assertEqual(result, 0)
        mock_print.assert_called_with("Linked: DOC-001 --DOCUMENTED_BY--> T-001")

    def test_link_failure(self):
        """Test failed document-task linking."""
        mock_manager = Mock()
        mock_manager.got_dir = "/fake/.got"

        with patch('cortical.got.cli.doc.link_document_to_task', return_value=False):
            args = Namespace(
                doc_id="DOC-999",
                task_id="T-999",
                edge_type="DOCUMENTED_BY"
            )

            with patch('builtins.print'):
                result = cmd_doc_link(args, mock_manager)

        self.assertEqual(result, 1)


class TestCmdDocTasks(unittest.TestCase):
    """Test cmd_doc_tasks command handler."""

    def test_no_tasks_linked(self):
        """Test showing tasks when none are linked."""
        mock_manager = Mock()
        mock_manager.got_dir = "/fake/.got"

        with patch('cortical.got.cli.doc.get_tasks_for_document', return_value=[]):
            args = Namespace(doc_id="DOC-001")

            with patch('builtins.print') as mock_print:
                result = cmd_doc_tasks(args, mock_manager)

        self.assertEqual(result, 0)
        mock_print.assert_called_with("No tasks linked to document: DOC-001")

    def test_tasks_linked(self):
        """Test showing linked tasks."""
        mock_manager = Mock()
        mock_manager.got_dir = "/fake/.got"

        # Create mock tasks
        task1 = Task(
            id="T-001",
            title="Implement feature",
            status="in_progress",
            priority="high",
        )
        task2 = Task(
            id="T-002",
            title="Write tests",
            status="pending",
            priority="medium",
        )

        with patch('cortical.got.cli.doc.get_tasks_for_document', return_value=[task1, task2]):
            args = Namespace(doc_id="DOC-001")

            with patch('builtins.print') as mock_print:
                result = cmd_doc_tasks(args, mock_manager)

        self.assertEqual(result, 0)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("T-001" in str(call) for call in print_calls))
        self.assertTrue(any("T-002" in str(call) for call in print_calls))


class TestCmdDocDocs(unittest.TestCase):
    """Test cmd_doc_docs command handler."""

    def test_no_documents_linked(self):
        """Test showing documents when none are linked."""
        mock_manager = Mock()
        mock_manager.got_dir = "/fake/.got"

        with patch('cortical.got.cli.doc.get_documents_for_task', return_value=[]):
            args = Namespace(task_id="T-001")

            with patch('builtins.print') as mock_print:
                result = cmd_doc_docs(args, mock_manager)

        self.assertEqual(result, 0)
        mock_print.assert_called_with("No documents linked to task: T-001")

    def test_documents_linked(self):
        """Test showing linked documents."""
        mock_manager = Mock()
        mock_manager.got_dir = "/fake/.got"

        # Create mock documents
        doc1 = Document(
            id="DOC-001",
            path="docs/arch.md",
            title="Architecture",
            doc_type="architecture",
        )
        doc2 = Document(
            id="DOC-002",
            path="docs/api.md",
            title="API Reference",
            doc_type="api",
        )

        with patch('cortical.got.cli.doc.get_documents_for_task', return_value=[doc1, doc2]):
            args = Namespace(task_id="T-001")

            with patch('builtins.print') as mock_print:
                result = cmd_doc_docs(args, mock_manager)

        self.assertEqual(result, 0)
        print_calls = [str(call) for call in mock_print.call_args_list]
        self.assertTrue(any("DOC-001" in str(call) for call in print_calls))
        self.assertTrue(any("DOC-002" in str(call) for call in print_calls))


class TestHandleDocCommand(unittest.TestCase):
    """Test handle_doc_command routing."""

    def test_no_subcommand(self):
        """Test error when no subcommand specified."""
        mock_manager = Mock()
        args = Namespace()  # No doc_command attribute

        with patch('builtins.print') as mock_print:
            result = handle_doc_command(args, mock_manager)

        self.assertEqual(result, 1)
        mock_print.assert_called_once()
        self.assertIn("No doc subcommand", str(mock_print.call_args))

    def test_route_to_scan(self):
        """Test routing to scan command."""
        mock_manager = Mock()
        args = Namespace(doc_command="scan", dirs=["docs"], dry_run=False, verbose=False)

        with patch('cortical.got.cli.doc.cmd_doc_scan', return_value=0) as mock_scan:
            result = handle_doc_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_scan.assert_called_once_with(args, mock_manager)

    def test_route_to_list(self):
        """Test routing to list command."""
        mock_manager = Mock()
        args = Namespace(doc_command="list", doc_type=None, tag=None, stale=False)

        with patch('cortical.got.cli.doc.cmd_doc_list', return_value=0) as mock_list:
            result = handle_doc_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_list.assert_called_once_with(args, mock_manager)

    def test_route_to_show(self):
        """Test routing to show command."""
        mock_manager = Mock()
        args = Namespace(doc_command="show", doc_id="DOC-001")

        with patch('cortical.got.cli.doc.cmd_doc_show', return_value=0) as mock_show:
            result = handle_doc_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_show.assert_called_once_with(args, mock_manager)

    def test_route_to_link(self):
        """Test routing to link command."""
        mock_manager = Mock()
        args = Namespace(
            doc_command="link",
            doc_id="DOC-001",
            task_id="T-001",
            edge_type="DOCUMENTED_BY"
        )

        with patch('cortical.got.cli.doc.cmd_doc_link', return_value=0) as mock_link:
            result = handle_doc_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_link.assert_called_once_with(args, mock_manager)

    def test_route_to_tasks(self):
        """Test routing to tasks command."""
        mock_manager = Mock()
        args = Namespace(doc_command="tasks", doc_id="DOC-001")

        with patch('cortical.got.cli.doc.cmd_doc_tasks', return_value=0) as mock_tasks:
            result = handle_doc_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_tasks.assert_called_once_with(args, mock_manager)

    def test_route_to_docs(self):
        """Test routing to docs command."""
        mock_manager = Mock()
        args = Namespace(doc_command="docs", task_id="T-001")

        with patch('cortical.got.cli.doc.cmd_doc_docs', return_value=0) as mock_docs:
            result = handle_doc_command(args, mock_manager)

        self.assertEqual(result, 0)
        mock_docs.assert_called_once_with(args, mock_manager)

    def test_unknown_command(self):
        """Test error for unknown subcommand."""
        mock_manager = Mock()
        args = Namespace(doc_command="invalid")

        with patch('builtins.print') as mock_print:
            result = handle_doc_command(args, mock_manager)

        self.assertEqual(result, 1)
        self.assertIn("Unknown doc subcommand", str(mock_print.call_args))


class TestSetupDocParser(unittest.TestCase):
    """Test setup_doc_parser function."""

    def test_parser_setup(self):
        """Test that parser is set up correctly."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")

        # Set up doc parser
        setup_doc_parser(subparsers)

        # Parse a scan command
        args = parser.parse_args(['doc', 'scan', '--dirs', 'docs', '--verbose'])
        self.assertEqual(args.command, 'doc')
        self.assertEqual(args.doc_command, 'scan')
        self.assertEqual(args.dirs, ['docs'])
        self.assertTrue(args.verbose)
        self.assertFalse(args.dry_run)

    def test_list_parser_setup(self):
        """Test list subcommand parser."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_doc_parser(subparsers)

        args = parser.parse_args(['doc', 'list', '--type', 'architecture', '--stale'])
        self.assertEqual(args.command, 'doc')
        self.assertEqual(args.doc_command, 'list')
        self.assertEqual(args.doc_type, 'architecture')
        self.assertTrue(args.stale)

    def test_show_parser_setup(self):
        """Test show subcommand parser."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_doc_parser(subparsers)

        args = parser.parse_args(['doc', 'show', 'DOC-001'])
        self.assertEqual(args.command, 'doc')
        self.assertEqual(args.doc_command, 'show')
        self.assertEqual(args.doc_id, 'DOC-001')

    def test_link_parser_setup(self):
        """Test link subcommand parser."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        setup_doc_parser(subparsers)

        args = parser.parse_args(['doc', 'link', 'DOC-001', 'T-001', '--type', 'PRODUCES'])
        self.assertEqual(args.command, 'doc')
        self.assertEqual(args.doc_command, 'link')
        self.assertEqual(args.doc_id, 'DOC-001')
        self.assertEqual(args.task_id, 'T-001')
        self.assertEqual(args.edge_type, 'PRODUCES')


if __name__ == '__main__':
    unittest.main()
