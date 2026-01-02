"""
Unit tests for cortical.got.cli.knowledge_transfer module.

Tests use mocked TransactionalGoTAdapter to avoid file system operations.
Covers markdown parsing, CLI command handlers, and argparse setup.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from argparse import Namespace, ArgumentParser
from pathlib import Path
import tempfile
import os

from cortical.got.cli.knowledge_transfer import (
    parse_markdown_file,
    cmd_kt_create,
    cmd_kt_append,
    cmd_kt_link,
    cmd_kt_list,
    cmd_kt_show,
    cmd_kt_import,
    cmd_kt_finalize,
    cmd_kt_history,
    setup_knowledge_transfer_parser,
    handle_knowledge_transfer_command,
)


class TestParseMarkdownFile(unittest.TestCase):
    """Test markdown parsing functionality."""

    def test_parse_basic_title(self):
        """Test extraction of title from # heading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# My Knowledge Transfer\n\nSome content.\n")
            f.flush()
            result = parse_markdown_file(Path(f.name))
            os.unlink(f.name)

        self.assertEqual(result['title'], 'My Knowledge Transfer')

    def test_parse_session_metadata(self):
        """Test extraction of session metadata from **Key:** lines."""
        content = """# Test Document

**Session Date:** 2025-12-29
**Session ID:** abc123
**Branch:** feature/test

## Summary

Some summary content.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            result = parse_markdown_file(Path(f.name))
            os.unlink(f.name)

        self.assertEqual(result['session_date'], '2025-12-29')
        self.assertEqual(result['session_id'], 'abc123')
        self.assertEqual(result['branch'], 'feature/test')

    def test_parse_sections(self):
        """Test extraction of content sections from ## headings."""
        content = """# Test Document

## Executive Summary

This is the summary.

## Technical Details

This is the technical content.

## Next Steps

These are next steps.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            result = parse_markdown_file(Path(f.name))
            os.unlink(f.name)

        self.assertIn('Technical Details', result['sections'])
        self.assertIn('Next Steps', result['sections'])
        self.assertIn('technical content', result['sections']['Technical Details'])

    def test_parse_code_references(self):
        """Test extraction of code references (file:line patterns)."""
        content = """# Test Document

See cortical/got/store.py:45 for implementation.
Also check tests/unit/test_got.py:123.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            result = parse_markdown_file(Path(f.name))
            os.unlink(f.name)

        self.assertIn('cortical/got/store.py:45', result['code_refs'])
        self.assertIn('tests/unit/test_got.py:123', result['code_refs'])

    def test_parse_generates_title_from_filename(self):
        """Test title generation from filename when no # heading."""
        content = "No title heading here.\n\nJust content."

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, prefix='my-session-notes_'
        ) as f:
            f.write(content)
            f.flush()
            result = parse_markdown_file(Path(f.name))
            os.unlink(f.name)

        # Should generate a title from filename
        self.assertTrue(len(result['title']) > 0)

    def test_parse_summary_section(self):
        """Test extraction of Executive Summary content."""
        content = """# Test

## Executive Summary

This is the executive summary content.
It spans multiple lines.

## Other Section

Other content.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()
            result = parse_markdown_file(Path(f.name))
            os.unlink(f.name)

        self.assertIn('executive summary content', result['summary'])
        self.assertIn('multiple lines', result['summary'])


class TestCmdKtCreate(unittest.TestCase):
    """Test kt create command handler."""

    def test_create_basic(self):
        """Test creating a knowledge transfer with basic fields."""
        manager = Mock()
        manager.create_knowledge_transfer.return_value = 'KT-20260101-120000'

        args = Namespace(
            title='Test KT',
            session='sess123',
            summary='A test summary',
            content='',
            sections=None,
            tags=['test', 'unit']
        )

        with patch('sys.stdout'):
            result = cmd_kt_create(args, manager)

        self.assertEqual(result, 0)
        manager.create_knowledge_transfer.assert_called_once_with(
            title='Test KT',
            session_id='sess123',
            summary='A test summary',
            sections={},
            tags=['test', 'unit']
        )

    def test_create_with_sections(self):
        """Test creating KT with sections specified."""
        manager = Mock()
        manager.create_knowledge_transfer.return_value = 'KT-20260101-120000'

        args = Namespace(
            title='Test KT',
            session='',
            summary='',
            content='',
            sections=['Technical:tech content', 'Notes:note content'],
            tags=[]
        )

        with patch('sys.stdout'):
            result = cmd_kt_create(args, manager)

        self.assertEqual(result, 0)
        call_kwargs = manager.create_knowledge_transfer.call_args[1]
        self.assertEqual(call_kwargs['sections']['Technical'], 'tech content')
        self.assertEqual(call_kwargs['sections']['Notes'], 'note content')


class TestCmdKtAppend(unittest.TestCase):
    """Test kt append command handler."""

    def test_append_success(self):
        """Test successfully appending a section."""
        manager = Mock()
        manager.append_kt_section.return_value = True

        args = Namespace(
            kt_id='KT-20260101-120000',
            section_heading='New Findings',
            content='Found something interesting.'
        )

        with patch('sys.stdout'):
            result = cmd_kt_append(args, manager)

        self.assertEqual(result, 0)
        manager.append_kt_section.assert_called_once_with(
            'KT-20260101-120000',
            'New Findings',
            'Found something interesting.'
        )

    def test_append_failure(self):
        """Test append failure when KT not found."""
        manager = Mock()
        manager.append_kt_section.return_value = False

        args = Namespace(
            kt_id='KT-nonexistent',
            section_heading='Section',
            content='Content'
        )

        with patch('sys.stdout'):
            result = cmd_kt_append(args, manager)

        self.assertEqual(result, 1)


class TestCmdKtLink(unittest.TestCase):
    """Test kt link command handler."""

    def test_link_handoff(self):
        """Test linking KT to a handoff."""
        manager = Mock()
        manager.link_kt_handoff.return_value = True

        args = Namespace(
            kt_id='KT-123',
            handoff='H-456',
            task=None,
            decision=None
        )

        with patch('sys.stdout'):
            result = cmd_kt_link(args, manager)

        self.assertEqual(result, 0)
        manager.link_kt_handoff.assert_called_once_with('KT-123', 'H-456')

    def test_link_task(self):
        """Test linking KT to a task."""
        manager = Mock()
        manager.link_kt_task.return_value = True

        args = Namespace(
            kt_id='KT-123',
            handoff=None,
            task='T-789',
            decision=None
        )

        with patch('sys.stdout'):
            result = cmd_kt_link(args, manager)

        self.assertEqual(result, 0)
        manager.link_kt_task.assert_called_once_with('KT-123', 'T-789')

    def test_link_decision(self):
        """Test linking KT to a decision."""
        manager = Mock()
        manager.link_kt_decision.return_value = True

        args = Namespace(
            kt_id='KT-123',
            handoff=None,
            task=None,
            decision='D-101'
        )

        with patch('sys.stdout'):
            result = cmd_kt_link(args, manager)

        self.assertEqual(result, 0)
        manager.link_kt_decision.assert_called_once_with('KT-123', 'D-101')

    def test_link_no_target(self):
        """Test link failure when no target specified."""
        manager = Mock()

        args = Namespace(
            kt_id='KT-123',
            handoff=None,
            task=None,
            decision=None
        )

        with patch('sys.stdout'):
            result = cmd_kt_link(args, manager)

        self.assertEqual(result, 1)


class TestCmdKtList(unittest.TestCase):
    """Test kt list command handler."""

    def test_list_empty(self):
        """Test listing when no KTs exist."""
        manager = Mock()
        manager.list_knowledge_transfers.return_value = []

        args = Namespace(status=None, tags=None, limit=None)

        with patch('sys.stdout'):
            result = cmd_kt_list(args, manager)

        self.assertEqual(result, 0)

    def test_list_with_results(self):
        """Test listing with results."""
        manager = Mock()
        manager.list_knowledge_transfers.return_value = [
            {'id': 'KT-1', 'title': 'First KT', 'status': 'draft', 'tags': ['test']},
            {'id': 'KT-2', 'title': 'Second KT', 'status': 'published', 'tags': []},
        ]

        args = Namespace(status=None, tags=None, limit=None)

        with patch('sys.stdout'):
            result = cmd_kt_list(args, manager)

        self.assertEqual(result, 0)
        manager.list_knowledge_transfers.assert_called_once_with(
            status=None, tags=None
        )

    def test_list_with_limit(self):
        """Test listing with limit applied."""
        manager = Mock()
        manager.list_knowledge_transfers.return_value = [
            {'id': 'KT-1', 'title': 'First KT', 'status': 'draft'},
            {'id': 'KT-2', 'title': 'Second KT', 'status': 'published'},
            {'id': 'KT-3', 'title': 'Third KT', 'status': 'published'},
        ]

        args = Namespace(status=None, tags=None, limit=2)

        with patch('sys.stdout'):
            result = cmd_kt_list(args, manager)

        self.assertEqual(result, 0)


class TestCmdKtShow(unittest.TestCase):
    """Test kt show command handler."""

    def test_show_not_found(self):
        """Test show when KT not found."""
        manager = Mock()
        manager.get_knowledge_transfer.return_value = None

        args = Namespace(kt_id='KT-nonexistent')

        with patch('sys.stdout'):
            result = cmd_kt_show(args, manager)

        self.assertEqual(result, 1)

    def test_show_full_details(self):
        """Test showing KT with all details."""
        manager = Mock()
        manager.get_knowledge_transfer.return_value = {
            'id': 'KT-123',
            'title': 'Test KT',
            'status': 'published',
            'session_id': 'sess456',
            'session_date': '2025-12-30',
            'tags': ['architecture', 'testing'],
            'summary': 'This is a summary.',
            'sections': {'Technical': 'Tech content'},
            'code_refs': ['file.py:10'],
            'related_handoffs': ['H-1'],
            'related_tasks': ['T-1'],
            'related_decisions': ['D-1'],
            'created_at': '2025-12-30T10:00:00Z',
            'modified_at': '2025-12-30T11:00:00Z',
        }

        args = Namespace(kt_id='KT-123')

        with patch('sys.stdout'):
            result = cmd_kt_show(args, manager)

        self.assertEqual(result, 0)


class TestCmdKtImport(unittest.TestCase):
    """Test kt import command handler."""

    def test_import_file_not_found(self):
        """Test import failure when file doesn't exist."""
        manager = Mock()

        args = Namespace(file='/nonexistent/path.md', tags=None)

        with patch('sys.stdout'):
            result = cmd_kt_import(args, manager)

        self.assertEqual(result, 1)

    def test_import_success(self):
        """Test successful import from markdown file."""
        content = """# Test Knowledge Transfer

**Session Date:** 2025-12-30
**Session ID:** test123

## Executive Summary

This is the summary.

## Technical Details

The implementation details.
"""
        manager = Mock()
        manager.create_knowledge_transfer.return_value = 'KT-20260101-120000'

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            f.flush()

            args = Namespace(file=f.name, tags=['imported'])

            with patch('sys.stdout'):
                result = cmd_kt_import(args, manager)

            os.unlink(f.name)

        self.assertEqual(result, 0)
        manager.create_knowledge_transfer.assert_called_once()


class TestCmdKtFinalize(unittest.TestCase):
    """Test kt finalize command handler."""

    def test_finalize_success(self):
        """Test successful finalization."""
        manager = Mock()
        manager.finalize_knowledge_transfer.return_value = True

        args = Namespace(
            kt_id='KT-123',
            handoff_to=None,
            instructions=''
        )

        with patch('sys.stdout'):
            result = cmd_kt_finalize(args, manager)

        self.assertEqual(result, 0)
        manager.finalize_knowledge_transfer.assert_called_once_with(
            kt_id='KT-123',
            handoff_to=None,
            instructions=''
        )

    def test_finalize_with_handoff(self):
        """Test finalization with handoff to another agent."""
        manager = Mock()
        manager.finalize_knowledge_transfer.return_value = True

        args = Namespace(
            kt_id='KT-123',
            handoff_to='agent2',
            instructions='Continue testing.'
        )

        with patch('sys.stdout'):
            result = cmd_kt_finalize(args, manager)

        self.assertEqual(result, 0)
        manager.finalize_knowledge_transfer.assert_called_once_with(
            kt_id='KT-123',
            handoff_to='agent2',
            instructions='Continue testing.'
        )

    def test_finalize_failure(self):
        """Test finalization failure."""
        manager = Mock()
        manager.finalize_knowledge_transfer.return_value = False

        args = Namespace(
            kt_id='KT-nonexistent',
            handoff_to=None,
            instructions=''
        )

        with patch('sys.stdout'):
            result = cmd_kt_finalize(args, manager)

        self.assertEqual(result, 1)


class TestCmdKtHistory(unittest.TestCase):
    """Test kt history command handler."""

    def test_history_empty(self):
        """Test history when no chain exists."""
        manager = Mock()
        manager.get_kt_history.return_value = []

        args = Namespace(kt_id='KT-123')

        with patch('sys.stdout'):
            result = cmd_kt_history(args, manager)

        self.assertEqual(result, 0)

    def test_history_with_chain(self):
        """Test history with a chain of entities."""
        manager = Mock()
        manager.get_kt_history.return_value = [
            ('knowledge_transfer', 'KT-1', 'Initial KT'),
            ('handoff', 'H-1', 'Handoff to Agent 2'),
            ('knowledge_transfer', 'KT-2', 'Continuation KT'),
        ]

        args = Namespace(kt_id='KT-2')

        with patch('sys.stdout'):
            result = cmd_kt_history(args, manager)

        self.assertEqual(result, 0)


class TestSetupKnowledgeTransferParser(unittest.TestCase):
    """Test argparse setup for knowledge transfer commands."""

    def test_parser_setup(self):
        """Test that all subcommands are registered."""
        parser = ArgumentParser()
        subparsers = parser.add_subparsers()

        setup_knowledge_transfer_parser(subparsers)

        # Verify the knowledge parser was added by attempting to parse
        # a known subcommand
        kt_parser = subparsers.choices.get('knowledge') or subparsers.choices.get('kt')
        self.assertIsNotNone(kt_parser)


class TestHandleKnowledgeTransferCommand(unittest.TestCase):
    """Test command routing."""

    def test_route_create(self):
        """Test routing to create handler."""
        manager = Mock()
        manager.create_knowledge_transfer.return_value = 'KT-123'

        args = Namespace(
            kt_command='create',
            title='Test',
            session='',
            summary='',
            content='',
            sections=None,
            tags=[]
        )

        with patch('sys.stdout'):
            result = handle_knowledge_transfer_command(args, manager)

        self.assertEqual(result, 0)

    def test_route_unknown_command(self):
        """Test routing with unknown command."""
        manager = Mock()
        args = Namespace(kt_command='unknown')

        with patch('sys.stdout'):
            result = handle_knowledge_transfer_command(args, manager)

        self.assertEqual(result, 1)

    def test_no_subcommand(self):
        """Test error when no subcommand specified."""
        manager = Mock()
        args = Namespace()  # No kt_command attribute

        with patch('sys.stdout'):
            result = handle_knowledge_transfer_command(args, manager)

        self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
