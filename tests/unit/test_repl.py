"""
Tests for the Cortical REPL.

Tests the interactive REPL interface for the Cortical Text Processor.
"""

import unittest
import sys
import io
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))

from repl import CorticalREPL
from cortical.processor import CorticalTextProcessor
from cortical.layers import CorticalLayer


class TestREPLBasics(unittest.TestCase):
    """Test basic REPL functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.repl = CorticalREPL()

    def test_initialization_no_corpus(self):
        """Test REPL initializes without corpus."""
        self.assertIsNone(self.repl.processor)
        self.assertIsNone(self.repl.corpus_file)

    def test_initialization_with_invalid_corpus(self):
        """Test REPL handles invalid corpus gracefully."""
        with patch('sys.stdout', new=io.StringIO()):
            repl = CorticalREPL(corpus_file='nonexistent.pkl')
        self.assertIsNone(repl.processor)

    def test_quit_command(self):
        """Test quit command returns True."""
        result = self.repl.do_quit('')
        self.assertTrue(result)

    def test_exit_command(self):
        """Test exit command is alias for quit."""
        result = self.repl.do_exit('')
        self.assertTrue(result)

    def test_eof_command(self):
        """Test EOF (Ctrl+D) exits."""
        result = self.repl.do_EOF('')
        self.assertTrue(result)

    def test_empty_line(self):
        """Test empty line does nothing."""
        result = self.repl.emptyline()
        self.assertFalse(result)

    def test_unknown_command(self):
        """Test unknown command handling."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.default('invalidcommand')
            output = mock_stdout.getvalue()
            self.assertIn('Unknown command', output)


class TestREPLWithCorpus(unittest.TestCase):
    """Test REPL commands with a loaded corpus."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a processor with test data
        self.processor = CorticalTextProcessor(enable_metrics=True)
        self.processor.process_document(
            "doc1.py",
            "def compute_pagerank(graph):\n"
            "    # PageRank algorithm implementation\n"
            "    return pagerank_values"
        )
        self.processor.process_document(
            "doc2.py",
            "class NeuralNetwork:\n"
            "    def __init__(self):\n"
            "        self.layers = []"
        )
        self.processor.compute_all()

        # Create temp file and save
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='wb', suffix='.pkl', delete=False
        )
        self.temp_file.close()
        self.processor.save(self.temp_file.name)

        # Create REPL with loaded corpus
        with patch('sys.stdout', new=io.StringIO()):
            self.repl = CorticalREPL(corpus_file=self.temp_file.name)

    def tearDown(self):
        """Clean up temp files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_load_command(self):
        """Test load command loads corpus."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_load(self.temp_file.name)
            output = mock_stdout.getvalue()
            self.assertIn('Loaded', output)
            self.assertIsNotNone(self.repl.processor)

    def test_load_nonexistent_file(self):
        """Test load command with nonexistent file."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_load('nonexistent.pkl')
            output = mock_stdout.getvalue()
            self.assertIn('not found', output.lower())

    def test_load_no_argument(self):
        """Test load command without file argument."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_load('')
            output = mock_stdout.getvalue()
            self.assertIn('specify a file', output.lower())

    def test_stats_command(self):
        """Test stats command shows corpus statistics."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_stats('')
            output = mock_stdout.getvalue()
            self.assertIn('Documents', output)
            self.assertIn('Tokens', output)
            self.assertIn('Bigrams', output)

    def test_stats_without_corpus(self):
        """Test stats command without loaded corpus."""
        repl = CorticalREPL()
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            repl.do_stats('')
            output = mock_stdout.getvalue()
            self.assertIn('No corpus loaded', output)

    def test_search_command(self):
        """Test search command finds documents."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_search('pagerank')
            output = mock_stdout.getvalue()
            self.assertIn('Results', output)
            # Should find doc1.py which contains pagerank
            self.assertIn('doc1.py', output.lower())

    def test_search_no_query(self):
        """Test search command without query."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_search('')
            output = mock_stdout.getvalue()
            self.assertIn('provide a search query', output.lower())

    def test_search_no_results(self):
        """Test search with no results."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_search('xyznonexistent')
            output = mock_stdout.getvalue()
            self.assertIn('No results', output)

    def test_expand_command(self):
        """Test expand command shows query expansion."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_expand('pagerank')
            output = mock_stdout.getvalue()
            self.assertIn('expansion', output.lower())

    def test_expand_no_term(self):
        """Test expand command without term."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_expand('')
            output = mock_stdout.getvalue()
            self.assertIn('provide a term', output.lower())

    def test_concepts_command(self):
        """Test concepts command lists clusters."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_concepts('')
            output = mock_stdout.getvalue()
            self.assertIn('Concept', output)

    def test_concepts_with_number(self):
        """Test concepts command with custom number."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_concepts('5')
            output = mock_stdout.getvalue()
            self.assertIn('Concept', output)

    def test_concepts_invalid_number(self):
        """Test concepts command with invalid number."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_concepts('abc')
            output = mock_stdout.getvalue()
            self.assertIn('valid number', output.lower())

    def test_fingerprint_command(self):
        """Test fingerprint command."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_fingerprint('neural network')
            output = mock_stdout.getvalue()
            self.assertIn('Fingerprint', output)
            self.assertIn('Top terms', output)

    def test_fingerprint_no_text(self):
        """Test fingerprint without text."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_fingerprint('')
            output = mock_stdout.getvalue()
            self.assertIn('provide text', output.lower())

    def test_patterns_command(self):
        """Test patterns command detects code patterns."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_patterns('doc1.py')
            output = mock_stdout.getvalue()
            # Should detect some patterns or show message
            self.assertTrue(len(output) > 0)

    def test_patterns_no_doc(self):
        """Test patterns without doc_id."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_patterns('')
            output = mock_stdout.getvalue()
            self.assertIn('provide a document', output.lower())

    def test_metrics_command(self):
        """Test metrics command shows timing metrics."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_metrics('')
            output = mock_stdout.getvalue()
            # Either shows metrics or "no metrics" message
            self.assertTrue(len(output) > 0)

    def test_stale_command(self):
        """Test stale command shows stale computations."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_stale('')
            output = mock_stdout.getvalue()
            # Should show either stale or up-to-date
            self.assertTrue(len(output) > 0)

    def test_relations_command(self):
        """Test relations command."""
        # First extract semantics
        self.repl.processor.extract_corpus_semantics()

        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_relations('')
            output = mock_stdout.getvalue()
            self.assertTrue(len(output) > 0)

    def test_relations_with_number(self):
        """Test relations with custom count."""
        self.repl.processor.extract_corpus_semantics()

        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_relations('5')
            output = mock_stdout.getvalue()
            self.assertTrue(len(output) > 0)

    def test_passages_command(self):
        """Test passages command finds relevant passages."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_passages('pagerank')
            output = mock_stdout.getvalue()
            self.assertIn('Passages', output)

    def test_passages_no_query(self):
        """Test passages without query."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_passages('')
            output = mock_stdout.getvalue()
            self.assertIn('provide a search query', output.lower())

    def test_docs_command(self):
        """Test docs command with documentation boost."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_docs('pagerank')
            output = mock_stdout.getvalue()
            self.assertIn('Results', output)

    def test_code_command(self):
        """Test code command with code-aware expansion."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_code('compute')
            output = mock_stdout.getvalue()
            self.assertIn('expansion', output.lower())

    def test_intent_command(self):
        """Test intent command parses query intent."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_intent('where do we compute pagerank')
            output = mock_stdout.getvalue()
            self.assertIn('Intent', output)

    def test_intent_no_query(self):
        """Test intent without query."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_intent('')
            output = mock_stdout.getvalue()
            self.assertIn('provide a query', output.lower())

    def test_similar_command(self):
        """Test similar command finds similar code."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_similar('doc1.py:1')
            output = mock_stdout.getvalue()
            # Should complete without error
            self.assertTrue(len(output) > 0)

    def test_similar_no_arg(self):
        """Test similar without argument."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_similar('')
            output = mock_stdout.getvalue()
            self.assertIn('provide file:line', output.lower())

    def test_similar_invalid_format(self):
        """Test similar with invalid format."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_similar('doc1.py')
            output = mock_stdout.getvalue()
            self.assertIn('format', output.lower())


class TestREPLComputeCommands(unittest.TestCase):
    """Test REPL computation commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = CorticalTextProcessor(enable_metrics=True)
        self.processor.process_document("doc1", "test content")

        self.temp_file = tempfile.NamedTemporaryFile(
            mode='wb', suffix='.pkl', delete=False
        )
        self.temp_file.close()
        self.processor.save(self.temp_file.name)

        with patch('sys.stdout', new=io.StringIO()):
            self.repl = CorticalREPL(corpus_file=self.temp_file.name)

    def tearDown(self):
        """Clean up temp files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_compute_all(self):
        """Test compute all command."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_compute('')
            output = mock_stdout.getvalue()
            self.assertIn('Computing', output)

    def test_compute_tfidf(self):
        """Test compute tfidf command."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_compute('tfidf')
            output = mock_stdout.getvalue()
            self.assertIn('TF-IDF', output)

    def test_compute_pagerank(self):
        """Test compute pagerank command."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_compute('pagerank')
            output = mock_stdout.getvalue()
            self.assertIn('PageRank', output)

    def test_compute_concepts(self):
        """Test compute concepts command."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_compute('concepts')
            output = mock_stdout.getvalue()
            self.assertIn('concepts', output.lower())

    def test_compute_semantics(self):
        """Test compute semantics command."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_compute('semantics')
            output = mock_stdout.getvalue()
            self.assertIn('semantic', output.lower())

    def test_compute_invalid_type(self):
        """Test compute with invalid type."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_compute('invalid')
            output = mock_stdout.getvalue()
            self.assertIn('Unknown', output)


class TestREPLSaveExport(unittest.TestCase):
    """Test REPL save and export commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1", "test content")
        self.processor.compute_all()

        self.temp_file = tempfile.NamedTemporaryFile(
            mode='wb', suffix='.pkl', delete=False
        )
        self.temp_file.close()
        self.processor.save(self.temp_file.name)

        with patch('sys.stdout', new=io.StringIO()):
            self.repl = CorticalREPL(corpus_file=self.temp_file.name)

    def tearDown(self):
        """Clean up temp files."""
        for f in [self.temp_file.name]:
            if os.path.exists(f):
                os.unlink(f)

    def test_save_command(self):
        """Test save command saves corpus."""
        temp_save = tempfile.NamedTemporaryFile(
            mode='wb', suffix='.pkl', delete=False
        )
        temp_save.close()

        try:
            with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
                self.repl.do_save(temp_save.name)
                output = mock_stdout.getvalue()
                self.assertIn('Saved', output)
                self.assertTrue(os.path.exists(temp_save.name))
        finally:
            if os.path.exists(temp_save.name):
                os.unlink(temp_save.name)

    def test_save_no_file(self):
        """Test save without filename."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_save('')
            output = mock_stdout.getvalue()
            self.assertIn('specify a file', output.lower())

    def test_export_json(self):
        """Test export to JSON."""
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()

        try:
            with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
                self.repl.do_export(f'{temp_dir} json')
                output = mock_stdout.getvalue()
                self.assertIn('Exported', output)
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def test_export_no_file(self):
        """Test export without filename."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_export('')
            output = mock_stdout.getvalue()
            self.assertIn('specify a path', output.lower())

    def test_export_invalid_format(self):
        """Test export with invalid format."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_export('test_dir invalid')
            output = mock_stdout.getvalue()
            self.assertIn('Unknown', output)


class TestREPLUtilityCommands(unittest.TestCase):
    """Test REPL utility commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = CorticalTextProcessor(enable_metrics=True)
        self.processor.process_document("doc1", "test content")
        self.processor.compute_all()

        self.temp_file = tempfile.NamedTemporaryFile(
            mode='wb', suffix='.pkl', delete=False
        )
        self.temp_file.close()
        self.processor.save(self.temp_file.name)

        with patch('sys.stdout', new=io.StringIO()):
            self.repl = CorticalREPL(corpus_file=self.temp_file.name)

    def tearDown(self):
        """Clean up temp files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_clear_command(self):
        """Test clear metrics command."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_clear('')
            output = mock_stdout.getvalue()
            self.assertIn('cleared', output.lower())

    def test_reset_command(self):
        """Test reset metrics command."""
        with patch('sys.stdout', new=io.StringIO()) as mock_stdout:
            self.repl.do_reset('')
            output = mock_stdout.getvalue()
            self.assertIn('reset', output.lower())


class TestREPLCompletion(unittest.TestCase):
    """Test REPL tab completion."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = CorticalTextProcessor()
        self.processor.process_document("doc1.py", "test")
        self.processor.process_document("doc2.py", "test")

        self.temp_file = tempfile.NamedTemporaryFile(
            mode='wb', suffix='.pkl', delete=False
        )
        self.temp_file.close()
        self.processor.save(self.temp_file.name)

        with patch('sys.stdout', new=io.StringIO()):
            self.repl = CorticalREPL(corpus_file=self.temp_file.name)

    def tearDown(self):
        """Clean up temp files."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_complete_compute(self):
        """Test completion for compute command."""
        completions = self.repl.complete_compute('t', 'compute t', 8, 9)
        self.assertIn('tfidf', completions)

    def test_complete_compute_all(self):
        """Test completion includes 'all'."""
        completions = self.repl.complete_compute('a', 'compute a', 8, 9)
        self.assertIn('all', completions)

    def test_complete_export_format(self):
        """Test completion for export formats."""
        completions = self.repl.complete_export('j', 'export dir j', 11, 12)
        self.assertIn('json', completions)


if __name__ == '__main__':
    unittest.main()
