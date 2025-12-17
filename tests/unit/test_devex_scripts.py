"""
Unit tests for developer experience scripts.

Tests the four DevEx scripts:
- find_similar.py
- explain_code.py
- suggest_related.py
- corpus_health.py
"""

import os
import unittest
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cortical.processor import CorticalTextProcessor


class TestDevExScripts(unittest.TestCase):
    """Tests for DevEx utility scripts."""

    @classmethod
    def setUpClass(cls):
        """Create a small test corpus."""
        cls.processor = CorticalTextProcessor()

        # Add test documents
        cls.processor.process_document(
            "file1.py",
            """
def compute_pagerank(graph, damping=0.85):
    '''Compute PageRank scores for graph nodes.'''
    nodes = list(graph.keys())
    ranks = {node: 1.0 / len(nodes) for node in nodes}
    return ranks
            """
        )

        cls.processor.process_document(
            "file2.py",
            """
def compute_similarity(vec1, vec2):
    '''Calculate cosine similarity between vectors.'''
    import math
    dot = sum(a * b for a, b in zip(vec1, vec2))
    return dot
            """
        )

        cls.processor.process_document(
            "file3.py",
            """
import file1
from file2 import compute_similarity

def analyze_graph(graph):
    '''Analyze graph structure using PageRank.'''
    scores = file1.compute_pagerank(graph)
    return scores
            """
        )

        cls.processor.process_document(
            "docs/README.md",
            """
# Graph Analysis

This module provides graph analysis algorithms including:
- PageRank computation
- Similarity metrics
- Graph clustering
            """
        )

        # Compute all analysis
        cls.processor.compute_all()

        # Save to temp directory for script testing
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_corpus_path = os.path.join(cls.temp_dir, "corpus_state")
        cls.processor.save(cls.temp_corpus_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up temp corpus."""
        import shutil
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_find_similar_basic(self):
        """Test basic similarity finding."""
        # Import the module
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
        import find_similar

        # Test finding similar code
        results = find_similar.find_similar_code(
            self.processor,
            "def compute_pagerank",
            top_n=3
        )

        self.assertIsInstance(results, list)
        # Should find at least one similar result
        if results:
            self.assertIn('file', results[0])
            self.assertIn('similarity', results[0])
            self.assertIn('passage', results[0])

    def test_explain_code_basic(self):
        """Test code explanation."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
        import explain_code

        # Test analyzing code
        analysis = explain_code.analyze_code(
            self.processor,
            "file1.py",
            self.processor.documents["file1.py"]
        )

        self.assertIsInstance(analysis, dict)
        self.assertIn('key_terms', analysis)
        self.assertIn('concepts', analysis)
        self.assertIn('related_docs', analysis)
        self.assertIn('fingerprint', analysis)

        # Should have some key terms
        self.assertGreater(len(analysis['key_terms']), 0)

    def test_suggest_related_imports(self):
        """Test import relationship detection."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
        import suggest_related

        # Test extracting imports
        imports = suggest_related.extract_imports(
            self.processor.documents["file3.py"],
            "file3.py"
        )

        self.assertIn('file1', imports)
        self.assertIn('file2', imports)

    def test_suggest_related_files(self):
        """Test related file suggestions."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
        import suggest_related

        # Test finding related files
        suggestions = suggest_related.suggest_related_files(
            "file1.py",
            self.processor.documents["file1.py"],
            self.processor,
            top_n=5
        )

        self.assertIsInstance(suggestions, dict)
        self.assertIn('imports', suggestions)

    def test_corpus_health_basic(self):
        """Test corpus health analysis."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
        import corpus_health

        # Test analyzing corpus health
        stats = corpus_health.analyze_corpus_health(self.processor)

        self.assertIsInstance(stats, dict)
        self.assertIn('document_count', stats)
        self.assertIn('layers', stats)
        self.assertIn('stale_computations', stats)

        # Should have 4 documents
        self.assertEqual(stats['document_count'], 4)

        # Should have layer statistics
        self.assertIn('tokens', stats['layers'])
        self.assertIn('bigrams', stats['layers'])

    def test_corpus_health_score(self):
        """Test health score calculation."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
        import corpus_health

        stats = corpus_health.analyze_corpus_health(self.processor)
        status, score = corpus_health.get_health_score(stats)

        self.assertIsInstance(status, str)
        self.assertIsInstance(score, int)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_concept_analysis(self):
        """Test concept cluster analysis."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
        import corpus_health

        # Build concept clusters first
        self.processor.build_concept_clusters()

        # Test analyzing concepts
        concept_stats = corpus_health.analyze_concepts(self.processor)

        self.assertIsInstance(concept_stats, dict)
        self.assertIn('total_concepts', concept_stats)
        self.assertIn('avg_concept_size', concept_stats)
        self.assertIn('large_concepts', concept_stats)

    def test_fingerprint_comparison(self):
        """Test fingerprint-based similarity."""
        # Get fingerprints for two files
        fp1 = self.processor.get_fingerprint(
            self.processor.documents["file1.py"]
        )
        fp2 = self.processor.get_fingerprint(
            self.processor.documents["file3.py"]
        )

        comparison = self.processor.compare_fingerprints(fp1, fp2)

        self.assertIsInstance(comparison, dict)
        self.assertIn('overall_similarity', comparison)
        self.assertIn('shared_terms', comparison)

        # file3 imports file1, so should have some similarity
        self.assertGreater(comparison['overall_similarity'], 0)

    def test_get_file_content(self):
        """Test file content retrieval helper."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
        import find_similar

        # Test exact match
        doc_id, content = find_similar.get_file_content("file1.py", self.processor)
        self.assertEqual(doc_id, "file1.py")
        self.assertEqual(content, self.processor.documents["file1.py"])

        # Test FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            find_similar.get_file_content("nonexistent.py", self.processor)

    def test_doc_type_labels(self):
        """Test document type label generation."""
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
        import find_similar

        self.assertEqual(find_similar.get_doc_type_label("file1.py"), "CODE")
        self.assertEqual(find_similar.get_doc_type_label("tests/test.py"), "TEST")
        self.assertEqual(find_similar.get_doc_type_label("docs/README.md"), "DOC")
        self.assertEqual(find_similar.get_doc_type_label("other.md"), "MD")


if __name__ == '__main__':
    unittest.main()
