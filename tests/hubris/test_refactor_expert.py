#!/usr/bin/env python3
"""
Tests for RefactorExpert

Tests the refactoring prediction expert including:
- Training from commit history
- Heuristic-based file analysis
- Signal type detection
- Co-refactoring patterns
"""

import tempfile
import unittest
from pathlib import Path
import sys

# Add scripts/hubris to path
SCRIPT_DIR = Path(__file__).parent.parent.parent / 'scripts' / 'hubris'
sys.path.insert(0, str(SCRIPT_DIR))

from experts.refactor_expert import RefactorExpert, suggest_refactoring


class TestRefactorExpertInit(unittest.TestCase):
    """Tests for RefactorExpert initialization."""

    def test_default_init(self):
        """Test default initialization."""
        expert = RefactorExpert()
        self.assertEqual(expert.expert_id, "refactor_expert")
        self.assertEqual(expert.expert_type, "refactor")
        self.assertEqual(expert.version, "1.0.0")
        self.assertIsNotNone(expert.model_data)

    def test_custom_thresholds(self):
        """Test custom threshold initialization."""
        custom_thresholds = {'large_file_lines': 1000}
        expert = RefactorExpert(thresholds=custom_thresholds)
        self.assertEqual(expert.thresholds['large_file_lines'], 1000)
        # Default thresholds should still be present
        self.assertIn('very_large_file_lines', expert.thresholds)

    def test_model_data_structure(self):
        """Test that model_data has required keys."""
        expert = RefactorExpert()
        required_keys = [
            'refactor_frequency',
            'co_refactor',
            'refactor_keywords',
            'refactor_types',
            'file_characteristics',
            'total_refactor_commits'
        ]
        for key in required_keys:
            self.assertIn(key, expert.model_data)


class TestRefactorExpertTraining(unittest.TestCase):
    """Tests for RefactorExpert training."""

    def setUp(self):
        self.expert = RefactorExpert()
        self.training_commits = [
            {
                'files': ['cortical/processor/compute.py', 'cortical/processor/query_api.py'],
                'message': 'refactor: Split large processor module into mixins'
            },
            {
                'files': ['cortical/analysis.py'],
                'message': 'refactor: Extract clustering logic to separate functions'
            },
            {
                'files': ['scripts/hubris/expert_consolidator.py'],
                'message': 'refactor: Simplify expert loading logic'
            },
            {
                'files': ['cortical/query/search.py', 'cortical/query/expansion.py'],
                'message': 'refactor: Move query expansion to dedicated module'
            },
            {
                'files': ['cortical/tokenizer.py'],
                'message': 'feat: Add new tokenization method'  # Not a refactor commit
            },
        ]

    def test_train_filters_refactor_commits(self):
        """Test that training only considers refactor commits."""
        self.expert.train(self.training_commits)
        # Should have 4 refactoring commits, not 5
        self.assertEqual(self.expert.trained_on_commits, 4)

    def test_train_builds_frequency_map(self):
        """Test that training builds file frequency map."""
        self.expert.train(self.training_commits)
        freq = self.expert.model_data['refactor_frequency']
        self.assertIn('cortical/analysis.py', freq)
        self.assertEqual(freq['cortical/analysis.py'], 1)

    def test_train_builds_co_refactor_matrix(self):
        """Test that training builds co-refactoring matrix."""
        self.expert.train(self.training_commits)
        co_refactor = self.expert.model_data['co_refactor']
        # Files from the same commit should be co-refactored
        self.assertIn('cortical/processor/compute.py', co_refactor)
        self.assertIn('cortical/processor/query_api.py',
                     co_refactor['cortical/processor/compute.py'])

    def test_train_detects_refactor_types(self):
        """Test that training detects refactor types from messages."""
        self.expert.train(self.training_commits)
        types = self.expert.model_data['refactor_types']
        # "Split" should be detected as extract
        self.assertIn('cortical/processor/compute.py', types)
        self.assertIn('extract', types['cortical/processor/compute.py'])
        # "Move" should be detected
        self.assertIn('cortical/query/search.py', types)
        self.assertIn('move', types['cortical/query/search.py'])


class TestRefactorExpertPrediction(unittest.TestCase):
    """Tests for RefactorExpert prediction."""

    def setUp(self):
        self.expert = RefactorExpert()
        # Train with some commits
        self.expert.train([
            {
                'files': ['cortical/analysis.py', 'cortical/processor/compute.py'],
                'message': 'refactor: Improve analysis module'
            },
            {
                'files': ['cortical/analysis.py'],
                'message': 'refactor: Simplify clustering'
            },
        ])

    def test_predict_returns_expert_prediction(self):
        """Test that predict returns ExpertPrediction."""
        prediction = self.expert.predict({'query': 'improve analysis'})
        self.assertEqual(prediction.expert_id, 'refactor_expert')
        self.assertEqual(prediction.expert_type, 'refactor')
        self.assertIsInstance(prediction.items, list)
        self.assertIsInstance(prediction.metadata, dict)

    def test_predict_with_historical_patterns(self):
        """Test prediction uses historical patterns."""
        prediction = self.expert.predict({
            'query': 'analysis improvements',
            'include_heuristics': False
        })
        # analysis.py was refactored twice, should rank high
        if prediction.items:
            file_paths = [f for f, _ in prediction.items]
            self.assertIn('cortical/analysis.py', file_paths)

    def test_predict_with_co_refactoring(self):
        """Test prediction uses co-refactoring patterns."""
        prediction = self.expert.predict({
            'files': ['cortical/analysis.py'],
            'include_heuristics': False
        })
        # compute.py was refactored with analysis.py
        if prediction.items:
            file_paths = [f for f, _ in prediction.items]
            # Should suggest compute.py based on co-refactoring
            self.assertIn('cortical/processor/compute.py', file_paths)

    def test_predict_respects_top_n(self):
        """Test that predict respects top_n limit."""
        prediction = self.expert.predict({
            'query': 'refactor',
            'top_n': 2
        })
        self.assertLessEqual(len(prediction.items), 2)

    def test_predict_empty_context(self):
        """Test prediction with empty context."""
        prediction = self.expert.predict({})
        # Should not crash, may return empty or historical-based results
        self.assertIsInstance(prediction.items, list)


class TestRefactorExpertHeuristics(unittest.TestCase):
    """Tests for RefactorExpert heuristic analysis."""

    def setUp(self):
        self.expert = RefactorExpert()

    def test_is_refactor_commit_explicit(self):
        """Test explicit refactor: prefix detection."""
        self.assertTrue(self.expert._is_refactor_commit('refactor: Split module'))
        self.assertTrue(self.expert._is_refactor_commit('refactor(core): Update'))
        self.assertFalse(self.expert._is_refactor_commit('feat: Add feature'))

    def test_is_refactor_commit_keywords(self):
        """Test refactoring keyword detection."""
        self.assertTrue(self.expert._is_refactor_commit('Split large file'))
        self.assertTrue(self.expert._is_refactor_commit('Extract common logic'))
        self.assertTrue(self.expert._is_refactor_commit('Cleanup code'))
        self.assertFalse(self.expert._is_refactor_commit('Add new feature'))

    def test_detect_refactor_type_extract(self):
        """Test extract signal detection."""
        types = self.expert._detect_refactor_type('Split large module into parts')
        self.assertIn('extract', types)

    def test_detect_refactor_type_move(self):
        """Test move signal detection."""
        types = self.expert._detect_refactor_type('Move function to utils')
        self.assertIn('move', types)

    def test_detect_refactor_type_simplify(self):
        """Test simplify signal detection."""
        types = self.expert._detect_refactor_type('Cleanup old code')
        self.assertIn('simplify', types)

    def test_detect_long_functions(self):
        """Test long function detection."""
        code = '''
def short_function():
    return 1

def long_function():
''' + '\n'.join(['    x = ' + str(i) for i in range(60)]) + '''
    return x
'''
        long_funcs = self.expert._detect_long_functions(code)
        self.assertIn('long_function', long_funcs)
        self.assertNotIn('short_function', long_funcs)

    def test_detect_max_indentation(self):
        """Test indentation depth detection."""
        code = '''
def func():
    if True:
        for i in range(10):
            while True:
                if x:
                    if y:
                        if z:
                            pass
'''
        lines = code.split('\n')
        max_indent = self.expert._detect_max_indentation(lines)
        self.assertGreaterEqual(max_indent, 6)

    def test_extract_keywords(self):
        """Test keyword extraction."""
        keywords = self.expert._extract_keywords('Improve the analysis module performance')
        self.assertIn('analysis', keywords)
        self.assertIn('module', keywords)
        self.assertIn('performance', keywords)
        # Stop words should be filtered
        self.assertNotIn('the', keywords)
        self.assertNotIn('refactor', keywords)


class TestRefactorExpertPersistence(unittest.TestCase):
    """Tests for RefactorExpert save/load."""

    def test_save_and_load(self):
        """Test saving and loading expert."""
        expert = RefactorExpert()
        expert.train([
            {
                'files': ['test.py'],
                'message': 'refactor: Test commit'
            }
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'refactor_expert.json'
            expert.save(save_path)

            loaded = RefactorExpert.load(save_path)
            self.assertEqual(loaded.expert_id, expert.expert_id)
            self.assertEqual(loaded.trained_on_commits, expert.trained_on_commits)
            self.assertEqual(
                loaded.model_data['refactor_frequency'],
                expert.model_data['refactor_frequency']
            )

    def test_to_dict_and_from_dict(self):
        """Test dictionary serialization."""
        expert = RefactorExpert()
        expert.train([
            {
                'files': ['a.py', 'b.py'],
                'message': 'refactor: Update'
            }
        ])

        data = expert.to_dict()
        loaded = RefactorExpert.from_dict(data)

        self.assertEqual(loaded.expert_type, 'refactor')
        self.assertEqual(loaded.trained_on_commits, expert.trained_on_commits)


class TestRefactorExpertFileAnalysis(unittest.TestCase):
    """Tests for file-level analysis with real files."""

    def setUp(self):
        self.expert = RefactorExpert()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_analyze_large_file(self):
        """Test analysis of a large file triggers extract signal."""
        large_file = Path(self.tmpdir) / 'large.py'
        # Create a file with 600 lines
        content = '\n'.join([f'line_{i} = {i}' for i in range(600)])
        large_file.write_text(content)

        result = self.expert._analyze_file_heuristics('large.py', self.tmpdir)
        self.assertGreater(result['score'], 0)
        self.assertIn('extract', result['signals'])

    def test_analyze_deeply_nested_file(self):
        """Test analysis of deeply nested code triggers simplify signal."""
        nested_file = Path(self.tmpdir) / 'nested.py'
        content = '''
def func():
    if True:
        for i in range(10):
            while True:
                if x:
                    if y:
                        if z:
                            pass
'''
        nested_file.write_text(content)

        result = self.expert._analyze_file_heuristics('nested.py', self.tmpdir)
        self.assertGreater(result['score'], 0)
        self.assertIn('simplify', result['signals'])

    def test_analyze_nonexistent_file(self):
        """Test analysis of nonexistent file returns empty result."""
        result = self.expert._analyze_file_heuristics('nonexistent.py', self.tmpdir)
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['signals'], [])


class TestRefactorExpertIntegration(unittest.TestCase):
    """Integration tests for RefactorExpert with ExpertConsolidator."""

    def test_consolidator_registers_refactor_expert(self):
        """Test that ExpertConsolidator knows about RefactorExpert."""
        from expert_consolidator import ExpertConsolidator
        self.assertIn('refactor', ExpertConsolidator.EXPERT_CLASSES)

    def test_consolidator_creates_refactor_expert(self):
        """Test that ExpertConsolidator can create RefactorExpert."""
        from expert_consolidator import ExpertConsolidator
        consolidator = ExpertConsolidator()
        consolidator.create_all_experts()
        self.assertIn('refactor', consolidator.experts)


class TestSuggestRefactoringConvenience(unittest.TestCase):
    """Tests for the suggest_refactoring convenience function."""

    def test_suggest_refactoring_without_model(self):
        """Test suggest_refactoring works without trained model."""
        # Should return empty list for nonexistent files
        results = suggest_refactoring(
            files=['nonexistent.py'],
            repo_root='/nonexistent',
            model_path=None
        )
        self.assertIsInstance(results, list)

    def test_suggest_refactoring_returns_tuples(self):
        """Test suggest_refactoring returns correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / 'test.py'
            test_file.write_text('x = 1\n' * 600)  # Large file

            results = suggest_refactoring(
                files=['test.py'],
                repo_root=tmpdir
            )

            if results:
                filepath, score, signals = results[0]
                self.assertIsInstance(filepath, str)
                self.assertIsInstance(score, float)
                self.assertIsInstance(signals, list)


if __name__ == '__main__':
    unittest.main()
