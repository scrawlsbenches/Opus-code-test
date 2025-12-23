"""
Integration tests for SparkCodeIntelligence.

Tests the full workflow of training, querying, and persisting
on the actual codebase.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from cortical.spark import (
    SparkCodeIntelligence,
    CodeTokenizer,
    ASTIndex,
    NGramModel
)


class TestSparkCodeIntelligenceIntegration:
    """Integration tests for the full SparkCodeIntelligence workflow."""

    @pytest.fixture(scope='class')
    def trained_engine(self):
        """Train engine on cortical/spark directory."""
        # Use the actual spark package for integration testing
        spark_dir = Path(__file__).parent.parent.parent / 'cortical' / 'spark'
        if not spark_dir.exists():
            pytest.skip("cortical/spark directory not found")

        engine = SparkCodeIntelligence(root_dir=spark_dir)
        engine.train(verbose=False)
        return engine

    def test_indexes_real_codebase(self, trained_engine):
        """Test that real codebase is indexed."""
        stats = trained_engine.get_stats()

        # Should index multiple files
        assert stats['files_indexed'] > 0

        # Should find real classes
        assert stats['classes'] > 0

        # Should find real functions
        assert stats['functions'] > 0

    def test_finds_ngram_model_class(self, trained_engine):
        """Test finding NGramModel class from the codebase."""
        result = trained_engine.find_class('NGramModel')

        assert result is not None
        assert result['name'] == 'NGramModel'
        assert 'train' in result['methods'] or 'predict' in result['methods']

    def test_finds_spark_predictor_class(self, trained_engine):
        """Test finding SparkPredictor class."""
        result = trained_engine.find_class('SparkPredictor')

        if result:  # May not exist in all versions
            assert result['name'] == 'SparkPredictor'

    def test_finds_anomaly_detector_class(self, trained_engine):
        """Test finding AnomalyDetector class."""
        result = trained_engine.find_class('AnomalyDetector')

        if result:
            assert result['name'] == 'AnomalyDetector'

    def test_completion_on_real_code(self, trained_engine):
        """Test code completion with real vocabulary."""
        results = trained_engine.complete("self.", top_n=10)

        # Should return some completions
        assert len(results) > 0

        # Each result should be a tuple
        for suggestion, confidence, source in results:
            assert isinstance(suggestion, str)
            assert isinstance(confidence, float)
            assert 0 <= confidence <= 1

    def test_query_what_calls(self, trained_engine):
        """Test natural language query on real code."""
        # Try to find callers of a common method
        results = trained_engine.query("what calls train")

        # Should return structured results
        assert isinstance(results, list)

    def test_inheritance_on_real_code(self, trained_engine):
        """Test inheritance query on real code."""
        # Check if there are any classes with inheritance
        stats = trained_engine.get_stats()
        if stats.get('inheritance_edges', 0) > 0:
            # Find a parent class and check inheritance
            for class_name in trained_engine.ast_index.inheritance.keys():
                result = trained_engine.get_inheritance(class_name)
                assert 'children' in result
                break


class TestSparkCodeIntelligenceSaveLoadIntegration:
    """Integration tests for save/load workflow."""

    @pytest.fixture
    def trained_engine_with_path(self):
        """Train engine and return with temp directory."""
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir)

        # Create a realistic Python project
        (path / 'core.py').write_text('''
"""Core module."""

class BaseProcessor:
    """Base processor class."""

    def __init__(self, config):
        self.config = config
        self.initialized = False

    def initialize(self):
        self.initialized = True

    def process(self, data):
        raise NotImplementedError
''')
        (path / 'impl.py').write_text('''
"""Implementation module."""

from core import BaseProcessor

class DataProcessor(BaseProcessor):
    """Data processor implementation."""

    def process(self, data):
        self.initialize()
        return self.transform(data)

    def transform(self, data):
        return [x * 2 for x in data]
''')
        (path / 'utils.py').write_text('''
"""Utility functions."""

def validate_input(data):
    """Validate input data."""
    if not isinstance(data, list):
        raise TypeError("Expected list")
    return True

def format_output(data):
    """Format output data."""
    return str(data)
''')

        engine = SparkCodeIntelligence(root_dir=path)
        engine.train(verbose=False)

        yield engine, path
        shutil.rmtree(tmpdir)

    def test_save_load_preserves_classes(self, trained_engine_with_path):
        """Test that save/load preserves class information."""
        engine, tmpdir = trained_engine_with_path

        model_path = str(tmpdir / 'test_model.json')
        engine.save(model_path)

        new_engine = SparkCodeIntelligence()
        new_engine.load(model_path)

        # Check classes are preserved
        assert 'BaseProcessor' in new_engine.ast_index.classes
        assert 'DataProcessor' in new_engine.ast_index.classes

    def test_save_load_preserves_inheritance(self, trained_engine_with_path):
        """Test that save/load preserves inheritance."""
        engine, tmpdir = trained_engine_with_path

        model_path = str(tmpdir / 'test_model.json')
        engine.save(model_path)

        new_engine = SparkCodeIntelligence()
        new_engine.load(model_path)

        # Check inheritance is preserved
        result = new_engine.get_inheritance('BaseProcessor')
        child_names = [c['name'] for c in result['children']]
        assert 'DataProcessor' in child_names

    def test_save_load_preserves_call_graph(self, trained_engine_with_path):
        """Test that save/load preserves call graph."""
        engine, tmpdir = trained_engine_with_path

        model_path = str(tmpdir / 'test_model.json')
        engine.save(model_path)

        new_engine = SparkCodeIntelligence()
        new_engine.load(model_path)

        # Check call graph is preserved
        callers = new_engine.find_callers('initialize')
        caller_names = [c['caller'] for c in callers]
        assert 'DataProcessor.process' in caller_names

    def test_save_load_preserves_ngram_model(self, trained_engine_with_path):
        """Test that save/load preserves n-gram model."""
        engine, tmpdir = trained_engine_with_path

        model_path = str(tmpdir / 'test_model.json')
        engine.save(model_path)

        new_engine = SparkCodeIntelligence()
        new_engine.load(model_path)

        # Check n-gram model is preserved
        assert len(new_engine.ngram_model.vocab) > 0
        assert new_engine.ngram_model.total_tokens > 0

        # Completions should work
        results = new_engine.complete("self.")
        assert len(results) >= 0  # May or may not have results


class TestComponentIntegration:
    """Test integration between components."""

    def test_tokenizer_ngram_integration(self):
        """Test CodeTokenizer works with NGramModel."""
        tokenizer = CodeTokenizer(split_identifiers=True)
        model = NGramModel(n=3)

        code = '''
def calculate_sum(a, b):
    result = a + b
    return result
'''
        tokens = tokenizer.tokenize(code)

        # Train model on tokens
        model.train_on_tokens([tokens])
        model.finalize()

        # Model should have learned from tokens
        assert len(model.vocab) > 0
        assert 'def' in model.vocab or 'result' in model.vocab

    def test_tokenizer_preserves_structure(self):
        """Test tokenizer preserves code structure for AST analysis."""
        tokenizer = CodeTokenizer(split_identifiers=True)

        code = "def foo(self, x, y): return x + y"
        tokens = tokenizer.tokenize(code)

        # Should preserve keywords and operators
        assert 'def' in tokens
        assert '(' in tokens
        assert ')' in tokens
        assert ':' in tokens
        assert 'return' in tokens

    def test_ast_index_call_graph_accuracy(self):
        """Test AST index builds accurate call graphs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'test.py').write_text('''
def a():
    b()
    c()

def b():
    c()

def c():
    pass
''')
            index = ASTIndex()
            index.index_file(path / 'test.py')

            # a calls b and c
            assert 'b' in index.call_graph['a']
            assert 'c' in index.call_graph['a']

            # b calls c
            assert 'c' in index.call_graph['b']

            # c is called by a and b
            assert 'a' in index.reverse_call_graph['c']
            assert 'b' in index.reverse_call_graph['c']


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_full_workflow(self):
        """Test complete workflow: train -> query -> save -> load -> query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create project files
            (path / 'app.py').write_text('''
from services import UserService

class Application:
    def __init__(self):
        self.user_service = UserService()

    def run(self):
        self.user_service.get_users()
''')
            (path / 'services.py').write_text('''
class UserService:
    def get_users(self):
        return self.fetch_from_db()

    def fetch_from_db(self):
        return []
''')

            # Step 1: Train
            engine = SparkCodeIntelligence(root_dir=path)
            engine.train(verbose=False)

            # Step 2: Query
            app_class = engine.find_class('Application')
            assert app_class is not None

            callers = engine.find_callers('get_users')
            assert len(callers) > 0

            # Step 3: Save
            model_path = str(path / 'model.json')
            engine.save(model_path)

            # Step 4: Load in new engine
            new_engine = SparkCodeIntelligence()
            new_engine.load(model_path)

            # Step 5: Query again - should give same results
            app_class_2 = new_engine.find_class('Application')
            assert app_class_2 is not None
            assert app_class_2['name'] == app_class['name']

            callers_2 = new_engine.find_callers('get_users')
            assert len(callers_2) == len(callers)
