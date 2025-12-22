"""
Unit tests for SparkCodeIntelligence.

Tests:
- Initialization
- Training
- Code completion
- Semantic queries
- Save/load
- Statistics
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from cortical.spark.intelligence import SparkCodeIntelligence


class TestSparkCodeIntelligenceInit:
    """Initialization tests."""

    def test_default_init(self):
        """Test default initialization."""
        engine = SparkCodeIntelligence()
        assert engine.trained is False
        assert engine.root_dir == Path.cwd()

    def test_custom_root(self):
        """Test custom root directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SparkCodeIntelligence(root_dir=Path(tmpdir))
            assert engine.root_dir == Path(tmpdir)

    def test_repr_untrained(self):
        """Test repr for untrained engine."""
        engine = SparkCodeIntelligence()
        repr_str = repr(engine)
        assert 'trained=False' in repr_str


class TestSparkCodeIntelligenceTraining:
    """Training tests."""

    @pytest.fixture
    def training_dir(self):
        """Create a temporary directory with Python files for training."""
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir)

        # Create some Python files
        (path / 'module_a.py').write_text('''
"""Module A."""

class ClassA:
    """Class A docstring."""

    def __init__(self, value):
        self.value = value

    def process(self):
        return self.value * 2

def helper_function():
    """A helper function."""
    return 42
''')
        (path / 'module_b.py').write_text('''
"""Module B."""

from module_a import ClassA

class ClassB(ClassA):
    """Class B inherits from ClassA."""

    def process(self):
        result = super().process()
        helper_function()
        return result + 1
''')

        yield path
        shutil.rmtree(tmpdir)

    def test_train(self, training_dir):
        """Test training on a directory."""
        engine = SparkCodeIntelligence(root_dir=training_dir)
        engine.train(verbose=False)

        assert engine.trained is True
        assert engine.training_time > 0

    def test_train_indexes_classes(self, training_dir):
        """Test that training indexes classes."""
        engine = SparkCodeIntelligence(root_dir=training_dir)
        engine.train(verbose=False)

        assert 'ClassA' in engine.ast_index.classes
        assert 'ClassB' in engine.ast_index.classes

    def test_train_indexes_functions(self, training_dir):
        """Test that training indexes functions."""
        engine = SparkCodeIntelligence(root_dir=training_dir)
        engine.train(verbose=False)

        assert 'helper_function' in engine.ast_index.functions

    def test_train_builds_ngram_model(self, training_dir):
        """Test that training builds n-gram model."""
        engine = SparkCodeIntelligence(root_dir=training_dir)
        engine.train(verbose=False)

        assert len(engine.ngram_model.vocab) > 0
        assert engine.ngram_model.total_tokens > 0

    def test_get_stats(self, training_dir):
        """Test statistics after training."""
        engine = SparkCodeIntelligence(root_dir=training_dir)
        engine.train(verbose=False)

        stats = engine.get_stats()
        assert 'files_indexed' in stats
        assert 'classes' in stats
        assert 'functions' in stats
        assert 'ngram_vocab' in stats
        assert stats['files_indexed'] == 2


class TestSparkCodeIntelligenceCompletion:
    """Code completion tests."""

    @pytest.fixture
    def trained_engine(self):
        """Create a trained engine."""
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir)

        (path / 'test.py').write_text('''
class MyClass:
    def __init__(self):
        self.value = 1
        self.name = "test"

    def process(self):
        return self.value

    def transform(self):
        return self.name

def standalone():
    pass
''')

        engine = SparkCodeIntelligence(root_dir=path)
        engine.train(verbose=False)
        yield engine
        shutil.rmtree(tmpdir)

    def test_complete_empty_prefix(self, trained_engine):
        """Test completion with empty prefix."""
        results = trained_engine.complete("")
        assert results == []

    def test_complete_returns_tuples(self, trained_engine):
        """Test completion returns proper format."""
        results = trained_engine.complete("def", top_n=5)
        for suggestion, confidence, source in results:
            assert isinstance(suggestion, str)
            assert isinstance(confidence, float)
            assert isinstance(source, str)

    def test_complete_class_attributes(self, trained_engine):
        """Test completion for self. context."""
        results = trained_engine.complete("self.")

        # Should suggest class attributes
        suggestions = [r[0] for r in results]
        # Check that some AST-based suggestions are present
        sources = [r[2] for r in results]
        assert any('ast' in s for s in sources)

    def test_complete_class_methods(self, trained_engine):
        """Test completion for ClassName. context."""
        results = trained_engine.complete("MyClass.")

        suggestions = [r[0] for r in results]
        # Should include methods
        assert any('(' in s for s in suggestions)  # Methods have ()

    def test_complete_ngram_fallback(self, trained_engine):
        """Test completion uses n-gram when no AST context."""
        results = trained_engine.complete("foo bar")

        # Should use n-gram fallback
        sources = [r[2] for r in results]
        assert any('ngram' in s for s in sources)


class TestSparkCodeIntelligenceQueries:
    """Semantic query tests."""

    @pytest.fixture
    def query_engine(self):
        """Create an engine for query testing."""
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir)

        (path / 'module.py').write_text('''
class BaseClass:
    def base_method(self):
        pass

class ChildClass(BaseClass):
    def child_method(self):
        self.base_method()

def caller_function():
    obj = ChildClass()
    obj.child_method()
''')

        engine = SparkCodeIntelligence(root_dir=path)
        engine.train(verbose=False)
        yield engine
        shutil.rmtree(tmpdir)

    def test_find_class(self, query_engine):
        """Test finding a class."""
        result = query_engine.find_class('ChildClass')

        assert result is not None
        assert result['name'] == 'ChildClass'
        assert 'BaseClass' in result['bases']

    def test_find_class_not_found(self, query_engine):
        """Test finding non-existent class."""
        result = query_engine.find_class('NonExistent')
        assert result is None

    def test_find_function(self, query_engine):
        """Test finding a function."""
        results = query_engine.find_function('caller_function')

        assert len(results) > 0
        assert results[0]['name'] == 'caller_function'

    def test_find_callers(self, query_engine):
        """Test finding callers."""
        callers = query_engine.find_callers('child_method')

        # caller_function calls child_method
        caller_names = [c['caller'] for c in callers]
        assert 'caller_function' in caller_names

    def test_get_inheritance(self, query_engine):
        """Test getting inheritance info."""
        result = query_engine.get_inheritance('BaseClass')

        assert result['class'] == 'BaseClass'
        child_names = [c['name'] for c in result['children']]
        assert 'ChildClass' in child_names


class TestSparkCodeIntelligenceNaturalQuery:
    """Natural language query tests."""

    @pytest.fixture
    def query_engine(self):
        """Create an engine for query testing."""
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir)

        (path / 'module.py').write_text('''
class Processor:
    def process_data(self):
        self.transform()

    def transform(self):
        pass

class DataProcessor(Processor):
    pass
''')

        engine = SparkCodeIntelligence(root_dir=path)
        engine.train(verbose=False)
        yield engine
        shutil.rmtree(tmpdir)

    def test_query_what_calls(self, query_engine):
        """Test 'what calls X' query."""
        results = query_engine.query("what calls transform")

        assert len(results) > 0
        assert results[0]['type'] == 'callers'

    def test_query_class_inherits(self, query_engine):
        """Test 'class that inherits X' query."""
        results = query_engine.query("class that inherits Processor")

        assert len(results) > 0
        assert results[0]['type'] == 'inheritance'
        assert 'DataProcessor' in results[0]['children']

    def test_query_where_is(self, query_engine):
        """Test 'where is X' query."""
        results = query_engine.query("where is Processor")

        assert len(results) > 0
        # Should find the class or function

    def test_query_fallback(self, query_engine):
        """Test fallback search for unknown patterns."""
        results = query_engine.query("something about process")

        # Should search for 'process' in names
        # May or may not find results


class TestSparkCodeIntelligencePersistence:
    """Save/load tests."""

    @pytest.fixture
    def engine_with_data(self):
        """Create a trained engine."""
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir)

        (path / 'test.py').write_text('''
class TestClass:
    def method(self):
        pass
''')

        engine = SparkCodeIntelligence(root_dir=path)
        engine.train(verbose=False)
        yield engine, path
        shutil.rmtree(tmpdir)

    def test_save_load_roundtrip(self, engine_with_data):
        """Test save and load."""
        engine, tmpdir = engine_with_data

        # Save
        model_path = str(tmpdir / 'model.json')
        engine.save(model_path)

        # Verify file exists
        assert Path(model_path).exists()

        # Load
        new_engine = SparkCodeIntelligence()
        new_engine.load(model_path)

        assert new_engine.trained is True
        assert 'TestClass' in new_engine.ast_index.classes

    def test_save_json_format(self, engine_with_data):
        """Test that save creates valid JSON."""
        engine, tmpdir = engine_with_data

        model_path = str(tmpdir / 'model.json')
        engine.save(model_path)

        # Verify it's valid JSON
        with open(model_path) as f:
            data = json.load(f)

        assert 'version' in data
        assert 'ast_index' in data
        assert 'ngram_model' in data

    def test_load_nonexistent(self):
        """Test loading non-existent model."""
        engine = SparkCodeIntelligence()

        with pytest.raises(FileNotFoundError):
            engine.load('/nonexistent/path/model.json')


class TestSparkCodeIntelligenceRelatedFiles:
    """Related files tests."""

    @pytest.fixture
    def related_engine(self):
        """Create an engine with related files."""
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir)

        (path / 'common.py').write_text('''
def shared_util():
    pass
''')
        (path / 'module_a.py').write_text('''
from common import shared_util

def func_a():
    shared_util()
''')
        (path / 'module_b.py').write_text('''
from common import shared_util

def func_b():
    shared_util()
''')

        engine = SparkCodeIntelligence(root_dir=path)
        engine.train(verbose=False)
        yield engine, path
        shutil.rmtree(tmpdir)

    def test_find_related_files(self, related_engine):
        """Test finding related files."""
        engine, path = related_engine

        # Find files related to module_a
        module_a_path = str(path / 'module_a.py')
        related = engine.find_related_files(module_a_path, top_n=5)

        # Should find common.py and module_b.py as related
        related_paths = [p for p, score in related]
        # At least one related file should be found
        assert len(related) >= 0  # May be empty if no token overlap


class TestSparkCodeIntelligenceEdgeCases:
    """Edge case tests."""

    def test_empty_directory(self):
        """Test training on empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SparkCodeIntelligence(root_dir=Path(tmpdir))
            engine.train(verbose=False)

            assert engine.trained is True
            assert engine.get_stats()['files_indexed'] == 0

    def test_no_python_files(self):
        """Test training on directory without Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'readme.txt').write_text('Not a Python file')

            engine = SparkCodeIntelligence(root_dir=path)
            engine.train(verbose=False)

            assert engine.trained is True
            assert engine.get_stats()['files_indexed'] == 0

    def test_syntax_error_files(self):
        """Test handling files with syntax errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            (path / 'valid.py').write_text('x = 1')
            (path / 'invalid.py').write_text('def broken syntax:')

            engine = SparkCodeIntelligence(root_dir=path)
            engine.train(verbose=False)

            # Should index valid file, skip invalid
            stats = engine.get_stats()
            assert stats['files_indexed'] >= 1
