"""
Unit tests for ASTIndex.

Tests:
- File indexing
- Class indexing
- Function indexing
- Import tracking
- Call graph building
- Inheritance tracking
- Query methods
- Serialization/deserialization
"""

import pytest
import tempfile
from pathlib import Path
from cortical.spark.ast_index import ASTIndex, FunctionInfo, ClassInfo, ImportInfo


class TestFunctionInfo:
    """Tests for FunctionInfo dataclass."""

    def test_function_info_creation(self):
        """Test basic FunctionInfo creation."""
        info = FunctionInfo(
            name='test_func',
            file_path='/path/to/file.py',
            lineno=10,
            args=['a', 'b'],
            decorators=['staticmethod'],
            class_name=None,
            docstring='Test function',
            calls=['other_func']
        )
        assert info.name == 'test_func'
        assert info.full_name == 'test_func'

    def test_method_full_name(self):
        """Test method full name includes class."""
        info = FunctionInfo(
            name='method',
            file_path='/path/to/file.py',
            lineno=10,
            args=[],
            decorators=[],
            class_name='MyClass',
        )
        assert info.full_name == 'MyClass.method'

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        info = FunctionInfo(
            name='test_func',
            file_path='/path/to/file.py',
            lineno=10,
            args=['a', 'b'],
            decorators=['staticmethod'],
            class_name='MyClass',
            docstring='Test function',
            calls=['other_func', 'another']
        )
        data = info.to_dict()
        restored = FunctionInfo.from_dict(data)

        assert restored.name == info.name
        assert restored.file_path == info.file_path
        assert restored.lineno == info.lineno
        assert restored.args == info.args
        assert restored.decorators == info.decorators
        assert restored.class_name == info.class_name
        assert restored.docstring == info.docstring
        assert restored.calls == info.calls


class TestClassInfo:
    """Tests for ClassInfo dataclass."""

    def test_class_info_creation(self):
        """Test basic ClassInfo creation."""
        info = ClassInfo(
            name='MyClass',
            file_path='/path/to/file.py',
            lineno=5,
            bases=['BaseClass'],
            methods=['__init__', 'method'],
            attributes={'attr1', 'attr2'},
            decorators=['dataclass'],
            docstring='My class'
        )
        assert info.name == 'MyClass'
        assert info.bases == ['BaseClass']

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        info = ClassInfo(
            name='MyClass',
            file_path='/path/to/file.py',
            lineno=5,
            bases=['BaseClass'],
            methods=['__init__', 'method'],
            attributes={'attr1', 'attr2'},
            decorators=['dataclass'],
            docstring='My class'
        )
        data = info.to_dict()
        restored = ClassInfo.from_dict(data)

        assert restored.name == info.name
        assert restored.bases == info.bases
        assert restored.attributes == info.attributes  # Should be set


class TestImportInfo:
    """Tests for ImportInfo dataclass."""

    def test_import_info_creation(self):
        """Test basic ImportInfo creation."""
        info = ImportInfo(
            module='cortical.spark',
            names=['NGramModel', 'SparkPredictor'],
            file_path='/path/to/file.py',
            lineno=1,
            is_from=True
        )
        assert info.module == 'cortical.spark'
        assert info.is_from is True

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        info = ImportInfo(
            module='cortical.spark',
            names=['NGramModel'],
            file_path='/path/to/file.py',
            lineno=1,
            is_from=True
        )
        data = info.to_dict()
        restored = ImportInfo.from_dict(data)

        assert restored.module == info.module
        assert restored.is_from == info.is_from


class TestASTIndexBasic:
    """Basic ASTIndex tests."""

    def test_empty_index(self):
        """Test empty index creation."""
        index = ASTIndex()
        assert index.files_indexed == 0
        assert len(index.functions) == 0
        assert len(index.classes) == 0

    def test_stats_empty(self):
        """Test stats for empty index."""
        index = ASTIndex()
        stats = index.get_stats()
        assert stats['files'] == 0
        assert stats['classes'] == 0
        assert stats['functions'] == 0

    def test_repr_empty(self):
        """Test repr of empty index."""
        index = ASTIndex()
        repr_str = repr(index)
        assert 'ASTIndex' in repr_str
        assert 'files=0' in repr_str


class TestASTIndexFileIndexing:
    """File indexing tests."""

    @pytest.fixture
    def temp_python_file(self):
        """Create a temporary Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
"""Test module."""

import os
from typing import List

class MyClass:
    """My class docstring."""

    def __init__(self, value):
        self.value = value
        self.computed = None

    def process(self, data):
        """Process data."""
        return self.value + data

def standalone_function(x, y):
    """A standalone function."""
    result = x + y
    return result
''')
            f.flush()
            yield Path(f.name)
        Path(f.name).unlink()

    def test_index_file(self, temp_python_file):
        """Test indexing a Python file."""
        index = ASTIndex()
        result = index.index_file(temp_python_file)

        assert result is True
        assert index.files_indexed == 1

    def test_index_file_classes(self, temp_python_file):
        """Test class indexing."""
        index = ASTIndex()
        index.index_file(temp_python_file)

        assert 'MyClass' in index.classes
        class_info = index.classes['MyClass']
        assert '__init__' in class_info.methods
        assert 'process' in class_info.methods
        assert 'value' in class_info.attributes
        assert 'computed' in class_info.attributes

    def test_index_file_functions(self, temp_python_file):
        """Test function indexing."""
        index = ASTIndex()
        index.index_file(temp_python_file)

        # Standalone function
        assert 'standalone_function' in index.functions

        # Method
        assert 'MyClass.__init__' in index.functions
        assert 'MyClass.process' in index.functions

    def test_index_file_imports(self, temp_python_file):
        """Test import indexing."""
        index = ASTIndex()
        index.index_file(temp_python_file)

        modules = [imp.module for imp in index.imports]
        assert 'os' in modules
        assert 'typing' in modules

    def test_index_syntax_error(self):
        """Test handling of syntax errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('def broken syntax here:')
            f.flush()
            path = Path(f.name)

        try:
            index = ASTIndex()
            result = index.index_file(path)
            assert result is False
            assert index.parse_errors == 1
        finally:
            path.unlink()


class TestASTIndexCallGraph:
    """Call graph tests."""

    @pytest.fixture
    def call_graph_file(self):
        """Create a file with function calls."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def caller():
    helper()
    process_data()

def helper():
    pass

def process_data():
    helper()
''')
            f.flush()
            yield Path(f.name)
        Path(f.name).unlink()

    def test_call_graph_building(self, call_graph_file):
        """Test call graph is built correctly."""
        index = ASTIndex()
        index.index_file(call_graph_file)

        # caller calls helper and process_data
        assert 'helper' in index.call_graph['caller']
        assert 'process_data' in index.call_graph['caller']

        # process_data calls helper
        assert 'helper' in index.call_graph['process_data']

    def test_reverse_call_graph(self, call_graph_file):
        """Test reverse call graph is built."""
        index = ASTIndex()
        index.index_file(call_graph_file)

        # helper is called by caller and process_data
        assert 'caller' in index.reverse_call_graph['helper']
        assert 'process_data' in index.reverse_call_graph['helper']


class TestASTIndexInheritance:
    """Inheritance tracking tests."""

    @pytest.fixture
    def inheritance_file(self):
        """Create a file with class inheritance."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
class BaseClass:
    pass

class ChildA(BaseClass):
    pass

class ChildB(BaseClass):
    pass

class GrandChild(ChildA):
    pass
''')
            f.flush()
            yield Path(f.name)
        Path(f.name).unlink()

    def test_inheritance_tracking(self, inheritance_file):
        """Test inheritance relationships are tracked."""
        index = ASTIndex()
        index.index_file(inheritance_file)

        # BaseClass has children ChildA and ChildB
        assert 'ChildA' in index.inheritance['BaseClass']
        assert 'ChildB' in index.inheritance['BaseClass']

        # ChildA has child GrandChild
        assert 'GrandChild' in index.inheritance['ChildA']

    def test_inheritance_tree(self, inheritance_file):
        """Test inheritance tree generation."""
        index = ASTIndex()
        index.index_file(inheritance_file)

        tree = index.get_inheritance_tree('BaseClass')
        assert tree['name'] == 'BaseClass'
        child_names = [c['name'] for c in tree['children']]
        assert 'ChildA' in child_names
        assert 'ChildB' in child_names


class TestASTIndexQueries:
    """Query method tests."""

    @pytest.fixture
    def indexed_file(self):
        """Create and index a test file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
import cortical

class Processor:
    def process(self, data):
        self.transform(data)

    def transform(self, data):
        return data

def main():
    p = Processor()
    p.process([1, 2, 3])
''')
            f.flush()
            path = Path(f.name)

        index = ASTIndex()
        index.index_file(path)
        yield index
        path.unlink()

    def test_find_class(self, indexed_file):
        """Test finding a class."""
        result = indexed_file.find_class('Processor')
        assert result is not None
        assert result.name == 'Processor'

    def test_find_class_not_found(self, indexed_file):
        """Test finding non-existent class."""
        result = indexed_file.find_class('NonExistent')
        assert result is None

    def test_find_function(self, indexed_file):
        """Test finding a function."""
        results = indexed_file.find_function('main')
        assert len(results) > 0
        assert results[0].name == 'main'

    def test_find_callers(self, indexed_file):
        """Test finding function callers."""
        callers = indexed_file.find_callers('process')
        # main calls p.process
        caller_names = [c[0] for c in callers]
        assert 'main' in caller_names

    def test_find_imports_of(self, indexed_file):
        """Test finding files that import a module."""
        results = indexed_file.find_imports_of('cortical')
        assert len(results) > 0


class TestASTIndexSerialization:
    """Serialization tests."""

    def test_to_dict_from_dict(self):
        """Test full index serialization round-trip."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
class MyClass:
    def method(self):
        pass

def function():
    pass
''')
            f.flush()
            path = Path(f.name)

        try:
            index = ASTIndex()
            index.index_file(path)

            # Serialize
            data = index.to_dict()

            # Deserialize
            restored = ASTIndex.from_dict(data)

            # Verify
            assert restored.files_indexed == index.files_indexed
            assert len(restored.classes) == len(index.classes)
            assert len(restored.functions) == len(index.functions)
            assert 'MyClass' in restored.classes
            assert 'function' in restored.functions

        finally:
            path.unlink()


class TestASTIndexDirectory:
    """Directory indexing tests."""

    @pytest.fixture
    def temp_directory(self):
        """Create a temporary directory with Python files."""
        import tempfile
        import shutil

        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir)

        # Create some Python files
        (path / 'module_a.py').write_text('''
class ClassA:
    pass
''')
        (path / 'module_b.py').write_text('''
from module_a import ClassA

class ClassB(ClassA):
    pass
''')

        yield path
        shutil.rmtree(tmpdir)

    def test_index_directory(self, temp_directory):
        """Test indexing a directory."""
        index = ASTIndex()
        index.index_directory(temp_directory)

        assert index.files_indexed == 2
        assert 'ClassA' in index.classes
        assert 'ClassB' in index.classes


class TestASTIndexEdgeCases:
    """Edge case tests."""

    def test_async_function(self):
        """Test async function indexing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
async def async_func():
    await something()
''')
            f.flush()
            path = Path(f.name)

        try:
            index = ASTIndex()
            index.index_file(path)
            assert 'async_func' in index.functions
        finally:
            path.unlink()

    def test_decorated_class(self):
        """Test decorated class indexing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
from dataclasses import dataclass

@dataclass
class DataClass:
    value: int
''')
            f.flush()
            path = Path(f.name)

        try:
            index = ASTIndex()
            index.index_file(path)
            assert 'DataClass' in index.classes
            assert 'dataclass' in index.classes['DataClass'].decorators
        finally:
            path.unlink()

    def test_class_attributes_from_init(self):
        """Test attribute extraction from __init__."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
class MyClass:
    def __init__(self):
        self.attr1 = 1
        self.attr2: int = 2
''')
            f.flush()
            path = Path(f.name)

        try:
            index = ASTIndex()
            index.index_file(path)
            attrs = index.get_class_attributes('MyClass')
            assert 'attr1' in attrs
            assert 'attr2' in attrs
        finally:
            path.unlink()
