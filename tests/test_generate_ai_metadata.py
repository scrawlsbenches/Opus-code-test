"""
Tests for the AI metadata generator script.
"""

import ast
import os
import sys
import tempfile
import shutil
import unittest

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from generate_ai_metadata import (
    FunctionInfo,
    ClassInfo,
    ModuleAnalyzer,
    dict_to_yaml,
    generate_metadata_for_file,
    should_regenerate,
    COMPLEXITY_HINTS,
    RELATED_FUNCTION_PATTERNS,
)


class TestFunctionInfo(unittest.TestCase):
    """Tests for FunctionInfo class."""

    def test_basic_function(self):
        """Test extraction of a basic function."""
        code = '''
def my_function(x: int, y: str = "default") -> bool:
    """A test function."""
    return True
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        info = FunctionInfo(func_node, code.split('\n'))

        self.assertEqual(info.name, 'my_function')
        self.assertEqual(info.line_start, 2)
        self.assertFalse(info.is_private)
        self.assertFalse(info.is_dunder)
        self.assertFalse(info.is_async)
        self.assertIn('int', info.signature)
        self.assertIn('str', info.signature)
        self.assertIn('bool', info.signature)
        self.assertEqual(info.docstring_summary, 'A test function.')

    def test_private_function(self):
        """Test detection of private functions."""
        code = '''
def _private_func():
    pass
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        info = FunctionInfo(func_node, code.split('\n'))

        self.assertTrue(info.is_private)
        self.assertFalse(info.is_dunder)

    def test_dunder_function(self):
        """Test detection of dunder methods."""
        code = '''
def __init__(self):
    pass
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        info = FunctionInfo(func_node, code.split('\n'))

        self.assertTrue(info.is_dunder)
        self.assertTrue(info.is_private)  # Dunders start with _

    def test_async_function(self):
        """Test extraction of async functions."""
        code = '''
async def async_func(x: int) -> int:
    """An async function."""
    return x
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        info = FunctionInfo(func_node, code.split('\n'))

        self.assertEqual(info.name, 'async_func')
        self.assertTrue(info.is_async)
        self.assertIn('int', info.signature)

    def test_complex_signature(self):
        """Test extraction of complex type signatures."""
        code = '''
from typing import Dict, List, Optional

def complex_func(
    data: Dict[str, List[int]],
    callback: Optional[callable] = None,
    *args,
    **kwargs
) -> tuple:
    pass
'''
        tree = ast.parse(code)
        func_node = tree.body[1]  # Skip import
        info = FunctionInfo(func_node, code.split('\n'))

        self.assertIn('Dict', info.signature)
        self.assertIn('List', info.signature)
        self.assertIn('Optional', info.signature)
        self.assertIn('*args', info.signature)
        self.assertIn('**kwargs', info.signature)

    def test_decorated_function(self):
        """Test extraction of decorators."""
        code = '''
@staticmethod
@custom_decorator
def decorated():
    pass
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        info = FunctionInfo(func_node, code.split('\n'))

        self.assertIn('staticmethod', info.decorators)
        self.assertIn('custom_decorator', info.decorators)

    def test_method_detection(self):
        """Test detection of methods vs functions."""
        code = '''
def regular_func():
    pass

class MyClass:
    def method(self):
        pass

    @classmethod
    def class_method(cls):
        pass
'''
        tree = ast.parse(code)

        # Regular function
        func_info = FunctionInfo(tree.body[0], code.split('\n'))
        self.assertFalse(func_info.is_method)

        # Instance method
        method_node = tree.body[1].body[0]
        method_info = FunctionInfo(method_node, code.split('\n'))
        self.assertTrue(method_info.is_method)

        # Class method
        classmethod_node = tree.body[1].body[1]
        classmethod_info = FunctionInfo(classmethod_node, code.split('\n'))
        self.assertTrue(classmethod_info.is_method)


class TestClassInfo(unittest.TestCase):
    """Tests for ClassInfo class."""

    def test_basic_class(self):
        """Test extraction of a basic class."""
        code = '''
class MyClass:
    """A test class."""

    def method1(self):
        pass

    def method2(self):
        pass
'''
        tree = ast.parse(code)
        class_node = tree.body[0]
        info = ClassInfo(class_node, code.split('\n'))

        self.assertEqual(info.name, 'MyClass')
        self.assertEqual(info.docstring_summary, 'A test class.')
        self.assertEqual(len(info.methods), 2)
        self.assertEqual(info.methods[0].name, 'method1')
        self.assertEqual(info.methods[1].name, 'method2')

    def test_class_with_bases(self):
        """Test extraction of base classes."""
        code = '''
class Child(Parent, Mixin):
    pass
'''
        tree = ast.parse(code)
        class_node = tree.body[0]
        info = ClassInfo(class_node, code.split('\n'))

        self.assertIn('Parent', info.bases)
        self.assertIn('Mixin', info.bases)

    def test_class_with_async_methods(self):
        """Test that async methods are captured."""
        code = '''
class AsyncClass:
    async def async_method(self):
        pass

    def sync_method(self):
        pass
'''
        tree = ast.parse(code)
        class_node = tree.body[0]
        info = ClassInfo(class_node, code.split('\n'))

        self.assertEqual(len(info.methods), 2)
        async_method = [m for m in info.methods if m.name == 'async_method'][0]
        sync_method = [m for m in info.methods if m.name == 'sync_method'][0]

        self.assertTrue(async_method.is_async)
        self.assertFalse(sync_method.is_async)

    def test_empty_class(self):
        """Test handling of empty class."""
        code = '''
class EmptyClass:
    """An empty class."""
    pass
'''
        tree = ast.parse(code)
        class_node = tree.body[0]
        info = ClassInfo(class_node, code.split('\n'))

        self.assertEqual(info.name, 'EmptyClass')
        self.assertEqual(len(info.methods), 0)


class TestModuleAnalyzer(unittest.TestCase):
    """Tests for ModuleAnalyzer class."""

    def setUp(self):
        """Create a temporary file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test_module.py')

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_basic_module(self):
        """Test analysis of a basic module."""
        code = '''"""Test module docstring."""

import os
from typing import Dict

def func1():
    """First function."""
    pass

def func2():
    """Second function."""
    pass

class MyClass:
    """A class."""
    pass
'''
        with open(self.test_file, 'w') as f:
            f.write(code)

        analyzer = ModuleAnalyzer(self.test_file)

        self.assertEqual(analyzer.module_docstring, 'Test module docstring.')
        self.assertEqual(len(analyzer.functions), 2)
        self.assertEqual(len(analyzer.classes), 1)
        self.assertIn('os', analyzer.imports)

    def test_find_related_functions(self):
        """Test finding related functions."""
        code = '''
def find_documents():
    pass

def find_passages():
    pass

def fast_find_documents():
    pass

def unrelated_func():
    pass
'''
        with open(self.test_file, 'w') as f:
            f.write(code)

        analyzer = ModuleAnalyzer(self.test_file)

        related = analyzer.find_related_functions('find_documents')
        self.assertIn('find_passages', related)
        self.assertIn('fast_find_documents', related)

    def test_complexity_hints(self):
        """Test retrieval of complexity hints."""
        code = '''
def compute_all():
    pass

def compute_pagerank():
    pass

def regular_func():
    pass
'''
        with open(self.test_file, 'w') as f:
            f.write(code)

        analyzer = ModuleAnalyzer(self.test_file)

        self.assertIsNotNone(analyzer.get_complexity_hint('compute_all'))
        self.assertIsNotNone(analyzer.get_complexity_hint('compute_pagerank'))
        self.assertIsNone(analyzer.get_complexity_hint('regular_func'))

    def test_section_detection(self):
        """Test detection of logical sections."""
        code = '''
def process_document():
    pass

def add_document():
    pass

def compute_pagerank():
    pass

def compute_tfidf():
    pass

def find_documents():
    pass
'''
        with open(self.test_file, 'w') as f:
            f.write(code)

        analyzer = ModuleAnalyzer(self.test_file)
        sections = analyzer.detect_sections()

        section_names = [s['name'] for s in sections]
        # Should detect document, computation, and query sections
        self.assertTrue(len(sections) >= 2)

    def test_generate_metadata(self):
        """Test complete metadata generation."""
        code = '''"""Module docstring."""

from typing import List

def my_func(x: int) -> List[str]:
    """A function."""
    pass

class MyClass:
    """A class."""

    def method(self):
        """A method."""
        pass
'''
        with open(self.test_file, 'w') as f:
            f.write(code)

        analyzer = ModuleAnalyzer(self.test_file)
        metadata = analyzer.generate_metadata()

        self.assertIn('file', metadata)
        self.assertIn('lines', metadata)
        self.assertIn('functions', metadata)
        self.assertIn('classes', metadata)
        self.assertIn('imports', metadata)

        # Check function metadata
        self.assertIn('my_func', metadata['functions'])
        func_meta = metadata['functions']['my_func']
        self.assertIn('line', func_meta)
        self.assertIn('signature', func_meta)

        # Check class metadata
        self.assertIn('MyClass', metadata['classes'])


class TestDictToYaml(unittest.TestCase):
    """Tests for YAML conversion."""

    def test_simple_dict(self):
        """Test conversion of simple dictionary."""
        data = {'key': 'value', 'number': 42}
        yaml = dict_to_yaml(data)

        self.assertIn('key: value', yaml)
        self.assertIn('number: 42', yaml)

    def test_nested_dict(self):
        """Test conversion of nested dictionary."""
        data = {
            'outer': {
                'inner': 'value'
            }
        }
        yaml = dict_to_yaml(data)

        self.assertIn('outer:', yaml)
        self.assertIn('inner: value', yaml)

    def test_list(self):
        """Test conversion of lists."""
        data = {'items': ['a', 'b', 'c']}
        yaml = dict_to_yaml(data)

        self.assertIn('items:', yaml)
        self.assertIn('- a', yaml)
        self.assertIn('- b', yaml)
        self.assertIn('- c', yaml)

    def test_special_characters(self):
        """Test handling of special characters in strings."""
        data = {'key': 'value: with colon'}
        yaml = dict_to_yaml(data)

        # Should be quoted
        self.assertIn('"value: with colon"', yaml)

    def test_boolean(self):
        """Test boolean conversion."""
        data = {'flag': True, 'other': False}
        yaml = dict_to_yaml(data)

        self.assertIn('flag: true', yaml)
        self.assertIn('other: false', yaml)

    def test_none(self):
        """Test None conversion."""
        data = {'empty': None}
        yaml = dict_to_yaml(data)

        self.assertIn('empty: null', yaml)


class TestShouldRegenerate(unittest.TestCase):
    """Tests for regeneration checking."""

    def setUp(self):
        """Create temporary files."""
        self.temp_dir = tempfile.mkdtemp()
        self.py_file = os.path.join(self.temp_dir, 'test.py')
        self.meta_file = os.path.join(self.temp_dir, 'test.py.ai_meta')

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_no_meta_file(self):
        """Should regenerate if no meta file exists."""
        with open(self.py_file, 'w') as f:
            f.write('# test')

        self.assertTrue(should_regenerate(self.py_file, self.meta_file))

    def test_meta_newer_than_source(self):
        """Should not regenerate if meta is newer."""
        with open(self.py_file, 'w') as f:
            f.write('# test')

        # Create meta file after
        import time
        time.sleep(0.01)
        with open(self.meta_file, 'w') as f:
            f.write('# meta')

        self.assertFalse(should_regenerate(self.py_file, self.meta_file))

    def test_source_newer_than_meta(self):
        """Should regenerate if source is newer."""
        with open(self.meta_file, 'w') as f:
            f.write('# meta')

        import time
        time.sleep(0.01)
        with open(self.py_file, 'w') as f:
            f.write('# test')

        self.assertTrue(should_regenerate(self.py_file, self.meta_file))


class TestGenerateMetadataForFile(unittest.TestCase):
    """Tests for file metadata generation."""

    def setUp(self):
        """Create a temporary file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, 'test_module.py')

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_generates_valid_yaml(self):
        """Test that generated output is valid YAML."""
        code = '''"""Test module."""

def test_func():
    """A test function."""
    pass
'''
        with open(self.test_file, 'w') as f:
            f.write(code)

        yaml_content = generate_metadata_for_file(self.test_file)

        # Should start with header comment
        self.assertTrue(yaml_content.startswith('#'))

        # Should contain expected keys
        self.assertIn('file:', yaml_content)
        self.assertIn('lines:', yaml_content)
        self.assertIn('functions:', yaml_content)

    def test_handles_empty_file(self):
        """Test handling of empty file."""
        with open(self.test_file, 'w') as f:
            f.write('')

        yaml_content = generate_metadata_for_file(self.test_file)

        # Empty file still has 1 line (empty string split gives [''])
        self.assertIn('lines: 1', yaml_content)
        self.assertIn('functions: {}', yaml_content)


class TestIntegration(unittest.TestCase):
    """Integration tests for the metadata generator."""

    def test_cortical_processor_metadata(self):
        """Test metadata generation for actual cortical/processor.py."""
        processor_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'cortical', 'processor.py'
        )

        if not os.path.exists(processor_path):
            self.skipTest("processor.py not found")

        analyzer = ModuleAnalyzer(processor_path)
        metadata = analyzer.generate_metadata()

        # Should find the main class
        self.assertIn('CorticalTextProcessor', metadata['classes'])

        # Should find key functions
        func_names = list(metadata['functions'].keys())
        self.assertTrue(any('process_document' in name for name in func_names))
        self.assertTrue(any('compute_all' in name for name in func_names))

        # Should have complexity hints for expensive functions
        compute_all_key = [k for k in func_names if 'compute_all' in k][0]
        self.assertIn('complexity', metadata['functions'][compute_all_key])


if __name__ == '__main__':
    unittest.main()
