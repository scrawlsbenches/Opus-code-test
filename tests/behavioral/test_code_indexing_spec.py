"""
Behavioral Tests for Code Indexing.

Specification:
    The CodeBridge converts ASTIndex data to cognitive graph atoms,
    storing REFERENCES to code (not code itself).

Test Categories:
    1. Atom Creation - code entities become atoms with correct metadata
    2. Link Creation - relationships become links
    3. Query Support - can traverse code structure
"""

import pytest
import tempfile
from pathlib import Path

from cortical.cognitive.graph import CognitiveGraph, AtomType, Atom


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def graph():
    """Fresh cognitive graph for each test."""
    return CognitiveGraph()


@pytest.fixture
def code_bridge(graph):
    """CodeBridge connected to a fresh graph."""
    from cortical.cognitive.code_bridge import CodeBridge
    return CodeBridge(graph)


@pytest.fixture
def simple_python_file(tmp_path):
    """A simple Python file with class and function."""
    code = '''"""Test module docstring."""

class MyClass:
    """My class docstring."""

    def __init__(self, value):
        self.value = value

    def process(self, data):
        """Process some data."""
        return self.value + data

def standalone_function(x, y):
    """A standalone function."""
    return x + y
'''
    file_path = tmp_path / "simple.py"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def call_graph_file(tmp_path):
    """Python file with function calls for call graph testing."""
    code = '''
def caller():
    helper()
    process_data()

def helper():
    pass

def process_data():
    helper()
'''
    file_path = tmp_path / "calls.py"
    file_path.write_text(code)
    return file_path


@pytest.fixture
def inheritance_file(tmp_path):
    """Python file with class inheritance."""
    code = '''
class BaseClass:
    """The base class."""
    pass

class ChildA(BaseClass):
    """First child."""
    pass

class ChildB(BaseClass):
    """Second child."""
    pass

class GrandChild(ChildA):
    """Grandchild through ChildA."""
    pass
'''
    file_path = tmp_path / "inheritance.py"
    file_path.write_text(code)
    return file_path


# =============================================================================
# Test: Atom Creation
# =============================================================================


class TestFileAtomCreation:
    """Verify FILE atoms are created correctly."""

    def test_indexing_creates_file_atom(self, code_bridge, simple_python_file):
        """Indexing a file should create a FILE atom."""
        stats = code_bridge.index_file(simple_python_file)

        assert stats.files == 1
        # FILE atom should exist with the file path as name
        file_atoms = code_bridge.graph.find_by_type(AtomType.FILE)
        assert len(file_atoms) == 1
        assert simple_python_file.name in file_atoms[0].name

    def test_file_atom_has_path_in_name(self, code_bridge, simple_python_file):
        """FILE atom name should contain the file path."""
        code_bridge.index_file(simple_python_file)

        file_atoms = code_bridge.graph.find_by_type(AtomType.FILE)
        file_atom = file_atoms[0]
        # Name should be or contain the path
        assert "simple.py" in file_atom.name


class TestClassAtomCreation:
    """Verify CLASS atoms are created with correct metadata."""

    def test_class_creates_class_atom(self, code_bridge, simple_python_file):
        """Indexing should create CLASS atoms for each class."""
        stats = code_bridge.index_file(simple_python_file)

        assert stats.classes == 1
        class_atoms = code_bridge.graph.find_by_type(AtomType.CLASS)
        assert len(class_atoms) == 1
        assert "MyClass" in class_atoms[0].name

    def test_class_has_file_path_metadata(self, code_bridge, simple_python_file):
        """CLASS atoms should have file_path in metadata."""
        code_bridge.index_file(simple_python_file)

        class_atoms = code_bridge.graph.find_by_type(AtomType.CLASS)
        class_atom = class_atoms[0]

        assert "file_path" in class_atom.metadata
        assert "simple.py" in class_atom.metadata["file_path"]

    def test_class_has_lineno_metadata(self, code_bridge, simple_python_file):
        """CLASS atoms should have lineno in metadata."""
        code_bridge.index_file(simple_python_file)

        class_atoms = code_bridge.graph.find_by_type(AtomType.CLASS)
        class_atom = class_atoms[0]

        assert "lineno" in class_atom.metadata
        assert isinstance(class_atom.metadata["lineno"], int)
        assert class_atom.metadata["lineno"] > 0

    def test_class_has_docstring_metadata(self, code_bridge, simple_python_file):
        """CLASS atoms should have docstring in metadata (if present)."""
        code_bridge.index_file(simple_python_file)

        class_atoms = code_bridge.graph.find_by_type(AtomType.CLASS)
        class_atom = class_atoms[0]

        assert "docstring" in class_atom.metadata
        assert "My class" in class_atom.metadata["docstring"]


class TestFunctionAtomCreation:
    """Verify FUNCTION atoms are created with correct metadata."""

    def test_function_creates_function_atoms(self, code_bridge, simple_python_file):
        """Indexing should create FUNCTION atoms for functions and methods."""
        stats = code_bridge.index_file(simple_python_file)

        # 1 standalone + 2 methods (__init__, process)
        assert stats.functions == 3
        func_atoms = code_bridge.graph.find_by_type(AtomType.FUNCTION)
        assert len(func_atoms) == 3

    def test_method_name_includes_class(self, code_bridge, simple_python_file):
        """Method atom names should include the class name."""
        code_bridge.index_file(simple_python_file)

        func_atoms = code_bridge.graph.find_by_type(AtomType.FUNCTION)
        method_names = [a.name for a in func_atoms]

        # Should have MyClass.process or similar
        assert any("MyClass" in name and "process" in name for name in method_names)

    def test_function_has_args_metadata(self, code_bridge, simple_python_file):
        """FUNCTION atoms should have args in metadata."""
        code_bridge.index_file(simple_python_file)

        func_atoms = code_bridge.graph.find_by_type(AtomType.FUNCTION)
        # Find standalone_function
        standalone = [a for a in func_atoms if "standalone" in a.name][0]

        assert "args" in standalone.metadata
        assert "x" in standalone.metadata["args"]
        assert "y" in standalone.metadata["args"]

    def test_function_has_file_and_lineno(self, code_bridge, simple_python_file):
        """FUNCTION atoms should have file_path and lineno."""
        code_bridge.index_file(simple_python_file)

        func_atoms = code_bridge.graph.find_by_type(AtomType.FUNCTION)
        for func_atom in func_atoms:
            assert "file_path" in func_atom.metadata
            assert "lineno" in func_atom.metadata


# =============================================================================
# Test: Link Creation
# =============================================================================


class TestDefinesLinks:
    """Verify FILE --DEFINES--> CLASS/FUNCTION links."""

    def test_file_defines_class_link(self, code_bridge, simple_python_file):
        """FILE should have DEFINES link to CLASS."""
        code_bridge.index_file(simple_python_file)

        file_atoms = code_bridge.graph.find_by_type(AtomType.FILE)
        class_atoms = code_bridge.graph.find_by_type(AtomType.CLASS)

        file_atom = file_atoms[0]
        class_atom = class_atoms[0]

        # Check for DEFINES link from file to class
        defines_links = code_bridge.graph.find_by_type(AtomType.DEFINES)
        file_to_class = [
            link for link in defines_links
            if file_atom.id in link.outgoing and class_atom.id in link.outgoing
        ]
        assert len(file_to_class) >= 1, "No DEFINES link from FILE to CLASS"

    def test_file_defines_standalone_function(self, code_bridge, simple_python_file):
        """FILE should have DEFINES link to standalone FUNCTION."""
        code_bridge.index_file(simple_python_file)

        file_atoms = code_bridge.graph.find_by_type(AtomType.FILE)
        func_atoms = code_bridge.graph.find_by_type(AtomType.FUNCTION)

        file_atom = file_atoms[0]
        standalone = [a for a in func_atoms if "standalone" in a.name][0]

        defines_links = code_bridge.graph.find_by_type(AtomType.DEFINES)
        file_to_func = [
            link for link in defines_links
            if file_atom.id in link.outgoing and standalone.id in link.outgoing
        ]
        assert len(file_to_func) >= 1, "No DEFINES link from FILE to standalone FUNCTION"


class TestContainsLinks:
    """Verify CLASS --CONTAINS--> FUNCTION links."""

    def test_class_contains_method_link(self, code_bridge, simple_python_file):
        """CLASS should have CONTAINS link to its methods."""
        code_bridge.index_file(simple_python_file)

        class_atoms = code_bridge.graph.find_by_type(AtomType.CLASS)
        func_atoms = code_bridge.graph.find_by_type(AtomType.FUNCTION)

        class_atom = class_atoms[0]
        method = [a for a in func_atoms if "process" in a.name][0]

        contains_links = code_bridge.graph.find_by_type(AtomType.CONTAINS)
        class_to_method = [
            link for link in contains_links
            if class_atom.id in link.outgoing and method.id in link.outgoing
        ]
        assert len(class_to_method) >= 1, "No CONTAINS link from CLASS to method"


class TestCallsLinks:
    """Verify FUNCTION --CALLS--> FUNCTION links."""

    def test_function_calls_function_link(self, code_bridge, call_graph_file):
        """CALLS links should reflect actual function calls."""
        stats = code_bridge.index_file(call_graph_file)

        assert stats.calls_links >= 3  # caller->helper, caller->process_data, process_data->helper

        func_atoms = code_bridge.graph.find_by_type(AtomType.FUNCTION)
        func_by_name = {a.name: a for a in func_atoms}

        # Find caller and helper
        caller = [a for a in func_atoms if "caller" in a.name][0]
        helper = [a for a in func_atoms if "helper" in a.name][0]

        # Check CALLS link exists
        calls_links = code_bridge.graph.find_by_type(AtomType.CALLS)
        caller_to_helper = [
            link for link in calls_links
            if caller.id in link.outgoing and helper.id in link.outgoing
        ]
        assert len(caller_to_helper) >= 1, "No CALLS link from caller to helper"


class TestInheritanceLinks:
    """Verify CLASS --INHERITANCE--> CLASS links."""

    def test_child_inherits_from_parent_link(self, code_bridge, inheritance_file):
        """INHERITANCE links should reflect class hierarchy."""
        stats = code_bridge.index_file(inheritance_file)

        assert stats.inheritance_links >= 3  # ChildA->Base, ChildB->Base, GrandChild->ChildA

        class_atoms = code_bridge.graph.find_by_type(AtomType.CLASS)

        base = [a for a in class_atoms if "BaseClass" in a.name][0]
        child_a = [a for a in class_atoms if "ChildA" in a.name][0]

        # Check INHERITANCE link: ChildA -> BaseClass
        inheritance_links = code_bridge.graph.find_by_type(AtomType.INHERITANCE)
        child_to_base = [
            link for link in inheritance_links
            if child_a.id in link.outgoing and base.id in link.outgoing
        ]
        assert len(child_to_base) >= 1, "No INHERITANCE link from ChildA to BaseClass"


# =============================================================================
# Test: Query Support
# =============================================================================


class TestCodeQueries:
    """Verify code structure queries work correctly."""

    def test_find_callers_of_function(self, code_bridge, call_graph_file):
        """query_code('callers_of', 'helper') should return caller functions."""
        code_bridge.index_file(call_graph_file)

        # Find what calls 'helper'
        callers = code_bridge.query_callers_of("helper")
        caller_names = [a.name for a in callers]

        # Both 'caller' and 'process_data' call 'helper'
        assert any("caller" in name for name in caller_names)
        assert any("process_data" in name for name in caller_names)

    def test_find_subclasses(self, code_bridge, inheritance_file):
        """query_code('subclasses_of', 'BaseClass') should return child classes."""
        code_bridge.index_file(inheritance_file)

        subclasses = code_bridge.query_subclasses_of("BaseClass")
        subclass_names = [a.name for a in subclasses]

        assert any("ChildA" in name for name in subclass_names)
        assert any("ChildB" in name for name in subclass_names)

    def test_find_methods_of_class(self, code_bridge, simple_python_file):
        """query_code('methods_of', 'MyClass') should return methods."""
        code_bridge.index_file(simple_python_file)

        methods = code_bridge.query_methods_of("MyClass")
        method_names = [a.name for a in methods]

        assert any("__init__" in name for name in method_names)
        assert any("process" in name for name in method_names)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case handling."""

    def test_syntax_error_handled(self, code_bridge, tmp_path):
        """Syntax errors should be counted, not crash."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken syntax here:")

        stats = code_bridge.index_file(bad_file)

        assert stats.parse_errors == 1
        assert stats.files == 0

    def test_empty_file_handled(self, code_bridge, tmp_path):
        """Empty files should index without error."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")

        stats = code_bridge.index_file(empty_file)

        assert stats.files == 1
        assert stats.classes == 0
        assert stats.functions == 0

    def test_directory_indexing(self, code_bridge, tmp_path):
        """Directory indexing should process all Python files."""
        # Create subdirectory structure
        subdir = tmp_path / "pkg"
        subdir.mkdir()

        (tmp_path / "a.py").write_text("class A: pass")
        (subdir / "b.py").write_text("class B: pass")
        (subdir / "__init__.py").write_text("")

        stats = code_bridge.index_directory(tmp_path)

        assert stats.files == 3
        assert stats.classes == 2  # A and B

    def test_excluded_directories_skipped(self, code_bridge, tmp_path):
        """Excluded directories should be skipped."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()

        (tmp_path / "good.py").write_text("class Good: pass")
        (pycache / "bad.py").write_text("class Bad: pass")

        stats = code_bridge.index_directory(tmp_path, exclude=["__pycache__"])

        assert stats.files == 1
        assert stats.classes == 1

        class_names = [a.name for a in code_bridge.graph.find_by_type(AtomType.CLASS)]
        assert any("Good" in name for name in class_names)
        assert not any("Bad" in name for name in class_names)


# =============================================================================
# Test: Statistics
# =============================================================================


class TestIndexStats:
    """Verify IndexStats are accurate."""

    def test_stats_count_entities(self, code_bridge, simple_python_file):
        """Stats should accurately count entities."""
        stats = code_bridge.index_file(simple_python_file)

        assert stats.files == 1
        assert stats.classes == 1
        assert stats.functions == 3  # __init__, process, standalone_function

    def test_stats_has_elapsed_time(self, code_bridge, simple_python_file):
        """Stats should include elapsed time."""
        stats = code_bridge.index_file(simple_python_file)

        assert hasattr(stats, 'elapsed_seconds')
        assert stats.elapsed_seconds >= 0


# =============================================================================
# Test: REFERS_TO Semantic Bridge
# =============================================================================


class TestRefersToLinks:
    """Verify REFERS_TO links bridge text vocabulary to code entities."""

    @pytest.fixture
    def graph_with_words(self, graph):
        """Graph pre-populated with some WORD atoms."""
        # Simulate words from text training
        graph.node("process", atom_type=AtomType.WORD)
        graph.node("data", atom_type=AtomType.WORD)
        graph.node("helper", atom_type=AtomType.WORD)
        graph.node("class", atom_type=AtomType.WORD)
        graph.node("standalone", atom_type=AtomType.WORD)
        graph.node("unrelated", atom_type=AtomType.WORD)
        return graph

    @pytest.fixture
    def bridge_with_words(self, graph_with_words):
        """CodeBridge with pre-existing WORD atoms."""
        from cortical.cognitive.code_bridge import CodeBridge
        return CodeBridge(graph_with_words)

    def test_word_refers_to_matching_function(self, bridge_with_words, simple_python_file):
        """WORD 'process' should REFERS_TO function containing 'process'."""
        bridge_with_words.index_file(simple_python_file)
        stats = bridge_with_words.create_refers_to_links()

        # Find REFERS_TO links
        refers_to_links = bridge_with_words.graph.find_by_type(AtomType.REFERS_TO)

        # 'process' word should link to 'MyClass.process' function
        process_word = bridge_with_words.graph.get_node("process")
        linked_targets = []
        for link in refers_to_links:
            if process_word.id in link.outgoing:
                for target_id in link.outgoing:
                    if target_id != process_word.id:
                        target = bridge_with_words.graph.get_atom(target_id)
                        if target:
                            linked_targets.append(target.name)

        assert any("process" in name for name in linked_targets)

    def test_word_refers_to_matching_class(self, bridge_with_words, simple_python_file):
        """WORD 'class' should REFERS_TO class containing 'class' in name."""
        bridge_with_words.index_file(simple_python_file)
        bridge_with_words.create_refers_to_links()

        refers_to_links = bridge_with_words.graph.find_by_type(AtomType.REFERS_TO)

        # 'class' word should link to 'MyClass'
        class_word = bridge_with_words.graph.get_node("class")
        linked_targets = []
        for link in refers_to_links:
            if class_word.id in link.outgoing:
                for target_id in link.outgoing:
                    if target_id != class_word.id:
                        target = bridge_with_words.graph.get_atom(target_id)
                        if target:
                            linked_targets.append(target.name)

        assert any("Class" in name for name in linked_targets)

    def test_unmatched_word_has_no_refers_to(self, bridge_with_words, simple_python_file):
        """WORD with no matching code entity should have no REFERS_TO links."""
        bridge_with_words.index_file(simple_python_file)
        bridge_with_words.create_refers_to_links()

        refers_to_links = bridge_with_words.graph.find_by_type(AtomType.REFERS_TO)

        # 'unrelated' word should not link to anything
        unrelated_word = bridge_with_words.graph.get_node("unrelated")
        links_from_unrelated = [
            link for link in refers_to_links
            if unrelated_word.id in link.outgoing
        ]

        assert len(links_from_unrelated) == 0

    def test_refers_to_stats_returned(self, bridge_with_words, simple_python_file):
        """create_refers_to_links should return statistics."""
        bridge_with_words.index_file(simple_python_file)
        stats = bridge_with_words.create_refers_to_links()

        assert hasattr(stats, 'refers_to_links')
        assert stats.refers_to_links > 0  # Should have some matches

    def test_short_words_skipped(self, graph):
        """Very short words (< 3 chars) should be skipped to avoid noise."""
        from cortical.cognitive.code_bridge import CodeBridge

        # Add short words
        graph.node("a", atom_type=AtomType.WORD)
        graph.node("do", atom_type=AtomType.WORD)
        graph.node("if", atom_type=AtomType.WORD)

        bridge = CodeBridge(graph)

        # Create a file with function named 'a'
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def a(): pass\ndef do_something(): pass")
            f.flush()
            bridge.index_file(Path(f.name))
            Path(f.name).unlink()

        stats = bridge.create_refers_to_links()

        # Short words should not create links
        a_word = graph.get_node("a")
        refers_to_links = graph.find_by_type(AtomType.REFERS_TO)
        links_from_a = [
            link for link in refers_to_links
            if a_word and a_word.id in link.outgoing
        ]

        assert len(links_from_a) == 0

    def test_case_insensitive_matching(self, graph):
        """Matching should be case-insensitive."""
        from cortical.cognitive.code_bridge import CodeBridge

        # Word in lowercase
        graph.node("myclass", atom_type=AtomType.WORD)

        bridge = CodeBridge(graph)

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("class MyClass: pass")
            f.flush()
            bridge.index_file(Path(f.name))
            Path(f.name).unlink()

        stats = bridge.create_refers_to_links()

        # Should match despite case difference
        myclass_word = graph.get_node("myclass")
        refers_to_links = graph.find_by_type(AtomType.REFERS_TO)
        links_from_myclass = [
            link for link in refers_to_links
            if myclass_word.id in link.outgoing
        ]

        assert len(links_from_myclass) > 0


class TestQueryWithRefersTo:
    """Test queries that use REFERS_TO links."""

    @pytest.fixture
    def indexed_with_refs(self, graph):
        """Graph with both text and code, linked via REFERS_TO."""
        from cortical.cognitive.code_bridge import CodeBridge

        # Add vocabulary
        graph.node("compute", atom_type=AtomType.WORD)
        graph.node("pagerank", atom_type=AtomType.WORD)
        graph.node("storage", atom_type=AtomType.WORD)

        bridge = CodeBridge(graph)

        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
class StorageBackend:
    def save(self): pass

def compute_pagerank(graph):
    """Compute PageRank scores."""
    pass
''')
            f.flush()
            bridge.index_file(Path(f.name))
            Path(f.name).unlink()

        bridge.create_refers_to_links()
        return bridge

    def test_find_code_for_word(self, indexed_with_refs):
        """Should find code entities that a word refers to."""
        results = indexed_with_refs.query_code_for_word("pagerank")

        result_names = [a.name for a in results]
        assert any("pagerank" in name.lower() for name in result_names)

    def test_find_words_for_code(self, indexed_with_refs):
        """Should find words that refer to a code entity."""
        results = indexed_with_refs.query_words_for_code("StorageBackend")

        result_names = [a.name for a in results]
        assert "storage" in result_names
