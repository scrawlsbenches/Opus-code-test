"""
Code Bridge: Converts ASTIndex to Cognitive Graph Atoms.

Stores REFERENCES to code, not code itself:
- FILE atoms point to file paths
- CLASS/FUNCTION atoms have file_path and lineno in metadata
- The actual source code stays in .py files

Example:
    >>> graph = CognitiveGraph()
    >>> bridge = CodeBridge(graph)
    >>> stats = bridge.index_directory(Path("cortical/"))
    >>> print(stats)
    IndexStats(files=200, classes=150, functions=500, ...)
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from cortical.cognitive.graph import AtomType, Atom, CognitiveGraph, TruthValue
from cortical.spark.ast_index import ASTIndex, FunctionInfo, ClassInfo


@dataclass
class IndexStats:
    """Statistics from indexing operation."""
    files: int = 0
    classes: int = 0
    functions: int = 0
    calls_links: int = 0
    inheritance_links: int = 0
    defines_links: int = 0
    contains_links: int = 0
    refers_to_links: int = 0
    parse_errors: int = 0
    elapsed_seconds: float = 0.0

    def __repr__(self) -> str:
        return (
            f"IndexStats(files={self.files}, classes={self.classes}, "
            f"functions={self.functions}, calls={self.calls_links}, "
            f"inheritance={self.inheritance_links}, refers_to={self.refers_to_links}, "
            f"errors={self.parse_errors}, elapsed={self.elapsed_seconds:.2f}s)"
        )


class CodeBridge:
    """
    Converts ASTIndex to CognitiveGraph atoms.

    Stores REFERENCES to code, not code itself.
    Atoms point to file paths and line numbers.

    Design Principles:
        1. FILE atoms use file path as name
        2. CLASS atoms use class name, store file_path/lineno in metadata
        3. FUNCTION atoms use full_name (Class.method), store args in metadata
        4. Links capture: DEFINES, CONTAINS, CALLS, INHERITANCE
    """

    def __init__(self, graph: CognitiveGraph):
        """
        Initialize CodeBridge.

        Args:
            graph: CognitiveGraph to add atoms to
        """
        self.graph = graph
        self._file_atoms: Dict[str, Atom] = {}      # path -> FILE atom
        self._class_atoms: Dict[str, Atom] = {}     # class_name -> CLASS atom
        self._function_atoms: Dict[str, Atom] = {}  # full_name -> FUNCTION atom

    def index_file(self, path: Path) -> IndexStats:
        """
        Index a single Python file.

        Args:
            path: Path to Python file

        Returns:
            IndexStats with counts
        """
        start_time = time.time()
        stats = IndexStats()

        ast_index = ASTIndex()
        success = ast_index.index_file(path)

        if not success:
            stats.parse_errors = 1
            stats.elapsed_seconds = time.time() - start_time
            return stats

        stats = self._convert_ast_to_atoms(ast_index)
        stats.elapsed_seconds = time.time() - start_time
        return stats

    def index_directory(
        self,
        path: Path,
        exclude: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> IndexStats:
        """
        Index all Python files in directory.

        Args:
            path: Directory to index
            exclude: Directory names to exclude (default: common ones)
            progress_callback: Optional callback(current, total)

        Returns:
            IndexStats with totals
        """
        start_time = time.time()
        exclude = exclude or ["__pycache__", ".git", "node_modules", "venv", ".venv", ".tox"]

        # Collect Python files, excluding specified directories
        py_files = []
        for py_file in path.rglob("*.py"):
            # Check if any parent directory is in exclude list
            skip = False
            for part in py_file.parts:
                if part in exclude:
                    skip = True
                    break
            if not skip:
                py_files.append(py_file)

        # Index with ASTIndex
        ast_index = ASTIndex()
        for i, py_file in enumerate(py_files):
            ast_index.index_file(py_file)
            if progress_callback:
                progress_callback(i + 1, len(py_files))

        stats = self._convert_ast_to_atoms(ast_index)
        stats.parse_errors = ast_index.parse_errors
        stats.elapsed_seconds = time.time() - start_time
        return stats

    def _convert_ast_to_atoms(self, ast_index: ASTIndex) -> IndexStats:
        """
        Convert ASTIndex entries to atoms and links.

        Args:
            ast_index: Populated ASTIndex

        Returns:
            IndexStats with counts
        """
        stats = IndexStats()
        stats.files = ast_index.files_indexed

        # Track atoms we create for linking
        file_atoms: Dict[str, Atom] = {}
        class_atoms: Dict[str, Atom] = {}
        function_atoms: Dict[str, Atom] = {}

        # 1. Create FILE atoms (deduped by path)
        all_file_paths: Set[str] = set()
        for class_info in ast_index.classes.values():
            all_file_paths.add(class_info.file_path)
        for func_info in ast_index.functions.values():
            all_file_paths.add(func_info.file_path)

        for file_path in all_file_paths:
            file_atom = self.graph.node(
                name=file_path,
                atom_type=AtomType.FILE,
                tv=TruthValue(strength=1.0, confidence=0.9)
            )
            file_atoms[file_path] = file_atom
            self._file_atoms[file_path] = file_atom

        # 2. Create CLASS atoms with metadata
        for class_name, class_info in ast_index.classes.items():
            class_atom = self.graph.node(
                name=class_name,
                atom_type=AtomType.CLASS,
                tv=TruthValue(strength=1.0, confidence=0.9)
            )
            # Store metadata
            class_atom.metadata["file_path"] = class_info.file_path
            class_atom.metadata["lineno"] = class_info.lineno
            if class_info.docstring:
                class_atom.metadata["docstring"] = class_info.docstring
            if class_info.bases:
                class_atom.metadata["bases"] = class_info.bases

            class_atoms[class_name] = class_atom
            self._class_atoms[class_name] = class_atom
            stats.classes += 1

            # Create DEFINES link: FILE -> CLASS
            file_atom = file_atoms.get(class_info.file_path)
            if file_atom:
                self.graph.link(
                    AtomType.DEFINES,
                    [file_atom, class_atom],
                    TruthValue(strength=1.0, confidence=0.9)
                )
                stats.defines_links += 1

        # 3. Create FUNCTION atoms with metadata
        for full_name, func_info in ast_index.functions.items():
            func_atom = self.graph.node(
                name=full_name,
                atom_type=AtomType.FUNCTION,
                tv=TruthValue(strength=1.0, confidence=0.9)
            )
            # Store metadata
            func_atom.metadata["file_path"] = func_info.file_path
            func_atom.metadata["lineno"] = func_info.lineno
            func_atom.metadata["args"] = func_info.args
            if func_info.docstring:
                func_atom.metadata["docstring"] = func_info.docstring
            if func_info.decorators:
                func_atom.metadata["decorators"] = func_info.decorators

            function_atoms[full_name] = func_atom
            self._function_atoms[full_name] = func_atom
            stats.functions += 1

            # Create CONTAINS or DEFINES link
            if func_info.class_name:
                # Method: CLASS -> FUNCTION (CONTAINS)
                class_atom = class_atoms.get(func_info.class_name)
                if class_atom:
                    self.graph.link(
                        AtomType.CONTAINS,
                        [class_atom, func_atom],
                        TruthValue(strength=1.0, confidence=0.9)
                    )
                    stats.contains_links += 1
            else:
                # Standalone function: FILE -> FUNCTION (DEFINES)
                file_atom = file_atoms.get(func_info.file_path)
                if file_atom:
                    self.graph.link(
                        AtomType.DEFINES,
                        [file_atom, func_atom],
                        TruthValue(strength=1.0, confidence=0.9)
                    )
                    stats.defines_links += 1

        # 4. Create CALLS links from call graph
        for caller_name, callees in ast_index.call_graph.items():
            caller_atom = function_atoms.get(caller_name)
            if not caller_atom:
                continue

            for callee_name in callees:
                # Try to find callee in our indexed functions
                callee_atom = function_atoms.get(callee_name)
                if not callee_atom:
                    # Try with class prefix variations
                    for full_name in function_atoms:
                        if full_name.endswith(f".{callee_name}") or full_name == callee_name:
                            callee_atom = function_atoms[full_name]
                            break

                if callee_atom:
                    self.graph.link(
                        AtomType.CALLS,
                        [caller_atom, callee_atom],
                        TruthValue(strength=1.0, confidence=0.8)
                    )
                    stats.calls_links += 1

        # 5. Create INHERITANCE links
        for parent_name, children in ast_index.inheritance.items():
            parent_atom = class_atoms.get(parent_name)
            if not parent_atom:
                continue

            for child_name in children:
                child_atom = class_atoms.get(child_name)
                if child_atom:
                    self.graph.link(
                        AtomType.INHERITANCE,
                        [child_atom, parent_atom],
                        TruthValue(strength=1.0, confidence=0.9)
                    )
                    stats.inheritance_links += 1

        return stats

    # =========================================================================
    # Query Methods
    # =========================================================================

    def query_callers_of(self, function_name: str) -> List[Atom]:
        """
        Find all functions that call the given function.

        Args:
            function_name: Name of function to find callers for

        Returns:
            List of FUNCTION atoms that call the target
        """
        # Find the target function atom
        target_atom = None
        for name, atom in self._function_atoms.items():
            if function_name in name:
                target_atom = atom
                break

        if not target_atom:
            return []

        # Find CALLS links pointing to target
        callers = []
        calls_links = self.graph.find_by_type(AtomType.CALLS)
        for link in calls_links:
            if target_atom.id in link.outgoing:
                # Find the caller (the other atom in the link)
                for atom_id in link.outgoing:
                    if atom_id != target_atom.id:
                        caller = self.graph.get_atom(atom_id)
                        if caller and caller.atom_type == AtomType.FUNCTION:
                            callers.append(caller)

        return callers

    def query_subclasses_of(self, class_name: str) -> List[Atom]:
        """
        Find all classes that inherit from the given class.

        Args:
            class_name: Name of parent class

        Returns:
            List of CLASS atoms that inherit from target
        """
        # Find the parent class atom
        parent_atom = self._class_atoms.get(class_name)
        if not parent_atom:
            return []

        # Find INHERITANCE links pointing to parent
        subclasses = []
        inheritance_links = self.graph.find_by_type(AtomType.INHERITANCE)
        for link in inheritance_links:
            if parent_atom.id in link.outgoing:
                # Find the child (the other atom in the link)
                for atom_id in link.outgoing:
                    if atom_id != parent_atom.id:
                        child = self.graph.get_atom(atom_id)
                        if child and child.atom_type == AtomType.CLASS:
                            subclasses.append(child)

        return subclasses

    def query_methods_of(self, class_name: str) -> List[Atom]:
        """
        Find all methods of the given class.

        Args:
            class_name: Name of class

        Returns:
            List of FUNCTION atoms that are methods of the class
        """
        # Find the class atom
        class_atom = self._class_atoms.get(class_name)
        if not class_atom:
            return []

        # Find CONTAINS links from this class
        methods = []
        contains_links = self.graph.find_by_type(AtomType.CONTAINS)
        for link in contains_links:
            if class_atom.id in link.outgoing:
                # Find the method (the other atom in the link)
                for atom_id in link.outgoing:
                    if atom_id != class_atom.id:
                        method = self.graph.get_atom(atom_id)
                        if method and method.atom_type == AtomType.FUNCTION:
                            methods.append(method)

        return methods

    def query_defined_in(self, file_path: str) -> List[Atom]:
        """
        Find all entities defined in the given file.

        Args:
            file_path: Path to file

        Returns:
            List of CLASS and FUNCTION atoms defined in the file
        """
        # Find the file atom
        file_atom = None
        for path, atom in self._file_atoms.items():
            if file_path in path:
                file_atom = atom
                break

        if not file_atom:
            return []

        # Find DEFINES links from this file
        entities = []
        defines_links = self.graph.find_by_type(AtomType.DEFINES)
        for link in defines_links:
            if file_atom.id in link.outgoing:
                for atom_id in link.outgoing:
                    if atom_id != file_atom.id:
                        entity = self.graph.get_atom(atom_id)
                        if entity:
                            entities.append(entity)

        return entities

    # =========================================================================
    # REFERS_TO Semantic Bridge
    # =========================================================================

    def create_refers_to_links(self, min_word_length: int = 3) -> IndexStats:
        """
        Create REFERS_TO links between WORD atoms and CODE atoms.

        This bridges natural language vocabulary to code entities:
        - "pagerank" WORD -> "compute_pagerank" FUNCTION
        - "storage" WORD -> "StorageBackend" CLASS

        Matching rules:
        - Case-insensitive substring match
        - Skip words shorter than min_word_length (avoid noise)
        - Match quality affects link strength

        Args:
            min_word_length: Minimum word length to consider (default: 3)

        Returns:
            IndexStats with refers_to_links count
        """
        stats = IndexStats()
        start_time = time.time()

        # Get all WORD atoms
        word_atoms = self.graph.find_by_type(AtomType.WORD)

        # Get all CODE atoms (CLASS, FUNCTION, FILE, MODULE)
        code_types = [AtomType.CLASS, AtomType.FUNCTION, AtomType.FILE, AtomType.MODULE]
        code_atoms: List[Atom] = []
        for code_type in code_types:
            code_atoms.extend(self.graph.find_by_type(code_type))

        # Build index of code names for faster matching
        # Map lowercase name -> (atom, original_name)
        code_index: Dict[str, List[Atom]] = {}
        for atom in code_atoms:
            if not atom.name:
                continue
            # Extract meaningful parts from name
            # e.g., "MyClass.process" -> ["myclass", "process"]
            name_parts = self._extract_name_parts(atom.name)
            for part in name_parts:
                if part not in code_index:
                    code_index[part] = []
                code_index[part].append(atom)

        # Create REFERS_TO links
        for word_atom in word_atoms:
            word = word_atom.name.lower()

            # Skip short words
            if len(word) < min_word_length:
                continue

            # Find matching code entities
            matched_atoms: Set[str] = set()  # Track by ID to avoid duplicates

            # Direct match: word is a name part
            if word in code_index:
                for code_atom in code_index[word]:
                    if code_atom.id not in matched_atoms:
                        self._create_refers_to_link(word_atom, code_atom, match_type="exact")
                        matched_atoms.add(code_atom.id)
                        stats.refers_to_links += 1

            # Substring match: word appears in name
            for name_part, atoms in code_index.items():
                if word in name_part and word != name_part:
                    for code_atom in atoms:
                        if code_atom.id not in matched_atoms:
                            self._create_refers_to_link(word_atom, code_atom, match_type="substring")
                            matched_atoms.add(code_atom.id)
                            stats.refers_to_links += 1

        stats.elapsed_seconds = time.time() - start_time
        return stats

    def _extract_name_parts(self, name: str) -> List[str]:
        """
        Extract meaningful parts from a code entity name.

        Examples:
            "MyClass.process_data" -> ["myclass", "my", "class", "process", "data"]
            "compute_pagerank" -> ["compute", "pagerank"]
            "cortical/got/api.py" -> ["cortical", "got", "api"]
        """
        import re

        # Split on non-alphanumeric characters
        parts = re.split(r'[^a-zA-Z0-9]+', name)

        # Also split camelCase, but keep the full word too
        expanded = []
        for part in parts:
            if part:
                # Add full part as-is (lowercased)
                expanded.append(part.lower())

                # Split camelCase: "MyClass" -> ["My", "Class"]
                camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', part)
                if camel_parts and len(camel_parts) > 1:
                    # Only add individual parts if there's more than one
                    expanded.extend(p.lower() for p in camel_parts)

        return [p for p in expanded if len(p) >= 2]  # Filter very short parts

    def _create_refers_to_link(
        self,
        word_atom: Atom,
        code_atom: Atom,
        match_type: str = "exact"
    ) -> Atom:
        """
        Create a REFERS_TO link from WORD to CODE atom.

        Args:
            word_atom: Source WORD atom
            code_atom: Target CODE atom
            match_type: "exact" or "substring" (affects strength)

        Returns:
            The created link atom
        """
        # Strength based on match quality
        strength = 0.9 if match_type == "exact" else 0.7
        confidence = 0.8 if match_type == "exact" else 0.6

        return self.graph.link(
            AtomType.REFERS_TO,
            [word_atom, code_atom],
            TruthValue(strength=strength, confidence=confidence)
        )

    def query_code_for_word(self, word: str) -> List[Atom]:
        """
        Find code entities that a word refers to.

        Args:
            word: The vocabulary word

        Returns:
            List of CODE atoms (CLASS, FUNCTION, etc.) the word refers to
        """
        word_atom = self.graph.get_node(word.lower())
        if not word_atom:
            return []

        # Find REFERS_TO links from this word
        code_entities = []
        refers_to_links = self.graph.find_by_type(AtomType.REFERS_TO)
        for link in refers_to_links:
            if word_atom.id in link.outgoing:
                for atom_id in link.outgoing:
                    if atom_id != word_atom.id:
                        target = self.graph.get_atom(atom_id)
                        if target and target.atom_type in [
                            AtomType.CLASS, AtomType.FUNCTION,
                            AtomType.FILE, AtomType.MODULE
                        ]:
                            code_entities.append(target)

        return code_entities

    def query_words_for_code(self, code_name: str) -> List[Atom]:
        """
        Find vocabulary words that refer to a code entity.

        Args:
            code_name: Name of the code entity

        Returns:
            List of WORD atoms that refer to the code entity
        """
        # Find the code atom
        code_atom = None
        for atom in self.graph.find_by_type(AtomType.CLASS):
            if code_name in atom.name:
                code_atom = atom
                break
        if not code_atom:
            for atom in self.graph.find_by_type(AtomType.FUNCTION):
                if code_name in atom.name:
                    code_atom = atom
                    break

        if not code_atom:
            return []

        # Find REFERS_TO links pointing to this code
        words = []
        refers_to_links = self.graph.find_by_type(AtomType.REFERS_TO)
        for link in refers_to_links:
            if code_atom.id in link.outgoing:
                for atom_id in link.outgoing:
                    if atom_id != code_atom.id:
                        word = self.graph.get_atom(atom_id)
                        if word and word.atom_type == AtomType.WORD:
                            words.append(word)

        return words
