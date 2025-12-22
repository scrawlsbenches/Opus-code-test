"""
AST Index
=========

AST-based code index for structural analysis of Python source files.

Indexes:
- Classes with inheritance, methods, attributes
- Functions with arguments, decorators, calls
- Imports and their usage
- Call graph (who calls what)
- Inheritance relationships

Example:
    >>> index = ASTIndex()
    >>> index.index_directory(Path("cortical/"))
    >>> callers = index.find_callers("compute_pagerank")
    >>> class_info = index.find_class("NGramModel")
"""

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple


@dataclass
class FunctionInfo:
    """Information about a function/method."""
    name: str
    file_path: str
    lineno: int
    args: List[str]
    decorators: List[str]
    class_name: Optional[str] = None
    docstring: Optional[str] = None
    calls: List[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        if self.class_name:
            return f"{self.class_name}.{self.name}"
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'name': self.name,
            'file_path': self.file_path,
            'lineno': self.lineno,
            'args': self.args,
            'decorators': self.decorators,
            'class_name': self.class_name,
            'docstring': self.docstring,
            'calls': self.calls,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FunctionInfo':
        """Create from dict."""
        return cls(
            name=data['name'],
            file_path=data['file_path'],
            lineno=data['lineno'],
            args=data['args'],
            decorators=data['decorators'],
            class_name=data.get('class_name'),
            docstring=data.get('docstring'),
            calls=data.get('calls', []),
        )


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    file_path: str
    lineno: int
    bases: List[str]
    methods: List[str]
    attributes: Set[str]
    decorators: List[str]
    docstring: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'name': self.name,
            'file_path': self.file_path,
            'lineno': self.lineno,
            'bases': self.bases,
            'methods': self.methods,
            'attributes': list(self.attributes),  # Set -> list for JSON
            'decorators': self.decorators,
            'docstring': self.docstring,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassInfo':
        """Create from dict."""
        return cls(
            name=data['name'],
            file_path=data['file_path'],
            lineno=data['lineno'],
            bases=data['bases'],
            methods=data['methods'],
            attributes=set(data['attributes']),  # list -> Set
            decorators=data['decorators'],
            docstring=data.get('docstring'),
        )


@dataclass
class ImportInfo:
    """Information about an import."""
    module: str
    names: List[str]
    file_path: str
    lineno: int
    is_from: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'module': self.module,
            'names': self.names,
            'file_path': self.file_path,
            'lineno': self.lineno,
            'is_from': self.is_from,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImportInfo':
        """Create from dict."""
        return cls(
            module=data['module'],
            names=data['names'],
            file_path=data['file_path'],
            lineno=data['lineno'],
            is_from=data.get('is_from', False),
        )


class ASTIndex:
    """
    AST-based code index for structural analysis.

    Indexes:
    - Classes with inheritance, methods, attributes
    - Functions with arguments, decorators, calls
    - Imports and their usage
    - Call graph (who calls what)
    """

    def __init__(self):
        self.functions: Dict[str, FunctionInfo] = {}  # full_name -> info
        self.classes: Dict[str, ClassInfo] = {}       # name -> info
        self.imports: List[ImportInfo] = []

        # Indices for fast lookup
        self.functions_by_file: Dict[str, List[str]] = defaultdict(list)
        self.classes_by_file: Dict[str, List[str]] = defaultdict(list)
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)  # caller -> callees
        self.reverse_call_graph: Dict[str, Set[str]] = defaultdict(set)  # callee -> callers
        self.inheritance: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self.module_index: Dict[str, Set[str]] = defaultdict(set)  # module -> files that import it

        # Attribute index: class -> attributes defined
        self.class_attributes: Dict[str, Set[str]] = defaultdict(set)

        # Stats
        self.files_indexed = 0
        self.parse_errors = 0

    def index_file(self, file_path: Path) -> bool:
        """
        Index a single Python file.

        Returns:
            True if successful, False if parse error
        """
        try:
            code = file_path.read_text(encoding='utf-8', errors='replace')
            tree = ast.parse(code, filename=str(file_path))
        except SyntaxError:
            self.parse_errors += 1
            return False
        except Exception:
            self.parse_errors += 1
            return False

        self.files_indexed += 1
        rel_path = str(file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._index_class(node, rel_path)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Determine if this is a method or standalone function
                class_name = self._find_enclosing_class(tree, node)
                self._index_function(node, rel_path, class_name)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info = ImportInfo(
                        module=alias.name,
                        names=[alias.asname or alias.name],
                        file_path=rel_path,
                        lineno=node.lineno,
                        is_from=False
                    )
                    self.imports.append(info)
                    self.module_index[alias.name].add(rel_path)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names = [a.name for a in node.names]
                    info = ImportInfo(
                        module=node.module,
                        names=names,
                        file_path=rel_path,
                        lineno=node.lineno,
                        is_from=True
                    )
                    self.imports.append(info)
                    self.module_index[node.module].add(rel_path)

        return True

    def _find_enclosing_class(self, tree: ast.AST, target_node: ast.AST) -> Optional[str]:
        """Find the class that contains a function node."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in ast.walk(node):
                    if child is target_node:
                        return node.name
        return None

    def _index_class(self, node: ast.ClassDef, file_path: str) -> None:
        """Index a class definition."""
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                bases.append(str(base))

        methods = []
        attributes = set()

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
                # Extract self.attr assignments in __init__
                if item.name == '__init__':
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if (isinstance(target, ast.Attribute) and
                                    isinstance(target.value, ast.Name) and
                                    target.value.id == 'self'):
                                    attributes.add(target.attr)
                        elif isinstance(stmt, ast.AnnAssign):
                            if (isinstance(stmt.target, ast.Attribute) and
                                isinstance(stmt.target.value, ast.Name) and
                                stmt.target.value.id == 'self'):
                                attributes.add(stmt.target.attr)

        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                pass

        docstring = ast.get_docstring(node)

        info = ClassInfo(
            name=node.name,
            file_path=file_path,
            lineno=node.lineno,
            bases=bases,
            methods=methods,
            attributes=attributes,
            decorators=decorators,
            docstring=docstring
        )

        self.classes[node.name] = info
        self.classes_by_file[file_path].append(node.name)
        self.class_attributes[node.name] = attributes

        # Build inheritance index
        for base in bases:
            # Handle simple base names (not qualified)
            base_name = base.split('.')[-1] if '.' in base else base
            self.inheritance[base_name].add(node.name)

    def _index_function(self, node: ast.FunctionDef, file_path: str, class_name: Optional[str]) -> None:
        """Index a function/method definition."""
        args = []
        for arg in node.args.args:
            if arg.arg != 'self' and arg.arg != 'cls':
                args.append(arg.arg)

        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                pass

        # Find function calls within this function
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._get_call_name(child)
                if call_name:
                    calls.append(call_name)

        docstring = ast.get_docstring(node)

        info = FunctionInfo(
            name=node.name,
            file_path=file_path,
            lineno=node.lineno,
            args=args,
            decorators=decorators,
            class_name=class_name,
            docstring=docstring,
            calls=calls
        )

        full_name = info.full_name
        self.functions[full_name] = info
        self.functions_by_file[file_path].append(full_name)

        # Build call graph
        for call in calls:
            self.call_graph[full_name].add(call)
            self.reverse_call_graph[call].add(full_name)

    def _get_call_name(self, node: ast.Call) -> Optional[str]:
        """Extract the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle method calls like self.foo() or obj.bar()
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        return None

    def index_directory(self, directory: Path, extensions: List[str] = None,
                       verbose: bool = False) -> 'ASTIndex':
        """
        Index all Python files in a directory.

        Args:
            directory: Root directory to index
            extensions: File extensions to include (default: ['.py'])
            verbose: Print progress

        Returns:
            self for method chaining
        """
        extensions = extensions or ['.py']

        files = []
        for ext in extensions:
            files.extend(directory.rglob(f'*{ext}'))

        for i, file_path in enumerate(files):
            if verbose and i % 100 == 0:
                print(f"  Indexing: {i}/{len(files)} files...", end='\r')
            self.index_file(file_path)

        if verbose:
            print(f"  Indexed {self.files_indexed} files ({self.parse_errors} parse errors)")

        return self

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def find_callers(self, function_name: str) -> List[Tuple[str, str, int]]:
        """
        Find all functions that call the given function.

        Returns:
            List of (caller_name, file_path, lineno)
        """
        results = []

        # Check both exact match and partial match (for method calls)
        callers = self.reverse_call_graph.get(function_name, set())

        # Also check for method-style calls
        for call, caller_set in self.reverse_call_graph.items():
            if call.endswith(f'.{function_name}'):
                callers = callers | caller_set

        for caller in callers:
            if caller in self.functions:
                info = self.functions[caller]
                results.append((caller, info.file_path, info.lineno))

        return sorted(results, key=lambda x: (x[1], x[2]))

    def find_class(self, class_name: str) -> Optional[ClassInfo]:
        """Find a class by name."""
        return self.classes.get(class_name)

    def find_function(self, function_name: str) -> List[FunctionInfo]:
        """Find functions by name (supports partial match)."""
        results = []
        for full_name, info in self.functions.items():
            if info.name == function_name or full_name == function_name:
                results.append(info)
        return results

    def get_inheritance_tree(self, class_name: str) -> Dict[str, Any]:
        """
        Get inheritance tree for a class.

        Returns:
            Nested dict showing class hierarchy
        """
        children = self.inheritance.get(class_name, set())

        tree = {'name': class_name, 'children': []}
        for child in sorted(children):
            tree['children'].append(self.get_inheritance_tree(child))

        return tree

    def get_class_attributes(self, class_name: str) -> Set[str]:
        """Get all attributes defined in a class."""
        attrs = self.class_attributes.get(class_name, set())

        # Also get inherited attributes
        if class_name in self.classes:
            for base in self.classes[class_name].bases:
                base_name = base.split('.')[-1]
                attrs = attrs | self.get_class_attributes(base_name)

        return attrs

    def find_imports_of(self, module_name: str) -> List[Tuple[str, int]]:
        """
        Find all files that import a module.

        Returns:
            List of (file_path, lineno)
        """
        results = []
        for imp in self.imports:
            if imp.module == module_name or imp.module.startswith(f'{module_name}.'):
                results.append((imp.file_path, imp.lineno))
        return sorted(set(results))

    def get_stats(self) -> Dict[str, int]:
        """Get index statistics."""
        return {
            'files': self.files_indexed,
            'classes': len(self.classes),
            'functions': len(self.functions),
            'imports': len(self.imports),
            'call_edges': sum(len(v) for v in self.call_graph.values()),
            'inheritance_edges': sum(len(v) for v in self.inheritance.values()),
            'parse_errors': self.parse_errors,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire index to JSON-serializable dict."""
        return {
            'functions': {k: v.to_dict() for k, v in self.functions.items()},
            'classes': {k: v.to_dict() for k, v in self.classes.items()},
            'imports': [imp.to_dict() for imp in self.imports],
            'call_graph': {k: list(v) for k, v in self.call_graph.items()},
            'reverse_call_graph': {k: list(v) for k, v in self.reverse_call_graph.items()},
            'inheritance': {k: list(v) for k, v in self.inheritance.items()},
            'class_attributes': {k: list(v) for k, v in self.class_attributes.items()},
            'functions_by_file': dict(self.functions_by_file),
            'classes_by_file': dict(self.classes_by_file),
            'files_indexed': self.files_indexed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ASTIndex':
        """Create index from dict."""
        index = cls()
        index.functions = {
            k: FunctionInfo.from_dict(v)
            for k, v in data['functions'].items()
        }
        index.classes = {
            k: ClassInfo.from_dict(v)
            for k, v in data['classes'].items()
        }
        index.imports = [
            ImportInfo.from_dict(imp)
            for imp in data['imports']
        ]
        index.call_graph = defaultdict(set, {k: set(v) for k, v in data['call_graph'].items()})
        index.reverse_call_graph = defaultdict(set, {k: set(v) for k, v in data['reverse_call_graph'].items()})
        index.inheritance = defaultdict(set, {k: set(v) for k, v in data['inheritance'].items()})
        index.class_attributes = defaultdict(set, {k: set(v) for k, v in data['class_attributes'].items()})
        index.functions_by_file = defaultdict(list, data.get('functions_by_file', {}))
        index.classes_by_file = defaultdict(list, data.get('classes_by_file', {}))
        index.files_indexed = data['files_indexed']
        return index

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"ASTIndex(files={stats['files']}, "
            f"classes={stats['classes']}, "
            f"functions={stats['functions']})"
        )
