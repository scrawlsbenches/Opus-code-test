#!/usr/bin/env python3
"""
SparkCodeIntelligence - Hybrid AST + N-gram Code Intelligence Engine
=====================================================================

A powerful code intelligence system combining:
1. AST-based structural analysis (classes, functions, calls, imports)
2. Code-aware tokenization (preserves punctuation and operators)
3. N-gram language model for pattern prediction
4. Semantic queries (find callers, inheritance, related code)

This is SparkSLM v2 - designed specifically for code understanding.

Usage:
    python scripts/spark_code_intelligence.py train [--verbose]
    python scripts/spark_code_intelligence.py complete "self."
    python scripts/spark_code_intelligence.py find-callers "method_name"
    python scripts/spark_code_intelligence.py find-class "ClassName"
    python scripts/spark_code_intelligence.py inheritance "ClassName"
    python scripts/spark_code_intelligence.py imports "module_name"
    python scripts/spark_code_intelligence.py related "file.py"
    python scripts/spark_code_intelligence.py query "natural language query"
    python scripts/spark_code_intelligence.py interactive
"""

import ast
import json
import os
import re
import sys
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cortical.spark.ngram import NGramModel


# =============================================================================
# CODE TOKENIZER - Preserves punctuation and structure
# =============================================================================

class CodeTokenizer:
    """
    Code-aware tokenizer that preserves punctuation and operators.

    Unlike natural language tokenizers, this:
    - Keeps . ( ) [ ] { } : , as separate tokens
    - Preserves operators: == != >= <= += -= etc.
    - Splits camelCase and snake_case
    - Keeps string literals as single tokens
    """

    # Operators to preserve as tokens
    OPERATORS = frozenset([
        '==', '!=', '>=', '<=', '+=', '-=', '*=', '/=', '//=', '%=',
        '**', '//', '->', '::', '...', '&&', '||', '<<', '>>', '**=',
        '&=', '|=', '^=', '>>=', '<<=', '@=',
    ])

    # Single-char punctuation to preserve
    PUNCTUATION = frozenset('.()[]{}:,;@#=+-*/<>|&^~%!')

    def __init__(self,
                 split_identifiers: bool = True,
                 preserve_case: bool = False,
                 include_strings: bool = False):
        """
        Initialize code tokenizer.

        Args:
            split_identifiers: Split camelCase/snake_case into parts
            preserve_case: Keep original case (default: lowercase)
            include_strings: Include string literal contents
        """
        self.split_identifiers = split_identifiers
        self.preserve_case = preserve_case
        self.include_strings = include_strings

        # Build operator pattern (longest first for correct matching)
        sorted_ops = sorted(self.OPERATORS, key=len, reverse=True)
        escaped_ops = [re.escape(op) for op in sorted_ops]
        self._op_pattern = re.compile('|'.join(escaped_ops))

        # Pattern for identifiers (including underscores)
        self._ident_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')

        # Pattern for numbers
        self._num_pattern = re.compile(r'\d+\.?\d*')

        # Pattern for strings
        self._string_pattern = re.compile(r'(""".*?"""|\'\'\'.*?\'\'\'|".*?"|\'.*?\')', re.DOTALL)

    def tokenize(self, code: str) -> List[str]:
        """
        Tokenize code preserving structure.

        Args:
            code: Source code string

        Returns:
            List of tokens
        """
        tokens = []

        # Handle strings first (replace with placeholder to avoid tokenizing contents)
        string_map = {}
        if not self.include_strings:
            def replace_string(match):
                placeholder = f"__STRING_{len(string_map)}__"
                string_map[placeholder] = match.group(0)
                return placeholder
            code = self._string_pattern.sub(replace_string, code)

        # Split by whitespace first
        parts = code.split()

        for part in parts:
            tokens.extend(self._tokenize_part(part))

        # Optionally restore string placeholders
        if self.include_strings:
            tokens = [string_map.get(t, t) for t in tokens]
        else:
            # Remove string placeholders
            tokens = [t for t in tokens if not t.startswith('__STRING_')]

        return tokens

    def _tokenize_part(self, part: str) -> List[str]:
        """Tokenize a single whitespace-separated part."""
        result = []
        i = 0

        while i < len(part):
            # Check for multi-char operators
            matched_op = None
            for op_len in [3, 2]:  # Check 3-char then 2-char operators
                candidate = part[i:i+op_len]
                if candidate in self.OPERATORS:
                    matched_op = candidate
                    break

            if matched_op:
                result.append(matched_op)
                i += len(matched_op)
                continue

            # Check for single punctuation
            if part[i] in self.PUNCTUATION:
                result.append(part[i])
                i += 1
                continue

            # Check for identifier
            ident_match = self._ident_pattern.match(part, i)
            if ident_match:
                ident = ident_match.group(0)
                if self.split_identifiers:
                    result.extend(self._split_identifier(ident))
                else:
                    result.append(ident if self.preserve_case else ident.lower())
                i = ident_match.end()
                continue

            # Check for number
            num_match = self._num_pattern.match(part, i)
            if num_match:
                result.append(num_match.group(0))
                i = num_match.end()
                continue

            # Skip unknown character
            i += 1

        return result

    def _split_identifier(self, ident: str) -> List[str]:
        """Split camelCase and snake_case identifiers."""
        result = []

        # First add the full identifier
        full = ident if self.preserve_case else ident.lower()
        result.append(full)

        # Split on underscores
        parts = ident.split('_')

        for part in parts:
            if not part:
                continue

            # Split camelCase
            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', part)

            for cp in camel_parts:
                lower = cp if self.preserve_case else cp.lower()
                if lower != full and lower not in result:
                    result.append(lower)

        return result


# =============================================================================
# AST INDEX - Structural code analysis
# =============================================================================

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

        # Track current class context
        current_class = None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._index_class(node, rel_path)

            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
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
            except:
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
            except:
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
            except:
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


# =============================================================================
# SPARK CODE INTELLIGENCE - Hybrid Engine
# =============================================================================

class SparkCodeIntelligence:
    """
    Hybrid AST + N-gram code intelligence engine.

    Combines structural understanding (AST) with pattern learning (n-grams)
    for powerful code intelligence:

    - Context-aware completion (self. -> class attributes)
    - Semantic search (find callers, inheritance, related files)
    - Code pattern prediction
    - Anomaly detection
    """

    MODEL_FILE = ".spark_intelligence_model.json"

    def __init__(self, root_dir: Path = None):
        """
        Initialize the intelligence engine.

        Args:
            root_dir: Root directory of the codebase (default: current)
        """
        self.root_dir = root_dir or Path.cwd()

        # Components
        self.tokenizer = CodeTokenizer(split_identifiers=True)
        self.ast_index = ASTIndex()
        self.ngram_model = NGramModel(n=3, smoothing=0.1)

        # Training state
        self.trained = False
        self.training_time = 0.0

        # File content cache for related file analysis
        self._file_tokens: Dict[str, List[str]] = {}

    def train(self, extensions: List[str] = None, verbose: bool = True) -> 'SparkCodeIntelligence':
        """
        Train the intelligence engine on the codebase.

        Args:
            extensions: File extensions to include
            verbose: Print progress

        Returns:
            self for method chaining
        """
        extensions = extensions or ['.py']
        start_time = time.time()

        if verbose:
            print("=" * 70)
            print("SparkCodeIntelligence Training")
            print("=" * 70)

        # Phase 1: AST Indexing
        if verbose:
            print("\n📊 Phase 1: AST Indexing...")
        self.ast_index.index_directory(self.root_dir, extensions, verbose)

        # Phase 2: Code Tokenization + N-gram Training
        if verbose:
            print("\n📝 Phase 2: N-gram Training...")

        token_lists = []
        file_count = 0

        for ext in extensions:
            for file_path in self.root_dir.rglob(f'*{ext}'):
                try:
                    code = file_path.read_text(encoding='utf-8', errors='replace')
                    tokens = self.tokenizer.tokenize(code)
                    if tokens:
                        token_lists.append(tokens)
                        self._file_tokens[str(file_path)] = tokens
                        file_count += 1
                except Exception:
                    continue

        if verbose:
            print(f"  Tokenized {file_count} files")

        self.ngram_model.train_on_tokens(token_lists)
        self.ngram_model.finalize()  # Build fallback cache

        self.training_time = time.time() - start_time
        self.trained = True

        if verbose:
            print(f"\n✅ Training complete in {self.training_time:.2f}s")
            stats = self.get_stats()
            print(f"\n📈 Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value:,}")

        return self

    def save(self, path: str = None) -> None:
        """Save trained model to JSON (git-friendly format)."""
        path = path or self.MODEL_FILE

        data = {
            'version': 1,  # Schema version for future compatibility
            'ast_index': {
                'functions': {k: v.to_dict() for k, v in self.ast_index.functions.items()},
                'classes': {k: v.to_dict() for k, v in self.ast_index.classes.items()},
                'imports': [imp.to_dict() for imp in self.ast_index.imports],
                'call_graph': {k: list(v) for k, v in self.ast_index.call_graph.items()},
                'reverse_call_graph': {k: list(v) for k, v in self.ast_index.reverse_call_graph.items()},
                'inheritance': {k: list(v) for k, v in self.ast_index.inheritance.items()},
                'class_attributes': {k: list(v) for k, v in self.ast_index.class_attributes.items()},
                'functions_by_file': dict(self.ast_index.functions_by_file),
                'classes_by_file': dict(self.ast_index.classes_by_file),
                'files_indexed': self.ast_index.files_indexed,
            },
            'ngram_model': {
                'n': self.ngram_model.n,
                'smoothing': self.ngram_model.smoothing,
                'vocab': list(self.ngram_model.vocab),
                'counts': {' '.join(k): dict(v) for k, v in self.ngram_model.counts.items()},
                'context_totals': {' '.join(k): v for k, v in self.ngram_model.context_totals.items()},
                'total_tokens': self.ngram_model.total_tokens,
                'total_documents': self.ngram_model.total_documents,
                'cached_frequent_words': self.ngram_model._cached_frequent_words,
            },
            'file_tokens': self._file_tokens,
            'training_time': self.training_time,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str = None) -> 'SparkCodeIntelligence':
        """Load trained model from JSON."""
        path = path or self.MODEL_FILE

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Restore AST index - convert dicts back to dataclasses
        self.ast_index.functions = {
            k: FunctionInfo.from_dict(v)
            for k, v in data['ast_index']['functions'].items()
        }
        self.ast_index.classes = {
            k: ClassInfo.from_dict(v)
            for k, v in data['ast_index']['classes'].items()
        }
        self.ast_index.imports = [
            ImportInfo.from_dict(imp)
            for imp in data['ast_index']['imports']
        ]
        self.ast_index.call_graph = defaultdict(set, {k: set(v) for k, v in data['ast_index']['call_graph'].items()})
        self.ast_index.reverse_call_graph = defaultdict(set, {k: set(v) for k, v in data['ast_index']['reverse_call_graph'].items()})
        self.ast_index.inheritance = defaultdict(set, {k: set(v) for k, v in data['ast_index']['inheritance'].items()})
        self.ast_index.class_attributes = defaultdict(set, {k: set(v) for k, v in data['ast_index']['class_attributes'].items()})
        self.ast_index.functions_by_file = defaultdict(list, data['ast_index'].get('functions_by_file', {}))
        self.ast_index.classes_by_file = defaultdict(list, data['ast_index'].get('classes_by_file', {}))
        self.ast_index.files_indexed = data['ast_index']['files_indexed']

        # Restore n-gram model
        ngram_data = data['ngram_model']
        self.ngram_model = NGramModel(n=ngram_data['n'], smoothing=ngram_data['smoothing'])
        self.ngram_model.vocab = set(ngram_data['vocab'])
        self.ngram_model.total_tokens = ngram_data['total_tokens']
        self.ngram_model.total_documents = ngram_data['total_documents']
        self.ngram_model._cached_frequent_words = ngram_data.get('cached_frequent_words')

        for key, counts in ngram_data['counts'].items():
            context = tuple(key.split())
            self.ngram_model.counts[context] = Counter(counts)

        for key, total in ngram_data['context_totals'].items():
            context = tuple(key.split())
            self.ngram_model.context_totals[context] = total

        self._file_tokens = data.get('file_tokens', {})
        self.training_time = data.get('training_time', 0)
        self.trained = True

        return self

    # =========================================================================
    # COMPLETION - Context-aware code completion
    # =========================================================================

    def complete(self, prefix: str, top_n: int = 10) -> List[Tuple[str, float, str]]:
        """
        Context-aware code completion.

        Args:
            prefix: Code prefix to complete
            top_n: Number of suggestions

        Returns:
            List of (suggestion, confidence, source) tuples
            source is 'ast' or 'ngram'
        """
        results = []
        tokens = self.tokenizer.tokenize(prefix)

        if not tokens:
            return []

        # Check for special contexts
        last_token = tokens[-1] if tokens else ''
        second_last = tokens[-2] if len(tokens) > 1 else ''

        # Context 1: self. -> class attributes
        if second_last == 'self' and last_token == '.':
            # Try to find current class from context
            for class_name, attrs in self.ast_index.class_attributes.items():
                for attr in sorted(attrs)[:top_n]:
                    results.append((attr, 0.9, 'ast:attribute'))
            if results:
                return results[:top_n]

        # Context 2: ClassName. -> methods/attributes
        if last_token == '.' and second_last in self.ast_index.classes:
            class_info = self.ast_index.classes[second_last]
            # Add methods
            for method in class_info.methods[:top_n // 2]:
                results.append((method + '()', 0.85, 'ast:method'))
            # Add attributes
            for attr in sorted(class_info.attributes)[:top_n // 2]:
                results.append((attr, 0.8, 'ast:attribute'))
            if results:
                return results[:top_n]

        # Context 3: from X import -> module contents
        if len(tokens) >= 3 and tokens[-3] == 'from' and tokens[-1] == 'import':
            module = tokens[-2]
            # Find what's exported from this module
            for imp in self.ast_index.imports:
                if imp.module == module and imp.is_from:
                    for name in imp.names:
                        results.append((name, 0.8, 'ast:import'))
            if results:
                return results[:top_n]

        # Context 4: import -> known modules
        if last_token == 'import' or (len(tokens) >= 2 and tokens[-2] == 'import'):
            modules = set()
            for imp in self.ast_index.imports:
                modules.add(imp.module.split('.')[0])
            for module in sorted(modules)[:top_n]:
                results.append((module, 0.7, 'ast:module'))
            if results:
                return results[:top_n]

        # Fallback: N-gram predictions
        predictions = self.ngram_model.predict(tokens, top_k=top_n)
        for word, prob in predictions:
            results.append((word, prob, 'ngram'))

        return results[:top_n]

    # =========================================================================
    # SEMANTIC QUERIES
    # =========================================================================

    def find_callers(self, function_name: str) -> List[Dict[str, Any]]:
        """Find all callers of a function."""
        results = []
        for caller, file_path, lineno in self.ast_index.find_callers(function_name):
            results.append({
                'caller': caller,
                'file': file_path,
                'line': lineno,
            })
        return results

    def find_class(self, class_name: str) -> Optional[Dict[str, Any]]:
        """Find class information."""
        info = self.ast_index.find_class(class_name)
        if not info:
            return None

        return {
            'name': info.name,
            'file': info.file_path,
            'line': info.lineno,
            'bases': info.bases,
            'methods': info.methods,
            'attributes': list(info.attributes),
            'docstring': info.docstring,
        }

    def find_function(self, function_name: str) -> List[Dict[str, Any]]:
        """Find function information."""
        results = []
        for info in self.ast_index.find_function(function_name):
            results.append({
                'name': info.full_name,
                'file': info.file_path,
                'line': info.lineno,
                'args': info.args,
                'calls': info.calls[:10],  # Limit calls shown
                'docstring': info.docstring,
            })
        return results

    def get_inheritance(self, class_name: str) -> Dict[str, Any]:
        """Get inheritance tree for a class."""
        # Find parent classes
        class_info = self.ast_index.find_class(class_name)
        parents = class_info.bases if class_info else []

        # Find child classes
        tree = self.ast_index.get_inheritance_tree(class_name)

        return {
            'class': class_name,
            'parents': parents,
            'children': tree['children'],
        }

    def find_imports(self, module_name: str) -> List[Dict[str, Any]]:
        """Find all imports of a module."""
        results = []
        for file_path, lineno in self.ast_index.find_imports_of(module_name):
            results.append({
                'file': file_path,
                'line': lineno,
            })
        return results

    def find_related_files(self, file_path: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find files related to the given file based on:
        - Shared imports
        - Call relationships
        - Token similarity
        """
        scores: Dict[str, float] = defaultdict(float)

        # Factor 1: Shared imports
        file_imports = set()
        for imp in self.ast_index.imports:
            if imp.file_path == file_path:
                file_imports.add(imp.module)

        for imp in self.ast_index.imports:
            if imp.file_path != file_path and imp.module in file_imports:
                scores[imp.file_path] += 0.3

        # Factor 2: Call relationships
        file_functions = self.ast_index.functions_by_file.get(file_path, [])
        for func_name in file_functions:
            # Files we call
            for callee in self.ast_index.call_graph.get(func_name, []):
                for other_func, info in self.ast_index.functions.items():
                    if info.name == callee and info.file_path != file_path:
                        scores[info.file_path] += 0.4

            # Files that call us
            for caller in self.ast_index.reverse_call_graph.get(func_name, []):
                if caller in self.ast_index.functions:
                    caller_file = self.ast_index.functions[caller].file_path
                    if caller_file != file_path:
                        scores[caller_file] += 0.4

        # Factor 3: Token similarity (Jaccard)
        if file_path in self._file_tokens:
            file_tokens = set(self._file_tokens[file_path])
            for other_path, other_tokens in self._file_tokens.items():
                if other_path != file_path:
                    other_set = set(other_tokens)
                    intersection = len(file_tokens & other_set)
                    union = len(file_tokens | other_set)
                    if union > 0:
                        scores[other_path] += 0.3 * (intersection / union)

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_n]

    def query(self, natural_query: str) -> List[Dict[str, Any]]:
        """
        Answer natural language queries about the codebase.

        Supports queries like:
        - "what calls process_document"
        - "class that inherits from Layer"
        - "where is PageRank implemented"
        """
        query_lower = natural_query.lower()
        results = []

        # Pattern: "what calls X" or "who calls X"
        call_match = re.search(r'(?:what|who)\s+calls?\s+(\w+)', query_lower)
        if call_match:
            func_name = call_match.group(1)
            callers = self.find_callers(func_name)
            return [{'type': 'callers', 'function': func_name, 'results': callers}]

        # Pattern: "class that inherits/extends X"
        inherit_match = re.search(r'class(?:es)?\s+(?:that\s+)?(?:inherit|extend)s?\s+(?:from\s+)?(\w+)', query_lower)
        if inherit_match:
            parent = inherit_match.group(1)
            children = list(self.ast_index.inheritance.get(parent, []))
            return [{'type': 'inheritance', 'parent': parent, 'children': children}]

        # Pattern: "where is X implemented/defined"
        where_match = re.search(r'where\s+is\s+(\w+)', query_lower)
        if where_match:
            name = where_match.group(1)
            # Check functions
            funcs = self.find_function(name)
            if funcs:
                return [{'type': 'function_location', 'results': funcs}]
            # Check classes
            cls = self.find_class(name)
            if cls:
                return [{'type': 'class_location', 'results': [cls]}]

        # Pattern: "what imports X" or "who uses X"
        import_match = re.search(r'(?:what|who)\s+(?:imports?|uses?)\s+(\w+)', query_lower)
        if import_match:
            module = import_match.group(1)
            imports = self.find_imports(module)
            return [{'type': 'imports', 'module': module, 'results': imports}]

        # Fallback: search for the term in function/class names
        terms = re.findall(r'\w+', query_lower)
        for term in terms:
            if len(term) > 3:  # Skip short words
                # Search functions
                for full_name, info in self.ast_index.functions.items():
                    if term in full_name.lower():
                        results.append({
                            'type': 'function',
                            'name': full_name,
                            'file': info.file_path,
                            'line': info.lineno,
                        })
                # Search classes
                for name, info in self.ast_index.classes.items():
                    if term in name.lower():
                        results.append({
                            'type': 'class',
                            'name': name,
                            'file': info.file_path,
                            'line': info.lineno,
                        })

        return results[:20]  # Limit results

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        ast_stats = self.ast_index.get_stats()

        return {
            'files_indexed': ast_stats['files'],
            'classes': ast_stats['classes'],
            'functions': ast_stats['functions'],
            'imports': ast_stats['imports'],
            'call_edges': ast_stats['call_edges'],
            'inheritance_edges': ast_stats['inheritance_edges'],
            'ngram_vocab': len(self.ngram_model.vocab),
            'ngram_contexts': len(self.ngram_model.counts),
            'ngram_tokens': self.ngram_model.total_tokens,
            'training_time_sec': round(self.training_time, 2),
        }

    # =========================================================================
    # INTERACTIVE MODE
    # =========================================================================

    def interactive(self):
        """Run interactive query mode."""
        print("\n" + "=" * 70)
        print("SparkCodeIntelligence Interactive Mode")
        print("=" * 70)
        print("\nCommands:")
        print("  complete <prefix>     - Code completion")
        print("  callers <function>    - Find callers")
        print("  class <name>          - Class info")
        print("  function <name>       - Function info")
        print("  inheritance <class>   - Inheritance tree")
        print("  imports <module>      - Find imports")
        print("  related <file>        - Find related files")
        print("  query <question>      - Natural language query")
        print("  stats                 - Show statistics")
        print("  quit                  - Exit")
        print()

        while True:
            try:
                line = input("spark> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not line:
                continue

            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ''

            if cmd in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            elif cmd == 'complete':
                results = self.complete(arg)
                for suggestion, conf, source in results:
                    print(f"  {suggestion:<30} ({conf:.2f}) [{source}]")

            elif cmd == 'callers':
                results = self.find_callers(arg)
                if results:
                    for r in results:
                        print(f"  {r['caller']:<40} {r['file']}:{r['line']}")
                else:
                    print(f"  No callers found for '{arg}'")

            elif cmd == 'class':
                result = self.find_class(arg)
                if result:
                    print(f"  Name: {result['name']}")
                    print(f"  File: {result['file']}:{result['line']}")
                    print(f"  Bases: {', '.join(result['bases']) or 'object'}")
                    print(f"  Methods: {', '.join(result['methods'][:10])}")
                    print(f"  Attributes: {', '.join(result['attributes'][:10])}")
                else:
                    print(f"  Class '{arg}' not found")

            elif cmd == 'function':
                results = self.find_function(arg)
                if results:
                    for r in results:
                        print(f"  {r['name']}")
                        print(f"    File: {r['file']}:{r['line']}")
                        print(f"    Args: {', '.join(r['args'])}")
                else:
                    print(f"  Function '{arg}' not found")

            elif cmd == 'inheritance':
                result = self.get_inheritance(arg)
                print(f"  Class: {result['class']}")
                print(f"  Parents: {', '.join(result['parents']) or 'object'}")
                if result['children']:
                    print(f"  Children:")
                    for child in result['children']:
                        print(f"    - {child['name']}")
                else:
                    print(f"  Children: (none)")

            elif cmd == 'imports':
                results = self.find_imports(arg)
                if results:
                    for r in results:
                        print(f"  {r['file']}:{r['line']}")
                else:
                    print(f"  No imports of '{arg}' found")

            elif cmd == 'related':
                results = self.find_related_files(arg)
                if results:
                    for path, score in results:
                        print(f"  {path:<50} (score: {score:.2f})")
                else:
                    print(f"  No related files found")

            elif cmd == 'query':
                results = self.query(arg)
                if results:
                    for r in results:
                        print(f"  {r}")
                else:
                    print(f"  No results for: {arg}")

            elif cmd == 'stats':
                stats = self.get_stats()
                for key, value in stats.items():
                    print(f"  {key}: {value:,}")

            else:
                print(f"  Unknown command: {cmd}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SparkCodeIntelligence - Hybrid AST + N-gram Code Intelligence"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train on codebase')
    train_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    train_parser.add_argument('--extensions', '-e', nargs='+', default=['.py'], help='File extensions')

    # Complete command
    complete_parser = subparsers.add_parser('complete', help='Code completion')
    complete_parser.add_argument('prefix', help='Code prefix to complete')
    complete_parser.add_argument('--top', '-n', type=int, default=10, help='Number of suggestions')

    # Find callers command
    callers_parser = subparsers.add_parser('find-callers', help='Find function callers')
    callers_parser.add_argument('function', help='Function name')

    # Find class command
    class_parser = subparsers.add_parser('find-class', help='Find class info')
    class_parser.add_argument('name', help='Class name')

    # Inheritance command
    inherit_parser = subparsers.add_parser('inheritance', help='Show inheritance tree')
    inherit_parser.add_argument('class_name', help='Class name')

    # Imports command
    imports_parser = subparsers.add_parser('imports', help='Find imports of module')
    imports_parser.add_argument('module', help='Module name')

    # Related files command
    related_parser = subparsers.add_parser('related', help='Find related files')
    related_parser.add_argument('file', help='File path')
    related_parser.add_argument('--top', '-n', type=int, default=10, help='Number of results')

    # Query command
    query_parser = subparsers.add_parser('query', help='Natural language query')
    query_parser.add_argument('question', help='Query string')

    # Interactive command
    subparsers.add_parser('interactive', help='Interactive mode')

    # Stats command
    subparsers.add_parser('stats', help='Show statistics')

    # Benchmark command
    subparsers.add_parser('benchmark', help='Run performance benchmarks')

    # Coverage commands
    coverage_parser = subparsers.add_parser('coverage', help='Estimate code coverage')
    coverage_parser.add_argument('--uncovered', '-u', action='store_true', help='Show uncovered files')
    coverage_parser.add_argument('--suggest', '-s', type=str, help='Suggest tests for a file')
    coverage_parser.add_argument('--changed', '-c', nargs='+', help='Changed files to analyze')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize engine
    engine = SparkCodeIntelligence()

    # Load or train
    if args.command == 'train':
        engine.train(extensions=args.extensions, verbose=args.verbose)
        engine.save()
        print(f"\n💾 Model saved to {SparkCodeIntelligence.MODEL_FILE}")
    else:
        # Try to load existing model
        try:
            engine.load()
        except FileNotFoundError:
            print("❌ No trained model found. Run 'train' first.")
            print(f"   python {__file__} train")
            return

        if args.command == 'complete':
            results = engine.complete(args.prefix, top_n=args.top)
            print(f"\nCompletions for '{args.prefix}':")
            for suggestion, conf, source in results:
                print(f"  {suggestion:<30} ({conf:.2f}) [{source}]")

        elif args.command == 'find-callers':
            results = engine.find_callers(args.function)
            print(f"\nCallers of '{args.function}':")
            if results:
                for r in results:
                    print(f"  {r['caller']:<40} {r['file']}:{r['line']}")
            else:
                print("  (none found)")

        elif args.command == 'find-class':
            result = engine.find_class(args.name)
            if result:
                print(f"\nClass: {result['name']}")
                print(f"  File: {result['file']}:{result['line']}")
                print(f"  Bases: {', '.join(result['bases']) or 'object'}")
                print(f"  Methods ({len(result['methods'])}):")
                for m in result['methods'][:15]:
                    print(f"    - {m}()")
                if len(result['methods']) > 15:
                    print(f"    ... and {len(result['methods']) - 15} more")
                print(f"  Attributes ({len(result['attributes'])}):")
                for a in sorted(result['attributes'])[:15]:
                    print(f"    - {a}")
            else:
                print(f"Class '{args.name}' not found")

        elif args.command == 'inheritance':
            result = engine.get_inheritance(args.class_name)
            print(f"\nInheritance for '{args.class_name}':")
            print(f"  Parents: {', '.join(result['parents']) or 'object'}")
            print(f"  Children:")
            if result['children']:
                def print_tree(node, indent=4):
                    print(' ' * indent + f"- {node['name']}")
                    for child in node.get('children', []):
                        print_tree(child, indent + 2)
                for child in result['children']:
                    print_tree(child)
            else:
                print("    (none)")

        elif args.command == 'imports':
            results = engine.find_imports(args.module)
            print(f"\nFiles importing '{args.module}':")
            if results:
                for r in results:
                    print(f"  {r['file']}:{r['line']}")
            else:
                print("  (none found)")

        elif args.command == 'related':
            results = engine.find_related_files(args.file, top_n=args.top)
            print(f"\nFiles related to '{args.file}':")
            if results:
                for path, score in results:
                    print(f"  {path:<50} (score: {score:.2f})")
            else:
                print("  (none found)")

        elif args.command == 'query':
            results = engine.query(args.question)
            print(f"\nResults for: '{args.question}'")
            if results:
                for r in results:
                    if r.get('type') == 'callers':
                        print(f"  Callers of {r['function']}:")
                        for caller in r['results'][:10]:
                            print(f"    - {caller['caller']} ({caller['file']}:{caller['line']})")
                    elif r.get('type') == 'inheritance':
                        print(f"  Classes inheriting from {r['parent']}:")
                        for child in r['children']:
                            print(f"    - {child}")
                    else:
                        print(f"  {r}")
            else:
                print("  (no results)")

        elif args.command == 'interactive':
            engine.interactive()

        elif args.command == 'stats':
            stats = engine.get_stats()
            print("\nSparkCodeIntelligence Statistics:")
            print("=" * 40)
            for key, value in stats.items():
                print(f"  {key}: {value:,}")

        elif args.command == 'benchmark':
            print("Running benchmarks...")
            results = run_benchmarks(engine)
            print_benchmarks(results)

        elif args.command == 'coverage':
            estimator = CoverageEstimator(engine)
            estimator.analyze(verbose=True)

            if args.suggest:
                print(f"\nSuggested tests for '{args.suggest}':")
                suggestions = estimator.suggest_tests(args.suggest)
                for s in suggestions:
                    print(f"  {s}")

            elif args.uncovered:
                print("\nUncovered source files (by function count):")
                uncovered = estimator.find_uncovered(top_n=20)
                for path, func_count in uncovered:
                    print(f"  {path:<60} ({func_count} functions)")

            else:
                print("\nCoverage Estimate:")
                estimate = estimator.estimate_coverage(changed_files=args.changed)
                print(f"  Estimated line coverage: {estimate['estimated_line_coverage']}%")
                print(f"  File coverage rate: {estimate['file_coverage_rate']}%")
                print(f"  Source files: {estimate['source_files']}")
                print(f"  Files with tests: {estimate['covered_files']}")
                print(f"  Files without tests: {estimate['uncovered_files']}")
                if estimate['change_impact'] is not None:
                    print(f"  Changed files coverage: {estimate['change_impact']}%")
                print(f"  Confidence: {estimate['confidence']}")


# =============================================================================
# COVERAGE ESTIMATOR - Fast coverage prediction
# =============================================================================

class CoverageEstimator:
    """
    Fast coverage estimator based on test file relationships.

    Instead of running the full test suite (which can take many minutes),
    this estimates coverage by analyzing:
    1. Test file <-> source file naming conventions
    2. Import relationships between tests and sources
    3. Historical patterns (if available)
    """

    def __init__(self, engine: SparkCodeIntelligence):
        """
        Initialize coverage estimator.

        Args:
            engine: Trained SparkCodeIntelligence instance
        """
        self.engine = engine
        self.test_source_map: Dict[str, List[str]] = {}  # test -> sources it tests
        self.source_test_map: Dict[str, List[str]] = {}  # source -> tests that cover it
        self.coverage_history: Dict[str, float] = {}     # file -> last known coverage

    def analyze(self, verbose: bool = True) -> 'CoverageEstimator':
        """
        Analyze test-source relationships.

        Returns:
            self for method chaining
        """
        if verbose:
            print("Analyzing test-source relationships...")

        # Find all test files and source files
        test_files = set()
        source_files = set()

        for file_path in self.engine.ast_index.functions_by_file.keys():
            filename = Path(file_path).name

            # Test file detection (broad)
            if '/tests/' in file_path or filename.startswith('test_'):
                test_files.add(file_path)
            elif file_path.endswith('.py'):
                # Source file if it's in cortical/ but not a test
                if '/cortical/' in file_path:
                    if not filename.startswith('test_'):
                        source_files.add(file_path)

        test_files = list(test_files)
        source_files = list(source_files)

        if verbose:
            print(f"  Found {len(test_files)} test files")
            print(f"  Found {len(source_files)} source files")

        # Build test-source relationships
        for test_file in test_files:
            sources = self._find_tested_sources(test_file, source_files)
            self.test_source_map[test_file] = sources
            for src in sources:
                if src not in self.source_test_map:
                    self.source_test_map[src] = []
                self.source_test_map[src].append(test_file)

        # Calculate coverage based on test existence
        covered = len([s for s in source_files if s in self.source_test_map])
        if verbose:
            print(f"  Source files with tests: {covered}/{len(source_files)}")
            if source_files:
                print(f"  Estimated file coverage: {100*covered/len(source_files):.1f}%")
            else:
                print(f"  No source files found in cortical/")

        return self

    def _find_tested_sources(self, test_file: str, source_files: List[str]) -> List[str]:
        """Find source files that a test file likely tests."""
        tested = []

        # Extract test file name pattern
        test_name = Path(test_file).stem  # e.g., "test_processor"

        # Pattern 1: test_X.py tests X.py
        if test_name.startswith('test_'):
            target = test_name[5:]  # Remove "test_" prefix
            for src in source_files:
                src_stem = Path(src).stem
                if src_stem == target or src_stem.endswith(f'/{target}'):
                    tested.append(src)

        # Pattern 2: Check imports in the test file
        for imp in self.engine.ast_index.imports:
            if imp.file_path == test_file:
                # Look for source files matching the import
                for src in source_files:
                    src_module = Path(src).stem
                    if src_module in imp.module or imp.module.endswith(src_module):
                        if src not in tested:
                            tested.append(src)

        return tested

    def estimate_coverage(self, changed_files: List[str] = None) -> Dict[str, Any]:
        """
        Estimate current coverage without running tests.

        Args:
            changed_files: Optional list of recently changed files

        Returns:
            Dict with coverage estimates
        """
        # Count files with test coverage
        total_source = len([f for f in self.engine.ast_index.functions_by_file
                          if '/cortical/' in f and not Path(f).name.startswith('test_')])

        covered_source = len(self.source_test_map)

        if total_source == 0:
            return {
                'estimated_line_coverage': 0,
                'file_coverage_rate': 0,
                'source_files': 0,
                'covered_files': 0,
                'uncovered_files': 0,
                'change_impact': None,
                'confidence': 'low',
            }

        # Estimate line coverage based on function coverage
        # Heuristic: files with tests typically have 70-90% line coverage
        # Files without tests have 0%
        avg_coverage_with_tests = 0.80  # Assume 80% for files with tests
        file_coverage_rate = covered_source / max(total_source, 1)

        estimated_line_coverage = file_coverage_rate * avg_coverage_with_tests * 100

        # Adjust for changed files
        if changed_files:
            changed_coverage = []
            for cf in changed_files:
                if cf in self.source_test_map:
                    changed_coverage.append(avg_coverage_with_tests)
                else:
                    changed_coverage.append(0.0)
            if changed_coverage:
                change_impact = sum(changed_coverage) / len(changed_coverage)
            else:
                change_impact = 0.5
        else:
            change_impact = None

        return {
            'estimated_line_coverage': round(estimated_line_coverage, 1),
            'file_coverage_rate': round(file_coverage_rate * 100, 1),
            'source_files': total_source,
            'covered_files': covered_source,
            'uncovered_files': total_source - covered_source,
            'change_impact': round(change_impact * 100, 1) if change_impact else None,
            'confidence': 'medium',  # Can improve with historical data
        }

    def find_uncovered(self, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Find source files without test coverage.

        Returns:
            List of (file_path, function_count) for uncovered files
        """
        uncovered = []

        for file_path, funcs in self.engine.ast_index.functions_by_file.items():
            # Only check cortical/ source files
            if '/cortical/' not in file_path:
                continue
            if Path(file_path).name.startswith('test_'):
                continue
            if not file_path.endswith('.py'):
                continue
            if file_path not in self.source_test_map:
                uncovered.append((file_path, len(funcs)))

        # Sort by function count (more functions = higher priority)
        uncovered.sort(key=lambda x: -x[1])
        return uncovered[:top_n]

    def suggest_tests(self, file_path: str) -> List[str]:
        """
        Suggest what tests should cover a file.

        Args:
            file_path: Source file path

        Returns:
            List of suggested test patterns
        """
        suggestions = []

        # Get functions in this file
        funcs = self.engine.ast_index.functions_by_file.get(file_path, [])

        file_stem = Path(file_path).stem

        # Suggest test file name
        suggestions.append(f"tests/test_{file_stem}.py")
        suggestions.append(f"tests/unit/test_{file_stem}.py")

        # Suggest test methods for key functions
        for func_name in funcs[:5]:
            if func_name.startswith('_'):
                continue
            name_parts = func_name.split('.')
            method_name = name_parts[-1]
            suggestions.append(f"  def test_{method_name}(self):")

        return suggestions


# =============================================================================
# BENCHMARKS
# =============================================================================

def run_benchmarks(engine: SparkCodeIntelligence) -> Dict[str, Any]:
    """
    Run comprehensive benchmarks on SparkCodeIntelligence.

    Returns:
        Dict with benchmark results
    """
    import time

    results = {}

    # Benchmark 1: Completion latency
    completions = ['self.', 'def ', 'import ', 'from cortical', 'NGramModel.']
    completion_times = []

    for prefix in completions:
        start = time.perf_counter()
        engine.complete(prefix, top_n=10)
        elapsed = (time.perf_counter() - start) * 1000
        completion_times.append(elapsed)

    results['completion'] = {
        'avg_ms': round(sum(completion_times) / len(completion_times), 2),
        'min_ms': round(min(completion_times), 2),
        'max_ms': round(max(completion_times), 2),
    }

    # Benchmark 2: Caller search
    test_funcs = ['compute_pagerank', 'process_document', 'expand_query', 'find_callers']
    caller_times = []

    for func in test_funcs:
        start = time.perf_counter()
        engine.find_callers(func)
        elapsed = (time.perf_counter() - start) * 1000
        caller_times.append(elapsed)

    results['find_callers'] = {
        'avg_ms': round(sum(caller_times) / len(caller_times), 2),
        'min_ms': round(min(caller_times), 2),
        'max_ms': round(max(caller_times), 2),
    }

    # Benchmark 3: Class lookup
    test_classes = ['NGramModel', 'CorticalTextProcessor', 'HierarchicalLayer', 'ASTIndex']
    class_times = []

    for cls in test_classes:
        start = time.perf_counter()
        engine.find_class(cls)
        elapsed = (time.perf_counter() - start) * 1000
        class_times.append(elapsed)

    results['find_class'] = {
        'avg_ms': round(sum(class_times) / len(class_times), 2),
        'min_ms': round(min(class_times), 2),
        'max_ms': round(max(class_times), 2),
    }

    # Benchmark 4: Natural language query
    test_queries = [
        'what calls compute_tfidf',
        'where is PageRank',
        'class that inherits Layer',
    ]
    query_times = []

    for q in test_queries:
        start = time.perf_counter()
        engine.query(q)
        elapsed = (time.perf_counter() - start) * 1000
        query_times.append(elapsed)

    results['query'] = {
        'avg_ms': round(sum(query_times) / len(query_times), 2),
        'min_ms': round(min(query_times), 2),
        'max_ms': round(max(query_times), 2),
    }

    # Benchmark 5: Related files
    test_files = list(engine._file_tokens.keys())[:3]
    related_times = []

    for f in test_files:
        start = time.perf_counter()
        engine.find_related_files(f, top_n=10)
        elapsed = (time.perf_counter() - start) * 1000
        related_times.append(elapsed)

    if related_times:
        results['find_related'] = {
            'avg_ms': round(sum(related_times) / len(related_times), 2),
            'min_ms': round(min(related_times), 2),
            'max_ms': round(max(related_times), 2),
        }
    else:
        results['find_related'] = {'avg_ms': 0, 'min_ms': 0, 'max_ms': 0}

    # Memory estimate
    import sys
    model_size = sys.getsizeof(engine.ngram_model.counts) / (1024 * 1024)
    results['memory'] = {
        'ngram_counts_mb': round(model_size, 1),
        'contexts': len(engine.ngram_model.counts),
        'vocab': len(engine.ngram_model.vocab),
    }

    return results


def print_benchmarks(results: Dict[str, Any]) -> None:
    """Print benchmark results in a nice format."""
    print("\n" + "=" * 60)
    print("SparkCodeIntelligence Benchmarks")
    print("=" * 60)

    print("\n📊 Latency (milliseconds):")
    print(f"  {'Operation':<20} {'Avg':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 50)

    for op in ['completion', 'find_callers', 'find_class', 'query', 'find_related']:
        if op in results:
            r = results[op]
            print(f"  {op:<20} {r['avg_ms']:>10.2f} {r['min_ms']:>10.2f} {r['max_ms']:>10.2f}")

    print("\n💾 Memory:")
    if 'memory' in results:
        m = results['memory']
        print(f"  N-gram counts: {m['ngram_counts_mb']:.1f} MB")
        print(f"  Contexts: {m['contexts']:,}")
        print(f"  Vocabulary: {m['vocab']:,}")

    print()


if __name__ == '__main__':
    main()
