"""
SparkCodeIntelligence
=====================

Hybrid AST + N-gram code intelligence engine.

Combines structural understanding (AST) with pattern learning (n-grams)
for powerful code intelligence:

- Context-aware completion (self. -> class attributes)
- Semantic search (find callers, inheritance, related files)
- Code pattern prediction
- Natural language queries

Example:
    >>> engine = SparkCodeIntelligence()
    >>> engine.train(verbose=True)
    >>> completions = engine.complete("self.", top_n=10)
    >>> callers = engine.find_callers("compute_pagerank")
"""

import json
import os
import re
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from .tokenizer import CodeTokenizer
from .ast_index import ASTIndex, FunctionInfo, ClassInfo, ImportInfo
from .ngram import NGramModel


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
            'ast_index': self.ast_index.to_dict(),
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

        # Restore AST index
        self.ast_index = ASTIndex.from_dict(data['ast_index'])

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
            parent_lower = inherit_match.group(1)
            # Find case-insensitive match in inheritance keys
            parent = parent_lower
            for key in self.ast_index.inheritance.keys():
                if key.lower() == parent_lower:
                    parent = key
                    break
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

    def __repr__(self) -> str:
        if self.trained:
            stats = self.get_stats()
            return (
                f"SparkCodeIntelligence(trained=True, "
                f"files={stats['files_indexed']}, "
                f"functions={stats['functions']}, "
                f"classes={stats['classes']})"
            )
        return "SparkCodeIntelligence(trained=False)"
