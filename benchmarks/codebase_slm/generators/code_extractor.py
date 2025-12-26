"""
Code Extractor - Parse Python files to extract structured patterns.

Extracts:
- Function definitions with signatures and docstrings
- Class definitions with methods and docstrings
- Import statements and dependencies
- Module-level docstrings and comments

Design:
- Batch processing (configurable batch size)
- Caching with file modification timestamps
- Resumable via checkpoint files
- Progress callbacks for monitoring
"""

import ast
import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Callable, Iterator, Any
from datetime import datetime


@dataclass
class FunctionPattern:
    """Extracted function information."""
    name: str
    signature: str
    docstring: Optional[str]
    decorators: List[str]
    file_path: str
    line_number: int
    is_async: bool = False
    is_method: bool = False
    class_name: Optional[str] = None


@dataclass
class ClassPattern:
    """Extracted class information."""
    name: str
    docstring: Optional[str]
    bases: List[str]
    methods: List[str]
    file_path: str
    line_number: int


@dataclass
class ImportPattern:
    """Extracted import information."""
    module: str
    names: List[str]
    is_from_import: bool
    file_path: str
    line_number: int


@dataclass
class CodePattern:
    """Container for all patterns extracted from a file."""
    file_path: str
    module_docstring: Optional[str]
    functions: List[FunctionPattern] = field(default_factory=list)
    classes: List[ClassPattern] = field(default_factory=list)
    imports: List[ImportPattern] = field(default_factory=list)
    file_hash: str = ""
    extracted_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'file_path': self.file_path,
            'module_docstring': self.module_docstring,
            'functions': [asdict(f) for f in self.functions],
            'classes': [asdict(c) for c in self.classes],
            'imports': [asdict(i) for i in self.imports],
            'file_hash': self.file_hash,
            'extracted_at': self.extracted_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodePattern':
        """Create from dictionary."""
        return cls(
            file_path=data['file_path'],
            module_docstring=data.get('module_docstring'),
            functions=[FunctionPattern(**f) for f in data.get('functions', [])],
            classes=[ClassPattern(**c) for c in data.get('classes', [])],
            imports=[ImportPattern(**i) for i in data.get('imports', [])],
            file_hash=data.get('file_hash', ''),
            extracted_at=data.get('extracted_at', ''),
        )


class CodeExtractor:
    """
    Extract code patterns from Python files.

    Features:
    - Batch processing with configurable size
    - Caching to avoid re-parsing unchanged files
    - Progress callbacks
    - Resumable via checkpoints

    Usage:
        extractor = CodeExtractor(cache_dir='benchmarks/codebase_slm/corpus')

        # Extract from all Python files
        for batch in extractor.extract_batched(source_dir='cortical/', batch_size=50):
            print(f"Processed {len(batch)} files")

        # Get all patterns
        patterns = extractor.get_all_patterns()
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the extractor.

        Args:
            cache_dir: Directory for caching extracted patterns
            exclude_patterns: File patterns to exclude (e.g., ['test_*', '*_test.py'])
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.exclude_patterns = exclude_patterns or ['__pycache__', '.git', 'venv', '.venv']
        self._patterns: Dict[str, CodePattern] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached patterns from disk."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / 'code_patterns.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    for item in data.get('patterns', []):
                        pattern = CodePattern.from_dict(item)
                        self._patterns[pattern.file_path] = pattern
            except (json.JSONDecodeError, KeyError):
                pass  # Start fresh if cache is corrupted

    def _save_cache(self) -> None:
        """Save patterns to cache."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / 'code_patterns.json'

        data = {
            'version': 1,
            'extracted_at': datetime.utcnow().isoformat(),
            'file_count': len(self._patterns),
            'patterns': [p.to_dict() for p in self._patterns.values()],
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of file contents for change detection."""
        content = file_path.read_bytes()
        return hashlib.md5(content).hexdigest()[:16]

    def _should_skip(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        path_str = str(file_path)
        for pattern in self.exclude_patterns:
            if pattern in path_str:
                return True
        return False

    def _is_cached_valid(self, file_path: Path) -> bool:
        """Check if cached pattern is still valid."""
        path_str = str(file_path)
        if path_str not in self._patterns:
            return False

        cached = self._patterns[path_str]
        current_hash = self._compute_file_hash(file_path)
        return cached.file_hash == current_hash

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature as string."""
        args = []

        # Regular args
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except:
                    pass
            args.append(arg_str)

        # *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        sig = f"{node.name}({', '.join(args)})"

        # Return type
        if node.returns:
            try:
                sig += f" -> {ast.unparse(node.returns)}"
            except:
                pass

        return sig

    def _extract_from_file(self, file_path: Path) -> Optional[CodePattern]:
        """Extract patterns from a single Python file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            return None

        pattern = CodePattern(
            file_path=str(file_path),
            module_docstring=ast.get_docstring(tree),
            file_hash=self._compute_file_hash(file_path),
            extracted_at=datetime.utcnow().isoformat(),
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Check if it's a method (inside a class)
                is_method = False
                class_name = None
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        if node in ast.walk(parent):
                            is_method = True
                            class_name = parent.name
                            break

                func = FunctionPattern(
                    name=node.name,
                    signature=self._get_function_signature(node),
                    docstring=ast.get_docstring(node),
                    decorators=[ast.unparse(d) if hasattr(ast, 'unparse') else str(d)
                               for d in node.decorator_list],
                    file_path=str(file_path),
                    line_number=node.lineno,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    is_method=is_method,
                    class_name=class_name,
                )
                pattern.functions.append(func)

            elif isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)

                cls = ClassPattern(
                    name=node.name,
                    docstring=ast.get_docstring(node),
                    bases=[ast.unparse(b) if hasattr(ast, 'unparse') else str(b)
                          for b in node.bases],
                    methods=methods,
                    file_path=str(file_path),
                    line_number=node.lineno,
                )
                pattern.classes.append(cls)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imp = ImportPattern(
                        module=alias.name,
                        names=[alias.asname or alias.name],
                        is_from_import=False,
                        file_path=str(file_path),
                        line_number=node.lineno,
                    )
                    pattern.imports.append(imp)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imp = ImportPattern(
                        module=node.module,
                        names=[alias.name for alias in node.names],
                        is_from_import=True,
                        file_path=str(file_path),
                        line_number=node.lineno,
                    )
                    pattern.imports.append(imp)

        return pattern

    def extract_batched(
        self,
        source_dir: Path,
        batch_size: int = 50,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        force_refresh: bool = False,
    ) -> Iterator[List[CodePattern]]:
        """
        Extract patterns in batches.

        Args:
            source_dir: Directory to scan for Python files
            batch_size: Number of files per batch
            progress_callback: Called with (processed, total, current_file)
            force_refresh: If True, ignore cache and re-extract all

        Yields:
            Batches of CodePattern objects
        """
        source_dir = Path(source_dir)

        # Find all Python files
        py_files = [
            f for f in source_dir.rglob('*.py')
            if not self._should_skip(f)
        ]

        total = len(py_files)
        batch = []
        processed = 0

        for file_path in py_files:
            processed += 1

            if progress_callback:
                progress_callback(processed, total, str(file_path))

            # Check cache
            if not force_refresh and self._is_cached_valid(file_path):
                batch.append(self._patterns[str(file_path)])
            else:
                pattern = self._extract_from_file(file_path)
                if pattern:
                    self._patterns[str(file_path)] = pattern
                    batch.append(pattern)

            # Yield batch when full
            if len(batch) >= batch_size:
                yield batch
                self._save_cache()  # Save after each batch
                batch = []

        # Yield remaining
        if batch:
            yield batch
            self._save_cache()

    def extract_all(
        self,
        source_dir: Path,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        force_refresh: bool = False,
    ) -> List[CodePattern]:
        """
        Extract all patterns (convenience method).

        Args:
            source_dir: Directory to scan
            progress_callback: Progress callback
            force_refresh: Force re-extraction

        Returns:
            List of all extracted patterns
        """
        all_patterns = []
        for batch in self.extract_batched(source_dir, batch_size=100,
                                          progress_callback=progress_callback,
                                          force_refresh=force_refresh):
            all_patterns.extend(batch)
        return all_patterns

    def get_all_patterns(self) -> List[CodePattern]:
        """Get all cached patterns."""
        return list(self._patterns.values())

    def get_statistics(self) -> Dict[str, int]:
        """Get extraction statistics."""
        total_functions = sum(len(p.functions) for p in self._patterns.values())
        total_classes = sum(len(p.classes) for p in self._patterns.values())
        total_imports = sum(len(p.imports) for p in self._patterns.values())

        return {
            'files': len(self._patterns),
            'functions': total_functions,
            'classes': total_classes,
            'imports': total_imports,
            'with_docstrings': sum(1 for p in self._patterns.values()
                                   for f in p.functions if f.docstring),
        }


if __name__ == '__main__':
    # Quick test
    import sys

    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('cortical/')
    cache = Path('benchmarks/codebase_slm/corpus')

    extractor = CodeExtractor(cache_dir=cache)

    def progress(done, total, current):
        print(f"\r[{done}/{total}] {current[:60]}...", end='', flush=True)

    patterns = extractor.extract_all(source, progress_callback=progress)
    print(f"\n\nExtracted from {len(patterns)} files")
    print(f"Statistics: {extractor.get_statistics()}")
