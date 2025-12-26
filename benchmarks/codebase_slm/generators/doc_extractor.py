"""
Doc Extractor - Parse Markdown files to extract structured patterns.

Extracts:
- Document titles and sections
- Code blocks with language annotations
- Lists and key-value patterns
- Links and references
- Decision records (ADRs)

Design:
- Batch processing with configurable size
- Caching with file modification timestamps
- Progress callbacks for monitoring
"""

import re
import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Callable, Iterator, Any, Tuple
from datetime import datetime


@dataclass
class SectionPattern:
    """Extracted markdown section."""
    title: str
    level: int  # 1-6 for H1-H6
    content: str
    file_path: str
    line_number: int


@dataclass
class CodeBlockPattern:
    """Extracted code block."""
    language: str
    code: str
    file_path: str
    line_number: int
    context: str = ""  # Text before the code block


@dataclass
class KeyValuePattern:
    """Extracted key-value pair (e.g., **Key:** Value)."""
    key: str
    value: str
    file_path: str
    line_number: int


@dataclass
class TablePattern:
    """Extracted markdown table."""
    headers: List[str]
    rows: List[List[str]]
    file_path: str
    line_number: int


@dataclass
class DocPattern:
    """Container for all patterns extracted from a markdown file."""
    file_path: str
    title: Optional[str]
    sections: List[SectionPattern] = field(default_factory=list)
    code_blocks: List[CodeBlockPattern] = field(default_factory=list)
    key_values: List[KeyValuePattern] = field(default_factory=list)
    tables: List[TablePattern] = field(default_factory=list)
    file_hash: str = ""
    extracted_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'file_path': self.file_path,
            'title': self.title,
            'sections': [asdict(s) for s in self.sections],
            'code_blocks': [asdict(c) for c in self.code_blocks],
            'key_values': [asdict(k) for k in self.key_values],
            'tables': [asdict(t) for t in self.tables],
            'file_hash': self.file_hash,
            'extracted_at': self.extracted_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocPattern':
        """Create from dictionary."""
        return cls(
            file_path=data['file_path'],
            title=data.get('title'),
            sections=[SectionPattern(**s) for s in data.get('sections', [])],
            code_blocks=[CodeBlockPattern(**c) for c in data.get('code_blocks', [])],
            key_values=[KeyValuePattern(**k) for k in data.get('key_values', [])],
            tables=[TablePattern(**t) for t in data.get('tables', [])],
            file_hash=data.get('file_hash', ''),
            extracted_at=data.get('extracted_at', ''),
        )


class DocExtractor:
    """
    Extract patterns from Markdown files.

    Features:
    - Batch processing with configurable size
    - Caching to avoid re-parsing unchanged files
    - Progress callbacks
    - Handles various markdown patterns

    Usage:
        extractor = DocExtractor(cache_dir='benchmarks/codebase_slm/corpus')

        for batch in extractor.extract_batched(source_dir='docs/', batch_size=50):
            print(f"Processed {len(batch)} files")
    """

    # Regex patterns
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'```(\w*)\n(.*?)```', re.DOTALL)
    KEY_VALUE_PATTERN = re.compile(r'\*\*([^*]+)\*\*:\s*(.+)$', re.MULTILINE)
    TABLE_ROW_PATTERN = re.compile(r'^\|(.+)\|$', re.MULTILINE)

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the extractor.

        Args:
            cache_dir: Directory for caching extracted patterns
            exclude_patterns: File patterns to exclude
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.exclude_patterns = exclude_patterns or ['.git', 'node_modules', 'venv']
        self._patterns: Dict[str, DocPattern] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cached patterns from disk."""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / 'doc_patterns.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    for item in data.get('patterns', []):
                        pattern = DocPattern.from_dict(item)
                        self._patterns[pattern.file_path] = pattern
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_cache(self) -> None:
        """Save patterns to cache."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / 'doc_patterns.json'

        data = {
            'version': 1,
            'extracted_at': datetime.utcnow().isoformat(),
            'file_count': len(self._patterns),
            'patterns': [p.to_dict() for p in self._patterns.values()],
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute hash of file contents."""
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

    def _extract_sections(self, content: str, file_path: str) -> Tuple[Optional[str], List[SectionPattern]]:
        """Extract headers and their content."""
        lines = content.split('\n')
        sections = []
        title = None

        current_section = None
        current_content = []
        current_line = 0

        for i, line in enumerate(lines, 1):
            match = self.HEADER_PATTERN.match(line)
            if match:
                # Save previous section
                if current_section:
                    current_section.content = '\n'.join(current_content).strip()
                    sections.append(current_section)

                level = len(match.group(1))
                section_title = match.group(2).strip()

                if level == 1 and title is None:
                    title = section_title

                current_section = SectionPattern(
                    title=section_title,
                    level=level,
                    content='',
                    file_path=file_path,
                    line_number=i,
                )
                current_content = []
            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            current_section.content = '\n'.join(current_content).strip()
            sections.append(current_section)

        return title, sections

    def _extract_code_blocks(self, content: str, file_path: str) -> List[CodeBlockPattern]:
        """Extract fenced code blocks."""
        blocks = []
        lines = content.split('\n')

        # Find code block positions
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            start_pos = match.start()
            line_number = content[:start_pos].count('\n') + 1

            # Get context (previous non-empty line)
            context_lines = content[:start_pos].strip().split('\n')
            context = context_lines[-1] if context_lines else ''

            blocks.append(CodeBlockPattern(
                language=match.group(1) or 'text',
                code=match.group(2).strip(),
                file_path=file_path,
                line_number=line_number,
                context=context,
            ))

        return blocks

    def _extract_key_values(self, content: str, file_path: str) -> List[KeyValuePattern]:
        """Extract bold key: value patterns."""
        key_values = []

        for match in self.KEY_VALUE_PATTERN.finditer(content):
            line_number = content[:match.start()].count('\n') + 1
            key_values.append(KeyValuePattern(
                key=match.group(1).strip(),
                value=match.group(2).strip(),
                file_path=file_path,
                line_number=line_number,
            ))

        return key_values

    def _extract_tables(self, content: str, file_path: str) -> List[TablePattern]:
        """Extract markdown tables."""
        tables = []
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check if this is a table row
            if line.strip().startswith('|') and line.strip().endswith('|'):
                table_start = i
                rows = []

                # Collect all table rows
                while i < len(lines) and lines[i].strip().startswith('|'):
                    cells = [c.strip() for c in lines[i].strip('|').split('|')]
                    rows.append(cells)
                    i += 1

                # Skip separator row (contains ---)
                if len(rows) >= 2:
                    headers = rows[0]
                    data_rows = []
                    for row in rows[1:]:
                        if not all('-' in c for c in row):  # Skip separator
                            data_rows.append(row)

                    if headers and data_rows:
                        tables.append(TablePattern(
                            headers=headers,
                            rows=data_rows,
                            file_path=file_path,
                            line_number=table_start + 1,
                        ))
            else:
                i += 1

        return tables

    def _extract_from_file(self, file_path: Path) -> Optional[DocPattern]:
        """Extract patterns from a single markdown file."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return None

        title, sections = self._extract_sections(content, str(file_path))

        pattern = DocPattern(
            file_path=str(file_path),
            title=title,
            sections=sections,
            code_blocks=self._extract_code_blocks(content, str(file_path)),
            key_values=self._extract_key_values(content, str(file_path)),
            tables=self._extract_tables(content, str(file_path)),
            file_hash=self._compute_file_hash(file_path),
            extracted_at=datetime.utcnow().isoformat(),
        )

        return pattern

    def extract_batched(
        self,
        source_dir: Path,
        batch_size: int = 50,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        force_refresh: bool = False,
    ) -> Iterator[List[DocPattern]]:
        """
        Extract patterns in batches.

        Args:
            source_dir: Directory to scan for Markdown files
            batch_size: Number of files per batch
            progress_callback: Called with (processed, total, current_file)
            force_refresh: If True, ignore cache and re-extract all

        Yields:
            Batches of DocPattern objects
        """
        source_dir = Path(source_dir)

        # Find all Markdown files
        md_files = [
            f for f in source_dir.rglob('*.md')
            if not self._should_skip(f)
        ]

        total = len(md_files)
        batch = []
        processed = 0

        for file_path in md_files:
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
                self._save_cache()
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
    ) -> List[DocPattern]:
        """Extract all patterns (convenience method)."""
        all_patterns = []
        for batch in self.extract_batched(source_dir, batch_size=100,
                                          progress_callback=progress_callback,
                                          force_refresh=force_refresh):
            all_patterns.extend(batch)
        return all_patterns

    def get_all_patterns(self) -> List[DocPattern]:
        """Get all cached patterns."""
        return list(self._patterns.values())

    def get_statistics(self) -> Dict[str, int]:
        """Get extraction statistics."""
        return {
            'files': len(self._patterns),
            'sections': sum(len(p.sections) for p in self._patterns.values()),
            'code_blocks': sum(len(p.code_blocks) for p in self._patterns.values()),
            'key_values': sum(len(p.key_values) for p in self._patterns.values()),
            'tables': sum(len(p.tables) for p in self._patterns.values()),
        }


if __name__ == '__main__':
    import sys

    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('docs/')
    cache = Path('benchmarks/codebase_slm/corpus')

    extractor = DocExtractor(cache_dir=cache)

    def progress(done, total, current):
        print(f"\r[{done}/{total}] {current[:60]}...", end='', flush=True)

    patterns = extractor.extract_all(source, progress_callback=progress)
    print(f"\n\nExtracted from {len(patterns)} files")
    print(f"Statistics: {extractor.get_statistics()}")
