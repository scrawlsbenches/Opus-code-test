"""
Diff Tokenizer
==============

Tokenize git diffs into structured token sequences with special markers.
Designed to preserve semantic structure of code changes for ML training.

Special Tokens:
- [FILE], [FILE_NEW], [FILE_DEL], [FILE_REN] - File-level markers
- [HUNK] - Hunk boundary marker
- [ADD], [DEL], [CTX] - Line-level change markers
- [FUNC], [CLASS] - Context markers
- [PATTERN:guard], [PATTERN:cache], etc. - Pattern annotations

Example:
    >>> tokenizer = DiffTokenizer()
    >>> tokens = tokenizer.tokenize(diff_text)
    >>> tokens[:5]
    ['[FILE]', 'cortical/processor.py', '[HUNK]', '@@ -10,5 +10,7 @@', '[CTX]']
"""

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict
from pathlib import Path


# Special token constants
SPECIAL_TOKENS = frozenset([
    # File-level markers
    '[FILE]', '[FILE_NEW]', '[FILE_DEL]', '[FILE_REN]',
    # Hunk-level markers
    '[HUNK]', '[FUNC]', '[CLASS]',
    # Change type markers
    '[ADD]', '[DEL]', '[CTX]',
    # Pattern markers
    '[PATTERN:guard]', '[PATTERN:cache]', '[PATTERN:error]', '[PATTERN:refactor]',
    # Language markers
    '[LANG:python]', '[LANG:javascript]', '[LANG:go]', '[LANG:rust]',
])


@dataclass
class DiffToken:
    """
    A single token in a diff sequence.

    Attributes:
        token: The token string
        token_type: Type of token (FILE, HUNK, ADD, DEL, CTX, CODE, PATTERN, META)
        line_number: Optional line number in the diff
        context: Optional contextual information
    """
    token: str
    token_type: str
    line_number: Optional[int] = None
    context: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'DiffToken':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DiffHunk:
    """
    A single hunk in a diff.

    Attributes:
        start_old: Starting line number in old file
        count_old: Number of lines in old file
        start_new: Starting line number in new file
        count_new: Number of lines in new file
        header: The full @@ header line
        lines: List of DiffToken objects for this hunk
    """
    start_old: int
    count_old: int
    start_new: int
    count_new: int
    header: str
    lines: List[DiffToken] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'start_old': self.start_old,
            'count_old': self.count_old,
            'start_new': self.start_new,
            'count_new': self.count_new,
            'header': self.header,
            'lines': [line.to_dict() for line in self.lines]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DiffHunk':
        """Create from dictionary."""
        lines = [DiffToken.from_dict(line) for line in data.get('lines', [])]
        return cls(
            start_old=data['start_old'],
            count_old=data['count_old'],
            start_new=data['start_new'],
            count_new=data['count_new'],
            header=data['header'],
            lines=lines
        )


@dataclass
class DiffFile:
    """
    A single file in a diff.

    Attributes:
        old_path: Path to old version of file
        new_path: Path to new version of file
        change_type: Type of change (modified, added, deleted, renamed)
        hunks: List of DiffHunk objects
        language: Detected programming language
    """
    old_path: str
    new_path: str
    change_type: str
    hunks: List[DiffHunk] = field(default_factory=list)
    language: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'old_path': self.old_path,
            'new_path': self.new_path,
            'change_type': self.change_type,
            'hunks': [hunk.to_dict() for hunk in self.hunks],
            'language': self.language
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DiffFile':
        """Create from dictionary."""
        hunks = [DiffHunk.from_dict(hunk) for hunk in data.get('hunks', [])]
        return cls(
            old_path=data['old_path'],
            new_path=data['new_path'],
            change_type=data['change_type'],
            hunks=hunks,
            language=data.get('language')
        )


class DiffTokenizer:
    """
    Tokenize git diffs into structured token sequences.

    Combines structural awareness (special tokens for file/hunk boundaries)
    with semantic hints (pattern detection) and intelligent chunking.

    Example:
        >>> tokenizer = DiffTokenizer()
        >>> diff_text = open('example.diff').read()
        >>> tokens = tokenizer.tokenize(diff_text)
        >>> print(tokens[:10])
        ['[FILE]', 'cortical/processor.py', '[HUNK]', ...]
    """

    # Language detection mapping
    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.md': 'markdown',
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.xml': 'xml',
        '.html': 'html',
        '.css': 'css',
        '.sh': 'shell',
        '.bash': 'shell',
        '.sql': 'sql',
    }

    def __init__(self, include_patterns: bool = True):
        """
        Initialize diff tokenizer.

        Args:
            include_patterns: Whether to detect and annotate code patterns
        """
        self.include_patterns = include_patterns

        # Compile patterns for efficiency
        self._file_header_pattern = re.compile(r'^diff --git a/(.*?) b/(.*?)$')
        self._hunk_header_pattern = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$')
        self._new_file_pattern = re.compile(r'^new file mode')
        self._deleted_file_pattern = re.compile(r'^deleted file mode')
        self._rename_pattern = re.compile(r'^rename from (.+)')

    def tokenize(self, diff: str) -> List[str]:
        """
        Tokenize a git diff into a flat list of tokens.

        Args:
            diff: Git diff text

        Returns:
            List of token strings
        """
        files = self.tokenize_structured(diff)
        tokens = []

        for file in files:
            # File marker
            if file.change_type == 'added':
                tokens.append('[FILE_NEW]')
            elif file.change_type == 'deleted':
                tokens.append('[FILE_DEL]')
            elif file.change_type == 'renamed':
                tokens.append('[FILE_REN]')
            else:
                tokens.append('[FILE]')

            tokens.append(file.new_path)

            # Language tag
            if file.language:
                tokens.append(f'[LANG:{file.language}]')

            # Process hunks
            for hunk in file.hunks:
                tokens.append('[HUNK]')
                tokens.append(hunk.header)

                # Extract function context if present
                func_match = re.search(r'@@.*?@@\s*(.+)', hunk.header)
                if func_match:
                    func_context = func_match.group(1).strip()
                    if func_context:
                        tokens.append('[FUNC]')
                        tokens.append(func_context)

                # Add hunk lines
                for line_token in hunk.lines:
                    tokens.append(line_token.token)

                # Pattern detection
                if self.include_patterns:
                    pattern = self._detect_pattern(hunk)
                    if pattern:
                        tokens.append(f'[PATTERN:{pattern}]')

        return tokens

    def tokenize_structured(self, diff: str) -> List[DiffFile]:
        """
        Tokenize a git diff into structured DiffFile objects.

        Args:
            diff: Git diff text

        Returns:
            List of DiffFile objects
        """
        if not diff or not diff.strip():
            return []

        files = []
        current_file = None
        current_hunk = None
        lines = diff.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i]

            # File header
            file_match = self._file_header_pattern.match(line)
            if file_match:
                # Save previous file
                if current_file:
                    if current_hunk:
                        current_file.hunks.append(current_hunk)
                        current_hunk = None
                    files.append(current_file)

                old_path, new_path = file_match.groups()
                current_file = DiffFile(
                    old_path=old_path,
                    new_path=new_path,
                    change_type='modified',
                    language=self._detect_language(new_path)
                )
                i += 1
                continue

            # File change type detection
            if current_file:
                if self._new_file_pattern.match(line):
                    current_file.change_type = 'added'
                elif self._deleted_file_pattern.match(line):
                    current_file.change_type = 'deleted'
                elif self._rename_pattern.match(line):
                    current_file.change_type = 'renamed'

            # Hunk header
            hunk_match = self._hunk_header_pattern.match(line)
            if hunk_match:
                # Save previous hunk
                if current_hunk and current_file:
                    current_file.hunks.append(current_hunk)

                start_old, count_old, start_new, count_new, context = self._parse_hunk_header(line)
                current_hunk = DiffHunk(
                    start_old=start_old,
                    count_old=count_old,
                    start_new=start_new,
                    count_new=count_new,
                    header=line
                )
                i += 1
                continue

            # Hunk content
            if current_hunk:
                if line.startswith('+'):
                    current_hunk.lines.append(DiffToken(
                        token='[ADD]',
                        token_type='ADD',
                        context=line[1:]
                    ))
                    current_hunk.lines.append(DiffToken(
                        token=line[1:],
                        token_type='CODE'
                    ))
                elif line.startswith('-'):
                    current_hunk.lines.append(DiffToken(
                        token='[DEL]',
                        token_type='DEL',
                        context=line[1:]
                    ))
                    current_hunk.lines.append(DiffToken(
                        token=line[1:],
                        token_type='CODE'
                    ))
                elif line.startswith(' '):
                    current_hunk.lines.append(DiffToken(
                        token='[CTX]',
                        token_type='CTX',
                        context=line[1:]
                    ))
                    current_hunk.lines.append(DiffToken(
                        token=line[1:],
                        token_type='CODE'
                    ))

            i += 1

        # Save last file
        if current_file:
            if current_hunk:
                current_file.hunks.append(current_hunk)
            files.append(current_file)

        return files

    def _parse_file_header(self, line: str) -> Tuple[str, str, str]:
        """
        Parse a diff file header line.

        Args:
            line: Header line starting with 'diff --git'

        Returns:
            Tuple of (old_path, new_path, change_type)
        """
        match = self._file_header_pattern.match(line)
        if not match:
            return '', '', 'unknown'

        old_path, new_path = match.groups()
        change_type = 'modified'

        return old_path, new_path, change_type

    def _parse_hunk_header(self, line: str) -> Tuple[int, int, int, int, str]:
        """
        Parse a hunk header line.

        Args:
            line: Header line starting with '@@'

        Returns:
            Tuple of (start_old, count_old, start_new, count_new, context)
        """
        match = self._hunk_header_pattern.match(line)
        if not match:
            return 0, 0, 0, 0, ''

        start_old_str, count_old_str, start_new_str, count_new_str, context = match.groups()

        start_old = int(start_old_str)
        count_old = int(count_old_str) if count_old_str else 1
        start_new = int(start_new_str)
        count_new = int(count_new_str) if count_new_str else 1
        context = context.strip()

        return start_old, count_old, start_new, count_new, context

    def _detect_language(self, path: str) -> str:
        """
        Detect programming language from file extension.

        Args:
            path: File path

        Returns:
            Language name or 'unknown'
        """
        suffix = Path(path).suffix.lower()
        return self.LANGUAGE_MAP.get(suffix, 'unknown')

    def _detect_patterns(self, lines: List[str]) -> List[str]:
        """
        Detect code patterns in a list of lines.

        Args:
            lines: List of code lines

        Returns:
            List of detected pattern names
        """
        patterns = []
        combined = ' '.join(lines).lower()

        # Guard pattern: early returns, if-not checks
        if any(keyword in combined for keyword in ['if not', 'if !', 'return if', 'guard']):
            patterns.append('guard')

        # Cache pattern: memoization, caching
        if any(keyword in combined for keyword in ['cache', 'memo', '@lru_cache', 'cached']):
            patterns.append('cache')

        # Error handling: try/except, error checks
        if any(keyword in combined for keyword in ['try:', 'except', 'catch', 'error', 'raise']):
            patterns.append('error')

        # Refactoring: rename, extract, inline
        if any(keyword in combined for keyword in ['rename', 'extract', 'inline', 'refactor']):
            patterns.append('refactor')

        return patterns

    def _detect_pattern(self, hunk: DiffHunk) -> Optional[str]:
        """
        Detect high-level pattern in a hunk.

        Args:
            hunk: DiffHunk object

        Returns:
            Pattern name or None
        """
        if not self.include_patterns:
            return None

        # Extract added and removed lines
        added_lines = [
            token.context or token.token
            for token in hunk.lines
            if token.token_type == 'ADD'
        ]
        removed_lines = [
            token.context or token.token
            for token in hunk.lines
            if token.token_type == 'DEL'
        ]

        added_text = ' '.join(added_lines).lower()
        removed_text = ' '.join(removed_lines).lower()

        # Guard pattern: adding if/check before operation
        if 'if' in added_text and 'if' not in removed_text:
            if any(keyword in added_text for keyword in ['is_stale', 'is_valid', 'exists', 'check', 'none']):
                return 'guard'

        # Cache pattern: storing result for reuse
        if any(keyword in added_text for keyword in ['cache', 'memo', '@lru_cache', 'store', 'save']):
            return 'cache'

        # Error handling: adding try/except
        if ('try:' in added_text or 'except' in added_text or 'catch' in added_text):
            return 'error'

        # Refactoring: extracting to function/method
        if 'def ' in added_text and len(added_lines) > 5:
            return 'refactor'

        return None

    @staticmethod
    def adaptive_context_size(total_changes: int) -> int:
        """
        Determine context size based on change magnitude.

        Args:
            total_changes: Total number of lines changed

        Returns:
            Number of context lines to include
        """
        if total_changes < 50:
            return 10  # Rich context for focused changes
        elif total_changes < 200:
            return 5   # Moderate context
        else:
            return 2   # Minimal context for mass changes

    def to_dict(self, files: List[DiffFile]) -> Dict:
        """
        Convert structured diff to dictionary.

        Args:
            files: List of DiffFile objects

        Returns:
            Dictionary representation
        """
        return {
            'files': [file.to_dict() for file in files]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> List[DiffFile]:
        """
        Create structured diff from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            List of DiffFile objects
        """
        files = [DiffFile.from_dict(file_data) for file_data in data.get('files', [])]
        return files

    def __repr__(self) -> str:
        return f"DiffTokenizer(include_patterns={self.include_patterns})"
