"""
Alignment Index
===============

Index of user definitions, patterns, and preferences to accelerate
human-AI alignment.

The alignment index stores:
- Definitions: "When I say X, I mean Y"
- Patterns: "In this codebase, we do X this way"
- Preferences: "I prefer X over Y because Z"
- Context: "The current goal is X"

This allows the AI to quickly understand user vocabulary and intent
without re-learning from scratch each session.

Example:
    >>> index = AlignmentIndex()
    >>> index.add_definition("spark", "fast statistical predictor, not neural")
    >>> index.add_pattern("error handling", "we use Result types, not exceptions")
    >>> index.add_preference("naming", "snake_case for functions, PascalCase for classes")
    >>>
    >>> # Query alignment context
    >>> index.lookup("spark")
    {'type': 'definition', 'value': 'fast statistical predictor, not neural'}
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


@dataclass
class AlignmentEntry:
    """Single entry in the alignment index."""
    key: str
    value: str
    entry_type: str  # 'definition', 'pattern', 'preference', 'context', 'goal'
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = ""  # Where this came from (session, file, user input)
    confidence: float = 1.0  # How confident we are in this entry
    tags: List[str] = field(default_factory=list)


class AlignmentIndex:
    """
    Index of human-AI alignment context.

    Stores definitions, patterns, and preferences that help the AI
    understand user vocabulary and intent quickly.
    """

    def __init__(self):
        self.entries: Dict[str, List[AlignmentEntry]] = defaultdict(list)
        self._inverse_index: Dict[str, set] = defaultdict(set)  # word -> keys

    def add_definition(self, term: str, meaning: str,
                      source: str = "user", tags: Optional[List[str]] = None) -> None:
        """
        Add a definition: "When I say X, I mean Y"

        Args:
            term: The term being defined
            meaning: What it means in this context
            source: Where this definition came from
            tags: Optional categorization tags
        """
        entry = AlignmentEntry(
            key=term.lower(),
            value=meaning,
            entry_type='definition',
            source=source,
            tags=tags or []
        )
        self.entries[term.lower()].append(entry)
        self._index_entry(entry)

    def add_pattern(self, pattern_name: str, description: str,
                   source: str = "user", tags: Optional[List[str]] = None) -> None:
        """
        Add a pattern: "In this codebase, we do X this way"

        Args:
            pattern_name: Name of the pattern
            description: How we implement this pattern
            source: Where this pattern came from
            tags: Optional categorization tags
        """
        entry = AlignmentEntry(
            key=pattern_name.lower(),
            value=description,
            entry_type='pattern',
            source=source,
            tags=tags or []
        )
        self.entries[pattern_name.lower()].append(entry)
        self._index_entry(entry)

    def add_preference(self, topic: str, preference: str,
                      source: str = "user", tags: Optional[List[str]] = None) -> None:
        """
        Add a preference: "I prefer X over Y because Z"

        Args:
            topic: What the preference is about
            preference: The actual preference
            source: Where this preference came from
            tags: Optional categorization tags
        """
        entry = AlignmentEntry(
            key=topic.lower(),
            value=preference,
            entry_type='preference',
            source=source,
            tags=tags or []
        )
        self.entries[topic.lower()].append(entry)
        self._index_entry(entry)

    def add_context(self, key: str, value: str,
                   source: str = "session", tags: Optional[List[str]] = None) -> None:
        """
        Add session context: temporary alignment info for current session.

        Args:
            key: Context key
            value: Context value
            source: Where this context came from
            tags: Optional categorization tags
        """
        entry = AlignmentEntry(
            key=key.lower(),
            value=value,
            entry_type='context',
            source=source,
            tags=tags or []
        )
        self.entries[key.lower()].append(entry)
        self._index_entry(entry)

    def add_goal(self, goal: str, description: str = "",
                source: str = "user", tags: Optional[List[str]] = None) -> None:
        """
        Add a goal: "The current objective is X"

        Args:
            goal: The goal statement
            description: Additional context about the goal
            source: Where this goal came from
            tags: Optional categorization tags
        """
        entry = AlignmentEntry(
            key="goal",
            value=f"{goal}: {description}" if description else goal,
            entry_type='goal',
            source=source,
            tags=tags or []
        )
        self.entries["goal"].append(entry)
        self._index_entry(entry)

    def _index_entry(self, entry: AlignmentEntry) -> None:
        """Build inverse index for fast lookup."""
        # Index key words
        for word in entry.key.lower().split():
            self._inverse_index[word].add(entry.key)

        # Index value words
        for word in entry.value.lower().split():
            if len(word) > 2:  # Skip very short words
                self._inverse_index[word].add(entry.key)

        # Index tags
        for tag in entry.tags:
            self._inverse_index[tag.lower()].add(entry.key)

    def lookup(self, term: str) -> List[AlignmentEntry]:
        """
        Look up alignment entries for a term.

        Args:
            term: Term to look up

        Returns:
            List of matching entries (most recent first)
        """
        entries = self.entries.get(term.lower(), [])
        return sorted(entries, key=lambda e: e.created, reverse=True)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, AlignmentEntry]]:
        """
        Search alignment index for relevant entries.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (key, entry) tuples
        """
        query_words = set(query.lower().split())

        # Find keys that match query words
        matching_keys: Dict[str, int] = defaultdict(int)
        for word in query_words:
            for key in self._inverse_index.get(word, set()):
                matching_keys[key] += 1

        # Sort by match count
        sorted_keys = sorted(matching_keys.items(), key=lambda x: x[1], reverse=True)

        results = []
        for key, _ in sorted_keys[:top_k]:
            entries = self.lookup(key)
            if entries:
                results.append((key, entries[0]))  # Most recent entry

        return results

    def get_all_definitions(self) -> List[AlignmentEntry]:
        """Get all definition entries."""
        return [e for entries in self.entries.values()
                for e in entries if e.entry_type == 'definition']

    def get_all_patterns(self) -> List[AlignmentEntry]:
        """Get all pattern entries."""
        return [e for entries in self.entries.values()
                for e in entries if e.entry_type == 'pattern']

    def get_all_preferences(self) -> List[AlignmentEntry]:
        """Get all preference entries."""
        return [e for entries in self.entries.values()
                for e in entries if e.entry_type == 'preference']

    def get_current_goals(self) -> List[AlignmentEntry]:
        """Get current goal entries."""
        return sorted(
            [e for e in self.entries.get('goal', [])],
            key=lambda e: e.created,
            reverse=True
        )

    def get_context_summary(self) -> str:
        """
        Generate a summary of alignment context for session start.

        Returns:
            Human-readable summary of definitions, patterns, and preferences
        """
        lines = ["# Alignment Context", ""]

        # Goals
        goals = self.get_current_goals()
        if goals:
            lines.append("## Current Goals")
            for g in goals[:3]:
                lines.append(f"- {g.value}")
            lines.append("")

        # Definitions
        defs = self.get_all_definitions()
        if defs:
            lines.append("## Key Definitions")
            for d in defs[:10]:
                lines.append(f"- **{d.key}**: {d.value}")
            lines.append("")

        # Patterns
        patterns = self.get_all_patterns()
        if patterns:
            lines.append("## Codebase Patterns")
            for p in patterns[:10]:
                lines.append(f"- **{p.key}**: {p.value}")
            lines.append("")

        # Preferences
        prefs = self.get_all_preferences()
        if prefs:
            lines.append("## User Preferences")
            for p in prefs[:10]:
                lines.append(f"- **{p.key}**: {p.value}")
            lines.append("")

        return "\n".join(lines)

    def load_from_markdown(self, path: str) -> int:
        """
        Load alignment entries from markdown file.

        Expected format:
        ```
        ## Definitions
        - **term**: meaning

        ## Patterns
        - **pattern**: description

        ## Preferences
        - **topic**: preference
        ```

        Args:
            path: Path to markdown file

        Returns:
            Number of entries loaded
        """
        if not os.path.exists(path):
            return 0

        with open(path, 'r') as f:
            content = f.read()

        count = 0
        current_section = None

        for line in content.split('\n'):
            line = line.strip()

            # Detect section headers
            if line.startswith('## '):
                section = line[3:].lower()
                if 'definition' in section:
                    current_section = 'definition'
                elif 'pattern' in section:
                    current_section = 'pattern'
                elif 'preference' in section:
                    current_section = 'preference'
                elif 'goal' in section:
                    current_section = 'goal'
                else:
                    current_section = None
                continue

            # Parse entries: "- **key**: value" or "- key: value"
            if line.startswith('- ') and current_section:
                line = line[2:]

                # Try **key**: value format
                if '**' in line and '**: ' in line:
                    start = line.index('**') + 2
                    end = line.index('**', start)
                    key = line[start:end]
                    value = line[end + 4:].strip()
                # Try key: value format
                elif ': ' in line:
                    parts = line.split(': ', 1)
                    key = parts[0].strip('*')
                    value = parts[1] if len(parts) > 1 else ""
                else:
                    continue

                if key and value:
                    if current_section == 'definition':
                        self.add_definition(key, value, source=path)
                    elif current_section == 'pattern':
                        self.add_pattern(key, value, source=path)
                    elif current_section == 'preference':
                        self.add_preference(key, value, source=path)
                    elif current_section == 'goal':
                        self.add_goal(key, value, source=path)
                    count += 1

        return count

    def save(self, path: str) -> None:
        """Save alignment index to JSON."""
        data = {
            'entries': {
                key: [
                    {
                        'key': e.key,
                        'value': e.value,
                        'entry_type': e.entry_type,
                        'created': e.created,
                        'source': e.source,
                        'confidence': e.confidence,
                        'tags': e.tags,
                    }
                    for e in entries
                ]
                for key, entries in self.entries.items()
            }
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'AlignmentIndex':
        """Load alignment index from JSON."""
        index = cls()

        if not os.path.exists(path):
            return index

        with open(path, 'r') as f:
            data = json.load(f)

        for key, entries in data.get('entries', {}).items():
            for e in entries:
                entry = AlignmentEntry(
                    key=e['key'],
                    value=e['value'],
                    entry_type=e['entry_type'],
                    created=e.get('created', ''),
                    source=e.get('source', ''),
                    confidence=e.get('confidence', 1.0),
                    tags=e.get('tags', []),
                )
                index.entries[key].append(entry)
                index._index_entry(entry)

        return index

    def __len__(self) -> int:
        return sum(len(entries) for entries in self.entries.values())

    def __repr__(self) -> str:
        return (
            f"AlignmentIndex("
            f"definitions={len(self.get_all_definitions())}, "
            f"patterns={len(self.get_all_patterns())}, "
            f"preferences={len(self.get_all_preferences())}, "
            f"goals={len(self.get_current_goals())})"
        )
