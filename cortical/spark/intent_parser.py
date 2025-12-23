"""
Intent Parser
=============

Parse conventional commit messages to extract structured intent information.
Supports conventional commits (feat/fix/etc), free-form messages with keyword extraction,
and reference detection (issues, PRs, task IDs).

Example:
    >>> parser = IntentParser()
    >>> result = parser.parse("feat(auth): Add OAuth2 login flow")
    >>> result.type
    'feat'
    >>> result.scope
    'auth'
    >>> result.action
    'add'
"""

import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class IntentResult:
    """
    Parsed intent from a commit message.

    Attributes:
        type: Commit type (feat, fix, refactor, docs, test, chore, perf, ci, build, style, revert)
        scope: Optional module/component scope
        action: Primary action verb (add, fix, update, remove, refactor, implement, etc.)
        entities: Extracted keywords/concepts from description
        description: Full description text
        breaking: Whether this is a breaking change
        priority: Inferred priority level (critical, high, medium, low)
        references: Issue/PR/task IDs (e.g., #123, T-20251222-...)
        confidence: Parsing confidence score (0.0-1.0)
        method: Classification method used (conventional, keyword, hybrid)
    """
    type: str
    scope: Optional[str]
    action: str
    entities: List[str]
    description: str
    breaking: bool = False
    priority: str = 'medium'
    references: List[str] = field(default_factory=list)
    confidence: float = 0.0
    method: str = 'unknown'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntentResult':
        """Create IntentResult from dictionary."""
        return cls(**data)


class IntentParser:
    """
    Parse commit messages to extract structured intent.

    Uses a hybrid approach:
    1. Rule-based parsing for conventional commits (fast, accurate)
    2. Keyword extraction for free-form messages
    3. Reference and priority detection from message body

    Example:
        >>> parser = IntentParser()
        >>> result = parser.parse("feat(spark): Add n-gram model")
        >>> result.type
        'feat'
        >>> result.scope
        'spark'
        >>> result.entities
        ['ngram', 'model']
    """

    # Conventional commit pattern
    CONVENTIONAL_COMMIT = re.compile(
        r'^(?P<type>feat|fix|refactor|docs|test|chore|perf|ci|build|style|revert|security)'
        r'(?:\((?P<scope>[^)]+)\))?'
        r'(?P<breaking>!)?'
        r':\s*'
        r'(?P<description>.+)',
        re.IGNORECASE
    )

    # Reference patterns (issues, PRs, task IDs)
    REFERENCE_PATTERN = re.compile(
        r'(?:#(\d+)|'                           # GitHub issue/PR: #123
        r'(T-\d{8}-\d{6}-[a-f0-9]+)|'          # GoT task: T-20251222-093045-a1b2
        r'(GH-\d+)|'                            # GH-123
        r'(JIRA-\d+)|'                          # JIRA-123
        r'[Tt]ask\s*#?(\d+))'                  # Task #123
    )

    # Action verbs (order matters - check specific patterns first)
    ACTION_VERBS = [
        (r'\badd(?:ing|ed|s)?\b', 'add'),
        (r'\bfix(?:ing|ed|es)?\b', 'fix'),
        (r'\brefactor(?:ing|ed|s)?\b', 'refactor'),
        (r'\bimplement(?:ing|ed|s)?\b', 'implement'),
        (r'\bupdate(?:ing|ed|s)?\b', 'update'),
        (r'\bremove(?:ing|d|s)?\b', 'remove'),
        (r'\bdelete(?:ing|d|s)?\b', 'delete'),
        (r'\bimprove(?:ing|d|s)?\b', 'improve'),
        (r'\boptimize(?:ing|d|s)?\b', 'optimize'),
        (r'\benhance(?:ing|d|s)?\b', 'enhance'),
        (r'\bcreate(?:ing|d|s)?\b', 'create'),
        (r'\bintroduce(?:ing|d|s)?\b', 'introduce'),
        (r'\bresolve(?:ing|d|s)?\b', 'resolve'),
        (r'\bcorrect(?:ing|ed|s)?\b', 'correct'),
        (r'\brepair(?:ing|ed|s)?\b', 'repair'),
        (r'\bpatch(?:ing|ed|es)?\b', 'patch'),
        (r'\brestructure(?:ing|d|s)?\b', 'restructure'),
        (r'\breorganize(?:ing|d|s)?\b', 'reorganize'),
        (r'\bsimplify(?:ing|ied|ies)?\b', 'simplify'),
        (r'\bextract(?:ing|ed|s)?\b', 'extract'),
        (r'\bdeprecate(?:ing|d|s)?\b', 'deprecate'),
        (r'\bdrop(?:ping|ped|s)?\b', 'drop'),
        (r'\bupgrade(?:ing|d|s)?\b', 'upgrade'),
        (r'\bdocument(?:ing|ed|s)?\b', 'document'),
        (r'\bexplain(?:ing|ed|s)?\b', 'explain'),
        (r'\bdescribe(?:ing|d|s)?\b', 'describe'),
        (r'\bclarify(?:ing|ied|ies)?\b', 'clarify'),
    ]

    # Stop words for entity extraction
    STOP_WORDS = frozenset([
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for',
        'on', 'at', 'by', 'with', 'from', 'as', 'be', 'or', 'and', 'but',
        'not', 'this', 'that', 'it', 'its', 'into', 'out', 'up', 'down',
    ])

    # Known module/component keywords for scope inference
    MODULE_KEYWORDS = frozenset([
        'spark', 'got', 'processor', 'query', 'analysis', 'persistence',
        'tokenizer', 'fingerprint', 'utils', 'reasoning', 'wal', 'semantics',
        'minicolumn', 'layers', 'config', 'embeddings', 'gaps', 'observability',
    ])

    # Priority keywords
    PRIORITY_CRITICAL = frozenset(['critical', 'security', 'vulnerability', 'data loss', 'crash'])
    PRIORITY_HIGH = frozenset(['urgent', 'blocking', 'breaks', 'regression'])
    PRIORITY_LOW = frozenset(['typo', 'formatting', 'whitespace', 'comment'])

    def __init__(self):
        """Initialize IntentParser."""
        pass

    def parse(self, message: str) -> IntentResult:
        """
        Parse commit message to extract structured intent.

        Args:
            message: Full commit message (subject + optional body)

        Returns:
            IntentResult with parsed components

        Example:
            >>> parser = IntentParser()
            >>> result = parser.parse("feat(auth): Add OAuth2 support")
            >>> result.type
            'feat'
            >>> result.scope
            'auth'
        """
        if not message or not message.strip():
            return IntentResult(
                type='unknown',
                scope=None,
                action='unknown',
                entities=[],
                description='',
                confidence=0.0,
                method='empty'
            )

        # Split into subject and body
        lines = message.strip().split('\n')
        subject = lines[0].strip()
        body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''

        # Try conventional commit pattern first
        result = self._parse_conventional(subject)

        if result.confidence > 0.8:
            # High confidence conventional commit
            self._enrich_from_body(result, body)
            return result

        # Fallback to free-form keyword extraction
        result = self._parse_freeform(subject)
        self._enrich_from_body(result, body)

        return result

    def _parse_conventional(self, subject: str) -> IntentResult:
        """
        Parse conventional commit format.

        Args:
            subject: Commit subject line

        Returns:
            IntentResult with high confidence if matched, low otherwise
        """
        match = self.CONVENTIONAL_COMMIT.match(subject)

        if not match:
            return IntentResult(
                type='unknown',
                scope=None,
                action='unknown',
                entities=[],
                description=subject,
                confidence=0.0,
                method='conventional_failed'
            )

        type_ = match.group('type').lower()
        scope = match.group('scope')
        breaking = bool(match.group('breaking'))
        description = match.group('description').strip()

        # Extract action and entities from description
        action = self._extract_action(description) or type_
        entities = self._extract_entities(description)

        # Infer priority
        priority = self._infer_priority(type_, breaking, description)

        return IntentResult(
            type=type_,
            scope=scope,
            action=action,
            entities=entities,
            description=description,
            breaking=breaking,
            priority=priority,
            confidence=0.95,
            method='conventional'
        )

    def _parse_freeform(self, subject: str) -> IntentResult:
        """
        Parse free-form commit message using keyword extraction.

        Args:
            subject: Commit subject line

        Returns:
            IntentResult with moderate confidence
        """
        action = self._extract_action(subject)
        entities = self._extract_entities(subject)

        # Infer type from action
        type_map = {
            'add': 'feat',
            'implement': 'feat',
            'create': 'feat',
            'introduce': 'feat',
            'fix': 'fix',
            'resolve': 'fix',
            'correct': 'fix',
            'repair': 'fix',
            'patch': 'fix',
            'refactor': 'refactor',
            'restructure': 'refactor',
            'reorganize': 'refactor',
            'simplify': 'refactor',
            'extract': 'refactor',
            'update': 'chore',
            'improve': 'chore',
            'enhance': 'chore',
            'optimize': 'perf',
            'remove': 'refactor',
            'delete': 'refactor',
            'deprecate': 'refactor',
            'drop': 'refactor',
            'upgrade': 'chore',
            'document': 'docs',
            'explain': 'docs',
            'describe': 'docs',
            'clarify': 'docs',
        }

        inferred_type = type_map.get(action, 'chore')
        scope = self._infer_scope(entities)
        priority = self._infer_priority(inferred_type, False, subject)

        return IntentResult(
            type=inferred_type,
            scope=scope,
            action=action or 'update',
            entities=entities,
            description=subject,
            priority=priority,
            confidence=0.6,
            method='keyword'
        )

    def _extract_action(self, text: str) -> Optional[str]:
        """
        Extract primary action verb from text.

        Args:
            text: Text to analyze

        Returns:
            Action verb or None if not found
        """
        text_lower = text.lower()

        for pattern, action in self.ACTION_VERBS:
            if re.search(pattern, text_lower):
                return action

        return None

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities (keywords, modules, concepts) from text.

        Args:
            text: Text to analyze

        Returns:
            List of extracted entities
        """
        # Tokenize (preserve underscores in identifiers)
        tokens = re.findall(r'\b[a-z_][a-z0-9_]*\b', text.lower())

        # Filter stop words and short tokens
        entities = [
            t for t in tokens
            if t not in self.STOP_WORDS and len(t) > 2
        ]

        # Limit to top 10 most meaningful entities
        return entities[:10]

    def _infer_scope(self, entities: List[str]) -> Optional[str]:
        """
        Infer scope from entities using known module keywords.

        Args:
            entities: Extracted entities

        Returns:
            Inferred scope or None
        """
        for entity in entities:
            if entity in self.MODULE_KEYWORDS:
                return entity

        return None

    def _infer_priority(self, type_: str, breaking: bool, text: str) -> str:
        """
        Infer priority level from type, breaking flag, and text content.

        Args:
            type_: Commit type
            breaking: Whether this is a breaking change
            text: Message text to scan for priority keywords

        Returns:
            Priority level: critical, high, medium, or low
        """
        text_lower = text.lower()

        # Breaking changes are at least high priority
        if breaking:
            return 'critical'

        # Check for critical keywords
        if any(keyword in text_lower for keyword in self.PRIORITY_CRITICAL):
            return 'critical'

        # Check for high priority keywords
        if any(keyword in text_lower for keyword in self.PRIORITY_HIGH):
            return 'high'

        # Check for low priority keywords
        if any(keyword in text_lower for keyword in self.PRIORITY_LOW):
            return 'low'

        # Type-based priority
        if type_ == 'fix':
            return 'high'
        elif type_ in ('feat', 'refactor', 'perf'):
            return 'medium'
        else:  # docs, test, chore, style, ci, build
            return 'low'

    def _enrich_from_body(self, result: IntentResult, body: str) -> None:
        """
        Enrich result with information from commit message body.

        Args:
            result: IntentResult to enrich (modified in-place)
            body: Commit message body
        """
        if not body:
            return

        # Extract references
        result.references = self._extract_references(body)

        # Check for breaking change markers in body
        if re.search(r'BREAKING\s*CHANGE:', body, re.IGNORECASE):
            result.breaking = True
            # Upgrade priority if not already critical
            if result.priority not in ('critical', 'high'):
                result.priority = 'critical'

        # Re-evaluate priority with body context
        body_priority = self._infer_priority(result.type, result.breaking, body)
        # Take the higher priority
        priority_order = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        if priority_order.get(body_priority, 0) > priority_order.get(result.priority, 0):
            result.priority = body_priority

    def _extract_references(self, text: str) -> List[str]:
        """
        Extract issue/PR/task references from text.

        Args:
            text: Text to scan for references

        Returns:
            List of reference IDs
        """
        refs = []

        # Find all matches
        for match in self.REFERENCE_PATTERN.finditer(text):
            # Get the first non-None group (handles different reference formats)
            for group in match.groups():
                if group:
                    refs.append(group)
                    break

        # Also match simple patterns that might be missed
        # GitHub issues/PRs: #123
        refs.extend(re.findall(r'#(\d+)', text))

        # Remove duplicates while preserving order
        seen = set()
        unique_refs = []
        for ref in refs:
            if ref not in seen:
                seen.add(ref)
                unique_refs.append(ref)

        return unique_refs

    def __repr__(self) -> str:
        """String representation of IntentParser."""
        return "IntentParser()"
