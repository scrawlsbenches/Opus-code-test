"""
Sample Suggester
================

Observes interactions and suggests new alignment entries.

The suggester watches for:
1. Undefined terms used frequently → suggest definitions
2. Repeated query patterns → suggest patterns
3. Consistent choices → suggest preferences

Example:
    >>> from cortical.spark import SampleSuggester
    >>> suggester = SampleSuggester()
    >>>
    >>> # Observe queries
    >>> suggester.observe_query("minicolumn activation", success=True)
    >>> suggester.observe_query("minicolumn connections", success=True)
    >>> suggester.observe_query("minicolumn pagerank", success=True)
    >>>
    >>> # Get suggestions
    >>> suggestions = suggester.suggest_definitions()
    >>> for s in suggestions:
    ...     print(f"Define '{s.term}' (seen {s.frequency} times)")
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Tuple


@dataclass
class Observation:
    """A single observed interaction."""
    query: str
    timestamp: str
    success: bool
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefinitionSuggestion:
    """Suggested definition for an undefined term."""
    term: str
    frequency: int
    contexts: List[str]  # Example queries containing the term
    confidence: float  # 0.0-1.0
    reason: str

    def to_markdown(self) -> str:
        """Convert to markdown format for alignment file."""
        return f"- **{self.term}**: [TODO: Add definition] (seen {self.frequency} times)"


@dataclass
class PatternSuggestion:
    """Suggested pattern from repeated structures."""
    pattern_name: str
    examples: List[str]
    frequency: int
    confidence: float
    reason: str

    def to_markdown(self) -> str:
        """Convert to markdown format for alignment file."""
        examples_str = ", ".join(f'"{e}"' for e in self.examples[:3])
        return f"- **{self.pattern_name}**: [TODO: Describe pattern] (e.g., {examples_str})"


@dataclass
class PreferenceSuggestion:
    """Suggested preference from consistent choices."""
    preference_name: str
    chosen: str
    over: str
    frequency: int
    confidence: float
    reason: str

    def to_markdown(self) -> str:
        """Convert to markdown format for alignment file."""
        return f"- **{self.preference_name}**: Prefer {self.chosen} over {self.over}"


class SampleSuggester:
    """
    Observes interactions and suggests alignment entries.

    The suggester maintains statistics about:
    - Term frequencies (for definition suggestions)
    - Query patterns (for pattern suggestions)
    - Choices made (for preference suggestions)

    Suggestions are drafts that require human review before
    being added to the alignment corpus.
    """

    def __init__(
        self,
        known_terms: Optional[Set[str]] = None,
        min_frequency: int = 3,
        min_confidence: float = 0.5
    ):
        """
        Initialize SampleSuggester.

        Args:
            known_terms: Set of already-defined terms (won't suggest these)
            min_frequency: Minimum occurrences before suggesting
            min_confidence: Minimum confidence for suggestions
        """
        self.known_terms = known_terms or set()
        self.min_frequency = min_frequency
        self.min_confidence = min_confidence

        # Observation storage
        self.observations: List[Observation] = []
        self.term_counts: Counter = Counter()
        self.term_contexts: Dict[str, List[str]] = defaultdict(list)
        self.bigram_counts: Counter = Counter()
        self.query_patterns: Dict[str, List[str]] = defaultdict(list)

        # Success/failure tracking for preferences
        self.success_contexts: List[str] = []
        self.failure_contexts: List[str] = []

    def observe_query(
        self,
        query: str,
        success: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record an observed query.

        Args:
            query: The query text
            success: Whether the query was successful
            context: Additional context (e.g., results found)
        """
        obs = Observation(
            query=query,
            timestamp=datetime.now().isoformat(),
            success=success,
            context=context or {}
        )
        self.observations.append(obs)

        # Extract and count terms
        terms = self._extract_terms(query)
        for term in terms:
            self.term_counts[term] += 1
            self.term_contexts[term].append(query)

        # Extract and count bigrams
        bigrams = self._extract_bigrams(query)
        for bigram in bigrams:
            self.bigram_counts[bigram] += 1

        # Track success/failure
        if success:
            self.success_contexts.append(query)
        else:
            self.failure_contexts.append(query)

        # Detect query pattern
        pattern = self._extract_pattern(query)
        if pattern:
            self.query_patterns[pattern].append(query)

    def observe_choice(
        self,
        choice_type: str,
        chosen: str,
        alternatives: List[str]
    ) -> None:
        """
        Record an observed choice (for preference detection).

        Args:
            choice_type: Type of choice (e.g., "naming", "approach")
            chosen: What was chosen
            alternatives: What could have been chosen instead
        """
        # Store for preference analysis
        context = {
            'type': 'choice',
            'choice_type': choice_type,
            'chosen': chosen,
            'alternatives': alternatives,
        }
        obs = Observation(
            query=f"choice:{choice_type}:{chosen}",
            timestamp=datetime.now().isoformat(),
            success=True,
            context=context
        )
        self.observations.append(obs)

    def add_known_term(self, term: str) -> None:
        """Add a term to the known set (won't suggest it)."""
        self.known_terms.add(term.lower())

    def add_known_terms(self, terms: Set[str]) -> None:
        """Add multiple terms to the known set."""
        self.known_terms.update(t.lower() for t in terms)

    def suggest_definitions(self) -> List[DefinitionSuggestion]:
        """
        Suggest definitions for frequently-used undefined terms.

        Returns:
            List of definition suggestions sorted by confidence
        """
        suggestions = []

        for term, count in self.term_counts.most_common():
            # Skip known terms
            if term.lower() in self.known_terms:
                continue

            # Skip if below threshold
            if count < self.min_frequency:
                continue

            # Calculate confidence based on frequency and uniqueness
            confidence = self._calculate_definition_confidence(term, count)

            if confidence >= self.min_confidence:
                suggestions.append(DefinitionSuggestion(
                    term=term,
                    frequency=count,
                    contexts=self.term_contexts[term][:5],  # Top 5 examples
                    confidence=confidence,
                    reason=f"Used {count} times without definition"
                ))

        return sorted(suggestions, key=lambda s: -s.confidence)

    def suggest_patterns(self) -> List[PatternSuggestion]:
        """
        Suggest patterns from repeated query structures.

        Returns:
            List of pattern suggestions sorted by confidence
        """
        suggestions = []

        for pattern, examples in self.query_patterns.items():
            if len(examples) < self.min_frequency:
                continue

            confidence = self._calculate_pattern_confidence(pattern, examples)

            if confidence >= self.min_confidence:
                suggestions.append(PatternSuggestion(
                    pattern_name=pattern,
                    examples=examples[:5],
                    frequency=len(examples),
                    confidence=confidence,
                    reason=f"Pattern seen {len(examples)} times"
                ))

        # Also check bigrams for common phrases
        for bigram, count in self.bigram_counts.most_common(20):
            if count >= self.min_frequency:
                confidence = min(0.9, count / 20)  # Cap at 0.9
                if confidence >= self.min_confidence:
                    suggestions.append(PatternSuggestion(
                        pattern_name=f"phrase:{bigram}",
                        examples=[bigram],
                        frequency=count,
                        confidence=confidence,
                        reason=f"Phrase used {count} times"
                    ))

        return sorted(suggestions, key=lambda s: -s.confidence)

    def suggest_preferences(self) -> List[PreferenceSuggestion]:
        """
        Suggest preferences from consistent choices.

        Returns:
            List of preference suggestions sorted by confidence
        """
        suggestions = []

        # Analyze choice observations
        choices = defaultdict(lambda: Counter())
        for obs in self.observations:
            if obs.context.get('type') == 'choice':
                choice_type = obs.context['choice_type']
                chosen = obs.context['chosen']
                choices[choice_type][chosen] += 1

        for choice_type, counter in choices.items():
            if len(counter) < 2:
                continue

            # Get top two choices
            top_two = counter.most_common(2)
            if len(top_two) < 2:
                continue

            winner, winner_count = top_two[0]
            runner_up, runner_up_count = top_two[1]

            # Calculate preference strength
            total = winner_count + runner_up_count
            if total < self.min_frequency:
                continue

            ratio = winner_count / total
            if ratio >= 0.7:  # Strong preference
                confidence = ratio
                suggestions.append(PreferenceSuggestion(
                    preference_name=choice_type,
                    chosen=winner,
                    over=runner_up,
                    frequency=winner_count,
                    confidence=confidence,
                    reason=f"Chose {winner} {winner_count}/{total} times ({ratio:.0%})"
                ))

        return sorted(suggestions, key=lambda s: -s.confidence)

    def get_all_suggestions(self) -> Dict[str, List]:
        """
        Get all suggestions organized by type.

        Returns:
            Dict with 'definitions', 'patterns', 'preferences' lists
        """
        return {
            'definitions': self.suggest_definitions(),
            'patterns': self.suggest_patterns(),
            'preferences': self.suggest_preferences(),
        }

    def export_suggestions_markdown(self) -> str:
        """
        Export all suggestions as a markdown file.

        Returns:
            Markdown string ready to save as alignment file
        """
        lines = [
            "# Suggested Alignment Entries",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            f"*Based on {len(self.observations)} observations*",
            "",
        ]

        definitions = self.suggest_definitions()
        if definitions:
            lines.append("## Suggested Definitions")
            lines.append("")
            for d in definitions:
                lines.append(d.to_markdown())
            lines.append("")

        patterns = self.suggest_patterns()
        if patterns:
            lines.append("## Suggested Patterns")
            lines.append("")
            for p in patterns:
                lines.append(p.to_markdown())
            lines.append("")

        preferences = self.suggest_preferences()
        if preferences:
            lines.append("## Suggested Preferences")
            lines.append("")
            for p in preferences:
                lines.append(p.to_markdown())
            lines.append("")

        if not definitions and not patterns and not preferences:
            lines.append("*No suggestions yet. Need more observations.*")
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get suggester statistics."""
        return {
            'total_observations': len(self.observations),
            'unique_terms': len(self.term_counts),
            'known_terms': len(self.known_terms),
            'unique_bigrams': len(self.bigram_counts),
            'query_patterns': len(self.query_patterns),
            'success_rate': (
                len(self.success_contexts) / len(self.observations)
                if self.observations else 0
            ),
            'pending_suggestions': {
                'definitions': len(self.suggest_definitions()),
                'patterns': len(self.suggest_patterns()),
                'preferences': len(self.suggest_preferences()),
            }
        }

    def clear(self) -> None:
        """Clear all observations and suggestions."""
        self.observations.clear()
        self.term_counts.clear()
        self.term_contexts.clear()
        self.bigram_counts.clear()
        self.query_patterns.clear()
        self.success_contexts.clear()
        self.failure_contexts.clear()

    def _extract_terms(self, text: str) -> List[str]:
        """Extract terms from text."""
        # Simple tokenization - lowercase, alphanumeric
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        # Filter common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does',
                     'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'must', 'shall', 'can', 'need', 'dare',
                     'ought', 'used', 'to', 'of', 'in', 'for', 'on',
                     'with', 'at', 'by', 'from', 'as', 'into', 'through',
                     'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then',
                     'once', 'here', 'there', 'when', 'where', 'why',
                     'how', 'all', 'each', 'few', 'more', 'most', 'other',
                     'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                     'same', 'so', 'than', 'too', 'very', 'just', 'and',
                     'but', 'if', 'or', 'because', 'as', 'until', 'while',
                     'it', 'its', 'this', 'that', 'these', 'those', 'i',
                     'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
                     'his', 'she', 'her', 'they', 'them', 'their', 'what',
                     'which', 'who', 'whom'}
        return [w for w in words if w not in stop_words and len(w) > 2]

    def _extract_bigrams(self, text: str) -> List[str]:
        """Extract bigrams from text."""
        terms = self._extract_terms(text)
        return [f"{terms[i]} {terms[i+1]}" for i in range(len(terms)-1)]

    def _extract_pattern(self, query: str) -> Optional[str]:
        """Extract a query pattern (abstracted structure)."""
        # Detect common patterns
        query_lower = query.lower()

        if query_lower.startswith("how do i "):
            return "how_to"
        elif query_lower.startswith("where "):
            return "location"
        elif query_lower.startswith("what is "):
            return "definition"
        elif query_lower.startswith("why "):
            return "explanation"
        elif query_lower.startswith("find ") or query_lower.startswith("search "):
            return "search"
        elif query_lower.startswith("show "):
            return "display"
        elif "error" in query_lower or "bug" in query_lower:
            return "debugging"
        elif "test" in query_lower:
            return "testing"

        return None

    def _calculate_definition_confidence(self, term: str, count: int) -> float:
        """Calculate confidence for a definition suggestion."""
        # Base confidence from frequency
        freq_score = min(1.0, count / 10)  # Max at 10 occurrences

        # Boost for longer terms (more likely to be domain-specific)
        length_score = min(1.0, len(term) / 10)

        # Boost for camelCase or snake_case (likely code terms)
        format_score = 0.2 if ('_' in term or any(c.isupper() for c in term[1:])) else 0

        return min(1.0, freq_score * 0.6 + length_score * 0.2 + format_score + 0.1)

    def _calculate_pattern_confidence(self, pattern: str, examples: List[str]) -> float:
        """Calculate confidence for a pattern suggestion."""
        # Base confidence from frequency
        freq_score = min(1.0, len(examples) / 10)

        # Boost for consistent examples
        consistency = 1.0  # Could measure variance in example lengths

        return min(1.0, freq_score * 0.7 + consistency * 0.3)
