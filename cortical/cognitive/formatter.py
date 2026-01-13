"""
Response Formatter for Unified Query Pipeline (Phase 4).

Formats aggregated results as natural language responses:
- Query type-specific templates (audit, cdg, code, semantic)
- PLN inference explanations and trace output
- "Why" question handling with reasoning chains
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from cortical.cognitive.unified_query import UnifiedQuery, QueryIntent
from cortical.cognitive.aggregator import AggregatedResult


@dataclass
class FormatterConfig:
    """Configuration for response formatting."""
    max_items_shown: int = 10
    show_confidence: bool = False
    show_sources: bool = True
    show_scores: bool = False
    include_trace: bool = False
    verbose: bool = False


class ResponseFormatter:
    """
    Formats execution results as natural language.

    Provides query type-specific formatting:
    - audit: Risk scores, PLN explanations, suggestions
    - cdg: Entity listings with properties
    - code: File locations, function signatures
    - semantic: Document associations, similarity scores

    Usage:
        formatter = ResponseFormatter()
        response = formatter.format(unified_query, aggregated_result)
    """

    def __init__(self, config: Optional[FormatterConfig] = None):
        """
        Initialize the response formatter.

        Args:
            config: Formatting configuration
        """
        self.config = config or FormatterConfig()

        # Dispatch table for query types
        self._formatters: Dict[str, Callable] = {
            "audit": self._format_audit_response,
            "cdg": self._format_cdg_response,
            "code": self._format_code_response,
            "semantic": self._format_semantic_response,
        }

    def format(
        self,
        query: UnifiedQuery,
        result: AggregatedResult,
    ) -> str:
        """
        Format an aggregated result as natural language.

        Args:
            query: The unified query that was executed
            result: The aggregated result from executors

        Returns:
            Human-readable response string
        """
        if result.is_empty:
            return self._format_empty_response(query)

        # Check for "why" questions first
        if self._is_why_question(query):
            return self._format_why_response(query, result)

        # Route to query type-specific formatter
        formatter = self._formatters.get(query.query_type)
        if formatter:
            return formatter(query, result)

        # Fallback to generic format
        return self._format_generic_response(query, result)

    def _is_why_question(self, query: UnifiedQuery) -> bool:
        """Check if this is a 'why' question requiring explanation."""
        question_lower = query.raw_question.lower()
        return (
            question_lower.startswith("why ") or
            " why " in question_lower or
            query.metadata.get("intent") == "explain"
        )

    def _format_empty_response(self, query: UnifiedQuery) -> str:
        """Format response when no results found."""
        question = query.raw_question
        query_type = query.query_type

        # Query type-specific empty messages
        if query_type == "audit":
            return (
                f"No audit results found for: {question}\n"
                "Hint: Run 'audit analyze' to scan files first."
            )
        elif query_type == "cdg":
            return f"No matching entities found for: {question}"
        elif query_type == "code":
            return f"No code locations found for: {question}"
        else:
            return f"No results found for: {question}"

    def _format_why_response(
        self,
        query: UnifiedQuery,
        result: AggregatedResult,
    ) -> str:
        """Format response for 'why' questions with explanation trace."""
        lines = []

        # Get the first item which should contain explanation
        if not result.items:
            return f"Unable to explain: {query.raw_question}"

        item = result.items[0]

        # Check if this is an audit explanation (has PLN trace)
        if self._is_audit_explanation(item):
            return self._format_pln_explanation(item, query)

        # Generic why response
        lines.append(f"Explanation for: {query.raw_question}")
        lines.append("")

        # Extract any explanation-like data
        if isinstance(item, dict):
            if "summary" in item:
                lines.append(item["summary"])
            elif "explanation" in item:
                lines.append(item["explanation"])
            elif "reason" in item:
                lines.append(item["reason"])
            else:
                # Show the item data as explanation
                lines.append(self._format_item_as_explanation(item))
        else:
            lines.append(str(item))

        # Add source attribution
        if self.config.show_sources and result.sources:
            lines.append("")
            lines.append(f"Source: {', '.join(result.sources)}")

        return "\n".join(lines)

    def _is_audit_explanation(self, item: Any) -> bool:
        """Check if item is an audit/PLN explanation."""
        if not isinstance(item, dict):
            return False
        return any(key in item for key in ["facts", "risk_level", "traces", "inferences"])

    def _format_pln_explanation(self, item: Dict[str, Any], query: UnifiedQuery) -> str:
        """Format PLN-based explanation with inference trace."""
        lines = []

        file_id = item.get("file_id", "unknown")
        lines.append(f"Risk Analysis: {file_id}")
        lines.append("=" * 40)

        # Risk level summary
        risk_level = item.get("risk_level")
        if risk_level:
            mean = risk_level.get("mean", 0)
            strength = risk_level.get("strength", 0)
            confidence = risk_level.get("confidence", 0)

            risk_category = self._categorize_risk(mean)
            lines.append(f"\nOverall Risk: {risk_category} ({mean:.2f})")
            if self.config.verbose:
                lines.append(f"  Strength: {strength:.2f}")
                lines.append(f"  Confidence: {confidence:.2f}")

        # Evidence (facts)
        facts = item.get("facts", [])
        if facts:
            lines.append(f"\nEvidence ({len(facts)} facts):")
            for i, fact in enumerate(facts[:self.config.max_items_shown], 1):
                atom = fact.get("atom", "")
                # Parse the atom for readable format
                readable = self._make_fact_readable(atom)
                lines.append(f"  {i}. {readable}")
            if len(facts) > self.config.max_items_shown:
                lines.append(f"  ... and {len(facts) - self.config.max_items_shown} more")

        # Inference chains
        inferences = item.get("inferences", [])
        if inferences:
            lines.append(f"\nInference Chain:")
            for inf in inferences[:5]:
                lines.append(f"  → {inf}")

        # Traces (if enabled and available)
        # TODO: Add detailed trace output when config.include_trace is True
        # This requires access to the InferenceTrace from PLN reasoner
        traces = item.get("traces", {})
        if self.config.include_trace and traces:
            lines.append("\nReasoning Trace:")
            for trace_name, trace_data in traces.items():
                lines.append(f"  [{trace_name}]: {trace_data}")

        # Suggestions
        suggestions = item.get("suggestions", [])
        if suggestions:
            lines.append("\nRecommendations:")
            for suggestion in suggestions:
                lines.append(f"  • {suggestion}")

        # Summary
        summary = item.get("summary", "")
        if summary:
            lines.append(f"\nSummary: {summary}")

        return "\n".join(lines)

    def _categorize_risk(self, score: float) -> str:
        """Convert risk score to category."""
        if score >= 0.8:
            return "CRITICAL"
        elif score >= 0.6:
            return "HIGH"
        elif score >= 0.4:
            return "MODERATE"
        elif score >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"

    def _make_fact_readable(self, atom: str) -> str:
        """Convert PLN atom notation to readable text."""
        # Pattern: has_trait(file_id, trait) -> "file_id has trait"
        if atom.startswith("has_trait(") and atom.endswith(")"):
            inner = atom[len("has_trait("):-1]
            parts = inner.split(",", 1)
            if len(parts) == 2:
                return f"{parts[0].strip()} has trait: {parts[1].strip()}"

        # Pattern: has_pattern(file_id, pattern) -> "file_id contains pattern"
        if atom.startswith("has_pattern(") and atom.endswith(")"):
            inner = atom[len("has_pattern("):-1]
            parts = inner.split(",", 1)
            if len(parts) == 2:
                return f"{parts[0].strip()} contains: {parts[1].strip()}"

        # Pattern: is_risky(file_id) -> "file_id is marked risky"
        if atom.startswith("is_risky(") and atom.endswith(")"):
            file_id = atom[len("is_risky("):-1].strip()
            return f"{file_id} is marked as risky"

        return atom

    def _format_item_as_explanation(self, item: Dict[str, Any]) -> str:
        """Format a generic item as explanation text."""
        parts = []
        for key, value in item.items():
            if key.startswith("_"):  # Skip internal keys
                continue
            if isinstance(value, (list, dict)):
                continue  # Skip complex values
            parts.append(f"{key}: {value}")
        return ", ".join(parts) if parts else str(item)

    def _format_audit_response(
        self,
        query: UnifiedQuery,
        result: AggregatedResult,
    ) -> str:
        """Format audit query results."""
        lines = []

        # Header with summary
        count = len(result.items)
        if count == 1:
            lines.append("Found 1 file:")
        else:
            lines.append(f"Found {count} files:")
        lines.append("")

        # List files with risk scores
        for i, item in enumerate(result.items[:self.config.max_items_shown], 1):
            file_name = item.get("file", item.get("file_path", "unknown"))
            risk_score = item.get("risk_score", item.get("_score", 0))

            risk_category = self._categorize_risk(risk_score)
            line = f"  {i}. {file_name}"

            if self.config.show_scores:
                line += f" (risk: {risk_score:.2f} - {risk_category})"
            else:
                line += f" [{risk_category}]"

            lines.append(line)

        # Show truncation notice
        if count > self.config.max_items_shown:
            remaining = count - self.config.max_items_shown
            lines.append(f"  ... and {remaining} more files")

        # Add confidence info if enabled
        if self.config.show_confidence:
            lines.append("")
            lines.append(f"Confidence: {result.total_confidence:.2f}")

        return "\n".join(lines)

    def _format_cdg_response(
        self,
        query: UnifiedQuery,
        result: AggregatedResult,
    ) -> str:
        """Format CDG query results."""
        lines = []

        count = len(result.items)
        lines.append(f"Query: {query.raw_question}")
        lines.append(f"Found {count} entities:")
        lines.append("")

        for i, item in enumerate(result.items[:self.config.max_items_shown], 1):
            # CDG items typically have id, type, and properties
            entity_id = item.get("id", "unknown")
            entity_type = item.get("type", item.get("entity_type", ""))

            if entity_type:
                lines.append(f"  {i}. [{entity_type}] {entity_id}")
            else:
                lines.append(f"  {i}. {entity_id}")

            # Show key properties if verbose
            if self.config.verbose:
                for key in ["status", "priority", "category", "title"]:
                    if key in item:
                        lines.append(f"       {key}: {item[key]}")

        if count > self.config.max_items_shown:
            lines.append(f"  ... and {count - self.config.max_items_shown} more")

        return "\n".join(lines)

    def _format_code_response(
        self,
        query: UnifiedQuery,
        result: AggregatedResult,
    ) -> str:
        """Format code intent query results."""
        lines = []

        count = len(result.items)
        lines.append(f"Code locations for: {query.raw_question}")
        lines.append(f"Found {count} matches:")
        lines.append("")

        for i, item in enumerate(result.items[:self.config.max_items_shown], 1):
            # Code items have file_path, name, line_number
            file_path = item.get("file_path", item.get("file", "unknown"))
            name = item.get("name", "")
            line_num = item.get("line_number", item.get("line", ""))

            if line_num:
                location = f"{file_path}:{line_num}"
            else:
                location = file_path

            if name:
                lines.append(f"  {i}. {name} at {location}")
            else:
                lines.append(f"  {i}. {location}")

            # Show context if available
            if self.config.verbose and "context" in item:
                context = item["context"]
                if isinstance(context, str) and context:
                    # Indent context lines
                    for ctx_line in context.split("\n")[:3]:
                        lines.append(f"       {ctx_line.strip()}")

        if count > self.config.max_items_shown:
            lines.append(f"  ... and {count - self.config.max_items_shown} more")

        return "\n".join(lines)

    def _format_semantic_response(
        self,
        query: UnifiedQuery,
        result: AggregatedResult,
    ) -> str:
        """Format semantic/association query results."""
        lines = []

        # Extract question type for context
        parsed = query.parsed
        question_type = ""
        if isinstance(parsed, QueryIntent):
            question_type = parsed.question_type

        count = len(result.items)
        lines.append(f"Results for: {query.raw_question}")
        lines.append(f"Found {count} related documents:")
        lines.append("")

        for i, item in enumerate(result.items[:self.config.max_items_shown], 1):
            # Semantic items have doc_id, score, excerpt from SemanticExecutor
            doc_id = item.get("doc_id", item.get("name", item.get("id", "unknown")))
            score = item.get("score", item.get("similarity", item.get("_score", 0)))
            excerpt = item.get("excerpt", "")

            # Show document header with optional score
            if self.config.show_scores:
                lines.append(f"  {i}. {doc_id} (relevance: {score:.2f})")
            else:
                lines.append(f"  {i}. {doc_id}")

            # Always show excerpt if available - this is the useful content!
            if excerpt:
                # Indent and wrap excerpt lines
                excerpt_lines = excerpt.split("\n")
                for j, exc_line in enumerate(excerpt_lines[:5]):  # Limit to 5 lines
                    # Trim long lines
                    trimmed = exc_line.strip()[:100]
                    if trimmed:
                        lines.append(f"       {trimmed}")
                if len(excerpt_lines) > 5:
                    lines.append("       ...")
                lines.append("")  # Blank line between items

            # Show associations if verbose (in addition to excerpt)
            if self.config.verbose:
                associations = item.get("associations", [])
                if associations:
                    assoc_str = ", ".join(str(a) for a in associations[:3])
                    lines.append(f"       Related: {assoc_str}")

        if count > self.config.max_items_shown:
            lines.append(f"  ... and {count - self.config.max_items_shown} more documents")

        # Add explanation from aggregator
        if result.explanation and self.config.verbose:
            lines.append("")
            lines.append(result.explanation)

        return "\n".join(lines)

    def _format_generic_response(
        self,
        query: UnifiedQuery,
        result: AggregatedResult,
    ) -> str:
        """Fallback generic formatting."""
        lines = []

        count = len(result.items)
        lines.append(f"Results for: {query.raw_question}")
        lines.append(f"Found {count} items:")
        lines.append("")

        for i, item in enumerate(result.items[:self.config.max_items_shown], 1):
            if isinstance(item, dict):
                # Try common display keys
                display = (
                    item.get("name") or
                    item.get("file") or
                    item.get("id") or
                    item.get("doc_id") or
                    str(item)
                )
            else:
                display = str(item)

            lines.append(f"  {i}. {display}")

        if count > self.config.max_items_shown:
            lines.append(f"  ... and {count - self.config.max_items_shown} more")

        if self.config.show_sources and result.sources:
            lines.append("")
            lines.append(f"Sources: {', '.join(result.sources)}")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

def format_response(
    query: UnifiedQuery,
    result: AggregatedResult,
    config: Optional[FormatterConfig] = None,
) -> str:
    """
    Convenience function to format a response.

    Args:
        query: The unified query
        result: The aggregated result
        config: Optional formatting configuration

    Returns:
        Formatted response string
    """
    formatter = ResponseFormatter(config)
    return formatter.format(query, result)
