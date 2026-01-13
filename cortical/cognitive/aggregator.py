"""
Result Aggregator for Unified Query Pipeline (Phase 3).

Combines results from multiple query executors:
- Deduplicates items across sources
- Ranks by confidence and relevance
- Merges explanations into coherent response
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from cortical.cognitive.executors.protocol import ExecutionResult


@dataclass
class AggregatedResult:
    """
    Combined result from multiple executors.

    Attributes:
        items: Deduplicated, ranked result items
        sources: Which executors contributed results
        total_confidence: Overall confidence (weighted average)
        explanation: Combined explanation
        source_results: Original results by source
    """
    items: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    total_confidence: float = 0.0
    explanation: str = ""
    source_results: Dict[str, ExecutionResult] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Check if aggregated result has no items."""
        return len(self.items) == 0

    def __len__(self) -> int:
        """Number of aggregated items."""
        return len(self.items)


class ResultAggregator:
    """
    Aggregates results from multiple query executors.

    Strategies:
    - MERGE: Combine all results, deduplicate, rank by confidence
    - BEST: Take results from highest-confidence executor only
    - WEIGHTED: Weight items by source confidence

    Usage:
        aggregator = ResultAggregator()
        results = [audit_result, semantic_result, code_result]
        aggregated = aggregator.aggregate(results)
    """

    def __init__(
        self,
        strategy: str = "merge",
        min_confidence: float = 0.2,
        max_items: int = 20,
    ):
        """
        Initialize the result aggregator.

        Args:
            strategy: Aggregation strategy (merge, best, weighted)
            min_confidence: Minimum confidence to include result
            max_items: Maximum items in final result
        """
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.max_items = max_items

        # Item key extractors for deduplication
        self._key_extractors: Dict[str, Callable[[Any], Optional[str]]] = {
            "audit": self._extract_audit_key,
            "semantic": self._extract_semantic_key,
            "code": self._extract_code_key,
            "cdg": self._extract_cdg_key,
        }

    def aggregate(self, results: List[ExecutionResult]) -> AggregatedResult:
        """
        Aggregate multiple execution results.

        Args:
            results: List of ExecutionResult from different executors

        Returns:
            AggregatedResult with deduplicated, ranked items
        """
        # Filter out low-confidence results
        valid_results = [
            r for r in results
            if r.confidence >= self.min_confidence
        ]

        if not valid_results:
            return AggregatedResult(
                explanation="No results met the minimum confidence threshold.",
            )

        # Route to strategy implementation
        if self.strategy == "best":
            return self._aggregate_best(valid_results)
        elif self.strategy == "weighted":
            return self._aggregate_weighted(valid_results)
        else:  # merge (default)
            return self._aggregate_merge(valid_results)

    def _aggregate_merge(self, results: List[ExecutionResult]) -> AggregatedResult:
        """
        Merge all results, deduplicate, rank by score.

        This is the default strategy that combines results from all sources.
        """
        # Collect items with metadata
        all_items: List[Tuple[Dict[str, Any], float, str]] = []  # (item, score, source)
        sources: List[str] = []
        source_results: Dict[str, ExecutionResult] = {}
        explanations: List[str] = []

        for result in results:
            source = result.source
            if source not in sources:
                sources.append(source)
            source_results[source] = result

            if result.explanation:
                explanations.append(f"[{source}] {result.explanation}")

            # Extract and score items
            for item in result.items:
                normalized = self._normalize_item(item, source)
                # Score = item's inherent score * source confidence
                item_score = self._get_item_score(item) * result.confidence
                all_items.append((normalized, item_score, source))

        # Deduplicate by key
        seen_keys: Set[str] = set()
        deduplicated: List[Tuple[Dict[str, Any], float, str]] = []

        for item, score, source in all_items:
            key = self._get_item_key(item, source)
            if key and key not in seen_keys:
                seen_keys.add(key)
                deduplicated.append((item, score, source))
            elif not key:
                # No key means can't deduplicate, include anyway
                deduplicated.append((item, score, source))

        # Sort by score descending
        deduplicated.sort(key=lambda x: x[1], reverse=True)

        # Take top items
        final_items = [
            {**item, "_score": score, "_source": source}
            for item, score, source in deduplicated[:self.max_items]
        ]

        # Calculate overall confidence
        if results:
            total_confidence = sum(r.confidence for r in results) / len(results)
        else:
            total_confidence = 0.0

        # Build combined explanation
        combined_explanation = self._build_explanation(
            final_items, sources, explanations
        )

        return AggregatedResult(
            items=final_items,
            sources=sources,
            total_confidence=total_confidence,
            explanation=combined_explanation,
            source_results=source_results,
        )

    def _aggregate_best(self, results: List[ExecutionResult]) -> AggregatedResult:
        """
        Take results from the highest-confidence executor only.

        Use when you want authoritative results from the best source.
        """
        if not results:
            return AggregatedResult()

        # Find highest confidence result
        best_result = max(results, key=lambda r: r.confidence)

        # Normalize items
        items = [
            {**self._normalize_item(item, best_result.source), "_source": best_result.source}
            for item in best_result.items[:self.max_items]
        ]

        return AggregatedResult(
            items=items,
            sources=[best_result.source],
            total_confidence=best_result.confidence,
            explanation=best_result.explanation or f"Results from {best_result.source}",
            source_results={best_result.source: best_result},
        )

    def _aggregate_weighted(self, results: List[ExecutionResult]) -> AggregatedResult:
        """
        Weight items by their source's confidence.

        Similar to merge but with stronger preference for high-confidence sources.
        """
        # Same as merge but with confidence^2 weighting
        all_items: List[Tuple[Dict[str, Any], float, str]] = []
        sources: List[str] = []
        source_results: Dict[str, ExecutionResult] = {}
        explanations: List[str] = []

        for result in results:
            source = result.source
            if source not in sources:
                sources.append(source)
            source_results[source] = result

            if result.explanation:
                explanations.append(f"[{source}] {result.explanation}")

            # Weight by confidence squared for stronger separation
            weight = result.confidence ** 2

            for item in result.items:
                normalized = self._normalize_item(item, source)
                item_score = self._get_item_score(item) * weight
                all_items.append((normalized, item_score, source))

        # Deduplicate
        seen_keys: Set[str] = set()
        deduplicated: List[Tuple[Dict[str, Any], float, str]] = []

        for item, score, source in all_items:
            key = self._get_item_key(item, source)
            if key and key not in seen_keys:
                seen_keys.add(key)
                deduplicated.append((item, score, source))
            elif not key:
                deduplicated.append((item, score, source))

        # Sort and limit
        deduplicated.sort(key=lambda x: x[1], reverse=True)

        final_items = [
            {**item, "_score": score, "_source": source}
            for item, score, source in deduplicated[:self.max_items]
        ]

        # Weighted average confidence
        if results:
            total_weight = sum(r.confidence for r in results)
            total_confidence = sum(r.confidence ** 2 for r in results) / total_weight if total_weight > 0 else 0
        else:
            total_confidence = 0.0

        combined_explanation = self._build_explanation(
            final_items, sources, explanations
        )

        return AggregatedResult(
            items=final_items,
            sources=sources,
            total_confidence=total_confidence,
            explanation=combined_explanation,
            source_results=source_results,
        )

    def _normalize_item(self, item: Any, source: str) -> Dict[str, Any]:
        """
        Normalize an item to a common dictionary format.

        Different executors return different item formats. This normalizes
        them for consistent handling.
        """
        if isinstance(item, dict):
            return item

        # Handle objects with common attributes
        result: Dict[str, Any] = {}

        if hasattr(item, "name"):
            result["name"] = item.name
        if hasattr(item, "id"):
            result["id"] = item.id
        if hasattr(item, "file_path"):
            result["file_path"] = item.file_path

        # If we couldn't extract anything, stringify
        if not result:
            result["value"] = str(item)

        return result

    def _get_item_score(self, item: Any) -> float:
        """
        Extract a score from an item.

        Items may have: score, risk_score, relevance, confidence, etc.
        """
        if isinstance(item, dict):
            # Try common score keys
            for key in ["score", "risk_score", "relevance", "confidence", "priority"]:
                if key in item:
                    try:
                        return float(item[key])
                    except (TypeError, ValueError):
                        pass
        return 1.0  # Default score if none found

    def _get_item_key(self, item: Dict[str, Any], source: str) -> Optional[str]:
        """
        Get a unique key for an item for deduplication.

        Uses source-specific extractors when available.
        """
        extractor = self._key_extractors.get(source)
        if extractor:
            key = extractor(item)
            if key:
                return key

        # Fallback: use common identifiers
        for key_field in ["id", "file", "doc_id", "name", "file_path"]:
            if key_field in item:
                return str(item[key_field])

        return None

    def _extract_audit_key(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract key from audit result item."""
        return item.get("file") or item.get("file_id")

    def _extract_semantic_key(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract key from semantic result item."""
        return item.get("doc_id") or item.get("path")

    def _extract_code_key(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract key from code result item."""
        name = item.get("name", "")
        file_path = item.get("file_path", "")
        if name and file_path:
            return f"{file_path}:{name}"
        return name or None

    def _extract_cdg_key(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract key from CDG result item."""
        return item.get("id")

    def _build_explanation(
        self,
        items: List[Dict[str, Any]],
        sources: List[str],
        source_explanations: List[str],
    ) -> str:
        """Build a combined explanation from multiple sources."""
        parts = []

        # Summary
        if items:
            parts.append(f"Found {len(items)} results from {len(sources)} source(s).")
        else:
            parts.append("No results found.")

        # Source breakdown
        if len(sources) > 1:
            source_counts: Dict[str, int] = {}
            for item in items:
                src = item.get("_source", "unknown")
                source_counts[src] = source_counts.get(src, 0) + 1

            breakdown = ", ".join(
                f"{src}: {count}" for src, count in source_counts.items()
            )
            parts.append(f"Sources: {breakdown}")

        # Include first source explanation if available
        if source_explanations:
            # Take first non-empty explanation
            for exp in source_explanations:
                if exp:
                    parts.append(exp)
                    break

        return " ".join(parts)


def aggregate_results(
    results: List[ExecutionResult],
    strategy: str = "merge",
    min_confidence: float = 0.2,
    max_items: int = 20,
) -> AggregatedResult:
    """
    Convenience function to aggregate results.

    Args:
        results: List of ExecutionResult from executors
        strategy: Aggregation strategy (merge, best, weighted)
        min_confidence: Minimum confidence threshold
        max_items: Maximum items in result

    Returns:
        AggregatedResult with combined items
    """
    aggregator = ResultAggregator(
        strategy=strategy,
        min_confidence=min_confidence,
        max_items=max_items,
    )
    return aggregator.aggregate(results)
