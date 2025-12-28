"""
LLM-Optimized Tool Designs

This module defines tool interfaces designed specifically for LLM limitations:
- Structured I/O for reliable parsing
- Sensible defaults with full configurability
- Progressive disclosure (concise by default, expandable)
- Error handling that enables reasoning about failures
- Composability for multi-step operations

Design Principles:
1. Tools should externalize what LLMs are bad at (memory, counting, verification)
2. Tools should leverage what LLMs are good at (reasoning, synthesis, planning)
3. Concise defaults, optional expansion
4. Single calls that handle pipelines internally when possible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator, Literal

from .types import SearchResult, SearchResponse, ToolError


# =============================================================================
# SEARCH CONFIG
# =============================================================================


@dataclass
class SearchConfig:
    """Configuration for search operations."""

    # Scope control
    top_n: int = 5
    max_depth: int = 5
    timeout_ms: int = 5000

    # Targeting
    include_patterns: list[str] = field(
        default_factory=lambda: ["**/*.py", "**/*.md"]
    )
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["**/node_modules/**", "**/.git/**"]
    )

    # Behavior
    expand_query: bool = True
    match_mode: Literal["exact", "fuzzy", "semantic"] = "semantic"
    context_lines: int = 3

    # Output
    include_snippets: bool = True
    deduplicate: bool = True
    verbosity: Literal["minimal", "summary", "full"] = "summary"


@dataclass
class SearchPlan:
    """Preview of what a search would do (dry run)."""

    query_interpreted: str
    files_to_search: list[str]
    estimated_time_ms: float
    expansions_planned: list[str]
    estimated_results: int


# =============================================================================
# FLUENT SEARCH BUILDER
# =============================================================================


class SearchBuilder:
    """
    Fluent API for building search queries.

    Mirrors natural language thinking about searches:
        Search("authentication")
            .in_files("**/*.py")
            .excluding("**/test_*")
            .with_expansion()
            .limit(10)
            .execute()
    """

    def __init__(self, query: str):
        self._query = query
        self._config = SearchConfig()

    def in_files(self, *patterns: str) -> SearchBuilder:
        """Specify file patterns to search in."""
        self._config.include_patterns = list(patterns)
        return self

    def excluding(self, *patterns: str) -> SearchBuilder:
        """Specify patterns to exclude."""
        self._config.exclude_patterns.extend(patterns)
        return self

    def with_expansion(self, enabled: bool = True) -> SearchBuilder:
        """Enable/disable query expansion."""
        self._config.expand_query = enabled
        return self

    def limit(self, n: int) -> SearchBuilder:
        """Limit number of results."""
        self._config.top_n = n
        return self

    def timeout(self, ms: int) -> SearchBuilder:
        """Set timeout in milliseconds."""
        self._config.timeout_ms = ms
        return self

    def with_context(self, lines: int) -> SearchBuilder:
        """Set context lines around matches."""
        self._config.context_lines = lines
        return self

    def match_mode(
        self,
        mode: Literal["exact", "fuzzy", "semantic"],
    ) -> SearchBuilder:
        """Set matching mode."""
        self._config.match_mode = mode
        return self

    def verbosity(
        self,
        level: Literal["minimal", "summary", "full"],
    ) -> SearchBuilder:
        """Set output verbosity."""
        self._config.verbosity = level
        return self

    def dry_run(self) -> SearchPlan:
        """Preview what the search would do without executing."""
        return SearchPlan(
            query_interpreted=self._parse_intent(),
            files_to_search=self._list_target_files(),
            estimated_time_ms=self._estimate_time(),
            expansions_planned=self._plan_expansions(),
            estimated_results=self._estimate_results(),
        )

    def execute(self) -> SearchResponse:
        """Execute the search."""
        searcher = SemanticSearch()
        return searcher.search(self._query, config=self._config)

    # Composability: chain from results
    def then_read(self, context_lines: int = 10) -> ReadBuilder:
        """Chain: search → read expanded context."""
        return ReadBuilder(self, context_lines)

    def then_find_related(
        self,
        relationship: str = "imports",
    ) -> SearchBuilder:
        """Chain: search → graph traversal for related files."""
        # Would execute search, then search for related
        return self

    # Private helpers
    def _parse_intent(self) -> str:
        return self._query

    def _list_target_files(self) -> list[str]:
        return []  # Would glob based on patterns

    def _estimate_time(self) -> float:
        return 100.0  # Placeholder

    def _plan_expansions(self) -> list[str]:
        if not self._config.expand_query:
            return []
        return []  # Would compute expansions

    def _estimate_results(self) -> int:
        return 10  # Placeholder


class ReadBuilder:
    """Builder for read operations chained from search."""

    def __init__(self, search_builder: SearchBuilder, context_lines: int):
        self._search = search_builder
        self._context_lines = context_lines

    def execute(self) -> dict[str, str]:
        """Execute search then read results."""
        results = self._search.execute()
        content = {}
        for result in results.results:
            content[result.file_path] = self._read_with_context(
                result.file_path,
                result.line_range,
                self._context_lines,
            )
        return content

    def _read_with_context(
        self,
        path: str,
        line_range: tuple[int, int],
        context: int,
    ) -> str:
        # Would read file with context
        return ""


# =============================================================================
# SEMANTIC SEARCH
# =============================================================================


class SemanticSearch:
    """
    Semantic search tool with full features.

    Designed for LLM use with:
    - Structured output for reliable parsing
    - Rich metadata for verification
    - Composable follow-up operations
    """

    def search(
        self,
        query: str,
        *,
        config: SearchConfig | None = None,
        trace_id: str | None = None,
    ) -> SearchResponse:
        """
        Execute a semantic search.

        Args:
            query: The search query
            config: Optional configuration (uses defaults if not provided)
            trace_id: Optional trace ID for observability

        Returns:
            SearchResponse with results and metadata
        """
        if config is None:
            config = SearchConfig()

        try:
            # 1. Interpret query
            interpreted = self._interpret_query(query)
            expansions = (
                self._expand_query(query)
                if config.expand_query
                else []
            )

            # 2. Execute search
            raw_results = self._execute_search(
                query=query,
                expansions=expansions,
                config=config,
            )

            # 3. Rank and filter
            ranked = self._rank_results(raw_results, query)
            truncated = len(ranked) > config.top_n
            final = ranked[:config.top_n]

            # 4. Build response
            return SearchResponse(
                results=final,
                query_interpreted=interpreted,
                expansions_used=expansions,
                total_matches=len(ranked),
                truncated=truncated,
                search_time_ms=0.0,  # Would measure
                corpus_stats=self._get_corpus_stats(),
                status="success",
            )

        except TimeoutError:
            return SearchResponse(
                status="failed",
                errors=[ToolError(
                    code="TIMEOUT",
                    message=f"Search timed out after {config.timeout_ms}ms",
                    recoverable=True,
                    suggestion="Try reducing scope or simplifying query",
                )],
            )

        except Exception as e:
            return SearchResponse(
                status="failed",
                errors=[ToolError(
                    code="UNKNOWN",
                    message=str(e),
                    recoverable=False,
                )],
            )

    def read_around(
        self,
        result: SearchResult,
        lines_before: int = 10,
        lines_after: int = 10,
    ) -> str:
        """Expand a result with more context."""
        return self._read_lines(
            result.file_path,
            result.line_range[0] - lines_before,
            result.line_range[1] + lines_after,
        )

    def find_related(
        self,
        result: SearchResult,
        relationship: Literal["imports", "references", "tests"] = "imports",
    ) -> list[SearchResult]:
        """Find files related to a search result."""
        return []  # Would traverse dependency graph

    def explain_ranking(self, result: SearchResult) -> dict[str, float]:
        """Explain why a result ranked where it did."""
        return {
            "tfidf_contribution": 0.0,
            "semantic_similarity": 0.0,
            "recency_boost": 0.0,
            "path_relevance": 0.0,
        }

    def verify_exists(self, result: SearchResult) -> bool:
        """Verify the file/content still exists."""
        return True  # Would check filesystem

    def get_corpus_stats(self) -> dict[str, Any]:
        """Get corpus-level statistics."""
        return self._get_corpus_stats()

    # Private helpers
    def _interpret_query(self, query: str) -> str:
        return query

    def _expand_query(self, query: str) -> list[str]:
        return []

    def _execute_search(
        self,
        query: str,
        expansions: list[str],
        config: SearchConfig,
    ) -> list[SearchResult]:
        return []

    def _rank_results(
        self,
        results: list[SearchResult],
        query: str,
    ) -> list[SearchResult]:
        return sorted(results, key=lambda r: -r.relevance_score)

    def _get_corpus_stats(self) -> dict[str, Any]:
        return {
            "total_documents": 0,
            "total_tokens": 0,
            "vocabulary_size": 0,
            "last_indexed": None,
        }

    def _read_lines(
        self,
        path: str,
        start: int,
        end: int,
    ) -> str:
        return ""


# =============================================================================
# PRACTICAL SEARCH (Optimized for LLM limitations)
# =============================================================================


@dataclass
class CompactResult:
    """Minimal footprint result - just enough to act on."""

    path: str
    line: int
    snippet: str  # 1-2 lines, not paragraphs
    score: float


@dataclass
class CompactResponse:
    """Minimal response for context efficiency."""

    results: list[CompactResult]
    ok: bool
    message: str | None = None
    total_found: int = 0
    truncated: bool = False


@dataclass
class SearchWithContentResponse:
    """Response from search_and_read operation."""

    query: str
    top_result: CompactResult | None
    expanded_content: str  # Full content of best match
    other_results: list[CompactResult]
    ok: bool
    message: str | None = None


@dataclass
class ExplorationSummary:
    """Narrative summary of an exploration."""

    starting_point: str
    goal: str
    steps_taken: int
    findings: list[str]
    related_files: list[str]
    suggested_next: list[str]
    narrative: str  # Human-readable summary


class PracticalSearch:
    """
    Search tools optimized for actual LLM limitations.

    Key differences from ideal design:
    - Returns concise results by default
    - Combined operations (search_and_read) reduce round trips
    - Text summaries instead of complex structured data
    - Progressive disclosure through follow-up calls
    """

    def search(
        self,
        query: str,
        top_n: int = 5,
        verbosity: Literal["minimal", "summary", "full"] = "summary",
    ) -> CompactResponse:
        """
        Simple search with concise output.

        Most searches need 5 results max, not 50.
        """
        try:
            results = self._do_search(query, top_n)
            return CompactResponse(
                results=results,
                ok=True,
                total_found=len(results),
                truncated=False,
            )
        except Exception as e:
            return CompactResponse(
                results=[],
                ok=False,
                message=str(e),
            )

    def search_and_read(
        self,
        query: str,
        auto_expand_top: int = 1,
    ) -> SearchWithContentResponse:
        """
        Single call: search → identify best → read context.

        Reduces 3 tool calls to 1.
        """
        # Search
        search_result = self.search(query, top_n=5)
        if not search_result.ok or not search_result.results:
            return SearchWithContentResponse(
                query=query,
                top_result=None,
                expanded_content="",
                other_results=[],
                ok=False,
                message=search_result.message or "No results found",
            )

        # Read best match
        top = search_result.results[0]
        content = self._read_file_context(top.path, top.line)

        return SearchWithContentResponse(
            query=query,
            top_result=top,
            expanded_content=content,
            other_results=search_result.results[1:auto_expand_top],
            ok=True,
        )

    def search_and_summarize(self, query: str) -> str:
        """
        Returns a TEXT SUMMARY, not structured data.

        LLMs can parse natural language more reliably than
        complex nested JSON.
        """
        results = self.search(query, top_n=10)

        if not results.ok:
            return f"Search failed: {results.message}"

        if not results.results:
            return f"No results found for: {query}"

        # Group by file
        by_file: dict[str, list[CompactResult]] = {}
        for r in results.results:
            if r.path not in by_file:
                by_file[r.path] = []
            by_file[r.path].append(r)

        # Build narrative
        lines = [
            f"Found {results.total_found} matches across {len(by_file)} files.",
            "",
            "Most relevant:",
        ]

        for path, file_results in list(by_file.items())[:3]:
            best = max(file_results, key=lambda r: r.score)
            lines.append(
                f"  - {path}:{best.line} (score: {best.score:.2f})"
            )
            lines.append(f"    {best.snippet[:100]}...")

        if results.truncated:
            lines.append("")
            lines.append(
                f"({results.total_found - len(results.results)} more results not shown)"
            )

        return "\n".join(lines)

    def explore(
        self,
        starting_point: str,
        goal: str,
        max_steps: int = 5,
    ) -> ExplorationSummary:
        """
        Multi-hop exploration done internally.

        Instead of LLM issuing 5 searches and losing track,
        the tool does the exploration and returns a narrative.
        """
        findings = []
        related_files = []
        steps = 0

        current = starting_point
        visited = set()

        while steps < max_steps:
            if current in visited:
                break
            visited.add(current)

            # Search from current point
            results = self.search(f"{goal} {current}", top_n=3)
            if results.ok and results.results:
                for r in results.results:
                    if r.path not in related_files:
                        related_files.append(r.path)
                        findings.append(f"Found {r.path}:{r.line} - {r.snippet[:50]}")

            steps += 1

        # Build narrative
        narrative = f"""Exploration from "{starting_point}" toward "{goal}":

Took {steps} steps, found {len(findings)} relevant items.

Key findings:
{chr(10).join('- ' + f for f in findings[:5])}

Related files to examine:
{chr(10).join('- ' + f for f in related_files[:5])}
"""

        return ExplorationSummary(
            starting_point=starting_point,
            goal=goal,
            steps_taken=steps,
            findings=findings,
            related_files=related_files,
            suggested_next=related_files[:3],
            narrative=narrative,
        )

    # Private helpers
    def _do_search(self, query: str, top_n: int) -> list[CompactResult]:
        return []  # Would call actual search

    def _read_file_context(self, path: str, line: int) -> str:
        return ""  # Would read file


# =============================================================================
# STREAMING SEARCH (For large result sets)
# =============================================================================


class StreamingSearch:
    """
    For large corpora, stream results instead of collecting all.

    LLM can process incrementally and stop early when satisfied.
    """

    def search_stream(
        self,
        query: str,
        **kwargs: Any,
    ) -> Iterator[SearchResult]:
        """
        Yields results one at a time.

        LLM can stop when it has enough without waiting for all.
        """
        for result in self._search_generator(query, **kwargs):
            yield result

    def search_batched(
        self,
        query: str,
        batch_size: int = 10,
        **kwargs: Any,
    ) -> Iterator[list[SearchResult]]:
        """
        Yields in batches - good for parallel processing.
        """
        batch = []
        for result in self._search_generator(query, **kwargs):
            batch.append(result)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _search_generator(
        self,
        query: str,
        **kwargs: Any,
    ) -> Iterator[SearchResult]:
        return iter([])  # Would generate results


# =============================================================================
# CACHED SEARCH
# =============================================================================


@dataclass
class CacheEntry:
    """A cached search result."""

    query: str
    response: SearchResponse
    cached_at: datetime
    ttl_seconds: int


class CachedSearch:
    """
    Cache repeated/similar queries.

    Critical for exploration patterns where LLM refines queries.
    """

    def __init__(self):
        self._cache: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def search(
        self,
        query: str,
        *,
        use_cache: bool = True,
        cache_ttl_seconds: int = 300,
        **kwargs: Any,
    ) -> SearchResponse:
        """Search with optional caching."""
        cache_key = self._make_key(query, kwargs)

        # Check cache
        if use_cache and cache_key in self._cache:
            entry = self._cache[cache_key]
            age = (datetime.now() - entry.cached_at).total_seconds()
            if age < entry.ttl_seconds:
                self._hits += 1
                return entry.response

        # Cache miss
        self._misses += 1
        searcher = SemanticSearch()
        response = searcher.search(query)

        # Store in cache
        if use_cache:
            self._cache[cache_key] = CacheEntry(
                query=query,
                response=response,
                cached_at=datetime.now(),
                ttl_seconds=cache_ttl_seconds,
            )

        return response

    def invalidate_cache(self, pattern: str = "*") -> int:
        """Clear cache entries matching pattern."""
        if pattern == "*":
            count = len(self._cache)
            self._cache.clear()
            return count

        count = 0
        to_remove = []
        for key in self._cache:
            if pattern in key:
                to_remove.append(key)
                count += 1
        for key in to_remove:
            del self._cache[key]
        return count

    def cache_stats(self) -> dict[str, Any]:
        """Visibility into cache performance."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hit_rate": hit_rate,
            "hits": self._hits,
            "misses": self._misses,
            "entries": len(self._cache),
            "oldest_entry": min(
                (e.cached_at for e in self._cache.values()),
                default=None,
            ),
        }

    def _make_key(self, query: str, kwargs: dict[str, Any]) -> str:
        return f"{query}:{hash(frozenset(kwargs.items()))}"


# =============================================================================
# OBSERVABLE SEARCH (With tracing)
# =============================================================================


@dataclass
class TraceStep:
    """A step in an execution trace."""

    name: str
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchTrace:
    """Detailed execution trace for debugging."""

    trace_id: str
    query: str
    steps: list[TraceStep] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    resource_usage: dict[str, Any] = field(default_factory=dict)


class ObservableSearch:
    """Search with observability hooks for debugging."""

    def __init__(self):
        self._traces: dict[str, SearchTrace] = {}

    def search(
        self,
        query: str,
        *,
        trace_id: str | None = None,
        log_level: Literal["debug", "info", "warn"] = "info",
        **kwargs: Any,
    ) -> SearchResponse:
        """Search with tracing enabled."""
        if trace_id is None:
            trace_id = self._generate_trace_id()

        trace = SearchTrace(trace_id=trace_id, query=query)
        self._traces[trace_id] = trace

        # Would instrument each step
        searcher = SemanticSearch()
        return searcher.search(query)

    def get_trace(self, trace_id: str) -> SearchTrace | None:
        """Get detailed execution trace for debugging."""
        return self._traces.get(trace_id)

    def _generate_trace_id(self) -> str:
        return f"trace-{datetime.now().strftime('%Y%m%d%H%M%S')}"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def Search(query: str) -> SearchBuilder:
    """Convenience function to start a search builder."""
    return SearchBuilder(query)


def quick_search(query: str, top_n: int = 5) -> CompactResponse:
    """Quick search with minimal setup."""
    return PracticalSearch().search(query, top_n=top_n)


def search_and_read(query: str) -> SearchWithContentResponse:
    """Search and automatically read the best result."""
    return PracticalSearch().search_and_read(query)
