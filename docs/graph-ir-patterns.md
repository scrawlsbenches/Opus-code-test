# Information Retrieval Patterns for Task/Decision Graph Databases

## Overview

This document details information retrieval patterns for systems that must handle both:
- **Structured queries** on task/decision graphs (graph traversal, dependency analysis)
- **Unstructured queries** on entity content (full-text search, semantic expansion)

Based on real implementations in the Cortical Text Processor and GoT (Graph of Thought) system, these patterns combine graph algorithms with text retrieval techniques.

---

## 1. Graph Traversal Queries

Graph traversal patterns answer questions about entity relationships: "What depends on X?", "Path from A to B", "All entities in X's influence sphere".

### 1.1 Basic Traversal Patterns

#### Forward Traversal (Following edges outward)
```python
def get_all_dependents(entity_id: str, max_depth: int = None) -> List[Entity]:
    """
    Get all entities that transitively depend on the given entity.
    Answers: "What depends on this task?"
    """
    visited = set()
    queue = [(entity_id, 0)]  # (id, depth)
    dependents = []

    while queue:
        current_id, depth = queue.pop(0)

        if current_id in visited:
            continue
        if max_depth and depth > max_depth:
            continue

        visited.add(current_id)

        # Find all edges where current_id is the target
        # (incoming edges = entities depending on this one)
        for edge in find_edges(target_id=current_id, edge_type="DEPENDS_ON"):
            dependent = get_entity(edge.source_id)
            if dependent:
                dependents.append(dependent)
                queue.append((edge.source_id, depth + 1))

    return dependents
```

#### Backward Traversal (Following edges inward)
```python
def get_all_dependencies(entity_id: str, max_depth: int = None) -> List[Entity]:
    """
    Get all entities this task transitively depends on.
    Answers: "What must be done before this task?"
    """
    visited = set()
    queue = [(entity_id, 0)]
    dependencies = []

    while queue:
        current_id, depth = queue.pop(0)

        if current_id in visited:
            continue
        if max_depth and depth > max_depth:
            continue

        visited.add(current_id)

        # Find all edges where current_id is the source
        # (outgoing edges = dependencies of this entity)
        for edge in find_edges(source_id=current_id, edge_type="DEPENDS_ON"):
            dependency = get_entity(edge.target_id)
            if dependency:
                dependencies.append(dependency)
                queue.append((edge.target_id, depth + 1))

    return dependencies
```

#### Bidirectional Traversal (All connected entities)
```python
def get_related_entities(entity_id: str, max_depth: int = 2) -> Dict[str, List[Entity]]:
    """
    Get all entities connected via any relationship.
    Answers: "What's in this task's influence sphere?"

    Returns: {
        'dependencies': [...],
        'dependents': [...],
        'blockers': [...],
        'blocked_by': [...]
    }
    """
    result = {
        'dependencies': get_all_dependencies(entity_id, max_depth),
        'dependents': get_all_dependents(entity_id, max_depth),
        'blockers': get_entities_with_edge_to(entity_id, "BLOCKS"),
        'blocked_by': get_entities_with_edge_from(entity_id, "BLOCKS")
    }
    return result
```

### 1.2 Shortest Path Queries

For "What's the shortest path from A to B?":

```python
def find_shortest_path(
    source_id: str,
    target_id: str,
    edge_types: Optional[List[str]] = None
) -> Optional[List[str]]:
    """
    Find shortest path between two entities using BFS.
    Answers: "How do these entities relate?"
    """
    from collections import deque

    if source_id == target_id:
        return [source_id]

    visited = {source_id}
    queue = deque([(source_id, [source_id])])

    while queue:
        current_id, path = queue.popleft()

        # Get all outgoing edges
        for edge in find_edges(source_id=current_id):
            if edge_types and edge.edge_type not in edge_types:
                continue
            if edge.target_id in visited:
                continue

            if edge.target_id == target_id:
                return path + [target_id]

            visited.add(edge.target_id)
            queue.append((edge.target_id, path + [edge.target_id]))

    return None  # No path exists
```

### 1.3 Strongly Connected Components (Cycle Detection)

Detect circular dependencies:

```python
def find_dependency_cycles() -> List[List[str]]:
    """
    Find all circular dependencies in the graph.
    Answers: "What tasks are blocking each other?"
    """
    from collections import defaultdict

    visited = set()
    rec_stack = set()
    cycles = []

    def dfs(node_id: str, path: List[str]) -> None:
        visited.add(node_id)
        rec_stack.add(node_id)
        path.append(node_id)

        for edge in find_edges(source_id=node_id, edge_type="DEPENDS_ON"):
            if edge.target_id not in visited:
                dfs(edge.target_id, path.copy())
            elif edge.target_id in rec_stack:
                # Found a cycle
                cycle_start = path.index(edge.target_id)
                cycle = path[cycle_start:] + [edge.target_id]
                cycles.append(cycle)

        rec_stack.discard(node_id)

    for entity_id in get_all_entity_ids():
        if entity_id not in visited:
            dfs(entity_id, [])

    return cycles
```

### 1.4 Critical Path Analysis

For project scheduling: "What's the longest dependency chain?"

```python
def find_critical_path(start_entity_id: str = None) -> List[str]:
    """
    Find the longest path in the dependency DAG.
    Answers: "What's the minimum time to completion?"
    """
    # Memoization for dynamic programming
    memo = {}

    def longest_path_from(node_id: str) -> Tuple[int, List[str]]:
        if node_id in memo:
            return memo[node_id]

        # Base case: leaf node
        deps = find_edges(source_id=node_id, edge_type="DEPENDS_ON")
        if not deps:
            return (1, [node_id])

        # Recursive case: longest path through any dependency
        max_length = 0
        best_path = []

        for edge in deps:
            child_length, child_path = longest_path_from(edge.target_id)
            if child_length + 1 > max_length:
                max_length = child_length + 1
                best_path = [node_id] + child_path

        memo[node_id] = (max_length, best_path)
        return (max_length, best_path)

    if start_entity_id:
        _, path = longest_path_from(start_entity_id)
        return path

    # Find global critical path
    longest = []
    for entity_id in get_all_entity_ids():
        _, path = longest_path_from(entity_id)
        if len(path) > len(longest):
            longest = path

    return longest
```

### 1.5 Influence Sphere (k-hop neighborhood)

All entities reachable within k hops:

```python
def get_k_hop_neighborhood(
    entity_id: str,
    k: int,
    direction: str = "both"  # "forward", "backward", "both"
) -> Dict[int, List[str]]:
    """
    Get all entities at each distance up to k hops.

    Returns:
        {0: [entity_id], 1: [neighbors], 2: [second_order], ...}
    """
    neighborhood = defaultdict(list)
    visited = set([entity_id])
    current_level = [entity_id]

    neighborhood[0] = [entity_id]

    for hop in range(1, k + 1):
        next_level = []

        for current_id in current_level:
            # Get next level neighbors
            neighbors = set()

            if direction in ["forward", "both"]:
                for edge in find_edges(source_id=current_id):
                    neighbors.add(edge.target_id)

            if direction in ["backward", "both"]:
                for edge in find_edges(target_id=current_id):
                    neighbors.add(edge.source_id)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    next_level.append(neighbor_id)

        if not next_level:
            break

        neighborhood[hop] = next_level
        current_level = next_level

    return neighborhood
```

---

## 2. Full-Text Search

Searching entity content (titles, descriptions, rationale) with ranking and relevance.

### 2.1 Basic Full-Text Search

```python
def search_entities(
    query: str,
    search_fields: List[str] = None,
    entity_types: List[str] = None,
    limit: int = 10
) -> List[Tuple[str, float]]:
    """
    Full-text search across entity content.

    Args:
        query: Search terms
        search_fields: Fields to search (title, description, rationale, content)
        entity_types: Filter by entity type (task, decision, etc.)
        limit: Maximum results

    Returns:
        List of (entity_id, relevance_score) tuples
    """
    from .tokenizer import Tokenizer

    tokenizer = Tokenizer()
    query_terms = set(tokenizer.tokenize(query))

    if not search_fields:
        search_fields = ['title', 'description', 'rationale', 'content']

    results = []

    for entity in get_all_entities():
        if entity_types and entity.entity_type not in entity_types:
            continue

        # Score entity against query
        score = 0.0
        match_count = 0

        for field in search_fields:
            if not hasattr(entity, field):
                continue

            field_value = getattr(entity, field, "")
            if not field_value:
                continue

            field_tokens = set(tokenizer.tokenize(str(field_value)))

            # Count matching terms
            matches = query_terms & field_tokens
            if matches:
                # Field boost for titles (higher weight)
                boost = 1.5 if field == 'title' else 1.0
                score += len(matches) * boost
                match_count += len(matches)

        if score > 0:
            # Normalize by query length to penalize partial matches
            normalized_score = score / len(query_terms)
            results.append((entity.id, normalized_score))

    # Sort by relevance and limit
    results.sort(key=lambda x: -x[1])
    return results[:limit]
```

### 2.2 TF-IDF Ranking

For better relevance ranking:

```python
from collections import defaultdict
import math

class TfIdfSearcher:
    """
    Full-text search using TF-IDF ranking.
    """

    def __init__(self):
        self.idf = {}
        self._build_idf()

    def _build_idf(self):
        """Compute IDF for all terms in corpus."""
        from .tokenizer import Tokenizer

        tokenizer = Tokenizer()
        doc_count = 0
        term_doc_count = defaultdict(int)

        # Count documents per term
        for entity in get_all_entities():
            doc_count += 1
            terms_in_doc = set()

            for field in ['title', 'description', 'rationale', 'content']:
                if hasattr(entity, field):
                    value = getattr(entity, field, "")
                    if value:
                        tokens = tokenizer.tokenize(str(value))
                        terms_in_doc.update(tokens)

            # Count once per document
            for term in terms_in_doc:
                term_doc_count[term] += 1

        # Compute IDF
        for term, count in term_doc_count.items():
            self.idf[term] = math.log(doc_count / count) if count > 0 else 0

    def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Search using TF-IDF scoring.
        """
        from .tokenizer import Tokenizer

        tokenizer = Tokenizer()
        query_terms = tokenizer.tokenize(query)
        query_term_counts = defaultdict(int)

        for term in query_terms:
            query_term_counts[term] += 1

        results = []

        for entity in get_all_entities():
            # Tokenize entity fields
            entity_text = ""
            for field in ['title', 'description', 'rationale', 'content']:
                if hasattr(entity, field):
                    value = getattr(entity, field, "")
                    if value:
                        entity_text += " " + str(value)

            entity_tokens = tokenizer.tokenize(entity_text)
            entity_term_counts = defaultdict(int)
            for term in entity_tokens:
                entity_term_counts[term] += 1

            # Compute TF-IDF score
            score = 0.0
            for query_term, query_freq in query_term_counts.items():
                if query_term in entity_term_counts:
                    # TF component
                    tf = entity_term_counts[query_term] / len(entity_tokens) if entity_tokens else 0

                    # IDF component
                    idf = self.idf.get(query_term, 0)

                    # TF-IDF
                    score += tf * idf

            if score >= min_score:
                results.append((entity.id, score))

        results.sort(key=lambda x: -x[1])
        return results[:limit]
```

### 2.3 Phrase Matching

Boost results that contain exact phrases:

```python
def search_with_phrases(
    query: str,
    limit: int = 10
) -> List[Tuple[str, float]]:
    """
    Search with boosted scoring for exact phrase matches.
    """
    # Split query into terms and phrases
    import re
    phrases = re.findall(r'"([^"]+)"', query)
    terms_only = re.sub(r'"[^"]*"', '', query).split()

    results = search_entities(query, limit=limit * 2)  # Get more results

    # Re-score with phrase boost
    rescored = []
    for entity_id, base_score in results:
        entity = get_entity(entity_id)
        text = " ".join([
            getattr(entity, field, "")
            for field in ['title', 'description', 'rationale', 'content']
            if hasattr(entity, field)
        ]).lower()

        phrase_score = 0
        for phrase in phrases:
            if phrase.lower() in text:
                # Significant boost for exact phrases
                phrase_score += 2.0

        final_score = base_score + phrase_score
        rescored.append((entity_id, final_score))

    rescored.sort(key=lambda x: -x[1])
    return rescored[:limit]
```

---

## 3. Faceted Search

Filtering by entity properties, status, priority, type, etc.

### 3.1 Basic Faceted Search

```python
class FacetedSearcher:
    """
    Faceted search combining graph and text queries.
    """

    def __init__(self):
        self.facets = {
            'status': set(),
            'priority': set(),
            'entity_type': set(),
            'sprint_id': set(),
            'assigned_to': set(),
        }
        self._scan_facets()

    def _scan_facets(self):
        """Index all unique facet values."""
        for entity in get_all_entities():
            if hasattr(entity, 'status'):
                self.facets['status'].add(entity.status)
            if hasattr(entity, 'priority'):
                self.facets['priority'].add(entity.priority)
            self.facets['entity_type'].add(entity.entity_type)
            if hasattr(entity, 'sprint_id'):
                self.facets['sprint_id'].add(entity.sprint_id)
            if hasattr(entity, 'assigned_to'):
                self.facets['assigned_to'].add(entity.assigned_to)

    def get_facet_values(self, facet_name: str) -> List[Tuple[str, int]]:
        """
        Get all values for a facet with document counts.
        """
        counts = defaultdict(int)

        for entity in get_all_entities():
            value = getattr(entity, facet_name, None)
            if value:
                counts[value] += 1

        # Sort by count descending
        return sorted(counts.items(), key=lambda x: -x[1])

    def search_with_facets(
        self,
        query: str = None,
        filters: Dict[str, List[str]] = None,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Combined text search + facet filtering.

        Args:
            query: Optional text query
            filters: {facet_name: [value1, value2, ...], ...}
            limit: Max results

        Returns:
            Ranked list of matching entities
        """
        # Start with all entities
        candidates = set(e.id for e in get_all_entities())

        # Apply facet filters (AND logic: must match all filters)
        if filters:
            for facet_name, facet_values in filters.items():
                matching = set()
                for entity in get_all_entities():
                    entity_value = getattr(entity, facet_name, None)
                    if entity_value in facet_values:
                        matching.add(entity.id)
                candidates &= matching

        # Apply text search
        if query:
            results = search_entities(query, limit=len(candidates))
            candidate_results = [
                (eid, score) for eid, score in results if eid in candidates
            ]
        else:
            candidate_results = [(eid, 1.0) for eid in candidates]

        # Sort and limit
        candidate_results.sort(key=lambda x: -x[1])
        return candidate_results[:limit]
```

### 3.2 Hierarchical Faceting (Status + Priority)

```python
def search_by_status_and_priority(
    statuses: List[str] = None,
    priorities: List[str] = None,
    query: str = None
) -> List[Dict[str, Any]]:
    """
    Multi-level filtering: status + priority + optional text search.
    """
    results = []

    for entity in get_all_entities():
        if not hasattr(entity, 'status') or not hasattr(entity, 'priority'):
            continue

        # Apply filters
        if statuses and entity.status not in statuses:
            continue
        if priorities and entity.priority not in priorities:
            continue

        # Apply text search if provided
        if query:
            if not _matches_query(entity, query):
                continue

        results.append({
            'id': entity.id,
            'title': entity.title,
            'status': entity.status,
            'priority': entity.priority,
            'type': entity.entity_type
        })

    return results
```

---

## 4. Ranking and Relevance

Combining multiple signals to rank results.

### 4.1 Multi-Signal Ranking

```python
def rank_entities(
    candidates: List[str],
    query: str = None,
    tfidf_weight: float = 0.4,
    pagerank_weight: float = 0.3,
    freshness_weight: float = 0.2,
    priority_weight: float = 0.1
) -> List[Tuple[str, float]]:
    """
    Rank entities combining multiple signals.

    Signals:
    - Text relevance (TF-IDF)
    - Graph importance (PageRank/dependency count)
    - Freshness (recent modifications)
    - Priority (explicit priority field)
    """
    from datetime import datetime, timedelta

    scores = {}

    # Normalize each signal to [0, 1]
    components = {
        'text': {},
        'graph': {},
        'freshness': {},
        'priority': {}
    }

    # 1. Text relevance
    if query:
        text_results = search_entities(query, limit=len(candidates))
        text_scores = {eid: score for eid, score in text_results}
        max_text_score = max(text_scores.values()) if text_scores else 1

        for eid in candidates:
            components['text'][eid] = (text_scores.get(eid, 0) / max_text_score) if max_text_score else 0
    else:
        for eid in candidates:
            components['text'][eid] = 0.5  # Neutral if no query

    # 2. Graph importance
    graph_scores = {}
    for eid in candidates:
        # Count dependencies
        dependents = len(get_all_dependents(eid))
        blockers = len(get_blockers(eid))
        # Entities with many dependents are important
        graph_scores[eid] = min((dependents + blockers) / 10, 1.0)

    max_graph = max(graph_scores.values()) if graph_scores else 1
    for eid in candidates:
        components['graph'][eid] = (graph_scores[eid] / max_graph) if max_graph else 0

    # 3. Freshness (recently modified = higher score)
    now = datetime.now()
    for eid in candidates:
        entity = get_entity(eid)
        if hasattr(entity, 'modified_at'):
            try:
                mod_time = datetime.fromisoformat(entity.modified_at)
                days_old = (now - mod_time).days
                # Decay with age: recent = 1.0, 30 days old = 0.5
                components['freshness'][eid] = max(0, 1 - (days_old / 60))
            except:
                components['freshness'][eid] = 0.5
        else:
            components['freshness'][eid] = 0.5

    # 4. Priority boost
    for eid in candidates:
        entity = get_entity(eid)
        priority = getattr(entity, 'priority', 'medium')
        priority_boost = {
            'critical': 1.0,
            'high': 0.75,
            'medium': 0.5,
            'low': 0.25
        }
        components['priority'][eid] = priority_boost.get(priority, 0.5)

    # Combine signals
    for eid in candidates:
        weighted_score = (
            tfidf_weight * components['text'].get(eid, 0) +
            pagerank_weight * components['graph'].get(eid, 0) +
            freshness_weight * components['freshness'].get(eid, 0) +
            priority_weight * components['priority'].get(eid, 0)
        )
        scores[eid] = weighted_score

    # Return sorted results
    result = sorted(scores.items(), key=lambda x: -x[1])
    return result
```

### 4.2 Personalized Ranking

Rank differently based on user context:

```python
def rank_entities_personalized(
    candidates: List[str],
    user_context: Dict[str, Any],
    query: str = None
) -> List[Tuple[str, float]]:
    """
    Rank entities considering user's recent activity and preferences.

    Context includes:
    - recently_viewed: Set of recently viewed entity IDs
    - preferred_statuses: List of preferred statuses
    - sprint_context: Current sprint ID
    """
    scores = {}

    for eid in candidates:
        entity = get_entity(eid)
        score = 0.0

        # Boost entities in user's sprint
        if user_context.get('sprint_id'):
            if hasattr(entity, 'sprint_id') and entity.sprint_id == user_context['sprint_id']:
                score += 0.3

        # Boost entities matching preferred statuses
        if user_context.get('preferred_statuses'):
            if hasattr(entity, 'status') and entity.status in user_context['preferred_statuses']:
                score += 0.2

        # Boost recently viewed entities
        if eid in user_context.get('recently_viewed', set()):
            score += 0.1

        # Penalize entities the user recently ignored
        if eid in user_context.get('ignored_entities', set()):
            score -= 0.2

        scores[eid] = score

    # Apply base ranking on top
    base_ranks = rank_entities(candidates, query)
    base_scores = {eid: idx for idx, (eid, _) in enumerate(base_ranks)}

    # Combine: personalized boost + base ranking
    for eid in scores:
        base_rank = base_scores.get(eid, len(candidates))
        # Personalization is a percentage boost, not absolute
        boost_factor = 1 + scores[eid]  # 0.8 to 1.3
        scores[eid] = (len(candidates) - base_rank) * boost_factor

    result = sorted(scores.items(), key=lambda x: -x[1])
    return result
```

---

## 5. Query Planning and Optimization

Choosing the best execution path for complex queries.

### 5.1 Query Optimizer

```python
from enum import Enum
from dataclasses import dataclass

class QueryStrategy(Enum):
    """Different execution strategies for queries."""
    INDEX_SCAN = "index_scan"      # Scan all entities
    FACET_FILTER = "facet_filter"  # Filter by facets first
    TEXT_SEARCH = "text_search"    # Text search with filters
    GRAPH_TRAVERSE = "graph_traverse"  # Start from entity, traverse graph
    HYBRID = "hybrid"               # Combine multiple strategies

@dataclass
class QueryPlan:
    """Execution plan for a query."""
    strategy: QueryStrategy
    estimated_cost: float
    estimated_results: int
    steps: List[str]

class QueryOptimizer:
    """
    Analyzes queries and determines optimal execution plan.
    """

    def __init__(self, stats: Dict[str, Any]):
        """
        Args:
            stats: Database statistics
                - total_entities
                - avg_entity_fields
                - facet_cardinalities
                - index_stats
        """
        self.stats = stats

    def optimize(self, query_spec: Dict[str, Any]) -> QueryPlan:
        """
        Determine optimal execution strategy.

        Query spec contains:
        - query_text: Optional full-text search
        - filters: {facet: [values]} facet filters
        - start_entity: Optional starting point for graph traversal
        - traverse_depth: For graph traversal
        """
        strategies = []

        # Strategy 1: Direct index scan (always possible)
        strategies.append(self._plan_index_scan(query_spec))

        # Strategy 2: Facet filter first (if filters provided)
        if query_spec.get('filters'):
            strategies.append(self._plan_facet_filter(query_spec))

        # Strategy 3: Text search with filters
        if query_spec.get('query_text'):
            strategies.append(self._plan_text_search(query_spec))

        # Strategy 4: Graph traversal (if starting entity provided)
        if query_spec.get('start_entity'):
            strategies.append(self._plan_graph_traverse(query_spec))

        # Choose strategy with lowest cost
        best = min(strategies, key=lambda p: p.estimated_cost)
        return best

    def _plan_index_scan(self, query_spec: Dict[str, Any]) -> QueryPlan:
        """Full table scan strategy."""
        total = self.stats['total_entities']
        cost = total  # Cost is O(n)
        return QueryPlan(
            strategy=QueryStrategy.INDEX_SCAN,
            estimated_cost=cost,
            estimated_results=total,
            steps=[
                "1. Scan all entities",
                "2. Filter by facets (if any)",
                "3. Rank by relevance"
            ]
        )

    def _plan_facet_filter(self, query_spec: Dict[str, Any]) -> QueryPlan:
        """Filter by facets first."""
        filters = query_spec.get('filters', {})

        # Estimate filtered set size
        total = self.stats['total_entities']
        estimated_size = total

        for facet, values in filters.items():
            # Use selectivity of each filter
            cardinality = self.stats.get(f'cardinality_{facet}', 10)
            selectivity = len(values) / cardinality
            estimated_size *= selectivity

        # Cost is O(filters * facet_values)
        filter_cost = sum(len(v) for v in filters.values())

        return QueryPlan(
            strategy=QueryStrategy.FACET_FILTER,
            estimated_cost=filter_cost + estimated_size,
            estimated_results=int(estimated_size),
            steps=[
                f"1. Apply {len(filters)} facet filters",
                f"2. Estimated result set: {int(estimated_size)} entities",
                "3. Rank results"
            ]
        )

    def _plan_text_search(self, query_spec: Dict[str, Any]) -> QueryPlan:
        """Text search strategy."""
        # Text search is typically O(n * fields) but with index is O(log n)
        # Assuming indexed search
        cost = 100 + (self.stats['total_entities'] * 0.01)  # Rough estimate

        return QueryPlan(
            strategy=QueryStrategy.TEXT_SEARCH,
            estimated_cost=cost,
            estimated_results=10,  # Text search typically returns ~10 results
            steps=[
                "1. Tokenize query",
                "2. Lookup in TF-IDF index",
                "3. Apply facet filters (if any)",
                "4. Rank by relevance"
            ]
        )

    def _plan_graph_traverse(self, query_spec: Dict[str, Any]) -> QueryPlan:
        """Graph traversal strategy."""
        depth = query_spec.get('traverse_depth', 1)

        # Cost is O(branching_factor ^ depth)
        # Rough estimate: assume avg 3 connections per node
        branching_factor = 3
        cost = branching_factor ** depth

        return QueryPlan(
            strategy=QueryStrategy.GRAPH_TRAVERSE,
            estimated_cost=cost,
            estimated_results=branching_factor ** depth,
            steps=[
                f"1. Lookup start entity",
                f"2. Traverse edges (depth={depth})",
                f"3. Estimated nodes: {branching_factor ** depth}",
                "4. Filter and rank results"
            ]
        )
```

### 5.2 Cost-Based Query Execution

```python
def execute_optimized_query(query_spec: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    Execute query using optimizer-chosen strategy.
    """
    optimizer = QueryOptimizer(get_db_stats())
    plan = optimizer.optimize(query_spec)

    print(f"Executing with strategy: {plan.strategy.value}")
    print(f"Estimated cost: {plan.estimated_cost}")
    print(f"Expected results: {plan.estimated_results}")

    if plan.strategy == QueryStrategy.FACET_FILTER:
        return search_with_facets(
            query=query_spec.get('query_text'),
            filters=query_spec.get('filters')
        )

    elif plan.strategy == QueryStrategy.TEXT_SEARCH:
        candidates = search_entities(query_spec.get('query_text'))
        if query_spec.get('filters'):
            # Apply facet filters
            candidates = [
                (eid, score) for eid, score in candidates
                if _matches_filters(get_entity(eid), query_spec.get('filters'))
            ]
        return candidates

    elif plan.strategy == QueryStrategy.GRAPH_TRAVERSE:
        start_id = query_spec.get('start_entity')
        depth = query_spec.get('traverse_depth', 1)
        neighbors = get_k_hop_neighborhood(start_id, depth)
        entity_ids = [eid for level_list in neighbors.values() for eid in level_list]
        return [(eid, 1.0) for eid in entity_ids]

    else:  # INDEX_SCAN or HYBRID
        return search_with_facets(
            query=query_spec.get('query_text'),
            filters=query_spec.get('filters')
        )
```

---

## 6. Caching Query Results

Strategies for caching when safe, invalidating when needed.

### 6.1 Query Result Cache

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib

@dataclass
class CachedQuery:
    """Cached query result with metadata."""
    key: str                           # Cache key
    results: List[Tuple[str, float]]  # Query results
    query_spec: Dict[str, Any]        # Original query
    created_at: datetime
    expires_at: datetime
    depends_on: Set[str]              # Entity IDs this depends on
    hit_count: int = 0
    valid: bool = True

class QueryResultCache:
    """
    LRU cache for query results with dependency tracking.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}  # key -> CachedQuery
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self.entity_to_queries = {}  # entity_id -> set of cache keys

    def _make_key(self, query_spec: Dict[str, Any]) -> str:
        """Generate cache key from query spec."""
        # Convert query spec to stable string representation
        import json

        # Remove non-deterministic fields
        spec_copy = query_spec.copy()
        spec_copy.pop('limit', None)  # May vary

        # Sort for determinism
        key_str = json.dumps(spec_copy, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query_spec: Dict[str, Any]) -> Optional[List[Tuple[str, float]]]:
        """
        Get cached query result if valid and not expired.
        """
        key = self._make_key(query_spec)

        if key not in self.cache:
            return None

        cached = self.cache[key]

        # Check expiration
        if datetime.now() > cached.expires_at:
            del self.cache[key]
            return None

        # Check validity (all dependent entities unchanged)
        if not self._check_dependencies(cached):
            cached.valid = False
            del self.cache[key]
            return None

        cached.hit_count += 1
        return cached.results

    def set(
        self,
        query_spec: Dict[str, Any],
        results: List[Tuple[str, float]],
        depends_on: Set[str] = None
    ) -> None:
        """
        Cache query results with dependency tracking.
        """
        key = self._make_key(query_spec)

        # Evict LRU entry if at capacity
        if len(self.cache) >= self.max_size:
            lru_key = min(
                self.cache.items(),
                key=lambda x: x[1].hit_count
            )[0]
            self._evict(lru_key)

        # Determine dependencies
        if depends_on is None:
            depends_on = self._infer_dependencies(query_spec)

        cached = CachedQuery(
            key=key,
            results=results,
            query_spec=query_spec,
            created_at=datetime.now(),
            expires_at=datetime.now() + self.ttl,
            depends_on=depends_on
        )

        self.cache[key] = cached

        # Track inverse mapping for invalidation
        for entity_id in depends_on:
            if entity_id not in self.entity_to_queries:
                self.entity_to_queries[entity_id] = set()
            self.entity_to_queries[entity_id].add(key)

    def _infer_dependencies(self, query_spec: Dict[str, Any]) -> Set[str]:
        """
        Infer which entities a query depends on.
        Conservative approach: include all filter targets and search results.
        """
        depends_on = set()

        # Depends on entities in filters
        for facet, values in query_spec.get('filters', {}).items():
            # Some facets may reference entity IDs
            if facet in ['sprint_id', 'epic_id']:
                depends_on.update(values)

        # If there's a starting entity for graph traversal, depend on it
        if query_spec.get('start_entity'):
            depends_on.add(query_spec['start_entity'])

        return depends_on

    def _check_dependencies(self, cached: CachedQuery) -> bool:
        """
        Check if dependent entities have been modified.
        """
        for entity_id in cached.depends_on:
            entity = get_entity(entity_id)
            if not entity:
                return False  # Entity deleted

            # Check if entity was modified after cache creation
            if hasattr(entity, 'modified_at'):
                try:
                    mod_time = datetime.fromisoformat(entity.modified_at)
                    if mod_time > cached.created_at:
                        return False  # Entity modified
                except:
                    pass

        return True

    def invalidate_entity(self, entity_id: str) -> int:
        """
        Invalidate all queries that depend on an entity.
        Returns count of invalidated queries.
        """
        if entity_id not in self.entity_to_queries:
            return 0

        invalidated = 0
        cache_keys = list(self.entity_to_queries[entity_id])

        for key in cache_keys:
            if key in self.cache:
                del self.cache[key]
                invalidated += 1

        del self.entity_to_queries[entity_id]
        return invalidated

    def invalidate_all_related(self, entity_id: str) -> int:
        """
        Invalidate queries depending on this entity and all its edges.
        """
        invalidated = self.invalidate_entity(entity_id)

        # Also invalidate queries depending on connected entities
        # (because graph structure changed)
        for edge in find_edges(source_id=entity_id) + find_edges(target_id=entity_id):
            # Check other endpoint
            other_id = edge.target_id if edge.source_id == entity_id else edge.source_id
            invalidated += self.invalidate_entity(other_id)

        return invalidated

    def _evict(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            cached = self.cache[key]

            # Clean up inverse mapping
            for entity_id in cached.depends_on:
                if entity_id in self.entity_to_queries:
                    self.entity_to_queries[entity_id].discard(key)

            del self.cache[key]

    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        total_hits = sum(c.hit_count for c in self.cache.values())

        return {
            'cached_queries': len(self.cache),
            'total_capacity': self.max_size,
            'total_hits': total_hits,
            'avg_hits_per_query': total_hits / len(self.cache) if self.cache else 0,
            'memory_estimate': sum(
                len(str(c.results)) for c in self.cache.values()
            )
        }
```

### 6.2 Smart Invalidation Strategy

```python
class CacheInvalidationStrategy:
    """
    Strategies for determining when to invalidate cached queries.
    """

    @staticmethod
    def on_entity_update(
        entity: Entity,
        cache: QueryResultCache,
        invalidation_level: str = "conservative"  # or "aggressive"
    ) -> int:
        """
        Update cache after entity modification.

        Levels:
        - conservative: Only invalidate direct dependencies
        - aggressive: Invalidate all related entities
        """

        if invalidation_level == "aggressive":
            # Invalidate all queries that touch this entity or its neighbors
            return cache.invalidate_all_related(entity.id)

        else:  # conservative
            # Only invalidate queries explicitly depending on this entity
            return cache.invalidate_entity(entity.id)

    @staticmethod
    def on_edge_update(
        source_id: str,
        target_id: str,
        edge_type: str,
        cache: QueryResultCache
    ) -> int:
        """
        Update cache after edge addition/removal.

        Edges affecting:
        - DEPENDS_ON: invalidate dependency queries for both ends
        - BLOCKS: invalidate blocking queries
        - CONTAINS: invalidate membership queries
        """

        invalidated = 0

        # Invalidate queries about this edge type
        invalidated += cache.invalidate_entity(source_id)
        invalidated += cache.invalidate_entity(target_id)

        # For graph traversal queries, may also need to invalidate
        # broader neighborhood
        if edge_type in ["DEPENDS_ON", "BLOCKS"]:
            # These affect transitive relationships
            for entity_id in [source_id, target_id]:
                neighbors = get_k_hop_neighborhood(entity_id, k=2)
                for level_ids in neighbors.values():
                    for nid in level_ids:
                        invalidated += cache.invalidate_entity(nid)

        return invalidated

    @staticmethod
    def should_cache(
        query_spec: Dict[str, Any],
        query_cost: float,
        frequency: int = 1
    ) -> bool:
        """
        Determine if a query result should be cached.

        Cache if:
        - Query is expensive (cost > threshold)
        - Query is frequently executed
        - Result stability is likely
        """

        expensive = query_cost > 100
        frequent = frequency > 5

        # Avoid caching queries with very recent data
        if query_spec.get('filters', {}).get('modified_within_hours'):
            return False

        return expensive or frequent
```

---

## Practical Integration Example

Combining all patterns into a unified query API:

```python
class GraphQueryEngine:
    """
    Unified query engine combining graph traversal, text search, and caching.
    """

    def __init__(self):
        self.cache = QueryResultCache(max_size=1000, ttl_seconds=3600)
        self.optimizer = QueryOptimizer(get_db_stats())
        self.text_searcher = TfIdfSearcher()

    def query(
        self,
        query_type: str,  # "graph", "text", "faceted", "combined"
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Execute query with automatic caching and optimization.
        """

        # Build normalized query spec for caching
        query_spec = {
            'query_type': query_type,
            **kwargs
        }

        # Check cache
        cached = self.cache.get(query_spec)
        if cached is not None:
            return cached

        # Execute query
        if query_type == "graph_traverse":
            results = self._execute_graph_query(**kwargs)

        elif query_type == "text_search":
            results = self._execute_text_query(**kwargs)

        elif query_type == "faceted":
            results = self._execute_faceted_query(**kwargs)

        elif query_type == "combined":
            results = self._execute_combined_query(**kwargs)

        else:
            raise ValueError(f"Unknown query type: {query_type}")

        # Cache results
        self.cache.set(query_spec, results)

        return results

    def _execute_graph_query(
        self,
        entity_id: str,
        depth: int = 2,
        direction: str = "both",
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Execute graph traversal query."""
        neighborhood = get_k_hop_neighborhood(entity_id, depth, direction)
        results = [(eid, 1.0 / (hop + 1)) for hop, ids in neighborhood.items() for eid in ids]
        return sorted(results, key=lambda x: -x[1])

    def _execute_text_query(
        self,
        text: str,
        limit: int = 10,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Execute text search query."""
        return self.text_searcher.search(text, limit=limit)

    def _execute_faceted_query(
        self,
        query: str = None,
        filters: Dict[str, List[str]] = None,
        limit: int = 10,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Execute faceted search query."""
        searcher = FacetedSearcher()
        return searcher.search_with_facets(query, filters, limit)

    def _execute_combined_query(
        self,
        query_text: str = None,
        filters: Dict[str, List[str]] = None,
        start_entity: str = None,
        traverse_depth: int = 1,
        limit: int = 10,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """Execute combined graph + text query."""

        # Start with text or graph results
        if query_text:
            candidates = self.text_searcher.search(query_text, limit=limit * 2)
            candidate_ids = [eid for eid, _ in candidates]
        elif start_entity:
            neighborhood = get_k_hop_neighborhood(start_entity, traverse_depth)
            candidate_ids = [eid for ids in neighborhood.values() for eid in ids]
        else:
            candidate_ids = [e.id for e in get_all_entities()]

        # Apply facet filters
        if filters:
            searcher = FacetedSearcher()
            filtered = searcher.search_with_facets(query_text, filters, limit * 2)
            candidate_ids = [eid for eid, _ in filtered]

        # Rank combined results
        results = rank_entities(candidate_ids, query_text, limit=limit)
        return results
```

---

## Summary: Key Patterns

| Pattern | Use Case | Complexity | Cache Safety |
|---------|----------|-----------|--------------|
| **BFS traversal** | "What depends on X?" | O(V+E) | Medium (invalidate on edge changes) |
| **Shortest path** | "How are A and B related?" | O(V+E) | Medium (invalidate on edge changes) |
| **TF-IDF search** | Text-based finding | O(n*m) with index | High (invalidate on content changes) |
| **Faceted search** | Multi-filter queries | O(facet_values * results) | High (invalidate on field changes) |
| **Multi-signal ranking** | Relevance combining | O(signals * results) | High (complex invalidation) |
| **Query optimizer** | Cost planning | O(strategies) | N/A (metadata only) |
| **LRU caching** | Result memorization | O(1) lookup | Depends on invalidation |

**Best Practices:**

1. **Combine strategies** - Graph for structure, text for content
2. **Cache aggressively** - With smart invalidation on entity changes
3. **Plan queries** - Use cost estimates for strategy selection
4. **Rank multi-signal** - Combine text relevance, graph importance, freshness
5. **Invalidate carefully** - Conservative by default, aggressive when needed
6. **Monitor cache** - Track hit rates and adjust TTL accordingly
