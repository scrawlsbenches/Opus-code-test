"""
ThoughtGraph: Main class for Graph of Thought operations.

This module implements the core thought graph data structure with operations for:
- Construction (adding/removing nodes and edges)
- Traversal (DFS, BFS, shortest paths)
- Analysis (cycles, orphans, hubs, bridges)
- Transformation (pruning, clustering)
- Queries (lookup by ID, type, relationships)
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from .graph_of_thought import (
    EdgeType,
    NodeType,
    ThoughtCluster,
    ThoughtEdge,
    ThoughtNode,
)


class ThoughtGraph:
    """
    A graph data structure for representing and manipulating networks of thoughts.

    The ThoughtGraph manages nodes (concepts, questions, decisions, etc.) and
    edges (relationships) between them, supporting graph algorithms and
    transformations for reasoning workflows.
    """

    def __init__(self):
        """Initialize an empty thought graph."""
        self.nodes: Dict[str, ThoughtNode] = {}
        self.edges: List[ThoughtEdge] = []
        self.clusters: Dict[str, ThoughtCluster] = {}

        # Index for fast edge lookups
        self._edges_from: Dict[str, List[ThoughtEdge]] = defaultdict(list)
        self._edges_to: Dict[str, List[ThoughtEdge]] = defaultdict(list)

    # =========================================================================
    # CONSTRUCTION OPERATIONS
    # =========================================================================

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        content: str,
        properties: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> ThoughtNode:
        """
        Add a node to the graph.

        Args:
            node_id: Unique identifier for the node
            node_type: Type of thought node
            content: Main content/description
            properties: Optional type-specific properties
            metadata: Optional additional metadata

        Returns:
            The created ThoughtNode

        Raises:
            ValueError: If node_id already exists
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")

        node = ThoughtNode(
            id=node_id,
            node_type=node_type,
            content=content,
            properties=properties or {},
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        confidence: float = 1.0,
        bidirectional: bool = False,
    ) -> ThoughtEdge:
        """
        Add an edge between two nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            edge_type: Type of relationship
            weight: Strength of relationship (0.0-1.0)
            confidence: Confidence in relationship (0.0-1.0)
            bidirectional: Whether relationship goes both ways

        Returns:
            The created ThoughtEdge

        Raises:
            ValueError: If either node doesn't exist
        """
        if from_id not in self.nodes:
            raise ValueError(f"Source node {from_id} not found")
        if to_id not in self.nodes:
            raise ValueError(f"Target node {to_id} not found")

        edge = ThoughtEdge(
            source_id=from_id,
            target_id=to_id,
            edge_type=edge_type,
            weight=weight,
            confidence=confidence,
            bidirectional=bidirectional,
        )

        self.edges.append(edge)
        self._edges_from[from_id].append(edge)
        self._edges_to[to_id].append(edge)

        # Add reverse edge if bidirectional
        if bidirectional:
            reverse_edge = ThoughtEdge(
                source_id=to_id,
                target_id=from_id,
                edge_type=edge_type,
                weight=weight,
                confidence=confidence,
                bidirectional=False,  # Avoid infinite recursion
            )
            self.edges.append(reverse_edge)
            self._edges_from[to_id].append(reverse_edge)
            self._edges_to[from_id].append(reverse_edge)

        return edge

    def remove_node(self, node_id: str) -> ThoughtNode:
        """
        Remove a node and all its edges from the graph.

        Args:
            node_id: ID of node to remove

        Returns:
            The removed ThoughtNode

        Raises:
            ValueError: If node doesn't exist
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        # Remove all edges connected to this node
        edges_to_remove = []
        for edge in self.edges:
            if edge.source_id == node_id or edge.target_id == node_id:
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            self.remove_edge(edge.source_id, edge.target_id, edge.edge_type)

        # Remove from clusters (only if node is in cluster)
        for cluster in self.clusters.values():
            if cluster.contains_node(node_id):
                cluster.node_ids.discard(node_id)

        # Remove node
        return self.nodes.pop(node_id)

    def remove_edge(self, from_id: str, to_id: str, edge_type: EdgeType) -> bool:
        """
        Remove an edge from the graph.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            edge_type: Type of relationship to remove

        Returns:
            True if edge was removed, False if not found
        """
        # Find and remove matching edge
        for i, edge in enumerate(self.edges):
            if (
                edge.source_id == from_id
                and edge.target_id == to_id
                and edge.edge_type == edge_type
            ):
                removed_edge = self.edges.pop(i)

                # Update indices
                self._edges_from[from_id].remove(removed_edge)
                self._edges_to[to_id].remove(removed_edge)

                return True

        return False

    def merge_nodes(self, node_id1: str, node_id2: str, merged_id: str) -> ThoughtNode:
        """
        Merge two nodes into one, combining their edges.

        Args:
            node_id1: First node to merge
            node_id2: Second node to merge
            merged_id: ID for the merged node

        Returns:
            The merged ThoughtNode

        Raises:
            ValueError: If either node doesn't exist or merged_id already exists
        """
        if node_id1 not in self.nodes or node_id2 not in self.nodes:
            raise ValueError("Both nodes must exist")
        if merged_id in self.nodes and merged_id not in (node_id1, node_id2):
            raise ValueError(f"Merged ID {merged_id} already exists")

        node1 = self.nodes[node_id1]
        node2 = self.nodes[node_id2]

        # Create merged node (use node1's type, combine content)
        merged_content = f"{node1.content} | {node2.content}"
        merged_properties = {**node1.properties, **node2.properties}
        merged_metadata = {**node1.metadata, **node2.metadata}

        # If merged_id is one of the original nodes, update in place
        if merged_id == node_id1:
            merged_node = node1
            merged_node.content = merged_content
            merged_node.properties = merged_properties
            merged_node.metadata = merged_metadata
            node_to_remove = node_id2
        elif merged_id == node_id2:
            merged_node = node2
            merged_node.content = merged_content
            merged_node.properties = merged_properties
            merged_node.metadata = merged_metadata
            node_to_remove = node_id1
        else:
            # Create new merged node
            merged_node = self.add_node(
                merged_id, node1.node_type, merged_content,
                merged_properties, merged_metadata
            )
            node_to_remove = None

        # Redirect all edges from both nodes to merged node
        edges_to_redirect = []
        for edge in self.edges[:]:  # Copy list to avoid modification during iteration
            if edge.source_id in (node_id1, node_id2):
                edges_to_redirect.append(
                    (merged_id, edge.target_id, edge.edge_type, edge.weight)
                )
            elif edge.target_id in (node_id1, node_id2):
                edges_to_redirect.append(
                    (edge.source_id, merged_id, edge.edge_type, edge.weight)
                )

        # Remove original nodes if needed
        if node_to_remove:
            self.remove_node(node_to_remove)
        elif merged_id not in (node_id1, node_id2):
            self.remove_node(node_id1)
            self.remove_node(node_id2)

        # Add redirected edges (deduplicate)
        seen_edges = set()
        for from_id, to_id, edge_type, weight in edges_to_redirect:
            edge_key = (from_id, to_id, edge_type)
            if edge_key not in seen_edges and from_id != to_id:  # Avoid self-loops
                seen_edges.add(edge_key)
                try:
                    self.add_edge(from_id, to_id, edge_type, weight)
                except ValueError:
                    pass  # Edge already exists

        return merged_node

    def split_node(
        self,
        node_id: str,
        split_id1: str,
        split_id2: str,
        content1: str,
        content2: str,
    ) -> Tuple[ThoughtNode, ThoughtNode]:
        """
        Split a node into two separate nodes.

        Args:
            node_id: ID of node to split
            split_id1: ID for first split node
            split_id2: ID for second split node
            content1: Content for first split node
            content2: Content for second split node

        Returns:
            Tuple of (first_node, second_node)

        Raises:
            ValueError: If node doesn't exist or split IDs already exist
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        if split_id1 in self.nodes or split_id2 in self.nodes:
            raise ValueError("Split IDs already exist")

        original = self.nodes[node_id]

        # Create two new nodes
        node1 = self.add_node(
            split_id1, original.node_type, content1,
            original.properties.copy(), original.metadata.copy()
        )
        node2 = self.add_node(
            split_id2, original.node_type, content2,
            original.properties.copy(), original.metadata.copy()
        )

        # Copy edges to both new nodes (user can prune later)
        for edge in self.get_edges_from(node_id):
            self.add_edge(split_id1, edge.target_id, edge.edge_type, edge.weight)
            self.add_edge(split_id2, edge.target_id, edge.edge_type, edge.weight)

        for edge in self.get_edges_to(node_id):
            self.add_edge(edge.source_id, split_id1, edge.edge_type, edge.weight)
            self.add_edge(edge.source_id, split_id2, edge.edge_type, edge.weight)

        # Remove original node
        self.remove_node(node_id)

        return node1, node2

    # =========================================================================
    # TRAVERSAL OPERATIONS
    # =========================================================================

    def dfs(self, start_id: str) -> List[str]:
        """
        Depth-first search traversal from a starting node.

        Args:
            start_id: ID of node to start from

        Returns:
            List of node IDs in DFS order

        Raises:
            ValueError: If start node doesn't exist
        """
        if start_id not in self.nodes:
            raise ValueError(f"Start node {start_id} not found")

        visited = set()
        result = []

        def dfs_recursive(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)
            result.append(node_id)

            for neighbor_id in self.get_neighbors(node_id):
                dfs_recursive(neighbor_id)

        dfs_recursive(start_id)
        return result

    def bfs(self, start_id: str) -> List[str]:
        """
        Breadth-first search traversal from a starting node.

        Args:
            start_id: ID of node to start from

        Returns:
            List of node IDs in BFS order

        Raises:
            ValueError: If start node doesn't exist
        """
        if start_id not in self.nodes:
            raise ValueError(f"Start node {start_id} not found")

        visited = {start_id}
        result = [start_id]
        queue = deque([start_id])

        while queue:
            node_id = queue.popleft()

            for neighbor_id in self.get_neighbors(node_id):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    result.append(neighbor_id)
                    queue.append(neighbor_id)

        return result

    def shortest_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """
        Find shortest path between two nodes using BFS.

        Args:
            from_id: Source node ID
            to_id: Target node ID

        Returns:
            List of node IDs forming the path, or None if no path exists

        Raises:
            ValueError: If either node doesn't exist
        """
        if from_id not in self.nodes:
            raise ValueError(f"Source node {from_id} not found")
        if to_id not in self.nodes:
            raise ValueError(f"Target node {to_id} not found")

        if from_id == to_id:
            return [from_id]

        visited = {from_id}
        queue = deque([(from_id, [from_id])])

        while queue:
            node_id, path = queue.popleft()

            for neighbor_id in self.get_neighbors(node_id):
                if neighbor_id == to_id:
                    return path + [neighbor_id]

                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None  # No path found

    def get_neighbors(self, node_id: str) -> List[str]:
        """
        Get all neighboring nodes (nodes connected by outgoing edges).

        Args:
            node_id: ID of node to get neighbors for

        Returns:
            List of neighbor node IDs

        Raises:
            ValueError: If node doesn't exist
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")

        neighbors = []
        for edge in self._edges_from.get(node_id, []):
            neighbors.append(edge.target_id)

        return neighbors

    # =========================================================================
    # ANALYSIS OPERATIONS
    # =========================================================================

    def find_cycles(self) -> List[List[str]]:
        """
        Find all cycles in the graph.

        Returns:
            List of cycles, where each cycle is a list of node IDs
        """
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs_cycle(node_id: str, path: List[str]):
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            for neighbor_id in self.get_neighbors(node_id):
                if neighbor_id not in visited:
                    dfs_cycle(neighbor_id, path[:])
                elif neighbor_id in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor_id)
                    cycle = path[cycle_start:] + [neighbor_id]
                    cycles.append(cycle)

            rec_stack.remove(node_id)

        for node_id in self.nodes:
            if node_id not in visited:
                dfs_cycle(node_id, [])

        return cycles

    def find_orphans(self) -> List[str]:
        """
        Find nodes with no incoming or outgoing edges.

        Returns:
            List of orphan node IDs
        """
        orphans = []
        for node_id in self.nodes:
            if not self._edges_from.get(node_id) and not self._edges_to.get(node_id):
                orphans.append(node_id)
        return orphans

    def find_hubs(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Find nodes with the most connections (highest degree).

        Args:
            top_n: Number of top hubs to return

        Returns:
            List of (node_id, degree) tuples sorted by degree descending
        """
        degrees = {}
        for node_id in self.nodes:
            in_degree = len(self._edges_to.get(node_id, []))
            out_degree = len(self._edges_from.get(node_id, []))
            degrees[node_id] = in_degree + out_degree

        # Sort by degree descending
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_n]

    def find_bridges(self) -> List[str]:
        """
        Find bridge nodes whose removal would disconnect the graph.

        Returns:
            List of bridge node IDs
        """
        bridges = []

        # For each node, check if removing it increases connected components
        original_components = self._count_connected_components()

        for node_id in self.nodes:
            # Temporarily remove node
            node = self.nodes[node_id]
            edges_from = self._edges_from.get(node_id, []).copy()
            edges_to = self._edges_to.get(node_id, []).copy()

            del self.nodes[node_id]
            if node_id in self._edges_from:
                del self._edges_from[node_id]
            if node_id in self._edges_to:
                del self._edges_to[node_id]

            # Remove edges referencing this node
            for edge in edges_from:
                if edge.target_id in self._edges_to:
                    self._edges_to[edge.target_id] = [
                        e for e in self._edges_to[edge.target_id] if e.source_id != node_id
                    ]
            for edge in edges_to:
                if edge.source_id in self._edges_from:
                    self._edges_from[edge.source_id] = [
                        e for e in self._edges_from[edge.source_id] if e.target_id != node_id
                    ]

            # Count components
            new_components = self._count_connected_components()

            # Restore node and edges
            self.nodes[node_id] = node
            if edges_from:
                self._edges_from[node_id] = edges_from
            if edges_to:
                self._edges_to[node_id] = edges_to

            for edge in edges_from:
                if edge.target_id in self.nodes:
                    if edge.target_id not in self._edges_to:
                        self._edges_to[edge.target_id] = []
                    self._edges_to[edge.target_id].append(edge)

            for edge in edges_to:
                if edge.source_id in self.nodes:
                    if edge.source_id not in self._edges_from:
                        self._edges_from[edge.source_id] = []
                    self._edges_from[edge.source_id].append(edge)

            # If removal increased components, it's a bridge
            if new_components > original_components:
                bridges.append(node_id)

        return bridges

    def get_cluster(self, node_id: str) -> Optional[ThoughtCluster]:
        """
        Get the cluster containing a node.

        Args:
            node_id: ID of node to find cluster for

        Returns:
            ThoughtCluster containing the node, or None if not in any cluster
        """
        for cluster in self.clusters.values():
            if cluster.contains_node(node_id):
                return cluster
        return None

    def _count_connected_components(self) -> int:
        """Count the number of connected components in the graph."""
        visited = set()
        components = 0

        for node_id in self.nodes:
            if node_id not in visited:
                components += 1
                # BFS to mark all connected nodes
                queue = deque([node_id])
                visited.add(node_id)

                while queue:
                    current = queue.popleft()
                    # Consider both outgoing and incoming edges for connectivity
                    for edge in self._edges_from.get(current, []):
                        if edge.target_id not in visited:
                            visited.add(edge.target_id)
                            queue.append(edge.target_id)
                    for edge in self._edges_to.get(current, []):
                        if edge.source_id not in visited:
                            visited.add(edge.source_id)
                            queue.append(edge.source_id)

        return components

    # =========================================================================
    # TRANSFORMATION OPERATIONS
    # =========================================================================

    def prune(self, node_ids: List[str]) -> List[ThoughtNode]:
        """
        Remove multiple nodes from the graph.

        Args:
            node_ids: List of node IDs to remove

        Returns:
            List of removed ThoughtNodes
        """
        removed = []
        for node_id in node_ids:
            try:
                removed.append(self.remove_node(node_id))
            except ValueError:
                pass  # Node doesn't exist, skip

        return removed

    def collapse_cluster(self, cluster_id: str) -> ThoughtNode:
        """
        Replace a cluster with a single representative node.

        Args:
            cluster_id: ID of cluster to collapse

        Returns:
            The representative node

        Raises:
            ValueError: If cluster doesn't exist or is empty
        """
        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster {cluster_id} not found")

        cluster = self.clusters[cluster_id]
        if not cluster.node_ids:
            raise ValueError(f"Cluster {cluster_id} is empty")

        # Create representative node
        rep_id = f"cluster_{cluster_id}"
        rep_node = self.add_node(
            rep_id,
            NodeType.CONCEPT,
            f"Collapsed cluster: {cluster.name}",
            properties={"cluster_id": cluster_id, "original_nodes": list(cluster.node_ids)},
        )

        # Find all external edges (edges crossing cluster boundary)
        external_edges_in = []
        external_edges_out = []

        for node_id in cluster.node_ids:
            for edge in self.get_edges_to(node_id):
                if edge.source_id not in cluster.node_ids:
                    external_edges_in.append((edge.source_id, edge.edge_type, edge.weight))

            for edge in self.get_edges_from(node_id):
                if edge.target_id not in cluster.node_ids:
                    external_edges_out.append((edge.target_id, edge.edge_type, edge.weight))

        # Remove all nodes in cluster
        for node_id in list(cluster.node_ids):
            self.remove_node(node_id)

        # Add external edges to representative node (deduplicate)
        seen_in = set()
        for from_id, edge_type, weight in external_edges_in:
            key = (from_id, edge_type)
            if key not in seen_in:
                seen_in.add(key)
                self.add_edge(from_id, rep_id, edge_type, weight)

        seen_out = set()
        for to_id, edge_type, weight in external_edges_out:
            key = (to_id, edge_type)
            if key not in seen_out:
                seen_out.add(key)
                self.add_edge(rep_id, to_id, edge_type, weight)

        # Update cluster to contain only representative
        cluster.node_ids.clear()
        cluster.add_node(rep_id)

        return rep_node

    def expand_cluster(self, cluster_id: str) -> List[ThoughtNode]:
        """
        Expand a collapsed cluster back to its original nodes.

        Args:
            cluster_id: ID of cluster to expand

        Returns:
            List of restored nodes

        Raises:
            ValueError: If cluster doesn't exist or wasn't collapsed
        """
        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster {cluster_id} not found")

        cluster = self.clusters[cluster_id]
        if len(cluster.node_ids) != 1:
            raise ValueError(f"Cluster {cluster_id} is not collapsed")

        # Get the representative node
        rep_id = list(cluster.node_ids)[0]
        rep_node = self.nodes.get(rep_id)

        if not rep_node or "original_nodes" not in rep_node.properties:
            raise ValueError(f"Cluster {cluster_id} cannot be expanded (missing metadata)")

        original_node_ids = rep_node.properties["original_nodes"]

        # We can't fully restore without original node data
        # This is a simplified version that creates placeholder nodes
        restored = []
        for i, orig_id in enumerate(original_node_ids):
            node = self.add_node(
                orig_id,
                NodeType.CONCEPT,
                f"Restored from {cluster.name} [{i+1}/{len(original_node_ids)}]",
                {},
            )
            restored.append(node)
            cluster.add_node(orig_id)

        # Remove representative node from cluster first, then from graph
        # Note: remove_node from graph will try to remove from all clusters
        # so we need to remove from cluster.node_ids manually
        cluster.node_ids.discard(rep_id)
        self.remove_node(rep_id)

        return restored

    # =========================================================================
    # QUERY OPERATIONS
    # =========================================================================

    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """
        Get a node by ID.

        Args:
            node_id: ID of node to retrieve

        Returns:
            ThoughtNode if found, None otherwise
        """
        return self.nodes.get(node_id)

    def get_edges_from(self, node_id: str) -> List[ThoughtEdge]:
        """
        Get all edges originating from a node.

        Args:
            node_id: ID of source node

        Returns:
            List of outgoing edges
        """
        return self._edges_from.get(node_id, []).copy()

    def get_edges_to(self, node_id: str) -> List[ThoughtEdge]:
        """
        Get all edges pointing to a node.

        Args:
            node_id: ID of target node

        Returns:
            List of incoming edges
        """
        return self._edges_to.get(node_id, []).copy()

    def nodes_of_type(self, node_type: NodeType) -> List[ThoughtNode]:
        """
        Get all nodes of a specific type.

        Args:
            node_type: Type of nodes to retrieve

        Returns:
            List of nodes matching the type
        """
        return [node for node in self.nodes.values() if node.node_type == node_type]

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def add_cluster(
        self,
        cluster_id: str,
        name: str,
        node_ids: Optional[Set[str]] = None,
    ) -> ThoughtCluster:
        """
        Create and add a cluster to the graph.

        Args:
            cluster_id: Unique cluster identifier
            name: Human-readable cluster name
            node_ids: Optional set of initial node IDs

        Returns:
            The created ThoughtCluster

        Raises:
            ValueError: If cluster_id already exists
        """
        if cluster_id in self.clusters:
            raise ValueError(f"Cluster {cluster_id} already exists")

        cluster = ThoughtCluster(
            id=cluster_id,
            name=name,
            node_ids=node_ids or set(),
        )
        self.clusters[cluster_id] = cluster
        return cluster

    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Return the number of edges in the graph."""
        return len(self.edges)

    def cluster_count(self) -> int:
        """Return the number of clusters in the graph."""
        return len(self.clusters)
