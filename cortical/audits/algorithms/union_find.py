"""
Union-Find Implementation for Audit Finding Clustering

This implements a disjoint-set data structure (Union-Find) optimized with:
1. Path compression in find() - O(α(n)) amortized time
2. Union by rank - keeps trees balanced

Used to cluster related audit findings for batch processing.
"""

from typing import Dict, List, Set, Optional


class FindingCluster:
    def __init__(self):
        """Initialize empty union-find structure."""
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def make_set(self, finding_id: str) -> None:
        """
        Create a new cluster containing only this finding. No-op if exists.

        Initially, each element is its own parent with rank 0.
        """
        if finding_id not in self._parent:
            self._parent[finding_id] = finding_id
            self._rank[finding_id] = 0

    def find(self, finding_id: str) -> str:
        """
        Find the cluster representative for this finding.

        IMPLEMENTS PATH COMPRESSION:
        - Walk up to root
        - On the way back, make all nodes on path point directly to root
        - This flattens the tree structure for future operations

        Time complexity: O(α(n)) amortized (inverse Ackermann, essentially constant)

        Raises ValueError if finding_id not in structure.
        """
        if finding_id not in self._parent:
            raise ValueError(f"Finding {finding_id} not in union-find structure")

        # Path compression: if not root, recursively find root and update parent
        if self._parent[finding_id] != finding_id:
            # Recursively find root and compress path
            # This makes all nodes on the path point directly to root
            self._parent[finding_id] = self.find(self._parent[finding_id])

        return self._parent[finding_id]

    def union(self, f1: str, f2: str) -> bool:
        """
        Merge clusters containing f1 and f2.

        IMPLEMENTS UNION BY RANK:
        - Find roots of both elements
        - Attach smaller rank tree under root of higher rank tree
        - If ranks equal, attach one under other and increment rank
        - This keeps trees balanced (logarithmic height)

        Returns True if clusters were different (merge happened).
        Returns False if already in same cluster.
        Creates sets for f1, f2 if they don't exist.
        """
        # Auto-create sets if they don't exist
        self.make_set(f1)
        self.make_set(f2)

        # Find roots
        root1 = self.find(f1)
        root2 = self.find(f2)

        # Already in same cluster
        if root1 == root2:
            return False

        # Union by rank: attach smaller tree under larger
        if self._rank[root1] < self._rank[root2]:
            # root2 has higher rank, make it parent of root1
            self._parent[root1] = root2
        elif self._rank[root1] > self._rank[root2]:
            # root1 has higher rank, make it parent of root2
            self._parent[root2] = root1
        else:
            # Equal rank: attach root2 under root1 and increment root1's rank
            self._parent[root2] = root1
            self._rank[root1] += 1

        return True

    def connected(self, f1: str, f2: str) -> bool:
        """
        Return True if f1 and f2 are in the same cluster.

        Two elements are connected if they have the same root.
        Auto-creates sets if they don't exist (for convenience).
        """
        self.make_set(f1)
        self.make_set(f2)
        return self.find(f1) == self.find(f2)

    def cluster_count(self) -> int:
        """
        Return number of distinct clusters.

        Count how many elements are their own parent (roots).
        """
        return sum(1 for finding_id in self._parent
                   if self._parent[finding_id] == finding_id)

    def get_cluster(self, finding_id: str) -> Set[str]:
        """
        Return all findings in the same cluster as finding_id.

        Finds the root of finding_id, then collects all elements with that root.
        """
        if finding_id not in self._parent:
            raise ValueError(f"Finding {finding_id} not in union-find structure")

        root = self.find(finding_id)
        cluster = {fid for fid in self._parent if self.find(fid) == root}
        return cluster

    def get_all_clusters(self) -> List[Set[str]]:
        """
        Return list of all clusters (each cluster is a set of finding_ids).

        Groups all elements by their root.
        """
        # Build map from root to cluster members
        clusters_map: Dict[str, Set[str]] = {}

        for finding_id in self._parent:
            root = self.find(finding_id)
            if root not in clusters_map:
                clusters_map[root] = set()
            clusters_map[root].add(finding_id)

        return list(clusters_map.values())
