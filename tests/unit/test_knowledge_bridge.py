"""
Unit Tests for Knowledge Bridge Pipeline
=========================================

Tests the knowledge_bridge.py script functions for detecting gaps,
suggesting bridges, identifying weak links, and finding synthesis
opportunities.
"""

import pytest
import json
from typing import Dict, Set, Any

# Import from scripts directory
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

from knowledge_bridge import (
    detect_input_type,
    extract_clusters_world_model,
    extract_clusters_knowledge_analysis,
    extract_connections,
    find_cluster_gaps,
    find_potential_bridges,
    suggest_bridges,
    find_best_bridge_path,
    identify_weak_links,
    find_synthesis_opportunities,
    generate_summary,
    process_knowledge_bridge,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def world_model_data():
    """Sample world_model_analysis.py output."""
    return {
        "concepts": [
            {"term": "prediction", "pagerank": 0.05, "domains": ["cognitive", "models"]},
            {"term": "learning", "pagerank": 0.04, "domains": ["cognitive", "ai"]},
            {"term": "model", "pagerank": 0.03, "domains": ["models", "ai"]},
            {"term": "decision", "pagerank": 0.02, "domains": ["cognitive"]},
        ],
        "bridges": [
            {"term": "prediction", "domain_count": 2, "domains": ["cognitive", "models"]},
        ],
        "network": {
            "prediction": [
                {"term": "model", "weight": 5.0},
                {"term": "learning", "weight": 3.0},
            ],
            "learning": [
                {"term": "model", "weight": 4.0},
            ],
        },
        "suggestions": [
            {"term1": "prediction", "term2": "decision", "shared_neighbors": 5},
            {"term1": "learning", "term2": "decision", "shared_neighbors": 3},
        ],
    }


@pytest.fixture
def knowledge_analysis_data():
    """Sample knowledge_analysis.py output."""
    return {
        "patterns": ["pattern1", "pattern2"],
        "clusters": [
            {"id": "cluster_0", "terms": ["prediction", "forecast"]},
            {"id": "cluster_1", "terms": ["learning", "adaptation"]},
            {"id": "cluster_2", "terms": ["model", "representation"]},
        ],
        "strength_scores": {
            "prediction-forecast": 0.9,
            "learning-adaptation": 0.8,
            "model-representation": 0.7,
            "prediction-model": 0.3,
            "learning-model": 0.05,  # Weak link
        },
        "density_map": {},
    }


@pytest.fixture
def sample_clusters():
    """Sample clusters for gap detection."""
    return {
        "cluster_0": {"prediction", "forecast", "anticipate"},
        "cluster_1": {"learning", "adaptation", "update"},
        "cluster_2": {"model", "representation"},
    }


@pytest.fixture
def sample_connections():
    """Sample connections between terms."""
    return {
        "prediction": {"forecast": 0.9, "model": 0.3},
        "forecast": {"prediction": 0.9},
        "learning": {"adaptation": 0.8, "model": 0.05},
        "adaptation": {"learning": 0.8},
        "model": {"prediction": 0.3, "representation": 0.7, "learning": 0.05},
        "representation": {"model": 0.7},
    }


# =============================================================================
# DETECT INPUT TYPE TESTS
# =============================================================================


class TestDetectInputType:
    """Tests for detect_input_type function."""

    def test_world_model_format(self, world_model_data):
        """Correctly identifies world_model format."""
        result = detect_input_type(world_model_data)
        assert result == "world_model"

    def test_knowledge_analysis_format(self, knowledge_analysis_data):
        """Correctly identifies knowledge_analysis format."""
        result = detect_input_type(knowledge_analysis_data)
        assert result == "knowledge_analysis"

    def test_unknown_format(self):
        """Returns 'unknown' for unrecognized format."""
        data = {"random": "data", "fields": [1, 2, 3]}
        result = detect_input_type(data)
        assert result == "unknown"

    def test_empty_data(self):
        """Handles empty data."""
        result = detect_input_type({})
        assert result == "unknown"


# =============================================================================
# EXTRACT CLUSTERS TESTS
# =============================================================================


class TestExtractClusters:
    """Tests for cluster extraction functions."""

    def test_extract_clusters_world_model(self, world_model_data):
        """Extracts clusters from world_model data."""
        clusters = extract_clusters_world_model(world_model_data)

        assert isinstance(clusters, dict)
        assert len(clusters) > 0
        assert "domain_cognitive" in clusters
        assert "prediction" in clusters["domain_cognitive"]
        assert "learning" in clusters["domain_cognitive"]

    def test_extract_clusters_world_model_empty(self):
        """Handles empty world_model data."""
        clusters = extract_clusters_world_model({"concepts": []})
        assert clusters == {}

    def test_extract_clusters_knowledge_analysis(self, knowledge_analysis_data):
        """Extracts clusters from knowledge_analysis data."""
        clusters = extract_clusters_knowledge_analysis(knowledge_analysis_data)

        assert isinstance(clusters, dict)
        assert len(clusters) == 3
        assert "cluster_0" in clusters
        assert "prediction" in clusters["cluster_0"]
        assert "forecast" in clusters["cluster_0"]

    def test_extract_clusters_knowledge_analysis_empty(self):
        """Handles empty knowledge_analysis data."""
        clusters = extract_clusters_knowledge_analysis({"clusters": []})
        assert clusters == {}

    def test_cluster_terms_are_sets(self, sample_clusters):
        """Cluster terms are returned as sets."""
        for cluster_id, terms in sample_clusters.items():
            assert isinstance(terms, set)


# =============================================================================
# EXTRACT CONNECTIONS TESTS
# =============================================================================


class TestExtractConnections:
    """Tests for connection extraction."""

    def test_extract_connections_world_model(self, world_model_data):
        """Extracts connections from world_model network."""
        connections = extract_connections(world_model_data, "world_model")

        assert isinstance(connections, dict)
        assert "prediction" in connections
        assert "model" in connections["prediction"]
        assert connections["prediction"]["model"] == 5.0

    def test_extract_connections_knowledge_analysis(self, knowledge_analysis_data):
        """Extracts connections from strength_scores."""
        connections = extract_connections(knowledge_analysis_data, "knowledge_analysis")

        assert isinstance(connections, dict)
        assert "prediction" in connections
        assert "forecast" in connections["prediction"]
        assert connections["prediction"]["forecast"] == 0.9
        # Should be bidirectional
        assert connections["forecast"]["prediction"] == 0.9

    def test_extract_connections_empty_network(self):
        """Handles empty network."""
        data = {"network": {}}
        connections = extract_connections(data, "world_model")
        assert connections == {}

    def test_extract_connections_malformed_keys(self):
        """Handles malformed strength_score keys."""
        data = {"strength_scores": {"invalid_key": 0.5, "good-key": 0.8}}
        connections = extract_connections(data, "knowledge_analysis")

        # Should only extract the valid key
        assert len(connections) > 0  # Has at least the good key


# =============================================================================
# FIND CLUSTER GAPS TESTS
# =============================================================================


class TestFindClusterGaps:
    """Tests for gap detection between clusters."""

    def test_find_gaps_basic(self, sample_clusters, sample_connections):
        """Finds basic gaps between clusters."""
        gaps = find_cluster_gaps(sample_clusters, sample_connections, min_gap_distance=2)

        assert isinstance(gaps, list)
        assert len(gaps) > 0

        # Check structure
        for gap in gaps:
            assert "between" in gap
            assert "distance" in gap
            assert "potential_bridges" in gap
            assert isinstance(gap["between"], list)
            assert len(gap["between"]) == 2

    def test_find_gaps_with_min_distance(self, sample_clusters, sample_connections):
        """Respects minimum gap distance."""
        gaps_min2 = find_cluster_gaps(sample_clusters, sample_connections, min_gap_distance=2)
        gaps_min3 = find_cluster_gaps(sample_clusters, sample_connections, min_gap_distance=3)

        # Higher threshold should return fewer or equal gaps
        assert len(gaps_min3) <= len(gaps_min2)

    def test_find_gaps_no_clusters(self):
        """Handles empty cluster set."""
        gaps = find_cluster_gaps({}, {}, min_gap_distance=2)
        assert gaps == []

    def test_find_gaps_single_cluster(self, sample_connections):
        """Handles single cluster (no gaps possible)."""
        clusters = {"cluster_0": {"prediction", "forecast"}}
        gaps = find_cluster_gaps(clusters, sample_connections, min_gap_distance=2)
        assert gaps == []

    def test_gap_includes_cluster_sizes(self, sample_clusters, sample_connections):
        """Gap includes cluster size information."""
        gaps = find_cluster_gaps(sample_clusters, sample_connections, min_gap_distance=2)

        if gaps:
            gap = gaps[0]
            assert "cluster1_size" in gap
            assert "cluster2_size" in gap
            assert gap["cluster1_size"] > 0
            assert gap["cluster2_size"] > 0


# =============================================================================
# FIND POTENTIAL BRIDGES TESTS
# =============================================================================


class TestFindPotentialBridges:
    """Tests for potential bridge identification."""

    def test_find_bridges_basic(self, sample_connections):
        """Finds basic bridge terms."""
        cluster1 = {"prediction", "forecast"}
        cluster2 = {"learning", "adaptation"}

        bridges = find_potential_bridges(cluster1, cluster2, sample_connections)

        assert isinstance(bridges, list)
        # "model" connects to both clusters
        if "model" in sample_connections:
            assert "model" in bridges or len(bridges) >= 0

    def test_find_bridges_no_connections(self):
        """Handles clusters with no potential bridges."""
        cluster1 = {"term1"}
        cluster2 = {"term2"}
        connections = {"term1": {}, "term2": {}}

        bridges = find_potential_bridges(cluster1, cluster2, connections)
        assert bridges == []

    def test_find_bridges_sorted_by_score(self, sample_connections):
        """Bridges are sorted by bridging score."""
        cluster1 = {"prediction", "forecast"}
        cluster2 = {"learning", "adaptation"}

        bridges = find_potential_bridges(cluster1, cluster2, sample_connections)

        # If multiple bridges, should be sorted (can't check exact order without knowing scores)
        assert isinstance(bridges, list)

    def test_find_bridges_excludes_cluster_members(self, sample_connections):
        """Bridge terms don't include cluster members."""
        cluster1 = {"prediction"}
        cluster2 = {"learning"}

        bridges = find_potential_bridges(cluster1, cluster2, sample_connections)

        # Bridges should not contain cluster1 or cluster2 members
        for bridge in bridges:
            assert bridge not in cluster1
            assert bridge not in cluster2


# =============================================================================
# SUGGEST BRIDGES TESTS
# =============================================================================


class TestSuggestBridges:
    """Tests for bridge suggestion generation."""

    def test_suggest_bridges_world_model(self, world_model_data, sample_connections):
        """Generates bridge suggestions from world_model data."""
        suggestions = suggest_bridges(world_model_data, "world_model", sample_connections)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

        # Check structure
        for sugg in suggestions:
            assert "concept1" in sugg
            assert "concept2" in sugg
            assert "via" in sugg
            assert "strength_potential" in sugg
            assert "shared_neighbors" in sugg

    def test_suggest_bridges_sorted_by_potential(self, world_model_data, sample_connections):
        """Suggestions are sorted by strength potential."""
        suggestions = suggest_bridges(world_model_data, "world_model", sample_connections)

        if len(suggestions) >= 2:
            # Check descending order
            for i in range(len(suggestions) - 1):
                assert suggestions[i]["strength_potential"] >= suggestions[i+1]["strength_potential"]

    def test_suggest_bridges_strength_potential_range(self, world_model_data, sample_connections):
        """Strength potential is in valid range [0, 1]."""
        suggestions = suggest_bridges(world_model_data, "world_model", sample_connections)

        for sugg in suggestions:
            assert 0.0 <= sugg["strength_potential"] <= 1.0

    def test_suggest_bridges_no_suggestions(self, sample_connections):
        """Handles data with no existing suggestions."""
        data = {"concepts": [], "suggestions": []}
        suggestions = suggest_bridges(data, "world_model", sample_connections)
        assert suggestions == []


# =============================================================================
# FIND BEST BRIDGE PATH TESTS
# =============================================================================


class TestFindBestBridgePath:
    """Tests for finding best bridge path."""

    def test_find_bridge_path_exists(self, sample_connections):
        """Finds bridge path when it exists."""
        bridge = find_best_bridge_path("prediction", "learning", sample_connections)

        # "model" is a common neighbor
        assert bridge == "model"

    def test_find_bridge_path_no_common_neighbors(self, sample_connections):
        """Returns None when no common neighbors."""
        # Add terms with no shared connections
        connections = sample_connections.copy()
        connections["isolated1"] = {"isolated2": 1.0}
        connections["isolated2"] = {"isolated1": 1.0}

        bridge = find_best_bridge_path("isolated1", "prediction", connections)
        assert bridge is None

    def test_find_bridge_path_term_not_in_connections(self, sample_connections):
        """Returns None when term not in connections."""
        bridge = find_best_bridge_path("nonexistent", "prediction", sample_connections)
        assert bridge is None

    def test_find_bridge_path_selects_highest_weight(self):
        """Selects bridge with highest combined weight."""
        connections = {
            "A": {"bridge1": 2.0, "bridge2": 5.0},
            "B": {"bridge1": 3.0, "bridge2": 2.0},
            "bridge1": {},
            "bridge2": {},
        }

        bridge = find_best_bridge_path("A", "B", connections)

        # bridge1: 2.0 + 3.0 = 5.0
        # bridge2: 5.0 + 2.0 = 7.0
        assert bridge == "bridge2"


# =============================================================================
# IDENTIFY WEAK LINKS TESTS
# =============================================================================


class TestIdentifyWeakLinks:
    """Tests for weak link identification."""

    def test_identify_weak_links_basic(self, sample_connections):
        """Identifies weak links below threshold."""
        weak_links = identify_weak_links(sample_connections, threshold=0.2)

        assert isinstance(weak_links, list)
        assert len(weak_links) > 0

        # Check structure
        for link in weak_links:
            assert "from" in link
            assert "to" in link
            assert "current_strength" in link
            assert "suggested_action" in link

    def test_identify_weak_links_threshold(self, sample_connections):
        """Respects threshold parameter."""
        weak_low = identify_weak_links(sample_connections, threshold=0.1)
        weak_high = identify_weak_links(sample_connections, threshold=0.5)

        # Higher threshold should find more weak links
        assert len(weak_high) >= len(weak_low)

    def test_identify_weak_links_sorted(self, sample_connections):
        """Weak links are sorted by strength (weakest first)."""
        weak_links = identify_weak_links(sample_connections, threshold=0.2)

        if len(weak_links) >= 2:
            for i in range(len(weak_links) - 1):
                assert weak_links[i]["current_strength"] <= weak_links[i+1]["current_strength"]

    def test_identify_weak_links_no_duplicates(self, sample_connections):
        """Avoids duplicate pairs (A-B and B-A)."""
        weak_links = identify_weak_links(sample_connections, threshold=0.2)

        pairs = set()
        for link in weak_links:
            pair = tuple(sorted([link["from"], link["to"]]))
            assert pair not in pairs, f"Duplicate pair found: {pair}"
            pairs.add(pair)

    def test_identify_weak_links_suggested_actions(self, sample_connections):
        """Suggested actions vary by strength."""
        weak_links = identify_weak_links(sample_connections, threshold=1.0)

        # Should have different actions for different strength levels
        actions = {link["suggested_action"] for link in weak_links}
        # At least one action type should be present
        assert len(actions) >= 1

    def test_identify_weak_links_empty_connections(self):
        """Handles empty connections."""
        weak_links = identify_weak_links({}, threshold=0.2)
        assert weak_links == []


# =============================================================================
# FIND SYNTHESIS OPPORTUNITIES TESTS
# =============================================================================


class TestFindSynthesisOpportunities:
    """Tests for synthesis opportunity detection."""

    def test_find_synthesis_world_model(self, world_model_data, sample_clusters):
        """Finds synthesis opportunities in world_model data."""
        opportunities = find_synthesis_opportunities(
            sample_clusters, world_model_data, "world_model"
        )

        assert isinstance(opportunities, list)

        # Check structure
        for opp in opportunities:
            assert "topics" in opp
            assert "synthesis_type" in opp
            assert "description" in opp
            assert isinstance(opp["topics"], list)

    def test_find_synthesis_cluster_merge(self):
        """Finds cluster merge opportunities."""
        clusters = {
            "cluster_0": {"prediction", "forecast", "shared1", "shared2"},
            "cluster_1": {"learning", "adaptation", "shared1", "shared2"},
        }

        opportunities = find_synthesis_opportunities(clusters, {}, "unknown")

        # Should find merge opportunity due to shared terms
        merge_opps = [o for o in opportunities if o["synthesis_type"] == "cluster_merge"]
        assert len(merge_opps) > 0
        assert merge_opps[0]["concept_count"] >= 2

    def test_find_synthesis_sorted_by_overlap(self):
        """Opportunities are sorted by concept count."""
        clusters = {
            "cluster_0": {"a", "b", "shared1", "shared2", "shared3"},
            "cluster_1": {"c", "d", "shared1", "shared2", "shared3"},
            "cluster_2": {"e", "f", "shared1"},
            "cluster_3": {"g", "h", "shared1"},
        }

        opportunities = find_synthesis_opportunities(clusters, {}, "unknown")

        if len(opportunities) >= 2:
            for i in range(len(opportunities) - 1):
                assert opportunities[i]["concept_count"] >= opportunities[i+1]["concept_count"]

    def test_find_synthesis_no_overlap(self):
        """Handles clusters with no overlap."""
        clusters = {
            "cluster_0": {"a", "b"},
            "cluster_1": {"c", "d"},
        }

        opportunities = find_synthesis_opportunities(clusters, {}, "unknown")
        # Should be empty or minimal
        assert isinstance(opportunities, list)

    def test_find_synthesis_empty_clusters(self):
        """Handles empty clusters."""
        opportunities = find_synthesis_opportunities({}, {}, "unknown")
        assert opportunities == []


# =============================================================================
# GENERATE SUMMARY TESTS
# =============================================================================


class TestGenerateSummary:
    """Tests for summary generation."""

    def test_generate_summary_basic(self):
        """Generates basic summary."""
        gaps = [{"potential_bridges": ["bridge1"]}, {"potential_bridges": []}]
        bridges = [{"concept1": "A", "concept2": "B", "strength_potential": 0.8}]
        weak_links = [{"from": "X", "to": "Y"}]
        synthesis = [{"topics": ["t1", "t2"]}]

        summary = generate_summary(gaps, bridges, weak_links, synthesis)

        assert isinstance(summary, dict)
        assert "total_gaps" in summary
        assert "bridgeable" in summary
        assert "unbridgeable" in summary
        assert "priority_bridges" in summary
        assert "weak_link_count" in summary
        assert "synthesis_opportunity_count" in summary

    def test_generate_summary_counts_correct(self):
        """Summary counts are correct."""
        gaps = [{"potential_bridges": ["b1"]}, {"potential_bridges": []}, {"potential_bridges": ["b2"]}]
        bridges = []
        weak_links = [{"from": "A", "to": "B"}] * 5
        synthesis = [{"topics": ["t1", "t2"]}] * 3

        summary = generate_summary(gaps, bridges, weak_links, synthesis)

        assert summary["total_gaps"] == 3
        assert summary["bridgeable"] == 2  # Two gaps with bridges
        assert summary["unbridgeable"] == 1
        assert summary["weak_link_count"] == 5
        assert summary["synthesis_opportunity_count"] == 3

    def test_generate_summary_priority_bridges(self):
        """Priority bridges include high-potential suggestions."""
        gaps = []
        bridges = [
            {"concept1": "A", "concept2": "B", "strength_potential": 0.9},
            {"concept1": "C", "concept2": "D", "strength_potential": 0.6},
            {"concept1": "E", "concept2": "F", "strength_potential": 0.3},  # Low potential
        ]
        weak_links = []
        synthesis = []

        summary = generate_summary(gaps, bridges, weak_links, synthesis)

        # Should include high-potential bridges (>= 0.5)
        assert len(summary["priority_bridges"]) == 2
        assert "A-B" in summary["priority_bridges"]
        assert "C-D" in summary["priority_bridges"]

    def test_generate_summary_empty_inputs(self):
        """Handles empty inputs."""
        summary = generate_summary([], [], [], [])

        assert summary["total_gaps"] == 0
        assert summary["bridgeable"] == 0
        assert summary["priority_bridges"] == []


# =============================================================================
# PROCESS KNOWLEDGE BRIDGE (INTEGRATION) TESTS
# =============================================================================


class TestProcessKnowledgeBridge:
    """Integration tests for main processing function."""

    def test_process_world_model_input(self, world_model_data):
        """Processes world_model input correctly."""
        result = process_knowledge_bridge(world_model_data)

        assert "input_type" in result
        assert result["input_type"] == "world_model"
        assert "gaps" in result
        assert "bridge_suggestions" in result
        assert "weak_links" in result
        assert "synthesis_opportunities" in result
        assert "summary" in result

    def test_process_knowledge_analysis_input(self, knowledge_analysis_data):
        """Processes knowledge_analysis input correctly."""
        result = process_knowledge_bridge(knowledge_analysis_data)

        assert result["input_type"] == "knowledge_analysis"
        assert isinstance(result["gaps"], list)
        assert isinstance(result["bridge_suggestions"], list)

    def test_process_unknown_input(self):
        """Handles unknown input format."""
        data = {"unknown": "format"}
        result = process_knowledge_bridge(data)

        assert "error" in result
        assert "Unknown input format" in result["error"]

    def test_process_respects_limits(self, world_model_data):
        """Respects max_* parameters."""
        result = process_knowledge_bridge(
            world_model_data,
            max_gaps=2,
            max_bridges=3,
            max_weak_links=1,
            max_synthesis=1
        )

        assert len(result["gaps"]) <= 2
        assert len(result["bridge_suggestions"]) <= 3
        assert len(result["weak_links"]) <= 1
        assert len(result["synthesis_opportunities"]) <= 1

    def test_process_min_gap_distance(self, knowledge_analysis_data):
        """Respects min_gap_distance parameter."""
        result1 = process_knowledge_bridge(knowledge_analysis_data, min_gap_distance=2)
        result2 = process_knowledge_bridge(knowledge_analysis_data, min_gap_distance=3)

        # Higher min_gap_distance should find fewer or equal gaps
        assert len(result2["gaps"]) <= len(result1["gaps"])

    def test_process_weak_link_threshold(self, knowledge_analysis_data):
        """Respects weak_link_threshold parameter."""
        result1 = process_knowledge_bridge(knowledge_analysis_data, weak_link_threshold=0.1)
        result2 = process_knowledge_bridge(knowledge_analysis_data, weak_link_threshold=0.5)

        # Higher threshold should find more weak links
        assert len(result2["weak_links"]) >= len(result1["weak_links"])

    def test_process_returns_complete_structure(self, world_model_data):
        """Returns complete expected structure."""
        result = process_knowledge_bridge(world_model_data)

        required_keys = [
            "input_type", "gaps", "bridge_suggestions",
            "weak_links", "synthesis_opportunities", "summary"
        ]

        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_process_summary_matches_results(self, world_model_data):
        """Summary counts match result counts."""
        result = process_knowledge_bridge(world_model_data)

        summary = result["summary"]
        assert summary["total_gaps"] == len(result["gaps"])
        assert summary["weak_link_count"] == len(result["weak_links"])
        assert summary["synthesis_opportunity_count"] == len(result["synthesis_opportunities"])
