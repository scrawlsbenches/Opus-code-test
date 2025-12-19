#!/usr/bin/env python3
"""
Unit tests for knowledge_analysis.py

Tests pattern detection, clustering, strength scoring, and density calculation
for concept network analysis.
"""

import json
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

# Import the module we're testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.knowledge_analysis import KnowledgeAnalyzer


class TestInputTypeDetection(unittest.TestCase):
    """Test input type detection from different sources."""

    def test_detect_question_connection_input(self):
        """Test detection of question_connection.py output."""
        data = {
            "query": "test query",
            "expanded_terms": ["term1", "term2"],
            "paths": [],
            "related_concepts": []
        }
        analyzer = KnowledgeAnalyzer(data)
        self.assertEqual(analyzer.input_type, "question_connection")

    def test_detect_world_model_analysis_input(self):
        """Test detection of world_model_analysis.py output."""
        data = {
            "concepts": [{"term": "model", "pagerank": 0.5}],
            "bridges": [],
            "network": {}
        }
        analyzer = KnowledgeAnalyzer(data)
        self.assertEqual(analyzer.input_type, "world_model_analysis")

    def test_detect_unknown_input(self):
        """Test detection when input type is unknown."""
        data = {"unknown_field": "value"}
        analyzer = KnowledgeAnalyzer(data)
        self.assertEqual(analyzer.input_type, "unknown")

    def test_question_connection_takes_precedence(self):
        """Test that query field takes precedence for detection."""
        data = {
            "query": "test",
            "concepts": []  # Both fields present, but query wins
        }
        analyzer = KnowledgeAnalyzer(data)
        self.assertEqual(analyzer.input_type, "question_connection")


class TestNetworkExtraction(unittest.TestCase):
    """Test network extraction from different input formats."""

    def test_extract_from_world_model_concepts(self):
        """Test extracting terms from world_model_analysis concepts."""
        data = {
            "concepts": [
                {"term": "model", "pagerank": 0.5},
                {"term": "prediction", "pagerank": 0.4}
            ],
            "network": {}
        }
        analyzer = KnowledgeAnalyzer(data)
        self.assertIn("model", analyzer.terms)
        self.assertIn("prediction", analyzer.terms)
        self.assertEqual(len(analyzer.terms), 2)

    def test_extract_from_world_model_network(self):
        """Test extracting connections from world_model_analysis network."""
        data = {
            "concepts": [],
            "network": {
                "model": [
                    {"term": "prediction", "weight": 5.0},
                    {"term": "learning", "weight": 3.0}
                ]
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        self.assertIn("model", analyzer.terms)
        self.assertIn("prediction", analyzer.terms)
        self.assertIn("learning", analyzer.terms)
        self.assertEqual(analyzer.connections["model"]["prediction"], 5.0)
        self.assertEqual(analyzer.connections["model"]["learning"], 3.0)

    def test_extract_from_question_connection_paths(self):
        """Test extracting connections from question_connection paths."""
        data = {
            "query": "test",
            "paths": [
                ["term1", "term2", "term3"],
                ["term2", "term4"]
            ]
        }
        analyzer = KnowledgeAnalyzer(data)
        self.assertIn("term1", analyzer.terms)
        self.assertIn("term2", analyzer.terms)
        self.assertIn("term3", analyzer.terms)
        self.assertIn("term4", analyzer.terms)
        # Check bidirectional connections
        self.assertGreater(analyzer.connections["term1"]["term2"], 0)
        self.assertGreater(analyzer.connections["term2"]["term1"], 0)

    def test_extract_domains_from_concepts(self):
        """Test extracting domain information from concepts."""
        data = {
            "concepts": [
                {
                    "term": "model",
                    "domains": ["cognitive_science", "world_models"]
                }
            ]
        }
        analyzer = KnowledgeAnalyzer(data)
        self.assertIn("cognitive_science", analyzer.term_domains["model"])
        self.assertIn("world_models", analyzer.term_domains["model"])

    def test_extract_domains_from_bridges(self):
        """Test extracting domain information from bridges."""
        data = {
            "bridges": [
                {
                    "term": "prediction",
                    "domains": ["cognitive_science", "ai_market_prediction", "world_models"]
                }
            ]
        }
        analyzer = KnowledgeAnalyzer(data)
        self.assertEqual(len(analyzer.term_domains["prediction"]), 3)

    def test_extract_with_dict_format_network(self):
        """Test extracting from dict-format network."""
        data = {
            "concepts": [],
            "network": {
                "term1": {"term2": 5.0, "term3": 3.0}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        self.assertEqual(analyzer.connections["term1"]["term2"], 5.0)
        self.assertEqual(analyzer.connections["term1"]["term3"], 3.0)


class TestPatternDetection(unittest.TestCase):
    """Test pattern detection (hubs, clusters, bridges)."""

    def test_detect_hub_pattern(self):
        """Test detection of hub patterns."""
        data = {
            "concepts": [],
            "network": {
                "hub": {
                    "term1": 1.0,
                    "term2": 1.0,
                    "term3": 1.0,
                    "term4": 1.0,
                    "term5": 1.0
                },
                "term1": {"hub": 1.0},
                "term2": {"hub": 1.0},
                "term3": {"hub": 1.0},
                "term4": {"hub": 1.0},
                "term5": {"hub": 1.0}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        patterns = analyzer.detect_patterns()

        hub_patterns = [p for p in patterns if p["type"] == "hub"]
        self.assertGreater(len(hub_patterns), 0)
        hub_terms = [p["term"] for p in hub_patterns]
        self.assertIn("hub", hub_terms)

    def test_detect_cluster_pattern(self):
        """Test detection of cluster patterns."""
        data = {
            "concepts": [],
            "network": {
                "term1": {"term2": 5.0, "term3": 5.0},
                "term2": {"term1": 5.0, "term3": 5.0},
                "term3": {"term1": 5.0, "term2": 5.0}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        patterns = analyzer.detect_patterns()

        cluster_patterns = [p for p in patterns if p["type"] == "cluster"]
        self.assertGreater(len(cluster_patterns), 0)

        # Verify cluster contains connected terms
        for pattern in cluster_patterns:
            self.assertIn("terms", pattern)
            self.assertIn("strength", pattern)
            self.assertGreaterEqual(len(pattern["terms"]), 2)

    def test_detect_bridge_pattern(self):
        """Test detection of bridge patterns (multi-domain terms)."""
        data = {
            "bridges": [
                {
                    "term": "model",
                    "domains": ["cognitive_science", "world_models", "ai_market_prediction"]
                }
            ]
        }
        analyzer = KnowledgeAnalyzer(data)
        patterns = analyzer.detect_patterns()

        bridge_patterns = [p for p in patterns if p["type"] == "bridge"]
        self.assertGreater(len(bridge_patterns), 0)

        model_bridge = next((p for p in bridge_patterns if p["term"] == "model"), None)
        self.assertIsNotNone(model_bridge)
        self.assertEqual(len(model_bridge["domains"]), 3)

    def test_patterns_with_minimal_data(self):
        """Test pattern detection with minimal data."""
        data = {"concepts": [{"term": "single"}]}
        analyzer = KnowledgeAnalyzer(data)
        patterns = analyzer.detect_patterns()

        # Should not crash, but won't find many patterns
        self.assertIsInstance(patterns, list)


class TestClustering(unittest.TestCase):
    """Test cluster building and coherence calculation."""

    def test_build_clusters_basic(self):
        """Test basic cluster building."""
        data = {
            "concepts": [],
            "network": {
                "term1": {"term2": 5.0},
                "term2": {"term1": 5.0, "term3": 5.0},
                "term3": {"term2": 5.0}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        clusters = analyzer.build_clusters()

        self.assertGreater(len(clusters), 0)
        for cluster in clusters:
            self.assertIn("id", cluster)
            self.assertIn("terms", cluster)
            self.assertIn("coherence", cluster)
            self.assertIn("size", cluster)

    def test_cluster_coherence_calculation(self):
        """Test cluster coherence calculation."""
        data = {
            "concepts": [],
            "network": {
                "term1": {"term2": 10.0, "term3": 10.0},
                "term2": {"term1": 10.0, "term3": 10.0},
                "term3": {"term1": 10.0, "term2": 10.0}
            }
        }
        analyzer = KnowledgeAnalyzer(data)

        # Highly connected cluster should have high coherence
        cluster = {"term1", "term2", "term3"}
        coherence = analyzer._calculate_cluster_coherence(cluster)
        self.assertGreater(coherence, 0.5)

    def test_cluster_with_no_connections(self):
        """Test cluster coherence when terms aren't connected."""
        data = {
            "concepts": [
                {"term": "isolated1"},
                {"term": "isolated2"}
            ]
        }
        analyzer = KnowledgeAnalyzer(data)

        cluster = {"isolated1", "isolated2"}
        coherence = analyzer._calculate_cluster_coherence(cluster)
        self.assertEqual(coherence, 0)

    def test_single_term_cluster(self):
        """Test coherence of single-term cluster."""
        data = {"concepts": [{"term": "single"}]}
        analyzer = KnowledgeAnalyzer(data)

        cluster = {"single"}
        coherence = analyzer._calculate_cluster_coherence(cluster)
        self.assertEqual(coherence, 1.0)  # Single term is perfectly coherent

    def test_clusters_sorted_by_size(self):
        """Test that clusters are sorted by size."""
        data = {
            "concepts": [],
            "network": {
                "a1": {"a2": 5.0},
                "a2": {"a1": 5.0},
                "b1": {"b2": 5.0, "b3": 5.0},
                "b2": {"b1": 5.0, "b3": 5.0},
                "b3": {"b1": 5.0, "b2": 5.0}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        clusters = analyzer.build_clusters()

        if len(clusters) >= 2:
            # First cluster should be largest or equal
            self.assertGreaterEqual(clusters[0]["size"], clusters[-1]["size"])


class TestStrengthScoring(unittest.TestCase):
    """Test connection strength scoring."""

    def test_calculate_overall_strength(self):
        """Test overall strength score calculation."""
        data = {
            "concepts": [],
            "network": {
                "term1": {"term2": 5.0, "term3": 3.0},
                "term2": {"term1": 5.0}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        scores = analyzer.calculate_strength_scores()

        self.assertIn("overall", scores)
        self.assertGreater(scores["overall"], 0)

    def test_calculate_domain_strength(self):
        """Test strength scores by domain."""
        data = {
            "concepts": [
                {"term": "term1", "domains": ["domain1"]},
                {"term": "term2", "domains": ["domain1"]}
            ],
            "network": {
                "term1": {"term2": 5.0},
                "term2": {"term1": 5.0}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        scores = analyzer.calculate_strength_scores()

        self.assertIn("by_domain", scores)
        self.assertIn("domain1", scores["by_domain"])
        self.assertGreater(scores["by_domain"]["domain1"], 0)

    def test_strength_with_no_connections(self):
        """Test strength calculation with no connections."""
        data = {"concepts": [{"term": "isolated"}]}
        analyzer = KnowledgeAnalyzer(data)
        scores = analyzer.calculate_strength_scores()

        self.assertEqual(scores["overall"], 0)
        self.assertEqual(scores["by_domain"], {})

    def test_strength_multiple_domains(self):
        """Test strength calculation across multiple domains."""
        data = {
            "concepts": [
                {"term": "term1", "domains": ["domain1", "domain2"]},
                {"term": "term2", "domains": ["domain1"]}
            ],
            "network": {
                "term1": {"term2": 10.0},
                "term2": {"term1": 10.0}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        scores = analyzer.calculate_strength_scores()

        self.assertIn("domain1", scores["by_domain"])
        self.assertIn("domain2", scores["by_domain"])


class TestDensityCalculation(unittest.TestCase):
    """Test knowledge density mapping."""

    def test_density_map_basic(self):
        """Test basic density map calculation."""
        data = {
            "concepts": [],
            "network": {
                "high": {"t1": 1, "t2": 1, "t3": 1, "t4": 1, "t5": 1},
                "medium": {"t1": 1, "t2": 1},
                "low": {"t1": 1},
                "t1": {}, "t2": {}, "t3": {}, "t4": {}, "t5": {}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        density_map = analyzer.calculate_density_map()

        self.assertIn("high", density_map)
        self.assertIn("medium", density_map)
        self.assertIn("low", density_map)

        # High density term should have many connections
        self.assertIn("high", density_map["high"])

    def test_density_map_empty(self):
        """Test density map with no data."""
        data = {"concepts": []}
        analyzer = KnowledgeAnalyzer(data)
        density_map = analyzer.calculate_density_map()

        self.assertEqual(density_map["high"], [])
        self.assertEqual(density_map["medium"], [])
        self.assertEqual(density_map["low"], [])

    def test_density_map_sorted(self):
        """Test that density map lists are sorted."""
        data = {
            "concepts": [],
            "network": {
                "z_term": {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1},
                "a_term": {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1},
                "a": {}, "b": {}, "c": {}, "d": {}, "e": {}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        density_map = analyzer.calculate_density_map()

        # Check alphabetical sorting in high density
        if len(density_map["high"]) >= 2:
            self.assertEqual(density_map["high"], sorted(density_map["high"]))

    def test_density_with_few_terms(self):
        """Test density calculation with very few terms."""
        data = {
            "concepts": [],
            "network": {
                "term1": {"term2": 1},
                "term2": {"term1": 1}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        density_map = analyzer.calculate_density_map()

        # Should not crash with few terms
        total_terms = len(density_map["high"]) + len(density_map["medium"]) + len(density_map["low"])
        self.assertEqual(total_terms, 2)


class TestFullAnalysis(unittest.TestCase):
    """Test complete analysis workflow."""

    def test_analyze_returns_all_fields(self):
        """Test that analyze() returns all required fields."""
        data = {
            "concepts": [
                {"term": "model", "pagerank": 0.5},
                {"term": "prediction", "pagerank": 0.4}
            ],
            "network": {
                "model": [{"term": "prediction", "weight": 5.0}]
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        results = analyzer.analyze()

        self.assertIn("patterns", results)
        self.assertIn("clusters", results)
        self.assertIn("strength_scores", results)
        self.assertIn("density_map", results)
        self.assertIn("input_type", results)
        self.assertIn("metadata", results)

    def test_metadata_accuracy(self):
        """Test that metadata counts are accurate."""
        data = {
            "concepts": [
                {"term": "term1"},
                {"term": "term2"},
                {"term": "term3"}
            ],
            "network": {
                "term1": {"term2": 1},
                "term2": {"term1": 1}
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        results = analyzer.analyze()

        self.assertEqual(results["metadata"]["term_count"], 3)
        self.assertGreater(results["metadata"]["connection_count"], 0)

    def test_analyze_with_verbose(self):
        """Test analyze with verbose mode."""
        data = {"concepts": [{"term": "test"}]}

        # Capture stderr
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            analyzer = KnowledgeAnalyzer(data, verbose=True)
            results = analyzer.analyze()

            stderr_output = mock_stderr.getvalue()
            self.assertIn("Detected input type", stderr_output)
            self.assertIn("Extracted", stderr_output)

    def test_analyze_minimal_input(self):
        """Test analysis with minimal valid input."""
        data = {"concepts": []}
        analyzer = KnowledgeAnalyzer(data)
        results = analyzer.analyze()

        # Should not crash
        self.assertIsInstance(results, dict)
        self.assertEqual(results["metadata"]["term_count"], 0)

    def test_full_pipeline_world_model_input(self):
        """Test full analysis with realistic world_model_analysis input."""
        data = {
            "concepts": [
                {"term": "model", "pagerank": 0.05, "domains": ["cognitive_science"]},
                {"term": "prediction", "pagerank": 0.04, "domains": ["cognitive_science", "world_models"]},
                {"term": "learning", "pagerank": 0.03, "domains": ["cognitive_science"]}
            ],
            "bridges": [
                {"term": "prediction", "domains": ["cognitive_science", "world_models", "ai_market_prediction"]}
            ],
            "network": {
                "model": [
                    {"term": "prediction", "weight": 5.0},
                    {"term": "learning", "weight": 3.0}
                ],
                "prediction": [
                    {"term": "model", "weight": 5.0},
                    {"term": "learning", "weight": 2.0}
                ],
                "learning": [
                    {"term": "model", "weight": 3.0},
                    {"term": "prediction", "weight": 2.0}
                ]
            }
        }
        analyzer = KnowledgeAnalyzer(data)
        results = analyzer.analyze()

        # Verify comprehensive analysis
        self.assertGreater(len(results["patterns"]), 0)
        self.assertEqual(results["input_type"], "world_model_analysis")
        self.assertEqual(results["metadata"]["term_count"], 3)

        # Should detect bridge pattern for "prediction"
        bridge_patterns = [p for p in results["patterns"] if p["type"] == "bridge"]
        self.assertGreater(len(bridge_patterns), 0)

    def test_full_pipeline_question_connection_input(self):
        """Test full analysis with question_connection-style input."""
        data = {
            "query": "how do models make predictions?",
            "expanded_terms": [
                {"term": "model", "weight": 1.0},
                {"term": "prediction", "weight": 0.8}
            ],
            "paths": [
                ["model", "learning", "prediction"],
                ["model", "inference", "prediction"]
            ],
            "related_concepts": ["uncertainty", "belief"]
        }
        analyzer = KnowledgeAnalyzer(data)
        results = analyzer.analyze()

        # Verify analysis
        self.assertEqual(results["input_type"], "question_connection")
        self.assertGreater(results["metadata"]["term_count"], 0)
        self.assertGreater(results["metadata"]["connection_count"], 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test with completely empty data."""
        data = {}
        analyzer = KnowledgeAnalyzer(data)
        results = analyzer.analyze()

        self.assertEqual(results["metadata"]["term_count"], 0)
        self.assertEqual(len(results["patterns"]), 0)

    def test_malformed_concepts(self):
        """Test with malformed concept entries."""
        data = {
            "concepts": [
                {"term": "valid"},
                {},  # Missing term
                {"no_term_field": "value"},
                {"term": ""}  # Empty term
            ]
        }
        analyzer = KnowledgeAnalyzer(data)

        # Should extract only valid term
        self.assertEqual(len(analyzer.terms), 1)
        self.assertIn("valid", analyzer.terms)

    def test_malformed_network(self):
        """Test with malformed network entries."""
        data = {
            "concepts": [],
            "network": {
                "term1": [
                    {"term": "term2", "weight": 5.0},
                    {},  # Missing term
                    {"weight": 3.0},  # Missing term field
                ]
            }
        }
        analyzer = KnowledgeAnalyzer(data)

        # Should extract valid connections only
        self.assertIn("term1", analyzer.terms)
        self.assertIn("term2", analyzer.terms)
        self.assertEqual(analyzer.connections["term1"]["term2"], 5.0)

    def test_zero_weight_connections(self):
        """Test handling of zero-weight connections."""
        data = {
            "concepts": [],
            "network": {
                "term1": {"term2": 0.0}
            }
        }
        analyzer = KnowledgeAnalyzer(data)

        # Zero-weight connections should still be tracked
        self.assertEqual(analyzer.connections["term1"]["term2"], 0.0)

    def test_negative_weights(self):
        """Test handling of negative weights."""
        data = {
            "concepts": [],
            "network": {
                "term1": {"term2": -5.0}
            }
        }
        analyzer = KnowledgeAnalyzer(data)

        # Negative weights should be preserved
        self.assertEqual(analyzer.connections["term1"]["term2"], -5.0)

    def test_very_large_network(self):
        """Test with a larger network (performance check)."""
        # Create a network with 100 terms
        concepts = [{"term": f"term{i}"} for i in range(100)]
        network = {}

        # Create connections: each term connects to next 5 terms
        for i in range(100):
            term = f"term{i}"
            network[term] = {}
            for j in range(1, 6):
                next_term = f"term{(i + j) % 100}"
                network[term][next_term] = float(j)

        data = {
            "concepts": concepts,
            "network": network
        }

        analyzer = KnowledgeAnalyzer(data)
        results = analyzer.analyze()

        # Should complete without timeout
        self.assertEqual(results["metadata"]["term_count"], 100)
        self.assertGreater(len(results["patterns"]), 0)


if __name__ == "__main__":
    unittest.main()
