#!/usr/bin/env python3
"""
Unit tests for ML file prediction module.

Tests feature extraction, model training, prediction, and evaluation.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from ml_file_prediction import (
    extract_commit_type,
    extract_keywords,
    extract_file_keywords,
    message_to_keywords,
    TrainingExample,
    FilePredictionModel,
    train_model,
    predict_files,
    evaluate_model,
    train_test_split,
    save_model,
    load_model,
)


class TestFeatureExtraction(unittest.TestCase):
    """Tests for feature extraction functions."""

    def test_extract_commit_type_feat(self):
        """Should extract 'feat' type."""
        self.assertEqual(extract_commit_type("feat: Add new feature"), "feat")
        self.assertEqual(extract_commit_type("feat(api): Add endpoint"), "feat")

    def test_extract_commit_type_fix(self):
        """Should extract 'fix' type."""
        self.assertEqual(extract_commit_type("fix: Fix bug"), "fix")
        self.assertEqual(extract_commit_type("fix(auth): Fix login"), "fix")

    def test_extract_commit_type_docs(self):
        """Should extract 'docs' type."""
        self.assertEqual(extract_commit_type("docs: Update README"), "docs")

    def test_extract_commit_type_merge(self):
        """Should extract 'merge' type."""
        self.assertEqual(extract_commit_type("Merge pull request #123"), "merge")
        self.assertEqual(extract_commit_type("Merge branch 'main'"), "merge")

    def test_extract_commit_type_task(self):
        """Should extract 'task' type."""
        self.assertEqual(extract_commit_type("Task #42: Implement feature"), "task")
        self.assertEqual(extract_commit_type("task 123: Fix issue"), "task")

    def test_extract_commit_type_add(self):
        """Should extract 'add' type."""
        self.assertEqual(extract_commit_type("Add new module"), "add")
        self.assertEqual(extract_commit_type("add tests"), "add")

    def test_extract_commit_type_none(self):
        """Should return None for unknown types."""
        self.assertIsNone(extract_commit_type("Random commit message"))

    def test_extract_keywords_modules(self):
        """Should extract module keywords."""
        keywords = extract_keywords("Add test coverage for API")
        self.assertIn("test", keywords)
        self.assertIn("api", keywords)

    def test_extract_keywords_task_reference(self):
        """Should extract task references."""
        keywords = extract_keywords("Task #42: Fix bug")
        self.assertIn("task_42", keywords)

    def test_extract_keywords_actions(self):
        """Should extract action verbs."""
        keywords = extract_keywords("Add new feature and fix bug")
        self.assertIn("action_add", keywords)
        self.assertIn("action_fix", keywords)

    def test_extract_file_keywords(self):
        """Should extract keywords from file paths."""
        files = ["tests/test_api.py", "docs/README.md", "cortical/analysis.py"]
        keywords = extract_file_keywords(files)

        self.assertIn("test", keywords)
        self.assertIn("documentation", keywords)
        self.assertIn("analysis", keywords)

    def test_message_to_keywords(self):
        """Should tokenize message into keywords."""
        keywords = message_to_keywords("Add authentication to the API endpoint")

        self.assertIn("authentication", keywords)
        self.assertIn("api", keywords)
        self.assertIn("endpoint", keywords)
        # Stop words should be filtered
        self.assertNotIn("the", keywords)
        self.assertNotIn("to", keywords)

    def test_message_to_keywords_filters_short(self):
        """Should filter short words."""
        keywords = message_to_keywords("Add a new API")

        self.assertNotIn("a", keywords)
        self.assertIn("api", keywords)


class TestTrainingExample(unittest.TestCase):
    """Tests for TrainingExample dataclass."""

    def test_create_example(self):
        """Should create training example."""
        example = TrainingExample(
            commit_hash="abc123",
            message="feat: Add feature",
            files_changed=["file1.py", "file2.py"],
            commit_type="feat",
            keywords=["action_add"],
            timestamp="2025-01-01T00:00:00",
            insertions=10,
            deletions=5
        )

        self.assertEqual(example.commit_hash, "abc123")
        self.assertEqual(example.message, "feat: Add feature")
        self.assertEqual(len(example.files_changed), 2)
        self.assertEqual(example.commit_type, "feat")


class TestFilePredictionModel(unittest.TestCase):
    """Tests for FilePredictionModel dataclass."""

    def test_to_dict(self):
        """Should serialize to dict."""
        model = FilePredictionModel(
            file_cooccurrence={"a.py": {"b.py": 5}},
            type_to_files={"feat": {"a.py": 3}},
            keyword_to_files={"test": {"tests/test.py": 2}},
            file_frequency={"a.py": 10},
            total_commits=100,
            trained_at="2025-01-01",
            version="1.0.0"
        )

        d = model.to_dict()

        self.assertEqual(d["total_commits"], 100)
        self.assertEqual(d["version"], "1.0.0")
        self.assertIn("a.py", d["file_cooccurrence"])

    def test_from_dict(self):
        """Should deserialize from dict."""
        d = {
            "file_cooccurrence": {"a.py": {"b.py": 5}},
            "type_to_files": {"feat": {"a.py": 3}},
            "keyword_to_files": {"test": {"tests/test.py": 2}},
            "file_frequency": {"a.py": 10},
            "total_commits": 100,
            "trained_at": "2025-01-01",
            "version": "1.0.0"
        }

        model = FilePredictionModel.from_dict(d)

        self.assertEqual(model.total_commits, 100)
        self.assertEqual(model.file_frequency["a.py"], 10)


class TestModelTraining(unittest.TestCase):
    """Tests for model training."""

    def setUp(self):
        """Create test training examples."""
        self.examples = [
            TrainingExample(
                commit_hash="abc1",
                message="feat: Add authentication",
                files_changed=["auth.py", "tests/test_auth.py"],
                commit_type="feat",
                keywords=["action_add"],
                timestamp="2025-01-01",
                insertions=100,
                deletions=0
            ),
            TrainingExample(
                commit_hash="abc2",
                message="fix: Fix login bug",
                files_changed=["auth.py", "login.py"],
                commit_type="fix",
                keywords=["action_fix"],
                timestamp="2025-01-02",
                insertions=10,
                deletions=5
            ),
            TrainingExample(
                commit_hash="abc3",
                message="docs: Update README",
                files_changed=["README.md"],
                commit_type="docs",
                keywords=[],
                timestamp="2025-01-03",
                insertions=20,
                deletions=10
            ),
        ]

    def test_train_model_basic(self):
        """Should train model from examples."""
        model = train_model(self.examples)

        self.assertEqual(model.total_commits, 3)
        self.assertIn("auth.py", model.file_frequency)
        self.assertEqual(model.file_frequency["auth.py"], 2)

    def test_train_model_cooccurrence(self):
        """Should build co-occurrence matrix."""
        model = train_model(self.examples)

        # auth.py and tests/test_auth.py should co-occur
        self.assertIn("tests/test_auth.py", model.file_cooccurrence.get("auth.py", {}))

    def test_train_model_type_mapping(self):
        """Should map commit types to files."""
        model = train_model(self.examples)

        self.assertIn("feat", model.type_to_files)
        self.assertIn("auth.py", model.type_to_files["feat"])

    def test_train_model_keyword_mapping(self):
        """Should map keywords to files."""
        model = train_model(self.examples)

        # Keywords from message should be mapped
        self.assertIn("authentication", model.keyword_to_files)

    def test_train_model_empty(self):
        """Should handle empty examples."""
        model = train_model([])

        self.assertEqual(model.total_commits, 0)
        self.assertEqual(len(model.file_frequency), 0)


class TestPrediction(unittest.TestCase):
    """Tests for file prediction."""

    def setUp(self):
        """Create trained model for testing."""
        self.model = FilePredictionModel(
            file_cooccurrence={
                "auth.py": {"tests/test_auth.py": 5, "login.py": 3},
                "tests/test_auth.py": {"auth.py": 5},
                "login.py": {"auth.py": 3}
            },
            type_to_files={
                "feat": {"auth.py": 5, "api.py": 3},
                "fix": {"auth.py": 2, "login.py": 4},
                "docs": {"README.md": 10}
            },
            keyword_to_files={
                "authentication": {"auth.py": 8, "login.py": 3},
                "api": {"api.py": 5, "routes.py": 2},
                "test": {"tests/test_auth.py": 4}
            },
            file_frequency={
                "auth.py": 10,
                "login.py": 5,
                "api.py": 8,
                "tests/test_auth.py": 4,
                "README.md": 15,
                "routes.py": 3
            },
            total_commits=50,
            trained_at="2025-01-01",
            version="1.0.0"
        )

    @patch('ml_file_prediction.Path.exists', return_value=True)
    def test_predict_by_type(self, mock_exists):
        """Should predict based on commit type."""
        predictions = predict_files("feat: Add new feature", self.model, top_n=5, use_ai_meta=False)

        # 'feat' type should boost auth.py and api.py
        file_names = [f for f, _ in predictions]
        self.assertIn("auth.py", file_names[:3])

    @patch('ml_file_prediction.Path.exists', return_value=True)
    def test_predict_by_keyword(self, mock_exists):
        """Should predict based on keywords."""
        predictions = predict_files("Add authentication system", self.model, top_n=5, use_ai_meta=False)

        # 'authentication' keyword should boost auth.py
        file_names = [f for f, _ in predictions]
        self.assertIn("auth.py", file_names[:3])

    @patch('ml_file_prediction.Path.exists', return_value=True)
    def test_predict_with_seed_files(self, mock_exists):
        """Should boost co-occurring files with seeds."""
        predictions = predict_files(
            "Update related files",
            self.model,
            top_n=5,
            seed_files=["auth.py"],
            use_ai_meta=False
        )

        # Files co-occurring with auth.py should be boosted
        file_names = [f for f, _ in predictions]
        self.assertIn("tests/test_auth.py", file_names[:3])

    def test_predict_returns_scores(self):
        """Should return files with scores."""
        # Use a query that matches known keywords
        predictions = predict_files("feat: Add authentication test", self.model, top_n=3)

        self.assertGreater(len(predictions), 0)
        for filename, score in predictions:
            self.assertIsInstance(filename, str)
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0)

    def test_predict_top_n(self):
        """Should limit results to top_n."""
        # Use a query that matches known keywords
        predictions = predict_files("feat: Add authentication", self.model, top_n=2)

        self.assertLessEqual(len(predictions), 2)


class TestEvaluation(unittest.TestCase):
    """Tests for model evaluation."""

    def setUp(self):
        """Create model and test examples."""
        self.model = FilePredictionModel(
            file_cooccurrence={},
            type_to_files={
                "feat": {"a.py": 5, "b.py": 3, "c.py": 2}
            },
            keyword_to_files={
                "test": {"a.py": 4, "d.py": 2}
            },
            file_frequency={
                "a.py": 10, "b.py": 5, "c.py": 3, "d.py": 2
            },
            total_commits=20,
            trained_at="2025-01-01",
            version="1.0.0"
        )

        self.test_examples = [
            TrainingExample(
                commit_hash="t1",
                message="feat: Add feature",
                files_changed=["a.py", "b.py"],
                commit_type="feat",
                keywords=[],
                timestamp="2025-01-01",
                insertions=10,
                deletions=0
            ),
            TrainingExample(
                commit_hash="t2",
                message="Add test coverage",
                files_changed=["a.py", "d.py"],
                commit_type=None,
                keywords=["test"],
                timestamp="2025-01-02",
                insertions=20,
                deletions=5
            ),
        ]

    def test_evaluate_returns_metrics(self):
        """Should return evaluation metrics."""
        results = evaluate_model(self.model, self.test_examples)

        self.assertIn("recall@1", results)
        self.assertIn("recall@5", results)
        self.assertIn("precision@1", results)
        self.assertIn("mrr", results)
        self.assertIn("total_examples", results)

    def test_evaluate_metrics_range(self):
        """Metrics should be in valid range [0, 1]."""
        results = evaluate_model(self.model, self.test_examples)

        for key, value in results.items():
            if key != "total_examples":
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_evaluate_empty(self):
        """Should handle empty test set."""
        results = evaluate_model(self.model, [])

        self.assertEqual(results["total_examples"], 0)


class TestTrainTestSplit(unittest.TestCase):
    """Tests for train/test splitting."""

    def test_split_ratio(self):
        """Should split according to ratio."""
        examples = [TrainingExample(
            commit_hash=f"h{i}",
            message=f"msg{i}",
            files_changed=[f"f{i}.py"],
            commit_type=None,
            keywords=[],
            timestamp="2025-01-01",
            insertions=1,
            deletions=0
        ) for i in range(100)]

        train, test = train_test_split(examples, test_ratio=0.2, shuffle=False)

        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)

    def test_split_shuffle(self):
        """Should shuffle when requested."""
        examples = [TrainingExample(
            commit_hash=f"h{i}",
            message=f"msg{i}",
            files_changed=[f"f{i}.py"],
            commit_type=None,
            keywords=[],
            timestamp="2025-01-01",
            insertions=1,
            deletions=0
        ) for i in range(100)]

        train1, _ = train_test_split(examples, test_ratio=0.2, shuffle=True)
        train2, _ = train_test_split(examples, test_ratio=0.2, shuffle=True)

        # Different shuffles should produce different orders
        # (with very high probability for 100 items)
        hashes1 = [e.commit_hash for e in train1]
        hashes2 = [e.commit_hash for e in train2]
        self.assertNotEqual(hashes1, hashes2)


class TestPersistence(unittest.TestCase):
    """Tests for model save/load."""

    def test_save_load_roundtrip(self):
        """Should preserve model through save/load."""
        model = FilePredictionModel(
            file_cooccurrence={"a.py": {"b.py": 5}},
            type_to_files={"feat": {"a.py": 3}},
            keyword_to_files={"test": {"t.py": 2}},
            file_frequency={"a.py": 10},
            total_commits=100,
            trained_at="2025-01-01",
            version="1.0.0"
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_model(model, path)
            loaded = load_model(path)

            self.assertEqual(loaded.total_commits, model.total_commits)
            self.assertEqual(loaded.version, model.version)
            self.assertEqual(loaded.file_frequency, model.file_frequency)
        finally:
            path.unlink()

    def test_load_nonexistent(self):
        """Should return None for missing file."""
        result = load_model(Path("/nonexistent/model.json"))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
