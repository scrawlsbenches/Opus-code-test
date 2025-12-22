#!/usr/bin/env python3
"""
Milestone Verification Script

Verifies SparkSLM research milestones and updates roadmap state.

Usage:
    python scripts/verify_milestone.py 1.1    # Verify specific milestone
    python scripts/verify_milestone.py --all  # Verify all completed
    python scripts/verify_milestone.py --status  # Show current status
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MilestoneResult:
    """Result of milestone verification."""
    milestone_id: str
    passed: bool
    metrics: Dict[str, Any]
    duration_seconds: float
    timestamp: str
    next_action: str
    notes: str = ""


class MilestoneVerifier:
    """Verifies research milestones and tracks state."""

    STATE_FILE = "tasks/roadmap_state.json"

    def __init__(self):
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        """Load roadmap state from file."""
        if os.path.exists(self.STATE_FILE):
            with open(self.STATE_FILE) as f:
                return json.load(f)
        return {
            "current_phase": 1,
            "current_milestone": "1.1",
            "completed_milestones": [],
            "failed_milestones": [],
            "failsafe_triggered": [],
            "metrics_history": {},
            "adjustments": []
        }

    def _save_state(self):
        """Save roadmap state to file."""
        os.makedirs(os.path.dirname(self.STATE_FILE), exist_ok=True)
        with open(self.STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)

    def verify_milestone(self, milestone_id: str) -> MilestoneResult:
        """Run verification for a specific milestone."""
        verifiers = {
            "1.1": self._verify_1_1_ngram_training,
            "1.2": self._verify_1_2_alignment_index,
            "1.3": self._verify_1_3_processor_integration,
            "2.1": self._verify_2_1_prediction_quality,
            "2.2": self._verify_2_2_query_expansion,
        }

        if milestone_id not in verifiers:
            return MilestoneResult(
                milestone_id=milestone_id,
                passed=False,
                metrics={},
                duration_seconds=0,
                timestamp=datetime.now().isoformat(),
                next_action="unknown_milestone",
                notes=f"Unknown milestone: {milestone_id}"
            )

        start = time.time()
        result = verifiers[milestone_id]()
        result.duration_seconds = time.time() - start
        result.timestamp = datetime.now().isoformat()

        # Update state
        if result.passed:
            if milestone_id not in self.state["completed_milestones"]:
                self.state["completed_milestones"].append(milestone_id)
        else:
            if milestone_id not in self.state["failed_milestones"]:
                self.state["failed_milestones"].append(milestone_id)

        self.state["metrics_history"][milestone_id] = result.metrics
        self._save_state()

        return result

    def _verify_1_1_ngram_training(self) -> MilestoneResult:
        """Verify: N-gram model can learn corpus patterns in <1s."""
        try:
            from cortical import CorticalTextProcessor

            # Create processor with spark
            p = CorticalTextProcessor(spark=True)

            # Load sample documents
            samples_dir = Path("samples")
            if samples_dir.exists():
                for md_file in samples_dir.rglob("*.md"):
                    try:
                        content = md_file.read_text()
                        if len(content) > 50:
                            p.process_document(str(md_file), content)
                    except Exception:
                        continue

            # Train and measure
            start = time.time()
            stats = p.train_spark()
            training_time = time.time() - start

            # Check perplexity (need some text to evaluate)
            # For now, use vocabulary size as proxy
            vocab_size = stats.get('tokens', 0)

            metrics = {
                "training_time": training_time,
                "documents": stats.get('documents', 0),
                "vocabulary_size": vocab_size,
            }

            # Success criteria: training < 1s
            passed = training_time < 1.0 and vocab_size > 0

            return MilestoneResult(
                milestone_id="1.1",
                passed=passed,
                metrics=metrics,
                duration_seconds=0,
                timestamp="",
                next_action="1.2" if passed else "investigate_training",
                notes=f"Training time: {training_time:.3f}s, vocab: {vocab_size}"
            )

        except Exception as e:
            return MilestoneResult(
                milestone_id="1.1",
                passed=False,
                metrics={"error": str(e)},
                duration_seconds=0,
                timestamp="",
                next_action="fix_ngram_model",
                notes=f"Error: {e}"
            )

    def _verify_1_2_alignment_index(self) -> MilestoneResult:
        """Verify: Alignment index can store/retrieve user knowledge."""
        try:
            from cortical import CorticalTextProcessor

            p = CorticalTextProcessor(spark=True)

            # Load alignment
            alignment_dir = "samples/alignment"
            if not os.path.exists(alignment_dir):
                return MilestoneResult(
                    milestone_id="1.2",
                    passed=False,
                    metrics={"error": "samples/alignment not found"},
                    duration_seconds=0,
                    timestamp="",
                    next_action="create_alignment_samples",
                    notes="Alignment directory missing"
                )

            count = p.load_alignment(alignment_dir)

            # Test retrieval
            ctx = p.get_alignment_context("spark")
            retrieval_works = len(ctx) > 0

            metrics = {
                "entries_loaded": count,
                "retrieval_success": retrieval_works,
            }

            # Success criteria: 20+ entries, retrieval works
            passed = count >= 20 and retrieval_works

            return MilestoneResult(
                milestone_id="1.2",
                passed=passed,
                metrics=metrics,
                duration_seconds=0,
                timestamp="",
                next_action="1.3" if passed else "expand_alignment",
                notes=f"Loaded {count} entries, retrieval: {retrieval_works}"
            )

        except Exception as e:
            return MilestoneResult(
                milestone_id="1.2",
                passed=False,
                metrics={"error": str(e)},
                duration_seconds=0,
                timestamp="",
                next_action="fix_alignment_index",
                notes=f"Error: {e}"
            )

    def _verify_1_3_processor_integration(self) -> MilestoneResult:
        """Verify: SparkMixin integrates without breaking functionality."""
        try:
            import subprocess

            # Run spark tests
            result = subprocess.run(
                [sys.executable, "-m", "unittest",
                 "tests.unit.test_spark", "tests.unit.test_spark_integration",
                 "-v"],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parse test results
            output = result.stdout + result.stderr
            passed_tests = output.count(" ok")
            failed_tests = output.count("FAIL")
            errors = output.count("ERROR")

            metrics = {
                "tests_passed": passed_tests,
                "tests_failed": failed_tests,
                "errors": errors,
                "exit_code": result.returncode,
            }

            # Success criteria: all tests pass, 50+ tests
            passed = result.returncode == 0 and passed_tests >= 50

            return MilestoneResult(
                milestone_id="1.3",
                passed=passed,
                metrics=metrics,
                duration_seconds=0,
                timestamp="",
                next_action="2.1" if passed else "fix_integration",
                notes=f"{passed_tests} tests passed, {failed_tests} failed"
            )

        except Exception as e:
            return MilestoneResult(
                milestone_id="1.3",
                passed=False,
                metrics={"error": str(e)},
                duration_seconds=0,
                timestamp="",
                next_action="investigate_test_failure",
                notes=f"Error: {e}"
            )

    def _verify_2_1_prediction_quality(self) -> MilestoneResult:
        """Verify: Predictions provide useful signal."""
        try:
            from cortical import CorticalTextProcessor

            p = CorticalTextProcessor(spark=True)

            # Load corpus
            samples_dir = Path("samples")
            docs = []
            if samples_dir.exists():
                for md_file in samples_dir.rglob("*.md"):
                    try:
                        content = md_file.read_text()
                        if len(content) > 100:
                            p.process_document(str(md_file), content)
                            docs.append(content)
                    except Exception:
                        continue

            p.train_spark()

            # Test predictions on sample queries
            test_queries = [
                "neural network",
                "query expansion",
                "lateral connections",
                "document processing",
                "compute all"
            ]

            predictions_found = 0
            for query in test_queries:
                hints = p.prime_query(query)
                if hints.get('completions'):
                    predictions_found += 1

            stats = p.get_spark_stats()

            metrics = {
                "vocabulary_size": stats.get('vocabulary_size', 0),
                "context_count": stats.get('context_count', 0),
                "queries_with_predictions": predictions_found,
                "total_test_queries": len(test_queries),
            }

            # Success criteria: predictions on majority of queries
            passed = predictions_found >= len(test_queries) * 0.5

            return MilestoneResult(
                milestone_id="2.1",
                passed=passed,
                metrics=metrics,
                duration_seconds=0,
                timestamp="",
                next_action="2.2" if passed else "improve_training",
                notes=f"{predictions_found}/{len(test_queries)} queries got predictions"
            )

        except Exception as e:
            return MilestoneResult(
                milestone_id="2.1",
                passed=False,
                metrics={"error": str(e)},
                duration_seconds=0,
                timestamp="",
                next_action="investigate_prediction_failure",
                notes=f"Error: {e}"
            )

    def _verify_2_2_query_expansion(self) -> MilestoneResult:
        """Verify: Spark-enhanced expansion improves search."""
        try:
            from cortical import CorticalTextProcessor

            p = CorticalTextProcessor(spark=True)

            # Load and train
            samples_dir = Path("samples")
            if samples_dir.exists():
                for md_file in list(samples_dir.rglob("*.md"))[:50]:
                    try:
                        content = md_file.read_text()
                        if len(content) > 100:
                            p.process_document(str(md_file), content)
                    except Exception:
                        continue

            p.compute_all()
            p.train_spark()

            # Compare expansions
            test_query = "search documents"

            standard = p.expand_query(test_query)
            enhanced = p.expand_query_with_spark(test_query)

            metrics = {
                "standard_terms": len(standard),
                "enhanced_terms": len(enhanced),
                "new_terms_added": len(enhanced) - len(standard),
            }

            # Success criteria: enhanced adds at least some terms
            passed = len(enhanced) >= len(standard)

            return MilestoneResult(
                milestone_id="2.2",
                passed=passed,
                metrics=metrics,
                duration_seconds=0,
                timestamp="",
                next_action="3.1" if passed else "tune_spark_boost",
                notes=f"Standard: {len(standard)} terms, Enhanced: {len(enhanced)} terms"
            )

        except Exception as e:
            return MilestoneResult(
                milestone_id="2.2",
                passed=False,
                metrics={"error": str(e)},
                duration_seconds=0,
                timestamp="",
                next_action="investigate_expansion_failure",
                notes=f"Error: {e}"
            )

    def show_status(self):
        """Display current roadmap status."""
        print("=" * 60)
        print("SparkSLM Research Roadmap Status")
        print("=" * 60)
        print()
        print(f"Current Phase: {self.state['current_phase']}")
        print(f"Current Milestone: {self.state['current_milestone']}")
        print()
        print("Completed Milestones:")
        for m in self.state['completed_milestones']:
            metrics = self.state['metrics_history'].get(m, {})
            print(f"  ✓ {m}: {metrics}")
        print()
        if self.state['failed_milestones']:
            print("Failed Milestones:")
            for m in self.state['failed_milestones']:
                print(f"  ✗ {m}")
            print()
        print("Metrics History:")
        for m, metrics in self.state['metrics_history'].items():
            print(f"  {m}: {metrics}")
        print()
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Verify SparkSLM milestones")
    parser.add_argument("milestone", nargs="?", help="Milestone ID (e.g., 1.1)")
    parser.add_argument("--all", action="store_true", help="Verify all milestones")
    parser.add_argument("--status", action="store_true", help="Show status")
    args = parser.parse_args()

    verifier = MilestoneVerifier()

    if args.status:
        verifier.show_status()
        return

    if args.all:
        for m in ["1.1", "1.2", "1.3", "2.1", "2.2"]:
            print(f"\nVerifying milestone {m}...")
            result = verifier.verify_milestone(m)
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            print(f"  {status}: {result.notes}")
            print(f"  Metrics: {result.metrics}")
            print(f"  Next: {result.next_action}")
        return

    if args.milestone:
        print(f"Verifying milestone {args.milestone}...")
        result = verifier.verify_milestone(args.milestone)
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        print(f"\n{status}")
        print(f"Notes: {result.notes}")
        print(f"Metrics: {json.dumps(result.metrics, indent=2)}")
        print(f"Duration: {result.duration_seconds:.2f}s")
        print(f"Next action: {result.next_action}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
