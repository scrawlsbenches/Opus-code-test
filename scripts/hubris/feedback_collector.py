#!/usr/bin/env python3
"""
Feedback Collector for Expert Credit Updates

Wires expert predictions to git hooks to enable real-time credit updates
based on commit and test outcomes. This closes the feedback loop between
expert predictions and their actual value.

Workflow:
1. Pre-commit: Expert makes prediction -> PredictionRecorder stores it
2. Post-commit: Actual files committed -> FeedbackProcessor evaluates accuracy
3. Credit update: ValueAttributor converts accuracy to credit/debit amounts
4. Ledger update: CreditLedger applies transactions to expert accounts

Storage:
- Pending predictions: .git-ml/predictions/pending.jsonl
- Resolved predictions: .git-ml/predictions/resolved.jsonl
- Uses atomic writes (temp + rename) for consistency

Classes:
    PredictionRecorder: Records predictions before commits
    FeedbackProcessor: Evaluates outcomes and updates credits
"""

import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

import sys
sys.path.insert(0, str(Path(__file__).parent))

from credit_account import CreditLedger
from value_signal import ValueAttributor, ValueSignal, SignalType


@dataclass
class Prediction:
    """
    Recorded prediction awaiting evaluation.

    Attributes:
        prediction_id: Unique identifier for this prediction
        expert_id: Expert that made the prediction
        predicted_files: List of files expert predicted would be modified
        confidence: Expert's confidence in prediction (0-1)
        timestamp: When prediction was made
        context: Additional metadata (task description, etc.)
    """
    prediction_id: str
    expert_id: str
    predicted_files: List[str]
    confidence: float
    timestamp: float
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Prediction':
        """Load from dict."""
        return cls(**data)


@dataclass
class ResolvedPrediction(Prediction):
    """
    Prediction with outcome data for calibration tracking.

    Extends Prediction with fields populated after evaluation:
    - actual_files: What files were actually modified
    - accuracy: Fraction of predicted files that were correct
    - outcome_timestamp: When the outcome was evaluated
    - commit_hash: Git commit that resolved this prediction

    These fields enable calibration analysis: comparing predicted
    confidence against actual accuracy to detect over/under-confidence.
    """
    actual_files: List[str] = None
    accuracy: float = 0.0
    outcome_timestamp: float = 0.0
    commit_hash: str = ""

    def __post_init__(self):
        """Initialize mutable default."""
        if self.actual_files is None:
            self.actual_files = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict including outcome fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResolvedPrediction':
        """Load from dict, handling both old and new formats."""
        # Handle old format without outcome fields
        if 'actual_files' not in data:
            data['actual_files'] = []
        if 'accuracy' not in data:
            data['accuracy'] = 0.0
        if 'outcome_timestamp' not in data:
            data['outcome_timestamp'] = 0.0
        if 'commit_hash' not in data:
            data['commit_hash'] = ""
        return cls(**data)

    @classmethod
    def from_prediction(
        cls,
        prediction: Prediction,
        actual_files: List[str],
        commit_hash: str
    ) -> 'ResolvedPrediction':
        """
        Create ResolvedPrediction from Prediction with outcome data.

        Args:
            prediction: Original prediction
            actual_files: Files actually modified in commit
            commit_hash: Git commit hash

        Returns:
            ResolvedPrediction with accuracy calculated
        """
        predicted_set = set(prediction.predicted_files)
        actual_set = set(actual_files)

        # Calculate accuracy: intersection / predicted
        if predicted_set:
            correct = len(predicted_set & actual_set)
            accuracy = correct / len(predicted_set)
        else:
            accuracy = 0.0

        return cls(
            prediction_id=prediction.prediction_id,
            expert_id=prediction.expert_id,
            predicted_files=prediction.predicted_files,
            confidence=prediction.confidence,
            timestamp=prediction.timestamp,
            context=prediction.context,
            actual_files=actual_files,
            accuracy=accuracy,
            outcome_timestamp=time.time(),
            commit_hash=commit_hash
        )


class PredictionRecorder:
    """
    Records expert predictions for later evaluation.

    Stores predictions to .git-ml/predictions/pending.jsonl in JSONL format
    (one prediction per line). Uses atomic writes to prevent corruption.

    Attributes:
        predictions_dir: Directory containing prediction files
        pending_path: Path to pending predictions file
        resolved_path: Path to resolved predictions file
    """

    def __init__(self, predictions_dir: Optional[Path] = None):
        """
        Initialize prediction recorder.

        Args:
            predictions_dir: Directory for prediction files (default: .git-ml/predictions)
        """
        if predictions_dir is None:
            # Default to .git-ml/predictions relative to project root
            git_ml_dir = Path(__file__).parent.parent.parent / '.git-ml'
            predictions_dir = git_ml_dir / 'predictions'

        self.predictions_dir = Path(predictions_dir)
        self.pending_path = self.predictions_dir / 'pending.jsonl'
        self.resolved_path = self.predictions_dir / 'resolved.jsonl'

        # Create directory if it doesn't exist
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def record_prediction(
        self,
        prediction_id: str,
        expert_id: str,
        predicted_files: List[str],
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Prediction:
        """
        Record a prediction for later evaluation.

        Args:
            prediction_id: Unique identifier for this prediction
            expert_id: Expert making the prediction
            predicted_files: Files predicted to be modified
            confidence: Prediction confidence (0-1)
            context: Additional metadata (optional)

        Returns:
            Prediction object that was recorded

        Raises:
            ValueError: If confidence not in [0, 1]
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {confidence}")

        prediction = Prediction(
            prediction_id=prediction_id,
            expert_id=expert_id,
            predicted_files=predicted_files,
            confidence=confidence,
            timestamp=time.time(),
            context=context or {}
        )

        # Atomic append to pending file
        self._append_jsonl(self.pending_path, prediction.to_dict())

        return prediction

    def get_pending_predictions(self, expert_id: Optional[str] = None) -> List[Prediction]:
        """
        Get predictions awaiting feedback.

        Args:
            expert_id: Optional filter by expert ID

        Returns:
            List of pending predictions
        """
        if not self.pending_path.exists():
            return []

        predictions = []
        with open(self.pending_path) as f:
            for line in f:
                if line.strip():
                    pred = Prediction.from_dict(json.loads(line))
                    if expert_id is None or pred.expert_id == expert_id:
                        predictions.append(pred)

        return predictions

    def get_prediction_by_id(self, prediction_id: str) -> Optional[Prediction]:
        """
        Get a specific prediction by ID.

        Args:
            prediction_id: Prediction ID to find

        Returns:
            Prediction or None if not found
        """
        predictions = self.get_pending_predictions()
        for pred in predictions:
            if pred.prediction_id == prediction_id:
                return pred
        return None

    def resolve_prediction(
        self,
        prediction_id: str,
        actual_files: Optional[List[str]] = None,
        commit_hash: str = ""
    ) -> bool:
        """
        Move prediction from pending to resolved with outcome data.

        Args:
            prediction_id: Prediction ID to resolve
            actual_files: Files actually modified (for calibration tracking)
            commit_hash: Git commit hash that resolved this prediction

        Returns:
            True if prediction was found and resolved, False otherwise

        Note:
            If actual_files is provided, creates a ResolvedPrediction with
            calibration data (accuracy, outcome_timestamp). Otherwise,
            stores basic prediction for backward compatibility.
        """
        if not self.pending_path.exists():
            return False

        # Read all pending predictions
        pending = []
        resolved_pred = None

        with open(self.pending_path) as f:
            for line in f:
                if line.strip():
                    pred_dict = json.loads(line)
                    pred = Prediction.from_dict(pred_dict)

                    if pred.prediction_id == prediction_id:
                        resolved_pred = pred
                    else:
                        pending.append(pred)

        if resolved_pred is None:
            return False

        # Atomic rewrite of pending file (without the resolved prediction)
        self._write_jsonl(self.pending_path, [p.to_dict() for p in pending])

        # Create ResolvedPrediction with outcome data if available
        if actual_files is not None:
            resolved_with_outcome = ResolvedPrediction.from_prediction(
                resolved_pred,
                actual_files=actual_files,
                commit_hash=commit_hash
            )
            self._append_jsonl(self.resolved_path, resolved_with_outcome.to_dict())
        else:
            # Backward compatibility: store without outcome data
            self._append_jsonl(self.resolved_path, resolved_pred.to_dict())

        return True

    def _append_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        """
        Atomically append JSON line to file.

        Args:
            path: File to append to
            data: Dictionary to serialize as JSON
        """
        # Write to temp file first
        fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix='.tmp_',
            suffix='.jsonl'
        )

        try:
            # If file exists, copy it first
            if path.exists():
                with open(path) as src:
                    with os.fdopen(fd, 'w') as dst:
                        dst.write(src.read())
                        dst.write(json.dumps(data) + '\n')
            else:
                with os.fdopen(fd, 'w') as dst:
                    dst.write(json.dumps(data) + '\n')

            # Atomic rename
            os.replace(temp_path, path)
        except:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise

    def _write_jsonl(self, path: Path, data_list: List[Dict[str, Any]]) -> None:
        """
        Atomically write JSON lines to file.

        Args:
            path: File to write to
            data_list: List of dictionaries to serialize
        """
        # Write to temp file
        fd, temp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix='.tmp_',
            suffix='.jsonl'
        )

        try:
            with os.fdopen(fd, 'w') as f:
                for data in data_list:
                    f.write(json.dumps(data) + '\n')

            # Atomic rename
            os.replace(temp_path, path)
        except:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise


class FeedbackProcessor:
    """
    Processes commit/test outcomes and updates expert credits.

    Evaluates prediction accuracy by comparing predicted files to actual
    files, then uses ValueAttributor to convert accuracy into credit amounts.

    Attributes:
        ledger: CreditLedger to update
        attributor: ValueAttributor for credit calculation
        recorder: PredictionRecorder for accessing predictions
    """

    def __init__(
        self,
        ledger: CreditLedger,
        attributor: Optional[ValueAttributor] = None,
        recorder: Optional[PredictionRecorder] = None
    ):
        """
        Initialize feedback processor.

        Args:
            ledger: CreditLedger to update
            attributor: ValueAttributor (creates default if None)
            recorder: PredictionRecorder (creates default if None)
        """
        self.ledger = ledger
        self.attributor = attributor if attributor is not None else ValueAttributor()
        self.recorder = recorder if recorder is not None else PredictionRecorder()

    def process_commit_outcome(
        self,
        commit_hash: str,
        actual_files: List[str],
        prediction_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compare predictions to actual committed files and update credits.

        Evaluates each pending prediction against the actual files committed,
        generates value signals, and updates expert credit accounts.

        Args:
            commit_hash: Git commit hash
            actual_files: Files that were actually modified in the commit
            prediction_id: Optional specific prediction to evaluate (default: all pending)

        Returns:
            Dictionary mapping expert_id -> credit/debit amount
        """
        # Get predictions to evaluate
        if prediction_id:
            prediction = self.recorder.get_prediction_by_id(prediction_id)
            predictions = [prediction] if prediction else []
        else:
            predictions = self.recorder.get_pending_predictions()

        if not predictions:
            return {}

        credit_updates = {}

        for prediction in predictions:
            # Calculate which files were correctly predicted
            predicted_set = set(prediction.predicted_files)
            actual_set = set(actual_files)

            files_correct = list(predicted_set & actual_set)

            # Generate value signal from commit result
            signal = self.attributor.attribute_from_commit_result(
                expert_id=prediction.expert_id,
                prediction_id=prediction.prediction_id,
                files_correct=files_correct,
                files_total=prediction.predicted_files,
                confidence=prediction.confidence,
                prediction_time=prediction.timestamp
            )

            # Process signal to update credits
            amount = self.attributor.process_signal(signal, self.ledger)
            credit_updates[prediction.expert_id] = credit_updates.get(
                prediction.expert_id, 0.0
            ) + amount

            # Move prediction to resolved with outcome data for calibration
            self.recorder.resolve_prediction(
                prediction.prediction_id,
                actual_files=actual_files,
                commit_hash=commit_hash
            )

        return credit_updates

    def process_test_outcome(
        self,
        test_results: Dict[str, bool],
        prediction_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Update credits based on test pass/fail predictions.

        Args:
            test_results: Dictionary mapping test_name -> passed (bool)
            prediction_id: Optional specific prediction to evaluate

        Returns:
            Dictionary mapping expert_id -> credit/debit amount
        """
        # Get predictions to evaluate
        if prediction_id:
            prediction = self.recorder.get_prediction_by_id(prediction_id)
            predictions = [prediction] if prediction else []
        else:
            predictions = self.recorder.get_pending_predictions()

        if not predictions:
            return {}

        credit_updates = {}

        # Determine overall test outcome
        all_passed = all(test_results.values())

        for prediction in predictions:
            # Generate value signal from test result
            signal = self.attributor.attribute_from_test_result(
                expert_id=prediction.expert_id,
                prediction_id=prediction.prediction_id,
                test_passed=all_passed,
                confidence=prediction.confidence,
                prediction_time=prediction.timestamp
            )

            # Process signal to update credits
            amount = self.attributor.process_signal(signal, self.ledger)
            credit_updates[prediction.expert_id] = credit_updates.get(
                prediction.expert_id, 0.0
            ) + amount

        return credit_updates


# Hook integration functions

def on_pre_commit(
    task_description: str,
    expert_id: str = "default_expert",
    predicted_files: Optional[List[str]] = None,
    confidence: float = 0.7,
    recorder: Optional[PredictionRecorder] = None
) -> str:
    """
    Called before commit - records expert prediction.

    This should be called from git prepare-commit-msg hook or similar.

    Args:
        task_description: Description of what's being committed
        expert_id: Expert making the prediction
        predicted_files: Files predicted to be modified
        confidence: Prediction confidence (0-1)
        recorder: Optional PredictionRecorder instance

    Returns:
        Prediction ID for later evaluation
    """
    if recorder is None:
        recorder = PredictionRecorder()

    # Generate unique prediction ID
    prediction_id = f"pred_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    # Record prediction
    recorder.record_prediction(
        prediction_id=prediction_id,
        expert_id=expert_id,
        predicted_files=predicted_files or [],
        confidence=confidence,
        context={
            'task_description': task_description,
            'timestamp': time.time()
        }
    )

    return prediction_id


def on_post_commit(
    commit_hash: str,
    actual_files: List[str],
    prediction_id: Optional[str] = None,
    ledger: Optional[CreditLedger] = None,
    attributor: Optional[ValueAttributor] = None,
    recorder: Optional[PredictionRecorder] = None
) -> Dict[str, float]:
    """
    Called after commit - evaluates predictions and updates credits.

    This should be called from git post-commit hook.

    Args:
        commit_hash: Git commit hash
        actual_files: Files that were actually modified
        prediction_id: Optional specific prediction to evaluate
        ledger: Optional CreditLedger instance
        attributor: Optional ValueAttributor instance
        recorder: Optional PredictionRecorder instance

    Returns:
        Dictionary mapping expert_id -> credit/debit amount
    """
    if ledger is None:
        ledger = CreditLedger()

    processor = FeedbackProcessor(
        ledger=ledger,
        attributor=attributor,
        recorder=recorder
    )

    return processor.process_commit_outcome(
        commit_hash=commit_hash,
        actual_files=actual_files,
        prediction_id=prediction_id
    )


if __name__ == '__main__':
    # Demo usage
    print("Feedback Collector Demo")
    print("=" * 60)

    # Create components
    recorder = PredictionRecorder()
    ledger = CreditLedger()
    attributor = ValueAttributor()
    processor = FeedbackProcessor(ledger, attributor, recorder)

    # Simulate pre-commit hook
    print("\n1. Pre-commit: Expert makes prediction")
    prediction_id = on_pre_commit(
        task_description="Add authentication feature",
        expert_id="file_expert",
        predicted_files=[
            "cortical/auth.py",
            "tests/test_auth.py",
            "docs/api.md"
        ],
        confidence=0.8,
        recorder=recorder
    )
    print(f"   Recorded prediction: {prediction_id}")

    # Check pending predictions
    pending = recorder.get_pending_predictions()
    print(f"   Pending predictions: {len(pending)}")

    # Simulate post-commit hook
    print("\n2. Post-commit: Evaluate prediction accuracy")
    actual_files = [
        "cortical/auth.py",
        "tests/test_auth.py"
    ]

    credit_updates = on_post_commit(
        commit_hash="abc123",
        actual_files=actual_files,
        prediction_id=prediction_id,
        ledger=ledger,
        attributor=attributor,
        recorder=recorder
    )

    print(f"   Credit updates: {credit_updates}")

    # Check expert balance
    account = ledger.get_or_create_account("file_expert")
    print(f"   Expert balance: {account.balance:.2f}")

    # Show transactions
    print("\n3. Recent transactions:")
    for tx in account.get_recent_transactions(5):
        print(f"   {tx.reason}: {tx.amount:+.2f} (balance: {tx.balance_after:.2f})")

    # Check pending predictions (should be resolved now)
    pending = recorder.get_pending_predictions()
    print(f"\n4. Pending predictions remaining: {len(pending)}")
