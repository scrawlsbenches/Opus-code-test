"""
Unit tests for VelocityPredictor class.

Tests cover:
- Initialization and configuration
- EMA-based velocity calculation
- Trend detection (increasing, stable, decreasing)
- Anomaly detection
- Prediction with confidence metrics
- Historical analysis
"""

import math
import pytest
from typing import Dict, Any

from llm_orchestration.agile import (
    VelocityPredictor,
    VelocityPrediction,
    VelocityAnomaly,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def predictor():
    """Create a predictor with default configuration."""
    return VelocityPredictor()


@pytest.fixture
def predictor_small_window():
    """Create a predictor with small window for testing."""
    return VelocityPredictor(history_window=5)


@pytest.fixture
def predictor_with_stable_history():
    """Create a predictor with stable velocity history."""
    pred = VelocityPredictor(history_window=10)
    # Record 10 sprints with stable velocity around 5.0
    for _ in range(10):
        pred.record_velocity(5.0)
    return pred


@pytest.fixture
def predictor_with_increasing_history():
    """Create a predictor with increasing velocity trend."""
    pred = VelocityPredictor(history_window=10)
    # Record 10 sprints with increasing velocity
    for i in range(10):
        pred.record_velocity(3.0 + i * 0.5)  # 3.0, 3.5, 4.0, ..., 7.5
    return pred


@pytest.fixture
def predictor_with_decreasing_history():
    """Create a predictor with decreasing velocity trend."""
    pred = VelocityPredictor(history_window=10)
    # Record 10 sprints with decreasing velocity
    for i in range(10):
        pred.record_velocity(8.0 - i * 0.5)  # 8.0, 7.5, 7.0, ..., 3.5
    return pred


@pytest.fixture
def predictor_with_anomaly():
    """Create a predictor with an anomalous velocity spike."""
    pred = VelocityPredictor(history_window=10)
    # Record stable velocities with one spike
    for i in range(10):
        if i == 5:
            pred.record_velocity(15.0)  # Anomaly: 3x normal
        else:
            pred.record_velocity(5.0)
    return pred


# =============================================================================
# TEST CLASS 1: INITIALIZATION
# =============================================================================


class TestVelocityPredictorInit:
    """Test VelocityPredictor initialization."""

    def test_default_initialization(self):
        """Test predictor initializes with default parameters."""
        pred = VelocityPredictor()

        assert pred._window == 10
        assert pred._alpha == 0.3
        assert pred._history == []
        assert pred._contexts == []

    def test_custom_alpha_parameter(self):
        """Test that alpha parameter is set to expected value."""
        pred = VelocityPredictor()

        # Alpha is hardcoded to 0.3 in the implementation
        assert pred._alpha == 0.3

    def test_window_size_configuration(self):
        """Test predictor respects custom window size."""
        pred = VelocityPredictor(history_window=15)

        assert pred._window == 15

        # Verify window is used for limiting history
        for i in range(50):  # Record more than 2x window
            pred.record_velocity(float(i))

        # Should keep only 2x window = 30 entries
        assert len(pred._history) == 30


# =============================================================================
# TEST CLASS 2: VELOCITY CALCULATION
# =============================================================================


class TestVelocityCalculation:
    """Test velocity calculation methods."""

    def test_ema_calculation(self):
        """Test exponential moving average calculation."""
        pred = VelocityPredictor(history_window=10)

        # Record known velocities
        velocities = [4.0, 5.0, 6.0, 5.0, 4.0]
        for v in velocities:
            pred.record_velocity(v)

        # Calculate expected EMA with alpha=0.3
        expected_ema = velocities[0]
        for v in velocities[1:]:
            expected_ema = 0.3 * v + 0.7 * expected_ema

        # Get EMA from predictor
        actual_ema = pred._exponential_moving_average()

        assert abs(actual_ema - expected_ema) < 0.001

    def test_velocity_with_no_history(self):
        """Test prediction with no historical data."""
        pred = VelocityPredictor()

        prediction = pred.predict_next()

        # Should return default with low confidence
        assert prediction.predicted_velocity == 5.0
        assert prediction.confidence == 0.0
        assert prediction.trend == "stable"
        assert "No historical data" in prediction.factors[0]

    def test_velocity_with_single_sprint(self):
        """Test prediction with only one data point."""
        pred = VelocityPredictor()
        pred.record_velocity(7.0)

        prediction = pred.predict_next()

        # Should return the single value
        assert prediction.predicted_velocity == 7.0
        assert prediction.confidence == 0.3
        assert prediction.trend == "stable"
        assert "Only one sprint" in prediction.factors[0]

        # Confidence interval should be 30% on either side
        assert prediction.confidence_interval[0] == pytest.approx(7.0 * 0.7)
        assert prediction.confidence_interval[1] == pytest.approx(7.0 * 1.3)

    def test_velocity_with_multiple_sprints(self):
        """Test prediction with multiple sprints."""
        pred = VelocityPredictor()

        # Record stable velocities
        for _ in range(5):
            pred.record_velocity(5.0)

        prediction = pred.predict_next()

        # Should predict around 5.0
        assert prediction.predicted_velocity == pytest.approx(5.0, abs=0.1)
        assert prediction.confidence > 0.5  # High confidence due to low variance

    def test_simple_moving_average(self):
        """Test simple moving average calculation."""
        pred = VelocityPredictor(history_window=5)

        # Record 10 velocities
        velocities = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        for v in velocities:
            pred.record_velocity(v)

        # SMA should use last 5 values (window size)
        expected_sma = sum(velocities[-5:]) / 5  # (8+9+10+11+12)/5 = 10.0
        actual_sma = pred._simple_moving_average()

        assert actual_sma == pytest.approx(expected_sma)


# =============================================================================
# TEST CLASS 3: TREND DETECTION
# =============================================================================


class TestTrendDetection:
    """Test trend detection functionality."""

    def test_detect_accelerating_trend(self, predictor_with_increasing_history):
        """Test detection of increasing velocity trend."""
        trend = predictor_with_increasing_history.get_trend()

        assert trend == "increasing"

    def test_detect_decelerating_trend(self, predictor_with_decreasing_history):
        """Test detection of decreasing velocity trend."""
        trend = predictor_with_decreasing_history.get_trend()

        assert trend == "decreasing"

    def test_detect_stable_trend(self, predictor_with_stable_history):
        """Test detection of stable velocity trend."""
        trend = predictor_with_stable_history.get_trend()

        assert trend == "stable"

    def test_trend_with_insufficient_data(self):
        """Test trend detection with less than 3 data points."""
        pred = VelocityPredictor()

        # No data
        assert pred.get_trend() == "stable"

        # One data point
        pred.record_velocity(5.0)
        assert pred.get_trend() == "stable"

        # Two data points
        pred.record_velocity(6.0)
        assert pred.get_trend() == "stable"

    def test_linear_regression_calculation(self):
        """Test linear regression for trend detection."""
        pred = VelocityPredictor()

        # Perfect linear increase: y = 2x + 3
        data = [3.0, 5.0, 7.0, 9.0, 11.0]
        slope, intercept = pred._linear_regression(data)

        assert slope == pytest.approx(2.0, abs=0.01)
        assert intercept == pytest.approx(3.0, abs=0.01)

    def test_trend_threshold_sensitivity(self):
        """Test that small changes don't trigger trend detection."""
        pred = VelocityPredictor(history_window=10)

        # Record velocities with very small increase (< 10% threshold)
        base = 10.0
        for i in range(10):
            pred.record_velocity(base + i * 0.01)  # Very small increase

        # Should still be stable (change < 10% threshold)
        trend = pred.get_trend()
        assert trend == "stable"


# =============================================================================
# TEST CLASS 4: ANOMALY DETECTION
# =============================================================================


class TestAnomalyDetection:
    """Test anomaly detection functionality."""

    def test_detect_velocity_spike(self, predictor_with_anomaly):
        """Test detection of velocity spike."""
        anomalies = predictor_with_anomaly.detect_anomalies()

        # Should detect the spike at index 5
        assert len(anomalies) > 0

        spike_anomaly = next((a for a in anomalies if a.velocity == 15.0), None)
        assert spike_anomaly is not None
        assert spike_anomaly.deviation > 2.0  # More than 2 std deviations
        assert "unusually high" in spike_anomaly.potential_causes[0].lower()

    def test_detect_velocity_drop(self):
        """Test detection of velocity drop."""
        pred = VelocityPredictor(history_window=10)

        # Record stable velocities with one drop
        for i in range(10):
            if i == 5:
                pred.record_velocity(1.0, {"impediments": 3})  # Drop with context
            else:
                pred.record_velocity(10.0)

        anomalies = pred.detect_anomalies()

        # Should detect the drop
        assert len(anomalies) > 0

        drop_anomaly = next((a for a in anomalies if a.velocity == 1.0), None)
        assert drop_anomaly is not None
        assert "unusually low" in drop_anomaly.potential_causes[0].lower()

    def test_normal_velocity_no_anomaly(self, predictor_with_stable_history):
        """Test that stable velocities don't trigger anomalies."""
        anomalies = predictor_with_stable_history.detect_anomalies()

        # No anomalies in perfectly stable data
        assert len(anomalies) == 0

    def test_anomaly_threshold_configuration(self):
        """Test anomaly detection threshold (2 std deviations)."""
        pred = VelocityPredictor(history_window=10)

        # Record velocities with known mean and std dev
        # Mean = 10, Std Dev ≈ 2
        velocities = [8.0, 9.0, 10.0, 11.0, 12.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        for v in velocities:
            pred.record_velocity(v)

        # Add a value just under 2 std devs away
        pred.record_velocity(13.5)  # Not quite anomalous

        anomalies = pred.detect_anomalies()

        # Should not detect 13.5 as anomaly (< 2 std devs)
        values = [a.velocity for a in anomalies]
        assert 13.5 not in values

    def test_anomaly_detection_insufficient_data(self):
        """Test anomaly detection with insufficient data."""
        pred = VelocityPredictor()

        # Less than 5 data points
        for i in range(4):
            pred.record_velocity(float(i))

        anomalies = pred.detect_anomalies()

        # Should return empty list
        assert anomalies == []

    def test_anomaly_detection_zero_variance(self):
        """Test anomaly detection when all values are identical."""
        pred = VelocityPredictor(history_window=10)

        # All identical values
        for _ in range(10):
            pred.record_velocity(5.0)

        anomalies = pred.detect_anomalies()

        # Should return empty list (std dev = 0)
        assert anomalies == []

    def test_anomaly_causes_with_context(self):
        """Test that anomaly causes use sprint context."""
        pred = VelocityPredictor(history_window=10)

        # Record stable velocities
        for _ in range(5):
            pred.record_velocity(10.0)

        # Record low velocity with context
        pred.record_velocity(
            2.0,
            {
                "impediments": 2,
                "complexity": "high",
                "team_changes": True,
            }
        )

        # More stable velocities
        for _ in range(4):
            pred.record_velocity(10.0)

        anomalies = pred.detect_anomalies()

        # Find the low velocity anomaly
        low_anomaly = next((a for a in anomalies if a.velocity == 2.0), None)
        assert low_anomaly is not None

        # Check that context was used for causes
        causes_text = " ".join(low_anomaly.potential_causes).lower()
        assert "impediments" in causes_text or "complexity" in causes_text


# =============================================================================
# TEST CLASS 5: PREDICTION
# =============================================================================


class TestPrediction:
    """Test velocity prediction functionality."""

    def test_predict_next_sprint(self, predictor_with_stable_history):
        """Test prediction for next sprint."""
        prediction = predictor_with_stable_history.predict_next()

        # Should predict around 5.0
        assert isinstance(prediction, VelocityPrediction)
        assert prediction.predicted_velocity == pytest.approx(5.0, abs=0.5)
        assert prediction.trend == "stable"

    def test_prediction_confidence(self):
        """Test confidence calculation based on variance."""
        # High variance = low confidence
        pred_high_var = VelocityPredictor(history_window=10)
        for i in range(10):
            pred_high_var.record_velocity(float(i * 2))  # 0, 2, 4, ..., 18

        prediction_high_var = pred_high_var.predict_next()

        # Low variance = high confidence
        pred_low_var = VelocityPredictor(history_window=10)
        for _ in range(10):
            pred_low_var.record_velocity(5.0)

        prediction_low_var = pred_low_var.predict_next()

        # Low variance should have higher confidence
        assert prediction_low_var.confidence > prediction_high_var.confidence

    def test_prediction_with_trend_adjustment(self, predictor_with_increasing_history):
        """Test that prediction accounts for trends."""
        prediction = predictor_with_increasing_history.predict_next()

        # Should recognize increasing trend
        assert prediction.trend == "increasing"
        assert "increasing" in prediction.factors or "Velocity is increasing" in prediction.factors

    def test_confidence_interval_calculation(self):
        """Test 95% confidence interval calculation."""
        pred = VelocityPredictor(history_window=10)

        # Record known velocities
        velocities = [4.0, 5.0, 6.0, 5.0, 4.0, 5.0, 6.0, 5.0, 4.0, 5.0]
        for v in velocities:
            pred.record_velocity(v)

        prediction = pred.predict_next()

        # Confidence interval should be around predicted value
        low, high = prediction.confidence_interval
        assert low < prediction.predicted_velocity < high

        # Interval should be reasonable (not negative)
        assert low >= 0

        # Interval should be roughly symmetric around prediction
        # (may not be exact due to EMA vs mean)
        assert abs((high - low) / 2) > 0

    def test_prediction_factors_analysis(self):
        """Test that prediction includes relevant factors."""
        pred = VelocityPredictor(history_window=10)

        # Limited data
        pred.record_velocity(5.0)
        pred.record_velocity(5.0)
        prediction = pred.predict_next()

        # Should mention limited data
        factors_text = " ".join(prediction.factors)
        assert "limited" in factors_text.lower() or "only" in factors_text.lower()

    def test_prediction_factors_variance(self):
        """Test that prediction factors mention variance."""
        # High variance case
        pred_high = VelocityPredictor(history_window=10)
        for i in range(10):
            pred_high.record_velocity(float(i) * 2)  # High variance

        prediction_high = pred_high.predict_next()
        factors_high = " ".join(prediction_high.factors).lower()

        # Low variance case
        pred_low = VelocityPredictor(history_window=10)
        for _ in range(10):
            pred_low.record_velocity(5.0)  # Low variance

        prediction_low = pred_low.predict_next()
        factors_low = " ".join(prediction_low.factors).lower()

        # One should mention variance/consistency
        assert "variance" in factors_high or "consistent" in factors_low

    def test_standard_deviation_calculation(self):
        """Test standard deviation calculation."""
        pred = VelocityPredictor(history_window=5)

        # Known data: [2, 4, 6, 8, 10] -> mean=6, variance=8, std=sqrt(8)≈2.83
        velocities = [2.0, 4.0, 6.0, 8.0, 10.0]
        for v in velocities:
            pred.record_velocity(v)

        std_dev = pred._standard_deviation()
        expected_std = math.sqrt(8.0)

        assert std_dev == pytest.approx(expected_std, abs=0.01)


# =============================================================================
# TEST CLASS 6: ADDITIONAL FUNCTIONALITY
# =============================================================================


class TestStatisticsAndHelpers:
    """Test statistical methods and helper functions."""

    def test_get_statistics(self):
        """Test get_statistics returns complete summary."""
        pred = VelocityPredictor(history_window=10)

        # Record known velocities
        velocities = [3.0, 5.0, 7.0, 5.0, 3.0, 5.0, 7.0, 5.0, 3.0, 5.0]
        for v in velocities:
            pred.record_velocity(v)

        stats = pred.get_statistics()

        # Check all required fields
        assert "mean" in stats
        assert "median" in stats
        assert "std_dev" in stats
        assert "min" in stats
        assert "max" in stats
        assert "trend_slope" in stats
        assert "count" in stats

        # Verify values
        assert stats["mean"] == pytest.approx(sum(velocities) / len(velocities))
        assert stats["median"] == 5.0  # Middle value when sorted
        assert stats["min"] == 3.0
        assert stats["max"] == 7.0
        assert stats["count"] == 10

    def test_get_statistics_empty(self):
        """Test get_statistics with no data."""
        pred = VelocityPredictor()

        stats = pred.get_statistics()

        # Should return zeros
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["std_dev"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["count"] == 0

    def test_record_velocity_validation(self):
        """Test that negative velocities are rejected."""
        pred = VelocityPredictor()

        with pytest.raises(ValueError, match="cannot be negative"):
            pred.record_velocity(-1.0)

    def test_record_velocity_with_context(self):
        """Test recording velocity with sprint context."""
        pred = VelocityPredictor()

        context = {
            "team_size": 3,
            "complexity": "high",
            "impediments": 1,
        }

        pred.record_velocity(5.0, context)

        assert len(pred._history) == 1
        assert len(pred._contexts) == 1
        assert pred._contexts[0] == context

    def test_history_window_trimming(self):
        """Test that history is trimmed to 2x window size."""
        pred = VelocityPredictor(history_window=5)

        # Record more than 2x window (> 10)
        for i in range(20):
            pred.record_velocity(float(i))

        # Should keep only 2x window = 10 entries
        assert len(pred._history) == 10
        assert len(pred._contexts) == 10

        # Should keep the most recent ones
        assert pred._history[-1] == 19.0
        assert pred._history[0] == 10.0

    def test_median_calculation_odd_count(self):
        """Test median calculation with odd number of values."""
        pred = VelocityPredictor(history_window=10)

        # Odd number: [1, 2, 3, 4, 5] -> median = 3
        velocities = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in velocities:
            pred.record_velocity(v)

        stats = pred.get_statistics()
        assert stats["median"] == 3.0

    def test_median_calculation_even_count(self):
        """Test median calculation with even number of values."""
        pred = VelocityPredictor(history_window=10)

        # Even number: [1, 2, 3, 4] -> median = (2+3)/2 = 2.5
        velocities = [1.0, 2.0, 3.0, 4.0]
        for v in velocities:
            pred.record_velocity(v)

        stats = pred.get_statistics()
        assert stats["median"] == 2.5
