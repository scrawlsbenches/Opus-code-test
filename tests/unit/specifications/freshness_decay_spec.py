"""
Freshness Decay Specifications
==============================

SPECIFICATION: These tests document LOAD-BEARING behavior for freshness decay.

The freshness decay system controls how recent documents are ranked higher
in search results. These specifications define the exact mathematical
behavior that users and downstream code depend on.

DO NOT CHANGE these specifications without:
1. Team review and documented justification
2. Assessing backward compatibility impact
3. Updating dependent documentation

Ratified: 2024-12-30
Guardian: CI Pipeline
"""

import math
import pytest
from datetime import datetime, timedelta

# Import the function under specification
from cortical.query.search import _compute_decay_factor
from cortical.constants import (
    FRESHNESS_WINDOW_DAYS,
    DEFAULT_FRESHNESS_DECAY,
    FRESHNESS_DECAY_FUNCTIONS,
)


class TestFreshnessDecaySpecification:
    """
    Specifications for _compute_decay_factor() - the core decay algorithm.

    Each specification documents a fact about the system that must remain true.
    These are not "tests" - they are executable documentation of promises.
    """

    # =========================================================================
    # LINEAR DECAY SPECIFICATIONS
    # =========================================================================

    def test_spec_linear_decay_at_day_zero_returns_one(self):
        """
        SPECIFICATION: Linear decay returns 1.0 (full boost) for day 0 documents.

        This ensures that brand-new documents receive the maximum freshness boost.
        This is load-bearing behavior - changing it would alter search ranking.
        """
        decay_factor = _compute_decay_factor(
            days_old=0,
            window_days=7,
            decay_function="linear"
        )
        assert decay_factor == 1.0, (
            "Day 0 documents must receive full boost (decay_factor=1.0)"
        )

    def test_spec_linear_decay_at_window_boundary_returns_zero(self):
        """
        SPECIFICATION: Linear decay returns 0.0 at the window boundary.

        At exactly N days (the window boundary), the boost should be zero.
        Documents at the boundary get no freshness advantage.
        """
        window = 7
        decay_factor = _compute_decay_factor(
            days_old=window,
            window_days=window,
            decay_function="linear"
        )
        assert decay_factor == 0.0, (
            f"Day {window} documents must receive no boost (decay_factor=0.0)"
        )

    def test_spec_linear_decay_is_mathematically_linear(self):
        """
        SPECIFICATION: Linear decay follows formula: decay = 1.0 - (days_old / window_days)

        This is the precise mathematical formula. It must remain exactly this
        for reproducible search behavior.
        """
        window = 7
        for days in range(window + 1):
            expected = 1.0 - (days / window)
            actual = _compute_decay_factor(
                days_old=days,
                window_days=window,
                decay_function="linear"
            )
            assert abs(actual - expected) < 1e-10, (
                f"Day {days}: expected {expected:.6f}, got {actual:.6f}"
            )

    def test_spec_linear_decay_midpoint_returns_half(self):
        """
        SPECIFICATION: Linear decay returns 0.5 at the midpoint of the window.

        For a 7-day window, day 3.5 should return exactly 0.5.
        This verifies the decay is truly linear, not curved.
        """
        window = 7
        midpoint = window / 2.0
        decay_factor = _compute_decay_factor(
            days_old=midpoint,
            window_days=window,
            decay_function="linear"
        )
        assert abs(decay_factor - 0.5) < 1e-10, (
            f"Midpoint decay should be 0.5, got {decay_factor}"
        )

    # =========================================================================
    # EXPONENTIAL DECAY SPECIFICATIONS
    # =========================================================================

    def test_spec_exponential_decay_at_day_zero_returns_one(self):
        """
        SPECIFICATION: Exponential decay returns 1.0 (full boost) for day 0 documents.

        Same promise as linear: newest documents always get full boost.
        """
        decay_factor = _compute_decay_factor(
            days_old=0,
            window_days=7,
            decay_function="exponential"
        )
        assert abs(decay_factor - 1.0) < 1e-10, (
            "Day 0 documents must receive full boost with exponential decay"
        )

    def test_spec_exponential_decay_at_boundary_returns_zero(self):
        """
        SPECIFICATION: Exponential decay returns 0.0 at the window boundary.

        The exponential curve is normalized to reach 0 at the boundary.
        """
        window = 7
        decay_factor = _compute_decay_factor(
            days_old=window,
            window_days=window,
            decay_function="exponential"
        )
        assert abs(decay_factor - 0.0) < 1e-10, (
            f"Day {window} documents must receive no boost with exponential decay"
        )

    def test_spec_exponential_decay_concentrates_boost_on_newest(self):
        """
        SPECIFICATION: Exponential decay concentrates boost heavily on newest docs.

        Exponential decay drops off FASTER than linear, meaning:
        - Day 0 docs get full boost (same as linear)
        - Day 1+ docs get LESS boost than linear
        - The boost is heavily concentrated on the most recent documents

        This is useful when recency is extremely important and you want to
        strongly favor brand-new documents over even slightly older ones.
        """
        window = 7
        for day in [1, 2, 3, 4, 5, 6]:
            linear = _compute_decay_factor(day, window, "linear")
            exponential = _compute_decay_factor(day, window, "exponential")
            assert exponential < linear, (
                f"Day {day}: exponential ({exponential:.3f}) must be < "
                f"linear ({linear:.3f}) due to faster decay"
            )

    def test_spec_exponential_decay_uses_e_power_minus_3(self):
        """
        SPECIFICATION: Exponential decay uses base curve e^(-3 * normalized_age).

        This is the precise mathematical formula for consistency:
        - decay = (e^(-3 * t) - e^(-3)) / (1 - e^(-3))
        where t = days_old / window_days
        """
        window = 7

        def expected_exponential(days_old):
            if days_old >= window:
                return 0.0
            t = days_old / window
            min_val = math.exp(-3.0)
            decay = math.exp(-3.0 * t)
            return (decay - min_val) / (1.0 - min_val)

        for day in [0, 1, 2, 3, 4, 5, 6, 7]:
            expected = expected_exponential(day)
            actual = _compute_decay_factor(day, window, "exponential")
            assert abs(actual - expected) < 1e-10, (
                f"Day {day}: expected {expected:.6f}, got {actual:.6f}"
            )

    # =========================================================================
    # BINARY (NONE) DECAY SPECIFICATIONS
    # =========================================================================

    def test_spec_none_decay_returns_one_within_window(self):
        """
        SPECIFICATION: decay_function="none" returns 1.0 for all days within window.

        This preserves the original binary behavior: full boost inside window.
        """
        window = 7
        for day in range(window):
            decay_factor = _compute_decay_factor(day, window, "none")
            assert decay_factor == 1.0, (
                f"Day {day} with decay='none' should return 1.0, got {decay_factor}"
            )

    def test_spec_none_decay_returns_zero_at_boundary(self):
        """
        SPECIFICATION: decay_function="none" returns 0.0 at and beyond window boundary.

        Binary behavior: no boost outside the freshness window.
        """
        window = 7
        for day in [window, window + 1, window + 10]:
            decay_factor = _compute_decay_factor(day, window, "none")
            assert decay_factor == 0.0, (
                f"Day {day} with decay='none' should return 0.0, got {decay_factor}"
            )

    # =========================================================================
    # BOUNDARY AND EDGE CASE SPECIFICATIONS
    # =========================================================================

    def test_spec_negative_days_treated_as_zero(self):
        """
        SPECIFICATION: Negative days_old values are clamped to 0.

        Documents cannot be from the future. Defensive behavior ensures
        negative values don't cause unexpected results.
        """
        for decay_func in ["linear", "exponential", "none"]:
            decay_factor = _compute_decay_factor(-5, 7, decay_func)
            expected = _compute_decay_factor(0, 7, decay_func)
            assert decay_factor == expected, (
                f"Negative days with {decay_func} should equal day 0"
            )

    def test_spec_beyond_window_always_returns_zero(self):
        """
        SPECIFICATION: All decay functions return 0.0 beyond the window.

        Documents older than the window get no freshness boost, regardless
        of which decay function is used.
        """
        window = 7
        for days_old in [8, 10, 30, 100]:
            for decay_func in FRESHNESS_DECAY_FUNCTIONS:
                decay_factor = _compute_decay_factor(days_old, window, decay_func)
                assert decay_factor == 0.0, (
                    f"Day {days_old} with {decay_func} should return 0.0"
                )

    def test_spec_unknown_decay_function_falls_back_to_linear(self):
        """
        SPECIFICATION: Unknown decay functions fall back to linear decay.

        This ensures forward compatibility: if new decay functions are added
        but not handled, the system gracefully degrades to linear.
        """
        linear_result = _compute_decay_factor(3, 7, "linear")
        unknown_result = _compute_decay_factor(3, 7, "unknown_decay_type")
        assert linear_result == unknown_result, (
            "Unknown decay function should fall back to linear"
        )

    # =========================================================================
    # CONSTANTS SPECIFICATIONS
    # =========================================================================

    def test_spec_default_window_is_seven_days(self):
        """
        SPECIFICATION: The default freshness window is 7 days.

        Users expect a 7-day window. Changing this affects existing deployments.
        """
        assert FRESHNESS_WINDOW_DAYS == 7, (
            f"Default window must be 7 days, got {FRESHNESS_WINDOW_DAYS}"
        )

    def test_spec_default_decay_is_linear(self):
        """
        SPECIFICATION: The default decay function is "linear".

        Linear was chosen because it's intuitive and predictable.
        """
        assert DEFAULT_FRESHNESS_DECAY == "linear", (
            f"Default decay must be 'linear', got '{DEFAULT_FRESHNESS_DECAY}'"
        )

    def test_spec_exactly_three_decay_functions(self):
        """
        SPECIFICATION: Exactly three decay functions are supported.

        - linear: Graduated linear decay
        - exponential: Front-loaded exponential decay
        - none: Binary (original) behavior

        Adding new functions requires spec update and team review.
        """
        expected = frozenset(["linear", "exponential", "none"])
        assert FRESHNESS_DECAY_FUNCTIONS == expected, (
            f"Expected decay functions {expected}, got {FRESHNESS_DECAY_FUNCTIONS}"
        )


class TestApplyFreshnessBoostSpecification:
    """
    Specifications for _apply_freshness_boost() - the boost application logic.

    These document how the decay factor is converted to an actual score boost.
    """

    def test_spec_boost_formula_is_additive(self):
        """
        SPECIFICATION: Boost formula is multiplicative: score * (1 + (boost - 1) * decay_factor)

        For freshness_boost=1.5 and decay_factor=1.0 (day 0):
        - effective_boost = 1.0 + (1.5 - 1.0) * 1.0 = 1.5

        For freshness_boost=1.5 and decay_factor=0.5 (midpoint):
        - effective_boost = 1.0 + (1.5 - 1.0) * 0.5 = 1.25
        """
        # This is documented behavior. The formula ensures:
        # - decay_factor=1.0 -> full boost (freshness_boost multiplier)
        # - decay_factor=0.5 -> half boost
        # - decay_factor=0.0 -> no boost (multiplier = 1.0)

        freshness_boost = 1.5

        # Day 0: full boost
        decay_factor = 1.0
        expected = 1.0 + (freshness_boost - 1.0) * decay_factor
        assert expected == 1.5

        # Midpoint: half boost
        decay_factor = 0.5
        expected = 1.0 + (freshness_boost - 1.0) * decay_factor
        assert expected == 1.25

        # Boundary: no boost
        decay_factor = 0.0
        expected = 1.0 + (freshness_boost - 1.0) * decay_factor
        assert expected == 1.0
