"""
Unit Tests for Experiments Package
===================================

Tests for cortical/experiments/ module:
- Profiler: timing, memory tracking, guard patterns, context manager
- Tokenizer: word-level tokenization, vocabulary building
- ExperimentKernel utilities: clip_gradients, compute_gradient_norm

Coverage target: 90%+
"""

import pytest
import time
import tracemalloc
import numpy as np

from cortical.experiments.profiler import Profiler, StepMetrics, ProfilingReport
from cortical.experiments.tokenizer import (
    tokenize,
    build_vocab,
    tokens_to_ids,
    ids_to_tokens,
    detokenize,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    SPECIAL_TOKENS,
)
from cortical.experiments.kernel import (
    clip_gradients,
    compute_gradient_norm,
    TrainingHistory,
)


# =============================================================================
# TOKENIZER TESTS
# =============================================================================


class TestTokenize:
    """Tests for tokenize function."""

    def test_basic_tokenization(self):
        """Basic text is split into words."""
        tokens = tokenize("hello world")
        assert tokens == ["hello", "world"]

    def test_lowercase_default(self):
        """Tokenization lowercases by default."""
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_preserve_case(self):
        """Can preserve case with lowercase=False."""
        tokens = tokenize("Hello World", lowercase=False)
        assert tokens == ["Hello", "World"]

    def test_punctuation_separate(self):
        """Punctuation is split as separate tokens."""
        tokens = tokenize("Hello, world!")
        assert "," in tokens
        assert "!" in tokens
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_string(self):
        """Empty string returns empty list."""
        tokens = tokenize("")
        assert tokens == []

    def test_whitespace_only(self):
        """Whitespace-only string returns empty list."""
        tokens = tokenize("   \t\n  ")
        assert tokens == []

    def test_numbers_preserved(self):
        """Numbers are preserved as tokens."""
        tokens = tokenize("there are 42 items")
        assert "42" in tokens


class TestBuildVocab:
    """Tests for build_vocab function."""

    def test_basic_vocab(self):
        """Builds vocab from token list."""
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        vocab, id_to_token = build_vocab(tokens)

        # Special tokens come first
        assert vocab[PAD_TOKEN] == 0
        assert vocab[UNK_TOKEN] == 1
        assert vocab[BOS_TOKEN] == 2
        assert vocab[EOS_TOKEN] == 3

        # Regular tokens start after special tokens
        assert "the" in vocab
        assert "cat" in vocab

    def test_min_freq_filter(self):
        """Tokens below min_freq are excluded."""
        tokens = ["a", "a", "a", "b", "b", "c"]
        vocab, _ = build_vocab(tokens, min_freq=2)

        assert "a" in vocab
        assert "b" in vocab
        assert "c" not in vocab  # Only appears once

    def test_max_vocab_size(self):
        """Limits vocabulary size."""
        tokens = ["a"] * 10 + ["b"] * 5 + ["c"] * 3 + ["d"] * 1
        vocab, _ = build_vocab(tokens, max_vocab_size=2)

        # Only top 2 by frequency (plus special tokens)
        assert "a" in vocab
        assert "b" in vocab
        assert "c" not in vocab
        assert "d" not in vocab

    def test_id_to_token_inverse(self):
        """id_to_token is inverse of vocab."""
        tokens = ["foo", "bar", "baz"]
        vocab, id_to_token = build_vocab(tokens)

        for token, id_ in vocab.items():
            assert id_to_token[id_] == token


class TestTokensToIds:
    """Tests for tokens_to_ids function."""

    def test_basic_conversion(self):
        """Converts tokens to IDs."""
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, "hello": 2, "world": 3}
        ids = tokens_to_ids(["hello", "world"], vocab)
        assert ids == [2, 3]

    def test_unknown_token(self):
        """Unknown tokens map to UNK."""
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, "hello": 2}
        ids = tokens_to_ids(["hello", "unknown"], vocab)
        assert ids == [2, 1]  # 1 is UNK_TOKEN

    def test_add_bos(self):
        """Can add BOS token."""
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, BOS_TOKEN: 2, "hello": 3}
        ids = tokens_to_ids(["hello"], vocab, add_bos=True)
        assert ids[0] == 2  # BOS first

    def test_add_eos(self):
        """Can add EOS token."""
        vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, EOS_TOKEN: 3, "hello": 4}
        ids = tokens_to_ids(["hello"], vocab, add_eos=True)
        assert ids[-1] == 3  # EOS last


class TestIdsToTokens:
    """Tests for ids_to_tokens function."""

    def test_basic_conversion(self):
        """Converts IDs back to tokens."""
        id_to_token = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: "hello", 3: "world"}
        tokens = ids_to_tokens([2, 3], id_to_token)
        assert tokens == ["hello", "world"]

    def test_skip_special_default(self):
        """Special tokens are skipped by default."""
        id_to_token = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: "hello"}
        tokens = ids_to_tokens([0, 2, 1], id_to_token)
        assert tokens == ["hello"]

    def test_include_special(self):
        """Can include special tokens."""
        id_to_token = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: "hello"}
        tokens = ids_to_tokens([0, 2], id_to_token, skip_special=False)
        assert tokens == [PAD_TOKEN, "hello"]


class TestDetokenize:
    """Tests for detokenize function."""

    def test_basic_join(self):
        """Joins tokens with spaces."""
        text = detokenize(["hello", "world"])
        assert text == "hello world"

    def test_punctuation_cleanup(self):
        """Punctuation spacing is cleaned."""
        text = detokenize(["hello", ",", "world", "!"])
        assert text == "hello, world!"

    def test_empty_list(self):
        """Empty list returns empty string."""
        text = detokenize([])
        assert text == ""


# =============================================================================
# PROFILER TESTS
# =============================================================================


class TestStepMetrics:
    """Tests for StepMetrics dataclass."""

    def test_to_dict(self):
        """StepMetrics converts to dictionary."""
        metrics = StepMetrics(
            step=0,
            loss=1.5,
            forward_time_ms=10.0,
            backward_time_ms=15.0,
            update_time_ms=5.0,
            total_time_ms=30.0,
            gradient_norm=0.5,
            memory_delta_bytes=1024,
        )
        d = metrics.to_dict()

        assert d["step"] == 0
        assert d["loss"] == 1.5
        assert d["forward_time_ms"] == 10.0
        assert d["gradient_norm"] == 0.5


class TestProfilingReport:
    """Tests for ProfilingReport dataclass."""

    def test_str_format(self):
        """Report has human-readable string."""
        report = ProfilingReport(
            total_steps=100,
            total_time_seconds=10.0,
            forward_time_mean=5.0,
            initial_loss=1.0,
            final_loss=0.1,
        )
        s = str(report)

        assert "PROFILING REPORT" in s
        assert "100" in s  # total_steps
        assert "10.0" in s  # total_time


class TestProfilerBasic:
    """Basic Profiler tests."""

    def test_disabled_profiler(self):
        """Disabled profiler has minimal overhead."""
        profiler = Profiler(enabled=False)

        with profiler.step(0) as metrics:
            pass

        # Metrics still returned but are zeros
        assert metrics.total_time_ms == 0.0
        profiler.close()

    def test_step_timing(self):
        """Step timing is recorded."""
        profiler = Profiler(enabled=True, track_memory=False)

        with profiler.step(0) as metrics:
            time.sleep(0.01)  # 10ms

        assert metrics.total_time_ms >= 10.0
        assert metrics.step == 0
        profiler.close()

    def test_forward_backward_update_timing(self):
        """Individual phase timing works."""
        profiler = Profiler(enabled=True, track_memory=False)

        with profiler.step(0) as metrics:
            with profiler.forward():
                time.sleep(0.005)
            with profiler.backward():
                time.sleep(0.005)
            with profiler.update():
                time.sleep(0.005)

        assert metrics.forward_time_ms >= 5.0
        assert metrics.backward_time_ms >= 5.0
        assert metrics.update_time_ms >= 5.0
        profiler.close()

    def test_report_generation(self):
        """Report aggregates multiple steps."""
        profiler = Profiler(enabled=True, track_memory=False)

        for i in range(5):
            with profiler.step(i) as metrics:
                metrics.loss = 1.0 / (i + 1)
                metrics.gradient_norm = 0.1 * (i + 1)

        report = profiler.report()

        assert report.total_steps == 5
        assert report.initial_loss == 1.0
        assert report.final_loss == 0.2
        profiler.close()


class TestProfilerContextManager:
    """Tests for Profiler context manager interface."""

    def test_context_manager_cleanup(self):
        """Context manager cleans up resources."""
        with Profiler(enabled=True, track_memory=False) as profiler:
            with profiler.step(0) as metrics:
                metrics.loss = 1.0

        # Should be closed after context exit
        assert profiler._closed

    def test_close_idempotent(self):
        """Close can be called multiple times safely."""
        profiler = Profiler(enabled=True, track_memory=False)
        profiler.close()
        profiler.close()  # Should not raise
        assert profiler._closed


class TestProfilerTracemalloc:
    """Tests for tracemalloc guard pattern."""

    def test_tracemalloc_guard_already_tracing(self):
        """Profiler doesn't start tracemalloc if already tracing."""
        # Start tracemalloc ourselves
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()

        try:
            profiler = Profiler(enabled=True, track_memory=True)
            # Should not own tracemalloc since it was already started
            assert not profiler._owns_tracemalloc
            profiler.close()
            # Tracemalloc should still be running (we started it)
            assert tracemalloc.is_tracing()
        finally:
            if not was_tracing:
                tracemalloc.stop()

    def test_tracemalloc_started_and_stopped(self):
        """Profiler starts/stops tracemalloc when not already tracing."""
        # Ensure tracemalloc is stopped
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        profiler = Profiler(enabled=True, track_memory=True)
        assert profiler._owns_tracemalloc
        assert tracemalloc.is_tracing()

        profiler.close()
        assert not tracemalloc.is_tracing()


class TestProfilerReset:
    """Tests for Profiler reset functionality."""

    def test_reset_clears_state(self):
        """Reset clears all recorded data."""
        profiler = Profiler(enabled=True, track_memory=False)

        with profiler.step(0) as metrics:
            metrics.loss = 1.0

        assert len(profiler._steps) == 1

        profiler.reset()

        assert len(profiler._steps) == 0
        assert profiler._start_time is None
        profiler.close()


# =============================================================================
# KERNEL UTILITY TESTS
# =============================================================================


class MockParameter:
    """Mock Parameter for testing gradient utilities."""

    def __init__(self, shape, grad_value=None):
        self.data = np.zeros(shape)
        self.grad = np.full(shape, grad_value) if grad_value is not None else None


class TestClipGradients:
    """Tests for clip_gradients utility."""

    def test_no_clipping_needed(self):
        """Gradients under max_norm are not clipped."""
        params = [MockParameter((10,), grad_value=0.1)]
        # norm = sqrt(10 * 0.01) = sqrt(0.1) ~ 0.316
        norm = clip_gradients(params, max_norm=1.0)

        assert norm < 1.0
        assert np.allclose(params[0].grad, 0.1)

    def test_clipping_applied(self):
        """Gradients over max_norm are clipped."""
        params = [MockParameter((10,), grad_value=1.0)]
        # norm = sqrt(10) ~ 3.16
        original_norm = clip_gradients(params, max_norm=1.0)

        assert original_norm > 1.0
        # After clipping, norm should be max_norm
        new_norm = compute_gradient_norm(params)
        assert np.isclose(new_norm, 1.0, atol=0.01)

    def test_none_gradients_skipped(self):
        """Parameters with None gradients are skipped."""
        params = [
            MockParameter((10,), grad_value=0.1),
            MockParameter((10,), grad_value=None),
        ]
        norm = clip_gradients(params, max_norm=1.0)
        # Only first param contributes to norm
        assert norm > 0

    def test_zero_max_norm(self):
        """Zero max_norm clips to zero."""
        params = [MockParameter((10,), grad_value=1.0)]
        clip_gradients(params, max_norm=0.0)
        # Gradients should be scaled to zero
        assert np.allclose(params[0].grad, 0.0)


class TestComputeGradientNorm:
    """Tests for compute_gradient_norm utility."""

    def test_single_parameter(self):
        """Computes norm of single parameter."""
        params = [MockParameter((4,), grad_value=1.0)]
        # norm = sqrt(4 * 1) = 2.0
        norm = compute_gradient_norm(params)
        assert np.isclose(norm, 2.0)

    def test_multiple_parameters(self):
        """Computes global norm across parameters."""
        params = [
            MockParameter((4,), grad_value=1.0),
            MockParameter((5,), grad_value=1.0),
        ]
        # norm = sqrt(4 + 5) = 3.0
        norm = compute_gradient_norm(params)
        assert np.isclose(norm, 3.0)

    def test_no_gradients(self):
        """Returns zero if no gradients."""
        params = [MockParameter((10,), grad_value=None)]
        norm = compute_gradient_norm(params)
        assert norm == 0.0


class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""

    def test_log_basic(self):
        """Log records train loss."""
        history = TrainingHistory()
        history.log(train_loss=1.0)
        history.log(train_loss=0.5)

        assert history.train_losses == [1.0, 0.5]

    def test_log_optional_fields(self):
        """Log can record optional fields."""
        history = TrainingHistory()
        history.log(
            train_loss=1.0,
            val_loss=1.1,
            lr=0.01,
            grad_norm=0.5,
        )

        assert history.val_losses == [1.1]
        assert history.learning_rates == [0.01]
        assert history.gradient_norms == [0.5]

    def test_log_with_metrics(self):
        """Log can record StepMetrics."""
        history = TrainingHistory()
        metrics = StepMetrics(
            step=0, loss=1.0, forward_time_ms=10.0,
            backward_time_ms=15.0, update_time_ms=5.0,
            total_time_ms=30.0, gradient_norm=0.5,
        )
        history.log(train_loss=1.0, metrics=metrics)

        assert len(history.step_metrics) == 1
        assert history.step_metrics[0].step == 0
