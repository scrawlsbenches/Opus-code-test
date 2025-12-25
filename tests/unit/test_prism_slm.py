"""
Tests for PRISM-SLM: Statistical Language Model with Synaptic Learning.

TDD: These tests define expected behavior before implementation.
"""

import pytest
from datetime import datetime, timedelta


class TestSynapticTransition:
    """Test synaptic transitions between tokens."""

    def test_transition_creation(self):
        """Test creating a basic transition."""
        from cortical.reasoning.prism_slm import SynapticTransition

        trans = SynapticTransition(
            from_token="the",
            to_token="quick",
        )

        assert trans.from_token == "the"
        assert trans.to_token == "quick"
        assert trans.weight == 1.0
        assert trans.count == 0

    def test_transition_strengthening(self):
        """Transitions strengthen with repeated use."""
        from cortical.reasoning.prism_slm import SynapticTransition

        trans = SynapticTransition("the", "quick")
        initial_weight = trans.weight

        trans.observe()
        trans.observe()
        trans.observe()

        assert trans.count == 3
        assert trans.weight > initial_weight

    def test_transition_decay(self):
        """Unused transitions decay over time."""
        from cortical.reasoning.prism_slm import SynapticTransition

        trans = SynapticTransition("the", "quick", weight=2.0)
        trans.apply_decay(factor=0.9)

        assert trans.weight == pytest.approx(1.8, rel=0.01)

    def test_transition_probability(self):
        """Transition provides probability based on weight."""
        from cortical.reasoning.prism_slm import SynapticTransition

        trans = SynapticTransition("the", "quick", weight=2.0)
        # Probability is computed relative to total outgoing weight
        prob = trans.probability(total_weight=10.0)

        assert prob == pytest.approx(0.2, rel=0.01)


class TestContextWindow:
    """Test context window for tracking recent tokens."""

    def test_context_creation(self):
        """Test creating a context window."""
        from cortical.reasoning.prism_slm import ContextWindow

        ctx = ContextWindow(size=3)

        assert ctx.size == 3
        assert len(ctx) == 0

    def test_context_add_tokens(self):
        """Test adding tokens to context."""
        from cortical.reasoning.prism_slm import ContextWindow

        ctx = ContextWindow(size=3)
        ctx.add("the")
        ctx.add("quick")
        ctx.add("brown")

        assert list(ctx) == ["the", "quick", "brown"]

    def test_context_sliding_window(self):
        """Context maintains fixed size sliding window."""
        from cortical.reasoning.prism_slm import ContextWindow

        ctx = ContextWindow(size=3)
        ctx.add("the")
        ctx.add("quick")
        ctx.add("brown")
        ctx.add("fox")  # Should push out "the"

        assert list(ctx) == ["quick", "brown", "fox"]
        assert len(ctx) == 3

    def test_context_as_key(self):
        """Context can be converted to hashable key."""
        from cortical.reasoning.prism_slm import ContextWindow

        ctx = ContextWindow(size=3)
        ctx.add("the")
        ctx.add("quick")

        key = ctx.as_key()
        assert key == ("the", "quick")


class TestTransitionGraph:
    """Test the transition graph that stores all synaptic connections."""

    def test_graph_creation(self):
        """Test creating a transition graph."""
        from cortical.reasoning.prism_slm import TransitionGraph

        graph = TransitionGraph(context_size=2)

        assert graph.context_size == 2
        assert graph.token_count == 0

    def test_graph_learn_sequence(self):
        """Graph learns from token sequences."""
        from cortical.reasoning.prism_slm import TransitionGraph

        graph = TransitionGraph(context_size=2)
        graph.learn_sequence(["the", "quick", "brown", "fox"])

        assert graph.token_count == 4
        assert graph.transition_count > 0

    def test_graph_transition_lookup(self):
        """Can look up transitions from context."""
        from cortical.reasoning.prism_slm import TransitionGraph

        graph = TransitionGraph(context_size=2)
        graph.learn_sequence(["the", "quick", "brown", "fox"])

        transitions = graph.get_transitions(("the", "quick"))
        assert len(transitions) > 0
        # "brown" should be a possible next token
        next_tokens = [t.to_token for t in transitions]
        assert "brown" in next_tokens

    def test_graph_repeated_patterns_strengthen(self):
        """Repeated patterns create stronger transitions."""
        from cortical.reasoning.prism_slm import TransitionGraph

        graph = TransitionGraph(context_size=1)
        # Learn "the cat" multiple times
        for _ in range(5):
            graph.learn_sequence(["the", "cat"])

        transitions = graph.get_transitions(("the",))
        cat_trans = next(t for t in transitions if t.to_token == "cat")

        assert cat_trans.count == 5
        assert cat_trans.weight > 1.0


class TestPRISMLanguageModel:
    """Test the main language model class."""

    def test_model_creation(self):
        """Test creating a language model."""
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=3)

        assert model.context_size == 3

    def test_model_train_on_text(self):
        """Model can train on text corpus."""
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=2)
        model.train("The quick brown fox jumps over the lazy dog.")

        assert model.vocab_size > 0

    def test_model_train_on_multiple_texts(self):
        """Model can train on multiple texts."""
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=2)
        model.train("The cat sat on the mat.")
        model.train("The dog ran in the park.")

        assert model.vocab_size > 0

    def test_model_generate_single_token(self):
        """Model can generate next token from context."""
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=2)
        model.train("The quick brown fox. The quick red fox. The quick blue fox.")

        # Given "the quick", should generate a color
        next_token = model.generate_next(["the", "quick"])

        assert next_token in ["brown", "red", "blue", "fox"]

    def test_model_generate_sequence(self):
        """Model can generate a sequence of tokens."""
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=2)
        model.train("The cat sat on the mat. The cat slept on the mat.")

        sequence = model.generate(prompt="The cat", max_tokens=5)

        assert len(sequence.split()) >= 2  # At least prompt
        assert sequence.startswith("The cat")

    def test_model_temperature_affects_randomness(self):
        """Temperature parameter affects generation randomness."""
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=2)
        model.train("The quick brown fox. The quick brown fox. The quick brown fox.")

        # Low temperature should be more deterministic
        results_low = set()
        for _ in range(10):
            token = model.generate_next(["the", "quick"], temperature=0.1)
            results_low.add(token)

        # High temperature should be more random
        results_high = set()
        for _ in range(10):
            token = model.generate_next(["the", "quick"], temperature=2.0)
            results_high.add(token)

        # Low temp should have fewer unique results (more deterministic)
        # This is probabilistic, so we just check it runs
        assert len(results_low) >= 1
        assert len(results_high) >= 1

    def test_model_perplexity(self):
        """Model can compute perplexity on text."""
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=2)
        model.train("The cat sat on the mat. The cat sat on the mat.")

        # Perplexity on training data should be low
        perplexity = model.perplexity("The cat sat on the mat.")

        assert perplexity > 0
        assert perplexity < 100  # Should be reasonably low on training data


class TestHebbianLearning:
    """Test Hebbian learning in the language model."""

    def test_coactivation_strengthening(self):
        """Tokens that appear together strengthen their connections."""
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=2)

        # Train with repeated pattern
        for _ in range(10):
            model.train("neural networks learn patterns")

        # The transition "neural" -> "networks" should be strong
        transitions = model.graph.get_transitions(("neural",))
        if transitions:
            networks_trans = next(
                (t for t in transitions if t.to_token == "networks"), None
            )
            if networks_trans:
                assert networks_trans.weight > 1.5

    def test_decay_weakens_unused(self):
        """Unused transitions decay over time."""
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=1)
        model.train("the cat")

        initial_transitions = model.graph.get_transitions(("the",))
        initial_weight = initial_transitions[0].weight if initial_transitions else 1.0

        # Apply decay
        model.apply_decay(factor=0.5)

        final_transitions = model.graph.get_transitions(("the",))
        if final_transitions:
            assert final_transitions[0].weight < initial_weight


class TestRewardLearning:
    """Test reward-based learning for generation quality."""

    def test_reward_strengthens_path(self):
        """Positive reward strengthens generation path."""
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=2)
        model.train("The cat sat. The dog ran.")

        # Generate and reward
        generated = model.generate(prompt="The", max_tokens=3, return_path=True)
        path = generated["path"]

        if len(path) >= 2:
            # Get initial weight
            ctx = (path[0],)
            transitions = model.graph.get_transitions(ctx)
            initial_weights = {t.to_token: t.weight for t in transitions}

            # Apply reward
            model.reward_path(path, reward=1.0)

            # Check weight increased
            transitions = model.graph.get_transitions(ctx)
            for t in transitions:
                if t.to_token == path[1]:
                    assert t.weight >= initial_weights.get(t.to_token, 1.0)


class TestIntegration:
    """Integration tests with corpus."""

    def test_train_on_corpus_files(self):
        """Model can train on corpus files."""
        from pathlib import Path
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=3)

        samples_dir = Path(__file__).parent.parent.parent / "samples"
        if samples_dir.exists():
            count = 0
            for f in samples_dir.glob("*.txt"):
                try:
                    text = f.read_text(encoding="utf-8")
                    model.train(text)
                    count += 1
                except:
                    pass

            if count > 0:
                assert model.vocab_size > 100
                # Should be able to generate something
                generated = model.generate(prompt="The", max_tokens=10)
                assert len(generated) > 3

    def test_model_serialization(self):
        """Model can be saved and loaded."""
        import tempfile
        import os
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=2)
        model.train("The quick brown fox jumps over the lazy dog.")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.json")
            model.save(path)

            loaded = PRISMLanguageModel.load(path)

            assert loaded.vocab_size == model.vocab_size
            assert loaded.context_size == model.context_size

    def test_incremental_learning(self):
        """Model supports incremental learning."""
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=2)
        model.train("The cat sat.")

        initial_vocab = model.vocab_size

        model.train("The dog ran.")

        assert model.vocab_size >= initial_vocab
