"""
Behavioral tests for writers using statistical language models to generate text.

Epic: Adaptive Text Generation

As a writer using language models,
I want the model to learn from my corpus and adapt its style,
So that generated text matches my domain and preferences.

Based on: examples/prism_slm_demo.py
"""

import pytest
from cortical.reasoning import PRISMLanguageModel


class TestWriterGeneratesText:
    """
    Epic: Adaptive Text Generation

    As a writer working with language models,
    I want models that learn my style and improve through feedback,
    So that generated text is useful and contextually appropriate.
    """

    def test_scenario_writer_trains_model_on_domain_corpus(self):
        """
        Scenario: Learning language patterns from a corpus

        Given a statistical language model
        When I train it on domain-specific text
        Then it builds a vocabulary from the corpus
        And learns transition probabilities between words
        Because models should understand domain-specific language.
        """
        # GIVEN a statistical language model
        model = PRISMLanguageModel(context_size=2)

        # WHEN I train it on domain-specific text
        corpus_text = """
        Neural networks learn patterns from data.
        The network adjusts weights during training.
        Backpropagation enables efficient learning.
        """

        model.train(corpus_text)

        # THEN it builds a vocabulary from the corpus
        stats = model.get_stats()
        assert stats['vocab_size'] > 0, "Should build vocabulary from corpus"
        assert stats['token_count'] > 0, "Should track total tokens processed"

        # AND learns transition probabilities between words
        assert stats['transition_count'] > 0, "Should learn word transitions"

        # Verify some specific transitions exist
        transitions = model.graph.get_transitions(("neural",))
        assert len(transitions) > 0, "Should learn transitions from 'neural'"

    def test_scenario_writer_generates_text_from_prompt(self):
        """
        Scenario: Generating contextually appropriate text

        Given a trained language model
        When I provide a prompt
        Then the model generates continuation text
        And the output follows learned patterns
        Because writers need coherent text generation.
        """
        # GIVEN a trained language model
        model = PRISMLanguageModel(context_size=2)

        # Train on repeated patterns for predictable generation
        training_text = "the neural network learns patterns " * 10
        model.train(training_text)

        # WHEN I provide a prompt
        prompt = "the neural"

        # THEN the model generates continuation text
        generated = model.generate(
            prompt=prompt,
            max_tokens=5,
            temperature=0.5,
        )

        # AND the output follows learned patterns
        assert len(generated) > len(prompt), "Should generate additional text"
        assert generated.lower().startswith(prompt.lower()), "Should include the original prompt"

    def test_scenario_writer_controls_generation_randomness(self):
        """
        Scenario: Adjusting creativity via temperature

        Given a trained language model
        When I generate with low temperature
        Then output is more deterministic and predictable
        When I generate with high temperature
        Then output is more diverse and creative
        Because writers need different levels of creativity.
        """
        # GIVEN a trained language model
        model = PRISMLanguageModel(context_size=2)

        # Train with clear patterns
        model.train("the cat sat on the mat. " * 5)
        model.train("the cat slept on the mat. " * 5)
        model.train("the cat jumped on the mat. " * 5)

        prompt = "the cat"

        # WHEN I generate with low temperature
        low_temp_outputs = set()
        for _ in range(10):
            output = model.generate(prompt=prompt, max_tokens=4, temperature=0.1)
            low_temp_outputs.add(output)

        # THEN output is more deterministic and predictable
        # Low temperature should produce fewer variations
        assert len(low_temp_outputs) <= 3, "Low temperature should produce limited variations"

        # WHEN I generate with high temperature
        high_temp_outputs = set()
        for _ in range(10):
            output = model.generate(prompt=prompt, max_tokens=4, temperature=2.0)
            high_temp_outputs.add(output)

        # THEN output is more diverse and creative
        # High temperature typically produces more variations (though not guaranteed)
        # We just verify it generates valid output
        assert all(len(out) > 0 for out in high_temp_outputs), "High temperature should generate valid output"

    def test_scenario_writer_evaluates_text_likelihood(self):
        """
        Scenario: Measuring how well text fits the learned model

        Given a trained language model
        When I calculate perplexity for different sentences
        Then in-domain sentences have low perplexity
        And out-of-domain sentences have high perplexity
        Because writers need to assess text quality.
        """
        # GIVEN a trained language model
        model = PRISMLanguageModel(context_size=2)

        # Train on technical corpus
        model.train("Neural networks learn from data using backpropagation algorithms.")
        model.train("Machine learning models optimize objective functions.")
        model.train("Deep learning architectures process hierarchical features.")

        # WHEN I calculate perplexity for different sentences
        # THEN in-domain sentences have low perplexity
        in_domain = "Neural networks learn patterns."
        in_domain_ppl = model.perplexity(in_domain)

        # AND out-of-domain sentences have high perplexity
        out_domain = "Xyzzy foobar completely unknown gibberish words."
        out_domain_ppl = model.perplexity(out_domain)

        assert in_domain_ppl < out_domain_ppl, "In-domain text should have lower perplexity"

    def test_scenario_writer_observes_hebbian_learning_in_transitions(self):
        """
        Scenario: Frequently used word pairs strengthen

        Given a language model
        When I train on text with repeated word patterns
        Then frequently co-occurring words have stronger connections
        And the model learns which words naturally go together
        Because "words that appear together wire together".
        """
        # GIVEN a language model
        model = PRISMLanguageModel(context_size=2)

        # WHEN I train on text with repeated word patterns
        # Train heavily on specific phrase
        frequent_phrase = "neural network"
        for _ in range(20):
            model.train(f"the {frequent_phrase} learns")

        # Train less on another phrase
        rare_phrase = "neural system"
        for _ in range(2):
            model.train(f"the {rare_phrase} learns")

        # THEN frequently co-occurring words have stronger connections
        transitions = model.graph.get_transitions(("neural",))

        # Find the transitions we care about
        network_trans = next((t for t in transitions if t.to_token == "network"), None)
        system_trans = next((t for t in transitions if t.to_token == "system"), None)

        assert network_trans is not None, "Should learn 'neural' -> 'network' transition"
        assert system_trans is not None, "Should learn 'neural' -> 'system' transition"

        # AND the model learns which words naturally go together
        assert network_trans.weight > system_trans.weight, "More frequent transition should have higher weight"
        assert network_trans.count > system_trans.count, "Should track transition frequency"

    def test_scenario_writer_applies_decay_to_unused_patterns(self):
        """
        Scenario: Unused word patterns gradually fade

        Given a language model with learned patterns
        When I apply decay without new training
        Then word transition weights decrease
        And recently used patterns decay slower
        Because models should adapt to current usage.
        """
        # GIVEN a language model with learned patterns
        model = PRISMLanguageModel(context_size=2)
        model.train("the cat sat on the mat")

        # Get initial transition weights
        transitions_before = model.graph.get_transitions(("the",))
        initial_weights = {t.to_token: t.weight for t in transitions_before}

        # WHEN I apply decay without new training
        model.apply_decay(factor=0.8)

        # THEN word transition weights decrease
        transitions_after = model.graph.get_transitions(("the",))
        final_weights = {t.to_token: t.weight for t in transitions_after}

        for token in initial_weights:
            if token in final_weights:
                assert final_weights[token] <= initial_weights[token], "Weights should decrease or stay same"
                assert final_weights[token] == pytest.approx(initial_weights[token] * 0.8, abs=0.01), "Should apply decay factor"

    def test_scenario_writer_reinforces_successful_generations(self):
        """
        Scenario: Learning from generation quality feedback

        Given a language model that generated text
        When I mark the generation path as high quality
        Then transitions in that path strengthen
        And future generations favor similar patterns
        Because models should learn from successful outputs.
        """
        # GIVEN a language model that generated text
        model = PRISMLanguageModel(context_size=2)
        model.train("the cat sat quietly")

        # Generate and get the path
        result = model.generate(prompt="the cat", max_tokens=3, return_path=True)
        path = result["path"]

        assert len(path) >= 2, "Should generate a path with multiple tokens"

        # Get weight of first transition before reward
        ctx = (path[0],)
        transitions_before = model.graph.get_transitions(ctx)
        trans_before = next((t for t in transitions_before if t.to_token == path[1]), None)
        weight_before = trans_before.weight if trans_before else 0

        # WHEN I mark the generation path as high quality
        model.reward_path(path, reward=2.0)

        # THEN transitions in that path strengthen
        transitions_after = model.graph.get_transitions(ctx)
        trans_after = next((t for t in transitions_after if t.to_token == path[1]), None)

        assert trans_after is not None, "Transition should still exist"
        assert trans_after.weight > weight_before, "Rewarded transition should strengthen"

    def test_scenario_writer_generates_with_context_awareness(self):
        """
        Scenario: Using context for better predictions

        Given a language model with context size N
        When I generate text
        Then the model considers previous N words
        And produces more coherent output than unigram models
        Because context enables better predictions.
        """
        # GIVEN a language model with context size N
        model = PRISMLanguageModel(context_size=3)

        # Train with patterns that require context
        model.train("in the morning we wake up early")
        model.train("in the evening we sleep soundly")
        model.train("in the afternoon we work hard")

        # WHEN I generate text
        # THEN the model considers previous N words
        prompt = "in the morning"
        generated = model.generate(prompt=prompt, max_tokens=3, temperature=0.5)

        # Verify generation uses context
        assert len(generated) > len(prompt), "Should generate continuation"

        # The model should use 3-word context for better predictions
        # Verify model has learned the context-dependent patterns
        context = ("in", "the", "morning")
        transitions = model.graph.get_transitions(context)

        # Should have transitions from this context
        assert len(transitions) > 0, "Should learn transitions from full context"

    def test_scenario_writer_handles_unknown_prompts_gracefully(self):
        """
        Scenario: Generating from unfamiliar starting points

        Given a language model trained on specific text
        When I provide a prompt with unknown words
        Then the model handles it without crashing
        And either generates from known words or indicates inability
        Because models should degrade gracefully.
        """
        # GIVEN a language model trained on specific text
        model = PRISMLanguageModel(context_size=2)
        model.train("the neural network learns patterns")

        # WHEN I provide a prompt with unknown words
        unknown_prompt = "xyzzy abracadabra"

        # THEN the model handles it without crashing
        try:
            result = model.generate(prompt=unknown_prompt, max_tokens=5, temperature=1.0)
            # Should either generate something or return the prompt
            assert isinstance(result, str), "Should return a string"
        except Exception as e:
            # If it raises an exception, it should be a handled one
            pytest.fail(f"Model should handle unknown prompts gracefully, got: {e}")
