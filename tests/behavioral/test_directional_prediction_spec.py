"""
BDD Specification: Directional Prediction with Uncertainty

Epic: Honest Next-Word Prediction from Graph Structure

As a cognitive agent processing natural language,
I want to predict the next word using directional transitions,
So that I can generate text while honestly representing uncertainty.

Background:
- SIMILARITY links are bidirectional (A ~ B) - good for semantic expansion (NLU)
- FOLLOWS links are directional (A -> B) - needed for prediction (NLG)
- Small corpus means sparse transitions - "I don't know" is often the honest answer
- Graph topology encodes confidence: peaked distribution = certain, flat = uncertain
- Natural boundaries exist: some words don't strongly predict forward

Design Decisions:
1. FOLLOWS LINKS (DIRECTIONAL)
   - Created for adjacent word pairs only (window=1)
   - Store transition count for probability calculation
   - Separate from SIMILARITY (different purpose, different structure)

2. PREDICTION WITH UNCERTAINTY
   - Returns candidates with probabilities
   - Confidence score: how peaked is the distribution?
   - is_boundary: few/weak outgoing transitions = natural stopping point
   - is_unknown: word not in vocabulary = honest "never seen this"

3. CONFIDENCE CALCULATION
   - High: one candidate dominates (>60% probability)
   - Medium: clear winner but alternatives exist
   - Low: flat distribution, multiple equally likely options
   - Zero: no transitions or unknown word

4. SMALL CORPUS FRIENDLY
   - Sparse data -> low confidence (not forced predictions)
   - Unknown words -> explicit is_unknown flag
   - Boundaries are common and valid

Acceptance Criteria:
[ ] FOLLOWS links created during training (adjacent pairs only)
[ ] Prediction returns candidates with probabilities
[ ] Confidence reflects distribution sharpness
[ ] is_boundary detects natural stopping points
[ ] is_unknown for out-of-vocabulary words
[ ] Small corpus produces honest uncertainty
"""

import pytest
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class Prediction:
    """
    Result of predicting the next atom.

    Attributes:
        candidates: List of (word, probability) pairs, sorted by probability desc
        confidence: 0.0-1.0, how certain the prediction is
        is_boundary: True if this is a natural stopping point
        is_unknown: True if the input word was not in vocabulary
    """
    candidates: List[Tuple[str, float]]
    confidence: float
    is_boundary: bool
    is_unknown: bool

    @property
    def top(self) -> Optional[str]:
        """Get top prediction, or None if unknown/boundary."""
        if self.is_unknown or not self.candidates:
            return None
        return self.candidates[0][0]

    @property
    def top_probability(self) -> float:
        """Get probability of top prediction."""
        if not self.candidates:
            return 0.0
        return self.candidates[0][1]


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def prediction_agent(memory_cognitive_container):
    """CognitiveAgent with FOLLOWS link support via DI container."""
    from cortical.cognitive.training import IncrementalTrainer

    trainer = memory_cognitive_container.resolve(IncrementalTrainer)
    return PredictionAgentWrapper(trainer)


class PredictionAgentWrapper:
    """
    Test wrapper providing prediction capabilities.

    Wraps trainer/agent to expose prediction API for testing.
    """

    def __init__(self, trainer: 'IncrementalTrainer'):
        self._trainer = trainer
        self._agent = trainer.agent
        self._bridge = trainer.bridge

    def train(self, texts: List[str]) -> None:
        """Train on texts, creating both SIMILARITY and FOLLOWS links."""
        self._bridge.learn_vocabulary(texts)
        for i, text in enumerate(texts):
            self._bridge.feed_text(text, doc_id=f"doc_{i}")
            self._trainer.manifest.total_documents += 1
        self._trainer.save()

    def predict_next(self, word: str, top_k: int = 10) -> Prediction:
        """Predict next word with uncertainty quantification."""
        return self._agent.predict_next(word, top_k=top_k)

    def get_follows_count(self, word: str) -> int:
        """Get number of FOLLOWS links from this word."""
        return self._agent.get_follows_count(word)


# =============================================================================
# STORY 1: FOLLOWS LINK CREATION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.prediction
class TestFollowsLinkCreation:
    """
    Story 1: FOLLOWS links created during training

    As a training system,
    I want to create directional FOLLOWS links for adjacent word pairs,
    So that prediction has transition data to work with.
    """

    def test_adjacent_pairs_create_follows_links(self, prediction_agent):
        """
        Scenario: Adjacent words create FOLLOWS links

        Given text "the quick brown fox"
        When training completes
        Then FOLLOWS links exist: the->quick, quick->brown, brown->fox
        And no FOLLOWS link exists: the->brown (not adjacent)

        Because FOLLOWS captures sequential transitions only.
        """
        # Given
        prediction_agent.train(["the quick brown fox"])

        # Then: Adjacent pairs have FOLLOWS
        pred_the = prediction_agent.predict_next("the")
        pred_quick = prediction_agent.predict_next("quick")
        pred_brown = prediction_agent.predict_next("brown")

        assert "quick" in [c[0] for c in pred_the.candidates]
        assert "brown" in [c[0] for c in pred_quick.candidates]
        assert "fox" in [c[0] for c in pred_brown.candidates]

        # And: Non-adjacent pairs don't have direct FOLLOWS
        # "the" should not directly predict "brown" or "fox"
        the_predictions = [c[0] for c in pred_the.candidates]
        assert "brown" not in the_predictions or pred_the.candidates[0][0] == "quick"

    def test_repeated_transitions_increase_probability(self, prediction_agent):
        """
        Scenario: Repeated transitions strengthen prediction

        Given text with repeated "neural network" pattern
        When training completes
        Then "neural" -> "network" has higher probability than other transitions

        Because frequency indicates likelihood.
        """
        # Given: "neural network" appears 5 times, other transitions once
        prediction_agent.train([
            "neural network",
            "neural network architecture",
            "deep neural network",
            "neural network training",
            "neural network model",
            "neural pathway different",  # "neural" -> "pathway" only once
        ])

        # When
        pred = prediction_agent.predict_next("neural")

        # Then: "network" dominates
        assert pred.candidates[0][0] == "network"
        assert pred.candidates[0][1] > 0.5  # Majority probability


# =============================================================================
# STORY 2: CONFIDENT PREDICTION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.prediction
class TestConfidentPrediction:
    """
    Story 2: High confidence when distribution is peaked

    As a prediction consumer,
    I want to know when the agent is confident,
    So that I can trust the prediction.
    """

    def test_single_dominant_transition_high_confidence(self, prediction_agent):
        """
        Scenario: Single strong transition yields high confidence

        Given "neural" always followed by "network" in training
        When predicting next word after "neural"
        Then confidence should be high (>0.7)
        And top prediction should be "network"

        Because consistent patterns warrant confidence.
        """
        # Given: Consistent pattern
        prediction_agent.train([
            "neural network",
            "neural network",
            "neural network",
        ])

        # When
        pred = prediction_agent.predict_next("neural")

        # Then
        assert pred.confidence > 0.7
        assert pred.top == "network"
        assert not pred.is_unknown
        assert not pred.is_boundary

    def test_confidence_reflects_distribution_sharpness(self, prediction_agent):
        """
        Scenario: Peaked distribution = high confidence

        Given word A always followed by B (peaked)
        And word X followed by Y, Z, W equally (flat)
        When predicting from A vs X
        Then A's confidence > X's confidence

        Because entropy measures uncertainty.
        """
        # Given
        prediction_agent.train([
            # A -> B always (peaked)
            "alpha beta",
            "alpha beta",
            "alpha beta",
            # X -> multiple equally (flat)
            "xray yankee",
            "xray zulu",
            "xray whiskey",
        ])

        # When
        pred_alpha = prediction_agent.predict_next("alpha")
        pred_xray = prediction_agent.predict_next("xray")

        # Then
        assert pred_alpha.confidence > pred_xray.confidence


# =============================================================================
# STORY 3: UNCERTAIN PREDICTION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.prediction
class TestUncertainPrediction:
    """
    Story 3: Low confidence when distribution is flat

    As a prediction consumer,
    I want to know when the agent is uncertain,
    So that I can seek more context or explore alternatives.
    """

    def test_flat_distribution_low_confidence(self, prediction_agent):
        """
        Scenario: Multiple equally likely transitions yield low confidence

        Given "the" followed by many different words equally
        When predicting next word after "the"
        Then confidence should be low (<0.4)
        And candidates should include multiple options

        Because "the" genuinely doesn't predict what follows.
        """
        # Given: "the" followed by many words
        prediction_agent.train([
            "the cat",
            "the dog",
            "the bird",
            "the fish",
            "the tree",
        ])

        # When
        pred = prediction_agent.predict_next("the")

        # Then
        assert pred.confidence < 0.4
        assert len(pred.candidates) >= 3
        # No single candidate dominates
        assert pred.top_probability < 0.4

    def test_uncertainty_is_not_failure(self, prediction_agent):
        """
        Scenario: Uncertainty is a valid, informative response

        Given common word with many transitions
        When prediction returns low confidence
        Then is_unknown should be False (we know the word)
        And candidates should still be provided (options exist)

        Because "I'm not sure" differs from "I don't know".
        """
        # Given
        prediction_agent.train([
            "is good",
            "is bad",
            "is okay",
            "is fine",
        ])

        # When
        pred = prediction_agent.predict_next("is")

        # Then
        assert not pred.is_unknown  # We know "is"
        assert len(pred.candidates) > 0  # Options exist
        assert pred.confidence < 0.5  # But we're uncertain which


# =============================================================================
# STORY 4: BOUNDARY DETECTION
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.prediction
class TestBoundaryDetection:
    """
    Story 4: Detect natural stopping points

    As a text generator,
    I want to know when a phrase is complete,
    So that I can stop gracefully instead of forcing continuation.
    """

    def test_word_with_no_outgoing_is_boundary(self, prediction_agent):
        """
        Scenario: Word at end of all training sentences is boundary

        Given "fox" always appears at end of sentences
        When predicting after "fox"
        Then is_boundary should be True
        And candidates should be empty or very weak

        Because some words are natural endpoints.
        """
        # Given: "fox" is always terminal
        prediction_agent.train([
            "quick brown fox",
            "lazy dog fox",
            "red fox",
        ])

        # When
        pred = prediction_agent.predict_next("fox")

        # Then
        assert pred.is_boundary
        assert pred.confidence == 0.0 or len(pred.candidates) == 0

    def test_phrase_completion_detection(self, prediction_agent):
        """
        Scenario: Complete phrases signal boundaries

        Given training on "neural network" as complete phrase
        When predicting after "network" (in "neural network" context)
        Then is_boundary should indicate phrase is complete

        Because "neural network" is a coherent unit.
        """
        # Given: "network" often terminal in these phrases
        prediction_agent.train([
            "neural network",
            "deep neural network",
            "convolutional neural network",
        ])

        # When
        pred = prediction_agent.predict_next("network")

        # Then: "network" is often a boundary
        assert pred.is_boundary or pred.confidence < 0.3


# =============================================================================
# STORY 5: UNKNOWN WORD HANDLING
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.prediction
class TestUnknownWordHandling:
    """
    Story 5: Honest handling of unknown words

    As a prediction consumer,
    I want explicit indication when a word is unknown,
    So that I don't mistake silence for confidence.
    """

    def test_unknown_word_explicit_flag(self, prediction_agent):
        """
        Scenario: Unknown word sets is_unknown flag

        Given training on limited vocabulary
        When predicting from word not in vocabulary
        Then is_unknown should be True
        And confidence should be 0.0
        And candidates should be empty

        Because honesty about limits builds trust.
        """
        # Given: Limited vocabulary
        prediction_agent.train(["cat dog bird"])

        # When: Query unknown word
        pred = prediction_agent.predict_next("quantum")

        # Then
        assert pred.is_unknown
        assert pred.confidence == 0.0
        assert len(pred.candidates) == 0
        assert pred.top is None

    def test_unknown_differs_from_boundary(self, prediction_agent):
        """
        Scenario: Unknown word vs boundary word

        Given "fox" is known but terminal
        And "quantum" is unknown
        When predicting from both
        Then "fox" is boundary (known, just no continuation)
        And "quantum" is unknown (never seen)

        Because the distinction matters for downstream handling.
        """
        # Given
        prediction_agent.train(["quick brown fox"])

        # When
        pred_fox = prediction_agent.predict_next("fox")
        pred_quantum = prediction_agent.predict_next("quantum")

        # Then
        assert pred_fox.is_boundary and not pred_fox.is_unknown
        assert pred_quantum.is_unknown and not pred_quantum.is_boundary


# =============================================================================
# STORY 6: SMALL CORPUS BEHAVIOR
# =============================================================================


@pytest.mark.behavioral
@pytest.mark.prediction
class TestSmallCorpusBehavior:
    """
    Story 6: Graceful behavior with limited training data

    As a system operating in Claude Code Web with small corpus,
    I want prediction to work honestly with sparse data,
    So that limited data produces uncertainty rather than hallucination.
    """

    def test_single_example_low_confidence(self, prediction_agent):
        """
        Scenario: Single training example yields low confidence

        Given only one example of "neural network"
        When predicting after "neural"
        Then prediction should exist but with moderate confidence

        Because one example isn't strong evidence.
        """
        # Given: Minimal training
        prediction_agent.train(["neural network"])

        # When
        pred = prediction_agent.predict_next("neural")

        # Then: Prediction exists but confidence is tempered
        assert pred.top == "network"
        assert pred.confidence < 0.8  # Not overconfident from one example

    def test_sparse_data_many_boundaries(self, prediction_agent):
        """
        Scenario: Small corpus has many boundary words

        Given very limited training data
        When examining vocabulary
        Then many words will be boundaries
        And this is correct behavior (not a bug)

        Because sparse data means sparse transitions.
        """
        # Given: Tiny corpus
        prediction_agent.train([
            "hello world",
            "foo bar",
        ])

        # When: Check boundary status
        pred_world = prediction_agent.predict_next("world")
        pred_bar = prediction_agent.predict_next("bar")

        # Then: Terminal words are boundaries
        assert pred_world.is_boundary
        assert pred_bar.is_boundary

    def test_no_hallucinated_transitions(self, prediction_agent):
        """
        Scenario: Never predict transitions that weren't seen

        Given training on "A B" and "C D"
        When predicting from "A"
        Then only "B" should be candidate (not "D")

        Because we only predict what we've actually seen.
        """
        # Given: Separate sequences
        prediction_agent.train([
            "alpha beta",
            "gamma delta",
        ])

        # When
        pred_alpha = prediction_agent.predict_next("alpha")
        pred_gamma = prediction_agent.predict_next("gamma")

        # Then: No cross-contamination
        alpha_candidates = [c[0] for c in pred_alpha.candidates]
        gamma_candidates = [c[0] for c in pred_gamma.candidates]

        assert "beta" in alpha_candidates
        assert "delta" not in alpha_candidates
        assert "delta" in gamma_candidates
        assert "beta" not in gamma_candidates


# =============================================================================
# IMPLEMENTATION NOTES
# =============================================================================

"""
Implementation Plan:

1. Add AtomType.FOLLOWS to graph.py (alongside SIMILARITY)

2. In TextToAtomsBridge.feed_text():
   - After tokenizing, iterate adjacent pairs
   - Create FOLLOWS link: word[i] -> word[i+1]
   - Increment count if link exists, create if not

3. In CognitiveAgent, add predict_next() method:

   def predict_next(self, word: str, top_k: int = 10) -> Prediction:
       # Check if word exists
       atom = self.graph.get_node(word)
       if not atom:
           return Prediction([], 0.0, False, is_unknown=True)

       # Get outgoing FOLLOWS links
       follows = self.graph.get_outgoing_follows(atom.id)
       if not follows:
           return Prediction([], 0.0, is_boundary=True, is_unknown=False)

       # Calculate probabilities
       total = sum(link.count for link in follows)
       candidates = [
           (self.graph.get_atom(link.target_id).name, link.count / total)
           for link in follows
       ]
       candidates.sort(key=lambda x: -x[1])

       # Calculate confidence (how peaked is distribution)
       confidence = calculate_confidence(candidates)

       # Boundary if very weak predictions
       is_boundary = (confidence < 0.1 or len(candidates) < 2)

       return Prediction(candidates[:top_k], confidence, is_boundary, False)

4. Confidence calculation:

   def calculate_confidence(candidates: List[Tuple[str, float]]) -> float:
       if not candidates:
           return 0.0
       if len(candidates) == 1:
           return candidates[0][1]  # Single option = its probability

       # How much does top candidate stand out?
       top_prob = candidates[0][1]
       second_prob = candidates[1][1] if len(candidates) > 1 else 0.0

       # Confidence = gap between top and rest
       # High gap = high confidence
       return min(1.0, top_prob - second_prob + top_prob * 0.5)

5. Storage: FOLLOWS links stored in graph.json alongside SIMILARITY
   - Link has: type=FOLLOWS, source_id, target_id, count
   - Count is transition frequency (for probability calculation)
"""
