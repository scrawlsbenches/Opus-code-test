from typing import Dict, List, Tuple, Optional
import random
import math

class CommentMarkovChain:
    def __init__(self):
        """Initialize empty Markov chain."""
        self._transitions: Dict[str, Dict[str, int]] = {}  # word -> {next_word: count}
        self._totals: Dict[str, int] = {}  # word -> total outgoing count

    def train(self, sequences: List[List[str]]) -> None:
        """
        Learn transitions from tokenized comment sequences.
        Training is ADDITIVE - multiple calls accumulate.
        Each sequence is a list of words from a comment.
        """
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                from_word = sequence[i]
                to_word = sequence[i + 1]

                # Initialize if needed
                if from_word not in self._transitions:
                    self._transitions[from_word] = {}
                    self._totals[from_word] = 0

                # Increment count (additive)
                if to_word not in self._transitions[from_word]:
                    self._transitions[from_word][to_word] = 0
                self._transitions[from_word][to_word] += 1
                self._totals[from_word] += 1

    def probability(self, from_word: str, to_word: str) -> float:
        """
        Return P(to_word | from_word).
        Returns 0.0 if transition was never observed.

        Probability normalization:
        P(to_word | from_word) = count(from_word -> to_word) / total_transitions_from(from_word)
        """
        if from_word not in self._transitions:
            return 0.0
        if to_word not in self._transitions[from_word]:
            return 0.0

        count = self._transitions[from_word][to_word]
        total = self._totals[from_word]
        return count / total

    def most_likely_next(self, word: str) -> Optional[str]:
        """
        Return most probable next word.
        Returns None if word has no outgoing transitions.

        Tie-breaking logic:
        When multiple words have same probability, return lexicographically smallest.
        This ensures deterministic behavior.
        """
        if word not in self._transitions or not self._transitions[word]:
            return None

        # Find maximum count
        max_count = max(self._transitions[word].values())

        # Find all words with max count, then return smallest alphabetically
        candidates = [w for w, count in self._transitions[word].items() if count == max_count]
        return min(candidates)  # Lexicographically smallest

    def likely_patterns(self, word: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Return top N most likely next words with probabilities.
        Sorted by probability descending, then alphabetically.
        """
        if word not in self._transitions or not self._transitions[word]:
            return []

        # Calculate probabilities for all transitions
        patterns = []
        for next_word in self._transitions[word]:
            prob = self.probability(word, next_word)
            patterns.append((next_word, prob))

        # Sort by probability descending, then alphabetically for ties
        patterns.sort(key=lambda x: (-x[1], x[0]))

        return patterns[:top_n]

    def generate(self, start: str, length: int) -> List[str]:
        """
        Generate sequence of given length starting from start word.

        Weighted random selection:
        At each step, choose next word with probability proportional to
        transition counts. This preserves the learned distribution.

        Uses random.choices with weights for proper probability distribution.
        Returns partial sequence if chain reaches dead end.
        """
        # Handle edge case: length 0
        if length == 0:
            return []

        sequence = [start]
        current = start

        for _ in range(length - 1):
            if current not in self._transitions or not self._transitions[current]:
                # Dead end - return partial sequence
                break

            # Get all possible next words and their counts
            next_words = list(self._transitions[current].keys())
            weights = [self._transitions[current][w] for w in next_words]

            # Weighted random selection
            next_word = random.choices(next_words, weights=weights, k=1)[0]
            sequence.append(next_word)
            current = next_word

        return sequence

    def transitions_from(self, word: str) -> Dict[str, float]:
        """
        Return all transitions from word as {next_word: probability}.
        Probabilities are normalized to sum to 1.0.
        """
        if word not in self._transitions or not self._transitions[word]:
            return {}

        result = {}
        total = self._totals[word]
        for next_word, count in self._transitions[word].items():
            result[next_word] = count / total

        return result

    def pattern_score(self, sequence: List[str]) -> float:
        """
        Score a sequence by average transition probability.
        Higher score = more likely pattern based on training data.
        Returns 0.0 for empty sequence or single word.

        Formula: average of P(word[i+1] | word[i]) for all transitions
        """
        if len(sequence) <= 1:
            return 0.0

        total_prob = 0.0
        num_transitions = 0

        for i in range(len(sequence) - 1):
            prob = self.probability(sequence[i], sequence[i + 1])
            total_prob += prob
            num_transitions += 1

        return total_prob / num_transitions if num_transitions > 0 else 0.0
