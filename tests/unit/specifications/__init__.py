"""
Unit Specifications - Precise Facts About Atomic Behavior

This module contains unit specifications that document the precise behavior
of individual components. Each specification is a fact about the system
that must remain true.

Specifications are different from regular unit tests:
- Regular tests verify implementation details that may change
- Specifications document LOAD-BEARING behavior that must never change

If you need to change a specification:
1. Understand WHY it was specified that way
2. Assess backward compatibility impact
3. Update the specification WITH documentation
4. Ensure dependent code still works

Example specification:
    def spec_bigrams_use_space_separator(self):
        '''
        SPECIFICATION: Bigrams are joined with spaces, never underscores.

        This is load-bearing behavior. Changing it breaks persistence
        compatibility and query expansion.
        '''
        tokenizer = Tokenizer()
        bigrams = tokenizer.extract_bigrams(["neural", "networks"])
        assert "neural networks" in bigrams
        assert "neural_networks" not in bigrams
"""
