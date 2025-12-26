"""
Integration tests for PRISM-SLM with real corpus files.

Moved from tests/unit/test_prism_slm.py::TestIntegration::test_train_on_corpus_files
because it reads actual files from samples/ directory.
"""

import pytest
from pathlib import Path


class TestPRISMCorpusIntegration:
    """Integration tests that use real corpus files."""

    def test_train_on_corpus_files(self):
        """Model can train on corpus files.

        This test reads actual files from samples/ directory,
        making it an integration test rather than a unit test.
        """
        from cortical.reasoning.prism_slm import PRISMLanguageModel

        model = PRISMLanguageModel(context_size=3)

        samples_dir = Path(__file__).parent.parent.parent / "samples"
        if not samples_dir.exists():
            pytest.skip("samples/ directory not found")

        count = 0
        for f in samples_dir.glob("*.txt"):
            try:
                text = f.read_text(encoding="utf-8")
                model.train(text)
                count += 1
            except Exception:
                pass

        if count == 0:
            pytest.skip("No .txt files found in samples/")

        assert model.vocab_size > 100
        # Should be able to generate something
        generated = model.generate(prompt="The", max_tokens=10)
        assert len(generated) > 3
