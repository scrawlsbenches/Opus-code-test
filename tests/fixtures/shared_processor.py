"""
Shared Processor with Full Sample Corpus
=========================================

A singleton processor loaded with the full samples/ directory.
Used for integration and behavioral tests that need realistic data.

This is slower to initialize (~10-20s) but is shared across all tests
that need it, so the cost is paid only once per test run.

Usage:
    from tests.fixtures.shared_processor import get_shared_processor

    processor = get_shared_processor()  # Returns cached instance
"""

import os
import sys

# Ensure cortical is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cortical import CorticalTextProcessor
from cortical.tokenizer import Tokenizer


# Module-level singleton
_SHARED_PROCESSOR = None
_SHARED_PROCESSOR_INITIALIZED = False


def get_shared_processor(force_reload: bool = False) -> CorticalTextProcessor:
    """
    Get or create the shared processor with full sample corpus.

    This singleton ensures we only load the corpus once per test run,
    dramatically reducing test time when multiple tests need the full corpus.

    Args:
        force_reload: If True, recreate the processor even if cached

    Returns:
        CorticalTextProcessor with full samples/ corpus loaded and computed
    """
    global _SHARED_PROCESSOR, _SHARED_PROCESSOR_INITIALIZED

    if _SHARED_PROCESSOR_INITIALIZED and not force_reload:
        return _SHARED_PROCESSOR

    # Find samples directory
    tests_dir = os.path.dirname(__file__)
    samples_dir = os.path.join(tests_dir, '..', '..', 'samples')
    samples_dir = os.path.abspath(samples_dir)

    if not os.path.isdir(samples_dir):
        raise RuntimeError(
            f"Samples directory not found: {samples_dir}\n"
            "The shared processor requires the samples/ directory."
        )

    # Create processor with code noise filtering
    tokenizer = Tokenizer(filter_code_noise=True)
    processor = CorticalTextProcessor(tokenizer=tokenizer)

    # Load all sample files (including subdirectories)
    loaded_count = 0
    for root, dirs, files in os.walk(samples_dir):
        # Skip hidden directories and special folders
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

        for filename in sorted(files):
            # Skip hidden files and non-text files
            if filename.startswith('.') or filename.endswith('.pyc'):
                continue
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Create doc_id from relative path (without extension)
                rel_path = os.path.relpath(filepath, samples_dir)
                doc_id = os.path.splitext(rel_path)[0].replace(os.sep, '/')
                processor.process_document(doc_id, content)
                loaded_count += 1
            except (IOError, UnicodeDecodeError):
                # Skip files that can't be read
                continue

    if loaded_count == 0:
        raise RuntimeError(
            f"No documents loaded from {samples_dir}\n"
            "Check that samples/ contains readable text files."
        )

    # Compute all network properties
    processor.compute_all(verbose=False)

    _SHARED_PROCESSOR = processor
    _SHARED_PROCESSOR_INITIALIZED = True

    return processor


def reset_shared_processor():
    """Reset the singleton so next get_shared_processor() creates fresh instance."""
    global _SHARED_PROCESSOR, _SHARED_PROCESSOR_INITIALIZED
    _SHARED_PROCESSOR = None
    _SHARED_PROCESSOR_INITIALIZED = False


def get_samples_dir() -> str:
    """Get the absolute path to the samples directory."""
    tests_dir = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(tests_dir, '..', '..', 'samples'))
