"""
Sharded Tokenizer Storage

Provides merge-conflict-free storage for BPE tokenizer vocabulary by
sharding words into separate files based on their prefix.

Design Principles:
    1. PREFIX-BASED SHARDING: Words are assigned to shards by their first
       2 characters. "neural" and "network" go to the same shard (ne),
       while "machine" goes to a different shard (ma).

    2. MERGE-CONFLICT-FREE: Since each word always maps to the same shard,
       two developers adding different words modify different files.
       Git can merge these changes automatically.

    3. SPARSE SHARDS: Only create shard files for prefixes that exist.
       Empty prefixes don't create files.

    4. INCREMENTAL SAVES: Track which shards are dirty and only write
       those during save operations.

Directory Structure:
    tokenizer/
    ├── meta.json           # Metadata: vocab size, shard list, version
    ├── vocab_aa.json       # Words starting with "aa"
    ├── vocab_ab.json       # Words starting with "ab"
    ├── vocab_ne.json       # Words starting with "ne" (neural, network)
    ├── vocab__.json        # Words with non-letter prefixes (numbers, etc.)
    └── merges.json         # BPE merge rules (separate file)

Usage:
    from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
    from cortical.common.filesystem import RealFileSystem

    fs = RealFileSystem(base_path)
    storage = ShardedTokenizerStorage(fs)

    # Save tokenizer to sharded directory
    storage.save(tokenizer, model_dir / "tokenizer")

    # Load tokenizer from sharded directory
    tokenizer = storage.load(model_dir / "tokenizer")

    # Migrate from legacy single-file format
    storage.migrate_to_sharded(model_dir)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from cortical.cognitive.text_bridge import BPETokenizer
    from cortical.common.filesystem import FileSystem


class ShardedTokenizerStorage:
    """
    Manages sharded storage for BPE tokenizer vocabulary.

    Shards vocabulary by 2-character prefix to enable:
    - Merge-conflict-free concurrent updates
    - Incremental saves (only dirty shards)
    - Efficient loading (can load specific shards)

    Attributes:
        filesystem: FileSystem abstraction for I/O
        _loaded_shards: Cache of loaded shard data for dirty tracking
    """

    VERSION = "1.0"
    SHARD_PREFIX_LENGTH = 2

    def __init__(self, filesystem: 'FileSystem'):
        """
        Initialize storage with filesystem.

        Args:
            filesystem: FileSystem for I/O operations
        """
        self.filesystem = filesystem
        self._loaded_shards: Dict[str, Dict[str, int]] = {}

    def _get_prefix(self, word: str) -> str:
        """
        Get the shard prefix for a word.

        Uses first 2 characters, lowercased. For words shorter than 2 chars
        or starting with non-letters, uses special prefix "__".

        Args:
            word: The vocabulary word

        Returns:
            2-character prefix for shard assignment
        """
        if len(word) < 2:
            return "__"  # Single char words go to special shard

        prefix = word[:2].lower()

        # Check if prefix is letters only
        if prefix.isalpha():
            return prefix

        # Non-letter prefixes (numbers, punctuation) go to special shard
        return "__"

    def _get_shard_filename(self, prefix: str) -> str:
        """
        Get the filename for a shard prefix.

        Args:
            prefix: 2-character prefix

        Returns:
            Filename like "vocab_ne.json"
        """
        return f"vocab_{prefix}.json"

    def _group_vocab_by_prefix(self, vocab: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """
        Group vocabulary entries by their prefix.

        Args:
            vocab: Full vocabulary dict {word: index}

        Returns:
            Dict of {prefix: {word: index}}
        """
        shards: Dict[str, Dict[str, int]] = {}

        for word, index in vocab.items():
            prefix = self._get_prefix(word)
            if prefix not in shards:
                shards[prefix] = {}
            shards[prefix][word] = index

        return shards

    def save(self, tokenizer: 'BPETokenizer', tokenizer_dir: Path) -> None:
        """
        Save tokenizer to sharded directory structure.

        Creates:
        - meta.json with metadata
        - vocab_XX.json files for each prefix
        - merges.json for BPE merge rules

        Args:
            tokenizer: BPETokenizer to save
            tokenizer_dir: Directory to save to
        """
        # Ensure directory exists
        self.filesystem.mkdir(tokenizer_dir, parents=True, exist_ok=True)

        # Convert vocab set to dict with indices
        if isinstance(tokenizer.vocab, set):
            vocab_dict = {word: i for i, word in enumerate(sorted(tokenizer.vocab))}
        else:
            vocab_dict = dict(tokenizer.vocab)

        # Group by prefix
        shards = self._group_vocab_by_prefix(vocab_dict)

        # Save each shard
        shard_files = []
        for prefix, shard_vocab in sorted(shards.items()):
            shard_filename = self._get_shard_filename(prefix)
            shard_path = tokenizer_dir / shard_filename
            self.filesystem.write_text(shard_path, json.dumps(shard_vocab, indent=2))
            shard_files.append(shard_filename)

        # Save merges separately
        # Handle both formats: [(pair, merged)] and [pair, pair, ...]
        merges_data = []
        for merge in tokenizer.merges:
            if isinstance(merge, tuple) and len(merge) == 2:
                # Check if it's (pair, merged_result) or just (char1, char2)
                if isinstance(merge[0], tuple):
                    # Format: ((char1, char2), merged_result)
                    pair, merged = merge
                    merges_data.append([[pair[0], pair[1]], merged])
                else:
                    # Format: (char1, char2) - simple pair without merged result
                    merges_data.append([merge[0], merge[1]])
            else:
                # Unknown format, try to serialize as-is
                merges_data.append(list(merge) if hasattr(merge, '__iter__') else merge)

        self.filesystem.write_text(
            tokenizer_dir / "merges.json",
            json.dumps(merges_data, indent=2)
        )

        # Save word counts if present
        if hasattr(tokenizer, '_word_counts') and tokenizer._word_counts:
            word_counts_shards = self._group_vocab_by_prefix(
                {word: count for word, count in tokenizer._word_counts.items()}
            )
            for prefix, counts in word_counts_shards.items():
                counts_path = tokenizer_dir / f"counts_{prefix}.json"
                self.filesystem.write_text(counts_path, json.dumps(counts, indent=2))

        # Save metadata
        meta = {
            "version": self.VERSION,
            "vocab_size": len(vocab_dict),
            "shard_count": len(shard_files),
            "shards": shard_files,
            "min_frequency": getattr(tokenizer, 'min_frequency', 2),
            "max_vocab_size": getattr(tokenizer, 'max_vocab_size', 10000),
        }
        self.filesystem.write_text(
            tokenizer_dir / "meta.json",
            json.dumps(meta, indent=2)
        )

        # Update cache
        self._loaded_shards = shards

    def load(self, tokenizer_dir: Path) -> 'BPETokenizer':
        """
        Load tokenizer from sharded directory structure.

        Args:
            tokenizer_dir: Directory containing sharded tokenizer

        Returns:
            Reconstructed BPETokenizer
        """
        from cortical.cognitive.text_bridge import BPETokenizer

        # Load metadata
        meta_path = tokenizer_dir / "meta.json"
        meta = json.loads(self.filesystem.read_text(meta_path))

        # Load all shards and reconstruct vocabulary
        vocab: Dict[str, int] = {}
        for shard_filename in meta["shards"]:
            shard_path = tokenizer_dir / shard_filename
            shard_data = json.loads(self.filesystem.read_text(shard_path))
            vocab.update(shard_data)

        # Load merges
        merges_path = tokenizer_dir / "merges.json"
        merges = []
        if self.filesystem.exists(merges_path):
            merges_data = json.loads(self.filesystem.read_text(merges_path))
            for item in merges_data:
                if len(item) == 2:
                    if isinstance(item[1], str) and len(item[1]) > 1:
                        # Format: [[char1, char2], merged_result]
                        merges.append((tuple(item[0]), item[1]))
                    else:
                        # Format: [char1, char2] - simple pair
                        merges.append((item[0], item[1]))

        # Load word counts if present
        word_counts: Counter = Counter()
        for shard_filename in meta["shards"]:
            counts_filename = shard_filename.replace("vocab_", "counts_")
            counts_path = tokenizer_dir / counts_filename
            if self.filesystem.exists(counts_path):
                counts_data = json.loads(self.filesystem.read_text(counts_path))
                word_counts.update(counts_data)

        # Create tokenizer
        tokenizer = BPETokenizer(
            min_frequency=meta.get("min_frequency", 2),
            max_vocab_size=meta.get("max_vocab_size", 10000),
        )
        tokenizer.vocab = set(vocab.keys())
        tokenizer.merges = merges
        tokenizer._word_counts = word_counts

        # Load document frequency data for IDF calculation
        doc_freq_path = tokenizer_dir / "doc_frequency.json"
        if self.filesystem.exists(doc_freq_path):
            tokenizer._doc_frequency = json.loads(
                self.filesystem.read_text(doc_freq_path)
            )
        else:
            tokenizer._doc_frequency = {}

        tokenizer._total_docs = meta.get("total_docs", 0)

        # Cache loaded shards for dirty tracking
        self._loaded_shards = self._group_vocab_by_prefix(vocab)

        return tokenizer

    def detect_format(self, model_dir: Path) -> str:
        """
        Detect whether model uses single-file or sharded tokenizer format.

        Args:
            model_dir: Model directory to check

        Returns:
            "sharded" if tokenizer/ directory exists with meta.json
            "single_file" if tokenizer.json exists
            "none" if neither exists
        """
        tokenizer_dir = model_dir / "tokenizer"
        meta_path = tokenizer_dir / "meta.json"

        if self.filesystem.exists(meta_path):
            return "sharded"

        single_file = model_dir / "tokenizer.json"
        if self.filesystem.exists(single_file):
            return "single_file"

        return "none"

    def migrate_to_sharded(self, model_dir: Path) -> None:
        """
        Migrate from legacy single-file tokenizer.json to sharded format.

        Reads tokenizer.json and creates tokenizer/ directory with shards.
        Does NOT delete the original tokenizer.json (for safety).

        Args:
            model_dir: Model directory containing tokenizer.json
        """
        from cortical.cognitive.text_bridge import BPETokenizer

        single_file = model_dir / "tokenizer.json"
        if not self.filesystem.exists(single_file):
            raise FileNotFoundError(f"No tokenizer.json found at {model_dir}")

        # Load from single file
        data = json.loads(self.filesystem.read_text(single_file))

        # Create tokenizer from legacy format
        tokenizer = BPETokenizer(
            min_frequency=data.get("min_frequency", 2),
            max_vocab_size=data.get("max_vocab_size", 10000),
        )

        # Handle vocab as either list or dict
        vocab_data = data.get("vocab", [])
        if isinstance(vocab_data, list):
            tokenizer.vocab = set(vocab_data)
        elif isinstance(vocab_data, dict):
            tokenizer.vocab = set(vocab_data.keys())
        else:
            tokenizer.vocab = set()

        # Load merges - handle both formats
        merges_data = data.get("merges", [])
        merges = []
        for item in merges_data:
            if len(item) == 2:
                if isinstance(item[0], list) and isinstance(item[1], str) and len(item[1]) > 1:
                    # Format: [[char1, char2], merged_result]
                    merges.append((tuple(item[0]), item[1]))
                else:
                    # Format: [char1, char2] - simple pair
                    merges.append((item[0], item[1]))
        tokenizer.merges = merges

        # Load word counts if present
        tokenizer._word_counts = Counter(data.get("word_counts", {}))

        # Save to sharded format
        tokenizer_dir = model_dir / "tokenizer"
        self.save(tokenizer, tokenizer_dir)

    def get_dirty_prefixes(
        self,
        tokenizer: 'BPETokenizer',
        tokenizer_dir: Path
    ) -> Set[str]:
        """
        Identify which prefixes have changed since last load/save.

        Args:
            tokenizer: Current tokenizer state
            tokenizer_dir: Directory where tokenizer is stored

        Returns:
            Set of prefixes that need to be saved
        """
        # Convert current vocab to dict
        if isinstance(tokenizer.vocab, set):
            current_vocab = {word: i for i, word in enumerate(sorted(tokenizer.vocab))}
        else:
            current_vocab = dict(tokenizer.vocab)

        # Group current vocab by prefix
        current_shards = self._group_vocab_by_prefix(current_vocab)

        # Load metadata to get existing shards
        meta_path = tokenizer_dir / "meta.json"
        if not self.filesystem.exists(meta_path):
            # No existing save, all prefixes are dirty
            return set(current_shards.keys())

        # Compare with cached/loaded shards
        dirty_prefixes: Set[str] = set()

        # Check for new or modified prefixes
        for prefix, current_words in current_shards.items():
            if prefix not in self._loaded_shards:
                dirty_prefixes.add(prefix)
            elif current_words != self._loaded_shards[prefix]:
                dirty_prefixes.add(prefix)

        # Check for removed prefixes (rare but possible)
        for prefix in self._loaded_shards:
            if prefix not in current_shards:
                dirty_prefixes.add(prefix)

        return dirty_prefixes

    def save_incremental(
        self,
        tokenizer: 'BPETokenizer',
        tokenizer_dir: Path
    ) -> None:
        """
        Save only the shards that have changed.

        More efficient than full save when only a few words were added.
        Always updates meta.json and doc_frequency.json even if vocab unchanged.

        Args:
            tokenizer: BPETokenizer to save
            tokenizer_dir: Directory to save to
        """
        # Ensure directory exists
        self.filesystem.mkdir(tokenizer_dir, parents=True, exist_ok=True)

        # Get dirty prefixes for vocab shards
        dirty_prefixes = self.get_dirty_prefixes(tokenizer, tokenizer_dir)

        # Convert vocab to dict
        if isinstance(tokenizer.vocab, set):
            vocab_dict = {word: i for i, word in enumerate(sorted(tokenizer.vocab))}
        else:
            vocab_dict = dict(tokenizer.vocab)

        # Group by prefix
        all_shards = self._group_vocab_by_prefix(vocab_dict)

        # Only write dirty vocab shards (if any)
        for prefix in dirty_prefixes:
            if prefix in all_shards:
                shard_filename = self._get_shard_filename(prefix)
                shard_path = tokenizer_dir / shard_filename
                self.filesystem.write_text(
                    shard_path,
                    json.dumps(all_shards[prefix], indent=2)
                )
            else:
                # Prefix was removed - delete the shard file
                shard_filename = self._get_shard_filename(prefix)
                shard_path = tokenizer_dir / shard_filename
                if self.filesystem.exists(shard_path):
                    self.filesystem.unlink(shard_path)

        # Update metadata
        shard_files = [
            self._get_shard_filename(prefix)
            for prefix in sorted(all_shards.keys())
        ]

        meta = {
            "version": self.VERSION,
            "vocab_size": len(vocab_dict),
            "shard_count": len(shard_files),
            "shards": shard_files,
            "min_frequency": getattr(tokenizer, 'min_frequency', 2),
            "max_vocab_size": getattr(tokenizer, 'max_vocab_size', 10000),
            "total_docs": getattr(tokenizer, '_total_docs', 0),
        }
        self.filesystem.write_text(
            tokenizer_dir / "meta.json",
            json.dumps(meta, indent=2)
        )

        # Save document frequency data for IDF calculation
        doc_freq = getattr(tokenizer, '_doc_frequency', {})
        if doc_freq:
            self.filesystem.write_text(
                tokenizer_dir / "doc_frequency.json",
                json.dumps(doc_freq, indent=2)
            )

        # Update cache
        self._loaded_shards = all_shards

        # Always save merges (they're append-only anyway)
        merges_data = []
        for merge in tokenizer.merges:
            if isinstance(merge, tuple) and len(merge) == 2:
                if isinstance(merge[0], tuple):
                    pair, merged = merge
                    merges_data.append([[pair[0], pair[1]], merged])
                else:
                    merges_data.append([merge[0], merge[1]])
            else:
                merges_data.append(list(merge) if hasattr(merge, '__iter__') else merge)

        self.filesystem.write_text(
            tokenizer_dir / "merges.json",
            json.dumps(merges_data, indent=2)
        )
