"""
=============================================================================
SPECIFICATION: Sharded Tokenizer Storage
=============================================================================

OVERVIEW
--------
This specification defines how the BPE tokenizer vocabulary and merge rules
are persisted to disk using a sharded directory structure instead of a
single monolithic JSON file.

PROBLEM STATEMENT
-----------------
The current tokenizer.json approach has significant issues at scale:

1. SIZE: A 10,000-word vocabulary produces a 6.7MB JSON file
2. PERFORMANCE: Parsing 6.7MB of JSON on every load is slow
3. GIT CONFLICTS: Any vocabulary change creates merge conflicts
4. INCREMENTAL SAVES: Cannot save just the new/changed entries
5. MEMORY: Must load entire vocabulary even if only checking one word

SOLUTION: SHARDED DIRECTORY STRUCTURE
-------------------------------------
Replace tokenizer.json with a tokenizer/ directory:

    models/cognitive_agent/
    └── tokenizer/
        ├── meta.json           # Metadata: shard count, vocab size, version
        ├── vocab_0000.json     # Entries 0-999 (alphabetically sorted)
        ├── vocab_0001.json     # Entries 1000-1999
        ├── vocab_0002.json     # Entries 2000-2999
        ├── ...
        └── merges.json         # BPE merge rules (separate file)

KEY DESIGN DECISIONS
--------------------
1. ALPHABETIC SHARDING: Words are assigned to shards based on their
   first character(s), not insertion order. This means:
   - "apple" always goes to the same shard
   - Adding "banana" doesn't affect "apple"'s shard
   - Merge conflicts are nearly impossible

2. SHARD BY PREFIX: Use first 2 characters as shard key:
   - aa-az → vocab_00.json
   - ba-bz → vocab_01.json
   - ...
   - This gives 26*26 = 676 potential shards (most will be sparse)

3. SPARSE SHARDS: Only create shard files that have content
   - Empty shards don't create files
   - meta.json tracks which shards exist

4. MERGES SEPARATE: BPE merge rules in their own file
   - Merges are append-only (new merges added at end)
   - Conflicts are rare (both sides usually add different merges)

DEPENDENCIES
------------
- FileSystem: I/O abstraction (RealFileSystem, InMemoryFileSystem)
- BPETokenizer: The tokenizer being persisted

USAGE EXAMPLE
-------------
    storage = ShardedTokenizerStorage(filesystem)

    # Save tokenizer to sharded directory
    storage.save(tokenizer, model_dir / "tokenizer")

    # Load tokenizer from sharded directory
    tokenizer = storage.load(model_dir / "tokenizer")

    # Incremental save (only dirty shards)
    storage.save_incremental(tokenizer, model_dir / "tokenizer", dirty_prefixes=["ne", "ma"])
"""

import pytest
from pathlib import Path


# =============================================================================
# EPIC: Sharded Vocabulary Persistence
# =============================================================================

class TestShardedVocabularyStorage:
    """
    EPIC: Sharded Vocabulary Persistence
    =====================================

    PERSONA: Developer training cognitive agents on large corpora
    GOAL: Save vocabulary to multiple files organized by word prefix
    VALUE: Eliminate merge conflicts and enable incremental saves

    ACCEPTANCE CRITERIA:
    - Vocabulary is split into multiple shard files by prefix
    - Each shard contains words with the same 2-character prefix
    - meta.json tracks shard metadata
    - Loading reconstructs the complete vocabulary
    """

    def test_scenario_save_vocabulary_creates_sharded_directory(self, tmp_path):
        """
        Scenario: Saving tokenizer creates sharded directory structure

        Given I have a tokenizer with vocabulary words
        When I save the tokenizer using sharded storage
        Then a tokenizer/ directory is created
        And meta.json contains shard metadata
        And vocabulary is split across shard files by prefix
        Because sharded storage eliminates merge conflicts.
        """
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I have a tokenizer with vocabulary words
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        tokenizer = BPETokenizer()
        # Add words with different prefixes
        tokenizer.vocab = {
            "apple": 0,
            "application": 1,
            "banana": 2,
            "cognitive": 3,
            "compute": 4,
            "neural": 5,
            "network": 6,
        }

        storage = ShardedTokenizerStorage(fs)
        tokenizer_dir = Path("/test/tokenizer")

        # When I save the tokenizer using sharded storage
        storage.save(tokenizer, tokenizer_dir)

        # Then a tokenizer/ directory is created
        assert fs.is_dir(tokenizer_dir)

        # And meta.json contains shard metadata
        assert fs.exists(tokenizer_dir / "meta.json")
        import json
        meta = json.loads(fs.read_text(tokenizer_dir / "meta.json"))
        assert "vocab_size" in meta
        assert meta["vocab_size"] == 7
        assert "shards" in meta

        # And vocabulary is split across shard files by prefix
        # Words starting with "ap" go to one shard
        # Words starting with "ba" go to another shard
        # etc.
        assert len(meta["shards"]) > 1  # Multiple shards created

    def test_scenario_load_vocabulary_from_shards(self, tmp_path):
        """
        Scenario: Loading tokenizer reconstructs vocabulary from shards

        Given I have a saved sharded tokenizer
        When I load the tokenizer from the sharded directory
        Then the complete vocabulary is reconstructed
        And all words are accessible
        And word indices are preserved
        Because loading must be lossless.
        """
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I have a saved sharded tokenizer
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        original = BPETokenizer()
        original.vocab = {
            "apple": 0,
            "banana": 1,
            "cherry": 2,
            "neural": 3,
            "network": 4,
        }
        original.merges = [("n", "e"), ("ne", "u"), ("neu", "r")]

        storage = ShardedTokenizerStorage(fs)
        tokenizer_dir = Path("/test/tokenizer")
        storage.save(original, tokenizer_dir)

        # When I load the tokenizer from the sharded directory
        loaded = storage.load(tokenizer_dir)

        # Then the complete vocabulary is reconstructed
        assert len(loaded.vocab) == len(original.vocab)

        # And all words are accessible
        for word in original.vocab:
            assert word in loaded.vocab

        # And word indices are preserved
        for word, idx in original.vocab.items():
            assert loaded.vocab[word] == idx

    def test_scenario_prefix_determines_shard_assignment(self, tmp_path):
        """
        Scenario: Words are assigned to shards by their prefix

        Given I have words with various prefixes
        When I save and examine the shard files
        Then words with the same prefix are in the same shard
        And words with different prefixes are in different shards
        Because prefix-based sharding prevents merge conflicts.
        """
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem
        import json

        # Given I have words with various prefixes
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        tokenizer = BPETokenizer()
        tokenizer.vocab = {
            # "ne" prefix group
            "neural": 0,
            "network": 1,
            "neuron": 2,
            # "ma" prefix group
            "machine": 3,
            "mapping": 4,
            # "co" prefix group
            "cognitive": 5,
            "compute": 6,
        }

        storage = ShardedTokenizerStorage(fs)
        tokenizer_dir = Path("/test/tokenizer")

        # When I save and examine the shard files
        storage.save(tokenizer, tokenizer_dir)

        # Read meta to find shard files
        meta = json.loads(fs.read_text(tokenizer_dir / "meta.json"))

        # Then words with the same prefix are in the same shard
        # Find the shard containing "neural"
        ne_shard = None
        for shard_name in meta["shards"]:
            shard_path = tokenizer_dir / shard_name
            shard_data = json.loads(fs.read_text(shard_path))
            if "neural" in shard_data:
                ne_shard = shard_data
                break

        assert ne_shard is not None
        assert "neural" in ne_shard
        assert "network" in ne_shard
        assert "neuron" in ne_shard

        # And words with different prefixes are in different shards
        assert "machine" not in ne_shard
        assert "cognitive" not in ne_shard


# =============================================================================
# EPIC: Merge-Conflict-Free Updates
# =============================================================================

class TestMergeConflictFreeUpdates:
    """
    EPIC: Merge-Conflict-Free Updates
    ==================================

    PERSONA: Team of developers training agents concurrently
    GOAL: Multiple developers can add vocabulary without conflicts
    VALUE: No git merge conflicts when training on different documents

    KEY INSIGHT:
    If Developer A adds "quantum" and Developer B adds "neural",
    these go to different shard files (qu_shard vs ne_shard).
    Git can merge both changes automatically.
    """

    def test_scenario_adding_new_word_only_modifies_its_shard(self, tmp_path):
        """
        Scenario: Adding a word only touches one shard file

        Given I have a saved sharded tokenizer
        When I add a new word and save
        Then only the shard for that word's prefix is modified
        And other shards remain unchanged
        Because isolated changes prevent merge conflicts.
        """
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem
        import json

        # Given I have a saved sharded tokenizer
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        tokenizer = BPETokenizer()
        tokenizer.vocab = {
            "apple": 0,
            "banana": 1,
            "cognitive": 2,
        }

        storage = ShardedTokenizerStorage(fs)
        tokenizer_dir = Path("/test/tokenizer")
        storage.save(tokenizer, tokenizer_dir)

        # Record original shard contents
        meta = json.loads(fs.read_text(tokenizer_dir / "meta.json"))
        original_shards = {}
        for shard_name in meta["shards"]:
            original_shards[shard_name] = fs.read_text(tokenizer_dir / shard_name)

        # When I add a new word and save
        tokenizer.vocab["neural"] = 3  # New word with "ne" prefix
        storage.save(tokenizer, tokenizer_dir)

        # Then only the shard for that word's prefix is modified
        new_meta = json.loads(fs.read_text(tokenizer_dir / "meta.json"))

        # Find which shards changed
        changed_shards = []
        unchanged_shards = []
        for shard_name in new_meta["shards"]:
            if shard_name in original_shards:
                current = fs.read_text(tokenizer_dir / shard_name)
                if current != original_shards[shard_name]:
                    changed_shards.append(shard_name)
                else:
                    unchanged_shards.append(shard_name)

        # The shard containing "neural" should be new or changed
        # Other existing shards should be unchanged
        assert len(unchanged_shards) >= 2  # ap, ba, co shards unchanged

    def test_scenario_concurrent_additions_to_different_prefixes(self, tmp_path):
        """
        Scenario: Concurrent additions to different prefixes don't conflict

        Given two developers start with the same tokenizer state
        When Developer A adds "quantum" (qu prefix)
        And Developer B adds "neural" (ne prefix)
        Then both changes can be merged without conflict
        Because they modify different shard files.
        """
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem
        import json

        # Given two developers start with the same tokenizer state
        fs_base = InMemoryFileSystem(Path("/base"))
        fs_base.mkdir(Path("/base"), parents=True, exist_ok=True)

        base_tokenizer = BPETokenizer()
        base_tokenizer.vocab = {"apple": 0, "banana": 1}

        storage = ShardedTokenizerStorage(fs_base)
        storage.save(base_tokenizer, Path("/base/tokenizer"))

        # Simulate Developer A's branch
        fs_a = InMemoryFileSystem(Path("/dev_a"))
        fs_a.mkdir(Path("/dev_a"), parents=True, exist_ok=True)

        tokenizer_a = BPETokenizer()
        tokenizer_a.vocab = {"apple": 0, "banana": 1, "quantum": 2}

        storage_a = ShardedTokenizerStorage(fs_a)
        storage_a.save(tokenizer_a, Path("/dev_a/tokenizer"))

        # Simulate Developer B's branch
        fs_b = InMemoryFileSystem(Path("/dev_b"))
        fs_b.mkdir(Path("/dev_b"), parents=True, exist_ok=True)

        tokenizer_b = BPETokenizer()
        tokenizer_b.vocab = {"apple": 0, "banana": 1, "neural": 2}

        storage_b = ShardedTokenizerStorage(fs_b)
        storage_b.save(tokenizer_b, Path("/dev_b/tokenizer"))

        # Then both changes can be merged without conflict
        # Get shard files from both
        meta_a = json.loads(fs_a.read_text(Path("/dev_a/tokenizer/meta.json")))
        meta_b = json.loads(fs_b.read_text(Path("/dev_b/tokenizer/meta.json")))

        # Find which shards each developer modified
        shards_a = set(meta_a["shards"])
        shards_b = set(meta_b["shards"])

        # The new shards should be different (qu vs ne prefix)
        # Intersection should only be the base shards (ap, ba)
        new_in_a = shards_a - {"vocab_ap.json", "vocab_ba.json"}
        new_in_b = shards_b - {"vocab_ap.json", "vocab_ba.json"}

        # These should not overlap
        assert new_in_a.isdisjoint(new_in_b), \
            f"Shards overlap: {new_in_a & new_in_b}"


# =============================================================================
# EPIC: BPE Merge Rules Persistence
# =============================================================================

class TestMergeRulesPersistence:
    """
    EPIC: BPE Merge Rules Persistence
    ==================================

    PERSONA: Developer using BPE tokenization
    GOAL: Merge rules are saved separately from vocabulary
    VALUE: Merge rules can be updated independently

    DESIGN:
    Merge rules are append-only and stored in merges.json.
    New training sessions add new merges at the end.
    """

    def test_scenario_merges_saved_separately(self, tmp_path):
        """
        Scenario: Merge rules are saved in a separate file

        Given I have a tokenizer with vocabulary and merge rules
        When I save using sharded storage
        Then merges.json is created separately from vocab shards
        And merges.json contains the merge rules in order
        Because merge rules have different update patterns than vocabulary.
        """
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem
        import json

        # Given I have a tokenizer with vocabulary and merge rules
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        tokenizer = BPETokenizer()
        tokenizer.vocab = {"neural": 0, "network": 1}
        tokenizer.merges = [
            ("n", "e"),
            ("ne", "u"),
            ("neu", "r"),
            ("neur", "a"),
            ("neura", "l"),
        ]

        storage = ShardedTokenizerStorage(fs)
        tokenizer_dir = Path("/test/tokenizer")

        # When I save using sharded storage
        storage.save(tokenizer, tokenizer_dir)

        # Then merges.json is created separately from vocab shards
        assert fs.exists(tokenizer_dir / "merges.json")

        # And merges.json contains the merge rules in order
        merges_data = json.loads(fs.read_text(tokenizer_dir / "merges.json"))
        assert len(merges_data) == 5
        assert merges_data[0] == ["n", "e"]
        assert merges_data[4] == ["neura", "l"]

    def test_scenario_merges_loaded_correctly(self, tmp_path):
        """
        Scenario: Loading reconstructs merge rules in correct order

        Given I have saved merge rules
        When I load the tokenizer
        Then merge rules are reconstructed in the original order
        Because BPE merge order affects tokenization.
        """
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I have saved merge rules
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        original = BPETokenizer()
        original.vocab = {"test": 0}
        original.merges = [("a", "b"), ("ab", "c"), ("abc", "d")]

        storage = ShardedTokenizerStorage(fs)
        tokenizer_dir = Path("/test/tokenizer")
        storage.save(original, tokenizer_dir)

        # When I load the tokenizer
        loaded = storage.load(tokenizer_dir)

        # Then merge rules are reconstructed in the original order
        assert loaded.merges == original.merges


# =============================================================================
# EPIC: Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """
    EPIC: Backward Compatibility
    =============================

    PERSONA: Developer with existing tokenizer.json files
    GOAL: Migrate smoothly from single-file to sharded format
    VALUE: No breaking changes for existing trained models

    MIGRATION PATH:
    1. Detect single-file format (tokenizer.json exists, tokenizer/ doesn't)
    2. Load from single file
    3. Save to sharded format
    4. Optionally delete old tokenizer.json
    """

    def test_scenario_detect_single_file_format(self, tmp_path):
        """
        Scenario: Detecting legacy single-file tokenizer format

        Given I have a model with legacy tokenizer.json
        When I check the storage format
        Then it is detected as single-file format
        Because we need to know when migration is needed.
        """
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem
        import json

        # Given I have a model with legacy tokenizer.json
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test/model"), parents=True, exist_ok=True)

        legacy_data = {
            "vocab": {"apple": 0, "banana": 1},
            "merges": [["a", "p"], ["ap", "p"]],
        }
        fs.write_text(Path("/test/model/tokenizer.json"), json.dumps(legacy_data))

        storage = ShardedTokenizerStorage(fs)

        # When I check the storage format
        format_type = storage.detect_format(Path("/test/model"))

        # Then it is detected as single-file format
        assert format_type == "single_file"

    def test_scenario_detect_sharded_format(self, tmp_path):
        """
        Scenario: Detecting sharded tokenizer format

        Given I have a model with sharded tokenizer directory
        When I check the storage format
        Then it is detected as sharded format
        Because we need to use the correct loading strategy.
        """
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem
        import json

        # Given I have a model with sharded tokenizer directory
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test/model/tokenizer"), parents=True, exist_ok=True)

        meta = {"vocab_size": 2, "shards": ["vocab_ap.json"]}
        fs.write_text(Path("/test/model/tokenizer/meta.json"), json.dumps(meta))

        storage = ShardedTokenizerStorage(fs)

        # When I check the storage format
        format_type = storage.detect_format(Path("/test/model"))

        # Then it is detected as sharded format
        assert format_type == "sharded"

    def test_scenario_migrate_single_file_to_sharded(self, tmp_path):
        """
        Scenario: Migrating from single-file to sharded format

        Given I have a legacy tokenizer.json
        When I run migration
        Then a sharded tokenizer/ directory is created
        And the vocabulary is correctly sharded
        And the original tokenizer.json can be removed
        Because migration should be seamless.
        """
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem
        import json

        # Given I have a legacy tokenizer.json
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test/model"), parents=True, exist_ok=True)

        legacy_data = {
            "vocab": {"apple": 0, "banana": 1, "neural": 2},
            "merges": [["n", "e"], ["ne", "u"]],
        }
        fs.write_text(Path("/test/model/tokenizer.json"), json.dumps(legacy_data))

        storage = ShardedTokenizerStorage(fs)

        # When I run migration
        storage.migrate_to_sharded(Path("/test/model"))

        # Then a sharded tokenizer/ directory is created
        assert fs.is_dir(Path("/test/model/tokenizer"))
        assert fs.exists(Path("/test/model/tokenizer/meta.json"))

        # And the vocabulary is correctly sharded
        loaded = storage.load(Path("/test/model/tokenizer"))
        assert len(loaded.vocab) == 3
        assert "apple" in loaded.vocab
        assert "neural" in loaded.vocab

        # And the original tokenizer.json can be removed
        # (migration doesn't delete it automatically for safety)
        assert fs.exists(Path("/test/model/tokenizer.json"))


# =============================================================================
# EPIC: Incremental Saves
# =============================================================================

class TestIncrementalSaves:
    """
    EPIC: Incremental Saves
    ========================

    PERSONA: Developer doing iterative training
    GOAL: Only save shards that have changed
    VALUE: Faster saves, less I/O, smaller git diffs

    IMPLEMENTATION:
    Track which prefixes have new/modified words since last save.
    Only write those shard files.
    """

    def test_scenario_track_dirty_shards(self, tmp_path):
        """
        Scenario: Tracking which shards need saving

        Given I have a saved tokenizer
        When I add new words
        Then I can identify which shards are dirty
        And only dirty shards need to be written
        Because incremental saves are faster.
        """
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I have a saved tokenizer
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        tokenizer = BPETokenizer()
        tokenizer.vocab = {"apple": 0, "banana": 1}

        storage = ShardedTokenizerStorage(fs)
        tokenizer_dir = Path("/test/tokenizer")
        storage.save(tokenizer, tokenizer_dir)

        # When I add new words
        tokenizer.vocab["neural"] = 2
        tokenizer.vocab["network"] = 3

        # Then I can identify which shards are dirty
        dirty_prefixes = storage.get_dirty_prefixes(tokenizer, tokenizer_dir)

        # Only the "ne" prefix shard is dirty
        assert "ne" in dirty_prefixes
        assert "ap" not in dirty_prefixes
        assert "ba" not in dirty_prefixes

    def test_scenario_incremental_save_only_writes_dirty_shards(self, tmp_path):
        """
        Scenario: Incremental save minimizes I/O

        Given I have a large saved tokenizer
        And I add a few new words
        When I do an incremental save
        Then only the affected shard files are written
        And unchanged shards are not touched
        Because minimizing I/O improves performance.
        """
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I have a large saved tokenizer
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        tokenizer = BPETokenizer()
        # Create vocabulary with many different prefixes
        words = ["apple", "banana", "cherry", "delta", "echo",
                 "foxtrot", "golf", "hotel", "india", "juliet"]
        tokenizer.vocab = {word: i for i, word in enumerate(words)}

        storage = ShardedTokenizerStorage(fs)
        tokenizer_dir = Path("/test/tokenizer")
        storage.save(tokenizer, tokenizer_dir)

        # Reset tracking
        fs.reset_tracking()

        # And I add a few new words
        tokenizer.vocab["neural"] = 10
        tokenizer.vocab["network"] = 11

        # When I do an incremental save
        storage.save_incremental(tokenizer, tokenizer_dir)

        # Then only the affected shard files are written
        written_files = fs.files_written
        written_names = [Path(f).name for f in written_files]

        # Should write: meta.json (updated count) and the "ne" shard
        assert "meta.json" in written_names
        # Only one vocab shard should be written (the "ne" prefix shard)
        vocab_shards_written = [n for n in written_names if n.startswith("vocab_")]
        assert len(vocab_shards_written) == 1


# =============================================================================
# EPIC: Special Character Handling
# =============================================================================

class TestSpecialCharacterHandling:
    """
    EPIC: Special Character Handling
    =================================

    PERSONA: Developer with diverse text corpora
    GOAL: Handle words with special prefixes correctly
    VALUE: Robust handling of numbers, punctuation, unicode

    EDGE CASES:
    - Numbers: "42", "3.14"
    - Punctuation: "don't", "e-mail"
    - Unicode: "café", "naïve", "日本語"
    - Single characters: "a", "I"
    """

    def test_scenario_numeric_prefixes(self, tmp_path):
        """
        Scenario: Words starting with numbers are handled correctly

        Given I have words starting with numbers
        When I save and load
        Then numeric-prefix words are correctly stored and retrieved
        Because corpora may contain version numbers, dates, etc.
        """
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I have words starting with numbers
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        tokenizer = BPETokenizer()
        tokenizer.vocab = {
            "42": 0,
            "3.14": 1,
            "2024": 2,
            "1st": 3,
        }

        storage = ShardedTokenizerStorage(fs)
        tokenizer_dir = Path("/test/tokenizer")

        # When I save and load
        storage.save(tokenizer, tokenizer_dir)
        loaded = storage.load(tokenizer_dir)

        # Then numeric-prefix words are correctly stored and retrieved
        assert "42" in loaded.vocab
        assert "3.14" in loaded.vocab
        assert loaded.vocab["2024"] == 2

    def test_scenario_single_character_words(self, tmp_path):
        """
        Scenario: Single character words are handled correctly

        Given I have single-character vocabulary entries
        When I save and load
        Then single characters are correctly stored and retrieved
        Because BPE starts with single characters.
        """
        from cortical.cognitive.text_bridge import BPETokenizer
        from cortical.cognitive.tokenizer_storage import ShardedTokenizerStorage
        from cortical.common.filesystem import InMemoryFileSystem

        # Given I have single-character vocabulary entries
        fs = InMemoryFileSystem(Path("/test"))
        fs.mkdir(Path("/test"), parents=True, exist_ok=True)

        tokenizer = BPETokenizer()
        tokenizer.vocab = {
            "a": 0, "b": 1, "c": 2,
            "I": 3,  # Single letter word
            "1": 4, "2": 5,
        }

        storage = ShardedTokenizerStorage(fs)
        tokenizer_dir = Path("/test/tokenizer")

        # When I save and load
        storage.save(tokenizer, tokenizer_dir)
        loaded = storage.load(tokenizer_dir)

        # Then single characters are correctly stored and retrieved
        assert len(loaded.vocab) == 6
        assert "a" in loaded.vocab
        assert "I" in loaded.vocab
        assert loaded.vocab["1"] == 4
