"""
Tests for CALI (Content-Addressable Log with Index) ML storage.
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from cortical.ml_storage import (
    BloomFilter,
    HashIndex,
    TimestampIndex,
    ObjectStore,
    PackedLog,
    MLStore,
    migrate_from_jsonl,
    # Validation functions and exceptions
    CALIError,
    CALIValidationError,
    CALISerializationError,
    CALIStorageError,
    validate_record_type,
    validate_record_id,
    validate_content_hash,
    validate_data,
    validate_timestamp,
)


class TestBloomFilter(unittest.TestCase):
    """Test BloomFilter for O(1) existence checks."""

    def test_add_and_check(self):
        """Basic add and check."""
        bf = BloomFilter()
        bf.add("test_key")
        self.assertTrue(bf.might_contain("test_key"))

    def test_false_negative_impossible(self):
        """Added keys must always be found."""
        bf = BloomFilter()
        keys = [f"key_{i}" for i in range(100)]
        for key in keys:
            bf.add(key)

        # All added keys must be found
        for key in keys:
            self.assertTrue(bf.might_contain(key))

    def test_non_added_usually_not_found(self):
        """Non-added keys usually not found (with some false positives)."""
        bf = BloomFilter()
        for i in range(100):
            bf.add(f"added_{i}")

        # Most non-added keys should not be found
        false_positives = sum(
            1 for i in range(1000)
            if bf.might_contain(f"not_added_{i}")
        )
        # With our parameters, false positive rate should be < 1%
        self.assertLess(false_positives / 1000, 0.01)

    def test_save_and_load(self):
        """Test persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bloom.bin"

            bf1 = BloomFilter()
            for i in range(50):
                bf1.add(f"key_{i}")
            bf1.save(path)

            bf2 = BloomFilter.load(path)
            self.assertEqual(bf1.count, bf2.count)
            for i in range(50):
                self.assertTrue(bf2.might_contain(f"key_{i}"))


class TestHashIndex(unittest.TestCase):
    """Test HashIndex for O(1) key lookups."""

    def test_put_and_get(self):
        """Basic put and get."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.idx"
            idx = HashIndex(path)

            idx.put("key1", "hash1", 1000.0, 0)
            entry = idx.get("key1")

            self.assertIsNotNone(entry)
            self.assertEqual(entry.key, "key1")
            self.assertEqual(entry.content_hash, "hash1")
            self.assertEqual(entry.timestamp, 1000.0)

    def test_persistence(self):
        """Index persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.idx"

            idx1 = HashIndex(path)
            idx1.put("key1", "hash1", 1000.0, 0)
            idx1.put("key2", "hash2", 2000.0, 100)

            # New instance should see the data
            idx2 = HashIndex(path)
            self.assertTrue(idx2.contains("key1"))
            self.assertTrue(idx2.contains("key2"))
            self.assertFalse(idx2.contains("key3"))

    def test_compact(self):
        """Compaction removes duplicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.idx"
            idx = HashIndex(path)

            # Add same key multiple times (simulates updates)
            idx.put("key1", "hash1", 1000.0, 0)
            idx.put("key1", "hash2", 2000.0, 100)  # Update
            idx.put("key1", "hash3", 3000.0, 200)  # Update again

            # File should have 3 lines
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 3)

            # Compact
            idx.compact()

            # File should have 1 line (latest version)
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)

            # Should still have the key
            entry = idx.get("key1")
            self.assertEqual(entry.content_hash, "hash3")


class TestTimestampIndex(unittest.TestCase):
    """Test TimestampIndex for range queries."""

    def test_range_query(self):
        """Test range query functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "time.idx"
            idx = TimestampIndex(path)

            # Add entries with different timestamps
            idx.add(1000.0, "key1")
            idx.add(2000.0, "key2")
            idx.add(3000.0, "key3")
            idx.add(4000.0, "key4")
            idx.add(5000.0, "key5")

            # Query middle range
            result = idx.query_range(1500.0, 3500.0)
            self.assertEqual(sorted(result), ["key2", "key3"])

            # Query from start
            result = idx.query_range(None, 2500.0)
            self.assertEqual(sorted(result), ["key1", "key2"])

            # Query to end
            result = idx.query_range(3500.0, None)
            self.assertEqual(sorted(result), ["key4", "key5"])


class TestObjectStore(unittest.TestCase):
    """Test content-addressable ObjectStore."""

    def test_put_and_get(self):
        """Basic put and get."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ObjectStore(Path(tmpdir) / "objects")

            data = {"name": "test", "value": 42}
            content_hash = store.put(data)

            retrieved = store.get(content_hash)
            self.assertEqual(retrieved, data)

    def test_deduplication(self):
        """Same content gets same hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ObjectStore(Path(tmpdir) / "objects")

            data = {"name": "test", "value": 42}
            hash1 = store.put(data)
            hash2 = store.put(data)

            self.assertEqual(hash1, hash2)

    def test_different_content_different_hash(self):
        """Different content gets different hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ObjectStore(Path(tmpdir) / "objects")

            hash1 = store.put({"a": 1})
            hash2 = store.put({"a": 2})

            self.assertNotEqual(hash1, hash2)

    def test_exists(self):
        """Test exists check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ObjectStore(Path(tmpdir) / "objects")

            content_hash = store.put({"test": True})
            self.assertTrue(store.exists(content_hash))
            self.assertFalse(store.exists("nonexistent_hash"))


class TestPackedLog(unittest.TestCase):
    """Test PackedLog for fast sequential iteration."""

    def test_append_and_iterate(self):
        """Test append and iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.packed.jsonl"
            log = PackedLog(path)

            # Append records
            log.append("id1", 1000.0, {"name": "record1"})
            log.append("id2", 2000.0, {"name": "record2"})
            log.append("id3", 3000.0, {"name": "record3"})

            # Iterate
            records = list(log.iterate())
            self.assertEqual(len(records), 3)

            record_id, timestamp, data = records[0]
            self.assertEqual(record_id, "id1")
            self.assertEqual(timestamp, 1000.0)
            self.assertEqual(data, {"name": "record1"})


class TestMLStore(unittest.TestCase):
    """Test complete MLStore functionality."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_path = Path(self.tmpdir) / "store"

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_put_and_get(self):
        """Basic put and get."""
        store = MLStore(self.store_path)

        data = {"hash": "abc123", "message": "test commit"}
        store.put("commit", "abc123", data)

        retrieved = store.get("commit", "abc123")
        self.assertEqual(retrieved, data)
        store.close()

    def test_exists_fast(self):
        """Existence check is O(1)."""
        store = MLStore(self.store_path)

        # Add many records
        for i in range(100):
            store.put("commit", f"hash_{i}", {"index": i})

        # Existence checks should be fast (bloom filter)
        start = time.perf_counter()
        for i in range(100):
            self.assertTrue(store.exists("commit", f"hash_{i}"))
            self.assertFalse(store.exists("commit", f"nonexistent_{i}"))
        elapsed = time.perf_counter() - start

        # Should complete very quickly (< 100ms for 200 checks)
        self.assertLess(elapsed, 0.1)
        store.close()

    def test_iterate(self):
        """Test sequential iteration."""
        store = MLStore(self.store_path)

        # Add records
        for i in range(10):
            store.put("commit", f"hash_{i}", {"index": i})

        # Iterate
        records = list(store.iterate("commit"))
        self.assertEqual(len(records), 10)
        store.close()

    def test_iterate_with_metadata(self):
        """Test iteration with metadata."""
        store = MLStore(self.store_path)

        # Use valid timestamp (2025-01-01)
        test_ts = 1735689600.0
        store.put("commit", "hash1", {"data": "test"}, timestamp=test_ts)

        records = list(store.iterate("commit", include_metadata=True))
        self.assertEqual(len(records), 1)

        record_id, timestamp, data = records[0]
        self.assertEqual(record_id, "hash1")
        self.assertEqual(timestamp, test_ts)
        self.assertEqual(data, {"data": "test"})
        store.close()

    def test_query_range(self):
        """Test timestamp range queries."""
        store = MLStore(self.store_path)

        # Use valid timestamps (2025-01-01, 2025-06-01, 2025-12-01)
        old_ts = 1735689600.0  # 2025-01-01
        mid_ts = 1748736000.0  # 2025-06-01
        new_ts = 1764547200.0  # 2025-12-01

        # Add records with different timestamps
        store.put("commit", "old", {"age": "old"}, timestamp=old_ts)
        store.put("commit", "mid", {"age": "mid"}, timestamp=mid_ts)
        store.put("commit", "new", {"age": "new"}, timestamp=new_ts)

        # Query range (between old and mid)
        results = list(store.query_range("commit", start_ts=old_ts + 1000, end_ts=mid_ts + 1000))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["age"], "mid")
        store.close()

    def test_deduplication(self):
        """Test automatic deduplication."""
        store = MLStore(self.store_path)

        data = {"identical": "data"}
        hash1 = store.put("commit", "id1", data)
        hash2 = store.put("commit", "id2", data)  # Same data, different ID

        # Both should have same content hash
        self.assertEqual(hash1, hash2)

        # Both should be retrievable
        self.assertEqual(store.get("commit", "id1"), data)
        self.assertEqual(store.get("commit", "id2"), data)
        store.close()

    def test_stats(self):
        """Test statistics."""
        store = MLStore(self.store_path)

        store.put("commit", "c1", {"type": "commit"})
        store.put("chat", "ch1", {"type": "chat"})

        stats = store.stats()
        self.assertEqual(stats["record_counts"]["commit"], 1)
        self.assertEqual(stats["record_counts"]["chat"], 1)
        self.assertIn("bloom_filter", stats)
        store.close()

    def test_compact(self):
        """Test compaction."""
        store = MLStore(self.store_path)

        # Add records
        for i in range(10):
            store.put("commit", f"hash_{i}", {"index": i})

        # Compact
        result = store.compact()
        self.assertIn("indices_compacted", result)
        store.close()


class TestSessionBasedStorage(unittest.TestCase):
    """Test git-friendly session-based storage."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_path = Path(self.tmpdir) / "store"

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_session_logs_have_unique_names(self):
        """Each session creates uniquely named log files."""
        store1 = MLStore(self.store_path, session_id='session_a')
        store1.put('commit', 'c1', {'data': 1})
        store1.close()

        store2 = MLStore(self.store_path, session_id='session_b')
        store2.put('commit', 'c2', {'data': 2})
        store2.close()

        # Check that two separate log files exist
        log_files = list((self.store_path / 'logs').glob('*_commit.jsonl'))
        self.assertEqual(len(log_files), 2)

        # Filenames should contain session IDs
        filenames = [f.name for f in log_files]
        self.assertTrue(any('session_a' in name for name in filenames))
        self.assertTrue(any('session_b' in name for name in filenames))

    def test_indices_in_local_dir(self):
        """Indices are stored in local/ (not git-tracked)."""
        store = MLStore(self.store_path)
        store.put('commit', 'c1', {'data': 1})
        store.close()

        # Indices should be in local/
        local_dir = self.store_path / 'local'
        self.assertTrue(local_dir.exists())
        self.assertTrue((local_dir / 'bloom.bin').exists())

        # Indices should NOT be in base dir
        self.assertFalse((self.store_path / 'bloom.bin').exists())
        self.assertFalse((self.store_path / 'indices').exists())

    def test_rebuild_indices_from_logs(self):
        """Can rebuild indices from session logs (after git pull)."""
        # Create store and add data
        store1 = MLStore(self.store_path)
        store1.put('commit', 'c1', {'data': 1})
        store1.put('commit', 'c2', {'data': 2})
        store1.close()

        # Simulate git pull - delete local indices
        local_dir = self.store_path / 'local'
        shutil.rmtree(local_dir)

        # Open store again - should rebuild from logs
        store2 = MLStore(self.store_path)

        # Data should still be accessible
        self.assertTrue(store2.exists('commit', 'c1'))
        self.assertTrue(store2.exists('commit', 'c2'))
        self.assertEqual(store2.get('commit', 'c1'), {'data': 1})
        store2.close()

    def test_no_merge_conflicts(self):
        """Session logs don't conflict across sessions."""
        # Two sessions write to same store
        store1 = MLStore(self.store_path, session_id='alice')
        store2 = MLStore(self.store_path, session_id='bob')

        store1.put('commit', 'alice_c1', {'author': 'alice'})
        store2.put('commit', 'bob_c1', {'author': 'bob'})

        store1.close()
        store2.close()

        # Both should have separate log files (no conflicts)
        log_files = list((self.store_path / 'logs').glob('*_commit.jsonl'))
        self.assertEqual(len(log_files), 2)

        # New session should see all data after index rebuild
        shutil.rmtree(self.store_path / 'local')  # Simulate fresh clone
        store3 = MLStore(self.store_path)

        self.assertEqual(store3.get('commit', 'alice_c1'), {'author': 'alice'})
        self.assertEqual(store3.get('commit', 'bob_c1'), {'author': 'bob'})
        store3.close()


class TestMigration(unittest.TestCase):
    """Test migration from JSON files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_migrate_from_jsonl(self):
        """Test JSONL migration."""
        # Create test JSONL file with valid timestamps (2025-01-01 + offset)
        base_ts = 1735689600.0  # 2025-01-01
        jsonl_path = Path(self.tmpdir) / "commits.jsonl"
        with open(jsonl_path, 'w') as f:
            for i in range(5):
                f.write(json.dumps({
                    "hash": f"commit_{i}",
                    "timestamp": base_ts + i * 86400,  # Each day apart
                    "message": f"commit {i}"
                }) + '\n')

        # Migrate
        store = MLStore(Path(self.tmpdir) / "store")
        result = migrate_from_jsonl(jsonl_path, store, "commit", id_field="hash")

        self.assertEqual(result["migrated"], 5)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(result["errors"], 0)

        # Verify data
        record = store.get("commit", "commit_0")
        self.assertEqual(record["message"], "commit 0")
        store.close()


class TestValidateRecordType(unittest.TestCase):
    """Test validate_record_type function."""

    def test_valid_record_type(self):
        """Valid record types pass validation."""
        self.assertEqual(validate_record_type("commit"), "commit")
        self.assertEqual(validate_record_type("Chat"), "chat")  # Normalized to lowercase
        self.assertEqual(validate_record_type("my_type"), "my_type")
        self.assertEqual(validate_record_type("type-1"), "type-1")
        self.assertEqual(validate_record_type("a"), "a")

    def test_none_raises(self):
        """None record_type raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_record_type(None)
        self.assertIn("cannot be None", str(ctx.exception))

    def test_non_string_raises(self):
        """Non-string record_type raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_record_type(123)
        self.assertIn("must be a string", str(ctx.exception))

        with self.assertRaises(CALIValidationError):
            validate_record_type(['list'])

    def test_empty_raises(self):
        """Empty record_type raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_record_type("")
        self.assertIn("cannot be empty", str(ctx.exception))

        with self.assertRaises(CALIValidationError):
            validate_record_type("   ")  # Whitespace only

    def test_invalid_pattern_raises(self):
        """Invalid patterns raise CALIValidationError."""
        # Must start with letter
        with self.assertRaises(CALIValidationError) as ctx:
            validate_record_type("1invalid")
        self.assertIn("invalid", str(ctx.exception))

        # No special characters
        with self.assertRaises(CALIValidationError):
            validate_record_type("type@name")

        # No spaces
        with self.assertRaises(CALIValidationError):
            validate_record_type("my type")

    def test_whitespace_stripped(self):
        """Whitespace is stripped before validation."""
        self.assertEqual(validate_record_type("  commit  "), "commit")


class TestValidateRecordId(unittest.TestCase):
    """Test validate_record_id function."""

    def test_valid_record_id(self):
        """Valid record IDs pass validation."""
        self.assertEqual(validate_record_id("abc123"), "abc123")
        self.assertEqual(validate_record_id("my_id"), "my_id")
        self.assertEqual(validate_record_id("id-with-dashes"), "id-with-dashes")
        self.assertEqual(validate_record_id("id.with.dots"), "id.with.dots")
        self.assertEqual(validate_record_id("id:with:colons"), "id:with:colons")

    def test_none_raises(self):
        """None record_id raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_record_id(None)
        self.assertIn("cannot be None", str(ctx.exception))

    def test_non_string_raises(self):
        """Non-string record_id raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_record_id(123)
        self.assertIn("must be a string", str(ctx.exception))

    def test_empty_raises(self):
        """Empty record_id raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_record_id("")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_invalid_pattern_raises(self):
        """Invalid patterns raise CALIValidationError."""
        # No special characters except allowed ones
        with self.assertRaises(CALIValidationError) as ctx:
            validate_record_id("id@invalid")
        self.assertIn("invalid", str(ctx.exception))

        # No spaces
        with self.assertRaises(CALIValidationError):
            validate_record_id("my id")

    def test_too_long_raises(self):
        """Record ID over 256 chars raises CALIValidationError."""
        long_id = "a" * 257
        with self.assertRaises(CALIValidationError):
            validate_record_id(long_id)

    def test_max_length_ok(self):
        """Record ID at exactly 256 chars is valid."""
        valid_id = "a" * 256
        self.assertEqual(validate_record_id(valid_id), valid_id)


class TestValidateContentHash(unittest.TestCase):
    """Test validate_content_hash function."""

    def test_valid_hash(self):
        """Valid SHA-256 hashes pass validation."""
        valid_hash = "a" * 64
        self.assertEqual(validate_content_hash(valid_hash), valid_hash)

        # Mixed case normalized to lowercase
        mixed = "A" * 32 + "b" * 32
        self.assertEqual(validate_content_hash(mixed), mixed.lower())

        # Real-looking hash
        real_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        self.assertEqual(validate_content_hash(real_hash), real_hash)

    def test_none_raises(self):
        """None content_hash raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_content_hash(None)
        self.assertIn("cannot be None", str(ctx.exception))

    def test_non_string_raises(self):
        """Non-string content_hash raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_content_hash(123)
        self.assertIn("must be a string", str(ctx.exception))

    def test_wrong_length_raises(self):
        """Content hash not 64 chars raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_content_hash("abc123")  # Too short
        self.assertIn("64-character", str(ctx.exception))

        with self.assertRaises(CALIValidationError):
            validate_content_hash("a" * 65)  # Too long

    def test_non_hex_raises(self):
        """Non-hex characters raise CALIValidationError."""
        with self.assertRaises(CALIValidationError):
            validate_content_hash("g" * 64)  # 'g' is not hex


class TestValidateData(unittest.TestCase):
    """Test validate_data function."""

    def test_valid_data(self):
        """Valid data passes validation."""
        data = {"key": "value", "number": 42}
        self.assertEqual(validate_data(data), data)

    def test_none_raises(self):
        """None data raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_data(None)
        self.assertIn("cannot be None", str(ctx.exception))

    def test_non_dict_raises(self):
        """Non-dict data raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_data("not a dict")
        self.assertIn("must be a dictionary", str(ctx.exception))

        with self.assertRaises(CALIValidationError):
            validate_data([1, 2, 3])

    def test_empty_dict_raises(self):
        """Empty dict raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_data({})
        self.assertIn("cannot be an empty", str(ctx.exception))

    def test_non_serializable_raises(self):
        """Non-JSON-serializable data raises CALISerializationError."""
        with self.assertRaises(CALISerializationError) as ctx:
            validate_data({"func": lambda x: x})
        self.assertIn("not JSON serializable", str(ctx.exception))

        # Object that can't be serialized
        class Custom:
            pass
        with self.assertRaises(CALISerializationError):
            validate_data({"obj": Custom()})

    def test_nested_data_valid(self):
        """Nested dicts and lists are valid."""
        data = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, {"inner": "value"}],
            "null": None,
            "bool": True
        }
        self.assertEqual(validate_data(data), data)


class TestValidateTimestamp(unittest.TestCase):
    """Test validate_timestamp function."""

    def test_none_returns_current_time(self):
        """None timestamp returns current time."""
        before = time.time()
        result = validate_timestamp(None)
        after = time.time()
        self.assertGreaterEqual(result, before)
        self.assertLessEqual(result, after)

    def test_valid_float(self):
        """Valid float timestamps pass validation."""
        valid_ts = 1735689600.0  # 2025-01-01
        self.assertEqual(validate_timestamp(valid_ts), valid_ts)

    def test_valid_int(self):
        """Valid int timestamps are converted to float."""
        valid_ts = 1735689600  # 2025-01-01
        result = validate_timestamp(valid_ts)
        self.assertIsInstance(result, float)
        self.assertEqual(result, float(valid_ts))

    def test_too_old_raises(self):
        """Timestamp before 2020 raises CALIValidationError."""
        old_ts = 1000.0  # 1970
        with self.assertRaises(CALIValidationError) as ctx:
            validate_timestamp(old_ts)
        self.assertIn("out of valid range", str(ctx.exception))

    def test_too_far_future_raises(self):
        """Timestamp after 2100 raises CALIValidationError."""
        future_ts = 5000000000.0  # ~2128
        with self.assertRaises(CALIValidationError) as ctx:
            validate_timestamp(future_ts)
        self.assertIn("out of valid range", str(ctx.exception))

    def test_valid_iso_string(self):
        """Valid ISO format string is converted to timestamp."""
        iso = "2025-01-01T00:00:00+00:00"
        result = validate_timestamp(iso)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 1735689600.0, delta=1)

    def test_iso_with_z_suffix(self):
        """ISO string with Z suffix is handled."""
        iso = "2025-01-01T00:00:00Z"
        result = validate_timestamp(iso)
        self.assertIsInstance(result, float)

    def test_invalid_iso_raises(self):
        """Invalid ISO format raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_timestamp("not-a-date")
        self.assertIn("not a valid ISO format", str(ctx.exception))

    def test_invalid_type_raises(self):
        """Invalid type raises CALIValidationError."""
        with self.assertRaises(CALIValidationError) as ctx:
            validate_timestamp([2025, 1, 1])
        self.assertIn("must be a float or ISO string", str(ctx.exception))


class TestExceptionHierarchy(unittest.TestCase):
    """Test CALI exception hierarchy."""

    def test_exception_inheritance(self):
        """All CALI exceptions inherit from CALIError."""
        self.assertTrue(issubclass(CALIValidationError, CALIError))
        self.assertTrue(issubclass(CALISerializationError, CALIError))
        self.assertTrue(issubclass(CALIStorageError, CALIError))

    def test_exceptions_are_catchable(self):
        """CALI exceptions can be caught by base class."""
        try:
            raise CALIValidationError("test")
        except CALIError as e:
            self.assertIn("test", str(e))

        try:
            raise CALISerializationError("serialize error")
        except CALIError as e:
            self.assertIn("serialize", str(e))


class TestMLStoreValidation(unittest.TestCase):
    """Test MLStore methods with validation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_path = Path(self.tmpdir) / "store"

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_put_validates_record_type(self):
        """MLStore.put validates record_type."""
        store = MLStore(self.store_path)
        with self.assertRaises(CALIValidationError):
            store.put(None, "id", {"data": 1})
        with self.assertRaises(CALIValidationError):
            store.put("", "id", {"data": 1})
        store.close()

    def test_put_validates_record_id(self):
        """MLStore.put validates record_id."""
        store = MLStore(self.store_path)
        with self.assertRaises(CALIValidationError):
            store.put("commit", None, {"data": 1})
        with self.assertRaises(CALIValidationError):
            store.put("commit", "", {"data": 1})
        store.close()

    def test_put_validates_data(self):
        """MLStore.put validates data."""
        store = MLStore(self.store_path)
        with self.assertRaises(CALIValidationError):
            store.put("commit", "id", None)
        with self.assertRaises(CALIValidationError):
            store.put("commit", "id", {})
        with self.assertRaises(CALIValidationError):
            store.put("commit", "id", "not a dict")
        store.close()

    def test_put_validates_timestamp(self):
        """MLStore.put validates timestamp."""
        store = MLStore(self.store_path)
        with self.assertRaises(CALIValidationError):
            store.put("commit", "id", {"data": 1}, timestamp=1000.0)  # Too old
        store.close()

    def test_get_validates_inputs(self):
        """MLStore.get validates inputs."""
        store = MLStore(self.store_path)
        with self.assertRaises(CALIValidationError):
            store.get(None, "id")
        with self.assertRaises(CALIValidationError):
            store.get("commit", None)
        store.close()

    def test_exists_validates_inputs(self):
        """MLStore.exists validates inputs."""
        store = MLStore(self.store_path)
        with self.assertRaises(CALIValidationError):
            store.exists(None, "id")
        with self.assertRaises(CALIValidationError):
            store.exists("commit", None)
        store.close()

    def test_get_by_hash_validates_input(self):
        """MLStore.get_by_hash validates content_hash."""
        store = MLStore(self.store_path)
        with self.assertRaises(CALIValidationError):
            store.get_by_hash(None)
        with self.assertRaises(CALIValidationError):
            store.get_by_hash("short")
        store.close()


if __name__ == '__main__':
    unittest.main()
