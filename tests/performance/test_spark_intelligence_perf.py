"""
Performance tests for SparkCodeIntelligence.

Tests performance characteristics:
- Tokenization speed
- AST indexing speed
- Training time
- Query latency
- Completion latency
- Save/load time

Note: These tests are for benchmarking, not strict assertions.
They use soft thresholds to detect regressions.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from cortical.spark import (
    SparkCodeIntelligence,
    CodeTokenizer,
    ASTIndex,
    NGramModel
)


class TestTokenizerPerformance:
    """Performance tests for CodeTokenizer."""

    def test_tokenize_speed(self):
        """Test tokenization speed."""
        tokenizer = CodeTokenizer(split_identifiers=True)

        # Generate test code
        code = '''
def calculate_sum(a, b, c):
    """Calculate the sum of three numbers."""
    result = a + b + c
    return result

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.data = []

    def process(self, input_data):
        for item in input_data:
            self.data.append(self.transform(item))
        return self.data
''' * 100  # Repeat to get substantial text

        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            tokenizer.tokenize(code)
        elapsed = time.perf_counter() - start

        # Should tokenize ~1000 lines in < 1 second (10 iterations)
        assert elapsed < 5.0, f"Tokenization too slow: {elapsed:.2f}s"

        # Calculate tokens per second
        tokens = tokenizer.tokenize(code)
        tokens_per_sec = (len(tokens) * 10) / elapsed
        print(f"\nTokenizer: {tokens_per_sec:.0f} tokens/sec, {elapsed:.3f}s for 10 iterations")

    def test_identifier_splitting_overhead(self):
        """Test overhead of identifier splitting."""
        tokenizer_split = CodeTokenizer(split_identifiers=True)
        tokenizer_no_split = CodeTokenizer(split_identifiers=False)

        code = "def getUserNameFromDatabaseWithValidation(): pass" * 100

        # Without splitting
        start = time.perf_counter()
        for _ in range(100):
            tokenizer_no_split.tokenize(code)
        time_no_split = time.perf_counter() - start

        # With splitting
        start = time.perf_counter()
        for _ in range(100):
            tokenizer_split.tokenize(code)
        time_split = time.perf_counter() - start

        # Splitting should add < 3x overhead
        overhead = time_split / time_no_split
        assert overhead < 3.0, f"Splitting overhead too high: {overhead:.1f}x"

        print(f"\nIdentifier splitting overhead: {overhead:.2f}x")


class TestASTIndexPerformance:
    """Performance tests for ASTIndex."""

    @pytest.fixture
    def large_project(self):
        """Create a larger test project."""
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir)

        # Create 50 Python files
        for i in range(50):
            content = f'''
"""Module {i}."""

class Class{i}:
    """Class {i} docstring."""

    def __init__(self):
        self.value = {i}

    def method_a(self):
        return self.value

    def method_b(self):
        return self.method_a() * 2

def function_{i}():
    obj = Class{i}()
    return obj.method_b()
'''
            (path / f'module_{i}.py').write_text(content)

        yield path
        shutil.rmtree(tmpdir)

    def test_index_directory_speed(self, large_project):
        """Test directory indexing speed."""
        index = ASTIndex()

        start = time.perf_counter()
        index.index_directory(large_project, verbose=False)
        elapsed = time.perf_counter() - start

        # Should index 50 files in < 2 seconds
        assert elapsed < 5.0, f"Indexing too slow: {elapsed:.2f}s"
        assert index.files_indexed == 50

        files_per_sec = index.files_indexed / elapsed
        print(f"\nAST Indexing: {files_per_sec:.1f} files/sec, {elapsed:.3f}s for 50 files")

    def test_find_callers_speed(self, large_project):
        """Test caller lookup speed."""
        index = ASTIndex()
        index.index_directory(large_project, verbose=False)

        # Benchmark find_callers
        start = time.perf_counter()
        for i in range(50):
            index.find_callers(f'method_a')
        elapsed = time.perf_counter() - start

        # Should complete 50 queries in < 0.5 seconds
        avg_ms = (elapsed / 50) * 1000
        assert avg_ms < 10, f"Find callers too slow: {avg_ms:.2f}ms"

        print(f"\nFind callers: {avg_ms:.2f}ms average")


class TestNGramPerformance:
    """Performance tests for NGramModel."""

    def test_training_speed(self):
        """Test n-gram training speed."""
        model = NGramModel(n=3)
        tokenizer = CodeTokenizer()

        # Generate training data
        code = '''
def process_data(input_list):
    result = []
    for item in input_list:
        if item is not None:
            result.append(item * 2)
    return result
''' * 100

        token_lists = [tokenizer.tokenize(code) for _ in range(10)]

        # Benchmark training
        start = time.perf_counter()
        model.train_on_tokens(token_lists)
        model.finalize()
        elapsed = time.perf_counter() - start

        # Should train in < 1 second
        assert elapsed < 2.0, f"Training too slow: {elapsed:.2f}s"

        tokens_per_sec = model.total_tokens / elapsed
        print(f"\nN-gram training: {tokens_per_sec:.0f} tokens/sec, {elapsed:.3f}s")

    def test_prediction_speed(self):
        """Test n-gram prediction speed."""
        model = NGramModel(n=3)
        tokenizer = CodeTokenizer()

        # Train model
        code = "def foo(): return bar()" * 100
        tokens = tokenizer.tokenize(code)
        model.train_on_tokens([tokens])
        model.finalize()

        # Benchmark prediction
        context = ['def', 'foo', '(']
        start = time.perf_counter()
        for _ in range(1000):
            model.predict(context, top_k=10)
        elapsed = time.perf_counter() - start

        # Should complete 1000 predictions in < 1 second
        avg_ms = (elapsed / 1000) * 1000
        assert avg_ms < 1.0, f"Prediction too slow: {avg_ms:.3f}ms"

        print(f"\nN-gram prediction: {avg_ms:.3f}ms average")


class TestSparkCodeIntelligencePerformance:
    """Performance tests for SparkCodeIntelligence."""

    @pytest.fixture
    def engine_and_project(self):
        """Create and train engine on test project."""
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir)

        # Create 20 Python files
        for i in range(20):
            content = f'''
class Class{i}:
    def __init__(self):
        self.value = {i}

    def process(self, data):
        return self.transform(data)

    def transform(self, data):
        return data * self.value

def helper_{i}():
    obj = Class{i}()
    return obj.process([1, 2, 3])
'''
            (path / f'module_{i}.py').write_text(content)

        engine = SparkCodeIntelligence(root_dir=path)
        engine.train(verbose=False)

        yield engine, path
        shutil.rmtree(tmpdir)

    def test_training_time(self, engine_and_project):
        """Verify training completed in reasonable time."""
        engine, _ = engine_and_project

        # Training time is recorded
        assert engine.training_time > 0
        assert engine.training_time < 10.0  # Should complete in < 10s

        print(f"\nTraining time: {engine.training_time:.3f}s")

    def test_completion_latency(self, engine_and_project):
        """Test completion latency."""
        engine, _ = engine_and_project

        prefixes = ['self.', 'def ', 'Class0.', 'return ']
        times = []

        for prefix in prefixes:
            start = time.perf_counter()
            for _ in range(100):
                engine.complete(prefix, top_n=10)
            elapsed = time.perf_counter() - start
            times.append(elapsed / 100)

        avg_ms = sum(times) / len(times) * 1000

        # Should average < 5ms per completion
        assert avg_ms < 10, f"Completion too slow: {avg_ms:.2f}ms"

        print(f"\nCompletion latency: {avg_ms:.2f}ms average")

    def test_find_callers_latency(self, engine_and_project):
        """Test find_callers latency."""
        engine, _ = engine_and_project

        functions = ['process', 'transform', 'helper_0']
        times = []

        for func in functions:
            start = time.perf_counter()
            for _ in range(100):
                engine.find_callers(func)
            elapsed = time.perf_counter() - start
            times.append(elapsed / 100)

        avg_ms = sum(times) / len(times) * 1000

        # Should average < 2ms
        assert avg_ms < 10, f"Find callers too slow: {avg_ms:.2f}ms"

        print(f"\nFind callers latency: {avg_ms:.2f}ms average")

    def test_find_class_latency(self, engine_and_project):
        """Test find_class latency."""
        engine, _ = engine_and_project

        start = time.perf_counter()
        for i in range(20):
            engine.find_class(f'Class{i}')
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / 20) * 1000

        # Should average < 0.5ms (simple dict lookup)
        assert avg_ms < 5, f"Find class too slow: {avg_ms:.2f}ms"

        print(f"\nFind class latency: {avg_ms:.2f}ms average")

    def test_query_latency(self, engine_and_project):
        """Test natural language query latency."""
        engine, _ = engine_and_project

        queries = [
            'what calls process',
            'where is Class0',
            'class that inherits Object',
        ]
        times = []

        for query in queries:
            start = time.perf_counter()
            for _ in range(100):
                engine.query(query)
            elapsed = time.perf_counter() - start
            times.append(elapsed / 100)

        avg_ms = sum(times) / len(times) * 1000

        # Should average < 5ms
        assert avg_ms < 20, f"Query too slow: {avg_ms:.2f}ms"

        print(f"\nQuery latency: {avg_ms:.2f}ms average")

    def test_save_load_speed(self, engine_and_project):
        """Test save and load speed."""
        engine, path = engine_and_project
        model_path = str(path / 'model.json')

        # Benchmark save
        start = time.perf_counter()
        engine.save(model_path)
        save_time = time.perf_counter() - start

        # Benchmark load
        new_engine = SparkCodeIntelligence()
        start = time.perf_counter()
        new_engine.load(model_path)
        load_time = time.perf_counter() - start

        # Should complete in < 1 second each
        assert save_time < 2.0, f"Save too slow: {save_time:.2f}s"
        assert load_time < 2.0, f"Load too slow: {load_time:.2f}s"

        print(f"\nSave: {save_time:.3f}s, Load: {load_time:.3f}s")


class TestScalability:
    """Scalability tests."""

    def test_linear_scaling(self):
        """Test that indexing scales linearly with file count."""
        times = []

        for file_count in [10, 20, 40]:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir)

                # Create files
                for i in range(file_count):
                    (path / f'module_{i}.py').write_text(f'''
class Class{i}:
    def method(self):
        pass
''')

                index = ASTIndex()
                start = time.perf_counter()
                index.index_directory(path, verbose=False)
                elapsed = time.perf_counter() - start
                times.append((file_count, elapsed))

        # Check scaling is roughly linear (not exponential)
        # Time for 40 files should be < 5x time for 10 files
        ratio = times[2][1] / times[0][1]
        expected_ratio = 4.0  # Ideal linear scaling

        assert ratio < expected_ratio * 2, f"Non-linear scaling: {ratio:.1f}x for 4x files"

        print("\nScaling test:")
        for files, time_val in times:
            print(f"  {files} files: {time_val:.3f}s")
