#!/usr/bin/env python3
"""
Tests for ML data collector orchestration extraction functionality.

Tests the orchestration.py module which extracts director orchestration
patterns from Claude Code transcripts including sub-agent detection,
batch classification (parallel vs sequential), and metrics collection.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from ml_collector.orchestration import (
    SubAgentExecution,
    OrchestrationBatch,
    ExtractedOrchestration,
    parse_agent_transcript,
    find_agent_transcripts,
    detect_batches,
    _create_batch,
    extract_orchestration_from_directory,
    save_orchestration,
    save_orchestration_lite,
    print_orchestration_summary,
    ORCHESTRATION_DIR,
    ORCHESTRATION_LITE_FILE,
)
from ml_collector.config import ML_DATA_DIR, TRACKED_DIR


class TestSubAgentExecution(unittest.TestCase):
    """Test SubAgentExecution dataclass."""

    def test_create_sub_agent_execution(self):
        """Test creating a SubAgentExecution instance."""
        agent = SubAgentExecution(
            agent_id="agent-123",
            session_id="session-456",
            model="claude-sonnet-4-20250514",
            started_at="2025-12-15T10:00:00Z",
            completed_at="2025-12-15T10:00:30Z",
            duration_ms=30000,
            tools_used=["Read", "Grep"],
            tool_count=5,
            thinking_blocks=2,
            has_error=False,
            error_summary=None,
            output_preview="Task completed successfully",
            transcript_path="/path/to/agent-123.jsonl",
        )

        self.assertEqual(agent.agent_id, "agent-123")
        self.assertEqual(agent.model, "claude-sonnet-4-20250514")
        self.assertEqual(agent.duration_ms, 30000)
        self.assertFalse(agent.has_error)
        self.assertEqual(len(agent.tools_used), 2)

    def test_sub_agent_to_dict(self):
        """Test serialization to dictionary."""
        agent = SubAgentExecution(
            agent_id="agent-test",
            session_id="session-test",
            model="test-model",
            started_at="2025-12-15T10:00:00Z",
            completed_at="2025-12-15T10:00:10Z",
            duration_ms=10000,
            tools_used=["Read"],
            tool_count=1,
            thinking_blocks=0,
            has_error=False,
            error_summary=None,
            output_preview="Done",
            transcript_path="/path/to/file.jsonl",
        )

        d = agent.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["agent_id"], "agent-test")
        self.assertEqual(d["tools_used"], ["Read"])
        self.assertIn("transcript_path", d)

    def test_sub_agent_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "agent_id": "restored-agent",
            "session_id": "restored-session",
            "model": "restored-model",
            "started_at": "2025-12-15T10:00:00Z",
            "completed_at": "2025-12-15T10:00:05Z",
            "duration_ms": 5000,
            "tools_used": ["Bash"],
            "tool_count": 3,
            "thinking_blocks": 1,
            "has_error": True,
            "error_summary": "Command failed",
            "output_preview": "Error occurred",
            "transcript_path": "/path/to/restored.jsonl",
        }

        agent = SubAgentExecution.from_dict(data)
        self.assertEqual(agent.agent_id, "restored-agent")
        self.assertTrue(agent.has_error)
        self.assertEqual(agent.error_summary, "Command failed")


class TestOrchestrationBatch(unittest.TestCase):
    """Test OrchestrationBatch dataclass."""

    def test_create_batch(self):
        """Test creating an OrchestrationBatch instance."""
        agent1 = SubAgentExecution(
            agent_id="agent-1", session_id="s1", model="model",
            started_at="2025-12-15T10:00:00Z", completed_at="2025-12-15T10:00:10Z",
            duration_ms=10000, tools_used=["Read"], tool_count=1,
            thinking_blocks=0, has_error=False, error_summary=None,
            output_preview="Done", transcript_path="/p1.jsonl",
        )
        agent2 = SubAgentExecution(
            agent_id="agent-2", session_id="s1", model="model",
            started_at="2025-12-15T10:00:01Z", completed_at="2025-12-15T10:00:15Z",
            duration_ms=14000, tools_used=["Grep"], tool_count=2,
            thinking_blocks=1, has_error=False, error_summary=None,
            output_preview="Done", transcript_path="/p2.jsonl",
        )

        batch = OrchestrationBatch(
            batch_index=0,
            execution_type="parallel",
            agents=[agent1, agent2],
            started_at="2025-12-15T10:00:00Z",
            completed_at="2025-12-15T10:00:15Z",
            duration_ms=15000,
            all_succeeded=True,
        )

        self.assertEqual(batch.execution_type, "parallel")
        self.assertEqual(len(batch.agents), 2)
        self.assertTrue(batch.all_succeeded)

    def test_batch_to_dict(self):
        """Test batch serialization."""
        agent = SubAgentExecution(
            agent_id="agent-1", session_id="s1", model="model",
            started_at="2025-12-15T10:00:00Z", completed_at="2025-12-15T10:00:10Z",
            duration_ms=10000, tools_used=[], tool_count=0,
            thinking_blocks=0, has_error=False, error_summary=None,
            output_preview="", transcript_path="/p.jsonl",
        )

        batch = OrchestrationBatch(
            batch_index=0, execution_type="sequential", agents=[agent],
            started_at="2025-12-15T10:00:00Z", completed_at="2025-12-15T10:00:10Z",
            duration_ms=10000, all_succeeded=True,
        )

        d = batch.to_dict()
        self.assertEqual(d["execution_type"], "sequential")
        self.assertEqual(len(d["agents"]), 1)
        self.assertIsInstance(d["agents"][0], dict)


class TestExtractedOrchestration(unittest.TestCase):
    """Test ExtractedOrchestration dataclass."""

    def test_create_empty_extraction(self):
        """Test creating an empty extraction result."""
        extraction = ExtractedOrchestration()

        self.assertFalse(extraction.orchestration_detected)
        self.assertEqual(extraction.total_sub_agents, 0)
        self.assertEqual(extraction.batches, [])
        self.assertEqual(extraction.version, 1)

    def test_extraction_to_dict(self):
        """Test extraction serialization."""
        extraction = ExtractedOrchestration(
            extracted_at="2025-12-15T10:00:00Z",
            parent_session_id="session-123",
            orchestration_detected=True,
            total_sub_agents=3,
            models_used=["claude-sonnet-4-20250514"],
            success_rate=100.0,
        )

        d = extraction.to_dict()
        self.assertEqual(d["version"], 1)
        self.assertEqual(d["parent_session_id"], "session-123")
        self.assertTrue(d["orchestration_detected"])


class TestParseAgentTranscript(unittest.TestCase):
    """Test parse_agent_transcript function."""

    def setUp(self):
        """Set up test environment with temporary directory."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()

    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        result = parse_agent_transcript(self.test_path / "nonexistent.jsonl")
        self.assertIsNone(result)

    def test_parse_empty_file(self):
        """Test parsing an empty file returns None."""
        empty_file = self.test_path / "agent-empty.jsonl"
        empty_file.touch()

        result = parse_agent_transcript(empty_file)
        self.assertIsNone(result)  # No timestamps = None

    def test_parse_valid_agent_transcript(self):
        """Test parsing a valid agent transcript."""
        transcript_file = self.test_path / "agent-test123.jsonl"

        # Create mock transcript entries
        entries = [
            {
                "timestamp": "2025-12-15T10:00:00Z",
                "agentId": "test123",
                "sessionId": "session-abc",
                "message": {
                    "model": "claude-sonnet-4-20250514",
                    "content": [
                        {"type": "thinking", "text": "Let me analyze..."},
                        {"type": "tool_use", "name": "Read", "input": {}},
                    ]
                }
            },
            {
                "timestamp": "2025-12-15T10:00:05Z",
                "message": {
                    "content": [
                        {"type": "tool_result", "content": "file contents"},
                        {"type": "text", "text": "I found the answer."},
                    ]
                }
            },
            {
                "timestamp": "2025-12-15T10:00:10Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "Task completed successfully."},
                    ]
                }
            }
        ]

        with open(transcript_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        result = parse_agent_transcript(transcript_file)

        self.assertIsNotNone(result)
        self.assertEqual(result.agent_id, "test123")
        self.assertEqual(result.session_id, "session-abc")
        self.assertEqual(result.model, "claude-sonnet-4-20250514")
        self.assertIn("Read", result.tools_used)
        self.assertEqual(result.thinking_blocks, 1)
        self.assertFalse(result.has_error)
        self.assertEqual(result.output_preview, "Task completed successfully.")

    def test_parse_transcript_with_error(self):
        """Test parsing transcript that has tool errors."""
        transcript_file = self.test_path / "agent-error.jsonl"

        entries = [
            {
                "timestamp": "2025-12-15T10:00:00Z",
                "agentId": "error-agent",
                "sessionId": "session-err",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Bash", "input": {}},
                    ]
                }
            },
            {
                "timestamp": "2025-12-15T10:00:05Z",
                "message": {
                    "content": [
                        {"type": "tool_result", "is_error": True, "content": "Command not found"},
                    ]
                }
            }
        ]

        with open(transcript_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        result = parse_agent_transcript(transcript_file)

        self.assertIsNotNone(result)
        self.assertTrue(result.has_error)
        self.assertIn("Command not found", result.error_summary)

    def test_parse_transcript_with_malformed_json(self):
        """Test parsing transcript with some malformed JSON lines."""
        transcript_file = self.test_path / "agent-partial.jsonl"

        with open(transcript_file, 'w') as f:
            # Valid entry
            f.write(json.dumps({"timestamp": "2025-12-15T10:00:00Z", "agentId": "partial"}) + '\n')
            # Malformed JSON - should be skipped
            f.write("not valid json\n")
            # Another valid entry
            f.write(json.dumps({"timestamp": "2025-12-15T10:00:05Z", "message": {}}) + '\n')

        result = parse_agent_transcript(transcript_file)

        # Should still parse successfully, skipping malformed lines
        self.assertIsNotNone(result)
        self.assertEqual(result.agent_id, "partial")


class TestFindAgentTranscripts(unittest.TestCase):
    """Test find_agent_transcripts function."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()

    def test_find_in_nonexistent_directory(self):
        """Test finding transcripts in non-existent directory."""
        result = find_agent_transcripts(self.test_path / "nonexistent")
        self.assertEqual(result, [])

    def test_find_in_empty_directory(self):
        """Test finding transcripts in empty directory."""
        result = find_agent_transcripts(self.test_path)
        self.assertEqual(result, [])

    def test_find_agent_transcripts(self):
        """Test finding agent transcript files."""
        # Create agent transcript files
        (self.test_path / "agent-001.jsonl").touch()
        (self.test_path / "agent-002.jsonl").touch()
        (self.test_path / "agent-003.jsonl").touch()
        # Create non-agent files (should be ignored)
        (self.test_path / "main-session.jsonl").touch()
        (self.test_path / "other.txt").touch()

        result = find_agent_transcripts(self.test_path)

        self.assertEqual(len(result), 3)
        self.assertTrue(all("agent-" in p.name for p in result))


class TestDetectBatches(unittest.TestCase):
    """Test detect_batches function."""

    def _create_agent(self, agent_id: str, start_time: str, duration_ms: int = 10000, has_error: bool = False) -> SubAgentExecution:
        """Helper to create agent execution."""
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end = start + timedelta(milliseconds=duration_ms)
        return SubAgentExecution(
            agent_id=agent_id,
            session_id="session-test",
            model="test-model",
            started_at=start_time,
            completed_at=end.isoformat().replace('+00:00', 'Z'),
            duration_ms=duration_ms,
            tools_used=[],
            tool_count=0,
            thinking_blocks=0,
            has_error=has_error,
            error_summary=None,
            output_preview="",
            transcript_path=f"/path/agent-{agent_id}.jsonl",
        )

    def test_detect_batches_empty_list(self):
        """Test batch detection with empty agent list."""
        result = detect_batches([])
        self.assertEqual(result, [])

    def test_detect_single_agent_sequential(self):
        """Test single agent is classified as sequential."""
        agent = self._create_agent("single", "2025-12-15T10:00:00Z")
        batches = detect_batches([agent])

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].execution_type, "sequential")
        self.assertEqual(len(batches[0].agents), 1)

    def test_detect_parallel_agents(self):
        """Test agents starting within threshold are classified as parallel."""
        # All start within 5 seconds (default threshold)
        agents = [
            self._create_agent("a1", "2025-12-15T10:00:00Z"),
            self._create_agent("a2", "2025-12-15T10:00:01Z"),
            self._create_agent("a3", "2025-12-15T10:00:02Z"),
        ]

        batches = detect_batches(agents)

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].execution_type, "parallel")
        self.assertEqual(len(batches[0].agents), 3)

    def test_detect_sequential_agents(self):
        """Test agents starting far apart are classified as sequential."""
        # Start times more than 5 seconds apart
        agents = [
            self._create_agent("a1", "2025-12-15T10:00:00Z"),
            self._create_agent("a2", "2025-12-15T10:00:30Z"),  # 30s later
            self._create_agent("a3", "2025-12-15T10:01:00Z"),  # 30s later
        ]

        batches = detect_batches(agents)

        self.assertEqual(len(batches), 3)
        self.assertTrue(all(b.execution_type == "sequential" for b in batches))

    def test_detect_mixed_batches(self):
        """Test mixed parallel and sequential batches."""
        agents = [
            # First parallel batch (start within 2 seconds)
            self._create_agent("a1", "2025-12-15T10:00:00Z"),
            self._create_agent("a2", "2025-12-15T10:00:02Z"),
            # Second parallel batch (starts 30s later, within 3 seconds)
            self._create_agent("a3", "2025-12-15T10:00:30Z"),
            self._create_agent("a4", "2025-12-15T10:00:33Z"),
        ]

        batches = detect_batches(agents)

        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].execution_type, "parallel")
        self.assertEqual(len(batches[0].agents), 2)
        self.assertEqual(batches[1].execution_type, "parallel")
        self.assertEqual(len(batches[1].agents), 2)

    def test_detect_batches_custom_threshold(self):
        """Test batch detection with custom threshold."""
        agents = [
            self._create_agent("a1", "2025-12-15T10:00:00Z"),
            self._create_agent("a2", "2025-12-15T10:00:08Z"),  # 8s later
        ]

        # With default 5s threshold, should be sequential
        batches_default = detect_batches(agents, threshold_ms=5000)
        self.assertEqual(len(batches_default), 2)

        # With 10s threshold, should be parallel
        batches_custom = detect_batches(agents, threshold_ms=10000)
        self.assertEqual(len(batches_custom), 1)
        self.assertEqual(batches_custom[0].execution_type, "parallel")

    def test_batch_all_succeeded_flag(self):
        """Test all_succeeded flag in batches."""
        # Batch with no errors
        agents_ok = [
            self._create_agent("ok1", "2025-12-15T10:00:00Z", has_error=False),
            self._create_agent("ok2", "2025-12-15T10:00:01Z", has_error=False),
        ]
        batches_ok = detect_batches(agents_ok)
        self.assertTrue(batches_ok[0].all_succeeded)

        # Batch with one error
        agents_err = [
            self._create_agent("ok", "2025-12-15T10:00:00Z", has_error=False),
            self._create_agent("err", "2025-12-15T10:00:01Z", has_error=True),
        ]
        batches_err = detect_batches(agents_err)
        self.assertFalse(batches_err[0].all_succeeded)


class TestExtractOrchestrationFromDirectory(unittest.TestCase):
    """Test extract_orchestration_from_directory function."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()

    def test_extract_from_empty_directory(self):
        """Test extraction from directory with no agent transcripts."""
        result = extract_orchestration_from_directory(self.test_path)

        self.assertFalse(result.orchestration_detected)
        self.assertEqual(result.total_sub_agents, 0)
        self.assertEqual(result.batches, [])

    def test_extract_from_directory_with_agents(self):
        """Test extraction from directory with agent transcripts."""
        # Create two agent transcript files
        for i, (agent_id, start_offset) in enumerate([("agent1", 0), ("agent2", 2)]):
            transcript_file = self.test_path / f"agent-{agent_id}.jsonl"
            start_time = f"2025-12-15T10:00:0{start_offset}Z"
            entries = [
                {
                    "timestamp": start_time,
                    "agentId": agent_id,
                    "sessionId": "shared-session",
                    "message": {
                        "model": "claude-sonnet-4-20250514",
                        "content": [{"type": "tool_use", "name": "Read", "input": {}}]
                    }
                },
                {
                    "timestamp": f"2025-12-15T10:00:1{i}Z",
                    "message": {
                        "content": [{"type": "text", "text": f"Agent {agent_id} done"}]
                    }
                }
            ]
            with open(transcript_file, 'w') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')

        result = extract_orchestration_from_directory(self.test_path)

        self.assertTrue(result.orchestration_detected)
        self.assertEqual(result.total_sub_agents, 2)
        self.assertEqual(result.parent_session_id, "shared-session")
        self.assertIn("claude-sonnet-4-20250514", result.models_used)
        self.assertIn("Read", result.unique_tools)

    def test_extract_with_session_filter(self):
        """Test extraction filtered by parent session ID."""
        # Create agents for different sessions
        for session, agent_id in [("session-A", "agent1"), ("session-B", "agent2")]:
            transcript_file = self.test_path / f"agent-{agent_id}.jsonl"
            entries = [
                {
                    "timestamp": "2025-12-15T10:00:00Z",
                    "agentId": agent_id,
                    "sessionId": session,
                    "message": {"content": []}
                },
                {
                    "timestamp": "2025-12-15T10:00:05Z",
                    "message": {"content": [{"type": "text", "text": "Done"}]}
                }
            ]
            with open(transcript_file, 'w') as f:
                for entry in entries:
                    f.write(json.dumps(entry) + '\n')

        # Filter by session-A
        result = extract_orchestration_from_directory(
            self.test_path,
            parent_session_id="session-A"
        )

        self.assertTrue(result.orchestration_detected)
        self.assertEqual(result.total_sub_agents, 1)
        self.assertEqual(result.parent_session_id, "session-A")


class TestSaveOrchestration(unittest.TestCase):
    """Test save_orchestration and save_orchestration_lite functions."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)
        self.output_dir = self.test_path / "orchestration"
        self.lite_file = self.test_path / "orchestration.jsonl"

    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()

    def test_save_orchestration_creates_file(self):
        """Test saving orchestration creates JSON file."""
        extraction = ExtractedOrchestration(
            extracted_at="2025-12-15T10:00:00Z",
            parent_session_id="session-save-test",
            orchestration_detected=True,
            total_sub_agents=2,
        )

        filepath = save_orchestration(extraction, output_dir=self.output_dir)

        self.assertTrue(filepath.exists())
        self.assertTrue(filepath.name.endswith(".json"))

        # Verify content
        with open(filepath) as f:
            data = json.load(f)
        self.assertEqual(data["parent_session_id"], "session-save-test")
        self.assertEqual(data["total_sub_agents"], 2)

    @patch('ml_collector.orchestration.cali_put')
    @patch('ml_collector.orchestration.cali_exists')
    def test_save_orchestration_lite_appends(self, mock_cali_exists, mock_cali_put):
        """Test saving lite orchestration appends to JSONL."""
        # Mock CALI functions to ensure file is written (not returning early)
        mock_cali_exists.return_value = False
        mock_cali_put.return_value = True

        # Create output directory
        self.test_path.mkdir(parents=True, exist_ok=True)
        lite_file = self.test_path / "orchestration.jsonl"

        extraction = ExtractedOrchestration(
            extracted_at="2025-12-15T10:00:00Z",
            parent_session_id="session-lite",
            orchestration_detected=True,
            total_sub_agents=3,
            models_used=["claude-sonnet-4-20250514"],
        )

        # Patch file paths to use test directory
        with patch('ml_collector.orchestration.ORCHESTRATION_LITE_FILE', lite_file):
            with patch('ml_collector.orchestration.TRACKED_DIR', self.test_path):
                result = save_orchestration_lite(extraction)

        self.assertIsNotNone(result)
        self.assertTrue(lite_file.exists())

        # Verify JSONL content
        with open(lite_file) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

        data = json.loads(lines[0])
        self.assertEqual(data["session_id"], "session-lite")
        self.assertEqual(data["total_sub_agents"], 3)

    def test_save_orchestration_lite_skips_empty(self):
        """Test lite save skips when no orchestration detected."""
        extraction = ExtractedOrchestration(orchestration_detected=False)

        with patch('ml_collector.orchestration.ORCHESTRATION_LITE_FILE', self.lite_file):
            result = save_orchestration_lite(extraction)

        self.assertIsNone(result)
        self.assertFalse(self.lite_file.exists())


class TestPrintOrchestrationSummary(unittest.TestCase):
    """Test print_orchestration_summary function."""

    def test_print_no_orchestration(self):
        """Test printing when no orchestration detected."""
        extraction = ExtractedOrchestration(orchestration_detected=False)

        # Should not raise, just print message
        from io import StringIO
        import sys
        captured = StringIO()
        sys.stdout = captured
        try:
            print_orchestration_summary(extraction)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("No orchestration detected", output)

    def test_print_with_orchestration(self):
        """Test printing orchestration summary."""
        agent = SubAgentExecution(
            agent_id="agent-print-test",
            session_id="session-print",
            model="claude-sonnet-4-20250514",
            started_at="2025-12-15T10:00:00Z",
            completed_at="2025-12-15T10:00:10Z",
            duration_ms=10000,
            tools_used=["Read", "Grep"],
            tool_count=5,
            thinking_blocks=2,
            has_error=False,
            error_summary=None,
            output_preview="Done",
            transcript_path="/path/to/file.jsonl",
        )

        batch = OrchestrationBatch(
            batch_index=0,
            execution_type="parallel",
            agents=[agent],
            started_at="2025-12-15T10:00:00Z",
            completed_at="2025-12-15T10:00:10Z",
            duration_ms=10000,
            all_succeeded=True,
        )

        extraction = ExtractedOrchestration(
            extracted_at="2025-12-15T10:00:00Z",
            parent_session_id="session-print",
            orchestration_detected=True,
            total_sub_agents=1,
            batches=[batch],
            models_used=["claude-sonnet-4-20250514"],
            total_tools_used=5,
            unique_tools=["Read", "Grep"],
            success_rate=100.0,
            total_duration_ms=10000,
        )

        from io import StringIO
        import sys
        captured = StringIO()
        sys.stdout = captured
        try:
            print_orchestration_summary(extraction)
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        self.assertIn("ORCHESTRATION EXTRACTION SUMMARY", output)
        self.assertIn("session-print", output)
        self.assertIn("Sub-agents spawned: 1", output)
        self.assertIn("Success rate: 100.0%", output)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_path = Path(self.test_dir.name)

    def tearDown(self):
        """Clean up test environment."""
        self.test_dir.cleanup()

    def test_parse_transcript_no_timestamps(self):
        """Test parsing transcript with no timestamp fields returns None."""
        transcript_file = self.test_path / "agent-no-ts.jsonl"
        entries = [
            {"agentId": "no-ts", "message": {"content": []}},
            {"message": {"content": [{"type": "text", "text": "done"}]}},
        ]
        with open(transcript_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        result = parse_agent_transcript(transcript_file)
        self.assertIsNone(result)

    def test_parse_transcript_invalid_timestamp_format(self):
        """Test parsing with invalid timestamp format handles gracefully."""
        transcript_file = self.test_path / "agent-bad-ts.jsonl"
        entries = [
            {"timestamp": "not-a-valid-timestamp", "agentId": "bad-ts"},
            {"timestamp": "also-invalid"},
        ]
        with open(transcript_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')

        result = parse_agent_transcript(transcript_file)

        # Should return agent but with 0 duration due to parse failure
        self.assertIsNotNone(result)
        self.assertEqual(result.duration_ms, 0)

    def test_detect_batches_agents_with_empty_timestamps(self):
        """Test batch detection handles agents with empty timestamps."""
        agent_good = SubAgentExecution(
            agent_id="good", session_id="s", model="m",
            started_at="2025-12-15T10:00:00Z", completed_at="2025-12-15T10:00:10Z",
            duration_ms=10000, tools_used=[], tool_count=0,
            thinking_blocks=0, has_error=False, error_summary=None,
            output_preview="", transcript_path="/p.jsonl",
        )
        agent_empty = SubAgentExecution(
            agent_id="empty", session_id="s", model="m",
            started_at="", completed_at="",  # Empty timestamps
            duration_ms=0, tools_used=[], tool_count=0,
            thinking_blocks=0, has_error=False, error_summary=None,
            output_preview="", transcript_path="/p2.jsonl",
        )

        # Should handle without crashing
        batches = detect_batches([agent_good, agent_empty])
        self.assertIsInstance(batches, list)
        self.assertGreater(len(batches), 0)


if __name__ == "__main__":
    unittest.main()
