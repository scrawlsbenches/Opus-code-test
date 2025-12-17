"""
Orchestration extraction module for ML Data Collector.

Extracts director orchestration patterns from Claude Code transcripts by:
1. Scanning for sub-agent transcript files (agent-*.jsonl)
2. Linking sub-agents to parent sessions via sessionId
3. Detecting parallel vs sequential execution patterns
4. Extracting sub-agent metadata, tools used, and outcomes

Architecture:
    Main transcript ({uuid}.jsonl):
        - isSidechain: false
        - Contains parent session tool calls

    Sub-agent transcript (agent-{id}.jsonl):
        - isSidechain: true
        - agentId: unique identifier
        - sessionId: links to parent session
        - Contains full sub-agent execution trace
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .config import ML_DATA_DIR, TRACKED_DIR, CALI_DIR
from .persistence import cali_put, cali_exists


logger = logging.getLogger(__name__)


# Git-tracked JSONL file for orchestration (append-only, git-friendly)
ORCHESTRATION_LITE_FILE = TRACKED_DIR / "orchestration.jsonl"


# Storage directory for extracted orchestration data
ORCHESTRATION_DIR = ML_DATA_DIR / "orchestration"


@dataclass
class SubAgentExecution:
    """Extracted data from a sub-agent transcript."""
    agent_id: str
    session_id: str
    model: str
    started_at: str
    completed_at: str
    duration_ms: int
    tools_used: List[str]
    tool_count: int
    thinking_blocks: int
    has_error: bool
    error_summary: Optional[str]
    output_preview: str  # First 500 chars of final output
    transcript_path: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SubAgentExecution':
        """Create from dictionary."""
        return cls(**d)


@dataclass
class OrchestrationBatch:
    """A batch of sub-agents that ran together (parallel or sequential)."""
    batch_index: int
    execution_type: str  # "parallel" or "sequential"
    agents: List[SubAgentExecution]
    started_at: str
    completed_at: str
    duration_ms: int
    all_succeeded: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "batch_index": self.batch_index,
            "execution_type": self.execution_type,
            "agents": [a.to_dict() for a in self.agents],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "all_succeeded": self.all_succeeded,
        }


@dataclass
class ExtractedOrchestration:
    """Complete orchestration extraction from a session."""
    version: int = 1
    extracted_at: str = ""
    parent_session_id: str = ""
    parent_transcript_path: str = ""
    orchestration_detected: bool = False
    total_sub_agents: int = 0
    batches: List[OrchestrationBatch] = field(default_factory=list)
    models_used: List[str] = field(default_factory=list)
    total_tools_used: int = 0
    unique_tools: List[str] = field(default_factory=list)
    total_duration_ms: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "extracted_at": self.extracted_at,
            "parent_session_id": self.parent_session_id,
            "parent_transcript_path": self.parent_transcript_path,
            "orchestration_detected": self.orchestration_detected,
            "total_sub_agents": self.total_sub_agents,
            "batches": [b.to_dict() for b in self.batches],
            "models_used": self.models_used,
            "total_tools_used": self.total_tools_used,
            "unique_tools": self.unique_tools,
            "total_duration_ms": self.total_duration_ms,
            "success_rate": self.success_rate,
        }


def parse_agent_transcript(filepath: Path) -> Optional[SubAgentExecution]:
    """
    Parse a sub-agent transcript file and extract execution data.

    Args:
        filepath: Path to agent-*.jsonl file

    Returns:
        SubAgentExecution object or None if parsing fails
    """
    if not filepath.exists():
        logger.warning(f"Agent transcript not found: {filepath}")
        return None

    agent_id = None
    session_id = None
    model = "unknown"
    timestamps = []
    tools_used = set()
    tool_count = 0
    thinking_count = 0
    has_error = False
    error_summary = None
    final_output = ""

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract metadata from first entry
                if agent_id is None:
                    agent_id = entry.get('agentId', filepath.stem.replace('agent-', ''))
                    session_id = entry.get('sessionId', '')

                # Get timestamp
                ts = entry.get('timestamp')
                if ts:
                    timestamps.append(ts)

                # Get model from message
                message = entry.get('message', {})
                if message.get('model'):
                    model = message['model']

                # Extract tool usage and content
                content = message.get('content', [])
                if isinstance(content, list):
                    for block in content:
                        block_type = block.get('type')

                        if block_type == 'tool_use':
                            tool_name = block.get('name', '')
                            if tool_name:
                                tools_used.add(tool_name)
                                tool_count += 1

                        elif block_type == 'thinking':
                            thinking_count += 1

                        elif block_type == 'text':
                            text = block.get('text', '')
                            if text:
                                final_output = text  # Keep updating to get last text

                        elif block_type == 'tool_result':
                            # Check for errors in tool results
                            if block.get('is_error'):
                                has_error = True
                                error_summary = str(block.get('content', ''))[:200]

        if not timestamps:
            return None

        # Parse timestamps
        try:
            start_time = datetime.fromisoformat(timestamps[0].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(timestamps[-1].replace('Z', '+00:00'))
            duration_ms = int((end_time - start_time).total_seconds() * 1000)
        except (ValueError, IndexError):
            duration_ms = 0

        return SubAgentExecution(
            agent_id=agent_id or filepath.stem,
            session_id=session_id or '',
            model=model,
            started_at=timestamps[0] if timestamps else '',
            completed_at=timestamps[-1] if timestamps else '',
            duration_ms=duration_ms,
            tools_used=list(tools_used),
            tool_count=tool_count,
            thinking_blocks=thinking_count,
            has_error=has_error,
            error_summary=error_summary,
            output_preview=final_output[:500] if final_output else '',
            transcript_path=str(filepath),
        )

    except IOError as e:
        logger.error(f"Error reading agent transcript {filepath}: {e}")
        return None


def find_agent_transcripts(project_dir: Path) -> List[Path]:
    """
    Find all sub-agent transcript files in a project directory.

    Args:
        project_dir: Path to Claude project transcript directory

    Returns:
        List of paths to agent-*.jsonl files
    """
    if not project_dir.exists():
        return []

    return sorted(project_dir.glob('agent-*.jsonl'))


def detect_batches(agents: List[SubAgentExecution], threshold_ms: int = 5000) -> List[OrchestrationBatch]:
    """
    Detect execution batches from a list of sub-agents.

    Agents that started within threshold_ms of each other are considered
    part of the same parallel batch.

    Args:
        agents: List of SubAgentExecution objects
        threshold_ms: Time window for parallel detection (default 5 seconds)

    Returns:
        List of OrchestrationBatch objects
    """
    if not agents:
        return []

    # Sort by start time
    sorted_agents = sorted(agents, key=lambda a: a.started_at)

    batches = []
    current_batch_agents = [sorted_agents[0]]

    for agent in sorted_agents[1:]:
        # Check if this agent started close to the first agent in current batch
        try:
            batch_start = datetime.fromisoformat(
                current_batch_agents[0].started_at.replace('Z', '+00:00')
            )
            agent_start = datetime.fromisoformat(
                agent.started_at.replace('Z', '+00:00')
            )
            gap_ms = abs((agent_start - batch_start).total_seconds() * 1000)

            if gap_ms <= threshold_ms:
                # Same batch (parallel)
                current_batch_agents.append(agent)
            else:
                # New batch
                batches.append(_create_batch(current_batch_agents, len(batches)))
                current_batch_agents = [agent]

        except (ValueError, AttributeError):
            # Can't parse timestamps, assume sequential
            batches.append(_create_batch(current_batch_agents, len(batches)))
            current_batch_agents = [agent]

    # Don't forget the last batch
    if current_batch_agents:
        batches.append(_create_batch(current_batch_agents, len(batches)))

    return batches


def _create_batch(agents: List[SubAgentExecution], index: int) -> OrchestrationBatch:
    """Create a batch from a list of agents."""
    execution_type = "parallel" if len(agents) > 1 else "sequential"

    # Get batch timing
    start_times = [a.started_at for a in agents if a.started_at]
    end_times = [a.completed_at for a in agents if a.completed_at]

    started_at = min(start_times) if start_times else ''
    completed_at = max(end_times) if end_times else ''

    # Calculate duration
    try:
        start = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
        end = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
        duration_ms = int((end - start).total_seconds() * 1000)
    except (ValueError, AttributeError):
        duration_ms = sum(a.duration_ms for a in agents)

    all_succeeded = not any(a.has_error for a in agents)

    return OrchestrationBatch(
        batch_index=index,
        execution_type=execution_type,
        agents=agents,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        all_succeeded=all_succeeded,
    )


def extract_orchestration_from_directory(
    project_dir: Path,
    parent_session_id: Optional[str] = None
) -> ExtractedOrchestration:
    """
    Extract orchestration data from a Claude project transcript directory.

    Args:
        project_dir: Path to directory containing transcript files
        parent_session_id: Optional session ID to filter by

    Returns:
        ExtractedOrchestration object with all extracted data
    """
    result = ExtractedOrchestration(
        extracted_at=datetime.now().isoformat(),
    )

    # Find agent transcripts
    agent_files = find_agent_transcripts(project_dir)

    if not agent_files:
        logger.info(f"No agent transcripts found in {project_dir}")
        return result

    # Parse each agent transcript
    agents: List[SubAgentExecution] = []
    for filepath in agent_files:
        agent = parse_agent_transcript(filepath)
        if agent:
            # Filter by session if specified
            if parent_session_id and agent.session_id != parent_session_id:
                continue
            agents.append(agent)

    if not agents:
        return result

    # Get parent session info
    session_ids = set(a.session_id for a in agents)
    if len(session_ids) == 1:
        result.parent_session_id = session_ids.pop()
        # Try to find parent transcript
        parent_transcript = project_dir / f"{result.parent_session_id}.jsonl"
        if parent_transcript.exists():
            result.parent_transcript_path = str(parent_transcript)

    # Detect batches
    batches = detect_batches(agents)

    # Aggregate statistics
    all_tools = set()
    total_tool_count = 0
    models = set()
    success_count = 0

    for agent in agents:
        all_tools.update(agent.tools_used)
        total_tool_count += agent.tool_count
        models.add(agent.model)
        if not agent.has_error:
            success_count += 1

    # Calculate total duration (from first start to last end)
    if agents:
        try:
            starts = [datetime.fromisoformat(a.started_at.replace('Z', '+00:00'))
                     for a in agents if a.started_at]
            ends = [datetime.fromisoformat(a.completed_at.replace('Z', '+00:00'))
                   for a in agents if a.completed_at]
            if starts and ends:
                total_duration = int((max(ends) - min(starts)).total_seconds() * 1000)
            else:
                total_duration = sum(a.duration_ms for a in agents)
        except (ValueError, AttributeError):
            total_duration = sum(a.duration_ms for a in agents)
    else:
        total_duration = 0

    # Build result
    result.orchestration_detected = True
    result.total_sub_agents = len(agents)
    result.batches = batches
    result.models_used = sorted(models)
    result.total_tools_used = total_tool_count
    result.unique_tools = sorted(all_tools)
    result.total_duration_ms = total_duration
    result.success_rate = (success_count / len(agents) * 100) if agents else 0.0

    return result


def save_orchestration(
    extraction: ExtractedOrchestration,
    output_dir: Optional[Path] = None
) -> Path:
    """
    Save extracted orchestration data to a JSON file.

    Args:
        extraction: ExtractedOrchestration object
        output_dir: Output directory (default: .git-ml/orchestration/)

    Returns:
        Path to saved file
    """
    out_dir = output_dir or ORCHESTRATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from session ID or timestamp
    if extraction.parent_session_id:
        filename = f"{extraction.parent_session_id}_orchestration.json"
    else:
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_orchestration.json"

    filepath = out_dir / filename

    with open(filepath, 'w') as f:
        json.dump(extraction.to_dict(), f, indent=2)

    logger.info(f"Saved orchestration data to {filepath}")
    return filepath


def save_orchestration_lite(extraction: ExtractedOrchestration) -> Optional[Path]:
    """
    Save orchestration data to git-tracked JSONL file (append-only).

    Creates a lightweight, single-line JSON entry suitable for git tracking.
    Skips if no orchestration was detected.

    Args:
        extraction: ExtractedOrchestration object

    Returns:
        Path to JSONL file, or None if nothing saved
    """
    if not extraction.orchestration_detected:
        return None

    TRACKED_DIR.mkdir(parents=True, exist_ok=True)

    # Create lightweight record (flatten for JSONL)
    lite_record = {
        "session_id": extraction.parent_session_id,
        "extracted_at": extraction.extracted_at,
        "total_sub_agents": extraction.total_sub_agents,
        "batch_count": len(extraction.batches),
        "models_used": extraction.models_used,
        "total_tools_used": extraction.total_tools_used,
        "unique_tools": extraction.unique_tools,
        "total_duration_ms": extraction.total_duration_ms,
        "success_rate": extraction.success_rate,
        # Flatten batch info
        "batches": [
            {
                "execution_type": b.execution_type,
                "agent_count": len(b.agents),
                "duration_ms": b.duration_ms,
                "all_succeeded": b.all_succeeded,
                "agent_ids": [a.agent_id for a in b.agents],
            }
            for b in extraction.batches
        ]
    }

    # CALI: O(1) existence check
    if cali_exists('orchestration', extraction.parent_session_id):
        logger.debug(f"CALI: orchestration {extraction.parent_session_id} already exists")
        return ORCHESTRATION_LITE_FILE

    # Append to JSONL file
    with open(ORCHESTRATION_LITE_FILE, 'a') as f:
        f.write(json.dumps(lite_record) + '\n')

    # Also write to CALI for O(1) lookups and git-friendly storage
    if cali_put('orchestration', extraction.parent_session_id, lite_record):
        logger.debug(f"CALI: stored orchestration {extraction.parent_session_id}")

    logger.info(f"Appended orchestration to {ORCHESTRATION_LITE_FILE}")
    return ORCHESTRATION_LITE_FILE


def extract_and_save(
    project_dir: Path,
    output_dir: Optional[Path] = None,
    parent_session_id: Optional[str] = None,
    save_lite: bool = True
) -> Tuple[ExtractedOrchestration, Optional[Path]]:
    """
    Extract orchestration from directory and save if data found.

    Args:
        project_dir: Path to Claude project transcript directory
        output_dir: Output directory for saved data
        parent_session_id: Optional session ID to filter by
        save_lite: Also save to git-tracked JSONL file (default True)

    Returns:
        Tuple of (ExtractedOrchestration, path_to_saved_file or None)
    """
    extraction = extract_orchestration_from_directory(project_dir, parent_session_id)

    if extraction.orchestration_detected:
        saved_path = save_orchestration(extraction, output_dir)
        # Also save lightweight version for git tracking
        if save_lite:
            save_orchestration_lite(extraction)
        return extraction, saved_path

    return extraction, None


def print_orchestration_summary(extraction: ExtractedOrchestration) -> None:
    """Print a human-readable summary of extracted orchestration."""
    if not extraction.orchestration_detected:
        print("No orchestration detected in this session.")
        return

    print("=" * 60)
    print("ORCHESTRATION EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"\nSession: {extraction.parent_session_id or 'Unknown'}")
    print(f"Extracted: {extraction.extracted_at}")
    print(f"\nSub-agents spawned: {extraction.total_sub_agents}")
    print(f"Batches detected: {len(extraction.batches)}")
    print(f"Success rate: {extraction.success_rate:.1f}%")
    print(f"Total duration: {extraction.total_duration_ms}ms")

    print(f"\nModels used: {', '.join(extraction.models_used)}")
    print(f"Tools used ({extraction.total_tools_used} calls): {', '.join(extraction.unique_tools)}")

    print("\nBatch Details:")
    for batch in extraction.batches:
        print(f"\n  Batch {batch.batch_index} ({batch.execution_type}):")
        print(f"    Agents: {len(batch.agents)}")
        print(f"    Duration: {batch.duration_ms}ms")
        print(f"    Success: {'Yes' if batch.all_succeeded else 'No'}")
        for agent in batch.agents:
            status = "OK" if not agent.has_error else "ERROR"
            print(f"      - {agent.agent_id}: {agent.model} [{status}] {agent.tool_count} tools")

    print("\n" + "=" * 60)
