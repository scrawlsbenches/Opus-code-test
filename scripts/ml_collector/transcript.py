"""
Transcript parsing module for ML Data Collector

Handles parsing of Claude Code transcript JSONL files and extraction of exchanges.
"""

import json
import shlex
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .data_classes import TranscriptExchange, ChatEntry
from .persistence import generate_chat_id, save_chat_entry
from .session import add_chat_to_session, generate_session_id


logger = logging.getLogger(__name__)


# ============================================================================
# TRANSCRIPT PARSING (for automatic session capture via Stop hook)
# ============================================================================

def parse_transcript_jsonl(filepath: Path) -> List[TranscriptExchange]:
    """Parse a Claude Code transcript JSONL file into exchanges.

    The JSONL format has entries with:
    - type: "user" or "assistant"
    - message.content: string (user) or array of content blocks (assistant)
    - timestamp: ISO timestamp

    Returns list of TranscriptExchange objects.
    """
    if not filepath.exists():
        logger.warning(f"Transcript file not found: {filepath}")
        return []

    exchanges = []
    current_query = None
    current_response_parts = []
    current_tools = []
    current_tool_inputs = []
    current_thinking = None
    current_timestamp = None

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

                entry_type = entry.get('type')
                message = entry.get('message', {})
                timestamp = entry.get('timestamp', '')

                if entry_type == 'user':
                    # Save previous exchange if we have one
                    if current_query and current_response_parts:
                        exchanges.append(TranscriptExchange(
                            query=current_query,
                            response=' '.join(current_response_parts),
                            tools_used=current_tools,
                            tool_inputs=current_tool_inputs,
                            timestamp=current_timestamp or timestamp,
                            thinking=current_thinking,
                        ))

                    # Start new exchange
                    content = message.get('content', '')
                    if isinstance(content, str):
                        current_query = content
                    elif isinstance(content, list):
                        # Extract text from content blocks
                        current_query = ' '.join(
                            c.get('text', '') for c in content
                            if c.get('type') == 'text'
                        )
                    current_response_parts = []
                    current_tools = []
                    current_tool_inputs = []
                    current_thinking = None
                    current_timestamp = timestamp

                elif entry_type == 'assistant':
                    content = message.get('content', [])
                    if isinstance(content, list):
                        for block in content:
                            block_type = block.get('type')

                            if block_type == 'text':
                                text = block.get('text', '')
                                if text:
                                    current_response_parts.append(text)

                            elif block_type == 'thinking':
                                current_thinking = block.get('thinking', '')

                            elif block_type == 'tool_use':
                                tool_name = block.get('name', '')
                                tool_input = block.get('input', {})
                                if tool_name and tool_name not in current_tools:
                                    current_tools.append(tool_name)
                                current_tool_inputs.append({
                                    'tool': tool_name,
                                    'input': tool_input,
                                })

        # Don't forget the last exchange
        if current_query and current_response_parts:
            exchanges.append(TranscriptExchange(
                query=current_query,
                response=' '.join(current_response_parts),
                tools_used=current_tools,
                tool_inputs=current_tool_inputs,
                timestamp=current_timestamp or '',
                thinking=current_thinking,
            ))

    except IOError as e:
        logger.error(f"Error reading transcript: {e}")
        return []

    return exchanges


def extract_files_from_tool_inputs(tool_inputs: List[Dict]) -> tuple:
    """Extract file references and modifications from tool inputs.

    Returns (files_referenced, files_modified) tuple.
    """
    files_referenced = set()
    files_modified = set()

    for ti in tool_inputs:
        tool = ti.get('tool', '')
        inp = ti.get('input', {})

        if tool == 'Read':
            path = inp.get('file_path', '')
            if path:
                files_referenced.add(path)

        elif tool in ('Edit', 'Write', 'MultiEdit'):
            path = inp.get('file_path', '')
            if path:
                files_modified.add(path)

        elif tool == 'NotebookEdit':
            # NotebookEdit modifies notebooks
            path = inp.get('notebook_path', '')
            if path:
                files_modified.add(path)

        elif tool == 'Bash':
            # Try to extract file paths from command
            cmd = inp.get('command', '')

            # Common file extensions to track
            FILE_EXTENSIONS = (
                '.py', '.md', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
                '.txt', '.rst', '.sh', '.bash', '.zsh',
                '.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs',
                '.html', '.css', '.scss', '.less',
                '.c', '.cpp', '.h', '.hpp', '.cc',
                '.java', '.kt', '.scala',
                '.go', '.rs', '.rb', '.php', '.pl',
                '.sql', '.graphql',
                '.xml', '.csv', '.env',
                '.dockerfile', 'Dockerfile', 'Makefile', 'Jenkinsfile'
            )

            # Use shlex.split() for safer parsing (handles quoted paths)
            try:
                words = shlex.split(cmd)
            except ValueError:
                # Fallback to simple split if shlex fails
                words = cmd.split()

            for word in words:
                # Strip quotes that might remain
                word = word.strip('\'"')

                # Skip flags/options
                if word.startswith('-'):
                    # But check if it's a flag with a value like --cov="file.py"
                    if '=' in word:
                        # Extract the value part after =
                        _, value = word.split('=', 1)
                        value = value.strip('\'"')
                        if any(value.endswith(ext) for ext in FILE_EXTENSIONS):
                            files_referenced.add(value)
                    continue

                # Check if it ends with a tracked extension
                if any(word.endswith(ext) for ext in FILE_EXTENSIONS):
                    files_referenced.add(word)
                # Also catch special files without extensions (case-insensitive)
                elif any(word.lower().endswith(name.lower()) for name in ('Dockerfile', 'Makefile', 'Jenkinsfile')):
                    files_referenced.add(word)

        elif tool == 'Glob':
            path = inp.get('path', '')
            if path:
                files_referenced.add(path)

        elif tool == 'Grep':
            path = inp.get('path', '')
            if path:
                files_referenced.add(path)

    return list(files_referenced), list(files_modified)


def process_transcript(
    filepath: Path,
    session_id: Optional[str] = None,
    save_exchanges: bool = True
) -> Dict[str, Any]:
    """Process a transcript file and optionally save exchanges.

    Args:
        filepath: Path to the JSONL transcript
        session_id: Optional session ID to use (extracted from transcript if not provided)
        save_exchanges: Whether to save exchanges to .git-ml/chats/

    Returns:
        Summary dict with counts and extracted data.
    """
    exchanges = parse_transcript_jsonl(filepath)

    if not exchanges:
        return {'status': 'empty', 'exchanges': 0}

    # Use provided session_id or generate one
    if not session_id:
        session_id = generate_session_id()

    saved_count = 0
    total_tools = set()
    all_files_ref = set()
    all_files_mod = set()

    for ex in exchanges:
        files_ref, files_mod = extract_files_from_tool_inputs(ex.tool_inputs)
        all_files_ref.update(files_ref)
        all_files_mod.update(files_mod)
        total_tools.update(ex.tools_used)

        if save_exchanges:
            try:
                entry = ChatEntry(
                    id=generate_chat_id(),
                    timestamp=ex.timestamp or datetime.now().isoformat(),
                    session_id=session_id,
                    query=ex.query[:10000],  # Limit query length
                    response=ex.response[:50000],  # Limit response length
                    files_referenced=files_ref,
                    files_modified=files_mod,
                    tools_used=ex.tools_used,
                    query_tokens=len(ex.query.split()),
                    response_tokens=len(ex.response.split()),
                )
                save_chat_entry(entry, validate=True)
                add_chat_to_session(entry.id)
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving exchange: {e}")

    return {
        'status': 'success',
        'exchanges': len(exchanges),
        'saved': saved_count,
        'session_id': session_id,
        'tools_used': list(total_tools),
        'files_referenced': list(all_files_ref),
        'files_modified': list(all_files_mod),
    }
