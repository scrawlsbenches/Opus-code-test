#!/usr/bin/env python3
"""
Training Data Exporter - Export GoT data for AI training

Exports Graph of Thought (GoT) data in formats suitable for machine learning training.
Focuses on high-value knowledge like decision rationales, task retrospectives,
handoff instructions, and knowledge transfers.

Usage:
    python scripts/training_data_exporter.py export --output ./training_data/
    python scripts/training_data_exporter.py stats
    python scripts/training_data_exporter.py export-decisions
    python scripts/training_data_exporter.py export-retrospectives

Output Formats:
    - JSONL (one JSON object per line) for ML pipelines
    - Markdown for human review

Quality Filtering:
    - Skips tasks without retrospectives
    - Skips decisions without rationales
    - Includes quality score based on completeness
"""

import glob
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Iterator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
REPO_ROOT = Path(__file__).parent.parent
GOT_DIR = REPO_ROOT / ".got" / "entities"

# Security and memory limits
MAX_ENTITY_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_OUTPUT_DIRS = [
    os.getcwd(),
    '/tmp',
    str(REPO_ROOT / 'training_data'),
    str(REPO_ROOT / 'exports')
]

# Sensitive field patterns to redact
SENSITIVE_PATTERNS = [
    'password', 'secret', 'token', 'key', 'credential',
    'api_key', 'auth', 'private'
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DecisionTrainingExample:
    """Decision rationale for training."""
    decision_id: str
    title: str
    rationale: str
    affects: List[str]
    created_at: str
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": f"Decision: {self.title}",
            "decision": self.title,
            "rationale": self.rationale,
            "affects": self.affects,
            "outcome": "documented",
            "quality_score": self.quality_score,
            "metadata": {
                "decision_id": self.decision_id,
                "created_at": self.created_at
            }
        }


@dataclass
class RetrospectiveTrainingExample:
    """Task retrospective for training."""
    task_id: str
    title: str
    description: str
    retrospective: str
    category: str
    priority: str
    success: bool
    created_at: str
    completed_at: Optional[str]
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.title,
            "description": self.description,
            "approach": self.category,
            "retrospective": self.retrospective,
            "success": self.success,
            "priority": self.priority,
            "quality_score": self.quality_score,
            "metadata": {
                "task_id": self.task_id,
                "created_at": self.created_at,
                "completed_at": self.completed_at
            }
        }


@dataclass
class HandoffTrainingExample:
    """Handoff instructions for training."""
    handoff_id: str
    task_id: str
    task_title: str
    instructions: str
    target_agent: str
    result: Dict[str, Any]
    success: bool
    created_at: str
    completed_at: Optional[str]
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task_title,
            "handoff_to": self.target_agent,
            "instructions": self.instructions,
            "result": self.result,
            "success": self.success,
            "quality_score": self.quality_score,
            "metadata": {
                "handoff_id": self.handoff_id,
                "task_id": self.task_id,
                "created_at": self.created_at,
                "completed_at": self.completed_at
            }
        }


@dataclass
class KnowledgeTransferTrainingExample:
    """Knowledge transfer for training."""
    kt_id: str
    title: str
    summary: str
    sections: Dict[str, Any]
    related_tasks: List[str]
    related_decisions: List[str]
    status: str
    created_at: str
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        # Extract key points from sections
        key_points = []
        if isinstance(self.sections, dict):
            for section_name, section_content in self.sections.items():
                if section_content:
                    key_points.append(f"{section_name}: {section_content}")

        return {
            "topic": self.title,
            "summary": self.summary,
            "key_points": key_points if key_points else [self.summary],
            "related_tasks": self.related_tasks,
            "related_decisions": self.related_decisions,
            "status": self.status,
            "quality_score": self.quality_score,
            "metadata": {
                "kt_id": self.kt_id,
                "created_at": self.created_at
            }
        }


@dataclass
class EdgeTrainingExample:
    """Edge relationship for graph knowledge."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    source_type: str
    target_type: str
    source_title: str
    target_title: str
    weight: float
    confidence: float
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": f"{self.source_type}:{self.source_title}",
            "relationship": self.edge_type,
            "to": f"{self.target_type}:{self.target_title}",
            "context": f"{self.source_id} {self.edge_type} {self.target_id}",
            "weight": self.weight,
            "confidence": self.confidence,
            "metadata": {
                "edge_id": self.edge_id,
                "source_id": self.source_id,
                "target_id": self.target_id,
                "created_at": self.created_at
            }
        }


@dataclass
class TrainingStats:
    """Statistics about training data."""
    total_decisions: int = 0
    quality_decisions: int = 0
    total_tasks: int = 0
    tasks_with_retrospectives: int = 0
    total_handoffs: int = 0
    successful_handoffs: int = 0
    total_kts: int = 0
    published_kts: int = 0
    total_edges: int = 0
    edge_types: Dict[str, int] = field(default_factory=dict)
    failed_entities: List[str] = field(default_factory=list)
    skipped_oversized: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decisions": {
                "total": self.total_decisions,
                "with_quality_rationale": self.quality_decisions,
                "quality_rate": self.quality_decisions / self.total_decisions if self.total_decisions > 0 else 0.0
            },
            "tasks": {
                "total": self.total_tasks,
                "with_retrospectives": self.tasks_with_retrospectives,
                "retrospective_rate": self.tasks_with_retrospectives / self.total_tasks if self.total_tasks > 0 else 0.0
            },
            "handoffs": {
                "total": self.total_handoffs,
                "successful": self.successful_handoffs,
                "success_rate": self.successful_handoffs / self.total_handoffs if self.total_handoffs > 0 else 0.0
            },
            "knowledge_transfers": {
                "total": self.total_kts,
                "published": self.published_kts,
                "publish_rate": self.published_kts / self.total_kts if self.total_kts > 0 else 0.0
            },
            "edges": {
                "total": self.total_edges,
                "by_type": self.edge_types
            },
            "errors": {
                "failed_entities": len(self.failed_entities),
                "skipped_oversized": self.skipped_oversized
            }
        }


# =============================================================================
# TRAINING DATA EXPORTER
# =============================================================================

class TrainingDataExporter:
    """
    Export GoT data in formats suitable for AI training.

    Exports decision rationales, task retrospectives, handoff instructions,
    knowledge transfers, and edge relationships with quality filtering.

    Uses streaming/generator-based iteration to avoid loading all entities
    into memory at once.
    """

    def __init__(self, got_dir: Path = GOT_DIR, sanitize_sensitive: bool = False):
        """
        Initialize exporter with GoT directory.

        Args:
            got_dir: Path to GoT entities directory
            sanitize_sensitive: Whether to redact sensitive data
        """
        self.got_dir = Path(got_dir)
        if not self.got_dir.exists():
            raise ValueError(f"GoT directory does not exist: {self.got_dir}")

        self.sanitize_sensitive = sanitize_sensitive
        self.failed_entities: List[str] = []
        self.skipped_oversized: int = 0

        # Lightweight cache for entity lookups (ID -> title mapping only)
        self._title_cache: Dict[str, str] = {}
        self._type_cache: Dict[str, str] = {}

    def _validate_output_path(self, output_path: str) -> Path:
        """
        Prevent directory traversal attacks.

        Args:
            output_path: User-provided output path

        Returns:
            Validated absolute path

        Raises:
            ValueError: If path is not in allowed directories
        """
        resolved = Path(output_path).resolve()

        # Check if path is within allowed base directories
        allowed = False
        for base in ALLOWED_OUTPUT_DIRS:
            base_path = Path(base).resolve()
            try:
                resolved.relative_to(base_path)
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise ValueError(
                f"Output path not allowed: {resolved}\n"
                f"Allowed base directories: {ALLOWED_OUTPUT_DIRS}"
            )

        return resolved

    def _load_single_entity(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """
        Load a single entity with size checking.

        Args:
            filepath: Path to entity JSON file

        Returns:
            Entity data dict or None if failed/oversized
        """
        try:
            # Check file size before loading
            file_size = filepath.stat().st_size
            if file_size > MAX_ENTITY_SIZE:
                logger.warning(
                    f"Skipping oversized entity {filepath.name}: "
                    f"{file_size / 1024 / 1024:.2f}MB > {MAX_ENTITY_SIZE / 1024 / 1024}MB"
                )
                self.skipped_oversized += 1
                return None

            with open(filepath, 'r', encoding='utf-8') as f:
                entity = json.load(f)

            # Extract entity data
            if isinstance(entity, dict) and 'data' in entity:
                entity_data = entity['data']

                # Sanitize if requested
                if self.sanitize_sensitive:
                    entity_data = self._sanitize_data(entity_data)

                return entity_data

            return None

        except (json.JSONDecodeError, KeyError, IOError, OSError) as e:
            logger.warning(f"Failed to load {filepath.name}: {e}")
            self.failed_entities.append(filepath.name)
            return None

    def _iter_entities(self, entity_type: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """
        Yield entities one at a time without loading all into memory.

        Args:
            entity_type: Filter by entity type (task, decision, etc.)

        Yields:
            Entity data dictionaries
        """
        for filepath in self.got_dir.glob("*.json"):
            entity_data = self._load_single_entity(filepath)

            if entity_data is None:
                continue

            # Filter by type if specified
            if entity_type and entity_data.get('entity_type') != entity_type:
                continue

            # Cache title and type for lookups
            entity_id = entity_data.get('id', filepath.stem)
            self._title_cache[entity_id] = entity_data.get('title', entity_id)
            self._type_cache[entity_id] = entity_data.get('entity_type', 'unknown')

            yield entity_data

    def _sanitize_data(self, data: Any) -> Any:
        """
        Recursively sanitize sensitive data from dictionaries.

        Args:
            data: Data to sanitize (dict, list, or primitive)

        Returns:
            Sanitized copy of data
        """
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Check if key contains sensitive patterns
                key_lower = key.lower()
                is_sensitive = any(pattern in key_lower for pattern in SENSITIVE_PATTERNS)

                if is_sensitive:
                    sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized

        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]

        elif isinstance(data, str):
            # Redact file paths containing home directories or sensitive locations
            if '/home/' in data or '/Users/' in data or 'C:\\Users\\' in data:
                # Replace with generic placeholder
                return data.replace(os.path.expanduser('~'), '~')
            return data

        else:
            return data

    def _get_entity_title(self, entity_id: str) -> str:
        """Get title for an entity by ID from cache."""
        return self._title_cache.get(entity_id, entity_id)

    def _get_entity_type(self, entity_id: str) -> str:
        """Get entity type by ID from cache."""
        return self._type_cache.get(entity_id, 'unknown')

    def _calculate_quality_score(self, text: str, min_length: int = 50) -> float:
        """
        Calculate quality score based on text completeness.

        Score factors:
        - Length (0-1): longer is better
        - Structure (0-1): has sentences, paragraphs
        - Information density (0-1): non-trivial content
        """
        if not text or not text.strip():
            return 0.0

        text = text.strip()
        length = len(text)

        # Length score
        length_score = min(1.0, length / (min_length * 3))

        # Structure score (has multiple sentences)
        sentences = text.count('.') + text.count('!') + text.count('?')
        structure_score = min(1.0, sentences / 3)

        # Information density (not just repeated words)
        words = text.lower().split()
        unique_words = len(set(words))
        total_words = len(words)
        density_score = unique_words / total_words if total_words > 0 else 0.0

        # Weighted average
        quality = (length_score * 0.4 + structure_score * 0.3 + density_score * 0.3)

        return round(quality, 2)

    def export_decisions(self, output_path: Path, limit: Optional[int] = None, dry_run: bool = False) -> List[DecisionTrainingExample]:
        """
        Export decision rationales.

        Args:
            output_path: Path to output JSONL file
            limit: Maximum number to export (for testing)
            dry_run: If True, don't write to file

        Returns:
            List of decision training examples
        """
        # Validate output path
        if not dry_run:
            output_path = self._validate_output_path(str(output_path))

        decisions = []
        processed = 0

        for entity in self._iter_entities(entity_type='decision'):
            try:
                entity_id = entity.get('id', 'unknown')
                title = entity.get('title', '')
                rationale = entity.get('rationale', '')

                # Skip if no rationale
                if not rationale or not rationale.strip():
                    continue

                quality_score = self._calculate_quality_score(rationale, min_length=30)

                # Skip low quality
                if quality_score < 0.2:
                    continue

                example = DecisionTrainingExample(
                    decision_id=entity_id,
                    title=title,
                    rationale=rationale,
                    affects=entity.get('affects', []),
                    created_at=entity.get('created_at', ''),
                    quality_score=quality_score
                )
                decisions.append(example)

                processed += 1
                if limit and processed >= limit:
                    logger.info(f"Reached limit of {limit} decisions")
                    break

            except Exception as e:
                logger.error(f"Error processing decision entity: {e}", exc_info=True)
                continue

        # Write JSONL
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for decision in decisions:
                    try:
                        f.write(json.dumps(decision.to_dict()) + '\n')
                    except Exception as e:
                        logger.error(f"Error writing decision {decision.decision_id}: {e}")

        logger.info(f"Exported {len(decisions)} decision rationales" +
                   (f" to {output_path}" if not dry_run else " (dry run)"))
        return decisions

    def export_retrospectives(self, output_path: Path, limit: Optional[int] = None, dry_run: bool = False) -> List[RetrospectiveTrainingExample]:
        """
        Export task retrospectives.

        Args:
            output_path: Path to output JSONL file
            limit: Maximum number to export (for testing)
            dry_run: If True, don't write to file

        Returns:
            List of retrospective training examples
        """
        # Validate output path
        if not dry_run:
            output_path = self._validate_output_path(str(output_path))

        retrospectives = []
        processed = 0

        for entity in self._iter_entities(entity_type='task'):
            try:
                entity_id = entity.get('id', 'unknown')

                # Get retrospective from properties
                properties = entity.get('properties', {})
                retrospective = properties.get('retrospective', '')

                # Skip if no retrospective
                if not retrospective or not retrospective.strip():
                    continue

                title = entity.get('title', '')
                description = entity.get('description', '')
                category = properties.get('category', 'unknown')
                priority = entity.get('priority', 'medium')
                status = entity.get('status', 'unknown')

                quality_score = self._calculate_quality_score(retrospective, min_length=50)

                # Skip low quality
                if quality_score < 0.2:
                    continue

                # Determine success from status and retrospective content
                success = status == 'completed'

                metadata = entity.get('metadata', {})
                example = RetrospectiveTrainingExample(
                    task_id=entity_id,
                    title=title,
                    description=description,
                    retrospective=retrospective,
                    category=category,
                    priority=priority,
                    success=success,
                    created_at=entity.get('created_at', ''),
                    completed_at=metadata.get('completed_at'),
                    quality_score=quality_score
                )
                retrospectives.append(example)

                processed += 1
                if limit and processed >= limit:
                    logger.info(f"Reached limit of {limit} retrospectives")
                    break

            except Exception as e:
                logger.error(f"Error processing task entity: {e}", exc_info=True)
                continue

        # Write JSONL
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for retro in retrospectives:
                    try:
                        f.write(json.dumps(retro.to_dict()) + '\n')
                    except Exception as e:
                        logger.error(f"Error writing retrospective {retro.task_id}: {e}")

        logger.info(f"Exported {len(retrospectives)} task retrospectives" +
                   (f" to {output_path}" if not dry_run else " (dry run)"))
        return retrospectives

    def export_handoffs(self, output_path: Path, limit: Optional[int] = None, dry_run: bool = False) -> List[HandoffTrainingExample]:
        """
        Export handoff instructions.

        Args:
            output_path: Path to output JSONL file
            limit: Maximum number to export (for testing)
            dry_run: If True, don't write to file

        Returns:
            List of handoff training examples
        """
        # Validate output path
        if not dry_run:
            output_path = self._validate_output_path(str(output_path))

        handoffs = []
        processed = 0

        for entity in self._iter_entities(entity_type='handoff'):
            try:
                entity_id = entity.get('id', 'unknown')
                instructions = entity.get('instructions', '')

                # Skip if no instructions
                if not instructions or not instructions.strip():
                    continue

                task_id = entity.get('task_id', '')
                context = entity.get('context', {})
                task_title = context.get('task_title', self._get_entity_title(task_id))

                quality_score = self._calculate_quality_score(instructions, min_length=100)

                # Skip low quality
                if quality_score < 0.2:
                    continue

                # Determine success from status
                status = entity.get('status', '')
                success = status == 'completed'

                example = HandoffTrainingExample(
                    handoff_id=entity_id,
                    task_id=task_id,
                    task_title=task_title,
                    instructions=instructions,
                    target_agent=entity.get('target_agent', 'unknown'),
                    result=entity.get('result', {}),
                    success=success,
                    created_at=entity.get('created_at', ''),
                    completed_at=entity.get('completed_at'),
                    quality_score=quality_score
                )
                handoffs.append(example)

                processed += 1
                if limit and processed >= limit:
                    logger.info(f"Reached limit of {limit} handoffs")
                    break

            except Exception as e:
                logger.error(f"Error processing handoff entity: {e}", exc_info=True)
                continue

        # Write JSONL
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for handoff in handoffs:
                    try:
                        f.write(json.dumps(handoff.to_dict()) + '\n')
                    except Exception as e:
                        logger.error(f"Error writing handoff {handoff.handoff_id}: {e}")

        logger.info(f"Exported {len(handoffs)} handoff instructions" +
                   (f" to {output_path}" if not dry_run else " (dry run)"))
        return handoffs

    def export_knowledge_transfers(self, output_path: Path, limit: Optional[int] = None, dry_run: bool = False) -> List[KnowledgeTransferTrainingExample]:
        """
        Export knowledge transfers.

        Args:
            output_path: Path to output JSONL file
            limit: Maximum number to export (for testing)
            dry_run: If True, don't write to file

        Returns:
            List of knowledge transfer training examples
        """
        # Validate output path
        if not dry_run:
            output_path = self._validate_output_path(str(output_path))

        kts = []
        processed = 0

        for entity in self._iter_entities(entity_type='knowledge_transfer'):
            try:
                entity_id = entity.get('id', 'unknown')
                title = entity.get('title', '')
                summary = entity.get('summary', '')

                # Skip if no summary
                if not summary or not summary.strip():
                    continue

                quality_score = self._calculate_quality_score(summary, min_length=100)

                # Skip low quality
                if quality_score < 0.2:
                    continue

                example = KnowledgeTransferTrainingExample(
                    kt_id=entity_id,
                    title=title,
                    summary=summary,
                    sections=entity.get('sections', {}),
                    related_tasks=entity.get('related_tasks', []),
                    related_decisions=entity.get('related_decisions', []),
                    status=entity.get('status', 'draft'),
                    created_at=entity.get('created_at', ''),
                    quality_score=quality_score
                )
                kts.append(example)

                processed += 1
                if limit and processed >= limit:
                    logger.info(f"Reached limit of {limit} knowledge transfers")
                    break

            except Exception as e:
                logger.error(f"Error processing knowledge transfer entity: {e}", exc_info=True)
                continue

        # Write JSONL
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for kt in kts:
                    try:
                        f.write(json.dumps(kt.to_dict()) + '\n')
                    except Exception as e:
                        logger.error(f"Error writing knowledge transfer {kt.kt_id}: {e}")

        logger.info(f"Exported {len(kts)} knowledge transfers" +
                   (f" to {output_path}" if not dry_run else " (dry run)"))
        return kts

    def export_edges(self, output_path: Path, limit: Optional[int] = None, dry_run: bool = False) -> List[EdgeTrainingExample]:
        """
        Export edge relationships.

        Args:
            output_path: Path to output JSONL file
            limit: Maximum number to export (for testing)
            dry_run: If True, don't write to file

        Returns:
            List of edge training examples
        """
        # Validate output path
        if not dry_run:
            output_path = self._validate_output_path(str(output_path))

        edges = []
        processed = 0

        for entity in self._iter_entities(entity_type='edge'):
            try:
                entity_id = entity.get('id', 'unknown')
                source_id = entity.get('source_id', '')
                target_id = entity.get('target_id', '')
                edge_type = entity.get('edge_type', '')

                # Get titles for source and target
                source_title = self._get_entity_title(source_id)
                target_title = self._get_entity_title(target_id)
                source_type = self._get_entity_type(source_id)
                target_type = self._get_entity_type(target_id)

                example = EdgeTrainingExample(
                    edge_id=entity_id,
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    source_type=source_type,
                    target_type=target_type,
                    source_title=source_title,
                    target_title=target_title,
                    weight=entity.get('weight', 1.0),
                    confidence=entity.get('confidence', 1.0),
                    created_at=entity.get('created_at', '')
                )
                edges.append(example)

                processed += 1
                if limit and processed >= limit:
                    logger.info(f"Reached limit of {limit} edges")
                    break

            except Exception as e:
                logger.error(f"Error processing edge entity: {e}", exc_info=True)
                continue

        # Write JSONL
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for edge in edges:
                    try:
                        f.write(json.dumps(edge.to_dict()) + '\n')
                    except Exception as e:
                        logger.error(f"Error writing edge {edge.edge_id}: {e}")

        logger.info(f"Exported {len(edges)} edge relationships" +
                   (f" to {output_path}" if not dry_run else " (dry run)"))
        return edges

    def export_all(self, output_dir: Path, limit: Optional[int] = None, dry_run: bool = False) -> Dict[str, int]:
        """
        Export all training data to a directory.

        Creates separate JSONL files for each data type plus markdown summaries.

        Args:
            output_dir: Directory to write files to
            limit: Maximum number of each type to export (for testing)
            dry_run: If True, don't write to files

        Returns:
            Dict with counts of exported items
        """
        # Validate output path
        output_dir = Path(output_dir)
        if not dry_run:
            output_dir = self._validate_output_path(str(output_dir))
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting all training data to {output_dir}" +
                   (" (dry run)" if dry_run else ""))

        # Export each type
        decisions = self.export_decisions(output_dir / "decisions.jsonl", limit=limit, dry_run=dry_run)
        retrospectives = self.export_retrospectives(output_dir / "retrospectives.jsonl", limit=limit, dry_run=dry_run)
        handoffs = self.export_handoffs(output_dir / "handoffs.jsonl", limit=limit, dry_run=dry_run)
        kts = self.export_knowledge_transfers(output_dir / "knowledge_transfers.jsonl", limit=limit, dry_run=dry_run)
        edges = self.export_edges(output_dir / "edges.jsonl", limit=limit, dry_run=dry_run)

        # Generate markdown summary
        if not dry_run:
            self._write_markdown_summary(
                output_dir / "README.md",
                decisions, retrospectives, handoffs, kts, edges
            )

        counts = {
            'decisions': len(decisions),
            'retrospectives': len(retrospectives),
            'handoffs': len(handoffs),
            'knowledge_transfers': len(kts),
            'edges': len(edges),
            'failed_entities': len(self.failed_entities),
            'skipped_oversized': self.skipped_oversized
        }

        logger.info(f"Export complete: {sum(counts.values())} total items")
        if self.failed_entities:
            logger.warning(f"Failed to process {len(self.failed_entities)} entities: {self.failed_entities[:10]}")
        if self.skipped_oversized:
            logger.warning(f"Skipped {self.skipped_oversized} oversized entities")

        return counts

    def _write_markdown_summary(
        self,
        output_path: Path,
        decisions: List[DecisionTrainingExample],
        retrospectives: List[RetrospectiveTrainingExample],
        handoffs: List[HandoffTrainingExample],
        kts: List[KnowledgeTransferTrainingExample],
        edges: List[EdgeTrainingExample]
    ):
        """Write a markdown summary of exported data."""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Training Data Export\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            # Summary statistics
            f.write("## Summary\n\n")
            f.write(f"- **Decisions**: {len(decisions)} rationales\n")
            f.write(f"- **Retrospectives**: {len(retrospectives)} task learnings\n")
            f.write(f"- **Handoffs**: {len(handoffs)} delegation examples\n")
            f.write(f"- **Knowledge Transfers**: {len(kts)} synthesis documents\n")
            f.write(f"- **Edges**: {len(edges)} relationships\n\n")

            # Quality distribution
            f.write("## Quality Distribution\n\n")

            if decisions:
                avg_quality = sum(d.quality_score for d in decisions) / len(decisions)
                f.write(f"### Decisions (avg quality: {avg_quality:.2f})\n\n")
                for d in sorted(decisions, key=lambda x: -x.quality_score)[:5]:
                    f.write(f"- **{d.title}** (Q={d.quality_score:.2f})\n")
                    f.write(f"  - {d.rationale[:100]}...\n\n")

            if retrospectives:
                avg_quality = sum(r.quality_score for r in retrospectives) / len(retrospectives)
                f.write(f"\n### Retrospectives (avg quality: {avg_quality:.2f})\n\n")
                for r in sorted(retrospectives, key=lambda x: -x.quality_score)[:5]:
                    f.write(f"- **{r.title}** (Q={r.quality_score:.2f})\n")
                    f.write(f"  - {r.retrospective[:100]}...\n\n")

            # Edge types
            if edges:
                edge_types = defaultdict(int)
                for e in edges:
                    edge_types[e.edge_type] += 1

                f.write("\n### Edge Types\n\n")
                for edge_type, count in sorted(edge_types.items(), key=lambda x: -x[1]):
                    f.write(f"- **{edge_type}**: {count}\n")

            f.write("\n## Files\n\n")
            f.write("- `decisions.jsonl` - Decision rationales\n")
            f.write("- `retrospectives.jsonl` - Task retrospectives\n")
            f.write("- `handoffs.jsonl` - Handoff instructions\n")
            f.write("- `knowledge_transfers.jsonl` - Knowledge transfers\n")
            f.write("- `edges.jsonl` - Graph relationships\n\n")

            f.write("## Format\n\n")
            f.write("All files are in JSONL format (one JSON object per line).\n")
            f.write("Each line is a complete training example with metadata.\n\n")

        logger.info(f"Wrote markdown summary to {output_path}")

    def get_training_stats(self) -> TrainingStats:
        """
        Get statistics about available training data.

        Returns:
            TrainingStats with counts and quality metrics
        """
        stats = TrainingStats()

        for entity in self._iter_entities():
            try:
                entity_type = entity.get('entity_type')

                if entity_type == 'decision':
                    stats.total_decisions += 1
                    rationale = entity.get('rationale', '')
                    if rationale and self._calculate_quality_score(rationale, 30) >= 0.2:
                        stats.quality_decisions += 1

                elif entity_type == 'task':
                    stats.total_tasks += 1
                    properties = entity.get('properties', {})
                    retrospective = properties.get('retrospective', '')
                    if retrospective and self._calculate_quality_score(retrospective, 50) >= 0.2:
                        stats.tasks_with_retrospectives += 1

                elif entity_type == 'handoff':
                    stats.total_handoffs += 1
                    if entity.get('status') == 'completed':
                        stats.successful_handoffs += 1

                elif entity_type == 'knowledge_transfer':
                    stats.total_kts += 1
                    if entity.get('status') == 'published':
                        stats.published_kts += 1

                elif entity_type == 'edge':
                    stats.total_edges += 1
                    edge_type = entity.get('edge_type', 'unknown')
                    stats.edge_types[edge_type] = stats.edge_types.get(edge_type, 0) + 1

            except Exception as e:
                logger.error(f"Error processing entity for stats: {e}")
                continue

        # Add error tracking
        stats.failed_entities = self.failed_entities.copy()
        stats.skipped_oversized = self.skipped_oversized

        return stats


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Export GoT data for AI training'
    )

    # Global options
    parser.add_argument('--sanitize', action='store_true',
                       help='Redact sensitive data (paths, usernames, etc.)')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show training data statistics')
    stats_parser.add_argument('--json', action='store_true', help='Output as JSON')

    # Export all command
    export_parser = subparsers.add_parser('export', help='Export all training data')
    export_parser.add_argument('--output', '-o', type=str, required=True,
                              help='Output directory')
    export_parser.add_argument('--limit', type=int,
                              help='Limit number of items per type (for testing)')
    export_parser.add_argument('--dry-run', action='store_true',
                              help='Count items without writing files')

    # Export individual types
    export_decisions_parser = subparsers.add_parser('export-decisions',
                                                    help='Export decision rationales')
    export_decisions_parser.add_argument('--output', '-o', type=str,
                                        default='./training_data/decisions.jsonl',
                                        help='Output file')
    export_decisions_parser.add_argument('--limit', type=int,
                                        help='Limit number of items (for testing)')
    export_decisions_parser.add_argument('--dry-run', action='store_true',
                                        help='Count items without writing files')

    export_retro_parser = subparsers.add_parser('export-retrospectives',
                                                help='Export task retrospectives')
    export_retro_parser.add_argument('--output', '-o', type=str,
                                     default='./training_data/retrospectives.jsonl',
                                     help='Output file')
    export_retro_parser.add_argument('--limit', type=int,
                                    help='Limit number of items (for testing)')
    export_retro_parser.add_argument('--dry-run', action='store_true',
                                    help='Count items without writing files')

    export_handoffs_parser = subparsers.add_parser('export-handoffs',
                                                   help='Export handoff instructions')
    export_handoffs_parser.add_argument('--output', '-o', type=str,
                                        default='./training_data/handoffs.jsonl',
                                        help='Output file')
    export_handoffs_parser.add_argument('--limit', type=int,
                                       help='Limit number of items (for testing)')
    export_handoffs_parser.add_argument('--dry-run', action='store_true',
                                       help='Count items without writing files')

    export_kts_parser = subparsers.add_parser('export-kts',
                                              help='Export knowledge transfers')
    export_kts_parser.add_argument('--output', '-o', type=str,
                                   default='./training_data/knowledge_transfers.jsonl',
                                   help='Output file')
    export_kts_parser.add_argument('--limit', type=int,
                                  help='Limit number of items (for testing)')
    export_kts_parser.add_argument('--dry-run', action='store_true',
                                  help='Count items without writing files')

    export_edges_parser = subparsers.add_parser('export-edges',
                                                help='Export edge relationships')
    export_edges_parser.add_argument('--output', '-o', type=str,
                                     default='./training_data/edges.jsonl',
                                     help='Output file')
    export_edges_parser.add_argument('--limit', type=int,
                                    help='Limit number of items (for testing)')
    export_edges_parser.add_argument('--dry-run', action='store_true',
                                    help='Count items without writing files')

    args = parser.parse_args()

    try:
        exporter = TrainingDataExporter(sanitize_sensitive=args.sanitize)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if args.command == 'stats':
        stats = exporter.get_training_stats()

        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print("\n" + "=" * 70)
            print("TRAINING DATA STATISTICS")
            print("=" * 70)

            stats_dict = stats.to_dict()

            print("\nDecisions:")
            print(f"  Total:                {stats_dict['decisions']['total']}")
            print(f"  Quality rationales:   {stats_dict['decisions']['with_quality_rationale']}")
            print(f"  Quality rate:         {stats_dict['decisions']['quality_rate']:.1%}")

            print("\nTasks:")
            print(f"  Total:                {stats_dict['tasks']['total']}")
            print(f"  With retrospectives:  {stats_dict['tasks']['with_retrospectives']}")
            print(f"  Retrospective rate:   {stats_dict['tasks']['retrospective_rate']:.1%}")

            print("\nHandoffs:")
            print(f"  Total:                {stats_dict['handoffs']['total']}")
            print(f"  Successful:           {stats_dict['handoffs']['successful']}")
            print(f"  Success rate:         {stats_dict['handoffs']['success_rate']:.1%}")

            print("\nKnowledge Transfers:")
            print(f"  Total:                {stats_dict['knowledge_transfers']['total']}")
            print(f"  Published:            {stats_dict['knowledge_transfers']['published']}")
            print(f"  Publish rate:         {stats_dict['knowledge_transfers']['publish_rate']:.1%}")

            print("\nEdges:")
            print(f"  Total:                {stats_dict['edges']['total']}")
            if stats_dict['edges']['by_type']:
                print("\n  By type:")
                for edge_type, count in sorted(stats_dict['edges']['by_type'].items(),
                                              key=lambda x: -x[1])[:10]:
                    print(f"    {edge_type:<20} {count:>5}")

            # Error tracking
            if stats_dict.get('errors'):
                print("\nErrors:")
                print(f"  Failed entities:      {stats_dict['errors']['failed_entities']}")
                print(f"  Skipped oversized:    {stats_dict['errors']['skipped_oversized']}")

            print("\n" + "=" * 70)

            # Estimate training value
            total_examples = (stats_dict['decisions']['with_quality_rationale'] +
                            stats_dict['tasks']['with_retrospectives'] +
                            stats_dict['handoffs']['successful'] +
                            stats_dict['knowledge_transfers']['published'])

            print(f"\nEstimated training examples: {total_examples}")
            print(f"Edge relationships:          {stats_dict['edges']['total']}")
            print(f"Total training data points:  {total_examples + stats_dict['edges']['total']}")

    elif args.command == 'export':
        output_dir = Path(args.output)
        limit = getattr(args, 'limit', None)
        dry_run = getattr(args, 'dry_run', False)

        counts = exporter.export_all(output_dir, limit=limit, dry_run=dry_run)

        print(f"\n{'[DRY RUN] ' if dry_run else ''}Exported training data" +
              (f" to {output_dir}" if not dry_run else ""))
        for data_type, count in counts.items():
            print(f"  {data_type}: {count}")

    elif args.command == 'export-decisions':
        limit = getattr(args, 'limit', None)
        dry_run = getattr(args, 'dry_run', False)
        decisions = exporter.export_decisions(Path(args.output), limit=limit, dry_run=dry_run)
        print(f"{'[DRY RUN] ' if dry_run else ''}Exported {len(decisions)} decision rationales" +
              (f" to {args.output}" if not dry_run else ""))

    elif args.command == 'export-retrospectives':
        limit = getattr(args, 'limit', None)
        dry_run = getattr(args, 'dry_run', False)
        retros = exporter.export_retrospectives(Path(args.output), limit=limit, dry_run=dry_run)
        print(f"{'[DRY RUN] ' if dry_run else ''}Exported {len(retros)} task retrospectives" +
              (f" to {args.output}" if not dry_run else ""))

    elif args.command == 'export-handoffs':
        limit = getattr(args, 'limit', None)
        dry_run = getattr(args, 'dry_run', False)
        handoffs = exporter.export_handoffs(Path(args.output), limit=limit, dry_run=dry_run)
        print(f"{'[DRY RUN] ' if dry_run else ''}Exported {len(handoffs)} handoff instructions" +
              (f" to {args.output}" if not dry_run else ""))

    elif args.command == 'export-kts':
        limit = getattr(args, 'limit', None)
        dry_run = getattr(args, 'dry_run', False)
        kts = exporter.export_knowledge_transfers(Path(args.output), limit=limit, dry_run=dry_run)
        print(f"{'[DRY RUN] ' if dry_run else ''}Exported {len(kts)} knowledge transfers" +
              (f" to {args.output}" if not dry_run else ""))

    elif args.command == 'export-edges':
        limit = getattr(args, 'limit', None)
        dry_run = getattr(args, 'dry_run', False)
        edges = exporter.export_edges(Path(args.output), limit=limit, dry_run=dry_run)
        print(f"{'[DRY RUN] ' if dry_run else ''}Exported {len(edges)} edge relationships" +
              (f" to {args.output}" if not dry_run else ""))

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
