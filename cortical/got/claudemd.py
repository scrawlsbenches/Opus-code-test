"""
CLAUDE.md Auto-Generation Engine.

Provides context-aware CLAUDE.md generation from GoT-stored layers with:
- Layer selection based on current work context
- Fault-tolerant generation with fallback
- Content validation
- Freshness management

Example:
    >>> from cortical.got.claudemd import ClaudeMdManager
    >>> manager = ClaudeMdManager(got_manager)
    >>> result = manager.generate()
    >>> print(result.success, result.path)
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class GenerationContext:
    """Context information for generation decisions."""
    current_branch: str = ""
    active_sprint_id: str = ""
    touched_files: List[str] = field(default_factory=list)
    detected_modules: List[str] = field(default_factory=list)
    in_progress_tasks: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result of CLAUDE.md generation."""
    success: bool
    path: Path = None
    layers_used: int = 0
    content_hash: str = ""
    error: str = ""
    fallback_used: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of content validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ContextAnalyzer:
    """Analyzes current work context for layer selection."""

    # Module detection patterns
    MODULE_PATTERNS = {
        "query": ["cortical/query/", "query.py", "search", "retrieval"],
        "reasoning": ["cortical/reasoning/", "thought", "cognitive"],
        "spark": ["cortical/spark/", "ngram", "predictor", "anomaly"],
        "got": ["cortical/got/", ".got/", "task", "sprint", "decision"],
        "analysis": ["cortical/analysis.py", "pagerank", "tfidf", "clustering"],
        "persistence": ["cortical/persistence.py", "save", "load", "chunk"],
    }

    def __init__(self, got_dir: Path = None):
        self.got_dir = got_dir or Path(".got")

    def analyze(self) -> GenerationContext:
        """Gather context for generation decisions."""
        return GenerationContext(
            current_branch=self._get_current_branch(),
            active_sprint_id=self._get_active_sprint_id(),
            touched_files=self._get_recently_touched_files(),
            detected_modules=self._detect_modules_from_files(),
            in_progress_tasks=self._get_in_progress_tasks(),
            user_preferences=self._load_user_preferences(),
        )

    def _get_current_branch(self) -> str:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    def _get_active_sprint_id(self) -> str:
        """Get active sprint ID from GoT."""
        try:
            entities_dir = self.got_dir / "entities"
            if not entities_dir.exists():
                return ""

            import json
            for sprint_file in entities_dir.glob("S-*.json"):
                with open(sprint_file) as f:
                    data = json.load(f)
                entity_data = data.get("data", data)
                if entity_data.get("status") == "in_progress":
                    return entity_data.get("id", "")
            return ""
        except Exception:
            return ""

    def _get_recently_touched_files(self) -> List[str]:
        """Get files modified recently (from git status)."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return []

            files = []
            for line in result.stdout.strip().split("\n"):
                if line and len(line) > 3:
                    files.append(line[3:].strip())
            return files
        except Exception:
            return []

    def _detect_modules_from_files(self) -> List[str]:
        """Detect which modules are relevant based on touched files."""
        touched = self._get_recently_touched_files()
        detected = set()

        for file_path in touched:
            for module, patterns in self.MODULE_PATTERNS.items():
                if any(pattern in file_path for pattern in patterns):
                    detected.add(module)

        return list(detected)

    def _get_in_progress_tasks(self) -> List[str]:
        """Get IDs of in-progress tasks."""
        try:
            entities_dir = self.got_dir / "entities"
            if not entities_dir.exists():
                return []

            import json
            tasks = []
            for task_file in entities_dir.glob("T-*.json"):
                with open(task_file) as f:
                    data = json.load(f)
                entity_data = data.get("data", data)
                if entity_data.get("status") == "in_progress":
                    tasks.append(entity_data.get("id", ""))
            return tasks
        except Exception:
            return []

    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences for layer inclusion."""
        try:
            prefs_file = self.got_dir / "claude-md" / "preferences.json"
            if prefs_file.exists():
                import json
                with open(prefs_file) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}


class LayerSelector:
    """Selects which layers to include based on context."""

    def select(
        self,
        context: GenerationContext,
        available_layers: List
    ) -> List:
        """
        Select layers based on inclusion rules and context.

        Args:
            context: Current generation context
            available_layers: All available ClaudeMdLayer entities

        Returns:
            List of selected layers, ordered by layer_number then section_id
        """
        selected = []

        for layer in available_layers:
            if self._should_include(layer, context):
                selected.append(layer)

        return self._order_layers(selected)

    def _should_include(self, layer, context: GenerationContext) -> bool:
        """Evaluate inclusion rule for a layer."""
        if layer.inclusion_rule == "always":
            return True

        if layer.inclusion_rule == "context":
            # Check if any context module matches layer's context_modules
            if layer.context_modules:
                return bool(
                    set(layer.context_modules) & set(context.detected_modules)
                )
            return True  # Include if no specific modules required

        if layer.inclusion_rule == "user_pref":
            pref_key = f"include_{layer.section_id}"
            return context.user_preferences.get(pref_key, True)

        return True

    def _order_layers(self, layers: List) -> List:
        """Order layers by layer_number, then section_id."""
        return sorted(layers, key=lambda l: (l.layer_number, l.section_id))


class ClaudeMdComposer:
    """Composes final CLAUDE.md from selected layers."""

    # Preferred section ordering within each layer
    SECTION_ORDER = [
        # Layer 0: Core
        "principles", "priorities", "critical-bugs",
        # Layer 1: Operational
        "quick-start", "got-commands", "development-workflow",
        "commit-conventions", "testing-patterns",
        # Layer 2: Contextual
        "project-overview", "environment-setup", "ai-onboarding",
        "architecture", "critical-knowledge", "scoring-algorithms",
        "code-search", "debugging", "quick-reference",
        # Layer 3: Persona
        "persona", "working-philosophy",
        # Layer 4: Ephemeral
        "session-state", "branch-context", "sprint-context",
        "dogfooding", "memories", "ml-collection",
    ]

    def compose(
        self,
        layers: List,
        context: GenerationContext
    ) -> str:
        """
        Compose layers into final CLAUDE.md content.

        Args:
            layers: Selected and ordered layers
            context: Generation context

        Returns:
            Complete CLAUDE.md content string
        """
        output = self._generate_header(context, len(layers))

        # Group by layer number for clear structure
        by_layer = {}
        for layer in layers:
            if layer.layer_number not in by_layer:
                by_layer[layer.layer_number] = []
            by_layer[layer.layer_number].append(layer)

        # Compose each layer group
        for layer_num in sorted(by_layer.keys()):
            layer_layers = by_layer[layer_num]
            # Sort within layer by section order
            layer_layers.sort(
                key=lambda l: (
                    self.SECTION_ORDER.index(l.section_id)
                    if l.section_id in self.SECTION_ORDER
                    else 999,
                    l.section_id
                )
            )

            for layer in layer_layers:
                if layer.content.strip():
                    output += layer.content
                    if not output.endswith("\n\n"):
                        output += "\n\n"

        return output.rstrip() + "\n"

    def _generate_header(self, context: GenerationContext, layer_count: int) -> str:
        """Generate dynamic header with generation metadata."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return f"""# CLAUDE.md - Cortical Text Processor Development Guide

<!--
  Auto-generated: {timestamp}
  Branch: {context.current_branch}
  Sprint: {context.active_sprint_id or 'None'}
  Layers: {layer_count}
  Modules: {', '.join(context.detected_modules) or 'all'}

  This file is auto-generated from GoT layers.
  To modify, edit the source layers in .got/entities/CML*.json
  Fallback: Original CLAUDE.md is preserved and never modified.
-->

---

"""


class ClaudeMdValidator:
    """Validates generated CLAUDE.md content."""

    REQUIRED_SECTIONS = [
        "Quick Session Start",
        "Work Priority Order",
        "Architecture",
        "Testing",
        "Quick Reference",
    ]

    CRITICAL_PATTERNS = [
        r"Security.*Bugs.*Features.*Docs",  # Priority order
        r"python\s+scripts/got_utils\.py",   # GoT commands
        r"pytest|unittest",                   # Test commands
    ]

    MIN_LINES = 200
    MAX_LINES = 10000

    def validate(self, content: str) -> ValidationResult:
        """
        Validate generated CLAUDE.md content.

        Args:
            content: Generated markdown content

        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []

        # Check minimum content
        if not content or len(content.strip()) < 100:
            errors.append("Content is empty or too short")
            return ValidationResult(is_valid=False, errors=errors)

        lines = content.splitlines()

        # Check required sections (case-insensitive partial match)
        content_lower = content.lower()
        for section in self.REQUIRED_SECTIONS:
            if section.lower() not in content_lower:
                errors.append(f"Missing required section: {section}")

        # Check critical patterns
        for pattern in self.CRITICAL_PATTERNS:
            if not re.search(pattern, content, re.IGNORECASE):
                warnings.append(f"Missing recommended pattern: {pattern}")

        # Check line count bounds
        if len(lines) < self.MIN_LINES:
            warnings.append(f"Content short: {len(lines)} lines (expected >{self.MIN_LINES})")
        if len(lines) > self.MAX_LINES:
            warnings.append(f"Content very long: {len(lines)} lines")

        # Check markdown structure
        headings = [l for l in lines if l.startswith("#")]
        if len(headings) < 5:
            warnings.append(f"Few headings found: {len(headings)}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


class ClaudeMdGenerator:
    """Main generator with fault tolerance."""

    def __init__(self, got_manager, got_dir: Path = None):
        """
        Initialize generator.

        Args:
            got_manager: GoTManager instance for layer access
            got_dir: GoT directory path (default: .got)
        """
        self.got = got_manager
        self.got_dir = got_dir or Path(".got")

        self.analyzer = ContextAnalyzer(self.got_dir)
        self.selector = LayerSelector()
        self.composer = ClaudeMdComposer()
        self.validator = ClaudeMdValidator()

        # Output paths
        self.output_dir = self.got_dir / "generated"
        self.output_path = self.output_dir / "CLAUDE.md"
        self.fallback_path = Path("CLAUDE.md")
        self.last_good_path = self.output_dir / "CLAUDE.md.last_good"
        self.hash_path = self.output_dir / "CLAUDE.md.hash"

    def generate(self, dry_run: bool = False) -> GenerationResult:
        """
        Generate CLAUDE.md with fault tolerance.

        Args:
            dry_run: If True, don't write to disk

        Returns:
            GenerationResult with success status and details
        """
        warnings = []

        try:
            # Phase 1: Analyze context
            logger.debug("Analyzing context...")
            context = self.analyzer.analyze()

            # Phase 2: Load and select layers
            logger.debug("Loading layers...")
            all_layers = self.got.list_claudemd_layers()

            if not all_layers:
                logger.warning("No layers found, using fallback")
                return self._use_fallback("No layers found in GoT")

            selected = self.selector.select(context, all_layers)
            logger.debug(f"Selected {len(selected)} of {len(all_layers)} layers")

            if not selected:
                logger.warning("No layers selected, using fallback")
                return self._use_fallback("No layers selected for context")

            # Phase 3: Compose
            logger.debug("Composing content...")
            content = self.composer.compose(selected, context)

            # Phase 4: Validate
            logger.debug("Validating content...")
            validation = self.validator.validate(content)

            if not validation.is_valid:
                logger.warning(f"Validation failed: {validation.errors}")
                return self._use_fallback(
                    f"Validation failed: {'; '.join(validation.errors)}"
                )

            warnings.extend(validation.warnings)

            # Phase 5: Write with backup (unless dry run)
            if not dry_run:
                logger.debug("Writing output...")
                self._write_with_backup(content)

            content_hash = self._compute_hash(content)

            return GenerationResult(
                success=True,
                path=self.output_path if not dry_run else None,
                layers_used=len(selected),
                content_hash=content_hash,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._use_fallback(str(e))

    def _use_fallback(self, reason: str) -> GenerationResult:
        """Use fallback when generation fails."""
        # Ensure fallback is available in output location
        if not self.output_path.exists() and self.fallback_path.exists():
            self._copy_fallback()

        path = self.output_path if self.output_path.exists() else self.fallback_path

        return GenerationResult(
            success=False,
            error=reason,
            path=path,
            fallback_used=True,
        )

    def _write_with_backup(self, content: str) -> None:
        """Write new content with backup of last known good."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Backup current if it exists and is valid
        if self.output_path.exists():
            current = self.output_path.read_text()
            if self.validator.validate(current).is_valid:
                self.last_good_path.write_text(current)

        # Write new content atomically
        temp_path = self.output_path.with_suffix('.tmp')
        temp_path.write_text(content)
        temp_path.rename(self.output_path)

        # Write hash
        self.hash_path.write_text(self._compute_hash(content))

    def _copy_fallback(self) -> None:
        """Copy original CLAUDE.md as fallback."""
        if self.fallback_path.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.fallback_path, self.output_path)

    def _compute_hash(self, content: str) -> str:
        """Compute content hash."""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ClaudeMdManager:
    """High-level API for CLAUDE.md layer management."""

    def __init__(self, got_manager, got_dir: Path = None):
        """
        Initialize manager.

        Args:
            got_manager: GoTManager instance
            got_dir: GoT directory path
        """
        self.got = got_manager
        self.got_dir = got_dir or Path(".got")
        self.generator = ClaudeMdGenerator(got_manager, self.got_dir)

    # === Generation ===

    def generate(self, dry_run: bool = False) -> GenerationResult:
        """Generate CLAUDE.md from layers."""
        return self.generator.generate(dry_run=dry_run)

    # === Layer Management ===

    def create_layer(self, **kwargs):
        """Create a new layer."""
        return self.got.create_claudemd_layer(**kwargs)

    def get_layer(self, layer_id: str):
        """Get a layer by ID."""
        return self.got.get_claudemd_layer(layer_id)

    def update_layer(self, layer_id: str, **kwargs):
        """Update a layer."""
        return self.got.update_claudemd_layer(layer_id, **kwargs)

    def list_layers(
        self,
        layer_type: str = None,
        freshness_status: str = None
    ):
        """List layers with optional filters."""
        return self.got.list_claudemd_layers(
            layer_type=layer_type,
            freshness_status=freshness_status
        )

    def delete_layer(self, layer_id: str) -> bool:
        """Delete a layer."""
        return self.got.delete_claudemd_layer(layer_id)

    # === Freshness Management ===

    def check_freshness(self) -> Dict[str, List]:
        """
        Check freshness of all layers.

        Returns:
            Dict with 'fresh', 'stale', 'regenerating' lists
        """
        layers = self.got.list_claudemd_layers()
        result = {"fresh": [], "stale": [], "regenerating": []}

        for layer in layers:
            if layer.freshness_status == "regenerating":
                result["regenerating"].append(layer)
            elif layer.is_stale():
                result["stale"].append(layer)
            else:
                result["fresh"].append(layer)

        return result

    def mark_layer_stale(self, layer_id: str, reason: str = "") -> bool:
        """Mark a layer as stale."""
        layer = self.get_layer(layer_id)
        if not layer:
            return False

        self.update_layer(
            layer_id,
            freshness_status="stale",
            regeneration_trigger=reason
        )
        return True

    # === Utility ===

    def get_output_path(self) -> Path:
        """Get the generated CLAUDE.md path."""
        return self.generator.output_path

    def get_fallback_path(self) -> Path:
        """Get the fallback CLAUDE.md path."""
        return self.generator.fallback_path
