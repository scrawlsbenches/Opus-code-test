# CLAUDE.md Auto-Generation System Design

**Status:** Draft
**Task:** T-20251222-204058-3675a72c (Epic)
**Date:** 2025-12-22

---

## 1. Overview

This document specifies the design for auto-generating CLAUDE.md content using the Graph of Thought (GoT) system with a layered architecture that supports:

- **Context-aware content**: Include only relevant sections based on current work
- **Versioned personas**: Track and evolve working style over time
- **Knowledge freshness**: Auto-decay stale information, promote new learnings
- **Fault tolerance**: Never break sessions, always have fallback

---

## 2. Layer Architecture

### 2.1 Layer Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLAUDE.MD LAYER HIERARCHY                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ LAYER 0: IMMUTABLE CORE (~500 lines)                        │    │
│  │ • Security principles (never change)                        │    │
│  │ • Work priority order (Security → Bugs → Features → Docs)   │    │
│  │ • Critical bugs to never reintroduce                        │    │
│  │ • Core ethical commitments                                  │    │
│  │ REGENERATION: Manual only, requires explicit approval       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓ always included                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ LAYER 1: OPERATIONAL (~1000 lines)                          │    │
│  │ • GoT commands and usage                                    │    │
│  │ • Test commands and patterns                                │    │
│  │ • Git conventions and commit format                         │    │
│  │ • Development workflow procedures                           │    │
│  │ REGENERATION: On command/API changes                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓ always included                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ LAYER 2: CONTEXTUAL (~1500 lines, modular)                  │    │
│  │ ├── architecture.md    - Module structure, data flow        │    │
│  │ ├── query-module.md    - Query/search specifics             │    │
│  │ ├── reasoning-module.md - Reasoning framework details       │    │
│  │ ├── spark-module.md    - SparkSLM specifics                 │    │
│  │ ├── got-module.md      - GoT system details                 │    │
│  │ └── critical-knowledge.md - Performance lessons, gotchas    │    │
│  │ REGENERATION: On code changes in respective modules         │    │
│  │ INCLUSION: Based on files being worked on                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓ conditionally included                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ LAYER 3: PERSONA (~300 lines, versioned)                    │    │
│  │ • Working philosophy and principles                         │    │
│  │ • Communication style preferences                           │    │
│  │ • Domain expertise emphasis                                 │    │
│  │ • Proactivity level                                         │    │
│  │ REGENERATION: Explicit persona update request               │    │
│  │ VERSION: Tracked with rationale for changes                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ↓ included based on user preference     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ LAYER 4: EPHEMERAL (~200 lines, auto-generated)             │    │
│  │ • Current branch state                                      │    │
│  │ • Active sprint context                                     │    │
│  │ • Recent session learnings                                  │    │
│  │ • ML collection metrics                                     │    │
│  │ • Corpus indexing status                                    │    │
│  │ REGENERATION: Every session start                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Layer Properties

| Layer | ID Prefix | Storage | Freshness | Inclusion Rule |
|-------|-----------|---------|-----------|----------------|
| L0 Core | `CML0-` | `.got/entities/` | Permanent | Always |
| L1 Operational | `CML1-` | `.got/entities/` | Monthly decay | Always |
| L2 Contextual | `CML2-` | `.got/entities/` | Weekly decay | Context-based |
| L3 Persona | `CML3-` | `.got/entities/` + versions/ | Manual | User preference |
| L4 Ephemeral | `CML4-` | `.got/entities/` | Session | Always (regenerated) |

---

## 3. GoT Entity Model

### 3.1 ClaudeMdLayer Entity

```python
@dataclass
class ClaudeMdLayer(Entity):
    """CLAUDE.md layer content entity."""

    # Core fields
    layer_type: str = ""          # "core", "operational", "contextual", "persona", "ephemeral"
    layer_number: int = 0         # 0-4
    section_id: str = ""          # e.g., "architecture", "query-module"
    title: str = ""               # Human-readable title
    content: str = ""             # Markdown content

    # Freshness tracking
    freshness_status: str = "fresh"  # "fresh", "stale", "regenerating"
    freshness_decay_days: int = 0    # 0 = never decay
    last_regenerated: str = ""       # ISO 8601 timestamp
    regeneration_trigger: str = ""   # What caused last regeneration

    # Inclusion rules
    inclusion_rule: str = "always"   # "always", "context", "user_pref"
    context_modules: List[str] = field(default_factory=list)  # For context-based
    context_branches: List[str] = field(default_factory=list)  # Branch patterns

    # Versioning
    content_hash: str = ""        # SHA256 of content (first 16 chars)
    version_number: int = 1       # Content version

    # Metadata
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.entity_type = "claudemd_layer"
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate layer ID: CML{layer_number}-{section_id}-{hash}."""
        import hashlib
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        hash_suffix = hashlib.sha256(
            f"{self.layer_type}{self.section_id}{timestamp}".encode()
        ).hexdigest()[:8]
        return f"CML{self.layer_number}-{self.section_id}-{hash_suffix}"

    def compute_content_hash(self) -> str:
        """Compute hash of content for change detection."""
        import hashlib
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def is_stale(self, current_date: datetime = None) -> bool:
        """Check if layer content is stale based on decay rules."""
        if self.freshness_decay_days == 0:
            return False  # Never decays
        if not self.last_regenerated:
            return True
        from datetime import datetime, timedelta
        current = current_date or datetime.now()
        regen_date = datetime.fromisoformat(self.last_regenerated)
        return (current - regen_date).days > self.freshness_decay_days
```

### 3.2 ClaudeMdVersion Entity (Persona Versioning)

```python
@dataclass
class ClaudeMdVersion(Entity):
    """Version snapshot for persona/layer evolution tracking."""

    layer_id: str = ""            # Reference to ClaudeMdLayer
    version_number: int = 1
    content_snapshot: str = ""     # Full content at this version
    content_hash: str = ""

    # Change tracking
    change_rationale: str = ""     # Why this version was created
    changed_by: str = ""           # Agent/user who made change
    changed_sections: List[str] = field(default_factory=list)

    # Diff info
    additions: int = 0             # Lines added
    deletions: int = 0             # Lines removed

    properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.entity_type = "claudemd_version"
        if not self.id:
            self.id = f"CMV-{self.layer_id}-v{self.version_number}"
```

### 3.3 Edge Types for CLAUDE.md

| Edge Type | Direction | Purpose | Example |
|-----------|-----------|---------|---------|
| `LAYER_CONTAINS` | Layer → Section | Parent-child within layer | CML2-root LAYER_CONTAINS CML2-query |
| `TRIGGERS_STALENESS` | Entity → Layer | What makes a layer stale | Task completion triggers layer refresh |
| `REFERENCES_MODULE` | Layer → Code | What code does layer describe | CML2-query REFERENCES cortical/query/ |
| `VERSIONED_FROM` | Version → Version | Version chain | CMV-v2 VERSIONED_FROM CMV-v1 |
| `GENERATED_BY` | Layer → Task | What task generated the layer | For audit trail |

---

## 4. Generation Pipeline

### 4.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GENERATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                │
│  │  CONTEXT    │   │  LAYER      │   │  COMPOSER   │                │
│  │  ANALYZER   │──▶│  SELECTOR   │──▶│             │                │
│  └─────────────┘   └─────────────┘   └──────┬──────┘                │
│        │                 │                   │                       │
│        │                 │                   ▼                       │
│        │                 │          ┌─────────────┐                 │
│        │                 │          │  VALIDATOR  │                 │
│        │                 │          └──────┬──────┘                 │
│        │                 │                 │                        │
│        │                 │        ┌────────┴────────┐               │
│        ▼                 ▼        ▼                 ▼               │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                      GOT STORAGE                          │      │
│  │  .got/entities/CML*.json                                  │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                      OUTPUT                               │      │
│  │  .got/generated/CLAUDE.md (always available)              │      │
│  │  .got/generated/CLAUDE.md.hash (for validation)           │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                      FALLBACK                             │      │
│  │  CLAUDE.md (original, never modified)                     │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Details

#### Context Analyzer

```python
class ContextAnalyzer:
    """Analyzes current work context for layer selection."""

    def analyze(self) -> GenerationContext:
        """Gather context for generation decisions."""
        return GenerationContext(
            current_branch=self._get_current_branch(),
            active_sprint=self._get_active_sprint(),
            touched_files=self._get_recently_touched_files(),
            detected_modules=self._detect_modules_from_files(),
            in_progress_tasks=self._get_in_progress_tasks(),
            user_preferences=self._load_user_preferences(),
        )

    def _detect_modules_from_files(self) -> List[str]:
        """Detect which modules are relevant based on touched files."""
        # Map file paths to module identifiers
        # e.g., cortical/query/expansion.py -> "query"
        pass
```

#### Layer Selector

```python
class LayerSelector:
    """Selects which layers to include based on context."""

    def select(
        self,
        context: GenerationContext,
        available_layers: List[ClaudeMdLayer]
    ) -> List[ClaudeMdLayer]:
        """Select layers based on inclusion rules and context."""
        selected = []

        for layer in available_layers:
            if self._should_include(layer, context):
                selected.append(layer)

        return self._order_layers(selected)

    def _should_include(
        self,
        layer: ClaudeMdLayer,
        context: GenerationContext
    ) -> bool:
        """Evaluate inclusion rule for a layer."""
        if layer.inclusion_rule == "always":
            return True

        if layer.inclusion_rule == "context":
            # Check if any touched module matches
            return bool(
                set(layer.context_modules) &
                set(context.detected_modules)
            )

        if layer.inclusion_rule == "user_pref":
            return context.user_preferences.get(
                f"include_{layer.section_id}", True
            )

        return True
```

#### Composer

```python
class ClaudeMdComposer:
    """Composes final CLAUDE.md from selected layers."""

    SECTION_ORDER = [
        "quick-start",
        "persona",
        "cognitive-continuity",
        "project-overview",
        "environment-setup",
        "ai-onboarding",
        "architecture",
        "critical-knowledge",
        "development-workflow",
        "testing-patterns",
        "common-tasks",
        "scoring-algorithms",
        "debugging",
        "quick-reference",
        "dogfooding",
        "memories",
        "ml-collection",
    ]

    def compose(
        self,
        layers: List[ClaudeMdLayer],
        context: GenerationContext
    ) -> str:
        """Compose layers into final CLAUDE.md content."""
        # Group by section
        sections = self._group_by_section(layers)

        # Order sections
        ordered = self._order_sections(sections)

        # Generate header
        output = self._generate_header(context)

        # Append each section
        for section_id in self.SECTION_ORDER:
            if section_id in ordered:
                output += ordered[section_id].content
                output += "\n\n"

        # Append any remaining sections
        for section_id, layer in ordered.items():
            if section_id not in self.SECTION_ORDER:
                output += layer.content
                output += "\n\n"

        return output

    def _generate_header(self, context: GenerationContext) -> str:
        """Generate dynamic header with generation metadata."""
        return f"""# CLAUDE.md - Cortical Text Processor Development Guide

<!--
  Auto-generated: {datetime.now().isoformat()}
  Branch: {context.current_branch}
  Sprint: {context.active_sprint.id if context.active_sprint else 'None'}
  Layers: {len(context.selected_layers)}

  DO NOT EDIT DIRECTLY - Use layer system for modifications
  Fallback: Original CLAUDE.md is preserved
-->

---

"""
```

### 4.3 Fault Tolerance

```python
class ClaudeMdGenerator:
    """Main generator with fault tolerance."""

    def __init__(self, got_manager: GoTManager):
        self.got = got_manager
        self.analyzer = ContextAnalyzer()
        self.selector = LayerSelector()
        self.composer = ClaudeMdComposer()
        self.validator = ClaudeMdValidator()

        # Paths
        self.output_path = Path(".got/generated/CLAUDE.md")
        self.fallback_path = Path("CLAUDE.md")
        self.last_good_path = Path(".got/generated/CLAUDE.md.last_good")

    def generate(self) -> GenerationResult:
        """Generate CLAUDE.md with fault tolerance."""
        try:
            # Phase 1: Analyze context
            context = self.analyzer.analyze()

            # Phase 2: Load and select layers
            all_layers = self.got.list_claudemd_layers()
            selected = self.selector.select(context, all_layers)

            # Phase 3: Compose
            content = self.composer.compose(selected, context)

            # Phase 4: Validate
            validation = self.validator.validate(content)
            if not validation.is_valid:
                raise ValidationError(validation.errors)

            # Phase 5: Write with backup
            self._write_with_backup(content)

            return GenerationResult(
                success=True,
                path=self.output_path,
                layers_used=len(selected),
                content_hash=self._compute_hash(content),
            )

        except Exception as e:
            # Log error
            logger.error(f"Generation failed: {e}")

            # Ensure fallback is available
            if not self.output_path.exists():
                self._copy_fallback()

            return GenerationResult(
                success=False,
                error=str(e),
                path=self.fallback_path,
                fallback_used=True,
            )

    def _write_with_backup(self, content: str) -> None:
        """Write new content with backup of last known good."""
        # Backup current if it exists and is valid
        if self.output_path.exists():
            current = self.output_path.read_text()
            if self.validator.validate(current).is_valid:
                self.last_good_path.write_text(current)

        # Write new content atomically
        temp_path = self.output_path.with_suffix('.tmp')
        temp_path.write_text(content)
        temp_path.rename(self.output_path)

    def _copy_fallback(self) -> None:
        """Copy original CLAUDE.md as fallback."""
        if self.fallback_path.exists():
            shutil.copy(self.fallback_path, self.output_path)
```

---

## 5. Freshness and Staleness System

### 5.1 Staleness Rules

```yaml
# .got/claude-md/staleness-rules.yaml
rules:
  - layer_type: core
    decay_days: 0  # Never decays
    triggers: []    # No automatic triggers

  - layer_type: operational
    decay_days: 30
    triggers:
      - event: command_api_changed
        action: mark_stale
      - event: test_pattern_added
        action: mark_stale

  - layer_type: contextual
    decay_days: 7
    triggers:
      - event: module_refactored
        module_pattern: "cortical/{module}/**"
        action: mark_stale
      - event: performance_lesson_learned
        action: append_knowledge

  - layer_type: persona
    decay_days: 0  # Never auto-decays
    triggers:
      - event: explicit_persona_update
        action: create_version

  - layer_type: ephemeral
    decay_days: 0  # Regenerated each session
    triggers:
      - event: session_start
        action: regenerate
```

### 5.2 Staleness Detection

```python
class StalenessManager:
    """Manages layer freshness and staleness."""

    def check_all_layers(self) -> StalenessReport:
        """Check staleness of all layers."""
        layers = self.got.list_claudemd_layers()
        report = StalenessReport()

        for layer in layers:
            if layer.is_stale():
                report.stale_layers.append(layer)
            elif self._has_pending_triggers(layer):
                report.triggered_layers.append(layer)
            else:
                report.fresh_layers.append(layer)

        return report

    def process_event(self, event: str, **kwargs) -> None:
        """Process an event that might trigger staleness."""
        rules = self._load_rules()

        for rule in rules:
            for trigger in rule.triggers:
                if trigger.event == event:
                    affected = self._find_affected_layers(rule, kwargs)
                    for layer in affected:
                        self._apply_action(trigger.action, layer)

    def auto_regenerate_stale(self) -> List[ClaudeMdLayer]:
        """Regenerate all stale layers that support auto-regeneration."""
        regenerated = []

        for layer in self.got.list_claudemd_layers(status="stale"):
            if layer.layer_type == "ephemeral":
                # Ephemeral layers can always be regenerated
                new_content = self._regenerate_ephemeral(layer)
                self._update_layer(layer, new_content)
                regenerated.append(layer)
            elif layer.properties.get("auto_regenerate", False):
                # Other layers only if explicitly allowed
                new_content = self._regenerate_layer(layer)
                self._update_layer(layer, new_content)
                regenerated.append(layer)

        return regenerated
```

---

## 6. Knowledge Promotion Pipeline

### 6.1 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE PROMOTION PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  SESSION LEARNINGS                                                   │
│  samples/memories/2025-12-22-*.md                                   │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ LEARNING ANALYZER                                           │    │
│  │ • Extract key insights                                      │    │
│  │ • Detect patterns across sessions                           │    │
│  │ • Score importance (frequency, impact, recency)             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ CANDIDATE KNOWLEDGE                                         │    │
│  │ .got/claude-md/candidates/                                  │    │
│  │ • Pending promotion                                         │    │
│  │ • Score > threshold                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│           │                                                          │
│           ├────────────────────────────────────────┐                │
│           ▼                                        ▼                │
│  ┌──────────────────┐                   ┌──────────────────┐       │
│  │ AUTO-PROMOTION   │                   │ MANUAL REVIEW    │       │
│  │ score > 0.9      │                   │ 0.7 < score < 0.9│       │
│  │ + 3+ occurrences │                   │ flagged for user │       │
│  └──────────────────┘                   └──────────────────┘       │
│           │                                        │                │
│           └────────────────────────────────────────┘                │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ CRITICAL KNOWLEDGE (Layer 2)                                │    │
│  │ Promoted insights become part of CLAUDE.md                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Promotion Criteria

```python
class KnowledgePromoter:
    """Manages knowledge promotion from sessions to layers."""

    PROMOTION_THRESHOLDS = {
        "auto_promote": 0.9,    # Auto-promote if score >= 0.9
        "manual_review": 0.7,   # Flag for review if score >= 0.7
        "discard": 0.3,         # Discard if score < 0.3
    }

    def score_insight(self, insight: Insight) -> float:
        """Score an insight for promotion potential."""
        score = 0.0

        # Frequency: How often has this come up?
        frequency_score = min(insight.occurrence_count / 5, 1.0) * 0.3
        score += frequency_score

        # Impact: Did it prevent bugs or improve outcomes?
        if insight.prevented_bug:
            score += 0.25
        if insight.improved_performance:
            score += 0.15

        # Recency: Recent insights weighted higher
        days_old = (datetime.now() - insight.first_seen).days
        recency_score = max(0, 1 - days_old / 30) * 0.15
        score += recency_score

        # Generality: Does it apply broadly?
        if insight.applies_to_multiple_modules:
            score += 0.15

        return score
```

---

## 7. Persona Versioning

### 7.1 Version Control

```python
class PersonaVersionManager:
    """Manages persona versioning and evolution."""

    def create_version(
        self,
        persona_layer: ClaudeMdLayer,
        rationale: str,
        changed_by: str = "system"
    ) -> ClaudeMdVersion:
        """Create a new version snapshot of persona."""
        # Get current version number
        current_versions = self.got.list_claudemd_versions(
            layer_id=persona_layer.id
        )
        next_version = max(v.version_number for v in current_versions) + 1

        # Compute diff from previous
        if current_versions:
            prev = max(current_versions, key=lambda v: v.version_number)
            diff = self._compute_diff(prev.content_snapshot, persona_layer.content)
        else:
            diff = DiffResult(additions=len(persona_layer.content.splitlines()))

        # Create version entity
        version = ClaudeMdVersion(
            layer_id=persona_layer.id,
            version_number=next_version,
            content_snapshot=persona_layer.content,
            content_hash=persona_layer.compute_content_hash(),
            change_rationale=rationale,
            changed_by=changed_by,
            additions=diff.additions,
            deletions=diff.deletions,
        )

        self.got.create_claudemd_version(version)

        # Create version chain edge
        if current_versions:
            self.got.add_edge(
                version.id,
                prev.id,
                "VERSIONED_FROM"
            )

        return version

    def rollback_to_version(
        self,
        persona_layer: ClaudeMdLayer,
        version_number: int,
        rationale: str
    ) -> ClaudeMdLayer:
        """Rollback persona to a previous version."""
        # Find the target version
        target = self.got.get_claudemd_version(
            f"CMV-{persona_layer.id}-v{version_number}"
        )

        if not target:
            raise VersionNotFoundError(version_number)

        # Create new version with rollback content
        self.create_version(
            persona_layer,
            rationale=f"Rollback to v{version_number}: {rationale}",
            changed_by="rollback"
        )

        # Update layer content
        persona_layer.content = target.content_snapshot
        persona_layer.content_hash = target.content_hash
        self.got.update_claudemd_layer(persona_layer)

        return persona_layer
```

---

## 8. Validation System

### 8.1 Validation Rules

```python
class ClaudeMdValidator:
    """Validates generated CLAUDE.md content."""

    REQUIRED_SECTIONS = [
        "Quick Session Start",
        "Work Priority Order",
        "Architecture Map",
        "Testing Patterns",
        "Quick Reference",
    ]

    CRITICAL_PATTERNS = [
        r"Security.*Bugs.*Features.*Docs",  # Priority order
        r"python\s+scripts/got_utils\.py",   # GoT commands
        r"python\s+-m\s+pytest",             # Test commands
    ]

    def validate(self, content: str) -> ValidationResult:
        """Validate generated CLAUDE.md content."""
        errors = []
        warnings = []

        # Check required sections
        for section in self.REQUIRED_SECTIONS:
            if section not in content:
                errors.append(f"Missing required section: {section}")

        # Check critical patterns
        for pattern in self.CRITICAL_PATTERNS:
            if not re.search(pattern, content):
                errors.append(f"Missing critical pattern: {pattern}")

        # Check length sanity
        lines = content.splitlines()
        if len(lines) < 500:
            warnings.append(f"Content suspiciously short: {len(lines)} lines")
        if len(lines) > 5000:
            warnings.append(f"Content very long: {len(lines)} lines")

        # Check for broken wiki-links
        broken_links = self._check_wiki_links(content)
        for link in broken_links:
            warnings.append(f"Broken wiki-link: {link}")

        # Check markdown validity
        md_errors = self._validate_markdown(content)
        errors.extend(md_errors)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
```

---

## 9. API Design

### 9.1 High-Level API

```python
# cortical/got/claudemd.py

class ClaudeMdManager:
    """High-level API for CLAUDE.md layer management."""

    def __init__(self, got_manager: GoTManager):
        self.got = got_manager
        self.generator = ClaudeMdGenerator(got_manager)
        self.staleness = StalenessManager(got_manager)
        self.promoter = KnowledgePromoter(got_manager)
        self.persona_mgr = PersonaVersionManager(got_manager)

    # Generation
    def generate(self) -> GenerationResult:
        """Generate CLAUDE.md from layers."""
        return self.generator.generate()

    def regenerate_stale(self) -> List[ClaudeMdLayer]:
        """Regenerate all stale layers."""
        return self.staleness.auto_regenerate_stale()

    # Layer CRUD
    def create_layer(self, **kwargs) -> ClaudeMdLayer:
        """Create a new layer."""
        return self.got.create_claudemd_layer(**kwargs)

    def get_layer(self, layer_id: str) -> Optional[ClaudeMdLayer]:
        """Get a layer by ID."""
        return self.got.get_claudemd_layer(layer_id)

    def update_layer(self, layer_id: str, **kwargs) -> ClaudeMdLayer:
        """Update a layer."""
        return self.got.update_claudemd_layer(layer_id, **kwargs)

    def list_layers(
        self,
        layer_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[ClaudeMdLayer]:
        """List layers with optional filters."""
        return self.got.list_claudemd_layers(
            layer_type=layer_type,
            status=status
        )

    # Freshness
    def check_freshness(self) -> StalenessReport:
        """Check freshness of all layers."""
        return self.staleness.check_all_layers()

    def mark_stale(self, layer_id: str, reason: str) -> None:
        """Manually mark a layer as stale."""
        layer = self.get_layer(layer_id)
        if layer:
            layer.freshness_status = "stale"
            layer.regeneration_trigger = reason
            self.update_layer(layer.id, **layer.__dict__)

    # Persona versioning
    def create_persona_version(
        self,
        rationale: str
    ) -> ClaudeMdVersion:
        """Create a new persona version."""
        persona = self.get_layer_by_type("persona")
        return self.persona_mgr.create_version(persona, rationale)

    def rollback_persona(
        self,
        version_number: int,
        rationale: str
    ) -> ClaudeMdLayer:
        """Rollback persona to a previous version."""
        persona = self.get_layer_by_type("persona")
        return self.persona_mgr.rollback_to_version(
            persona, version_number, rationale
        )

    # Knowledge promotion
    def promote_knowledge(
        self,
        insight: str,
        source: str,
        auto: bool = False
    ) -> PromotionResult:
        """Promote an insight to critical knowledge."""
        return self.promoter.promote(insight, source, auto=auto)
```

### 9.2 CLI Commands

```bash
# Generation
python scripts/claudemd_utils.py generate              # Generate CLAUDE.md
python scripts/claudemd_utils.py generate --dry-run    # Preview without writing

# Layer management
python scripts/claudemd_utils.py layer list            # List all layers
python scripts/claudemd_utils.py layer list --stale    # List stale layers
python scripts/claudemd_utils.py layer show CML2-xxx   # Show layer details
python scripts/claudemd_utils.py layer update CML2-xxx # Update layer

# Freshness
python scripts/claudemd_utils.py freshness check       # Check all layers
python scripts/claudemd_utils.py freshness regenerate  # Regenerate stale

# Persona
python scripts/claudemd_utils.py persona versions      # List versions
python scripts/claudemd_utils.py persona rollback 3    # Rollback to v3

# Knowledge
python scripts/claudemd_utils.py knowledge candidates  # View pending
python scripts/claudemd_utils.py knowledge promote ID  # Promote candidate
```

---

## 10. File Structure

```
.got/
├── entities/
│   ├── CML0-*.json          # Layer 0 (Core) entities
│   ├── CML1-*.json          # Layer 1 (Operational) entities
│   ├── CML2-*.json          # Layer 2 (Contextual) entities
│   ├── CML3-*.json          # Layer 3 (Persona) entities
│   ├── CML4-*.json          # Layer 4 (Ephemeral) entities
│   ├── CMV-*.json           # Version snapshots
│   └── E-CML*.json          # CLAUDE.md related edges
│
├── claude-md/
│   ├── rules/
│   │   ├── staleness-rules.yaml
│   │   ├── inclusion-rules.yaml
│   │   └── promotion-rules.yaml
│   │
│   ├── candidates/          # Pending knowledge promotions
│   │   └── K-*.json
│   │
│   └── templates/           # Section templates
│       ├── quick-start.md.tmpl
│       ├── architecture.md.tmpl
│       └── ...
│
├── generated/
│   ├── CLAUDE.md            # Generated output
│   ├── CLAUDE.md.hash       # Content hash for validation
│   └── CLAUDE.md.last_good  # Last known good version
│
└── versions/
    └── persona/
        ├── v1.0/
        │   ├── content.md
        │   └── rationale.md
        ├── v1.5/
        └── current -> v2.0/

CLAUDE.md                     # Original (NEVER MODIFIED, fallback)
```

---

## 11. Testing Strategy

### 11.1 Test Categories

| Category | Purpose | Coverage Target |
|----------|---------|-----------------|
| **Unit** | Individual components | 90% |
| **Integration** | Component interaction | 80% |
| **Behavioral** | User workflows | Key workflows |
| **Performance** | Generation speed | <500ms |
| **Regression** | Quality preservation | All critical sections |

### 11.2 Test Files

```
tests/
├── unit/
│   ├── test_claudemd_layer.py          # Layer entity tests
│   ├── test_claudemd_generator.py      # Generator tests
│   ├── test_claudemd_selector.py       # Layer selector tests
│   ├── test_claudemd_composer.py       # Composer tests
│   ├── test_claudemd_staleness.py      # Staleness manager tests
│   ├── test_claudemd_promoter.py       # Knowledge promoter tests
│   └── test_claudemd_validator.py      # Validator tests
│
├── integration/
│   ├── test_claudemd_pipeline.py       # Full pipeline tests
│   ├── test_claudemd_got_integration.py # GoT integration tests
│   └── test_claudemd_fault_tolerance.py # Recovery tests
│
├── behavioral/
│   └── test_claudemd_workflows.py      # User workflow tests
│
├── performance/
│   └── test_claudemd_performance.py    # Speed tests
│
└── regression/
    └── test_claudemd_quality.py        # Quality preservation tests
```

---

## 12. Implementation Phases

### Phase 1: Core Entity Types (Week 1)
- [ ] Add ClaudeMdLayer entity to types.py
- [ ] Add ClaudeMdVersion entity to types.py
- [ ] Add CRUD methods to GoTManager
- [ ] Add edge types for layer relationships
- [ ] Unit tests for entities and CRUD

### Phase 2: Generation Pipeline (Week 2)
- [ ] Implement ContextAnalyzer
- [ ] Implement LayerSelector
- [ ] Implement ClaudeMdComposer
- [ ] Implement ClaudeMdValidator
- [ ] Implement ClaudeMdGenerator with fault tolerance
- [ ] Integration tests for pipeline

### Phase 3: Freshness System (Week 3)
- [ ] Implement StalenessManager
- [ ] Implement staleness rules parser
- [ ] Implement auto-regeneration
- [ ] Add CLI commands for freshness
- [ ] Tests for staleness system

### Phase 4: Knowledge Promotion (Week 4)
- [ ] Implement KnowledgePromoter
- [ ] Implement insight scoring
- [ ] Implement promotion workflow
- [ ] Add CLI commands
- [ ] Tests for promotion system

### Phase 5: Persona Versioning (Week 5)
- [ ] Implement PersonaVersionManager
- [ ] Implement version diffing
- [ ] Implement rollback
- [ ] Add CLI commands
- [ ] Tests for versioning

### Phase 6: Integration & Polish (Week 6)
- [ ] Extract initial layers from current CLAUDE.md
- [ ] Full integration testing
- [ ] Performance optimization
- [ ] Documentation
- [ ] User acceptance testing

---

## 13. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Generation failure breaks sessions | High | Always maintain fallback to original CLAUDE.md |
| Content quality degradation | High | Validation with required sections, regression tests |
| Performance regression | Medium | <500ms generation target, caching |
| Layer corruption | Medium | Checksums, WAL, recovery procedures |
| Version conflicts | Medium | Optimistic locking, conflict resolution |

---

## 14. Success Criteria

1. **Functional**: Generated CLAUDE.md is valid and useful
2. **Reliable**: Zero session failures due to generation
3. **Fast**: Generation completes in <500ms
4. **Maintainable**: Layers can be updated independently
5. **Observable**: Clear visibility into layer status and history

---

*Document Status: Draft - Ready for Review*
