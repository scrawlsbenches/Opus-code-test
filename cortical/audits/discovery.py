"""
WovenMind Pattern Discovery for Codebase Audits.

Uses the WovenMind dual-process cognitive architecture to discover
emergent patterns in audit findings through:
- Hebbian learning for co-occurrence patterns
- Abstraction formation for higher-level concepts
- Novelty detection for anomalies

This module provides the business logic for pattern discovery.
CLI commands and scripts should import from here.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Protocol
from collections import defaultdict
from dataclasses import dataclass, field

from cortical.reasoning.woven_mind import WovenMind, WovenMindConfig
from cortical.reasoning.loom import ThinkingMode


# =============================================================================
# CONSTANTS
# =============================================================================

# TODO: Revisit storage location - consider moving to CDG for proper
# transactional persistence instead of raw JSON file in .got/
DEFAULT_MIND_STATE_FILE = ".got/woven_audit_mind.json"
DEFAULT_DISCOVERY_LOG_FILE = ".got/woven_discoveries.json"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DiscoveryResult:
    """Results from a discovery session."""
    findings_fed: int = 0
    patterns_observed: int = 0
    surprising_files: List[Dict[str, Any]] = field(default_factory=list)
    abstractions: List[Dict[str, Any]] = field(default_factory=list)
    session_num: int = 1
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'findings_fed': self.findings_fed,
            'patterns_observed': self.patterns_observed,
            'surprising_files': self.surprising_files,
            'abstractions': self.abstractions,
            'session_num': self.session_num,
            'error': self.error,
        }


@dataclass
class DiscoveryConfig:
    """Configuration for WovenMind discovery."""
    surprise_threshold: float = 0.3
    min_frequency: int = 3
    k_winners: int = 10
    novelty_threshold: float = 0.6
    enable_auto_consolidation: bool = True
    max_tokens: int = 10000


# =============================================================================
# PERSISTENCE
# =============================================================================

class DiscoveryPersistence(Protocol):
    """Protocol for discovery state persistence."""

    def load_mind_state(self) -> Optional[Dict[str, Any]]:
        """Load persisted WovenMind state."""
        ...

    def save_mind_state(self, mind: WovenMind, metadata: Dict[str, Any]) -> None:
        """Save WovenMind state."""
        ...

    def load_discovery_log(self) -> Dict[str, Any]:
        """Load discovery log."""
        ...

    def save_discovery_log(self, log: Dict[str, Any]) -> None:
        """Save discovery log."""
        ...

    def clear_state(self) -> None:
        """Clear all persisted state."""
        ...


class FileDiscoveryPersistence:
    """File-based persistence for discovery state."""

    def __init__(
        self,
        mind_state_file: Optional[Path] = None,
        discovery_log_file: Optional[Path] = None,
    ):
        self._mind_state_file = mind_state_file or Path(DEFAULT_MIND_STATE_FILE)
        self._discovery_log_file = discovery_log_file or Path(DEFAULT_DISCOVERY_LOG_FILE)

    def load_mind_state(self) -> Optional[Dict[str, Any]]:
        """Load persisted WovenMind state."""
        if self._mind_state_file.exists():
            try:
                with open(self._mind_state_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def save_mind_state(self, mind: WovenMind, metadata: Dict[str, Any]) -> None:
        """Save WovenMind state to disk."""
        self._mind_state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "mind": mind.to_dict(),
            "metadata": {
                **metadata,
                "saved_at": datetime.now().isoformat(),
            }
        }
        with open(self._mind_state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_discovery_log(self) -> Dict[str, Any]:
        """Load discovery log."""
        if self._discovery_log_file.exists():
            try:
                with open(self._discovery_log_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "sessions": [],
            "total_findings_processed": 0,
            "surprising_files": [],
        }

    def save_discovery_log(self, log: Dict[str, Any]) -> None:
        """Save discovery log."""
        self._discovery_log_file.parent.mkdir(parents=True, exist_ok=True)
        log["updated_at"] = datetime.now().isoformat()
        with open(self._discovery_log_file, 'w') as f:
            json.dump(log, f, indent=2)

    def clear_state(self) -> None:
        """Clear all persisted state."""
        for f in [self._mind_state_file, self._discovery_log_file]:
            if f.exists():
                f.unlink()


class InMemoryDiscoveryPersistence:
    """In-memory persistence for testing."""

    def __init__(self):
        self._mind_state: Optional[Dict[str, Any]] = None
        self._discovery_log: Dict[str, Any] = {
            "sessions": [],
            "total_findings_processed": 0,
            "surprising_files": [],
        }

    def load_mind_state(self) -> Optional[Dict[str, Any]]:
        return self._mind_state

    def save_mind_state(self, mind: WovenMind, metadata: Dict[str, Any]) -> None:
        self._mind_state = {
            "mind": mind.to_dict(),
            "metadata": {**metadata, "saved_at": datetime.now().isoformat()},
        }

    def load_discovery_log(self) -> Dict[str, Any]:
        return self._discovery_log

    def save_discovery_log(self, log: Dict[str, Any]) -> None:
        self._discovery_log = log

    def clear_state(self) -> None:
        self._mind_state = None
        self._discovery_log = {
            "sessions": [],
            "total_findings_processed": 0,
            "surprising_files": [],
        }


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize_finding(
    finding: Dict[str, Any],
    file_info: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Convert an audit finding into tokens for WovenMind.

    Creates semantic tokens that capture different aspects:
    - Pattern type (e.g., "pattern:should_be", "pattern:TODO")
    - File path components (e.g., "dir:got", "file:api.py")
    - File characteristics (e.g., "trait:high_churn", "trait:bug_prone")

    This lets WovenMind learn associations like:
    "files in got/ with TODO patterns tend to have high churn"
    """
    tokens = []

    # Pattern token - normalize
    pattern = finding.get('pattern', 'unknown')
    normalized_pattern = pattern.replace(' ', '_').lower().rstrip(':')
    tokens.append(f"pattern:{normalized_pattern}")

    # Extract file info from finding ID (format: "path/file.py:line")
    finding_id = finding.get('id', '')
    if ':' in finding_id:
        file_path = finding_id.rsplit(':', 1)[0]

        # Directory components
        parts = file_path.split('/')
        if len(parts) > 1:
            for part in parts[:-1]:
                tokens.append(f"dir:{part}")

        # File name
        if parts:
            tokens.append(f"file:{parts[-1]}")

    # Add file characteristics if available
    if file_info:
        if file_info.get('high_churn'):
            tokens.append("trait:high_churn")
        if file_info.get('bug_prone'):
            tokens.append("trait:bug_prone")
        if file_info.get('stale'):
            tokens.append("trait:stale_code")

    return tokens


def tokenize_file_context(file_path: str, git_analysis: Dict[str, Any]) -> List[str]:
    """Create tokens representing a file's characteristics."""
    tokens = [f"file:{Path(file_path).name}"]

    # Directory context
    parts = file_path.split('/')
    for part in parts[:-1]:
        tokens.append(f"dir:{part}")

    # Check if high churn
    high_churn_files = {f for f, _ in git_analysis.get('high_churn_files', [])}
    if file_path in high_churn_files:
        tokens.append("trait:high_churn")

    return tokens


# =============================================================================
# NOVELTY DETECTION
# =============================================================================

def compute_novelty_score(
    file_tokens: List[str],
    abstractions: List[Any],
    all_file_tokens: Dict[str, List[str]],
) -> float:
    """
    Compute novelty score for a file based on abstraction coverage.

    A file is novel/surprising if:
    1. Its tokens don't match well with any learned abstraction
    2. It contains unusual token combinations not seen in other files

    Returns:
        Novelty score in [0, 1] where 1 = highly novel/surprising
    """
    if not file_tokens:
        return 0.0

    file_token_set = set(file_tokens)

    # Score 1: Abstraction coverage (how well do tokens match abstractions?)
    abstraction_coverage = 0.0
    if abstractions:
        best_match = 0.0
        for abstraction in abstractions:
            source_nodes = set(abstraction.source_nodes)
            # Jaccard similarity
            intersection = len(file_token_set & source_nodes)
            union = len(file_token_set | source_nodes)
            if union > 0:
                similarity = intersection / union
                best_match = max(best_match, similarity)
        abstraction_coverage = best_match

    # Score 2: Token combination rarity
    combination_rarity = 1.0
    if all_file_tokens:
        similar_files = 0
        for other_path, other_tokens in all_file_tokens.items():
            other_set = set(other_tokens)
            if file_token_set and other_set:
                overlap = len(file_token_set & other_set) / len(file_token_set)
                if overlap > 0.5:
                    similar_files += 1
        combination_rarity = 1.0 - (similar_files / max(len(all_file_tokens), 1))

    # Score 3: Pattern diversity
    pattern_tokens = [t for t in file_tokens if t.startswith('pattern:')]
    pattern_count = len(set(pattern_tokens))
    pattern_diversity = min((pattern_count - 1) / 3.0, 1.0) if pattern_count > 0 else 0.0

    # Score 4: Cross-domain signal
    has_trait = any(t.startswith('trait:') for t in file_tokens)
    cross_domain = 1.0 if (has_trait and pattern_count >= 2) else 0.0

    # Combine scores
    novelty = (
        (1.0 - abstraction_coverage) * 0.25 +
        combination_rarity * 0.25 +
        pattern_diversity * 0.35 +
        cross_domain * 0.15
    )

    return min(1.0, max(0.0, novelty))


# =============================================================================
# PATTERN FEEDING
# =============================================================================

def feed_findings_to_mind(
    mind: WovenMind,
    findings: List[Dict[str, Any]],
    git_analysis: Dict[str, Any],
    novelty_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Feed audit findings to WovenMind for pattern learning.

    Feeds findings at multiple granularities:
    1. Full file patterns (specific but rarely repeat)
    2. Generic patterns without file names (can form abstractions)
    3. Individual pattern types (for tracking frequency)

    Returns statistics about what was learned.
    """
    stats = {
        "findings_fed": 0,
        "patterns_observed": 0,
        "surprising_inputs": [],
        "mode_switches": 0,
    }

    all_file_tokens: Dict[str, List[str]] = {}

    # Group findings by file
    findings_by_file = defaultdict(list)
    for f in findings:
        finding_id = f.get('id', '')
        if ':' in finding_id:
            file_path = finding_id.rsplit(':', 1)[0]
            findings_by_file[file_path].append(f)

    # Prepare file characteristics
    high_churn = {f for f, _ in git_analysis.get('high_churn_files', [])}

    # Feed each file's findings
    for file_path, file_findings in findings_by_file.items():
        specific_tokens = []
        generic_tokens = []

        # File context
        parts = file_path.split('/')
        for part in parts[:-1]:
            specific_tokens.append(f"dir:{part}")
            generic_tokens.append(f"dir:{part}")
        if parts:
            specific_tokens.append(f"file:{parts[-1]}")

        # File traits
        if file_path in high_churn:
            specific_tokens.append("trait:high_churn")
            generic_tokens.append("trait:high_churn")

        # All pattern types in this file
        for finding in file_findings:
            pattern = finding.get('pattern', 'unknown')
            normalized = pattern.replace(' ', '_').lower().rstrip(':')
            pattern_token = f"pattern:{normalized}"
            specific_tokens.append(pattern_token)
            generic_tokens.append(pattern_token)

        all_file_tokens[file_path] = generic_tokens.copy()

        # Train Hive on the text representation
        text = " ".join(specific_tokens)
        mind.train(text)

        # Process through full system
        mind.process(specific_tokens)
        stats["findings_fed"] += len(file_findings)

        # Observe generic pattern for Cortex abstraction
        if len(generic_tokens) >= 2:
            mind.observe_pattern(generic_tokens)
            stats["patterns_observed"] += 1

        # Observe individual pattern+directory combos
        for finding in file_findings:
            pattern = finding.get('pattern', 'unknown')
            normalized = pattern.replace(' ', '_').lower().rstrip(':')
            pattern_token = f"pattern:{normalized}"

            if parts and len(parts) > 1:
                combo = [f"dir:{parts[0]}", pattern_token]
                mind.observe_pattern(combo)

            if file_path in high_churn:
                combo = [pattern_token, "trait:high_churn"]
                mind.observe_pattern(combo)

    # Run consolidation
    mind.consolidate()

    # Detect surprising files using abstraction-based novelty
    abstractions = mind.cortex.get_abstractions()

    for file_path, tokens in all_file_tokens.items():
        novelty = compute_novelty_score(tokens, abstractions, all_file_tokens)

        if novelty > novelty_threshold:
            stats["surprising_inputs"].append({
                "file": file_path,
                "tokens": tokens,
                "surprise_magnitude": novelty,
                "mode": "NOVELTY",
            })

    return stats


# =============================================================================
# DISCOVERY EXTRACTION
# =============================================================================

def extract_discoveries(mind: WovenMind) -> Dict[str, Any]:
    """
    Extract what WovenMind has discovered.

    Returns:
        Dictionary with discovered patterns, abstractions, and insights.
    """
    discoveries = {
        "abstractions": [],
        "hive_stats": {},
        "cortex_stats": {},
        "consolidation_stats": {},
    }

    # Get formed abstractions
    abstractions = mind.cortex.get_abstractions()
    for abstraction in abstractions:
        source_patterns = list(abstraction.source_nodes)

        discoveries["abstractions"].append({
            "id": abstraction.id,
            "level": abstraction.level,
            "frequency": abstraction.frequency,
            "strength": abstraction.strength,
            "source_nodes": source_patterns,
            "interpretation": interpret_abstraction(source_patterns),
        })

    # Sort by strength
    discoveries["abstractions"].sort(key=lambda x: -x["strength"])

    # Get system stats
    full_stats = mind.get_stats()
    discoveries["hive_stats"] = full_stats.get("hive", {})
    discoveries["cortex_stats"] = full_stats.get("cortex", {})
    discoveries["consolidation_stats"] = full_stats.get("consolidation", {})

    return discoveries


def interpret_abstraction(source_nodes: List[str]) -> str:
    """
    Attempt to interpret what an abstraction means in human terms.
    """
    dirs = [n.split(':')[1] for n in source_nodes if n.startswith('dir:')]
    files = [n.split(':')[1] for n in source_nodes if n.startswith('file:')]
    patterns = [n.split(':')[1] for n in source_nodes if n.startswith('pattern:')]
    traits = [n.split(':')[1] for n in source_nodes if n.startswith('trait:')]

    parts = []

    if dirs:
        parts.append(f"in {'/'.join(dirs)}")
    if files:
        parts.append(f"files like {', '.join(files[:2])}")
    if patterns:
        parts.append(f"with patterns [{', '.join(patterns)}]")
    if traits:
        parts.append(f"having traits [{', '.join(traits)}]")

    if not parts:
        return "Unknown pattern cluster"

    return " ".join(parts)


# =============================================================================
# MAIN DISCOVERY CLASS
# =============================================================================

class WovenMindDiscovery:
    """
    Main class for WovenMind-based pattern discovery.

    Provides a clean interface for:
    - Running discovery sessions
    - Managing WovenMind state
    - Extracting and interpreting patterns
    """

    def __init__(
        self,
        config: Optional[DiscoveryConfig] = None,
        persistence: Optional[DiscoveryPersistence] = None,
    ):
        """
        Initialize the discovery system.

        Args:
            config: Configuration for WovenMind
            persistence: Backend for state persistence
        """
        self.config = config or DiscoveryConfig()
        self._persistence = persistence or FileDiscoveryPersistence()
        self._mind: Optional[WovenMind] = None
        self._session_num = 0
        self._total_findings = 0

    def load_or_create_mind(self) -> WovenMind:
        """Load existing mind state or create new."""
        state = self._persistence.load_mind_state()

        if state:
            self._mind = WovenMind.from_dict(state["mind"])
            self._session_num = state["metadata"].get("sessions", 0) + 1
            self._total_findings = state["metadata"].get("total_findings", 0)
        else:
            wm_config = WovenMindConfig(
                surprise_threshold=self.config.surprise_threshold,
                min_frequency=self.config.min_frequency,
                k_winners=self.config.k_winners,
                enable_auto_consolidation=self.config.enable_auto_consolidation,
            )
            self._mind = WovenMind(config=wm_config)
            self._session_num = 1
            self._total_findings = 0

        return self._mind

    @property
    def mind(self) -> Optional[WovenMind]:
        """Get the WovenMind instance."""
        return self._mind

    @property
    def session_num(self) -> int:
        """Get current session number."""
        return self._session_num

    def run_discovery(
        self,
        findings: List[Dict[str, Any]],
        git_analysis: Optional[Dict[str, Any]] = None,
    ) -> DiscoveryResult:
        """
        Run a discovery session on audit findings.

        Args:
            findings: List of audit findings
            git_analysis: Optional git analysis data

        Returns:
            DiscoveryResult with patterns and insights
        """
        if self._mind is None:
            self.load_or_create_mind()

        git_analysis = git_analysis or {}

        # Feed findings to mind
        stats = feed_findings_to_mind(
            self._mind,
            findings,
            git_analysis,
            novelty_threshold=self.config.novelty_threshold,
        )

        # Extract discoveries
        discoveries = extract_discoveries(self._mind)

        # Update totals
        self._total_findings += stats["findings_fed"]

        # Build result
        result = DiscoveryResult(
            findings_fed=stats["findings_fed"],
            patterns_observed=stats["patterns_observed"],
            surprising_files=stats["surprising_inputs"],
            abstractions=discoveries["abstractions"],
            session_num=self._session_num,
        )

        return result

    def consolidate(self) -> Dict[str, Any]:
        """Run memory consolidation cycle."""
        if self._mind is None:
            self.load_or_create_mind()

        result = self._mind.consolidate()
        return {
            "patterns_transferred": result.patterns_transferred,
            "abstractions_formed": result.abstractions_formed,
            "cycle_duration_ms": result.cycle_duration_ms,
        }

    def save_state(self) -> None:
        """Save current state to persistence."""
        if self._mind is not None:
            self._persistence.save_mind_state(self._mind, {
                "sessions": self._session_num,
                "total_findings": self._total_findings,
            })

    def reset(self) -> None:
        """Clear all state and start fresh."""
        self._persistence.clear_state()
        self._mind = None
        self._session_num = 0
        self._total_findings = 0

    def get_mind_stats(self) -> Dict[str, Any]:
        """Get current mind statistics."""
        if self._mind is None:
            return {}
        return self._mind.get_stats()

    def get_abstractions(self) -> List[Dict[str, Any]]:
        """Get formed abstractions."""
        if self._mind is None:
            return []
        discoveries = extract_discoveries(self._mind)
        return discoveries["abstractions"]


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_discovery_report(
    result: DiscoveryResult,
    mind: Optional[WovenMind] = None,
) -> str:
    """Generate a human-readable discovery report."""
    lines = []
    lines.append("=" * 70)
    lines.append("  WOVEN AUDIT DISCOVERY - Pattern Discovery Report")
    lines.append("=" * 70)

    # Session info
    lines.append(f"\n[Session Info]")
    lines.append(f"  Session number: {result.session_num}")
    lines.append(f"  Findings processed: {result.findings_fed}")
    lines.append(f"  Patterns observed: {result.patterns_observed}")

    if mind:
        lines.append(f"  Current mode: {mind.get_current_mode().name}")

    # Surprising files
    if result.surprising_files:
        lines.append(f"\n[Surprising Files - Deviate from Learned Patterns]")
        lines.append(f"  Found {len(result.surprising_files)} files with unusual combinations:")
        for s in sorted(result.surprising_files, key=lambda x: -x['surprise_magnitude'])[:10]:
            lines.append(f"    ! {s['file']}")
            lines.append(f"      Surprise: {s['surprise_magnitude']:.2f}")
            patterns = [t for t in s['tokens'] if t.startswith('pattern:')]
            if patterns:
                lines.append(f"      Patterns: {', '.join(patterns)}")
    else:
        lines.append(f"\n[Surprising Files]")
        lines.append(f"  None detected yet (need more data to establish baseline)")

    # Discovered abstractions
    if result.abstractions:
        lines.append(f"\n[Discovered Abstractions - Emergent Patterns]")
        lines.append(f"  WovenMind has formed {len(result.abstractions)} abstractions:")
        for a in result.abstractions[:10]:
            lines.append(f"\n  [{a['id']}] (strength: {a['strength']:.2f}, seen {a['frequency']}x)")
            lines.append(f"    → {a['interpretation']}")
    else:
        lines.append(f"\n[Discovered Abstractions]")
        lines.append(f"  None yet - need more data (min 3 observations per pattern)")

    lines.append("\n" + "=" * 70)
    lines.append("  This is EXPERIMENTAL - treat as hints to investigate, not ground truth")
    lines.append("=" * 70)

    return "\n".join(lines)
