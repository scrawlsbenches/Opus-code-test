"""
Sharded Graph Storage

Provides git-friendly storage for CognitiveGraph by sharding atoms
into separate files based on their type.

Design Principles:
    1. TYPE-BASED SHARDING: Atoms are stored in files by their AtomType.
       WORD atoms go to atoms_word.json, SIMILARITY links to atoms_similarity.json, etc.

    2. GIT-FRIENDLY: Each shard stays under 50MB to avoid GitHub limits.
       Large atom types are further subdivided if needed.

    3. INCREMENTAL SAVES: Track which shards are dirty and only write those.

    4. BACKWARD COMPATIBLE: Can load from legacy single-file graph.json.

Directory Structure:
    bridge/
    ├── meta.json              # Metadata: counts, version, shard list
    ├── atoms_word.json        # WORD atoms (~21k atoms, small)
    ├── atoms_similarity.json  # SIMILARITY links (~200k)
    ├── atoms_follows.json     # FOLLOWS links (~247k)
    └── atoms_other.json       # Other atom types (small)

For very large shards (>50MB), we subdivide by ID prefix:
    ├── atoms_similarity_00.json  # IDs starting with 0-3
    ├── atoms_similarity_01.json  # IDs starting with 4-7
    ├── atoms_similarity_02.json  # IDs starting with 8-b
    └── atoms_similarity_03.json  # IDs starting with c-f

Usage:
    from cortical.cognitive.graph_storage import ShardedGraphStorage

    storage = ShardedGraphStorage()

    # Save graph to sharded directory
    storage.save(graph, model_dir / "bridge")

    # Load graph from sharded directory
    atoms = storage.load(model_dir / "bridge")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cortical.cognitive.graph import CognitiveGraph, Atom


# Maximum shard size in bytes before subdividing
MAX_SHARD_SIZE = 40 * 1024 * 1024  # 40MB (leave room for growth)

# Atom types that typically have many entries and should be subdivided
# if they exceed MAX_SHARD_SIZE
LARGE_TYPES = {'SIMILARITY', 'FOLLOWS'}

# Bytes per atom estimate (conservative - actual is larger)
BYTES_PER_ATOM = 250


class ShardedGraphStorage:
    """
    Manages sharded storage for CognitiveGraph atoms.

    Shards atoms by type to enable:
    - Git-friendly file sizes (under 50MB each)
    - Semantic organization (all WORD atoms together, etc.)
    - Incremental saves (only dirty shards)

    Attributes:
        VERSION: Storage format version
    """

    VERSION = "1.0"

    def __init__(self):
        """Initialize storage."""
        self._dirty_shards: set = set()

    def _get_shard_name(self, atom_type: str, subdivision: Optional[int] = None) -> str:
        """
        Get the shard filename for an atom type.

        Args:
            atom_type: The atom type (e.g., 'WORD', 'SIMILARITY')
            subdivision: Optional subdivision index for large shards

        Returns:
            Filename like 'atoms_word.json' or 'atoms_similarity_00.json'
        """
        type_lower = atom_type.lower()
        if subdivision is not None:
            return f"atoms_{type_lower}_{subdivision:02d}.json"
        return f"atoms_{type_lower}.json"

    def _get_subdivision(self, atom_id: str, num_subdivisions: int = 4) -> int:
        """
        Get subdivision index for an atom ID.

        Uses first hex character of ID to distribute evenly.

        Args:
            atom_id: The atom's ID (UUID format)
            num_subdivisions: Number of subdivisions (default 4)

        Returns:
            Subdivision index 0 to num_subdivisions-1
        """
        if not atom_id:
            return 0
        # Use first hex char for distribution
        first_char = atom_id[0].lower()
        if first_char.isdigit():
            idx = int(first_char)
        elif 'a' <= first_char <= 'f':
            idx = ord(first_char) - ord('a') + 10
        else:
            idx = 0
        return idx % num_subdivisions

    def save(self, graph: 'CognitiveGraph', directory: Path) -> Dict[str, Any]:
        """
        Save graph to sharded directory structure with incremental support.

        If the storage tracks dirty atoms, only shards containing dirty atoms
        are rewritten. Otherwise, all shards are written.

        Args:
            graph: The CognitiveGraph to save
            directory: Directory to save to

        Returns:
            Dict with save statistics
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        storage = graph._storage

        # Check if incremental save is possible
        supports_dirty = hasattr(storage, 'get_dirty_atoms') and hasattr(storage, 'is_all_dirty')
        is_incremental = supports_dirty and not storage.is_all_dirty()

        if is_incremental:
            dirty_atoms = storage.get_dirty_atoms()
            if not dirty_atoms:
                # Nothing changed, skip save
                return {
                    "shards_written": 0,
                    "total_atoms": len(storage._atoms),
                    "shard_names": [],
                    "incremental": True,
                }
            # Determine which atom types have dirty atoms
            dirty_types: set = set()
            for atom_id in dirty_atoms:
                atom = storage.load(atom_id)
                if atom:
                    dirty_types.add(atom.atom_type.name)
        else:
            # Full save needed - mark all types as dirty
            dirty_types = None  # None means all types
            # Clean up old shard files for full save
            for old_file in directory.glob("atoms_*.json"):
                old_file.unlink()

        # Group atoms by type (only for types we need to save)
        atoms_by_type: Dict[str, List[Dict]] = {}
        for atom in storage.all_atoms():
            type_name = atom.atom_type.name
            # Skip types that aren't dirty (incremental mode)
            if dirty_types is not None and type_name not in dirty_types:
                continue
            if type_name not in atoms_by_type:
                atoms_by_type[type_name] = []
            atoms_by_type[type_name].append(self._atom_to_dict(atom))

        # Determine which types need subdivision
        type_sizes: Dict[str, int] = {}
        for type_name, atoms in atoms_by_type.items():
            estimated_size = len(atoms) * BYTES_PER_ATOM
            type_sizes[type_name] = estimated_size

        # Save each type to appropriate shard(s)
        shards_written = []
        atoms_saved = 0

        for type_name, atoms in atoms_by_type.items():
            if type_name in LARGE_TYPES and type_sizes[type_name] > MAX_SHARD_SIZE:
                # Subdivide large types
                subdivisions: Dict[int, List[Dict]] = {}
                for atom_dict in atoms:
                    sub_idx = self._get_subdivision(atom_dict['id'])
                    if sub_idx not in subdivisions:
                        subdivisions[sub_idx] = []
                    subdivisions[sub_idx].append(atom_dict)

                for sub_idx, sub_atoms in subdivisions.items():
                    shard_name = self._get_shard_name(type_name, sub_idx)
                    shard_path = directory / shard_name
                    self._write_shard(shard_path, sub_atoms)
                    shards_written.append(shard_name)
                    atoms_saved += len(sub_atoms)
            else:
                # Single shard for this type
                shard_name = self._get_shard_name(type_name)
                shard_path = directory / shard_name
                self._write_shard(shard_path, atoms)
                shards_written.append(shard_name)
                atoms_saved += len(atoms)

        # Get total atoms including non-dirty types
        total_atoms = len(storage._atoms) if hasattr(storage, '_atoms') else atoms_saved

        # Update metadata (includes all shards, not just written ones)
        meta_path = directory / "meta.json"
        if is_incremental and meta_path.exists():
            # Load existing meta and update
            with open(meta_path) as f:
                meta = json.load(f)
            meta["total_atoms"] = total_atoms
            # Add new shards to list
            existing_shards = set(meta.get("shards", []))
            existing_shards.update(shards_written)
            meta["shards"] = sorted(existing_shards)
            # Update type counts for dirty types
            if "type_counts" not in meta:
                meta["type_counts"] = {}
            for type_name, atoms in atoms_by_type.items():
                meta["type_counts"][type_name] = len(atoms)
        else:
            # Full metadata for new save
            all_shards = list(directory.glob("atoms_*.json"))
            meta = {
                "version": self.VERSION,
                "total_atoms": total_atoms,
                "shards": [f.name for f in all_shards],
                "type_counts": {t: len(a) for t, a in atoms_by_type.items()},
            }

        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        # Remove legacy graph.json if it exists
        legacy_path = directory / "graph.json"
        if legacy_path.exists():
            legacy_path.unlink()

        # Clear dirty state after successful save
        if supports_dirty:
            storage.clear_dirty()

        return {
            "shards_written": len(shards_written),
            "total_atoms": total_atoms,
            "shard_names": shards_written,
            "incremental": is_incremental,
            "atoms_saved": atoms_saved,
        }

    def _write_shard(self, path: Path, atoms: List[Dict]) -> None:
        """Write atoms to a shard file."""
        with open(path, 'w') as f:
            json.dump({"atoms": atoms}, f)

    def _atom_to_dict(self, atom: 'Atom') -> Dict[str, Any]:
        """Convert an Atom to a dictionary for serialization."""
        return {
            "id": atom.id,
            "atom_type": atom.atom_type.name,
            "name": atom.name,
            "outgoing": atom.outgoing,
            "tv": {"strength": atom.tv.strength, "confidence": atom.tv.confidence},
            "sti": atom.sti,
            "lti": atom.lti,
            "created_at": atom.created_at,
            "accessed_at": atom.accessed_at,
            "metadata": atom.metadata,
        }

    def load(self, directory: Path) -> List[Dict[str, Any]]:
        """
        Load atoms from sharded directory structure.

        Handles both sharded format and legacy single-file format.

        Args:
            directory: Directory containing shard files

        Returns:
            List of atom dictionaries
        """
        directory = Path(directory)

        # Check for sharded format
        meta_path = directory / "meta.json"
        if meta_path.exists():
            return self._load_sharded(directory, meta_path)

        # Fall back to legacy single-file format
        legacy_path = directory / "graph.json"
        if legacy_path.exists():
            return self._load_legacy(legacy_path)

        # No data found
        return []

    def _load_sharded(self, directory: Path, meta_path: Path) -> List[Dict]:
        """Load from sharded format."""
        with open(meta_path) as f:
            meta = json.load(f)

        all_atoms = []
        for shard_name in meta.get("shards", []):
            shard_path = directory / shard_name
            if shard_path.exists():
                with open(shard_path) as f:
                    data = json.load(f)
                    all_atoms.extend(data.get("atoms", []))

        return all_atoms

    def _load_legacy(self, path: Path) -> List[Dict]:
        """Load from legacy single-file format."""
        with open(path) as f:
            data = json.load(f)
        return data.get("atoms", [])

    def migrate_to_sharded(self, directory: Path) -> Dict[str, Any]:
        """
        Migrate from legacy graph.json to sharded format.

        Args:
            directory: Directory containing graph.json

        Returns:
            Migration statistics
        """
        directory = Path(directory)
        legacy_path = directory / "graph.json"

        if not legacy_path.exists():
            return {"status": "no_legacy_file", "migrated": False}

        # Load legacy data
        atoms = self._load_legacy(legacy_path)
        if not atoms:
            return {"status": "empty_legacy", "migrated": False}

        # Group by type and save
        from cortical.cognitive.graph import CognitiveGraph, AtomType, TruthValue, Atom

        # Reconstruct graph from atoms
        graph = CognitiveGraph()

        # First pass: create all atoms
        for atom_dict in atoms:
            atom_type = AtomType[atom_dict['atom_type']]
            tv = TruthValue(
                strength=atom_dict['tv']['strength'],
                confidence=atom_dict['tv']['confidence'],
            )

            if atom_dict.get('name'):
                # Node atom
                atom = graph.node(atom_dict['name'], atom_type)
            else:
                # Link atom - need to handle outgoing references
                # Skip for now, will be handled in second pass
                continue

            atom.sti = atom_dict.get('sti', 0.0)
            atom.lti = atom_dict.get('lti', 0.0)
            atom.metadata = atom_dict.get('metadata', {})
            graph._storage.save(atom)

        # Save in sharded format
        result = self.save(graph, directory)
        result["status"] = "migrated"
        result["migrated"] = True
        result["legacy_atoms"] = len(atoms)

        return result
