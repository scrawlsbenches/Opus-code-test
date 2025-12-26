"""
Pattern Generator - Transform extracted patterns into training data.

Generates:
- Q&A pairs (What/Where/How questions)
- Completion patterns (code/text completions)
- Association pairs (term relationships)
- Explanation pairs (code → description)

This is the "knowledge crystallization" step that transforms raw
extractions into patterns optimized for SLM training.
"""

import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Iterator
import json
from datetime import datetime

from .code_extractor import CodePattern, FunctionPattern, ClassPattern
from .doc_extractor import DocPattern, SectionPattern
from .meta_extractor import MetaPattern, TaskPattern, CommitPattern


@dataclass
class TrainingPattern:
    """A single training pattern for the SLM."""
    pattern_type: str  # 'qa', 'completion', 'association', 'explanation'
    input_text: str
    target_text: str
    source_file: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_training_format(self) -> str:
        """Format for direct SLM training (input → target)."""
        return f"{self.input_text} {self.target_text}"


class PatternGenerator:
    """
    Generate training patterns from extracted code, docs, and metadata.

    Usage:
        generator = PatternGenerator()
        patterns = generator.generate_all(
            code_patterns=code_extractor.get_all_patterns(),
            doc_patterns=doc_extractor.get_all_patterns(),
            meta_patterns=meta_extractor.extract_all(),
        )

        # Save for training
        generator.save_corpus('benchmarks/codebase_slm/corpus/training.jsonl')
    """

    # Question templates for Q&A generation
    QA_TEMPLATES = {
        'function_location': [
            ("Where is {name} defined?", "{file_path}"),
            ("What file contains {name}?", "{file_path}"),
            ("Find {name} function", "{file_path}:{line_number}"),
        ],
        'function_purpose': [
            ("What does {name} do?", "{docstring}"),
            ("Explain {name}", "{docstring}"),
            ("Purpose of {name}?", "{docstring}"),
        ],
        'class_info': [
            ("What is {name}?", "{docstring}"),
            ("Describe the {name} class", "{docstring}"),
            ("What methods does {name} have?", "{methods}"),
        ],
        'task_info': [
            ("What is task {id}?", "{title}"),
            ("Status of {id}?", "{status}"),
            ("Task {title}", "{status} - {description}"),
        ],
        'commit_info': [
            ("What did commit {hash} change?", "{message}"),
            ("Files changed in {hash}?", "{files}"),
        ],
    }

    # Completion templates
    COMPLETION_TEMPLATES = {
        'import': [
            ("from {module} import", "{names}"),
            ("import {module}", ""),
        ],
        'function_call': [
            ("{class_name}.", "{method_name}"),
            ("processor.", "{method_name}"),
        ],
        'command': [
            ("python scripts/got_utils.py task", "create"),
            ("python scripts/got_utils.py", "{subcommand}"),
            ("make test-", "quick"),
        ],
    }

    def __init__(self, seed: int = 42):
        """Initialize generator with optional random seed."""
        self.seed = seed
        random.seed(seed)
        self._patterns: List[TrainingPattern] = []

    def _generate_qa_from_functions(
        self,
        code_patterns: List[CodePattern]
    ) -> List[TrainingPattern]:
        """Generate Q&A pairs from function definitions."""
        patterns = []

        for cp in code_patterns:
            for func in cp.functions:
                # Skip private/magic methods
                if func.name.startswith('_'):
                    continue

                # Location questions
                for q_template, a_template in self.QA_TEMPLATES['function_location']:
                    q = q_template.format(name=func.name)
                    a = a_template.format(
                        file_path=func.file_path,
                        line_number=func.line_number
                    )
                    patterns.append(TrainingPattern(
                        pattern_type='qa',
                        input_text=q,
                        target_text=a,
                        source_file=func.file_path,
                        metadata={'function': func.name}
                    ))

                # Purpose questions (only if docstring exists)
                if func.docstring:
                    # Truncate docstring to first sentence
                    docstring = func.docstring.split('.')[0] + '.'
                    for q_template, a_template in self.QA_TEMPLATES['function_purpose']:
                        q = q_template.format(name=func.name)
                        a = a_template.format(docstring=docstring)
                        patterns.append(TrainingPattern(
                            pattern_type='qa',
                            input_text=q,
                            target_text=a,
                            source_file=func.file_path,
                            confidence=0.9,  # Slightly lower for generated text
                            metadata={'function': func.name}
                        ))

        return patterns

    def _generate_qa_from_classes(
        self,
        code_patterns: List[CodePattern]
    ) -> List[TrainingPattern]:
        """Generate Q&A pairs from class definitions."""
        patterns = []

        for cp in code_patterns:
            for cls in cp.classes:
                # Skip private classes
                if cls.name.startswith('_'):
                    continue

                # Class info questions
                for q_template, a_template in self.QA_TEMPLATES['class_info']:
                    if '{docstring}' in a_template and not cls.docstring:
                        continue
                    if '{methods}' in a_template and not cls.methods:
                        continue

                    q = q_template.format(name=cls.name)

                    docstring = ''
                    if cls.docstring:
                        docstring = cls.docstring.split('.')[0] + '.'

                    methods = ', '.join(m for m in cls.methods if not m.startswith('_'))

                    a = a_template.format(
                        docstring=docstring,
                        methods=methods
                    )

                    patterns.append(TrainingPattern(
                        pattern_type='qa',
                        input_text=q,
                        target_text=a,
                        source_file=cls.file_path,
                        metadata={'class': cls.name}
                    ))

        return patterns

    def _generate_qa_from_docs(
        self,
        doc_patterns: List[DocPattern]
    ) -> List[TrainingPattern]:
        """Generate Q&A pairs from documentation."""
        patterns = []

        for dp in doc_patterns:
            # Generate from sections
            for section in dp.sections:
                if len(section.content) < 50:  # Skip short sections
                    continue

                # Create Q&A from section title and content
                q = f"What is {section.title}?"
                a = section.content[:200] + ('...' if len(section.content) > 200 else '')

                patterns.append(TrainingPattern(
                    pattern_type='qa',
                    input_text=q,
                    target_text=a,
                    source_file=dp.file_path,
                    confidence=0.85,
                    metadata={'section': section.title}
                ))

            # Generate from code blocks with context
            for cb in dp.code_blocks:
                if cb.context and cb.language in ('python', 'bash', 'shell'):
                    q = f"How to {cb.context.lower().rstrip(':.?')}?"
                    a = f"```{cb.language}\n{cb.code[:200]}\n```"

                    patterns.append(TrainingPattern(
                        pattern_type='qa',
                        input_text=q,
                        target_text=a,
                        source_file=dp.file_path,
                        confidence=0.8,
                        metadata={'language': cb.language}
                    ))

        return patterns

    def _generate_qa_from_meta(
        self,
        meta_patterns: MetaPattern
    ) -> List[TrainingPattern]:
        """Generate Q&A pairs from metadata."""
        patterns = []

        # From tasks
        for task in meta_patterns.tasks:
            patterns.append(TrainingPattern(
                pattern_type='qa',
                input_text=f"What is {task.id}?",
                target_text=task.title,
                source_file='.got/entities',
                metadata={'task_id': task.id}
            ))

            if task.description:
                patterns.append(TrainingPattern(
                    pattern_type='qa',
                    input_text=f"Describe task {task.title[:50]}",
                    target_text=task.description[:200],
                    source_file='.got/entities',
                    confidence=0.9,
                    metadata={'task_id': task.id}
                ))

        # From commits
        for commit in meta_patterns.commits:
            files = ', '.join(commit.files_changed[:3])
            if len(commit.files_changed) > 3:
                files += f' (+{len(commit.files_changed) - 3} more)'

            patterns.append(TrainingPattern(
                pattern_type='qa',
                input_text=f"What changed in {commit.hash}?",
                target_text=commit.message,
                source_file='.git-ml',
                metadata={'commit': commit.hash}
            ))

        return patterns

    def _generate_completions(
        self,
        code_patterns: List[CodePattern]
    ) -> List[TrainingPattern]:
        """Generate completion patterns from code."""
        patterns = []

        for cp in code_patterns:
            # Import completions
            for imp in cp.imports:
                if imp.is_from_import:
                    names = ', '.join(imp.names[:5])
                    patterns.append(TrainingPattern(
                        pattern_type='completion',
                        input_text=f"from {imp.module} import",
                        target_text=names,
                        source_file=imp.file_path,
                        metadata={'import': imp.module}
                    ))

            # Method completions for classes
            for cls in cp.classes:
                public_methods = [m for m in cls.methods if not m.startswith('_')]
                for method in public_methods[:10]:  # Limit per class
                    patterns.append(TrainingPattern(
                        pattern_type='completion',
                        input_text=f"{cls.name}.",
                        target_text=method,
                        source_file=cls.file_path,
                        metadata={'class': cls.name, 'method': method}
                    ))

        return patterns

    def _generate_associations(
        self,
        code_patterns: List[CodePattern],
        doc_patterns: List[DocPattern]
    ) -> List[TrainingPattern]:
        """Generate association patterns (term → related terms)."""
        patterns = []

        # Build term co-occurrence from imports
        module_terms: Dict[str, List[str]] = {}
        for cp in code_patterns:
            for imp in cp.imports:
                if imp.module not in module_terms:
                    module_terms[imp.module] = []
                module_terms[imp.module].extend(imp.names)

        for module, terms in module_terms.items():
            if len(terms) >= 2:
                unique_terms = list(set(terms))[:10]
                patterns.append(TrainingPattern(
                    pattern_type='association',
                    input_text=module,
                    target_text=', '.join(unique_terms),
                    source_file='associations',
                    metadata={'type': 'module_exports'}
                ))

        # Build term associations from documentation
        for dp in doc_patterns:
            if dp.title:
                # Associate title with section names
                section_titles = [s.title for s in dp.sections[:5]]
                if section_titles:
                    patterns.append(TrainingPattern(
                        pattern_type='association',
                        input_text=dp.title,
                        target_text=', '.join(section_titles),
                        source_file=dp.file_path,
                        metadata={'type': 'doc_structure'}
                    ))

        return patterns

    def _generate_explanations(
        self,
        code_patterns: List[CodePattern]
    ) -> List[TrainingPattern]:
        """Generate code → explanation patterns."""
        patterns = []

        for cp in code_patterns:
            # Function signature → docstring
            for func in cp.functions:
                if func.docstring and not func.name.startswith('_'):
                    patterns.append(TrainingPattern(
                        pattern_type='explanation',
                        input_text=f"def {func.signature}",
                        target_text=func.docstring.split('.')[0] + '.',
                        source_file=func.file_path,
                        metadata={'function': func.name}
                    ))

            # Class → docstring
            for cls in cp.classes:
                if cls.docstring and not cls.name.startswith('_'):
                    patterns.append(TrainingPattern(
                        pattern_type='explanation',
                        input_text=f"class {cls.name}",
                        target_text=cls.docstring.split('.')[0] + '.',
                        source_file=cls.file_path,
                        metadata={'class': cls.name}
                    ))

        return patterns

    def generate_all(
        self,
        code_patterns: List[CodePattern],
        doc_patterns: List[DocPattern],
        meta_patterns: Optional[MetaPattern] = None,
    ) -> List[TrainingPattern]:
        """
        Generate all training patterns.

        Args:
            code_patterns: Extracted code patterns
            doc_patterns: Extracted documentation patterns
            meta_patterns: Extracted metadata patterns (optional)

        Returns:
            List of training patterns
        """
        all_patterns = []

        # Q&A patterns
        all_patterns.extend(self._generate_qa_from_functions(code_patterns))
        all_patterns.extend(self._generate_qa_from_classes(code_patterns))
        all_patterns.extend(self._generate_qa_from_docs(doc_patterns))

        if meta_patterns:
            all_patterns.extend(self._generate_qa_from_meta(meta_patterns))

        # Completion patterns
        all_patterns.extend(self._generate_completions(code_patterns))

        # Association patterns
        all_patterns.extend(self._generate_associations(code_patterns, doc_patterns))

        # Explanation patterns
        all_patterns.extend(self._generate_explanations(code_patterns))

        self._patterns = all_patterns
        return all_patterns

    def get_patterns_by_type(self, pattern_type: str) -> List[TrainingPattern]:
        """Get patterns of a specific type."""
        return [p for p in self._patterns if p.pattern_type == pattern_type]

    def get_statistics(self) -> Dict[str, int]:
        """Get generation statistics."""
        stats = {
            'total': len(self._patterns),
            'qa': len(self.get_patterns_by_type('qa')),
            'completion': len(self.get_patterns_by_type('completion')),
            'association': len(self.get_patterns_by_type('association')),
            'explanation': len(self.get_patterns_by_type('explanation')),
        }
        return stats

    def save_corpus(
        self,
        output_path: Path,
        format: str = 'jsonl'
    ) -> None:
        """
        Save training corpus to file.

        Args:
            output_path: Output file path
            format: 'jsonl' or 'text'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'jsonl':
            with open(output_path, 'w') as f:
                for pattern in self._patterns:
                    f.write(json.dumps(pattern.to_dict()) + '\n')
        else:  # text format for direct training
            with open(output_path, 'w') as f:
                for pattern in self._patterns:
                    f.write(pattern.to_training_format() + '\n')

    def sample_patterns(self, n: int = 10) -> List[TrainingPattern]:
        """Get a random sample of patterns for inspection."""
        return random.sample(self._patterns, min(n, len(self._patterns)))


if __name__ == '__main__':
    from .code_extractor import CodeExtractor
    from .doc_extractor import DocExtractor
    from .meta_extractor import MetaExtractor

    print("Testing pattern generator...")

    # Quick extraction
    code_ext = CodeExtractor()
    doc_ext = DocExtractor()
    meta_ext = MetaExtractor()

    # Extract from small subset
    code_patterns = code_ext.extract_all(Path('cortical/'))
    doc_patterns = doc_ext.extract_all(Path('docs/'))
    meta_patterns = meta_ext.extract_all(commit_limit=100)

    print(f"Code: {len(code_patterns)} files")
    print(f"Docs: {len(doc_patterns)} files")
    print(f"Meta: {meta_ext.get_statistics()}")

    # Generate patterns
    generator = PatternGenerator()
    patterns = generator.generate_all(code_patterns, doc_patterns, meta_patterns)

    print(f"\nGenerated patterns: {generator.get_statistics()}")

    # Show samples
    print("\n=== Sample Patterns ===")
    for p in generator.sample_patterns(5):
        print(f"\n[{p.pattern_type}] {p.input_text}")
        print(f"  → {p.target_text[:100]}...")
