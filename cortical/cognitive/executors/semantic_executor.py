"""
Semantic Executor: Document retrieval via TF-IDF.

Retrieves relevant documents from the trained corpus:
- Loads training manifest to find documents
- Uses TF-IDF similarity to rank documents
- Returns content excerpts for answers
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .protocol import BaseExecutor, ExecutionResult

# Import the QueryIntent from unified_query
from cortical.cognitive.unified_query import QueryIntent


class SemanticExecutor(BaseExecutor):
    """
    Executes semantic queries via document retrieval.

    Capabilities:
    - Find documents relevant to query concepts
    - Return content excerpts
    - Rank by TF-IDF similarity
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        samples_dir: Optional[Path] = None,
    ):
        """
        Initialize semantic executor.

        Args:
            model_dir: Directory containing training manifest
            samples_dir: Base directory for document content
        """
        self._model_dir = model_dir or Path("models/cognitive_agent")
        self._samples_dir = samples_dir or Path("samples")
        self._manifest: Optional[Dict[str, Any]] = None
        self._idf_weights: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def manifest(self) -> Dict[str, Any]:
        """Get or load the training manifest."""
        if self._manifest is None:
            manifest_path = self._model_dir / "training_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    self._manifest = json.load(f)
            else:
                self._manifest = {"documents": {}}
        return self._manifest

    def execute(self, query: QueryIntent) -> ExecutionResult:
        """
        Execute a semantic query.

        Args:
            query: QueryIntent from the router

        Returns:
            ExecutionResult with relevant documents and excerpts
        """
        concepts = query.concepts
        raw_question = query.raw_question

        if not concepts:
            return ExecutionResult(
                items=[],
                confidence=0.0,
                source=self.name,
                explanation="No concepts extracted from query.",
            )

        # Find relevant documents
        scored_docs = self._find_relevant_documents(concepts)

        if not scored_docs:
            return ExecutionResult(
                items=[],
                confidence=0.3,
                source=self.name,
                explanation=f"No documents found matching concepts: {', '.join(concepts)}",
            )

        # Get content excerpts for top documents
        items = []
        for doc_path, score in scored_docs[:10]:
            content = self._get_document_content(doc_path)
            excerpt = self._extract_relevant_excerpt(content, concepts, raw_question)

            items.append({
                "doc_id": doc_path,
                "score": score,
                "excerpt": excerpt,
                "word_count": self.manifest.get("documents", {}).get(doc_path, {}).get("word_count", 0),
            })

        # Calculate confidence based on top score
        top_score = scored_docs[0][1] if scored_docs else 0
        confidence = min(0.9, 0.3 + top_score * 0.6)

        return ExecutionResult(
            items=items,
            confidence=confidence,
            source=self.name,
            explanation=f"Found {len(items)} relevant documents for: {raw_question}",
            metadata={
                "concepts": concepts,
                "question_type": query.question_type,
                "total_docs_searched": len(self.manifest.get("documents", {})),
            }
        )

    def _find_relevant_documents(
        self,
        concepts: List[str],
        max_results: int = 10
    ) -> List[tuple]:
        """
        Find documents matching the query concepts.

        Uses simple TF-IDF-like scoring based on concept matches.

        Args:
            concepts: List of concept words to search for
            max_results: Maximum number of documents to return

        Returns:
            List of (doc_path, score) tuples, sorted by score descending
        """
        documents = self.manifest.get("documents", {})

        # Score each document by concept match
        doc_scores: Dict[str, float] = {}

        for doc_path, doc_info in documents.items():
            # Score based on path match
            path_lower = doc_path.lower()
            score = 0.0

            for concept in concepts:
                concept_lower = concept.lower()

                # Exact word match in path
                if concept_lower in path_lower:
                    score += 2.0

                # Check path components
                path_parts = re.split(r'[/_\-\.]', path_lower)
                for part in path_parts:
                    if concept_lower == part:
                        score += 3.0
                    elif concept_lower in part:
                        score += 1.0

            # Boost for larger documents (more likely to have detail)
            word_count = doc_info.get("word_count", 0)
            if word_count > 500:
                score *= 1.2

            if score > 0:
                doc_scores[doc_path] = score

        # Sort by score descending
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_docs[:max_results]

    def _get_document_content(self, doc_path: str) -> str:
        """
        Get the content of a document.

        Args:
            doc_path: Relative path from manifest

        Returns:
            Document content or empty string
        """
        # Try samples directory first
        full_path = self._samples_dir / doc_path

        if not full_path.exists():
            # Try other common locations
            for base in [Path("."), Path("docs"), Path("cortical")]:
                alt_path = base / doc_path
                if alt_path.exists():
                    full_path = alt_path
                    break

        if full_path.exists():
            try:
                return full_path.read_text()
            except Exception:
                return ""

        return ""

    def _extract_relevant_excerpt(
        self,
        content: str,
        concepts: List[str],
        question: str,
        max_length: int = 500
    ) -> str:
        """
        Extract the most relevant excerpt from document content.

        Args:
            content: Full document content
            concepts: Query concepts to match
            question: Original question for context
            max_length: Maximum excerpt length

        Returns:
            Relevant excerpt from the document
        """
        if not content:
            return "(Content not available)"

        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)

        # Score paragraphs by concept density
        scored_paragraphs = []
        for para in paragraphs:
            if len(para.strip()) < 20:
                continue

            para_lower = para.lower()
            score = 0

            for concept in concepts:
                concept_lower = concept.lower()
                # Count occurrences
                count = para_lower.count(concept_lower)
                score += count * 2

                # Bonus for being at start of paragraph
                if para_lower.strip().startswith(concept_lower):
                    score += 3

            if score > 0:
                scored_paragraphs.append((para, score))

        # Sort by score and take best
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)

        if scored_paragraphs:
            best_para = scored_paragraphs[0][0]
            # Truncate if needed
            if len(best_para) > max_length:
                best_para = best_para[:max_length] + "..."
            return best_para.strip()

        # Fall back to first substantial paragraph
        for para in paragraphs:
            if len(para.strip()) > 50:
                if len(para) > max_length:
                    para = para[:max_length] + "..."
                return para.strip()

        return content[:max_length] + "..." if len(content) > max_length else content

    def format_result(self, result: ExecutionResult) -> str:
        """Format semantic results as natural language."""
        if result.is_empty:
            return result.explanation or "No relevant documents found."

        lines = []
        if result.explanation:
            lines.append(result.explanation)
            lines.append("")

        for i, item in enumerate(result.items[:5], 1):
            doc_id = item.get("doc_id", "unknown")
            score = item.get("score", 0)
            excerpt = item.get("excerpt", "")

            lines.append(f"{i}. {doc_id} (relevance: {score:.1f})")

            if excerpt:
                # Indent excerpt
                excerpt_lines = excerpt.split("\n")
                for line in excerpt_lines[:3]:
                    lines.append(f"   {line[:80]}")
                if len(excerpt_lines) > 3:
                    lines.append("   ...")

            lines.append("")

        if len(result.items) > 5:
            lines.append(f"... and {len(result.items) - 5} more documents")

        return "\n".join(lines)
