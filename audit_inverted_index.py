"""
AuditInvertedIndex - Inverted index for audit finding search.

Implements O(1) term lookup with position tracking for phrase search.
Case-insensitive search with sorted results.
"""

from typing import Dict, List, Set, Tuple


class AuditInvertedIndex:
    """
    Inverted index for searching audit findings.

    Tracks term positions to enable phrase search (consecutive term matching).
    All searches are case-insensitive.
    """

    def __init__(self):
        """Initialize empty index."""
        # Map: term -> finding_id -> list of positions
        # This enables O(1) term lookup
        self._term_to_findings: Dict[str, Dict[str, List[int]]] = {}

        # Map: finding_id -> set of terms
        # Used for efficient finding removal (cleanup all term references)
        self._finding_terms: Dict[str, Set[str]] = {}

    def add(self, term: str, finding_id: str, position: int) -> None:
        """
        Add a term occurrence at a specific position in a finding.

        Args:
            term: The term to index (will be normalized to lowercase)
            finding_id: Identifier for the finding
            position: Zero-based position of term in the finding
        """
        # Normalize term to lowercase for case-insensitive search
        term = term.lower()

        # Initialize term entry if this is first occurrence of term
        if term not in self._term_to_findings:
            self._term_to_findings[term] = {}

        # Initialize finding entry for this term if needed
        if finding_id not in self._term_to_findings[term]:
            self._term_to_findings[term][finding_id] = []

        # Add position to the list (positions may not be in order if added out of sequence)
        self._term_to_findings[term][finding_id].append(position)

        # Track which terms belong to this finding (for efficient removal)
        if finding_id not in self._finding_terms:
            self._finding_terms[finding_id] = set()
        self._finding_terms[finding_id].add(term)

    def search(self, term: str) -> List[Tuple[str, List[int]]]:
        """
        Return list of (finding_id, [positions]) for findings containing term.

        Search is case-insensitive. Results sorted by finding_id.

        Args:
            term: Term to search for (case-insensitive)

        Returns:
            List of (finding_id, positions) tuples, sorted by finding_id.
            Returns empty list if term not found.
        """
        # Normalize term to lowercase
        term = term.lower()

        # Get findings for this term (O(1) lookup)
        findings = self._term_to_findings.get(term, {})

        # Convert to list of tuples, copying position lists to avoid mutation
        result = [(finding_id, positions[:]) for finding_id, positions in findings.items()]

        # Sort by finding_id as required
        result.sort(key=lambda x: x[0])

        return result

    def search_phrase(self, terms: List[str]) -> List[str]:
        """
        Return finding_ids where terms appear consecutively.

        Checks that all terms appear in sequence (position n, n+1, n+2, ...).
        Empty terms list returns empty list.

        Args:
            terms: List of terms that must appear consecutively

        Returns:
            List of finding_ids containing the phrase, sorted
        """
        if not terms:
            return []

        # Normalize all terms to lowercase
        terms = [t.lower() for t in terms]

        # Get findings for first term
        first_term = terms[0]
        if first_term not in self._term_to_findings:
            return []

        # Candidates are findings that contain the first term
        candidates = set(self._term_to_findings[first_term].keys())

        # For each candidate finding, check if all terms appear consecutively
        result = []
        for finding_id in candidates:
            # Get positions of first term in this finding
            first_positions = self._term_to_findings[first_term][finding_id]

            # Try each occurrence of the first term as a potential phrase start
            for start_pos in first_positions:
                # Check if all subsequent terms appear at consecutive positions
                found_phrase = True
                for i, term in enumerate(terms):
                    expected_pos = start_pos + i

                    # Check if this term exists in the index
                    if term not in self._term_to_findings:
                        found_phrase = False
                        break

                    # Check if this finding has this term
                    if finding_id not in self._term_to_findings[term]:
                        found_phrase = False
                        break

                    # Check if the expected position exists in the positions list
                    if expected_pos not in self._term_to_findings[term][finding_id]:
                        found_phrase = False
                        break

                # If we found the complete phrase starting at start_pos
                if found_phrase:
                    result.append(finding_id)
                    break  # Found phrase in this finding, no need to check other positions

        return sorted(result)

    def remove_finding(self, finding_id: str) -> None:
        """
        Remove all entries for a finding.

        No-op if finding doesn't exist.

        Args:
            finding_id: Identifier of finding to remove
        """
        # No-op if finding doesn't exist
        if finding_id not in self._finding_terms:
            return

        # Get all terms associated with this finding
        terms = self._finding_terms[finding_id]

        # Remove this finding from each term's index
        for term in terms:
            if term in self._term_to_findings:
                if finding_id in self._term_to_findings[term]:
                    del self._term_to_findings[term][finding_id]

                # Clean up term entry if no findings remain for this term
                if not self._term_to_findings[term]:
                    del self._term_to_findings[term]

        # Remove finding from tracking dictionary
        del self._finding_terms[finding_id]

    def term_frequency(self, term: str, finding_id: str) -> int:
        """
        Return number of times term appears in finding.

        Returns 0 if term not found or finding doesn't exist.

        Args:
            term: Term to count (case-insensitive)
            finding_id: Identifier of finding

        Returns:
            Count of term occurrences, or 0 if not found
        """
        # Normalize term to lowercase
        term = term.lower()

        # Check if term exists in index
        if term not in self._term_to_findings:
            return 0

        # Check if finding exists for this term
        if finding_id not in self._term_to_findings[term]:
            return 0

        # Return count of positions (number of occurrences)
        return len(self._term_to_findings[term][finding_id])

    def index_text(self, finding_id: str, text: str) -> None:
        """
        Tokenize text and add all terms with positions.

        Simple whitespace tokenization. Terms are normalized to lowercase.

        Args:
            finding_id: Identifier for this finding
            text: Text to tokenize and index
        """
        words = text.lower().split()
        for pos, word in enumerate(words):
            self.add(word.lower(), finding_id, pos)
