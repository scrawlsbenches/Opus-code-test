"""
Evaluation Framework
====================

Comprehensive evaluation of cortical text processing capabilities.
"""

import os
import re
from typing import Dict, List, Tuple, Any, Optional
import sys
sys.path.insert(0, '..')

from cortical import CorticalTextProcessor


class CorticalEvaluator:
    """Evaluates cortical text processor against test cases."""
    
    def __init__(self, processor: CorticalTextProcessor = None, samples_dir: str = None):
        self.processor = processor or CorticalTextProcessor()
        self.samples_dir = samples_dir
        self.results: List[Dict] = []
        
    def load_samples(self, samples_dir: str = None) -> int:
        """Load sample documents from directory."""
        samples_dir = samples_dir or self.samples_dir
        if not samples_dir or not os.path.exists(samples_dir):
            return 0
            
        count = 0
        for filename in os.listdir(samples_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(samples_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                doc_id = filename.replace('.txt', '')
                self.processor.process_document(doc_id, content)
                count += 1
        
        return count
    
    def prepare(self, verbose: bool = True):
        """Run all computations to prepare for evaluation."""
        self.processor.compute_all(verbose=verbose)
        self.processor.extract_corpus_semantics(verbose=verbose)
        self.processor.retrofit_connections(verbose=verbose)
        self.processor.compute_graph_embeddings(verbose=verbose)
        self.processor.retrofit_embeddings(verbose=verbose)
    
    def evaluate_factual_retrieval(self) -> Dict:
        """Test factual retrieval capability."""
        tests = [
            ("neural networks", ["graph_neural_networks", "brain_inspired_computing"]),
            ("machine learning", ["deep_learning_revolution", "knowledge_enhanced_nlp"]),
            ("deep learning", ["deep_learning_revolution", "graph_neural_networks"]),
        ]
        
        passed = 0
        total = len(tests)
        
        for query, expected in tests:
            results = self.processor.find_documents_for_query(query, top_n=5)
            found_docs = [doc_id for doc_id, _ in results]
            if any(exp in found_docs for exp in expected):
                passed += 1
        
        return {
            'category': 'Factual Retrieval',
            'passed': passed,
            'total': total,
            'score': passed / total if total > 0 else 0
        }
    
    def evaluate_query_expansion(self) -> Dict:
        """Test query expansion capability."""
        tests = [
            ("neural", 3),  # Should expand to at least 3 terms
            ("learning", 3),
            ("bread", 2),
        ]
        
        passed = 0
        total = len(tests)
        
        for query, min_expansions in tests:
            expanded = self.processor.expand_query(query, max_expansions=10)
            if len(expanded) >= min_expansions:
                passed += 1
        
        return {
            'category': 'Query Expansion',
            'passed': passed,
            'total': total,
            'score': passed / total if total > 0 else 0
        }
    
    def evaluate_gap_detection(self) -> Dict:
        """Test gap detection capability."""
        gaps = self.processor.analyze_knowledge_gaps()
        
        tests = [
            ('coverage_score' in gaps, "Has coverage score"),
            ('isolated_documents' in gaps, "Has isolated documents"),
            ('weak_topics' in gaps, "Has weak topics"),
        ]
        
        passed = sum(1 for result, _ in tests if result)
        total = len(tests)
        
        return {
            'category': 'Gap Detection',
            'passed': passed,
            'total': total,
            'score': passed / total if total > 0 else 0
        }
    
    def run_all(self, verbose: bool = True) -> Dict:
        """Run all evaluations and return summary."""
        evaluations = [
            self.evaluate_factual_retrieval,
            self.evaluate_query_expansion,
            self.evaluate_gap_detection,
        ]
        
        results = []
        total_passed = 0
        total_tests = 0
        
        for eval_func in evaluations:
            result = eval_func()
            results.append(result)
            total_passed += result['passed']
            total_tests += result['total']
            
            if verbose:
                print(f"  {result['category']}: {result['passed']}/{result['total']} ({result['score']:.1%})")
        
        overall_score = total_passed / total_tests if total_tests > 0 else 0
        
        return {
            'results': results,
            'total_passed': total_passed,
            'total_tests': total_tests,
            'overall_score': overall_score
        }
