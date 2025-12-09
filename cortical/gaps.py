"""
Gaps Module
===========

Knowledge gap detection and anomaly analysis.

Identifies:
- Isolated documents that don't connect well to the corpus
- Weakly covered topics (few documents)
- Bridge opportunities between document clusters
- Anomalous documents that may be miscategorized
"""

import math
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

from .layers import CorticalLayer, HierarchicalLayer
from .analysis import cosine_similarity


def analyze_knowledge_gaps(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str]
) -> Dict:
    """
    Analyze the corpus to identify potential knowledge gaps.
    
    Args:
        layers: Dictionary of layers
        documents: Dictionary of documents
        
    Returns:
        Dict with gap analysis results including isolated_documents,
        weak_topics, bridge_opportunities, coverage_score, etc.
    """
    layer0 = layers[CorticalLayer.TOKENS]
    doc_ids = list(documents.keys())
    
    # 1. Find isolated documents
    isolated_docs = []
    doc_similarities: Dict[str, Dict[str, float]] = {}
    
    for doc_id in doc_ids:
        doc_vector = {col.content: col.tfidf_per_doc[doc_id] 
                     for col in layer0.minicolumns.values() 
                     if doc_id in col.tfidf_per_doc}
        
        similarities = []
        for other_id in doc_ids:
            if other_id != doc_id:
                other_vector = {col.content: col.tfidf_per_doc[other_id]
                               for col in layer0.minicolumns.values()
                               if other_id in col.tfidf_per_doc}
                sim = cosine_similarity(doc_vector, other_vector)
                similarities.append((other_id, sim))
        
        avg_sim = sum(s for _, s in similarities) / len(similarities) if similarities else 0
        max_sim = max((s for _, s in similarities), default=0)
        doc_similarities[doc_id] = {'avg': avg_sim, 'max': max_sim}
        
        if avg_sim < 0.02:
            isolated_docs.append({
                'doc_id': doc_id,
                'avg_similarity': avg_sim,
                'max_similarity': max_sim,
                'most_similar': max(similarities, key=lambda x: x[1])[0] if similarities else None
            })
    
    isolated_docs.sort(key=lambda x: x['avg_similarity'])
    
    # 2. Find weakly covered topics
    weak_topics = []
    for col in layer0.minicolumns.values():
        doc_count = len(col.document_ids)
        if col.tfidf > 0.005 and 1 <= doc_count <= 2:
            weak_topics.append({
                'term': col.content,
                'tfidf': col.tfidf,
                'doc_count': doc_count,
                'documents': list(col.document_ids),
                'pagerank': col.pagerank
            })
    weak_topics.sort(key=lambda x: x['tfidf'] * x['pagerank'], reverse=True)
    
    # 3. Find bridge opportunities
    bridge_opportunities = []
    for i, doc1 in enumerate(doc_ids):
        vec1 = {col.content: col.tfidf_per_doc[doc1] 
               for col in layer0.minicolumns.values() 
               if doc1 in col.tfidf_per_doc}
        
        for doc2 in doc_ids[i+1:]:
            vec2 = {col.content: col.tfidf_per_doc[doc2]
                   for col in layer0.minicolumns.values()
                   if doc2 in col.tfidf_per_doc}
            
            sim = cosine_similarity(vec1, vec2)
            if 0.005 < sim < 0.03:
                shared = set(vec1.keys()) & set(vec2.keys())
                bridge_opportunities.append({
                    'doc1': doc1,
                    'doc2': doc2,
                    'similarity': sim,
                    'shared_terms': list(shared)[:5]
                })
    
    bridge_opportunities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 4. Connector terms
    connector_terms = []
    isolated_doc_ids = {d['doc_id'] for d in isolated_docs[:5]}
    if isolated_doc_ids:
        for col in layer0.minicolumns.values():
            in_isolated = col.document_ids & isolated_doc_ids
            in_connected = col.document_ids - isolated_doc_ids
            if in_isolated and in_connected:
                connector_terms.append({
                    'term': col.content,
                    'bridges_isolated': list(in_isolated),
                    'connects_to': list(in_connected)[:3],
                    'pagerank': col.pagerank
                })
    connector_terms.sort(key=lambda x: len(x['bridges_isolated']), reverse=True)
    
    # 5. Coverage metrics
    total_docs = len(doc_ids)
    isolated_count = len([d for d in doc_similarities.values() if d['avg'] < 0.02])
    well_connected = len([d for d in doc_similarities.values() if d['avg'] >= 0.03])
    coverage_score = well_connected / total_docs if total_docs > 0 else 0
    
    all_avg_sims = [d['avg'] for d in doc_similarities.values()]
    connectivity_score = sum(all_avg_sims) / len(all_avg_sims) if all_avg_sims else 0
    
    return {
        'isolated_documents': isolated_docs[:10],
        'weak_topics': weak_topics[:10],
        'bridge_opportunities': bridge_opportunities[:10],
        'connector_terms': connector_terms[:10],
        'coverage_score': coverage_score,
        'connectivity_score': connectivity_score,
        'summary': {
            'total_documents': total_docs,
            'isolated_count': isolated_count,
            'well_connected_count': well_connected,
            'weak_topic_count': len(weak_topics)
        }
    }


def detect_anomalies(
    layers: Dict[CorticalLayer, HierarchicalLayer],
    documents: Dict[str, str],
    threshold: float = 0.3
) -> List[Dict]:
    """
    Detect documents that don't fit well with the rest of the corpus.
    
    Args:
        layers: Dictionary of layers
        documents: Dictionary of documents
        threshold: Similarity threshold below which docs are anomalies
        
    Returns:
        List of anomaly reports with explanations
    """
    layer0 = layers[CorticalLayer.TOKENS]
    layer3 = layers.get(CorticalLayer.DOCUMENTS)
    anomalies = []
    
    for doc_id in documents:
        doc_col = layer3.get_minicolumn(doc_id) if layer3 else None
        connection_count = doc_col.connection_count() if doc_col else 0
        
        doc_vector = {col.content: col.tfidf_per_doc[doc_id]
                     for col in layer0.minicolumns.values()
                     if doc_id in col.tfidf_per_doc}
        
        similarities = []
        for other_id in documents:
            if other_id != doc_id:
                other_vector = {col.content: col.tfidf_per_doc[other_id]
                               for col in layer0.minicolumns.values()
                               if other_id in col.tfidf_per_doc}
                similarities.append(cosine_similarity(doc_vector, other_vector))
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0
        
        is_anomaly = False
        reasons = []
        
        if avg_similarity < threshold:
            is_anomaly = True
            reasons.append(f"Low average similarity ({avg_similarity:.1%})")
        if connection_count <= 1:
            is_anomaly = True
            reasons.append(f"Few document connections ({connection_count})")
        if max_similarity < threshold * 1.5:
            is_anomaly = True
            reasons.append("No closely related documents")
        
        if is_anomaly:
            sig_terms = sorted(doc_vector.items(), key=lambda x: x[1], reverse=True)[:5]
            anomalies.append({
                'doc_id': doc_id,
                'avg_similarity': avg_similarity,
                'max_similarity': max_similarity,
                'connections': connection_count,
                'reasons': reasons,
                'distinctive_terms': [t for t, _ in sig_terms]
            })
    
    anomalies.sort(key=lambda x: x['avg_similarity'])
    return anomalies
