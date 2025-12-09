"""
Tokenizer Module
================

Text tokenization with stemming and word variant support.

Like early visual processing, the tokenizer extracts basic features
(words) from raw input, filtering noise (stop words) and normalizing
representations (lowercase, stemming).
"""

import re
from typing import List, Set, Optional, Dict


class Tokenizer:
    """
    Text tokenizer with stemming and word variant support.
    
    Extracts tokens from text, filters stop words, and provides
    word variant expansion for query normalization.
    
    Attributes:
        stop_words: Set of words to filter out
        min_word_length: Minimum word length to keep
        
    Example:
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Neural networks process information")
        # ['neural', 'networks', 'process', 'information']
        
        variants = tokenizer.get_word_variants("bread")
        # ['bread', 'sourdough', 'dough', 'flour', 'baking', 'breads']
    """
    
    DEFAULT_STOP_WORDS = frozenset({
        # Articles and conjunctions
        'the', 'a', 'an', 'and', 'or', 'but', 'nor', 'yet', 'so',
        # Prepositions
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'over', 'again', 'against', 'about', 'within',
        'without', 'toward', 'towards', 'upon', 'across', 'along', 'around',
        'behind', 'beside', 'beyond', 'down', 'inside', 'outside', 'throughout',
        # Verbs (auxiliary and common)
        'is', 'was', 'are', 'were', 'been', 'be', 'being',
        'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'done',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
        'need', 'needs', 'needed',
        'get', 'gets', 'got', 'getting',
        'make', 'makes', 'made', 'making',
        'take', 'takes', 'took', 'taking', 'taken',
        'come', 'comes', 'came', 'coming',
        'give', 'gives', 'gave', 'giving', 'given',
        'use', 'uses', 'used', 'using',
        # Pronouns
        'that', 'this', 'these', 'those', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves',
        'he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself',
        'we', 'us', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves',
        'i', 'me', 'my', 'mine', 'myself',
        'who', 'whom', 'whose', 'what', 'which', 'when', 'where', 'why', 'how',
        # Adverbs and modifiers
        'not', 'no', 'yes', 'so', 'if', 'then', 'than', 'too', 'very', 'just',
        'also', 'only', 'even', 'still', 'already', 'always', 'never', 'ever',
        'often', 'sometimes', 'usually', 'now', 'here', 'there', 'well', 'much',
        'more', 'most', 'less', 'least', 'rather', 'quite', 'almost', 'nearly',
        'really', 'actually', 'especially', 'particularly', 'generally',
        # Common transitional words
        'while', 'although', 'though', 'however', 'therefore', 'thus', 'hence',
        'moreover', 'furthermore', 'nevertheless', 'nonetheless', 'meanwhile',
        'otherwise', 'instead', 'besides', 'whereas', 'whether', 'unless',
        # Common verbs
        'include', 'includes', 'including', 'included',
        'provide', 'provides', 'provided', 'providing',
        'require', 'requires', 'required', 'requiring',
        'enable', 'enables', 'enabled', 'enabling',
        'allow', 'allows', 'allowed', 'allowing',
        'create', 'creates', 'created', 'creating',
        'become', 'becomes', 'became', 'becoming',
        'remain', 'remains', 'remained', 'remaining',
        'offer', 'offers', 'offered', 'offering',
        'support', 'supports', 'supported', 'supporting',
        # Quantifiers and determiners
        'each', 'every', 'any', 'some', 'all', 'both', 'few', 'many', 'several',
        'such', 'other', 'another', 'same', 'different', 'own', 'certain',
        'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next',
        # Common nouns (too generic)
        'way', 'ways', 'thing', 'things', 'time', 'times', 'year', 'years',
        'day', 'days', 'place', 'part', 'parts', 'case', 'cases', 'point',
        'fact', 'kind', 'type', 'form', 'forms', 'level', 'area', 'areas',
        # Common adjectives (too generic)
        'new', 'old', 'good', 'bad', 'great', 'small', 'large', 'big', 'long',
        'high', 'low', 'right', 'left', 'possible', 'important', 'major',
        'available', 'able', 'like', 'different', 'similar'
    })
    
    def __init__(self, stop_words: Optional[Set[str]] = None, min_word_length: int = 3):
        """
        Initialize tokenizer.
        
        Args:
            stop_words: Set of words to filter out. Uses defaults if None.
            min_word_length: Minimum word length to keep.
        """
        self.stop_words = stop_words if stop_words is not None else self.DEFAULT_STOP_WORDS
        self.min_word_length = min_word_length
        
        # Simple suffix rules for stemming (Porter-lite)
        self._suffix_rules = [
            ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
            ('anci', 'ance'), ('izer', 'ize'), ('isation', 'ize'),
            ('ization', 'ize'), ('ation', 'ate'), ('ator', 'ate'),
            ('alism', 'al'), ('iveness', 'ive'), ('fulness', 'ful'),
            ('ousness', 'ous'), ('aliti', 'al'), ('iviti', 'ive'),
            ('biliti', 'ble'), ('ement', ''), ('ment', ''), ('ness', ''),
            ('ling', ''), ('ing', ''), ('ies', 'y'), ('ied', 'y'),
            ('es', ''), ('ed', ''), ('ly', ''), ('er', ''), ('est', ''),
            ('ful', ''), ('less', ''), ('able', ''), ('ible', ''),
            ('ness', ''), ('ment', ''), ('ity', ''),
        ]
        
        # Common word mappings for query normalization
        self._word_mappings: Dict[str, List[str]] = {
            # Bread/baking related
            'bread': ['sourdough', 'dough', 'flour', 'baking', 'loaf'],
            'baking': ['sourdough', 'bread', 'dough', 'flour'],
            # Neural/brain related
            'brain': ['neural', 'cortical', 'neurons', 'cognitive'],
            'ai': ['neural', 'learning', 'artificial', 'intelligence'],
            'ml': ['learning', 'machine', 'neural', 'training'],
            # Database/storage
            'database': ['storage', 'data', 'query', 'index'],
            'db': ['database', 'storage', 'data'],
            # Common abbreviations
            'nlp': ['natural', 'language', 'processing', 'text'],
            'api': ['interface', 'endpoint', 'service'],
            # Synonyms
            'fast': ['quick', 'rapid', 'speed'],
            'slow': ['latency', 'delay'],
            'big': ['large', 'scale', 'massive'],
            'small': ['tiny', 'minimal', 'compact'],
        }
    
    def tokenize(self, text: str) -> List[str]:
        """
        Extract tokens from text.
        
        Args:
            text: Input text to tokenize.
            
        Returns:
            List of filtered, lowercase tokens.
        """
        # Convert to lowercase and extract words (including alphanumeric like word2vec)
        words = re.findall(r'\b[a-z][a-z0-9]*\b', text.lower())
        
        # Filter stop words and short words
        return [
            w for w in words 
            if w not in self.stop_words and len(w) >= self.min_word_length
        ]
    
    def extract_ngrams(self, tokens: List[str], n: int = 2) -> List[str]:
        """
        Extract n-grams from token list.
        
        Args:
            tokens: List of tokens.
            n: Size of n-grams to extract.
            
        Returns:
            List of n-gram strings (tokens joined by space).
        """
        if len(tokens) < n:
            return []
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def stem(self, word: str) -> str:
        """
        Apply simple suffix stripping (Porter-lite stemming).
        
        Args:
            word: Word to stem
            
        Returns:
            Stemmed word
        """
        if len(word) <= 4:
            return word
        
        for suffix, replacement in self._suffix_rules:
            if word.endswith(suffix):
                stemmed = word[:-len(suffix)] + replacement
                if len(stemmed) >= 3:
                    return stemmed
        
        return word
    
    def get_word_variants(self, word: str) -> List[str]:
        """
        Get related words/variants for query expansion.
        
        Args:
            word: Input word
            
        Returns:
            List of related words including the original
        """
        word_lower = word.lower()
        variants = [word_lower]
        
        # Add mapped variants
        if word_lower in self._word_mappings:
            variants.extend(self._word_mappings[word_lower])
        
        # Add stemmed version
        stemmed = self.stem(word_lower)
        if stemmed != word_lower:
            variants.append(stemmed)
        
        # Add common variations
        if word_lower.endswith('s') and len(word_lower) > 3:
            variants.append(word_lower[:-1])  # Remove plural
        elif not word_lower.endswith('s'):
            variants.append(word_lower + 's')  # Add plural
        
        return list(set(variants))
    
    def add_word_mapping(self, word: str, variants: List[str]) -> None:
        """
        Add a custom word mapping for query expansion.
        
        Args:
            word: The source word
            variants: List of variant words to map to
        """
        word_lower = word.lower()
        if word_lower in self._word_mappings:
            self._word_mappings[word_lower].extend(variants)
            self._word_mappings[word_lower] = list(set(self._word_mappings[word_lower]))
        else:
            self._word_mappings[word_lower] = variants
