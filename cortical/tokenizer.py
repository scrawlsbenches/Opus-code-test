"""
Tokenizer Module
================

Text tokenization with stemming and word variant support.

Like early visual processing, the tokenizer extracts basic features
(words) from raw input, filtering noise (stop words) and normalizing
representations (lowercase, stemming).
"""

import re
from typing import List, Set, Optional, Dict, Tuple


# Ubiquitous code tokens that pollute query expansion
# These appear in almost every Python method/function, so they add noise
# rather than signal when expanding queries for code search
CODE_EXPANSION_STOP_WORDS = frozenset({
    'self', 'cls',              # Class method parameters
    'args', 'kwargs',           # Variadic parameters
    'none', 'true', 'false',    # Literals (too common)
    'return', 'pass',           # Control flow (too common)
    'def', 'class',             # Definitions (search for these explicitly)
})

# Very common code tokens that should be filtered from corpus analysis
# when mixed text/code documents are present. These dominate PageRank/TF-IDF
# due to appearing in almost every method/function.
CODE_NOISE_TOKENS = frozenset({
    # Python-specific
    'self', 'cls', 'args', 'kwargs',
    'def', 'class', 'return', 'pass',
    'none', 'true', 'false',
    'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
    'len', 'range', 'print', 'type', 'isinstance', 'hasattr',
    # Test framework noise
    'assertequal', 'asserttrue', 'assertfalse', 'assertnone',
    'assertis', 'assertisnot', 'assertin', 'assertnotin',
    'assertraises', 'setup', 'teardown', 'unittest',
    # Common variable names that are too generic
    'result', 'value', 'item', 'obj', 'data', 'func',
})


# Programming keywords that should be preserved even if in stop words
PROGRAMMING_KEYWORDS = frozenset({
    'def', 'class', 'function', 'return', 'import', 'from', 'if', 'else',
    'elif', 'for', 'while', 'try', 'except', 'finally', 'with', 'as',
    'yield', 'async', 'await', 'lambda', 'pass', 'break', 'continue',
    'raise', 'assert', 'global', 'nonlocal', 'del', 'true', 'false',
    'none', 'null', 'void', 'int', 'str', 'float', 'bool', 'list',
    'dict', 'set', 'tuple', 'self', 'cls', 'init', 'main', 'args',
    'kwargs', 'super', 'property', 'staticmethod', 'classmethod',
    'isinstance', 'hasattr', 'getattr', 'setattr', 'len', 'range',
    'enumerate', 'zip', 'map', 'filter', 'print', 'open', 'read',
    'write', 'close', 'append', 'extend', 'insert', 'remove', 'pop',
    # Dunder method components (for __init__, __slots__, etc.)
    'repr', 'slots', 'name', 'doc', 'call', 'iter', 'next', 'enter',
    'exit', 'getitem', 'setitem', 'delitem', 'contains', 'hash', 'eq',
    'const', 'let', 'var', 'public', 'private', 'protected', 'static',
    'final', 'abstract', 'interface', 'implements', 'extends', 'new',
    'this', 'constructor', 'module', 'export', 'require', 'package',
    # Common identifier components that shouldn't be filtered
    'get', 'set', 'add', 'put', 'has', 'can', 'run', 'max', 'min',
})


def split_identifier(identifier: str) -> List[str]:
    """
    Split a code identifier into component words.

    Handles camelCase, PascalCase, underscore_style, and CONSTANT_STYLE.

    Args:
        identifier: A code identifier like "getUserCredentials" or "get_user_data"

    Returns:
        List of component words in lowercase

    Examples:
        >>> split_identifier("getUserCredentials")
        ['get', 'user', 'credentials']
        >>> split_identifier("get_user_data")
        ['get', 'user', 'data']
        >>> split_identifier("XMLParser")
        ['xml', 'parser']
        >>> split_identifier("parseHTTPResponse")
        ['parse', 'http', 'response']
    """
    if not identifier:
        return []

    # Handle underscore_style and CONSTANT_STYLE
    if '_' in identifier:
        parts = [p for p in identifier.split('_') if p]
        # Recursively split any camelCase parts
        result = []
        for part in parts:
            if any(c.isupper() for c in part):  # Has any capitals - could be camelCase
                result.extend(split_identifier(part))
            else:
                result.append(part.lower())
        return [p for p in result if p]

    # Handle camelCase and PascalCase
    # Insert space before uppercase letters, handling acronyms
    # "parseHTTPResponse" -> "parse HTTP Response" -> ["parse", "http", "response"]
    result = []
    current = []

    for i, char in enumerate(identifier):
        if char.isupper():
            # Check if this starts a new word
            if current:
                # If previous was lowercase, this starts a new word
                if current[-1].islower():
                    result.append(''.join(current).lower())
                    current = [char]
                # If next char is lowercase, this uppercase starts a new word (end of acronym)
                elif i + 1 < len(identifier) and identifier[i + 1].islower():
                    result.append(''.join(current).lower())
                    current = [char]
                else:
                    # Continue building acronym
                    current.append(char)
            else:
                current.append(char)
        else:
            current.append(char)

    if current:
        result.append(''.join(current).lower())

    return [p for p in result if p]


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
    
    def __init__(
        self,
        stop_words: Optional[Set[str]] = None,
        min_word_length: int = 3,
        split_identifiers: bool = False,
        filter_code_noise: bool = False
    ):
        """
        Initialize tokenizer.

        Args:
            stop_words: Set of words to filter out. Uses defaults if None.
            min_word_length: Minimum word length to keep.
            split_identifiers: If True, split camelCase/underscore_style and include
                               both original and component tokens.
            filter_code_noise: If True, filter out common code tokens (self, def, etc.)
                              that dominate PageRank/TF-IDF in mixed text/code corpora.
        """
        base_stop_words = stop_words if stop_words is not None else self.DEFAULT_STOP_WORDS
        # Add code noise tokens to stop words if filtering is enabled
        if filter_code_noise:
            self.stop_words = base_stop_words | CODE_NOISE_TOKENS
        else:
            self.stop_words = base_stop_words
        self.min_word_length = min_word_length
        self.split_identifiers = split_identifiers
        self.filter_code_noise = filter_code_noise
        
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
    
    def tokenize(self, text: str, split_identifiers: Optional[bool] = None) -> List[str]:
        """
        Extract tokens from text.

        Args:
            text: Input text to tokenize.
            split_identifiers: Override instance setting. If True, split
                              camelCase/underscore_style identifiers into components.

        Returns:
            List of filtered, lowercase tokens.

        Examples:
            >>> t = Tokenizer(split_identifiers=True)
            >>> t.tokenize("getUserCredentials fetches data")
            ['getusercredentials', 'get', 'user', 'credentials', 'fetches', 'data']
        """
        should_split = split_identifiers if split_identifiers is not None else self.split_identifiers

        # Extract potential identifiers (including camelCase with internal caps)
        # Pattern matches: word2vec, getUserData, get_user_data, XMLParser
        # Also matches underscore-prefixed: __init__, _private, __slots__
        raw_tokens = re.findall(r'\b_*[a-zA-Z][a-zA-Z0-9_]*\b', text)

        result = []
        seen_splits = set()  # Only track splits to avoid duplicates from them

        for token in raw_tokens:
            token_lower = token.lower()

            # Skip stop words and short words
            if token_lower in self.stop_words or len(token_lower) < self.min_word_length:
                continue

            # Add the original token (allow duplicates for proper bigram extraction)
            result.append(token_lower)
            # Track this token to prevent splits from duplicating it
            seen_splits.add(token_lower)

            # Split identifier if enabled and token looks like an identifier
            if should_split and (
                '_' in token or
                any(c.isupper() for c in token[1:])  # Has internal capitals
            ):
                parts = split_identifier(token)
                for part in parts:
                    # Allow programming keywords even if in stop words
                    is_programming_keyword = part in PROGRAMMING_KEYWORDS
                    # Only add split parts once per token to avoid bloating
                    if (
                        part not in seen_splits and
                        part != token_lower and  # Don't duplicate the original
                        (is_programming_keyword or part not in self.stop_words) and
                        len(part) >= self.min_word_length
                    ):
                        result.append(part)
                        seen_splits.add(part)

        return result
    
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
