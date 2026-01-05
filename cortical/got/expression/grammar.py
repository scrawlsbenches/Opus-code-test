"""
Formal grammar definition for the GoT query expression language.

This module documents the complete grammar in EBNF notation and provides
grammar validation utilities.
"""

# EBNF Grammar (canonical reference)
GRAMMAR = '''
query           ::= expression [order_clause] [limit_clause]

expression      ::= and_expr ('OR' and_expr)*

and_expr        ::= not_expr ('AND' not_expr)*

not_expr        ::= 'NOT' not_expr
                  | primary

primary         ::= comparison
                  | function_call
                  | '(' expression ')'

comparison      ::= field_ref operator value

field_ref       ::= IDENTIFIER

operator        ::= '=' | '!=' | '>' | '<' | '>=' | '<='
                  | 'IN' | 'NOT' 'IN'
                  | 'LIKE' | 'NOT' 'LIKE'

value           ::= STRING
                  | NUMBER
                  | IDENTIFIER
                  | list

list            ::= '[' [value (',' value)*] ']'

function_call   ::= IDENTIFIER '(' [arg_list] ')'

arg_list        ::= arg (',' arg)*

arg             ::= [IDENTIFIER '='] value

order_clause    ::= 'ORDER' 'BY' IDENTIFIER [direction]

direction       ::= 'ASC' | 'DESC'

limit_clause    ::= 'LIMIT' NUMBER ['OFFSET' NUMBER]

(* Terminals *)
STRING          ::= "'" [^']* "'" | '"' [^"]* '"'
NUMBER          ::= [0-9]+ ('.' [0-9]+)?
IDENTIFIER      ::= [a-zA-Z_][a-zA-Z0-9_-]*
'''


# Valid examples that should parse successfully
VALID_EXAMPLES = {
    "simple_comparison": "status = 'pending'",
    "and_expression": "status = 'pending' AND priority = 'high'",
    "or_expression": "a = 1 OR b = 2",
    "mixed_precedence": "a = 1 OR b = 2 AND c = 3",  # AND binds tighter
    "parenthesized": "(a = 1 OR b = 2) AND c = 3",
    "not_expression": "NOT status = 'completed'",
    "in_operator": "status IN ['pending', 'active']",
    "not_in_operator": "status NOT IN ['deleted']",
    "like_operator": "title LIKE '%bug%'",
    "function_call": "connected_to(T-123)",
    "function_with_kwargs": "path(T-1, T-2, max_depth=5)",
    "order_by": "status = 'pending' ORDER BY created_at DESC",
    "limit_offset": "category = 'bug' LIMIT 10 OFFSET 20",
    "entity_id": "id = T-123",
    "complex": "status IN ['pending', 'active'] AND NOT priority = 'low' ORDER BY created_at DESC LIMIT 50",

    # Additional valid cases
    "double_quotes": 'status = "pending"',
    "number_comparison": "priority > 3",
    "greater_equal": "priority >= 3",
    "less_than": "priority < 5",
    "less_equal": "priority <= 5",
    "not_equal": "status != 'deleted'",
    "multiple_and": "a = 1 AND b = 2 AND c = 3",
    "multiple_or": "a = 1 OR b = 2 OR c = 3",
    "nested_parens": "((a = 1))",
    "complex_parens": "(a = 1 OR b = 2) AND (c = 3 OR d = 4)",
    "double_not": "NOT NOT status = 'pending'",
    "empty_list": "status IN []",
    "single_item_list": "status IN ['pending']",
    "numeric_list": "priority IN [1, 2, 3]",
    "mixed_whitespace": "  status  =  'pending'  ",
    "no_whitespace": "status='pending'",
    "order_asc": "status = 'pending' ORDER BY created_at ASC",
    "limit_only": "status = 'pending' LIMIT 10",
    "offset_only_limit": "status = 'pending' LIMIT 10",
    "function_no_args": "orphan_nodes()",
    "function_one_arg": "children(T-123)",
    "function_multi_args": "path(T-1, T-2, T-3)",
    "function_kwargs_only": "path(from=T-1, to=T-2)",
    "function_mixed_args": "path(T-1, T-2, max_depth=5, edge_type='DEPENDS_ON')",
    "hyphenated_id": "id = T-123-sub",
    "underscored_field": "field_name = 'value'",
}


# Invalid examples that should raise ParseError
INVALID_EXAMPLES = {
    "missing_value": "status =",
    "unclosed_paren": "(a = 1",
    "invalid_operator": "status == 'pending'",
    "unclosed_string": "status = 'pending",
    "missing_operator": "status 'pending'",
    "unclosed_list": "status IN ['pending'",
    "missing_list_value": "status IN [,]",
    "trailing_comma_list": "status IN ['pending',]",
    "missing_field": "= 'value'",
    "missing_comparison": "status AND",
    "invalid_keyword": "status IS 'pending'",
    "double_operator": "status = = 'pending'",
    "invalid_not": "status NOT 'pending'",  # NOT without IN/LIKE
    "missing_order_field": "status = 'pending' ORDER BY",
    "invalid_direction": "status = 'pending' ORDER BY created_at ASCENDING",
    "missing_limit_value": "status = 'pending' LIMIT",
    "invalid_limit": "status = 'pending' LIMIT abc",
    "offset_without_limit": "status = 'pending' OFFSET 10",  # OFFSET requires LIMIT
    "missing_function_paren": "connected_to",
    "unclosed_function_args": "connected_to(T-123",
    "invalid_kwarg": "path(123=value)",  # Keyword must be identifier
    "double_equals_kwarg": "path(key==value)",
    "empty_expression": "",
    "only_whitespace": "   ",
    "only_keyword": "AND",
    "missing_rparen": "(status = 'pending'",
    "extra_rparen": "status = 'pending')",
    "unmatched_bracket": "status IN ['pending']]",
}


def get_valid_examples():
    """Return all valid grammar examples."""
    return VALID_EXAMPLES


def get_invalid_examples():
    """Return all invalid grammar examples."""
    return INVALID_EXAMPLES


def get_grammar():
    """Return the EBNF grammar definition."""
    return GRAMMAR
