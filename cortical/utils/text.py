"""Text utility functions for string manipulation."""


def slugify(text: str, max_length: int = 50) -> str:
    """
    Convert text to URL-friendly slug.

    Args:
        text: Text to convert to slug
        max_length: Maximum length of slug (default: 50)

    Returns:
        Slugified text (lowercase, hyphens, alphanumeric only)

    Examples:
        >>> slugify("Hello World")
        'hello-world'
        >>> slugify("Task: Fix Bug #123")
        'task-fix-bug-123'
        >>> slugify("Multiple   Spaces", max_length=10)
        'multiple-s'
    """
    # Simple slugification: lowercase, replace spaces with hyphens
    slug = text.lower().strip()
    slug = slug.replace(" ", "-")
    # Remove non-alphanumeric except hyphens
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    # Remove duplicate hyphens
    while "--" in slug:
        slug = slug.replace("--", "-")
    # Truncate to reasonable length
    return slug[:max_length]
