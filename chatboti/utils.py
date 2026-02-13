"""Utility functions for chatboti."""

import re


def make_slug(text: str, strip_latest: bool = False) -> str:
    """Convert text to filesystem-safe slug.

    Converts to lowercase, replaces non-alphanumeric with dashes,
    collapses multiple dashes, and strips leading/trailing dashes.

    :param text: Text to convert (e.g., model name, filename)
    :param strip_latest: If True, remove ':latest' suffix first
    :return: Slug (e.g., 'nomic-embed-text', 'my-data-file')
    """
    slug = text
    if strip_latest:
        slug = re.sub(r':latest$', '', slug)
    slug = re.sub(r'[^a-z0-9]+', '-', slug.lower())
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug
