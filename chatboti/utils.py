"""Utility functions for chatboti."""

import re
from pathlib import Path

try:
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # backport


def get_version() -> str:
    """Get version from package metadata.

    :return: Version string or '0+unknown' if not found
    """
    try:
        return version("chatboti")
    except PackageNotFoundError:
        return "0+unknown"


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
        slug = re.sub(r":latest$", "", slug)
    slug = re.sub(r"[^a-z0-9]+", "-", slug.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug
