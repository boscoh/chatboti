"""Utility functions for chatboti."""

import re
import tomllib
from pathlib import Path


def get_version() -> str:
    """Get version from pyproject.toml.

    :return: Version string or 'unknown' if not found
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
            return pyproject.get("project", {}).get("version", "unknown")
    return "unknown"


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
