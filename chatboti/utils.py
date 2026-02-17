"""Utility functions for chatboti."""
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

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


def parse_response_as_json_list(response: Dict[str, Any] | str) -> Optional[Dict[str, Any] | List[Any]]:
    """Parse JSON from text response, extracting from markdown or .transactions if needed.

    Args:
        response: Response dict with 'text' key or raw string

    Returns:
        Parsed JSON object (dict or list) or None if parsing fails
    """
    if isinstance(response, dict):
        response_text = response.get("text", "")
    elif isinstance(response, str):
        response_text = response
    else:
        return None

    if not response_text:
        return None

    def try_parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    parsed = try_parse(response_text)
    if parsed:
        return parsed

    patterns = [
        r"```(?:json|python)?\s*([\s\S]*?)\s*```",
        r"```(?:json)?\s*({[\s\S]*})\s*```",
        r"\{[\s\S]*\}",
        r"({[\s\S]*})",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        for match in matches:
            parsed = try_parse(match)
            if parsed:
                return parsed

    return None
