"""TinyRAG - Tiny RAG starter kit"""

try:
    from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # backport

try:
    __version__ = version("chatboti")
except PackageNotFoundError:
    __version__ = "0+unknown"
