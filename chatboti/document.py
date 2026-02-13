"""Document and chunk data structures for unified RAG system."""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class DocumentChunk:
    """Chunk reference with optional text indices.

    Field-level: dict key is field name, i_start/i_end are None
    Chunk-level: dict key is index, i_start/i_end locate text
    Global ID: (document.id, dict_key)
    """
    faiss_id: int
    i_start: Optional[int] = None
    i_end: Optional[int] = None


@dataclass(frozen=True)
class ChunkRef:
    """Maps faiss_id to document location."""
    document_id: str
    chunk_key: str


@dataclass
class ChunkResult:
    """Search result with chunk text."""
    document_id: str
    chunk_key: str
    text: str
    document_text: Optional[str] = None
    content: Optional[Dict] = None


class Document:
    """Document with flexible chunking support.

    Supports both field-level chunking (structured documents) and
    chunk-level chunking (long text documents).
    """

    def __init__(
        self,
        id: str,
        content: Optional[Dict[str, str]] = None,
        full_text: str = "",
        metadata: Optional[dict] = None,
        chunks: Optional[Dict[str, DocumentChunk]] = None
    ):
        """Initialize a document.

        :param id: Unique document identifier
        :param content: Flexible fields for field-level chunking
        :param full_text: Complete text for chunk-level chunking
        :param metadata: Source, timestamp, and other metadata
        :param chunks: Mapping of field names or indices to chunks
        """
        self.id = id
        self.content = content if content is not None else {}
        self.full_text = full_text
        self.metadata = metadata if metadata is not None else {}
        self.chunks = chunks if chunks is not None else {}

    def get_chunk_text(self, key: str) -> str:
        """Get chunk text by field name or index.

        :param key: Field name ("bio") or chunk index ("0")
        :return: Chunk text
        """
        chunk = self.chunks[key]
        if chunk.i_start is not None:
            return self.full_text[chunk.i_start:chunk.i_end]
        return self.content[key]

    def get_chunk_with_context(self, key: str, context_chars: int = 200) -> tuple[str, str, str]:
        """Get chunk text with surrounding context.

        :param key: Field name or chunk index
        :param context_chars: Characters to include before/after
        :return: (before, chunk, after) tuple
        """
        chunk = self.chunks[key]
        if chunk.i_start is None:
            return ("", self.content[key], "")

        before_start = max(0, chunk.i_start - context_chars)
        after_end = min(len(self.full_text), chunk.i_end + context_chars)
        return (
            self.full_text[before_start:chunk.i_start],
            self.full_text[chunk.i_start:chunk.i_end],
            self.full_text[chunk.i_end:after_end]
        )

    def to_dict(self) -> dict:
        """Serialize document to dictionary.

        :return: Dictionary representation of document
        """
        return {
            "id": self.id,
            "content": self.content,
            "full_text": self.full_text,
            "metadata": self.metadata,
            "chunks": {
                key: {
                    "faiss_id": chunk.faiss_id,
                    "i_start": chunk.i_start,
                    "i_end": chunk.i_end
                }
                for key, chunk in self.chunks.items()
            }
        }

    @staticmethod
    def from_dict(data: dict) -> 'Document':
        """Deserialize document from dictionary.

        :param data: Dictionary representation of document
        :return: Document instance
        """
        chunks = {
            key: DocumentChunk(
                faiss_id=chunk_data["faiss_id"],
                i_start=chunk_data.get("i_start"),
                i_end=chunk_data.get("i_end")
            )
            for key, chunk_data in data.get("chunks", {}).items()
        }

        return Document(
            id=data["id"],
            content=data.get("content", {}),
            full_text=data.get("full_text", ""),
            metadata=data.get("metadata", {}),
            chunks=chunks
        )
