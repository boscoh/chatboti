"""Document loaders for various file formats."""

import csv
from typing import List, Optional

from chatboti.document import Document, DocumentChunk


class DocumentLoader:
    """Base class for loading documents from various sources."""

    async def load(self, source: str, doc_type: str) -> List[Document]:
        """Load documents from source.

        :param source: File path
        :param doc_type: Document type identifier
        :return: List of documents with chunks
        """
        raise NotImplementedError


class CSVDocumentLoader(DocumentLoader):
    """Load documents from CSV files with field-level embeddings."""

    def __init__(self, embed_fields: Optional[List[str]] = None):
        """Initialize CSV loader.

        :param embed_fields: Fields to embed (default: all text fields)
        """
        self.embed_fields = embed_fields

    async def load(self, source: str, doc_type: str) -> List[Document]:
        """Load CSV rows as documents.

        For speakers.csv: Each row becomes a Document with content dict,
        chunks created for specified embed_fields (bio, abstract).

        :param source: Path to CSV file
        :param doc_type: Document type identifier
        :return: List of documents with chunks
        """
        documents = []
        with open(source) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Create document with content dict
                doc = Document(
                    id=f"{doc_type}-{i}",
                    content=dict(row),
                    chunks={}
                )

                # Create chunks for embed fields
                embed_fields = self.embed_fields or [k for k, v in row.items() if v]
                for field in embed_fields:
                    if field in row and row[field]:
                        doc.chunks[field] = DocumentChunk(faiss_id=-1)  # Assigned later

                documents.append(doc)

        return documents
