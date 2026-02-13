"""Document loaders for various file formats."""

import csv
from pathlib import Path
from typing import List, Optional

from chatboti.document import Document, DocumentChunk


def infer_doc_type(source: str) -> str:
    """Infer document type from filename.

    :param source: File path
    :return: Document type (filename stem)
    """
    return Path(source).stem


async def load_csv(source: str, doc_type: Optional[str] = None, embed_fields: Optional[List[str]] = None) -> List[Document]:
    """Load CSV rows as documents.

    Each row becomes a Document with content dict,
    chunks created for specified embed_fields.

    :param source: Path to CSV file
    :param doc_type: Document type identifier (default: inferred from filename)
    :param embed_fields: Fields to embed (default: all text fields)
    :return: List of documents with chunks
    """
    if doc_type is None:
        doc_type = infer_doc_type(source)

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
            fields = embed_fields or [k for k, v in row.items() if v]
            for field in fields:
                if field in row and row[field]:
                    doc.chunks[field] = DocumentChunk(faiss_id=-1)  # Assigned later

            documents.append(doc)

    return documents


async def load_documents(source: str, doc_type: Optional[str] = None, embed_fields: Optional[List[str]] = None) -> List[Document]:
    """Load documents from source, dispatching to appropriate loader.

    :param source: File path
    :param doc_type: Document type identifier (default: inferred from filename)
    :param embed_fields: Fields to embed for CSV files (default: all text fields)
    :return: List of documents with chunks
    """
    ext = Path(source).suffix.lower()

    if ext == '.csv':
        return await load_csv(source, doc_type, embed_fields)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
