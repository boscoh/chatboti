"""HDF5-based RAG service for single-file storage.

This module provides an alternative to the multi-file FAISS+JSON approach,
storing all RAG data (vectors, chunks, documents) in a single HDF5 file.

Advantages:
- Single file for complete dataset
- Efficient binary format with compression
- Partial loading capability
- Self-describing metadata
- Industry standard format (h5py)

See docs/single-file-rag-backend-spec.md section 2.1 for format details.
"""

import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

# Third-party
import faiss
import h5py
import numpy as np
from chatboti.llm import SimpleLLMClient

# Local
from chatboti.document import ChunkRef, Document
from chatboti.faiss_rag import FaissRAGService


class HDF5RAGService(FaissRAGService):
    """RAG service using HDF5 single-file backend.

    Storage format:
        /metadata (attributes: model_name, embedding_dim, created_at, etc.)
        /vectors (dataset: float32 array [n_chunks × embedding_dim])
        /chunks (dataset: structured array with faiss_id, document_id, chunk_key)
        /documents (group with nested groups for each document)

    Usage:
        from chatboti.llm import get_llm_client

        embed_client = get_llm_client("ollama", model="nomic-embed-text")
        await embed_client.connect()

        async with HDF5RAGService(embed_client=embed_client,
                                   hdf5_path=Path("embeddings.h5")) as rag:
            results = await rag.search("query")

        await embed_client.close()
    """

    def __init__(
        self,
        embed_client: SimpleLLMClient,
        data_dir: Optional[Path] = None,
        hdf5_path: Optional[Path] = None,
    ):
        """Initialize HDF5 RAG service.

        :param embed_client: Connected embedding client (from microeval.llm.get_llm_client)
        :param data_dir: Data directory (default: chatboti/data)
        :param hdf5_path: Path to HDF5 file (overrides auto-detection)
        """
        # Call parent
        super().__init__(
            embed_client=embed_client,
            data_dir=data_dir,
            index_path=None,  # Not used
            metadata_path=None,  # Not used
        )
        self.hdf5_path = hdf5_path

    async def __aenter__(self):
        """Async context manager entry - performs async initialization."""
        # First call parent to set up embed client and paths
        await super().__aenter__()

        # Set HDF5 path if not provided
        if not self.hdf5_path:
            from chatboti.utils import make_slug

            model_slug = (
                make_slug(self.model_name, strip_latest=True)
                if self.model_name
                else "default"
            )
            self.hdf5_path = self.data_dir / f"embeddings-{model_slug}.h5"

        return self

    def initialize_search_backend(self):
        """Load or create search backend from HDF5 file."""

        if self.hdf5_path.exists():
            self.load_from_hdf5(self.hdf5_path)
        else:
            # Create empty index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.chunk_refs = []
            self.documents = {}

    def load_from_hdf5(self, path: Path) -> None:
        """Load FAISS index, chunks, and documents from HDF5 file.

        :param path: Path to HDF5 file
        """
        with h5py.File(path, "r") as f:
            # Load metadata from attributes
            self.model_name = f.attrs.get("model_name", self.model_name)
            self.embedding_dim = int(f.attrs["embedding_dim"])

            # Load vectors and rebuild FAISS index
            if "vectors" in f:
                vectors = f["vectors"][:]
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                if len(vectors) > 0:
                    self.index.add(vectors)
            else:
                self.index = faiss.IndexFlatIP(self.embedding_dim)

            # Load chunks from structured array
            if "chunks" in f:
                chunk_data = f["chunks"][:]
                self.chunk_refs = []
                for row in chunk_data:
                    # Properly decode bytes to strings
                    doc_id = row["document_id"]
                    chunk_key = row["chunk_key"]
                    if isinstance(doc_id, bytes):
                        doc_id = doc_id.decode("utf-8")
                    if isinstance(chunk_key, bytes):
                        chunk_key = chunk_key.decode("utf-8")
                    self.chunk_refs.append(
                        ChunkRef(document_id=doc_id, chunk_key=chunk_key)
                    )
            else:
                self.chunk_refs = []

            # Load documents from nested groups
            if "documents" in f:
                self.documents = {}
                docs_group = f["documents"]
                for doc_id_raw in docs_group.keys():
                    # Decode doc_id if it's bytes
                    doc_id = doc_id_raw
                    if isinstance(doc_id, bytes):
                        doc_id = doc_id.decode("utf-8")

                    doc_group = docs_group[doc_id_raw]

                    # Load attributes
                    doc_data = {
                        "id": doc_id,
                        "source": doc_group.attrs.get("source", ""),
                        "full_text": "",
                        "content": {},
                        "metadata": {},
                        "chunks": {},
                    }

                    # Load full_text if present
                    if "full_text" in doc_group:
                        doc_data["full_text"] = doc_group["full_text"][()]
                        if isinstance(doc_data["full_text"], bytes):
                            doc_data["full_text"] = doc_data["full_text"].decode(
                                "utf-8"
                            )

                    # Load content (JSON) if present
                    if "content" in doc_group:
                        content_str = doc_group["content"][()]
                        if isinstance(content_str, bytes):
                            content_str = content_str.decode("utf-8")
                        doc_data["content"] = json.loads(content_str)

                    # Load metadata (JSON) if present
                    if "metadata" in doc_group:
                        metadata_str = doc_group["metadata"][()]
                        if isinstance(metadata_str, bytes):
                            metadata_str = metadata_str.decode("utf-8")
                        doc_data["metadata"] = json.loads(metadata_str)

                    # Load chunks (JSON) if present
                    if "chunks" in doc_group:
                        chunks_str = doc_group["chunks"][()]
                        if isinstance(chunks_str, bytes):
                            chunks_str = chunks_str.decode("utf-8")
                        doc_data["chunks"] = json.loads(chunks_str)

                    # Create document from dict
                    self.documents[doc_id] = Document.from_dict(doc_data)
            else:
                self.documents = {}

    def save_to_hdf5(self, path: Path) -> None:
        """Save FAISS index, chunks, and documents to HDF5 file.

        :param path: Path to HDF5 file
        """
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            # Store metadata as attributes
            f.attrs["model_name"] = self.model_name or ""
            f.attrs["embedding_dim"] = self.embedding_dim
            f.attrs["created_at"] = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
            f.attrs["format_version"] = "1.0"
            f.attrs["index_type"] = self.index.__class__.__name__
            f.attrs["vector_count"] = self.index.ntotal
            f.attrs["document_count"] = len(self.documents)

            # Store vectors as float32 array with compression
            n_vectors = self.index.ntotal
            if n_vectors > 0:
                vectors = np.zeros((n_vectors, self.embedding_dim), dtype=np.float32)
                for i in range(n_vectors):
                    vectors[i] = self.index.reconstruct(i)
                f.create_dataset("vectors", data=vectors, compression="gzip")
            else:
                # Create empty dataset
                f.create_dataset(
                    "vectors", shape=(0, self.embedding_dim), dtype=np.float32
                )

            # Store chunks as structured array
            if self.chunk_refs:
                chunk_dtype = np.dtype(
                    [
                        ("faiss_id", "i8"),
                        ("document_id", h5py.string_dtype(encoding="utf-8")),
                        ("chunk_key", h5py.string_dtype(encoding="utf-8")),
                    ]
                )
                chunk_array = np.array(
                    [
                        (i, ref.document_id, ref.chunk_key)
                        for i, ref in enumerate(self.chunk_refs)
                    ],
                    dtype=chunk_dtype,
                )
                f.create_dataset("chunks", data=chunk_array)
            else:
                # Create empty dataset
                chunk_dtype = np.dtype(
                    [
                        ("faiss_id", "i8"),
                        ("document_id", h5py.string_dtype(encoding="utf-8")),
                        ("chunk_key", h5py.string_dtype(encoding="utf-8")),
                    ]
                )
                f.create_dataset("chunks", shape=(0,), dtype=chunk_dtype)

            # Store documents as nested groups
            docs_group = f.create_group("documents")
            for doc_id, doc in self.documents.items():
                doc_group = docs_group.create_group(doc_id)

                # Store attributes
                doc_group.attrs["source"] = doc.source

                # Store full_text as dataset
                if doc.full_text:
                    doc_group.create_dataset("full_text", data=doc.full_text)

                # Store content as JSON string
                if doc.content:
                    content_str = json.dumps(doc.content)
                    doc_group.create_dataset("content", data=content_str)

                # Store metadata as JSON string
                if doc.metadata:
                    metadata_str = json.dumps(doc.metadata)
                    doc_group.create_dataset("metadata", data=metadata_str)

                # Store chunks as JSON string (contains faiss_id, i_start, i_end)
                if doc.chunks:
                    chunks_dict = {
                        key: {
                            "faiss_id": chunk.faiss_id,
                            "i_start": chunk.i_start,
                            "i_end": chunk.i_end,
                        }
                        for key, chunk in doc.chunks.items()
                    }
                    chunks_str = json.dumps(chunks_dict)
                    doc_group.create_dataset("chunks", data=chunks_str)

    def save(self) -> None:
        """Persist index and metadata to HDF5 file."""
        self.save_to_hdf5(self.hdf5_path)
