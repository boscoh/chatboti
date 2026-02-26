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
import h5py
import numpy as np
from chatboti.llm import SimpleLLMClient

# Local
from chatboti.document import ChunkRef, Document
from chatboti.faiss_rag import FaissRAGService


class VectorBuffer:
    """In-memory buffer of pre-normalized float32 embedding vectors.

    Vectors are L2-normalized on insert so that cosine similarity reduces to
    a single dot product (matrix multiply) at search time.

    Vectors are accumulated in a list and materialized into a contiguous array
    lazily, so add() is O(1) per call rather than O(n) copy.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self._rows: list[np.ndarray] = []
        self._data: Optional[np.ndarray] = None

    def add(self, vectors: np.ndarray) -> None:
        """Append and L2-normalize one or more vectors.

        :param vectors: Array of shape (dim,), (1, dim), or (n, dim)
        """
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.where(norms > 0, norms, 1.0)
        self._rows.append(arr)
        self._data = None  # invalidate materialized cache

    @property
    def data(self) -> np.ndarray:
        """Contiguous float32 array of shape (ntotal, embedding_dim)."""
        if self._data is None:
            self._data = (
                np.vstack(self._rows)
                if self._rows
                else np.empty((0, self.embedding_dim), dtype=np.float32)
            )
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """Assign a pre-built array (e.g. loaded from HDF5) directly."""
        self._data = value
        self._rows = []

    @property
    def ntotal(self) -> int:
        if self._data is not None:
            return len(self._data)
        return sum(len(r) for r in self._rows)


def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Dot-product similarity against pre-normalized vectors.

    Assumes vectors are already L2-normalized (as stored by VectorBuffer).

    :param query: Query vector shape (1, dim) or (dim,)
    :param vectors: Pre-normalized matrix shape (n, dim)
    :return: Array of similarities shape (n,)
    """
    q = query.flatten()
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    return vectors @ q_norm


class HDF5RAGService(FaissRAGService):
    """RAG service using HDF5 single-file backend with numpy vector store.

    Storage format:
        /metadata (attributes: model_name, embedding_dim, created_at, etc.)
        /vectors (dataset: float32 array [n_chunks × embedding_dim])
        /chunks (dataset: structured array with id, document_id, chunk_key)
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
        super().__init__(
            embed_client=embed_client,
            data_dir=data_dir,
            index_path=None,
            metadata_path=None,
        )
        self.hdf5_path = hdf5_path
        self.vectors: Optional[VectorBuffer] = None

    async def __aenter__(self):
        """Async context manager entry - performs async initialization."""
        await super().__aenter__()

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
        """Load or create vector buffer from HDF5 file."""
        if self.hdf5_path.exists():
            self.load_from_hdf5(self.hdf5_path)
        else:
            self.vectors = VectorBuffer(self.embedding_dim)
            self.chunk_refs = []
            self.documents = {}

    async def add_document(self, doc: Document) -> None:
        """Add document and its chunk embeddings to the vector buffer.

        :param doc: Document with chunks to add
        """
        doc.i_vector_start = len(self.chunk_refs)
        for chunk_key, chunk in doc.chunks.items():
            chunk_id = len(self.chunk_refs)
            chunk_text = doc.get_chunk_text(chunk_key)
            embedding = await self.get_embedding(chunk_text)
            self.vectors.add(embedding)
            self.chunk_refs.append(ChunkRef(document_id=doc.id, chunk_key=chunk_key))
            chunk.id = chunk_id
        doc.i_vector_end = len(self.chunk_refs)
        self.documents[doc.id] = doc

    def vector_search(
        self, query_emb: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cosine similarity search over the vector buffer.

        :param query_emb: Query embedding shape (1, dim)
        :param k: Number of results
        :return: (similarities, indices) both shape (k,)
        """
        if self.vectors.ntotal == 0:
            return np.array([]), np.array([], dtype=np.int64)
        sims = cosine_similarity(query_emb, self.vectors.data)
        k = min(k, self.vectors.ntotal)
        top_k = np.argpartition(sims, -k)[-k:]
        top_k = top_k[np.argsort(sims[top_k])[::-1]]
        return sims[top_k], top_k.astype(np.int64)

    def load_from_hdf5(self, path: Path) -> None:
        """Load vector buffer, chunks, and documents from HDF5 file.

        :param path: Path to HDF5 file
        """
        with h5py.File(path, "r") as f:
            self.model_name = f.attrs.get("model_name", self.model_name)
            self.embedding_dim = int(f.attrs["embedding_dim"])

            self.vectors = VectorBuffer(self.embedding_dim)
            if "vectors" in f:
                data = f["vectors"][:].astype(np.float32)
                if len(data) > 0:
                    # Normalize on load to handle both old (raw) and new (pre-normalized) files
                    norms = np.linalg.norm(data, axis=1, keepdims=True)
                    self.vectors.data = data / np.where(norms > 0, norms, 1.0)

            if "chunks" in f:
                chunk_data = f["chunks"][:]
                self.chunk_refs = []
                for row in chunk_data:
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

            if "documents" in f:
                self.documents = {}
                docs_group = f["documents"]
                for doc_id_raw in docs_group.keys():
                    doc_id = doc_id_raw
                    if isinstance(doc_id, bytes):
                        doc_id = doc_id.decode("utf-8")

                    doc_group = docs_group[doc_id_raw]

                    doc_data = {
                        "id": doc_id,
                        "i_vector_start": doc_group.attrs.get("i_vector_start", None),
                        "i_vector_end": doc_group.attrs.get("i_vector_end", None),
                        "full_text": "",
                        "content": {},
                        "metadata": {},
                        "chunks": {},
                    }
                    # Back-compat: old files stored source as HDF5 attribute
                    if "source" in doc_group.attrs:
                        doc_data["source"] = doc_group.attrs["source"]

                    if "full_text" in doc_group:
                        doc_data["full_text"] = doc_group["full_text"][()]
                        if isinstance(doc_data["full_text"], bytes):
                            doc_data["full_text"] = doc_data["full_text"].decode(
                                "utf-8"
                            )

                    if "content" in doc_group:
                        content_str = doc_group["content"][()]
                        if isinstance(content_str, bytes):
                            content_str = content_str.decode("utf-8")
                        doc_data["content"] = json.loads(content_str)

                    if "metadata" in doc_group:
                        metadata_str = doc_group["metadata"][()]
                        if isinstance(metadata_str, bytes):
                            metadata_str = metadata_str.decode("utf-8")
                        doc_data["metadata"] = json.loads(metadata_str)

                    if "chunks" in doc_group:
                        chunks_str = doc_group["chunks"][()]
                        if isinstance(chunks_str, bytes):
                            chunks_str = chunks_str.decode("utf-8")
                        doc_data["chunks"] = json.loads(chunks_str)

                    self.documents[doc_id] = Document.from_dict(doc_data)
            else:
                self.documents = {}

    def save_to_hdf5(self, path: Path) -> None:
        """Save vector buffer, chunks, and documents to HDF5 file.

        :param path: Path to HDF5 file
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as f:
            f.attrs["model_name"] = self.model_name or ""
            f.attrs["embedding_dim"] = self.embedding_dim
            f.attrs["created_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            f.attrs["format_version"] = "1.0"
            f.attrs["vector_count"] = self.vectors.ntotal
            f.attrs["document_count"] = len(self.documents)

            # Write vectors directly from buffer
            if self.vectors.ntotal > 0:
                f.create_dataset("vectors", data=self.vectors.data, compression="gzip")
            else:
                f.create_dataset(
                    "vectors", shape=(0, self.embedding_dim), dtype=np.float32
                )

            if self.chunk_refs:
                chunk_dtype = np.dtype(
                    [
                        ("id", "i8"),
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
                chunk_dtype = np.dtype(
                    [
                        ("id", "i8"),
                        ("document_id", h5py.string_dtype(encoding="utf-8")),
                        ("chunk_key", h5py.string_dtype(encoding="utf-8")),
                    ]
                )
                f.create_dataset("chunks", shape=(0,), dtype=chunk_dtype)

            docs_group = f.create_group("documents")
            for doc_id, doc in self.documents.items():
                doc_group = docs_group.create_group(doc_id)
                if doc.i_vector_start is not None:
                    doc_group.attrs["i_vector_start"] = doc.i_vector_start
                if doc.i_vector_end is not None:
                    doc_group.attrs["i_vector_end"] = doc.i_vector_end

                if doc.full_text:
                    doc_group.create_dataset("full_text", data=doc.full_text)

                if doc.content:
                    doc_group.create_dataset(
                        "content", data=json.dumps(doc.content)
                    )

                if doc.metadata:
                    doc_group.create_dataset(
                        "metadata", data=json.dumps(doc.metadata)
                    )

                if doc.chunks:
                    chunks_dict = {
                        key: {
                            "id": chunk.id,
                            "i_start": chunk.i_start,
                            "i_end": chunk.i_end,
                        }
                        for key, chunk in doc.chunks.items()
                    }
                    doc_group.create_dataset(
                        "chunks", data=json.dumps(chunks_dict)
                    )

    async def search(
        self,
        query: str,
        k: int = 5,
        include_documents: bool = False,
        doc_ids: Optional[list[str]] = None,
    ) -> list:
        """Search for relevant chunks, optionally restricted to selected documents.

        :param query: Search query text
        :param k: Number of results to return
        :param include_documents: Include full document in results
        :param doc_ids: If given, restrict search to these document IDs
        :return: List of chunk results with text
        """
        query_emb = await self.get_embedding(query)

        if doc_ids is not None:
            valid_ids = self._filtered_search(query_emb, k, doc_ids)
        else:
            _, faiss_ids = self.vector_search(query_emb, k)
            valid_ids = [fid for fid in faiss_ids.tolist() if fid >= 0]

        if not valid_ids:
            return []

        results = []
        for ref in self.get_chunk_refs(valid_ids):
            doc = self.documents[ref.document_id]
            from chatboti.document import ChunkResult
            result = ChunkResult(
                document_id=ref.document_id,
                chunk_key=ref.chunk_key,
                text=self.get_chunk_text(ref),
            )
            if doc.full_text:
                result.document_text = doc.full_text
            if include_documents and doc.content:
                result.content = doc.content
            results.append(result)
        return results

    def _filtered_search(
        self, query_emb: np.ndarray, k: int, doc_ids: list[str]
    ) -> list[int]:
        """Return global vector indices of top-k results within selected documents.

        Slices each document's contiguous vector range directly from the buffer,
        so only the selected documents' vectors are touched.

        :param query_emb: Query embedding shape (1, dim)
        :param k: Number of results
        :param doc_ids: Document IDs to search within
        :return: List of global vector indices (sorted by similarity, descending)
        """
        global_indices = []
        for doc_id in doc_ids:
            doc = self.documents[doc_id]
            if doc.i_vector_start is None:
                continue
            n = len(doc.chunks)
            global_indices.extend(range(doc.i_vector_start, doc.i_vector_start + n))

        if not global_indices:
            return []

        idx = np.array(global_indices, dtype=np.int64)
        sims = cosine_similarity(query_emb, self.vectors.data[idx])
        k_actual = min(k, len(idx))
        top_local = np.argpartition(sims, -k_actual)[-k_actual:]
        top_local = top_local[np.argsort(sims[top_local])[::-1]]
        return idx[top_local].tolist()

    def save(self) -> None:
        """Persist vector buffer and metadata to HDF5 file."""
        self.save_to_hdf5(self.hdf5_path)
