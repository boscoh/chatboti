"""SQLite-based RAG service using sqlite-vec for native KNN vector search.

All state lives in a single SQLite database - no in-memory docs/chunks after init.
Vector index ranges per document enable efficient document-scoped filtered search
via rowid BETWEEN.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np
import sqlite_vec
from chatboti.llm import SimpleLLMClient

from chatboti.document import ChunkResult, Document
from chatboti.faiss_rag import FaissRAGService


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector so cosine similarity = dot product on the unit sphere.

    :param vec: Array of shape (dim,) or (1, dim)
    :return: Normalized array of the same shape
    """
    arr = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


class SQLiteRAGService(FaissRAGService):
    """RAG service backed by SQLite + sqlite-vec for single-file storage.

    Schema overview:
    - ``vec_items``: sqlite-vec virtual table holding normalized float32 embeddings
    - ``documents``: document metadata and vector range (i_vector_start/i_vector_end)
    - ``chunks``: per-chunk metadata; ``id`` matches rowid in vec_items
    - ``rag_metadata``: key/value store for model_name and embedding_dim

    All writes are committed transactionally inside ``add_document`` so the DB
    is always in a consistent state without calling ``save()`` first.

    Usage::

        async with SQLiteRAGService(embed_client=embed_client) as rag:
            results = await rag.search("query")
    """

    def __init__(
        self,
        embed_client: SimpleLLMClient,
        data_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        """Initialize SQLite RAG service.

        :param embed_client: Connected embedding client
        :param data_dir: Data directory (default: chatboti/data)
        :param db_path: Path to SQLite database file (overrides auto-detection)
        """
        super().__init__(
            embed_client=embed_client,
            data_dir=data_dir,
            index_path=None,
            metadata_path=None,
        )
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    async def __aenter__(self):
        """Async context manager entry - performs async initialization."""
        await super().__aenter__()

        if not self.db_path:
            from chatboti.utils import make_slug

            model_slug = (
                make_slug(self.model_name, strip_latest=True)
                if self.model_name
                else "default"
            )
            self.db_path = self.data_dir / f"embeddings-{model_slug}.db"

        return self

    def initialize_search_backend(self):
        """Open connection, load extension, create tables, load stored metadata."""
        self._conn = sqlite3.connect(str(self.db_path) if self.db_path else ":memory:")
        sqlite_vec.load(self._conn)

        # Create tables if they don't exist yet.  vec_items requires knowing
        # embedding_dim at table-creation time; we skip it until __aenter__
        # sets self.embedding_dim and calls this method via super().__aenter__.
        self._conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                full_text TEXT,
                content TEXT,
                metadata TEXT,
                i_vector_start INTEGER,
                i_vector_end INTEGER
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                document_id TEXT REFERENCES documents(id),
                chunk_key TEXT,
                i_start INTEGER,
                i_end INTEGER
            );
            CREATE TABLE IF NOT EXISTS rag_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)

        # Create vec_items only when it doesn't already exist.
        # sqlite-vec virtual tables cannot use IF NOT EXISTS directly in all
        # versions, so we check via sqlite_master first.
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_items'"
        )
        if cur.fetchone() is None:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{self.embedding_dim}])"
            )
        self._conn.commit()

        # Load model_name / embedding_dim from stored metadata if available.
        cur = self._conn.execute(
            "SELECT key, value FROM rag_metadata WHERE key IN ('model_name', 'embedding_dim')"
        )
        for key, value in cur.fetchall():
            if key == "model_name":
                self.model_name = value
            elif key == "embedding_dim":
                stored_dim = int(value)
                if stored_dim != self.embedding_dim:
                    import logging

                    logging.getLogger(__name__).warning(
                        f"Stored embedding_dim {stored_dim} != current {self.embedding_dim}"
                    )
                self.embedding_dim = stored_dim

        # Keep in-memory structures empty - DB is the source of truth.
        self.chunk_refs = []
        self.documents = {}

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        return False

    async def add_document(self, doc: Document) -> None:
        """Embed all chunks of *doc* and persist to the database.

        :param doc: Document with chunks to add
        """
        cur = self._conn.execute("SELECT COUNT(*) FROM chunks")
        doc.i_vector_start = cur.fetchone()[0]

        chunk_counter = doc.i_vector_start
        for chunk_key, chunk in doc.chunks.items():
            chunk_text = doc.get_chunk_text(chunk_key)
            embedding = await self.get_embedding(chunk_text)
            norm_vec = _l2_normalize(embedding)

            self._conn.execute(
                "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                (chunk_counter, norm_vec.tobytes()),
            )
            self._conn.execute(
                "INSERT INTO chunks(id, document_id, chunk_key, i_start, i_end) VALUES (?, ?, ?, ?, ?)",
                (chunk_counter, doc.id, chunk_key, chunk.i_start, chunk.i_end),
            )
            chunk.id = chunk_counter
            chunk_counter += 1

        doc.i_vector_end = chunk_counter

        self._conn.execute(
            """INSERT OR REPLACE INTO documents
               (id, full_text, content, metadata, i_vector_start, i_vector_end)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                doc.id,
                doc.full_text or "",
                json.dumps(doc.content),
                json.dumps(doc.metadata),
                doc.i_vector_start,
                doc.i_vector_end,
            ),
        )
        self._conn.commit()

    def vector_search(
        self, query_emb: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full-collection KNN search using sqlite-vec.

        :param query_emb: Query embedding shape (1, dim) or (dim,)
        :param k: Number of results
        :return: (distances, rowids) both shape (k,)
        """
        norm_vec = _l2_normalize(query_emb)
        cur = self._conn.execute(
            "SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (norm_vec.tobytes(), k),
        )
        rows = cur.fetchall()
        if not rows:
            return np.array([]), np.array([], dtype=np.int64)
        distances = np.array([r[1] for r in rows], dtype=np.float32)
        indices = np.array([r[0] for r in rows], dtype=np.int64)
        return distances, indices

    def _rowid_search(
        self, norm_vec: np.ndarray, k: int, i_start: int, i_end_exclusive: int
    ) -> list[tuple[int, float]]:
        """KNN search restricted to rowid range [i_start, i_end_exclusive - 1].

        :param norm_vec: Normalized query vector (1-D float32)
        :param k: Number of results
        :param i_start: Inclusive lower bound for rowid
        :param i_end_exclusive: Exclusive upper bound for rowid
        :return: List of (rowid, distance) pairs
        """
        cur = self._conn.execute(
            """SELECT rowid, distance FROM vec_items
               WHERE embedding MATCH ?
                 AND rowid BETWEEN ? AND ?
               ORDER BY distance LIMIT ?""",
            (norm_vec.tobytes(), i_start, i_end_exclusive - 1, k),
        )
        return cur.fetchall()

    async def search(
        self,
        query: str,
        k: int = 5,
        include_documents: bool = False,
        doc_ids: Optional[List[str]] = None,
    ) -> List[ChunkResult]:
        """Search for relevant chunks, optionally restricted to selected documents.

        :param query: Search query text
        :param k: Number of results to return
        :param include_documents: Include full document text in results
        :param doc_ids: If given, restrict search to these document IDs
        :return: List of ChunkResult instances
        """
        query_emb = await self.get_embedding(query)
        norm_vec = _l2_normalize(query_emb)

        if doc_ids is not None:
            # Per-document rowid-range search, then merge and re-rank.
            all_rows: list[tuple[int, float]] = []
            for doc_id in doc_ids:
                cur = self._conn.execute(
                    "SELECT i_vector_start, i_vector_end FROM documents WHERE id = ?",
                    (doc_id,),
                )
                row = cur.fetchone()
                if row is None:
                    continue
                i_start, i_end = row
                all_rows.extend(self._rowid_search(norm_vec, k, i_start, i_end))

            # Sort merged results by distance ascending, keep top-k.
            all_rows.sort(key=lambda r: r[1])
            rowids = [r[0] for r in all_rows[:k]]
        else:
            _, idx_arr = self.vector_search(query_emb, k)
            rowids = idx_arr.tolist()

        if not rowids:
            return []

        results: List[ChunkResult] = []
        for rowid in rowids:
            chunk_cur = self._conn.execute(
                "SELECT document_id, chunk_key, i_start, i_end FROM chunks WHERE id = ?",
                (rowid,),
            )
            chunk_row = chunk_cur.fetchone()
            if chunk_row is None:
                continue
            document_id, chunk_key, i_start, i_end = chunk_row

            doc_cur = self._conn.execute(
                "SELECT full_text, content FROM documents WHERE id = ?",
                (document_id,),
            )
            doc_row = doc_cur.fetchone()
            if doc_row is None:
                continue
            full_text, content_json = doc_row
            content = json.loads(content_json) if content_json else {}

            # Determine chunk text.
            if i_start is not None:
                text = full_text[i_start:i_end]
            else:
                text = content.get(chunk_key, "")

            result = ChunkResult(
                document_id=document_id,
                chunk_key=chunk_key,
                text=text,
            )
            if full_text:
                result.document_text = full_text
            if include_documents and content:
                result.content = content
            results.append(result)

        return results

    def save(self) -> None:
        """Persist model_name and embedding_dim into rag_metadata.

        All document/chunk/vector data is already committed during add_document.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO rag_metadata(key, value) VALUES ('model_name', ?)",
            (self.model_name or "",),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO rag_metadata(key, value) VALUES ('embedding_dim', ?)",
            (str(self.embedding_dim),),
        )
        self._conn.commit()
