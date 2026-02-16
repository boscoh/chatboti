"""FAISS-based RAG service with FAISS index and JSON metadata storage."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from microeval.llm import SimpleLLMClient

from chatboti.document import ChunkRef, ChunkResult, Document
from chatboti.loaders import load_documents
from chatboti.utils import make_slug

logger = logging.getLogger(__name__)


class FaissRAGService:
    """RAG service using FAISS for vector indexing and JSON for metadata storage.

    This is the standard multi-file backend format. For single-file storage,
    use HDF5RAGService instead.
    """

    def __init__(
        self,
        embed_client: SimpleLLMClient,
        data_dir: Optional[Path] = None,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ):
        """Initialize RAG service (lazy - call via context manager for async setup).

        Usage:
            from microeval.llm import get_llm_client

            embed_client = get_llm_client("ollama", model="nomic-embed-text")
            await embed_client.connect()

            async with FaissRAGService(embed_client=embed_client) as rag:
                results = await rag.search("query")

            await embed_client.close()

        :param embed_client: Connected embedding client (from microeval.llm.get_llm_client)
        :param data_dir: Data directory (default: chatboti/data)
        :param index_path: Path to FAISS index file (overrides auto-detection)
        :param metadata_path: Path to metadata JSON file (overrides auto-detection)
        """
        # Store parameters (initialization happens in __aenter__)
        self.embed_client = embed_client
        self.data_dir = data_dir
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_dim = None
        self.model_name = None
        self._initialized = False

        # These will be initialized in __aenter__
        self.index = None
        self.chunk_refs = []
        self.documents = {}

    async def get_embedding(self, text: str) -> np.ndarray:
        """Get text embedding as numpy array.

        :param text: Text to embed
        :return: Embedding array with shape (1, embedding_dim)
        """
        embedding_list = await self.embed_client.embed(text)
        return np.array(embedding_list, dtype=np.float32).reshape(1, -1)

    async def __aenter__(self):
        """Async context manager entry - performs async initialization."""
        if self._initialized:
            return self

        # Get model name from embed client
        self.model_name = getattr(self.embed_client, 'model', None)

        # Create model slug for file paths
        if self.model_name:
            model_slug = make_slug(self.model_name, strip_latest=True)
        else:
            model_slug = "default"

        # Set up paths
        if not self.data_dir:
            import chatboti
            self.data_dir = Path(chatboti.__file__).parent / "data"

        if not self.index_path:
            self.index_path = self.data_dir / f"vectors-{model_slug}.faiss"
        if not self.metadata_path:
            self.metadata_path = self.data_dir / f"metadata-{model_slug}.json"

        # Detect or load embedding dimension
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                metadata = json.load(f)
                self.embedding_dim = metadata.get('embedding_dim', 768)
        else:
            test_embedding = await self.embed_client.embed("test")
            self.embedding_dim = len(test_embedding)

        # Load or create index and metadata
        self.initialize_search_backend()

        self._initialized = True
        return self

    def initialize_search_backend(self):
        """Load or create search backend (FAISS index) and metadata (sync operation)."""
        # Load or create FAISS index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Load or create JSON metadata
        if self.metadata_path.exists():
            data = json.load(open(self.metadata_path))
            self.chunk_refs = [ChunkRef(**r) for r in data['chunk_refs']]
            self.documents = {d['id']: Document.from_dict(d) for d in data['documents']}
            self.model_name = data.get('model_name', self.model_name)
            stored_dim = data.get('embedding_dim')
            if stored_dim and stored_dim != self.embedding_dim:
                logger.warning(f"Stored dimension {stored_dim} != provided {self.embedding_dim}")
        else:
            self.chunk_refs = []
            self.documents = {}

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit.

        Note: Does not close embed_client since it's managed externally.
        """
        return False

    async def add_document(self, doc: Document) -> None:
        """Add document and its chunk embeddings to index.

        :param doc: Document with chunks to add
        """
        for chunk_key, chunk in doc.chunks.items():
            faiss_id = len(self.chunk_refs)

            # Generate embedding
            chunk_text = doc.get_chunk_text(chunk_key)
            embedding = await self.get_embedding(chunk_text)

            # Add to FAISS index
            self.index.add(embedding)

            # Track metadata
            self.chunk_refs.append(ChunkRef(document_id=doc.id, chunk_key=chunk_key))
            chunk.faiss_id = faiss_id

        self.documents[doc.id] = doc

    async def build_embeddings_from_documents(self, source: str) -> None:
        """Load documents from source and build embeddings.

        :param source: File path or pattern (e.g., "papers.json", "docs/*.md")
        """
        documents = await load_documents(source)
        for doc in documents:
            await self.add_document(doc)
        self.save()

    def save(self) -> None:
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        self.save_metadata()

    def vector_search(self, query_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Vector similarity search wrapper (override for cloud backends).

        :param query_emb: Query embedding shape (1, dim)
        :param k: Number of results
        :return: (distances, faiss_ids) both shape (k,) - first result set
        """
        distances, faiss_ids = self.index.search(query_emb, k)
        return distances[0], faiss_ids[0]

    def get_chunk_refs(self, faiss_ids: List[int]) -> List[ChunkRef]:
        """Fetch chunk references (override in subclasses).

        :param faiss_ids: List of FAISS indices
        :return: List of chunk references
        """
        return [self.chunk_refs[fid] for fid in faiss_ids]

    def get_chunk_text(self, ref: ChunkRef) -> str:
        """Fetch text for a single chunk (override in subclasses).

        :param ref: Chunk reference
        :return: Chunk text
        """
        doc = self.documents[ref.document_id]
        chunk = doc.chunks[ref.chunk_key]
        if chunk.i_start is not None:
            # Chunk-level: slice from full_text
            return doc.full_text[chunk.i_start:chunk.i_end]
        else:
            # Field-level: get from content
            return doc.content[ref.chunk_key]

    def get_document_texts(self, doc_ids: List[str]) -> Dict[str, str]:
        """Fetch full document text (override in subclasses).

        :param doc_ids: List of document IDs
        :return: Dict mapping document_id to full text
        """
        result = {}
        for doc_id in doc_ids:
            doc = self.documents[doc_id]
            # Return full_text if available, otherwise JSON of content
            if doc.full_text:
                result[doc_id] = doc.full_text
            else:
                result[doc_id] = json.dumps(doc.content, indent=2)
        return result

    def save_metadata(self) -> None:
        """Save metadata to JSON (override in subclasses)."""
        data = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'chunk_refs': [{'document_id': r.document_id, 'chunk_key': r.chunk_key} for r in self.chunk_refs],
            'documents': [doc.to_dict() for doc in self.documents.values()]
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)

    async def search(self, query: str, k: int = 5, include_documents: bool = False) -> List[ChunkResult]:
        """Search for relevant chunks.

        :param query: Search query text
        :param k: Number of results to return
        :param include_documents: Include full document in results
        :return: List of chunk results with text
        """
        # 1. Embed query → shape (1, embedding_dim) for batch processing
        query_emb = await self.get_embedding(query)

        # 2. Vector search returns (distances, faiss_ids) each shape (k,)
        # distances: k distances to nearest neighbors (float32)
        # faiss_ids: k indices into vector index (int64)
        distances, faiss_ids = self.vector_search(query_emb, k)

        # 3. Filter out -1 indices (returned when index is empty or not enough results)
        valid_ids = [fid for fid in faiss_ids.tolist() if fid >= 0]
        if not valid_ids:
            return []

        # 5. Build results
        results = []
        for ref in self.get_chunk_refs(valid_ids):
            doc = self.documents[ref.document_id]
            result = ChunkResult(
                document_id=ref.document_id,
                chunk_key=ref.chunk_key,
                text=self.get_chunk_text(ref)
            )
            if doc.full_text:
                result.document_text = doc.full_text
            if include_documents and doc.content:
                result.content = doc.content
            results.append(result)
        return results
