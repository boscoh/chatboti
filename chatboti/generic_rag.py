"""Generic RAG service with FAISS index and JSON metadata storage."""

from pathlib import Path
from typing import List, Dict, Optional
import json
import faiss
import numpy as np
from collections import defaultdict

from chatboti.document import Document, DocumentChunk, ChunkRef, ChunkResult
from chatboti.loaders import CSVDocumentLoader, DocumentLoader
from chatboti.embed_client import EmbedClient


class GenericRAGService:
    """RAG service with FAISS index and JSON metadata storage.

    For SQLite storage, use SQLiteRAGService subclass.
    """

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_dim: int = 1536,
        embed_client: Optional[EmbedClient] = None
    ):
        """Initialize RAG service.

        :param index_path: Path to FAISS index file
        :param metadata_path: Path to metadata JSON file
        :param embedding_dim: Embedding dimension
        :param embed_client: Embedding client (e.g., OpenAI, Ollama)
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embed_client = embed_client

        # Load or create FAISS index
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)

        # Load or create JSON metadata
        if metadata_path.exists():
            data = json.load(open(metadata_path))
            self.chunk_refs: List[ChunkRef] = [ChunkRef(**r) for r in data['chunk_refs']]
            self.documents: Dict[str, Document] = {d['id']: Document.from_dict(d) for d in data['documents']}
        else:
            self.chunk_refs = []
            self.documents = {}

    async def add_document(self, doc: Document) -> None:
        """Add document and its chunk embeddings to index.

        :param doc: Document with chunks to add
        """
        for chunk_key, chunk in doc.chunks.items():
            faiss_id = len(self.chunk_refs)

            # Generate embedding
            chunk_text = doc.get_chunk_text(chunk_key)
            embedding_list = await self.embed_client.embed(chunk_text)
            embedding = np.array(embedding_list, dtype=np.float32).reshape(1, -1)

            # Add to FAISS index
            self.index.add(embedding)

            # Track metadata
            self.chunk_refs.append(ChunkRef(document_id=doc.id, chunk_key=chunk_key))
            chunk.faiss_id = faiss_id

        self.documents[doc.id] = doc

    def _get_loader(self, source: str) -> DocumentLoader:
        """Get appropriate loader based on file extension.

        :param source: File path
        :return: Document loader instance
        """
        ext = Path(source).suffix.lower()
        if ext == '.csv':
            return CSVDocumentLoader()
        raise ValueError(f"Unsupported file type: {ext}. Only .csv supported currently.")

    async def build_embeddings_from_documents(self, source: str, doc_type: str) -> None:
        """Load documents from source and build embeddings.

        :param source: File path or pattern (e.g., "papers.json", "docs/*.md")
        :param doc_type: Document type identifier
        """
        loader = self._get_loader(source)
        documents = await loader.load(source, doc_type)
        for doc in documents:
            await self.add_document(doc)
        self.save()

    def save(self) -> None:
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, str(self.index_path))
        self._save_metadata()

    def _vector_search(self, query_emb: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Vector similarity search wrapper (override for cloud backends).

        :param query_emb: Query embedding shape (1, dim)
        :param k: Number of results
        :return: (distances, faiss_ids) both shape (1, k)
        """
        return self.index.search(query_emb, k)

    def _get_chunk_refs(self, faiss_ids: List[int]) -> List[ChunkRef]:
        """Fetch chunk references (override in subclasses).

        :param faiss_ids: List of FAISS indices
        :return: List of chunk references
        """
        return [self.chunk_refs[fid] for fid in faiss_ids]

    def _get_chunk_texts(self, refs: List[ChunkRef]) -> Dict[ChunkRef, str]:
        """Fetch only text needed for chunks (override in subclasses).

        :param refs: Chunk references
        :return: Dict mapping ref to chunk text
        """
        # Group by document for efficient fetching
        refs_by_doc = defaultdict(list)
        for ref in refs:
            refs_by_doc[ref.document_id].append(ref)

        # Fetch and extract text
        result = {}
        for doc_id, doc_refs in refs_by_doc.items():
            doc = self.documents[doc_id]
            for ref in doc_refs:
                chunk = doc.chunks[ref.chunk_key]
                if chunk.i_start is not None:
                    # Chunk-level: slice from full_text
                    result[ref] = doc.full_text[chunk.i_start:chunk.i_end]
                else:
                    # Field-level: get from content
                    result[ref] = doc.content[ref.chunk_key]
        return result

    def _get_document_texts(self, doc_ids: List[str]) -> Dict[str, str]:
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

    def _save_metadata(self) -> None:
        """Save metadata to JSON (override in subclasses)."""
        data = {
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
        query_emb = np.array(await self.embed_client.embed(query), dtype=np.float32).reshape(1, -1)

        # 2. Vector search returns (distances, indices)
        # Shape: (n_queries, k) where n_queries=1, k=number of results
        # distances[0]: k distances to nearest neighbors (float32)
        # faiss_ids[0]: k indices into vector index (int64)
        distances, faiss_ids = self._vector_search(query_emb, k)

        # 3. Fetch chunk references (extract [0] since single query: (1, k) → (k,))
        refs: List[ChunkRef] = self._get_chunk_refs(faiss_ids[0].tolist())

        # 4. Fetch chunk texts (optimized)
        chunk_texts = self._get_chunk_texts(refs)

        # 5. Optionally fetch full document texts
        document_texts = None
        if include_documents:
            doc_ids = list(set(ref.document_id for ref in refs))
            document_texts = self._get_document_texts(doc_ids)

        # 6. Build results
        return [
            ChunkResult(
                document_id=ref.document_id,
                chunk_key=ref.chunk_key,
                text=chunk_texts[ref],
                document_text=document_texts[ref.document_id] if document_texts else None
            )
            for ref in refs
        ]
