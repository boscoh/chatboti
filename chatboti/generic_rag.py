"""Generic RAG service with FAISS index and JSON metadata storage."""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import json
import faiss
import numpy as np
from collections import defaultdict

from chatboti.document import Document, DocumentChunk, ChunkRef, ChunkResult
from chatboti.loaders import CSVDocumentLoader, DocumentLoader


class GenericRAGService:
    """RAG service with FAISS index and JSON metadata storage.

    For SQLite storage, use SQLiteRAGService subclass.
    """

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_dim: int = 1536,
        embed_client = None,
        _owns_client: bool = False
    ):
        """Initialize RAG service.

        :param index_path: Path to FAISS index file
        :param metadata_path: Path to metadata JSON file
        :param embedding_dim: Embedding dimension
        :param embed_client: Embedding client (e.g., OpenAI, Ollama)
        :param _owns_client: Internal flag - whether this instance owns the client lifecycle
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embed_client = embed_client
        self.embedding_dim = embedding_dim
        self.model_name = getattr(embed_client, 'model', None)
        self._owns_client = _owns_client

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
            # Load model info if present
            self.model_name = data.get('model_name', self.model_name)
            stored_dim = data.get('embedding_dim')
            if stored_dim and stored_dim != embedding_dim:
                print(f"⚠ Warning: Stored dimension {stored_dim} != provided {embedding_dim}")
        else:
            self.chunk_refs = []
            self.documents = {}

    @staticmethod
    def make_model_slug(model_name: str) -> str:
        """Convert model name to filesystem-safe slug.

        :param model_name: Model name (e.g., 'nomic-embed-text', 'text-embedding-3-small')
        :return: Slug (e.g., 'nomic-embed-text', 'text-embedding-3-small')
        """
        slug = re.sub(r':latest$', '', model_name)
        slug = re.sub(r'[^a-z0-9]+', '-', slug.lower())
        slug = re.sub(r'-+', '-', slug).strip('-')
        return slug

    @staticmethod
    async def detect_embedding_dim(embed_client) -> int:
        """Detect embedding dimension by running a test query.

        :param embed_client: Embedding client
        :return: Embedding dimension
        """
        test_embedding = await embed_client.embed("test")
        return len(test_embedding)

    @classmethod
    async def from_service(
        cls,
        service_name: str,
        model: Optional[str] = None,
        data_dir: Optional[Path] = None,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None
    ) -> "GenericRAGService":
        """Factory method to create RAG service from service name.

        This method handles:
        - Loading model configuration
        - Creating model slug for file paths
        - Connecting embed client
        - Detecting embedding dimensions
        - Loading existing RAG or creating new one

        :param service_name: Service name (e.g., 'ollama', 'openai', 'bedrock')
        :param model: Model name (optional, will load from config if not provided)
        :param data_dir: Data directory (default: chatboti/data)
        :param index_path: Path to FAISS index (overrides data_dir default)
        :param metadata_path: Path to metadata JSON (overrides data_dir default)
        :return: Initialized GenericRAGService instance
        """
        from microeval.llm import get_llm_client, load_config

        # Load model configuration
        model_config = load_config()
        embed_models = model_config.get("embed_models", {})

        # Get model name
        if not model:
            model = os.getenv("EMBED_MODEL") or embed_models.get(service_name)
        if not model:
            raise ValueError(
                f"Model not specified and EMBED_MODEL not set for service '{service_name}'. "
                f"Available models in config: {list(embed_models.keys())}"
            )

        # Create model slug for file paths
        model_slug = cls.make_model_slug(model)

        # Set up paths
        if not data_dir:
            # Default to chatboti/data
            import chatboti
            data_dir = Path(chatboti.__file__).parent / "data"

        if not index_path:
            index_path = data_dir / f"vectors-{model_slug}.faiss"
        if not metadata_path:
            metadata_path = data_dir / f"metadata-{model_slug}.json"

        # Create and connect embed client
        embed_client = get_llm_client(service_name, model=model)
        await embed_client.connect()

        # Detect or load embedding dimension
        if metadata_path.exists():
            # Load from metadata
            with open(metadata_path) as f:
                metadata = json.load(f)
                embedding_dim = metadata.get('embedding_dim', 768)
        else:
            # Detect by running test query
            embedding_dim = await cls.detect_embedding_dim(embed_client)

        # Create service instance (marks that it owns the client)
        return cls(
            index_path=index_path,
            metadata_path=metadata_path,
            embedding_dim=embedding_dim,
            embed_client=embed_client,
            _owns_client=True
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup embed client if we own it."""
        if self._owns_client and self.embed_client:
            await self.embed_client.close()
        return False

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
        query_emb = np.array(await self.embed_client.embed(query), dtype=np.float32).reshape(1, -1)

        # 2. Vector search returns (distances, indices)
        # Shape: (n_queries, k) where n_queries=1, k=number of results
        # distances[0]: k distances to nearest neighbors (float32)
        # faiss_ids[0]: k indices into vector index (int64)
        distances, faiss_ids = self._vector_search(query_emb, k)

        # 3. Filter out -1 indices (returned when index is empty or not enough results)
        valid_ids = [fid for fid in faiss_ids[0].tolist() if fid >= 0]
        if not valid_ids:
            return []

        # 4. Fetch chunk references
        refs: List[ChunkRef] = self._get_chunk_refs(valid_ids)

        # 5. Fetch chunk texts (optimized)
        chunk_texts = self._get_chunk_texts(refs)

        # 6. Optionally fetch full document texts
        document_texts = None
        if include_documents:
            doc_ids = list(set(ref.document_id for ref in refs))
            document_texts = self._get_document_texts(doc_ids)

        # 7. Build results
        return [
            ChunkResult(
                document_id=ref.document_id,
                chunk_key=ref.chunk_key,
                text=chunk_texts[ref],
                document_text=document_texts[ref.document_id] if document_texts else None
            )
            for ref in refs
        ]
