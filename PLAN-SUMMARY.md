# Generic Document Storage - Plan Summary

## Goal
Transform chatboti RAG from speaker-specific → generic document storage system

## Current Problems
1. Hardcoded speaker fields (bio_max_120_words, final_abstract_max_150_words)
2. Single CSV source - can't load JSON, PDF, web pages
3. Fixed structure - exactly 2 embeddings per document
4. Inefficient storage - List[float] (8x memory, 6-8x storage waste)
5. Slow search - linear scan, no indexing (50-100ms for 10K docs)

## Solution Architecture

### Storage Evolution

**Phase 1 - JSON Metadata:**
```
documents/
├─ vectors.faiss    # FAISS index (float32 vectors)
└─ metadata.json    # Document metadata
```
• Simple, human-readable, O(1) dict lookup
• Good for <1K documents

**Phase 2 - SQLite Metadata:**
```
documents/
├─ vectors.faiss    # FAISS index (float32 vectors)
└─ metadata.db      # SQLite with indexed queries
```
• ACID transactions, indexed queries
• Better for >1K documents
• **CRITICAL**: faiss_id must be PRIMARY KEY for O(log n) lookups

### Multi-Model Support

Model-specific storage (384-3072 dimensions):
```
├─ vectors_openai-text-embedding-3-small.faiss  # 1536-dim
├─ metadata_openai-text-embedding-3-small.json
├─ vectors_nomic-nomic-embed-text.faiss         # 768-dim
└─ metadata_nomic-nomic-embed-text.json
```

Common models:
| Model | Dims | Storage (1K) | Speed |
|-------|------|-------------|-------|
| all-MiniLM-L6-v2 | 384 | 1.5 MB | 0.5ms |
| nomic-embed-text | 768 | 3.0 MB | 1.0ms |
| text-embedding-3-small | 1536 | 6.0 MB | 2.0ms |
| text-embedding-3-large | 3072 | 12.0 MB | 4.0ms |

### Embedding Strategies

**Field-level** (structured rows like CSV)
• One document per row
• Multiple embeddings per document (one per field)
• Returns entire row when any field matches

**Chunk-level** (long documents like PDFs)
• Document split into chunks
• One embedding per chunk
• Returns matching chunks with text

### Core Abstractions

```python
class Document:
    id: str
    content: dict          # Flexible fields (for field-level)
    full_text: str         # Complete text (for chunk-level)
    metadata: dict         # Source, timestamp, etc.
    chunks: Dict[str, DocumentChunk]  # Key = field name (field-level) or index (chunk-level)

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

class DocumentChunk:
    """Chunk reference with optional text indices.

    Field-level: dict key is field name, i_start/i_end are None
    Chunk-level: dict key is index, i_start/i_end locate text
    Global ID: (document.id, dict_key)
    """
    faiss_id: int
    i_start: Optional[int] = None
    i_end: Optional[int] = None

@dataclass
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
    document_text: Optional[str] = None  # Full document text if include_documents=True

class DocumentLoader:
    """Base class for loading documents from various sources."""

    def load(self, source: str, doc_type: str) -> List[Document]:
        """Load documents from source.

        :param source: File path
        :param doc_type: Document type identifier
        :return: List of documents with chunks
        """
        raise NotImplementedError

class CSVDocumentLoader(DocumentLoader):
    """Load documents from CSV files with field-level embeddings."""

    def __init__(self, embed_fields: List[str] = None):
        """Initialize CSV loader.

        :param embed_fields: Fields to embed (default: all text fields)
        """
        self.embed_fields = embed_fields

    def load(self, source: str, doc_type: str) -> List[Document]:
        """Load CSV rows as documents.

        For speakers.csv: Each row becomes a Document with content dict,
        chunks created for specified embed_fields (bio, abstract).
        """
        # Read CSV
        import csv
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

class GenericRAGService:
    """RAG service with FAISS index and JSON metadata storage.

    For SQLite storage, use SQLiteRAGService subclass.
    """

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_dim: int = 1536,
        embed_client = None
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

    def add_document(self, doc: Document) -> None:
        """Add document and its chunk embeddings to index."""
        for chunk_key, chunk in doc.chunks.items():
            faiss_id = len(self.chunk_refs)
            embedding = self.embed(doc.get_chunk_text(chunk_key))
            self.index.add(embedding)
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
            self.add_document(doc)
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
        from collections import defaultdict
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
                import json
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

class SQLiteRAGService(GenericRAGService):
    """RAG service with SQLite metadata storage (for >1K documents)."""

    def __init__(
        self,
        index_path: Path,
        db_path: Path,
        embedding_dim: int = 1536,
        embed_client = None
    ):
        """Initialize with SQLite storage.

        :param index_path: Path to FAISS index
        :param db_path: Path to SQLite database
        :param embedding_dim: Embedding dimension
        :param embed_client: Embedding client
        """
        self.index_path = index_path
        self.db_path = db_path
        self.embed_client = embed_client
        self.conn = sqlite3.connect(str(db_path))

        # Load or create FAISS index
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)

        # Create tables if new database
        if not db_path.exists():
            self._create_tables()

    def _create_tables(self) -> None:
        """Create SQLite schema."""
        self.conn.execute("""
            CREATE TABLE chunk_refs (
                faiss_id INTEGER PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_key TEXT NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                full_text TEXT,
                metadata TEXT
            )
        """)
        self.conn.commit()

    def _get_chunk_refs(self, faiss_ids: List[int]) -> List[ChunkRef]:
        """Fetch from SQLite."""
        placeholders = ','.join('?' * len(faiss_ids))
        rows = self.conn.execute(
            f"SELECT document_id, chunk_key FROM chunk_refs WHERE faiss_id IN ({placeholders})",
            faiss_ids
        ).fetchall()
        return [ChunkRef(document_id=r[0], chunk_key=r[1]) for r in rows]

    def _get_document_texts(self, doc_ids: List[str]) -> Dict[str, str]:
        """Fetch full document texts from SQLite (optimized)."""
        placeholders = ','.join('?' * len(doc_ids))
        rows = self.conn.execute(
            f"SELECT id, full_text, content FROM documents WHERE id IN ({placeholders})",
            doc_ids
        ).fetchall()
        result = {}
        for row in rows:
            doc_id, full_text, content = row
            # Return full_text if available, otherwise JSON of content
            if full_text:
                result[doc_id] = full_text
            else:
                result[doc_id] = content  # Already JSON string in SQLite
        return result

    def _get_documents(self, doc_ids: List[str]) -> Dict[str, Document]:
        """Fetch full documents from SQLite (for operations needing Document objects)."""
        placeholders = ','.join('?' * len(doc_ids))
        rows = self.conn.execute(
            f"SELECT * FROM documents WHERE id IN ({placeholders})",
            doc_ids
        ).fetchall()
        return {row[0]: Document.from_dict(json.loads(row[1])) for row in rows}

    def _save_metadata(self) -> None:
        """Commit SQLite transaction."""
        self.conn.commit()
```

### Storage Architecture

**Three-layer storage:**

**Layer 1 - FAISS Index** (vectors only)
```
faiss_id → embedding vector (float32[])
```

**Layer 2 - Chunk Metadata** (faiss_id → document location)
```python
# Phase 1: JSON array where array index IS faiss_id
{
    "chunk_refs": [
        {"document_id": "speaker-1", "chunk_key": "bio"},      # index=0 = faiss_id 0
        {"document_id": "speaker-1", "chunk_key": "abstract"}, # index=1 = faiss_id 1
        {"document_id": "speaker-2", "chunk_key": "bio"}       # index=2 = faiss_id 2
    ],
    "documents": [...]
}

# Loaded as: List[ChunkRef] - FAISS assigns sequential IDs (0, 1, 2...)
# Lookup: chunk_refs[faiss_id] - O(1) array access
```

# Phase 2: SQLite (query on demand)
CREATE TABLE chunk_refs (
    faiss_id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_key TEXT NOT NULL
);
SELECT * FROM chunk_refs WHERE faiss_id IN (0, 1, 2);
```

**Layer 3 - Document Storage** (full document data)
```python
# Phase 1: JSON (in same file, loaded on init)
{
    "chunk_refs": [...],
    "documents": [
        {
            "id": "speaker-1",
            "content": {"name": "John", "bio": "...", "abstract": "..."},
            "chunks": {
                "bio": {"faiss_id": 0},
                "abstract": {"faiss_id": 1}
            }
        }
    ]
}

# Phase 2: SQLite
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    content TEXT,
    full_text TEXT,
    metadata TEXT
);
```

**Complete flow:**
```
FAISS search → faiss_ids [0, 1]
    ↓
Chunk refs → [(doc_id="s-1", key="bio"), (doc_id="s-1", key="abstract")]
    ↓
Documents → {s-1: Document(...)}
    ↓
Access → doc.get_chunk_text("bio")
```

**Key principles:**
• Embeddings only in FAISS (single source of truth)
• Text stored once per document (no duplication)
• Chunks reference text via field name or (i_start, i_end)
• Reconstruct on-demand: content[field] or full_text[i_start:i_end]

**Storage efficiency (10-page PDF, 30 chunks):**
| Approach | Storage |
|----------|---------|
| Duplicate text | 31× document size |
| Index pairs (i_start, i_end) | 1.0001× document size |

### Performance

**Retrieval (get_chunk_refs from faiss_ids):**
| Storage | Method | Complexity | Speed (k=10) |
|---------|--------|------------|--------------|
| JSON | Array index | O(k) | 0.1ms |
| SQLite (no index) | Table scan | O(n*k) | 10-50ms |
| SQLite (PRIMARY KEY) | B-tree | O(k log n) | 1ms |

**Critical: chunk_refs needs PRIMARY KEY on faiss_id:**
```sql
CREATE TABLE chunk_refs (
    faiss_id INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_key TEXT NOT NULL
);

CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    content TEXT,
    full_text TEXT,
    metadata TEXT
);
```

**End-to-end search:**
• FAISS vector search: 1-5ms (10K docs)
• Metadata lookup: 0.1-1ms (indexed)
• Total: 1-6ms

### Type Safety

```python
# Use numpy arrays with explicit dtype
embedding: NDArray[np.float32] = np.array([0.1, 0.2, ...], dtype=np.float32)
```
Benefits vs List[float]: 8x less memory, 100x faster, SIMD optimization

## Implementation Phases

**Phase 1**: Core abstractions (Document, DocumentChunk, DocumentLoader)
**Phase 2**: GenericRAGService + YAML config
**Phase 3**: SpeakerRAGService wrapper (backwards compatibility)
**Phase 4**: FAISS + JSON storage
**Phase 5**: Advanced features (PDF/web loaders, chunking strategies)
**Phase 6**: SQLite migration (optional, for >1K docs)
**Phase 7**: Documentation

Total: 4 weeks

## Example: speakers.csv (Field-Level)

**CSV structure:**
```csv
name,title,bio,abstract
John Doe,Professor,Short bio...,Research abstract...
```

**Storage:**
```python
Document(
    id="speaker-1",
    content={"name": "John Doe", "title": "Professor", "bio": "...", "abstract": "..."},
    chunks={
        "bio": DocumentChunk(faiss_id=0),
        "abstract": DocumentChunk(faiss_id=1)
    }
)
```

**Search flow:**
```python
# Query: "quantum computing" → FAISS returns faiss_id=1
metadata = get_chunk_metadata(faiss_id=1)  # Returns: {document_id: "speaker-1", chunk_key: "abstract"}
speaker = get_document("speaker-1")        # Full document with chunks
text = speaker.get_chunk_text("abstract")  # Use chunk_key to access

# Result: entire speaker + matched field
{"name": "John Doe", ..., "matched_field": "abstract"}
```

Benefits: Multiple embeddings per row, returns entire row, no duplication

## Migration Strategy

**Legacy speaker-specific API:**
```python
from chatboti.rag import RAGService
rag = RAGService()  # Wrapper around GenericRAGService
```

**Generic document API:**
```python
from chatboti.generic_rag import GenericRAGService, SQLiteRAGService
from chatboti.embed import OpenAIEmbedClient

# Initialize embedding client
embed_client = OpenAIEmbedClient(model="text-embedding-3-small")

# Default: JSON storage (simple, good for <1K docs)
rag = GenericRAGService(
    index_path=Path("vectors.faiss"),
    metadata_path=Path("metadata.json"),
    embedding_dim=1536,
    embed_client=embed_client
)

# For scale: SQLite storage (good for >1K docs)
rag = SQLiteRAGService(
    index_path=Path("vectors.faiss"),
    db_path=Path("metadata.db"),
    embedding_dim=1536,
    embed_client=embed_client
)

# Build embeddings from source documents
await rag.build_embeddings_from_documents("papers.json", doc_type="research_paper")

# Search
results = await rag.search("quantum computing", k=5)
```

## Benefits Summary

| Aspect | Current | After |
|--------|---------|-------|
| Memory | 48 KB/emb | 6 KB/emb (8x) |
| Search | 50-100ms | 1-5ms (10-100x) |
| Storage | 40 KB/emb | 6 KB/emb (6x) |
| Flexibility | Speakers only | Any document |
| Dependencies | NumPy: 20 MB | +FAISS: +15 MB |
| Scalability | <1K docs | 10K-100K docs |
| Models | Single | Any (384-3072 dims) |

## Key Decisions

1. **FAISS** for vectors (purpose-built, fast, memory-efficient)
2. **SQLite** for metadata (ACID, flexible queries, single file)
3. **Separate storage** (vectors vs metadata - optimized for each)
4. **ndarray[float32]** for embeddings (8x memory, 100x performance vs List[float])
5. **Zero breaking changes** (SpeakerRAGService wrapper)
6. **Multi-model support** (flexible, future-proof, cost optimization)

## Non-Goals

• Cloud vector databases (Pinecone, Weaviate)
• Real-time updates (batch ingestion sufficient)
• Distributed search (single-node sufficient)
• Complex query DSL (simple text search only)
• Multi-tenancy (single user system)

## Success Criteria

• Existing speaker search works unchanged
• Load documents from CSV, JSON, PDF, web
• 10-100x faster search with FAISS
• 8x memory reduction (no List[float])
• Docker image +15 MB only
• Support any embedding model (384-3072 dims)
• Comprehensive tests and documentation

## Reference

**`docs/generic-document-storage-spec.md`** (4,020 lines)
• Complete architecture and implementation details
• Multi-model support (Section 13)
• Migration strategies and code examples
