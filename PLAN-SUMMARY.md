# Generic Document Storage - Plan Summary

## 🎯 Goal
Transform chatboti RAG from **speaker-specific** → **generic document storage** system

## 📊 Current Problems
1. **Hardcoded speaker fields**: `bio_max_120_words`, `final_abstract_max_150_words`
2. **Single CSV source**: Can't load JSON, PDF, web pages, multiple files
3. **Fixed structure**: Exactly 2 embeddings per document, no flexibility
4. **Inefficient storage**: JSON with `List[float]` (8x memory waste, 6-8x storage waste)
5. **Slow search**: Linear scan, no indexing (50-100ms for 10K docs)

## ✅ Solution Architecture

### Storage Layer
```
documents/
├── vectors.faiss    # FAISS index (all vectors as float32)
└── metadata.db      # SQLite (all metadata, no vectors)
```

**Why FAISS?**
- ✅ Only +15 MB dependency (+10% Docker image)
- ✅ 10-100x faster search (1-5ms for 10K docs)
- ✅ 0.3% memory overhead
- ✅ Memory-mapped (instant load)
- ✅ Always float32 (type safe)

**Why SQLite?**
- ✅ ACID transactions
- ✅ Flexible metadata queries
- ✅ No separate server needed
- ✅ Standard relational model

### Core Abstractions

```python
# 1. Generic Document (domain-agnostic)
class Document:
    id: str
    content: dict          # Flexible fields
    embeddings: dict       # Named embeddings
    metadata: dict         # Source, timestamp, etc.

# 2. Document Loaders (pluggable)
class DocumentLoader:
    def load(self, source: str) -> List[Document]:
        ...

# Implementations: CSV, JSON, PDF, Web, ...

# 3. Generic RAG Service
class GenericRAGService:
    def __init__(self, index_path: Path, db_path: Path):
        self.index = faiss.read_index(index_path)    # Fast
        self.db = sqlite3.connect(db_path)

    def add_document(self, doc: Document) -> int:
        # Add to FAISS, save metadata to SQLite

    def search(self, query: str, k: int) -> List[Document]:
        # FAISS search → fetch metadata from SQLite
```

### Type Safety

```python
# NEVER use List[float]
embedding: List[float] = [0.1, 0.2, ...]  ❌

# ALWAYS use ndarray[float32]
embedding: NDArray[np.float32] = np.array([0.1, 0.2, ...], dtype=np.float32)  ✅
```

**Benefits**: 8x less memory, 100x faster operations, SIMD optimization

## 📅 Implementation Plan

### Phase 1: Core Abstractions (Week 1)
- `Document`, `DocumentChunk`, `EmbeddingConfig` classes
- `DocumentLoader` ABC + CSV/JSON loaders
- Unit tests

### Phase 2: Generic Service (Week 1-2)
- `GenericRAGService` implementation
- Configuration file support (YAML)
- Integration tests

### Phase 3: Backwards Compatibility (Week 2)
- `SpeakerRAGService` wrapper (zero breaking changes)
- Data migration utilities
- Adapter for legacy code

### Phase 4: FAISS Integration (Week 3)
- Migrate from JSON → FAISS vector storage
- FAISS index creation/loading
- Migration utility for existing data
- Performance benchmarks

### Phase 5: Advanced Features (Week 3-4)
- Text chunking strategies
- PDF and web loaders
- Index optimization (IVF, PQ)
- Quantization for compression

### Phase 6: Documentation (Week 4)
- Migration guide
- Configuration examples
- API documentation
- Tutorials

**Total: 4 weeks**

## 🔄 Migration Strategy

### For Existing Code (Zero Breakage)
```python
# Old code continues to work
from chatboti.rag import RAGService
rag = RAGService()  # Uses SpeakerRAGService internally

# Or explicit
from chatboti.rag import SpeakerRAGService
rag = SpeakerRAGService()  # Same API, better name
```

### For New Use Cases
```python
# Generic approach
from chatboti.generic_rag import GenericRAGService

rag = GenericRAGService(
    index_path="vectors.faiss",
    db_path="metadata.db"
)

# Load from any source
await rag.load_documents("papers.json", doc_type="research_paper")
await rag.load_documents("products.csv", doc_type="product")

# Search works the same
results = await rag.search("quantum computing", k=5)
```

## 🎁 Benefits Summary

| Aspect | Current | After Refactor |
|--------|---------|----------------|
| **Memory** | List[float]: 48 KB/emb | ndarray: 6 KB/emb (8x better) |
| **Search** | Linear: 50-100ms | FAISS: 1-5ms (10-100x faster) |
| **Storage** | JSON: 40 KB/emb | FAISS: 6 KB/emb (6x smaller) |
| **Flexibility** | Speakers only | Any document type |
| **Dependencies** | NumPy: 20 MB | +FAISS: +15 MB (+10%) |
| **Type Safety** | List confusion | Always float32 |
| **Scalability** | <1K docs | 10K-100K docs |

## 📦 Example Use Cases

### 1. Speaker Data (Current - Backwards Compatible)
```python
rag = SpeakerRAGService()
speaker = await rag.get_best_speaker("quantum computing")
```

### 2. Research Papers
```python
rag = GenericRAGService()
await rag.load_documents("papers.pdf", doc_type="research_paper")
results = await rag.search("transformer architecture", k=10)
```

### 3. Product Catalog
```python
rag = GenericRAGService()
await rag.load_documents("products.json", doc_type="product")
results = await rag.search("wireless headphones", k=5)
```

### 4. Knowledge Base
```python
rag = GenericRAGService()
await rag.load_documents("docs/*.md", doc_type="documentation")
results = await rag.search("how to install", k=3)
```

### 5. Multi-Domain Search
```python
rag = GenericRAGService()
await rag.load_documents("speakers.csv", doc_type="speaker")
await rag.load_documents("papers.pdf", doc_type="paper")
await rag.load_documents("products.json", doc_type="product")

# Search across all domains
results = await rag.search("AI applications", k=10)
```

## 🚀 Key Decisions

1. **FAISS as primary vector storage** (not Parquet/SQLite BLOB)
   - Reason: Purpose-built, fast, small, memory-efficient

2. **SQLite for metadata** (not Parquet/separate files)
   - Reason: ACID, flexible queries, single file

3. **Separate vectors and metadata** (not unified Parquet)
   - Reason: Separation of concerns, optimized for each use case

4. **Never use List[float]** (always ndarray[float32])
   - Reason: 8x memory, 100x performance, type safety

5. **Zero breaking changes** (SpeakerRAGService wrapper)
   - Reason: Smooth migration, existing code works

## ⚠️ Non-Goals

- ❌ Cloud vector databases (Pinecone, Weaviate) - keep it simple
- ❌ Real-time updates - batch ingestion is fine
- ❌ Distributed search - single-node is sufficient
- ❌ Complex query DSL - simple text search only
- ❌ Multi-tenancy - single user system

## 📋 Success Criteria

- ✅ Existing speaker search continues to work without changes
- ✅ Can load documents from CSV, JSON, PDF, web
- ✅ Search is 10-100x faster with FAISS
- ✅ Memory usage reduced by 8x (no List[float])
- ✅ Docker image only +15 MB larger
- ✅ All embeddings stored as float32 in FAISS
- ✅ Metadata in SQLite with flexible schema
- ✅ Comprehensive tests and documentation

## 🔗 Related Specs

- `docs/faiss-multiple-documents-spec.md` - FAISS migration details
- `docs/metadata-storage-design.md` - SQLite schema design
- `docs/vector-storage-comparison.md` - Parquet vs FAISS analysis
- `docs/analysis/embedding-storage-architecture.md` - Storage format comparison
- `docs/analysis/faiss-vs-numpy-size-comparison.md` - Size analysis
