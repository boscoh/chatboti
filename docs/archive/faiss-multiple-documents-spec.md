# FAISS-CPU and Multiple Documents Specification

## Executive Summary

This specification outlines the migration from JSON-based embedding storage to faiss-cpu vector indexing, while adding support for multiple document sources beyond the current single speaker CSV file. The proposed architecture maintains deployment simplicity while significantly improving scalability and query performance.

**Key Changes**:
1. Replace JSON vector storage with faiss-cpu IndexFlatIP (Inner Product) index
2. Support multiple document types (CSV, JSON, text files, PDFs, web pages)
3. Implement document chunking and ingestion pipeline
4. Integrate with SQLite metadata storage (see `metadata-storage-design.md`)
5. Maintain backward compatibility during migration

**Expected Benefits**:
- 10-100x faster similarity search for large datasets (>1000 documents)
- Support for arbitrary document sources
- Reduced memory footprint with memory-mapped indices
- Clear path to GPU acceleration if needed
- Industry-standard vector search implementation

---

## 1. Current System Analysis

### 1.1 Current Architecture

```
CSV Source → Embedding Generation → JSON Storage → In-Memory Search
    ↓              ↓                      ↓                ↓
speakers.csv   OpenAI/Bedrock    embeddings-{model}.json  Numpy cosine
```

**File**: `chatboti/rag.py` (225 lines)

### 1.2 Current Data Flow

1. **Embedding Generation** (`generate_and_save_embeddings`):
   - Reads `2025-09-02-speaker-bio.csv` (hardcoded)
   - Generates 2 embeddings per speaker (abstract, bio)
   - Saves to `embeddings-{model}.json` with inline vectors
   - File size: 750KB - 1.1MB for 63 speakers

2. **Loading** (`connect`):
   - Loads entire JSON file into memory
   - Parses into `speakers_with_embeddings` list
   - Strips embeddings to create `speakers` list

3. **Querying** (`get_best_speaker`):
   - Embeds query text
   - Linear scan: compute cosine distance for each speaker
   - Returns best match (lowest distance)

### 1.3 Current Limitations

#### Performance
- **O(n) search complexity**: Linear scan through all embeddings
- **Full dataset in memory**: ~1MB+ loaded for every query
- **No indexing**: Cold start penalty on each service initialization
- **Single-threaded search**: No batch query optimization

#### Scalability
- **Single document source**: Hardcoded to speakers CSV
- **No chunking**: Each speaker = 2 fixed embeddings (abstract + bio)
- **No document tracking**: No source metadata, timestamps, versions
- **Limited to 1000s of documents**: Performance degrades linearly

#### Maintainability
- **Fragile file path handling**: Hardcoded CSV filename
- **No deduplication**: Same document can be embedded multiple times
- **No versioning**: Can't track which model generated embeddings
- **Mixed concerns**: Storage format coupled with search logic

### 1.4 Performance Baseline

Current system (63 speakers = 126 embeddings):
- **Load time**: 50-100ms (JSON parse)
- **Query time**: <1ms (numpy cosine distance)
- **Memory usage**: ~2MB (data + embeddings)
- **Storage**: 750KB - 1.1MB (JSON)

**Note**: Performance acceptable at current scale but won't scale to 10K+ documents.

---

## 2. Proposed Architecture

### 2.1 Overview

```
Multiple Sources → Document Ingestion → FAISS Index + SQLite → Hybrid Search
       ↓                  ↓                      ↓                    ↓
CSV/PDF/Web    Chunking + Embedding    embeddings.index      Vector + Metadata
                                       metadata.db           retrieval
```

### 2.2 Core Components

#### Component 1: Document Ingestion Pipeline
**Purpose**: Load, parse, and chunk documents from multiple sources

```python
class DocumentIngestionPipeline:
    """Load and process documents from various sources."""

    async def ingest_csv(self, file_path: str, text_columns: List[str]) -> List[Document]
    async def ingest_pdf(self, file_path: str, chunk_size: int = 512) -> List[Document]
    async def ingest_text(self, file_path: str, chunk_size: int = 512) -> List[Document]
    async def ingest_json(self, file_path: str, text_field: str) -> List[Document]
    async def ingest_web(self, url: str, chunk_size: int = 512) -> List[Document]
```

**Features**:
- Pluggable loaders for different file types
- Configurable chunking strategies
- Document metadata extraction
- Content hashing for deduplication

#### Component 2: FAISS Vector Index
**Purpose**: Fast similarity search over embeddings

```python
class FAISSVectorStore:
    """Manage FAISS index for vector similarity search."""

    def __init__(self, dimension: int, index_type: str = "Flat")
    async def add_vectors(self, vectors: np.ndarray, ids: List[str])
    async def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]
    def save(self, path: str)
    def load(self, path: str)
    def reset(self)
```

**Index Types**:
- **IndexFlatIP** (default): Inner product similarity, exact search
- **IndexFlatL2**: L2 distance, exact search
- **IndexIVFFlat**: Approximate search for 10K+ vectors (future)

#### Component 3: SQLite Metadata Store
**Purpose**: Store document metadata, chunks, and embedding metadata

See `metadata-storage-design.md` for full schema. Key tables:
- `documents`: Source files, import timestamps, content hashes
- `chunks`: Text chunks with positions and metadata
- `embeddings`: Embedding metadata (not vectors, just references)

#### Component 4: Hybrid Retrieval Service
**Purpose**: Coordinate vector search and metadata lookup

```python
class RAGService:
    """Enhanced RAG service with FAISS and multiple document support."""

    def __init__(self, llm_service: str = None)
    async def ingest_document(self, source: str, doc_type: str, **kwargs)
    async def search(self, query: str, k: int = 5, filters: dict = None) -> List[SearchResult]
    async def get_context(self, chunk_id: str, context_size: int = 1) -> str
```

### 2.3 Data Flow

#### Ingestion Flow
```
1. Load document from source
2. Parse and extract text
3. Chunk text (if needed)
4. Generate embeddings for chunks
5. Add vectors to FAISS index
6. Store metadata in SQLite
7. Save index to disk
```

#### Query Flow
```
1. Embed query text
2. Search FAISS index for top-K chunk IDs
3. Lookup metadata in SQLite
4. Optionally expand context (surrounding chunks)
5. Return ranked results with metadata
```

### 2.4 File Structure

```
chatboti/data/
├── embeddings-{model}.index          # FAISS vector index
├── metadata.db                        # SQLite database
├── documents/                         # Optional: raw documents
│   ├── speakers.csv
│   ├── policy-docs.pdf
│   └── faq.json
└── legacy/
    └── embeddings-{model}.json       # Old format (for migration)
```

---

## 3. FAISS Integration

### 3.1 Why FAISS?

**Advantages over JSON**:
- **Performance**: 10-100x faster for large datasets
- **Memory efficiency**: Memory-mapped indices (don't load full index)
- **Scalability**: Proven at billion-vector scale
- **Flexibility**: Multiple index types for different trade-offs
- **Industry standard**: Used by Meta, OpenAI, Anthropic

**From vector-storage-comparison.md**: While Parquet was recommended for the current small dataset, FAISS becomes beneficial at 1000+ documents, especially with multiple document sources.

### 3.2 Index Selection

#### For Current Scale (<1000 documents, ~2000 embeddings)

**Recommended: IndexFlatIP (Inner Product)**
```python
import faiss
import numpy as np

dimension = 1536  # OpenAI text-embedding-3-small
index = faiss.IndexFlatIP(dimension)

# Normalize vectors for cosine similarity via inner product
faiss.normalize_L2(vectors)
index.add(vectors)
```

**Why IndexFlatIP?**
- Exact search (no approximation errors)
- Cosine similarity via normalized inner product
- <1ms query time for 1000s of vectors
- Simple: no hyperparameters to tune
- Direct replacement for current numpy cosine distance

**Alternative: IndexFlatL2 (L2 Distance)**
- Use if L2 distance preferred over cosine
- Slightly faster than IP
- Same exact search guarantees

#### For Future Scale (10K-100K documents)

**IndexIVFFlat (Inverted File Index)**
```python
# Train on sample vectors
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.train(training_vectors)
index.add(vectors)

# Search with probe parameter
index.nprobe = 10  # Trade-off: speed vs accuracy
```

**When to migrate**:
- Dataset > 10K documents (>20K embeddings)
- Query latency > 10ms with Flat index
- Memory constraints (large dimensional vectors)

**Trade-offs**:
- 10-100x faster search
- 95-99% accuracy (rarely misses true top-K)
- Requires training phase
- More complex to configure

### 3.3 Distance Metrics

**Current System**: Cosine distance
```python
cosine_similarity = dot_product / (norm_a * norm_b)
cosine_distance = 1.0 - cosine_similarity
```

**FAISS Equivalent**: Inner Product with normalized vectors
```python
import faiss
import numpy as np

# Normalize vectors (in-place)
faiss.normalize_L2(vectors)

# Inner product on normalized vectors = cosine similarity
index = faiss.IndexFlatIP(dimension)
index.add(vectors)

# Search returns (distances, indices)
# distances = cosine similarities (higher = more similar)
distances, indices = index.search(query_vector, k=5)

# Convert to cosine distance if needed
cosine_distances = 1.0 - distances
```

**Why normalize?**
- Converts inner product to cosine similarity
- Makes scores interpretable (0-1 range)
- Consistent with current implementation

### 3.4 Vector Storage

**Option A: Store in FAISS only** (Recommended)
```python
# Vectors only in FAISS index
index.add(vectors)
faiss.write_index(index, "embeddings.index")

# Metadata in SQLite (no vector column)
# Chunk ID serves as link between systems
```

**Pros**:
- Single source of truth for vectors
- Optimized binary format
- Memory-mapped loading (fast cold start)

**Cons**:
- Need to rebuild index if FAISS file corrupted
- Can't easily inspect individual vectors

**Option B: Store in both FAISS and SQLite**
```python
# FAISS for search
index.add(vectors)

# SQLite for backup/inspection
INSERT INTO embeddings (chunk_id, vector) VALUES (?, ?)
```

**Pros**:
- Redundancy (can rebuild index from DB)
- Easy to inspect vectors
- Supports vector DB migration later

**Cons**:
- Duplicated storage (~2x size)
- Sync complexity

**Recommendation**: Start with Option A (FAISS only), add SQLite backup if reliability concerns arise.

### 3.5 Index Persistence

```python
import faiss

# Save index
faiss.write_index(index, "embeddings.index")

# Load index (fast, memory-mapped)
index = faiss.read_index("embeddings.index")

# File size: dimension * num_vectors * 4 bytes (float32)
# Example: 1536 dims * 2000 vectors * 4 = ~12MB
```

**Performance**:
- Save: <100ms for 10K vectors
- Load: <50ms (memory-mapped, doesn't load full index)
- Query: <1ms for flat index

### 3.6 ID Mapping

**Challenge**: FAISS uses integer indices (0, 1, 2, ...), but we need string chunk IDs (UUIDs).

**Solution**: IndexIDMap wrapper
```python
import faiss

# Base index
base_index = faiss.IndexFlatIP(dimension)

# Wrap with ID mapper
index = faiss.IndexIDMap(base_index)

# Add with explicit IDs
chunk_ids = [hash(uuid) for uuid in chunk_uuids]  # Convert to int64
index.add_with_ids(vectors, np.array(chunk_ids, dtype=np.int64))

# Search returns original IDs
distances, ids = index.search(query, k=5)
chunk_uuids = [reverse_lookup[id] for id in ids[0]]
```

**Alternative**: Maintain separate mapping
```python
# FAISS uses sequential indices
index.add(vectors)

# Separate mapping: FAISS index -> chunk UUID
id_mapping = {0: "chunk-uuid-1", 1: "chunk-uuid-2", ...}

# After search
indices = search_results[1][0]  # FAISS indices
chunk_ids = [id_mapping[i] for i in indices]
```

**Recommendation**: Use separate mapping for simplicity. Store in SQLite or in-memory dict.

---

## 4. Multiple Document Support

### 4.1 Document Types

#### Type 1: CSV Files (Current)
**Use Case**: Tabular data with text columns

```python
async def ingest_csv(
    self,
    file_path: str,
    text_columns: List[str],
    id_column: str = None,
    metadata_columns: List[str] = None
) -> List[str]:
    """
    Ingest CSV file with specified text columns for embedding.

    Args:
        file_path: Path to CSV file
        text_columns: Columns to embed separately (e.g., ["abstract", "bio"])
        id_column: Column to use as document ID (or generate UUID)
        metadata_columns: Additional columns to store as metadata

    Returns:
        List of chunk IDs created
    """
```

**Example**: Current speaker CSV
```csv
name,bio_max_120_words,final_abstract_max_150_words,country
Jonathan Passé,"Bio text...","Abstract text...",Germany & UK
```

**Processing**:
- Each row = 1 document
- Each text column = 1 chunk
- 63 rows * 2 text columns = 126 chunks

#### Type 2: PDF Files
**Use Case**: Research papers, documentation, reports

```python
async def ingest_pdf(
    self,
    file_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    extract_metadata: bool = True
) -> List[str]:
    """
    Ingest PDF file with automatic text extraction and chunking.

    Args:
        file_path: Path to PDF file
        chunk_size: Max tokens per chunk
        chunk_overlap: Token overlap between chunks
        extract_metadata: Extract PDF metadata (title, author, date)
    """
```

**Dependencies**: `pypdf2` or `pdfplumber`

**Processing**:
- Extract text from each page
- Chunk based on token count or paragraph boundaries
- Preserve page numbers in metadata
- Example: 10-page PDF → 30 chunks (3 per page avg)

#### Type 3: Text Files (Markdown, TXT)
**Use Case**: Documentation, notes, articles

```python
async def ingest_text(
    self,
    file_path: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    preserve_structure: bool = True  # Respect paragraphs/sections
) -> List[str]:
    """Ingest plain text or Markdown files."""
```

**Processing**:
- Respect paragraph/section boundaries
- Split long paragraphs if > chunk_size
- Markdown: preserve headers as metadata

#### Type 4: JSON Documents
**Use Case**: API responses, structured data

```python
async def ingest_json(
    self,
    file_path: str,
    text_field: str,
    id_field: str = None,
    metadata_fields: List[str] = None,
    array_path: str = None  # JSONPath for array of objects
) -> List[str]:
    """
    Ingest JSON file or array of JSON objects.

    Args:
        text_field: Field containing text to embed
        id_field: Field to use as document ID
        metadata_fields: Fields to store as metadata
        array_path: JSONPath if data is nested (e.g., "$.data[*]")
    """
```

**Example**:
```json
{
  "data": [
    {
      "id": "faq-1",
      "question": "What is RAG?",
      "answer": "Retrieval-Augmented Generation...",
      "category": "AI"
    }
  ]
}
```

**Processing**:
- Extract text from specified field
- Store other fields as metadata
- Each object = 1 document (unless chunking needed)

#### Type 5: Web Pages
**Use Case**: Documentation sites, articles, FAQs

```python
async def ingest_web(
    self,
    url: str,
    chunk_size: int = 512,
    extract_links: bool = False,
    max_depth: int = 0  # Crawl depth
) -> List[str]:
    """
    Ingest web page content.

    Args:
        url: URL to fetch
        chunk_size: Max tokens per chunk
        extract_links: Follow links for crawling
        max_depth: Max crawl depth (0 = single page)
    """
```

**Dependencies**: `beautifulsoup4`, `requests` or `httpx`

**Processing**:
- Fetch HTML
- Extract main content (strip nav, footer, ads)
- Convert to Markdown
- Chunk and embed

### 4.2 Document Ingestion API

```python
class RAGService:
    async def ingest_document(
        self,
        source: str,
        doc_type: str = "auto",  # auto-detect or explicit
        **kwargs
    ) -> IngestResult:
        """
        Ingest a document from any supported source.

        Args:
            source: File path or URL
            doc_type: "csv", "pdf", "text", "json", "web", or "auto"
            **kwargs: Type-specific parameters

        Returns:
            IngestResult with document_id, chunk_ids, stats
        """
        # Auto-detect type if needed
        if doc_type == "auto":
            doc_type = self._detect_type(source)

        # Dispatch to type-specific ingester
        ingester = self.ingesters[doc_type]
        documents = await ingester.ingest(source, **kwargs)

        # Generate embeddings
        chunks = await self._chunk_documents(documents, **kwargs)
        embeddings = await self._embed_chunks(chunks)

        # Add to FAISS index
        await self.vector_store.add_vectors(embeddings, chunk_ids)

        # Store metadata in SQLite
        await self.metadata_store.save_documents(documents)
        await self.metadata_store.save_chunks(chunks)

        # Save index to disk
        self.vector_store.save()

        return IngestResult(
            document_ids=[d.id for d in documents],
            chunk_ids=chunk_ids,
            num_chunks=len(chunks),
            elapsed_time=elapsed
        )
```

**Usage**:
```python
# CSV
await rag.ingest_document(
    "speakers.csv",
    doc_type="csv",
    text_columns=["bio_max_120_words", "final_abstract_max_150_words"],
    id_column="name"
)

# PDF
await rag.ingest_document(
    "research-paper.pdf",
    chunk_size=512,
    chunk_overlap=50
)

# Web
await rag.ingest_document(
    "https://docs.anthropic.com/claude",
    doc_type="web",
    max_depth=1
)
```

### 4.3 Chunking Strategies

#### Strategy 1: Fixed-Size Chunking
**Best for**: General text, PDFs, articles

```python
def chunk_fixed_size(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    tokenizer: Optional[Callable] = None
) -> List[Chunk]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text: Input text
        chunk_size: Max tokens per chunk
        chunk_overlap: Tokens to overlap between chunks
        tokenizer: Optional custom tokenizer (default: tiktoken)
    """
```

**Example**: 1000-token document → 2 chunks (tokens 0-512, 462-974)

**Pros**:
- Predictable chunk sizes (fits in LLM context)
- Overlap preserves context across boundaries
- Simple to implement

**Cons**:
- May split mid-sentence or mid-paragraph
- No semantic awareness

#### Strategy 2: Semantic Chunking
**Best for**: Markdown, documentation, structured text

```python
def chunk_semantic(
    text: str,
    max_chunk_size: int = 512,
    split_points: List[str] = ["\n\n", "\n", ". "],
    preserve_structure: bool = True
) -> List[Chunk]:
    """
    Split text at semantic boundaries (paragraphs, sentences).

    Args:
        text: Input text
        max_chunk_size: Maximum tokens per chunk
        split_points: Boundary markers (in priority order)
        preserve_structure: Try to keep paragraphs together
    """
```

**Example**: Split on `\n\n` (paragraphs), respect max_chunk_size

**Pros**:
- Natural boundaries (complete thoughts)
- Better for QA and summarization
- Preserves document structure

**Cons**:
- Variable chunk sizes
- May require padding or splitting large paragraphs

#### Strategy 3: Field-Based Chunking (Current)
**Best for**: Structured data (CSV, JSON with predefined fields)

```python
def chunk_by_field(
    document: dict,
    text_fields: List[str]
) -> List[Chunk]:
    """
    Create one chunk per text field.

    Args:
        document: Document dict
        text_fields: Fields to extract as separate chunks
    """
```

**Example**: CSV row → 1 chunk per column (abstract, bio)

**Pros**:
- Preserves field semantics
- No need to recombine chunks
- Clean metadata (field name → chunk type)

**Cons**:
- Only works for structured data
- Large fields may exceed context limits

#### Strategy 4: Recursive Chunking
**Best for**: Complex documents with nested structure

```python
def chunk_recursive(
    text: str,
    max_chunk_size: int = 512,
    separators: List[str] = ["\n\n", "\n", ". ", " "]
) -> List[Chunk]:
    """
    Recursively split text using separator hierarchy.

    Try separators in order:
    1. Double newline (paragraphs)
    2. Single newline (lines)
    3. Sentence boundaries
    4. Word boundaries
    """
```

**Pros**:
- Adapts to document structure
- Maximizes chunk coherence
- Used by LangChain (proven approach)

**Cons**:
- More complex implementation
- Unpredictable chunk sizes

**Recommendation**: Start with fixed-size (Strategy 1) for PDFs/text, field-based (Strategy 3) for CSV/JSON.

### 4.4 Metadata Extraction

Each chunk should include metadata for filtering and context:

```python
@dataclass
class Chunk:
    id: str                           # UUID
    document_id: str                  # Parent document
    text: str                         # Chunk content
    chunk_index: int                  # Position in document
    start_char: int                   # Character offset
    end_char: int                     # End offset
    token_count: int                  # Approx tokens
    chunk_type: Optional[str] = None  # "abstract", "bio", "paragraph"
    metadata: dict = field(default_factory=dict)
```

**Metadata Examples**:
- **CSV**: `{"row_index": 5, "field": "bio", "speaker_name": "John Doe"}`
- **PDF**: `{"page": 3, "section": "Results", "file_name": "paper.pdf"}`
- **Web**: `{"url": "...", "crawl_depth": 1, "last_modified": "2026-02-10"}`

### 4.5 Document Deduplication

**Challenge**: Same document ingested multiple times

**Solution**: Content-based hashing
```python
import hashlib

def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of document content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Before ingestion
content_hash = compute_content_hash(document.text)

# Check if exists
existing = await metadata_store.get_document_by_hash(content_hash)
if existing:
    logger.info(f"Document already exists: {existing.id}")
    return existing.id

# Proceed with ingestion
```

**Storage**: Store hash in `documents.content_hash` column (SQLite)

---

## 5. Integration with Metadata Storage

This section describes how FAISS vector indices integrate with SQLite metadata storage (see `metadata-storage-design.md` for full schema).

### 5.1 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      RAGService                              │
│  (Orchestrates vector search + metadata retrieval)           │
└───────────────┬────────────────────────┬─────────────────────┘
                │                        │
                ▼                        ▼
    ┌───────────────────────┐  ┌─────────────────────────┐
    │  FAISSVectorStore     │  │  SQLiteMetadataStore    │
    │  (embeddings.index)   │  │  (metadata.db)          │
    └───────────────────────┘  └─────────────────────────┘
             │                           │
             ▼                           ▼
    Vector search returns         Metadata lookup by IDs
    chunk IDs + distances         returns full chunk data
```

### 5.2 Data Synchronization

**Critical Invariant**: FAISS index position ↔ chunk ID mapping must be consistent

**Approach**: Maintain mapping in SQLite
```sql
CREATE TABLE faiss_index_mapping (
    faiss_index INTEGER PRIMARY KEY,  -- FAISS internal index (0, 1, 2, ...)
    chunk_id TEXT NOT NULL,            -- UUID from chunks table
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE INDEX idx_faiss_mapping_chunk ON faiss_index_mapping(chunk_id);
```

**Add vectors workflow**:
```python
# 1. Store chunks in SQLite
chunk_ids = await metadata_store.insert_chunks(chunks)

# 2. Add vectors to FAISS
vectors = np.array([chunk.embedding for chunk in chunks])
faiss_indices = index.ntotal + np.arange(len(vectors))  # Sequential
index.add(vectors)

# 3. Store mapping
for faiss_idx, chunk_id in zip(faiss_indices, chunk_ids):
    await metadata_store.insert_mapping(faiss_idx, chunk_id)
```

**Search workflow**:
```python
# 1. Search FAISS
distances, faiss_indices = index.search(query_vector, k=5)

# 2. Lookup chunk IDs
chunk_ids = await metadata_store.get_chunk_ids_by_faiss_indices(faiss_indices[0])

# 3. Fetch full metadata
chunks = await metadata_store.get_chunks_by_ids(chunk_ids)

# 4. Return results
return [SearchResult(chunk=c, score=d) for c, d in zip(chunks, distances[0])]
```

### 5.3 Schema Integration

From `metadata-storage-design.md`, we use these tables:

#### documents
```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    source_type TEXT NOT NULL,
    content_hash TEXT,
    imported_at TIMESTAMP NOT NULL,
    metadata JSON
);
```

**FAISS relationship**: None (documents are parents of chunks)

#### chunks
```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_type TEXT,
    text TEXT NOT NULL,
    token_count INTEGER,
    metadata JSON,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
```

**FAISS relationship**: Each chunk has 0-1 embeddings in FAISS (some chunks may not be embedded)

#### embeddings (metadata only, no vectors)
```sql
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    vector_dimension INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL,
    metadata JSON,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);
```

**FAISS relationship**: `embeddings.id` maps to FAISS index via `faiss_index_mapping`

**Note**: We do NOT store the vector BLOB in SQLite (see Section 3.4). Vectors live only in FAISS.

### 5.4 Multi-Model Support

**Use Case**: Support multiple embedding models (e.g., OpenAI, Bedrock, Cohere)

**Approach**: Separate FAISS index per model
```
chatboti/data/
├── embeddings-text-embedding-3-small.index
├── embeddings-amazon-titan-embed-text-v1.index
├── metadata.db  # Single metadata DB, multiple embedding records
```

**Schema**:
```sql
-- embeddings table tracks which model generated embedding
SELECT chunk_id, model_name FROM embeddings;
-- chunk-1, text-embedding-3-small
-- chunk-1, amazon.titan-embed-text-v1  # Same chunk, 2 models
```

**Code**:
```python
class RAGService:
    def __init__(self, llm_service: str):
        self.embed_client = get_llm_client(llm_service, model=...)

        # Load model-specific FAISS index
        index_path = f"embeddings-{model_slug}.index"
        self.vector_store = FAISSVectorStore.load(index_path)
```

**Query**: Uses active model's index only
```python
# Embed with current model
query_vec = await self.embed_client.embed(query)

# Search in corresponding index
results = await self.vector_store.search(query_vec, k=5)
```

### 5.5 Metadata Filtering

**Use Case**: Filter search results by document type, date, or custom metadata

**Approach**: Post-filter FAISS results using SQLite
```python
async def search(
    self,
    query: str,
    k: int = 5,
    filters: Optional[dict] = None
) -> List[SearchResult]:
    """
    Search with optional metadata filters.

    Args:
        query: Search query
        k: Number of results (before filtering)
        filters: {"source_type": "pdf", "imported_after": "2026-01-01"}
    """
    # 1. Vector search (over-fetch to account for filtering)
    query_vec = await self.embed_client.embed(query)
    distances, faiss_indices = self.index.search(query_vec, k * 3)

    # 2. Get chunk IDs
    chunk_ids = await self.metadata_store.get_chunk_ids(faiss_indices[0])

    # 3. Apply filters in SQLite
    if filters:
        chunks = await self.metadata_store.get_chunks_filtered(chunk_ids, filters)
    else:
        chunks = await self.metadata_store.get_chunks(chunk_ids)

    # 4. Return top-K after filtering
    return chunks[:k]
```

**SQLite filter example**:
```sql
SELECT c.*, d.source_type, d.imported_at
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE c.id IN (?, ?, ?, ...)  -- From FAISS
  AND d.source_type = 'pdf'
  AND d.imported_at > '2026-01-01'
ORDER BY FIELD(c.id, ?, ?, ?, ...);  -- Preserve FAISS ranking
```

**Alternative**: Pre-filter with FAISS IndexShards (advanced)
- Separate index per document type
- Only search relevant shard
- More complex, only needed for 100K+ documents

---

## 6. API Changes

### 6.1 Current API

```python
class RAGService:
    async def generate_and_save_embeddings(self)
    async def get_best_speaker(self, query: str) -> Optional[dict]
    async def get_speakers(self) -> List[dict]
```

**Issues**:
- Hardcoded to speakers
- No multi-document support
- No metadata filtering
- Single result only

### 6.2 Proposed API (Backward Compatible)

```python
class RAGService:
    # NEW: Document ingestion
    async def ingest_document(
        self,
        source: str,
        doc_type: str = "auto",
        **kwargs
    ) -> IngestResult

    # NEW: General search
    async def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[dict] = None,
        include_context: bool = False
    ) -> List[SearchResult]

    # NEW: Context expansion
    async def get_context(
        self,
        chunk_id: str,
        context_size: int = 1
    ) -> str

    # DEPRECATED: Use ingest_document instead
    async def generate_and_save_embeddings(self)

    # DEPRECATED: Use search with filters instead
    async def get_best_speaker(self, query: str) -> Optional[dict]

    # DEPRECATED: Query metadata DB directly
    async def get_speakers(self) -> List[dict]
```

### 6.3 New Data Structures

```python
@dataclass
class SearchResult:
    """Result from vector search."""
    chunk_id: str
    chunk_text: str
    score: float  # Cosine similarity (0-1, higher = better)
    document_id: str
    document_source: str
    chunk_type: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    context: Optional[str] = None  # Surrounding chunks if requested

@dataclass
class IngestResult:
    """Result from document ingestion."""
    document_ids: List[str]
    chunk_ids: List[str]
    num_chunks: int
    num_embeddings: int
    elapsed_time: float
    errors: List[str] = field(default_factory=list)
```

### 6.4 Migration Strategy

**Phase 1: Dual Write** (Keep JSON for safety)
```python
async def generate_and_save_embeddings(self):
    # Generate embeddings
    speakers = await self._generate_speaker_embeddings()

    # Write to both formats
    # 1. Legacy JSON
    self.save_text_file(json.dumps(speakers, indent=2), self.embed_json)

    # 2. New FAISS + SQLite
    await self.ingest_speakers_to_faiss(speakers)
```

**Phase 2: Dual Read** (Prefer FAISS, fallback to JSON)
```python
async def connect(self):
    # Try loading FAISS index
    if self.vector_store.exists():
        logger.info("Loading FAISS index")
        self.vector_store.load()
        return

    # Fallback to JSON
    if self.is_exists(self.embed_json):
        logger.warning("FAISS not found, loading JSON (deprecated)")
        await self._load_from_json()
        # Optionally: migrate to FAISS
        await self._migrate_json_to_faiss()
```

**Phase 3: FAISS Only** (Remove JSON support)
```python
async def connect(self):
    if not self.vector_store.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run 'chatboti rag ingest' to build index."
        )
    self.vector_store.load()
```

### 6.5 CLI Changes

**Current**:
```bash
chatboti rag           # Generate embeddings from hardcoded CSV
chatboti rag-search    # Interactive search
```

**Proposed**:
```bash
# Ingest documents
chatboti rag ingest speakers.csv --type csv \
    --text-columns bio_max_120_words,final_abstract_max_150_words

chatboti rag ingest policy-docs.pdf --chunk-size 512

chatboti rag ingest https://docs.example.com --type web

# Search
chatboti rag search                        # Interactive
chatboti rag search "quantum computing"    # Single query

# Manage index
chatboti rag rebuild                       # Rebuild FAISS index from DB
chatboti rag stats                         # Show index stats
chatboti rag migrate                       # Migrate JSON → FAISS

# Backward compatible (deprecated warnings)
chatboti rag                               # Calls ingest with speakers.csv
```

### 6.6 Backward Compatibility

**Guarantees**:
1. Existing `chatboti rag` command still works (with deprecation warning)
2. Old JSON files can be migrated automatically
3. `get_best_speaker()` method continues to work (wraps `search()`)
4. No breaking changes to agent.py integration

**Breaking Changes** (acceptable for v2.0):
- JSON format no longer updated (read-only migration)
- `get_speakers()` returns different format (query DB instead)
- Environment variable changes (if any)

---

## 7. Performance Considerations

### 7.1 Expected Performance

**Current System** (63 speakers, 126 embeddings):
| Operation | Time |
|-----------|------|
| Load embeddings | 50-100ms |
| Single query | <1ms |
| Memory usage | ~2MB |

**With FAISS** (IndexFlatIP):
| Operation | 100 docs | 1K docs | 10K docs | 100K docs |
|-----------|----------|---------|----------|-----------|
| Build index | 10ms | 50ms | 500ms | 5s |
| Save index | 5ms | 20ms | 200ms | 2s |
| Load index | <5ms | <10ms | <50ms | <200ms |
| Query (k=5) | <1ms | <1ms | 1-5ms | 10-20ms |
| Memory (index) | <1MB | 6MB | 60MB | 600MB |

**Notes**:
- Assumes 1536-dim embeddings (OpenAI text-embedding-3-small)
- IndexFlatIP (exact search)
- Query time scales linearly with index size
- Memory usage: `dimension * num_vectors * 4 bytes`

### 7.2 Scalability Targets

**Small Scale** (100-1000 documents):
- IndexFlatIP (exact search)
- <5ms query latency
- <100MB memory
- Single FAISS index

**Medium Scale** (1K-10K documents):
- IndexFlatIP or IndexIVFFlat
- <10ms query latency
- <1GB memory
- Memory-mapped index

**Large Scale** (10K-100K documents):
- IndexIVFFlat (approximate search)
- <20ms query latency
- <5GB memory
- Consider index sharding

**Very Large Scale** (100K+ documents):
- IndexHNSW or IndexIVFPQ
- <50ms query latency
- GPU acceleration (faiss-gpu)
- Distributed vector DB (migration to Pinecone/Weaviate)

### 7.3 Memory Optimization

**Technique 1: Memory-Mapped Indices**
```python
# FAISS automatically memory-maps large indices
index = faiss.read_index("embeddings.index", faiss.IO_FLAG_MMAP)

# Only loads accessed vectors into RAM
# Reduces startup time for large indices
```

**Technique 2: On-Disk Indices** (future)
```python
# For very large indices that don't fit in RAM
index = faiss.read_index("embeddings.index", faiss.IO_FLAG_ONDISK_SAME_DIR)
```

**Technique 3: Product Quantization** (lossy compression)
```python
# Reduce memory by 8-32x with minimal accuracy loss
pq = faiss.IndexPQ(dimension, M=8, nbits=8)
pq.train(training_vectors)
pq.add(vectors)

# Memory: dimension / M * nbits / 8 bytes per vector
# Example: 1536 / 8 * 1 = 192 bytes (vs 6144 bytes uncompressed)
```

**Recommendation**: Start with memory-mapped IndexFlatIP, add compression only if needed.

### 7.4 Query Optimization

**Batch Queries** (for parallel requests):
```python
# Single query
distances, indices = index.search(query_vector, k=5)

# Batch queries (10-100x faster for many queries)
query_vectors = np.array([vec1, vec2, vec3, ...])  # Shape: (n_queries, dimension)
distances, indices = index.search(query_vectors, k=5)  # Vectorized
```

**GPU Acceleration** (if available):
```bash
pip install faiss-gpu
```

```python
import faiss

# Move index to GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# 10-100x faster for large indices
distances, indices = gpu_index.search(query_vector, k=5)
```

**Pre-filtering** (reduce search space):
```python
# If most queries filter by document type, use separate indices
pdf_index = faiss.read_index("embeddings-pdf.index")
web_index = faiss.read_index("embeddings-web.index")

# Search only relevant index
if doc_type == "pdf":
    results = pdf_index.search(query_vector, k=5)
```

### 7.5 Benchmarking Plan

**Metrics to Track**:
1. Index build time
2. Index save/load time
3. Query latency (p50, p95, p99)
4. Memory usage (peak, steady-state)
5. Index file size

**Test Datasets**:
1. Current: 63 speakers (126 embeddings) - baseline
2. Small: 100 documents (~200 chunks)
3. Medium: 1000 documents (~2000 chunks)
4. Large: 10000 documents (~20000 chunks)

**Benchmark Script**:
```python
import time
import faiss
import numpy as np

def benchmark_index(dimension, num_vectors, num_queries=100):
    # Generate random data
    vectors = np.random.rand(num_vectors, dimension).astype(np.float32)
    faiss.normalize_L2(vectors)

    queries = np.random.rand(num_queries, dimension).astype(np.float32)
    faiss.normalize_L2(queries)

    # Build
    start = time.time()
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    build_time = time.time() - start

    # Save
    start = time.time()
    faiss.write_index(index, "benchmark.index")
    save_time = time.time() - start

    # Load
    start = time.time()
    index = faiss.read_index("benchmark.index")
    load_time = time.time() - start

    # Query
    start = time.time()
    for q in queries:
        index.search(q.reshape(1, -1), k=5)
    query_time = (time.time() - start) / num_queries

    print(f"Vectors: {num_vectors}, Dim: {dimension}")
    print(f"Build: {build_time*1000:.1f}ms")
    print(f"Save: {save_time*1000:.1f}ms")
    print(f"Load: {load_time*1000:.1f}ms")
    print(f"Query: {query_time*1000:.3f}ms")
```

---

## 8. Implementation Roadmap

### Phase 1: FAISS Integration (Week 1-2)

**Goal**: Replace JSON with FAISS for existing speaker data

**Tasks**:
1. Add `faiss-cpu` dependency to `pyproject.toml`
2. Create `FAISSVectorStore` class
   - `__init__`, `add_vectors`, `search`, `save`, `load`
3. Implement FAISS index for current speaker embeddings
   - Use IndexFlatIP with normalized vectors
   - Maintain chunk ID mapping
4. Update `RAGService.connect()` to load FAISS index
5. Update `get_best_speaker()` to use FAISS search
6. Add migration script: JSON → FAISS
7. Test with existing speaker data

**Deliverables**:
- `chatboti/vector_store.py` (FAISS wrapper)
- Updated `chatboti/rag.py`
- Migration script: `scripts/migrate_json_to_faiss.py`
- Tests: `tests/test_faiss_integration.py`

**Acceptance Criteria**:
- All existing RAG tests pass
- FAISS search returns same results as JSON (within floating point error)
- Query latency ≤ current system
- Backward compatible with JSON fallback

### Phase 2: SQLite Metadata Storage (Week 3-4)

**Goal**: Implement metadata storage as per `metadata-storage-design.md`

**Tasks**:
1. Create SQLite schema (documents, chunks, embeddings, faiss_index_mapping)
2. Create `SQLiteMetadataStore` class
   - Document CRUD operations
   - Chunk CRUD operations
   - Index mapping management
3. Update ingestion to write to SQLite + FAISS
4. Update search to query SQLite for metadata
5. Add database migration system (e.g., Alembic or custom)
6. Implement document deduplication (content hashing)

**Deliverables**:
- `chatboti/metadata_store.py`
- SQLite schema: `chatboti/schema.sql`
- Migration tools: `chatboti/migrations/`
- Tests: `tests/test_metadata_store.py`

**Acceptance Criteria**:
- All document/chunk metadata stored in SQLite
- FAISS-SQLite synchronization works correctly
- Can rebuild FAISS index from SQLite
- Database handles concurrent reads (tested)

### Phase 3: Multi-Document Support (Week 5-6)

**Goal**: Add support for multiple document types beyond CSV

**Tasks**:
1. Create `DocumentIngestionPipeline` with pluggable loaders
2. Implement CSV ingester (generalize current speaker logic)
3. Implement PDF ingester (using pypdf2)
4. Implement text/Markdown ingester
5. Implement JSON ingester
6. Add chunking strategies (fixed-size, semantic, field-based)
7. Update CLI with `chatboti rag ingest` command
8. Add document type auto-detection

**Deliverables**:
- `chatboti/ingesters/` (base class + implementations)
- `chatboti/chunking.py` (chunking strategies)
- Updated CLI: `chatboti/cli.py`
- Tests for each ingester: `tests/test_ingesters/`

**Acceptance Criteria**:
- Can ingest CSV, PDF, text, JSON files
- Chunking produces reasonable results
- All chunks properly linked to parent documents
- CLI commands work end-to-end

### Phase 4: Enhanced Search & Filtering (Week 7)

**Goal**: Add metadata filtering and context expansion

**Tasks**:
1. Implement metadata filtering in search
2. Add context expansion (retrieve surrounding chunks)
3. Update `search()` API with filters and options
4. Add search result ranking/reranking (optional)
5. Implement query logging (optional)
6. Add search analytics queries (optional)

**Deliverables**:
- Updated `RAGService.search()` with filters
- Context expansion utility
- Tests: `tests/test_search_filtering.py`

**Acceptance Criteria**:
- Can filter by document type, date, custom metadata
- Context expansion returns coherent text
- Search quality maintained or improved

### Phase 5: Web Ingestion (Week 8, Optional)

**Goal**: Support web page ingestion and crawling

**Tasks**:
1. Implement web page fetcher (httpx + beautifulsoup4)
2. Add HTML → Markdown conversion
3. Implement simple crawler (follow links)
4. Add URL deduplication
5. Handle errors gracefully (404s, timeouts)

**Deliverables**:
- `chatboti/ingesters/web.py`
- Tests: `tests/test_web_ingester.py`

**Acceptance Criteria**:
- Can fetch and chunk web pages
- Crawler respects max_depth
- Handles common errors

### Phase 6: Performance Optimization (Week 9)

**Goal**: Optimize for production use

**Tasks**:
1. Run benchmarks (see Section 7.5)
2. Implement memory-mapped index loading
3. Add batch query support
4. Optimize SQLite queries (add indexes)
5. Profile and fix bottlenecks
6. Add monitoring/logging for performance

**Deliverables**:
- Benchmark results: `docs/performance-benchmarks.md`
- Optimized code
- Performance monitoring utilities

**Acceptance Criteria**:
- Query latency <10ms for 1K documents
- Memory usage <100MB for 1K documents
- Can handle 100 queries/second

### Phase 7: Migration & Cleanup (Week 10)

**Goal**: Complete migration from JSON, finalize API

**Tasks**:
1. Remove dual-write to JSON (FAISS only)
2. Remove JSON loading code (or mark deprecated)
3. Update documentation
4. Update examples and tutorials
5. Add migration guide for users
6. Final testing and bug fixes

**Deliverables**:
- Updated README
- Migration guide: `docs/migration-guide.md`
- Final release: v2.0

**Acceptance Criteria**:
- All JSON code removed or deprecated
- Documentation complete and accurate
- All tests passing
- Ready for production use

---

## 9. Testing Strategy

### 9.1 Unit Tests

**FAISS Integration** (`tests/test_faiss_vector_store.py`):
```python
def test_add_vectors():
    # Test adding vectors and retrieving by ID

def test_search_accuracy():
    # Test search returns correct top-K results

def test_save_load():
    # Test index persistence

def test_id_mapping():
    # Test FAISS index → chunk ID mapping
```

**Metadata Store** (`tests/test_metadata_store.py`):
```python
def test_document_crud():
    # Create, read, update, delete documents

def test_chunk_crud():
    # CRUD operations for chunks

def test_deduplication():
    # Verify content hashing prevents duplicates

def test_concurrent_access():
    # Multiple readers, single writer
```

**Ingesters** (`tests/test_ingesters/`):
```python
def test_csv_ingestion():
    # Load CSV, verify chunks created

def test_pdf_ingestion():
    # Load PDF, verify text extraction and chunking

def test_chunking_strategies():
    # Fixed-size, semantic, field-based
```

### 9.2 Integration Tests

**End-to-End Ingestion** (`tests/test_e2e_ingestion.py`):
```python
async def test_ingest_and_search():
    # 1. Ingest document
    # 2. Search for known content
    # 3. Verify results

async def test_multi_document_ingestion():
    # Ingest CSV + PDF + text, search across all
```

**Backward Compatibility** (`tests/test_backward_compat.py`):
```python
async def test_json_migration():
    # Load old JSON file, migrate to FAISS, verify equivalence

async def test_get_best_speaker():
    # Deprecated method still works
```

### 9.3 Performance Tests

**Benchmark Suite** (`tests/benchmark_faiss.py`):
```python
def benchmark_index_build(num_vectors):
    # Measure index build time

def benchmark_query_latency(num_vectors, num_queries):
    # Measure query performance

def benchmark_memory_usage(num_vectors):
    # Measure memory consumption
```

### 9.4 Acceptance Tests

**User Stories**:
1. "As a user, I can search speaker bios and get relevant results"
2. "As a user, I can ingest a PDF and search its content"
3. "As a user, I can filter search results by document type"
4. "As a user, I can migrate from JSON to FAISS without data loss"

---

## 10. Risks and Mitigations

### Risk 1: FAISS Index Corruption

**Impact**: High - Search fails if index corrupted

**Mitigation**:
- Store vectors in SQLite as backup (Option B from Section 3.4)
- Implement index rebuild from metadata DB
- Add checksum validation on index load
- Regular backups of index files

### Risk 2: FAISS-SQLite Synchronization Bugs

**Impact**: High - Wrong search results if mapping incorrect

**Mitigation**:
- Extensive integration tests
- Validate mapping consistency on startup
- Add repair tool to detect and fix mismatches
- Use transactions to ensure atomic updates

### Risk 3: Performance Regression

**Impact**: Medium - Slower than current system

**Mitigation**:
- Benchmark at each phase
- Keep JSON loading as fallback initially
- Profile and optimize before removing JSON
- Set clear performance targets (Section 7.1)

### Risk 4: Breaking Changes for Users

**Impact**: Medium - User code breaks after upgrade

**Mitigation**:
- Maintain backward compatibility (Section 6.6)
- Provide migration guide
- Deprecation warnings, not immediate removal
- Semantic versioning (v2.0 for breaking changes)

### Risk 5: Large Document Chunking Issues

**Impact**: Medium - Poor quality chunks hurt search quality

**Mitigation**:
- Test with diverse document types
- Implement multiple chunking strategies
- Allow custom chunking functions
- Add chunk quality metrics (token count, coherence)

### Risk 6: SQLite Concurrency Issues

**Impact**: Low - Database locks under high load

**Mitigation**:
- SQLite handles concurrent reads well
- Use WAL mode for better concurrency
- Document single-writer limitation
- Plan migration to PostgreSQL if needed (>10K docs)

---

## 11. Open Questions

1. **Should we store vectors in SQLite as backup?**
   - Pro: Can rebuild FAISS index if corrupted
   - Con: 2x storage, sync complexity
   - Decision: Start without, add if reliability issues arise

2. **Which chunking strategy as default?**
   - Fixed-size (512 tokens, 50 overlap) seems safest
   - Document type-specific defaults?
   - Allow user override?

3. **Should we support multiple embedding models simultaneously?**
   - Increases storage/compute
   - Useful for comparing models
   - Decision: Yes, but make it optional

4. **When to migrate from IndexFlatIP to IndexIVFFlat?**
   - Auto-detect at 10K documents?
   - Make it a config option?
   - Provide migration tool?

5. **Should we implement a caching layer?**
   - Cache frequent queries
   - Cache chunk metadata
   - Use Redis or in-memory LRU?
   - Decision: No caching in v2.0, add if needed

6. **How to handle embedding model updates?**
   - New model version changes embeddings
   - Versioning strategy?
   - Decision: Store model version in embeddings table, rebuild if changed

---

## 12. Success Metrics

**Functional Metrics**:
- Can ingest CSV, PDF, text, JSON documents
- Search returns relevant results (manual validation)
- Backward compatible with existing speaker search
- Zero data loss during JSON → FAISS migration

**Performance Metrics**:
- Query latency <10ms for 1K documents
- Index build time <1s for 1K documents
- Memory usage <100MB for 1K documents
- Support 10K documents without degradation

**Quality Metrics**:
- Code coverage >80%
- All unit tests passing
- All integration tests passing
- Documentation complete

**User Experience Metrics**:
- Clear error messages
- Simple CLI commands
- Easy migration path
- Good logging/debugging

---

## 13. Future Enhancements

**Post v2.0 Features**:

1. **Hybrid Search** (vector + full-text)
   - Combine FAISS with SQLite FTS5
   - Weighted score merging (e.g., 0.7 * vector + 0.3 * keyword)

2. **Reranking Models**
   - Post-filter FAISS results with cross-encoder
   - Improve top-K quality

3. **Multi-Modal Embeddings**
   - Image embeddings (CLIP)
   - Audio embeddings (Whisper)

4. **Distributed Vector Search**
   - Shard FAISS index across multiple machines
   - Or migrate to Pinecone/Weaviate for cloud scale

5. **Query Analytics**
   - Track popular queries
   - A/B test chunking strategies
   - Measure search quality over time

6. **Auto-Tuning**
   - Detect optimal chunk size per document type
   - Auto-select index type based on dataset size
   - Self-optimizing hyperparameters

7. **Version Control for Documents**
   - Track document updates
   - Rebuild embeddings for changed documents only
   - Query specific document versions

8. **Streaming Ingestion**
   - Process large files without loading into memory
   - Incremental index updates

---

## Appendix A: Dependencies

**New Dependencies**:
```toml
[project.dependencies]
faiss-cpu = "^1.7.4"        # Vector similarity search
pypdf2 = "^3.0.0"           # PDF text extraction (optional)
beautifulsoup4 = "^4.12.0"  # HTML parsing (optional)
httpx = "^0.25.0"           # Async HTTP client (optional)
```

**Existing Dependencies**:
- `numpy`: Vector operations (already used)
- `sqlite3`: Metadata storage (built-in)
- `microeval`: LLM client (already used)

**Optional Dependencies**:
```toml
[project.optional-dependencies]
pdf = ["pypdf2"]
web = ["beautifulsoup4", "httpx"]
gpu = ["faiss-gpu"]  # Alternative to faiss-cpu
```

---

## Appendix B: References

**FAISS Resources**:
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [FAISS Index Types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

**RAG Resources**:
- [LangChain: Retrieval](https://python.langchain.com/docs/modules/data_connection/)
- [OpenAI: Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Building Production-Ready RAG Applications](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)

**Chunking Strategies**:
- [Five Levels of Text Splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
- [Semantic Chunking](https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/character.py)

**Related Documents**:
- `vector-storage-comparison.md`: Parquet vs FAISS comparison
- `metadata-storage-design.md`: SQLite schema design

---

**Document Version**: 1.0
**Author**: Claude Sonnet 4.5
**Date**: 2026-02-12
**Status**: Proposed Specification
