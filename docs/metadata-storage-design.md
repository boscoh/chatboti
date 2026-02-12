# Metadata Storage Design for RAG System

## Executive Summary

This document outlines the design for metadata storage in the Chatboti RAG system. The current implementation stores embeddings inline with document data in JSON format, which limits scalability and query efficiency. This design proposes a structured metadata architecture that supports document tracking, embedding index mapping, and efficient text retrieval.

**Recommendation**: SQLite with separate tables for documents, chunks, and embeddings, with JSON as a fallback for simple deployments.

## Current System Overview

### Architecture
The current RAG implementation (`chatboti/rag.py`) uses a simple in-memory architecture:

```
CSV Source (speakers) → JSON with embedded vectors → In-memory search
```

### Data Model
- **Source**: `2025-09-02-speaker-bio.csv` contains speaker information
- **Storage**: `embeddings-{model-name}.json` stores documents with inline embeddings
- **Structure**: Each speaker entry contains:
  - Document fields (name, bio, abstract, etc.)
  - Two embedding vectors (abstract_embedding, bio_embedding)
  - No explicit IDs or chunk metadata

### Current Limitations
1. **No document tracking**: No source file metadata, timestamps, or versioning
2. **No index mapping**: Embeddings stored inline without separate vector-to-document mapping
3. **No chunking support**: Single document = single embedding pair (abstract + bio)
4. **No retrieval metadata**: No tracking of which chunks were returned or relevance scores
5. **Scalability issues**: Full dataset loaded into memory; large embeddings duplicate data
6. **Query inefficiency**: Linear scan through all embeddings; no indexing support

## Design Requirements

### Functional Requirements
1. **Document Source Tracking**
   - Source file path/URL
   - Import timestamp
   - Document version/hash
   - Source type (CSV, PDF, web, etc.)

2. **Embedding Index Mapping**
   - Unique vector ID
   - Document ID reference
   - Chunk ID reference
   - Embedding model metadata
   - Vector storage or reference

3. **Chunked Text Storage**
   - Original document text
   - Chunk boundaries and metadata
   - Chunk overlap information
   - Chunk-specific metadata (e.g., section, paragraph)

4. **Retrieval Index**
   - Query history
   - Retrieved chunks per query
   - Relevance scores
   - User feedback (optional)

### Non-Functional Requirements
1. **Performance**: Sub-100ms metadata lookups
2. **Scalability**: Support 10K+ documents, 100K+ chunks
3. **Maintainability**: Clear schema, easy migrations
4. **Portability**: Work in Docker, serverless, local dev
5. **Simplicity**: Minimal dependencies, straightforward deployment

## Proposed Metadata Schema

### 1. Documents Table
Primary table tracking source documents.

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,              -- UUID or hash-based ID
    source_path TEXT NOT NULL,        -- File path or URL
    source_type TEXT NOT NULL,        -- csv, pdf, web, text, etc.
    content_hash TEXT,                -- SHA256 of content for deduplication
    imported_at TIMESTAMP NOT NULL,   -- When document was ingested
    updated_at TIMESTAMP,             -- Last update time
    metadata JSON,                    -- Flexible field for source-specific metadata
    raw_text TEXT                     -- Optional: store full text
);

CREATE INDEX idx_documents_source ON documents(source_path);
CREATE INDEX idx_documents_hash ON documents(content_hash);
```

**Example metadata JSON**:
```json
{
    "speaker_name": "Jonathan Passé",
    "country": "Germany & UK",
    "email": "jpasse@thoughtworks.com",
    "file_size": 2048,
    "encoding": "utf-8"
}
```

### 2. Chunks Table
Tracks text chunks extracted from documents.

```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,              -- UUID
    document_id TEXT NOT NULL,        -- Foreign key to documents
    chunk_index INTEGER NOT NULL,     -- Position in document (0-based)
    chunk_type TEXT,                  -- abstract, bio, paragraph, section
    text TEXT NOT NULL,               -- The actual chunk text
    start_char INTEGER,               -- Character offset in original document
    end_char INTEGER,                 -- End character offset
    token_count INTEGER,              -- Approximate tokens (for context limits)
    metadata JSON,                    -- Chunk-specific metadata
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX idx_chunks_document ON chunks(document_id);
CREATE INDEX idx_chunks_type ON chunks(chunk_type);
```

**Example metadata JSON**:
```json
{
    "overlap_with_prev": 50,
    "overlap_with_next": 50,
    "section_title": "Biography",
    "word_count": 120
}
```

### 3. Embeddings Table
Stores embedding vectors and metadata.

```sql
CREATE TABLE embeddings (
    id TEXT PRIMARY KEY,              -- UUID
    chunk_id TEXT NOT NULL,           -- Foreign key to chunks
    model_name TEXT NOT NULL,         -- e.g., text-embedding-3-small
    model_version TEXT,               -- Model version/variant
    vector_dimension INTEGER NOT NULL,-- 1536, 768, etc.
    vector BLOB NOT NULL,             -- Binary-encoded numpy array or JSON
    created_at TIMESTAMP NOT NULL,    -- When embedding was generated
    metadata JSON,                    -- Model-specific metadata
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE INDEX idx_embeddings_chunk ON embeddings(chunk_id);
CREATE INDEX idx_embeddings_model ON embeddings(model_name);
```

**Example metadata JSON**:
```json
{
    "api_version": "v1",
    "latency_ms": 245,
    "cost": 0.00001,
    "normalization": "l2"
}
```

### 4. Queries Table (Optional)
Tracks retrieval queries for analytics and debugging.

```sql
CREATE TABLE queries (
    id TEXT PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_embedding BLOB,
    model_name TEXT,
    timestamp TIMESTAMP NOT NULL,
    metadata JSON
);

CREATE INDEX idx_queries_timestamp ON queries(timestamp);
```

### 5. Retrievals Table (Optional)
Links queries to retrieved chunks.

```sql
CREATE TABLE retrievals (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    rank INTEGER NOT NULL,            -- Retrieval rank (1 = best match)
    score REAL NOT NULL,              -- Cosine similarity or distance
    metadata JSON,
    FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE INDEX idx_retrievals_query ON retrievals(query_id);
CREATE INDEX idx_retrievals_chunk ON retrievals(chunk_id);
```

## Storage Approach Comparison

### Option 1: SQLite (Recommended)

**Pros**:
- ACID transactions ensure data consistency
- Built-in indexing for fast lookups
- Foreign key constraints maintain referential integrity
- Single-file database (easy deployment, backups)
- Zero configuration, no server required
- JSON column support for flexible metadata
- Excellent Python support (`sqlite3` built-in)
- Full-text search via FTS5 extension
- Supports concurrent reads (single writer)

**Cons**:
- No native vector similarity search (requires external indexes)
- Single writer limitation (not suitable for high-concurrency writes)
- BLOB storage for vectors less efficient than specialized formats
- Maximum database size ~281 TB (not an issue for most use cases)

**Best for**: Production deployments, persistent storage, complex queries, data integrity

### Option 2: JSON Files

**Pros**:
- Simple to implement and understand
- Human-readable for debugging
- No external dependencies
- Easy to version control (small datasets)
- Works well for current single-file approach

**Cons**:
- No indexing (full file read for every query)
- No transactions (corruption risk on partial writes)
- Memory intensive (entire file loaded)
- Poor performance with large datasets (>1000 documents)
- No efficient updates (rewrite entire file)
- No relationship enforcement

**Best for**: Development, simple demos, datasets <100 documents

### Option 3: Parquet Files

**Pros**:
- Columnar format (efficient for large datasets)
- Excellent compression ratios
- Fast column-specific reads
- Schema enforcement
- Popular in data science workflows
- Good for batch processing

**Cons**:
- Requires `pyarrow` or `fastparquet` dependency
- Not ideal for frequent updates (append-only)
- No ACID transactions
- No foreign key constraints
- Limited query capabilities (no indexes)
- Binary format (not human-readable)

**Best for**: Batch processing, data lake integration, read-heavy workloads

### Option 4: Separate Files (CSV + JSON + NPY)

**Pros**:
- Specialized format per data type
- Efficient vector storage (`.npy` for numpy arrays)
- Human-readable text files
- Simple backup/restore

**Cons**:
- Manual relationship management across files
- No transactional consistency
- Complex synchronization logic
- Difficult to query relationships
- Higher risk of data corruption
- Fragmented storage

**Best for**: File-based pipelines, offline processing, legacy compatibility

### Option 5: Vector Databases (Pinecone, Weaviate, Chroma)

**Pros**:
- Native vector similarity search
- Optimized for embedding retrieval
- Horizontal scalability
- Built-in filtering and metadata
- Real-time indexing

**Cons**:
- External service dependency (operational overhead)
- Cost considerations (hosted services)
- Network latency for queries
- Overkill for simple use cases
- More complex deployment

**Best for**: Large-scale production, real-time search, multi-tenant systems

## Comparison Matrix

| Feature               | SQLite | JSON | Parquet | Separate Files | Vector DB |
|-----------------------|--------|------|---------|----------------|-----------|
| Setup Complexity      | Low    | Low  | Medium  | Medium         | High      |
| Query Performance     | High   | Low  | Medium  | Low            | Very High |
| Update Performance    | High   | Low  | Low     | Medium         | High      |
| Data Integrity        | High   | Low  | Low     | Low            | High      |
| Deployment Simplicity | High   | High | Medium  | Medium         | Low       |
| Scalability           | Medium | Low  | High    | Low            | Very High |
| Vector Search         | Low    | Low  | Low     | Low            | Very High |
| Cost                  | Free   | Free | Free    | Free           | $$-$$$    |

## Query Patterns and Access Patterns

### Primary Access Patterns

1. **Vector Similarity Search**
   - Query: Embedding vector
   - Returns: Top-K most similar chunks with metadata
   - Frequency: Every user query (high)
   - Optimization: Requires external vector index (FAISS, HNSW)

2. **Document Retrieval**
   - Query: Document ID or source path
   - Returns: Document metadata and all chunks
   - Frequency: Moderate
   - Optimization: Index on `documents.id` and `documents.source_path`

3. **Chunk Context Retrieval**
   - Query: Chunk ID
   - Returns: Surrounding chunks for context expansion
   - Frequency: After similarity search (high)
   - Optimization: Index on `chunks.document_id` and `chunks.chunk_index`

4. **Metadata Filtering**
   - Query: Filter by document type, date range, or custom metadata
   - Returns: Filtered document set for search scope
   - Frequency: Moderate
   - Optimization: JSON path indexes on metadata fields

5. **Model-Specific Embedding Lookup**
   - Query: Chunk ID + model name
   - Returns: Specific embedding vector
   - Frequency: High (during search)
   - Optimization: Compound index on `(chunk_id, model_name)`

### Example Queries

#### 1. Retrieve Top-K Similar Chunks
```sql
-- Pseudocode (vector similarity handled externally)
-- 1. External: Find top K chunk IDs using vector index
-- 2. Fetch metadata from SQLite:

SELECT
    c.id AS chunk_id,
    c.text,
    c.chunk_type,
    c.metadata AS chunk_metadata,
    d.id AS document_id,
    d.source_path,
    d.metadata AS document_metadata
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE c.id IN (?, ?, ?, ...)  -- IDs from vector search
ORDER BY FIELD(c.id, ?, ?, ?, ...);  -- Preserve ranking
```

#### 2. Get Document with All Chunks
```sql
SELECT
    d.*,
    c.id AS chunk_id,
    c.chunk_index,
    c.chunk_type,
    c.text,
    c.metadata AS chunk_metadata
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
WHERE d.id = ?
ORDER BY c.chunk_index;
```

#### 3. Expand Context Around Chunk
```sql
SELECT
    id,
    chunk_index,
    text,
    chunk_type
FROM chunks
WHERE document_id = (SELECT document_id FROM chunks WHERE id = ?)
  AND chunk_index BETWEEN
    (SELECT chunk_index - 1 FROM chunks WHERE id = ?) AND
    (SELECT chunk_index + 1 FROM chunks WHERE id = ?)
ORDER BY chunk_index;
```

#### 4. Filter by Metadata and Date
```sql
SELECT d.id, d.source_path, d.imported_at
FROM documents d
WHERE json_extract(d.metadata, '$.country') = 'Germany'
  AND d.imported_at > datetime('now', '-7 days');
```

## Recommended Architecture

### Hybrid Approach: SQLite + Vector Index

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│                     Application Layer                   │
└───────────────┬─────────────────────────────────────────┘
                │
      ┌─────────┴──────────┐
      │                    │
      ▼                    ▼
┌─────────────┐    ┌──────────────┐
│   SQLite    │    │ Vector Index │
│  Metadata   │    │   (FAISS)    │
│   Storage   │    │              │
└─────────────┘    └──────────────┘
  Documents           Embeddings
  Chunks              (vectors only)
  Embeddings
  (metadata only)
```

**Components**:
1. **SQLite Database** (`chatboti.db`): Stores all metadata (documents, chunks, embedding metadata)
2. **FAISS Index** (`embeddings.index`): Stores vectors for fast similarity search
3. **Synchronization**: Vector IDs in FAISS map to chunk IDs in SQLite

**Rationale**:
- SQLite handles structured metadata with ACID guarantees
- FAISS provides optimized vector similarity search
- Clear separation of concerns
- Both are embedded (no external services)
- Portable (single directory with 2-3 files)

### Alternative: JSON for Simple Cases

For small datasets (<500 documents), continue using JSON:
- Single file: `metadata-{model}.json`
- Include document source, chunks, and embeddings
- Trade simplicity for performance

**Migration path**: Start with JSON, migrate to SQLite when needed

## Implementation Considerations

### Vector Storage in SQLite

**Option A: Store vectors as BLOB**
```python
import numpy as np
import sqlite3

# Serialize
vector_blob = np.array(embedding, dtype=np.float32).tobytes()

# Deserialize
vector = np.frombuffer(vector_blob, dtype=np.float32)
```

**Option B: Store vectors as JSON**
```python
import json

# Serialize
vector_json = json.dumps(embedding)

# Deserialize
vector = json.loads(vector_json)
```

**Recommendation**: Use BLOB for space efficiency (4 bytes per float vs ~8-12 bytes in JSON)

### Chunking Strategy

Current system uses pre-defined chunks (abstract, bio). For general documents:

1. **Fixed-size chunking**: 512 tokens with 50-token overlap
2. **Semantic chunking**: Paragraph or section boundaries
3. **Hybrid**: Respect boundaries, max 512 tokens

**Metadata to track**:
- Chunk boundaries (start/end chars)
- Overlap with adjacent chunks
- Token count (for LLM context limits)

### Document Deduplication

Use `content_hash` to detect duplicates:
```python
import hashlib

content_hash = hashlib.sha256(document_text.encode()).hexdigest()
```

Before ingestion, check if hash exists:
```sql
SELECT id FROM documents WHERE content_hash = ?
```

### Migration from Current System

**Phase 1**: Add metadata layer without breaking current code
```python
# Keep existing JSON format
# Add parallel SQLite storage
# Dual-write during transition
```

**Phase 2**: Read from SQLite, write to both
```python
# Primary reads from SQLite
# Fallback to JSON if missing
# Continue dual-write
```

**Phase 3**: SQLite only
```python
# Remove JSON read/write
# Delete legacy files after backup
```

## Deployment Considerations

### Docker
- Include `chatboti.db` in image or mount as volume
- FAISS index can be in image (read-only) or volume (if updated)
- SQLite works well in containers (single file)

### AWS Lambda / Serverless
- Store `chatboti.db` in `/tmp` or EFS mount
- Consider read-only deployment (pre-built index)
- Watch for 512MB Lambda limit (use EFS for large indexes)

### Local Development
- SQLite requires no setup (built into Python)
- Database file can be in git (if small) or gitignored (if large)
- Easy to reset: delete `.db` file and regenerate

## Performance Estimates

Based on typical hardware (modern laptop, SSD):

| Operation                    | SQLite | JSON | Parquet |
|------------------------------|--------|------|---------|
| Lookup 1 document by ID      | <1ms   | 50ms | 10ms    |
| Lookup 10 chunks by ID       | <5ms   | 50ms | 15ms    |
| Insert 100 documents         | 50ms   | 200ms| 100ms   |
| Full-text search             | 10ms   | 500ms| N/A     |
| Load entire dataset to memory| 100ms  | 50ms | 80ms    |

**Note**: Vector similarity search dominates query time (10-100ms for FAISS on 10K vectors), so metadata lookup overhead is negligible.

## Future Enhancements

1. **Vector Database Integration**: Migrate to Chroma or Weaviate when scaling beyond 100K chunks
2. **Multi-Modal Support**: Store image embeddings, audio transcripts
3. **Versioning**: Track document updates and embedding regenerations
4. **Analytics**: Query logs, retrieval quality metrics
5. **Distributed Storage**: Shard across multiple SQLite files or migrate to PostgreSQL
6. **Caching Layer**: Redis for hot chunks/queries

## Conclusion

**Recommendation**: Implement SQLite-based metadata storage with FAISS for vector search.

**Rationale**:
- Provides robust metadata tracking without external dependencies
- Maintains simplicity of deployment (embedded database)
- Scales to 10K-100K documents (sufficient for current needs)
- Clear migration path to vector databases if needed
- Industry-standard tools with strong Python support

**Next Steps**:
1. Define exact schema based on this design
2. Implement database initialization and migration scripts
3. Build ingestion pipeline with SQLite writes
4. Integrate FAISS for vector search
5. Update RAG service to use new metadata layer
6. Add tests for concurrent access and data integrity
7. Document backup and restore procedures

---

**Document Version**: 1.0
**Author**: Claude Sonnet 4.5
**Date**: 2026-02-12
**Status**: Proposed Design
