# Single-File RAG Backend Format Specification

## Executive Summary

This specification defines a **single markdown file format** for RAG (Retrieval-Augmented Generation) backend storage as an alternative to the current multi-file approach (FAISS `.faiss` + JSON/SQLite metadata). The format embeds both vector data and document metadata in a human-readable, version-controllable markdown file with structured frontmatter and base64-encoded binary sections.

**Use Case**: Small-to-medium document collections (10-1000 documents) where simplicity, portability, and version control are prioritized over maximum performance.

**Tradeoff**: Sacrifices some performance and scalability for simplicity, readability, and git-friendliness.

**Status**: Proposed Design
**Author**: Claude Sonnet 4.5
**Date**: 2026-02-14
**Related**: generic-document-storage-spec.md

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Alternative Standard Formats](#2-alternative-standard-formats)
3. [Format Specification](#3-format-specification)
4. [File Structure](#4-file-structure)
5. [Encoding Scheme](#5-encoding-scheme)
6. [API Compatibility](#6-api-compatibility)
7. [Implementation](#7-implementation)
8. [Performance Characteristics](#8-performance-characteristics)
9. [Limitations](#9-limitations)
10. [Migration Path](#10-migration-path)
11. [Example File](#11-example-file)

---

## 1. Motivation

### 1.1 Problems with Current Multi-File Approach

**Current Storage**: `vectors-{model}.faiss` + `metadata-{model}.json`

**Issues**:
- **Two files to manage**: Index and metadata must be kept in sync
- **Binary FAISS file**: Not human-readable, not git-friendly, hard to inspect
- **Sync risks**: Possible to have mismatched index/metadata after partial updates
- **Portability**: Requires both files for complete dataset
- **Version control**: Binary FAISS files bloat git history

### 1.2 Benefits of Single-File Format

✓ **Single source of truth**: All data in one file
✓ **Human-readable metadata**: YAML frontmatter + markdown sections
✓ **Inspectable vectors**: Base64-encoded, can be decoded for debugging
✓ **Git-friendly**: Text-based format with meaningful diffs (metadata changes visible)
✓ **Portable**: Copy one file, get complete dataset
✓ **Self-documenting**: Embedded schema and statistics
✓ **No sync issues**: Atomically consistent

### 1.3 When to Use

**Good fit**:
- Small-to-medium collections (10-1000 documents)
- Prototype/research projects
- Version-controlled knowledge bases
- Sharing datasets with collaborators
- Embedded deployments

**Not suitable**:
- Large-scale production systems (>10K documents)
- High-throughput search services
- Frequent updates (embeddings regeneration)
- Memory-constrained environments

---

## 2. Alternative Standard Formats

**Note**: The `.ragmd` format specified in this document is a **custom design**, not an industry standard. Before implementing it, consider these proven single-file alternatives:

### 2.1 HDF5 (.h5) - Industry Standard for Scientific Data

**What it is**: Hierarchical Data Format version 5 - binary container for large scientific datasets with metadata.

**Status**: ✓ Industry standard, mature (since 1998), widely used in ML/scientific computing

**Structure**:
```
embeddings.h5
├── /metadata
│   ├── model_name (attribute: "nomic-embed-text")
│   ├── embedding_dim (attribute: 768)
│   └── created_at (attribute: "2026-02-14T10:00:00Z")
├── /vectors (dataset: float32 array [n_chunks × embedding_dim])
├── /chunks (dataset: structured array)
│   ├── faiss_id (int64)
│   ├── document_id (string)
│   └── chunk_key (string)
└── /documents (group)
    ├── /doc-001 (group)
    │   ├── id (attribute)
    │   ├── source (attribute)
    │   ├── full_text (dataset: string)
    │   └── chunks (dataset: JSON or structured array)
    └── /doc-002 (group)
        └── ...
```

**Advantages**:
- ✓ Highly efficient binary format (no base64 overhead)
- ✓ Partial loading (load only needed vectors/metadata)
- ✓ Compression built-in (gzip, lzf)
- ✓ Excellent Python support (`h5py`, `tables`)
- ✓ Self-describing (embedded metadata)
- ✓ Cross-platform, cross-language (C/C++/Java/R/Julia)
- ✓ Proven at massive scale (used by NASA, CERN)

**Disadvantages**:
- ✗ Binary format (not human-readable)
- ✗ Not git-friendly (binary diffs)
- ✗ Requires external library (`h5py`)
- ✗ File corruption risk (not as robust as SQLite)

**Python Example**:
```python
import h5py
import numpy as np

# Write
with h5py.File('embeddings.h5', 'w') as f:
    # Metadata
    f.attrs['model_name'] = 'nomic-embed-text'
    f.attrs['embedding_dim'] = 768

    # Vectors (n_chunks × dim)
    f.create_dataset('vectors', data=vectors_array, compression='gzip')

    # Chunks (structured array)
    chunk_dtype = np.dtype([
        ('faiss_id', 'i8'),
        ('document_id', 'U64'),
        ('chunk_key', 'U64')
    ])
    f.create_dataset('chunks', data=chunk_refs_array, dtype=chunk_dtype)

    # Documents (nested groups)
    docs_group = f.create_group('documents')
    for doc in documents:
        doc_group = docs_group.create_group(doc.id)
        doc_group.attrs['source'] = doc.source
        doc_group.create_dataset('full_text', data=doc.full_text)

# Read
with h5py.File('embeddings.h5', 'r') as f:
    model_name = f.attrs['model_name']
    vectors = f['vectors'][:]  # Load all vectors
    chunks = f['chunks'][:]    # Load all chunks

    # Partial loading (efficient)
    vector_10 = f['vectors'][10]  # Load single vector
    doc_text = f['documents/doc-001/full_text'][()]
```

**Use when**:
- Working with large datasets (>1GB)
- Need partial loading (don't load all vectors into memory)
- Performance is critical
- Using scientific Python stack (NumPy, pandas, scikit-learn)

**Integration with FAISS**:
```python
# Build FAISS index from HDF5 vectors
with h5py.File('embeddings.h5', 'r') as f:
    vectors = f['vectors'][:]
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
```

---

### 2.3 DuckDB with Parquet - Modern SQL + Columnar Storage

**What it is**: Embedded SQL database (like SQLite) optimized for analytics, using Parquet columnar format for efficient vector storage.

**Status**: ✓ Production-ready, modern (since 2019), designed for analytics workloads

**Structure**:
```sql
-- Single .duckdb file with embedded Parquet
CREATE TABLE embeddings (
    faiss_id INTEGER PRIMARY KEY,
    document_id VARCHAR,
    chunk_key VARCHAR,
    vector FLOAT[768],  -- Array type for embeddings
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE TABLE documents (
    id VARCHAR PRIMARY KEY,
    source VARCHAR,
    full_text TEXT,
    content JSON,
    created_at TIMESTAMP
);

-- Or export to separate Parquet file
COPY embeddings TO 'embeddings.parquet' (FORMAT PARQUET, COMPRESSION 'zstd');
```

**Advantages**:
- ✓ Single file (`.duckdb`) with all data
- ✓ SQL interface (familiar, powerful queries)
- ✓ ACID transactions (safe concurrent access)
- ✓ Native array/vector types
- ✓ Excellent compression (Parquet with zstd)
- ✓ Fast analytical queries (columnar storage)
- ✓ No server needed (embedded)
- ✓ Export to Parquet (interop with pandas, Arrow, Spark)
- ✓ Zero-copy integration with NumPy/pandas

**Disadvantages**:
- ✗ Not human-readable
- ✗ Not git-friendly
- ✗ No native FAISS integration (must extract to NumPy)
- ✗ Slower than raw FAISS for vector search

**Python Example**:
```python
import duckdb
import numpy as np

# Write
con = duckdb.connect('embeddings.duckdb')

con.execute("""
    CREATE TABLE embeddings (
        faiss_id INTEGER PRIMARY KEY,
        document_id VARCHAR,
        chunk_key VARCHAR,
        vector FLOAT[768]
    )
""")

con.execute("""
    CREATE TABLE documents (
        id VARCHAR PRIMARY KEY,
        source VARCHAR,
        full_text TEXT,
        content JSON
    )
""")

# Insert vectors (convert numpy to list)
for i, (doc_id, chunk_key, vec) in enumerate(data):
    con.execute(
        "INSERT INTO embeddings VALUES (?, ?, ?, ?)",
        (i, doc_id, chunk_key, vec.tolist())
    )

# Insert documents
con.execute(
    "INSERT INTO documents VALUES (?, ?, ?, ?)",
    (doc.id, doc.source, doc.full_text, json.dumps(doc.content))
)

con.close()

# Read
con = duckdb.connect('embeddings.duckdb')

# Load all vectors as numpy array
vectors = con.execute("""
    SELECT vector FROM embeddings ORDER BY faiss_id
""").fetchnumpy()['vector']

# Efficient: export to Parquet, load with pyarrow
con.execute("COPY embeddings TO 'vectors.parquet' (FORMAT PARQUET)")
import pyarrow.parquet as pq
table = pq.read_table('vectors.parquet')
vectors_np = np.stack(table['vector'].to_numpy())

# Build FAISS index
index = faiss.IndexFlatIP(768)
index.add(vectors_np)

# Query with SQL
results = con.execute("""
    SELECT d.full_text, e.chunk_key
    FROM embeddings e
    JOIN documents d ON e.document_id = d.id
    WHERE e.faiss_id IN (10, 23, 45)
""").fetchall()

con.close()
```

**Use when**:
- Need SQL querying capabilities
- Want metadata filtering before vector search
- Working with structured document metadata
- Need ACID transactions
- Integrating with data analytics tools

**Hybrid workflow** (DuckDB + FAISS):
```python
# 1. Store everything in DuckDB
# 2. At startup, load vectors into FAISS for fast search
# 3. Use FAISS for vector search → get faiss_ids
# 4. Use DuckDB for metadata lookup and filtering

con = duckdb.connect('embeddings.duckdb')

# Load vectors into FAISS once
vectors = con.execute("SELECT vector FROM embeddings ORDER BY faiss_id").fetchnumpy()['vector']
index = faiss.IndexFlatIP(768)
index.add(np.stack(vectors))

# Search with FAISS
query_vec = embed_model.embed("query")
distances, faiss_ids = index.search(query_vec, k=5)

# Fetch metadata with DuckDB (fast indexed lookups)
results = con.execute(f"""
    SELECT e.document_id, e.chunk_key, d.full_text
    FROM embeddings e
    JOIN documents d ON e.document_id = d.id
    WHERE e.faiss_id IN ({','.join(map(str, faiss_ids[0]))})
""").fetchall()
```

---

### 2.4 Comparison: Custom .ragmd vs Standard Formats

| Feature | .ragmd (Custom) | HDF5 | DuckDB + Parquet |
|---------|-----------------|------|------------------|
| **Human-readable** | ✓ (YAML/JSON) | ✗ | ✗ |
| **Git-friendly** | ✓ (text diffs) | ✗ | ✗ |
| **Single file** | ✓ | ✓ | ✓ |
| **Performance** | Slow (base64) | Fast | Medium |
| **Partial loading** | ✗ | ✓ | ✓ |
| **Compression** | ✗ | ✓ | ✓ |
| **Industry standard** | ✗ | ✓ | ✓ |
| **SQL queries** | ✗ | ✗ | ✓ |
| **FAISS integration** | Direct | Easy | Medium |
| **File size** | Large (+33%) | Small | Small |
| **External deps** | None | h5py | duckdb |
| **Learning curve** | Low | Medium | Medium |

**Recommendation**:
- **Prototyping/education**: Custom `.ragmd` (simple, transparent)
- **Production/scale**: HDF5 (performance, partial loading)
- **Analytics/queries**: DuckDB (SQL interface, metadata filtering)
- **Simple use case**: Current FAISS + JSON (works fine)

---

## 3. Format Specification

### 2.1 File Extension

`.ragmd` (RAG Markdown) or `.md` with special frontmatter marker

### 2.2 MIME Type

`text/markdown` or custom `application/vnd.rag-markdown`

### 2.3 Character Encoding

UTF-8

### 2.4 Structure

```
┌─────────────────────────────┐
│ YAML Frontmatter            │  ← Metadata: model, dimensions, stats
├─────────────────────────────┤
│ ## Documents                │  ← Human-readable document list
├─────────────────────────────┤
│ ## Chunks                   │  ← Chunk references (doc_id → chunk_key)
├─────────────────────────────┤
│ ## Vectors (Base64)         │  ← Binary FAISS index, base64-encoded
└─────────────────────────────┘
```

---

## 4. File Structure

### 3.1 YAML Frontmatter

```yaml
---
format_version: "1.0"
model_name: "nomic-embed-text"
embedding_dim: 768
vector_count: 245
document_count: 42
index_type: "IndexFlatIP"
created_at: "2026-02-14T10:30:00Z"
updated_at: "2026-02-14T10:30:00Z"
---
```

**Fields**:
- `format_version`: Spec version (semantic versioning)
- `model_name`: Embedding model identifier
- `embedding_dim`: Vector dimensionality
- `vector_count`: Total embeddings stored
- `document_count`: Total documents indexed
- `index_type`: FAISS index type (e.g., `IndexFlatIP`, `IndexIVFFlat`)
- `created_at`: ISO 8601 timestamp
- `updated_at`: ISO 8601 timestamp

### 3.2 Documents Section

Human-readable markdown table of all documents:

```markdown
## Documents

| Document ID | Source | Chunks | Title |
|-------------|--------|--------|-------|
| doc-001 | papers/smith2024.pdf | 12 | Neural Architecture Search |
| doc-002 | papers/jones2024.pdf | 8 | Attention Mechanisms |
```

**Purpose**: Quick overview, debugging, manual inspection

### 3.3 Chunks Section

JSON array of chunk references mapping FAISS index positions to document chunks:

```markdown
## Chunks

```json
[
  {"faiss_id": 0, "document_id": "doc-001", "chunk_key": "chunk-0"},
  {"faiss_id": 1, "document_id": "doc-001", "chunk_key": "chunk-1"},
  {"faiss_id": 2, "document_id": "doc-002", "chunk_key": "abstract"}
]
```
```

**Schema**:
- `faiss_id`: Index position in vector array (0-based)
- `document_id`: Reference to document
- `chunk_key`: Chunk identifier within document

### 3.4 Document Metadata Section

Full document metadata as JSON:

```markdown
## Document Metadata

```json
{
  "doc-001": {
    "id": "doc-001",
    "source": "papers/smith2024.pdf",
    "content": {"title": "Neural Architecture Search", "authors": ["Smith, J."]},
    "chunks": {
      "chunk-0": {"i_start": 0, "i_end": 512, "faiss_id": 0},
      "chunk-1": {"i_start": 512, "i_end": 1024, "faiss_id": 1}
    }
  }
}
```
```

**Schema**: Serialized `Document` objects (see `document.py`)

### 3.5 Vectors Section

Base64-encoded FAISS index:

```markdown
## Vectors

```base64
TVRMMj... [truncated for brevity] ...Qo=
```
```

**Encoding**:
1. Serialize FAISS index with `faiss.write_index()` to bytes
2. Base64-encode the binary data
3. Wrap in markdown code fence with `base64` language tag

**Decoding**:
1. Extract base64 string from code fence
2. Decode to bytes
3. Load with `faiss.read_index_binary()` or write to temp file

---

## 5. Encoding Scheme

### 4.1 FAISS Index Serialization

```python
import faiss
import base64
import io

def serialize_index(index: faiss.Index) -> str:
    """Serialize FAISS index to base64 string."""
    # Write index to bytes
    index_bytes = io.BytesIO()
    faiss.write_index(index, faiss.BufferedIOWriter(
        faiss.PyCallbackIOWriter(index_bytes.write)
    ))
    # Encode as base64
    return base64.b64encode(index_bytes.getvalue()).decode('utf-8')

def deserialize_index(b64_string: str) -> faiss.Index:
    """Deserialize FAISS index from base64 string."""
    # Decode base64
    index_bytes = base64.b64decode(b64_string)
    # Load index from bytes
    reader = faiss.BufferedIOReader(
        faiss.PyCallbackIOReader(io.BytesIO(index_bytes).read)
    )
    return faiss.read_index(reader)
```

### 4.2 Document Serialization

Use existing `Document.to_dict()` / `Document.from_dict()` methods:

```python
documents_json = json.dumps(
    {doc_id: doc.to_dict() for doc_id, doc in documents.items()},
    indent=2
)
```

---

## 6. API Compatibility

### 5.1 Backend Interface

The single-file backend implements the same interface as `GenericRAGService`:

```python
class SingleFileRAGService(GenericRAGService):
    """RAG service using single markdown file backend."""

    def __init__(self, service_name: str, model: str, ragmd_path: Path):
        """Initialize with path to .ragmd file."""
        ...

    def initialize_search_backend(self):
        """Load from .ragmd file instead of FAISS + JSON."""
        ...

    def save(self):
        """Save to .ragmd file."""
        ...
```

### 5.2 Backwards Compatibility

**Option 1: Separate class**
- `SingleFileRAGService` for `.ragmd` files
- `GenericRAGService` for FAISS + JSON (default)
- Users choose based on use case

**Option 2: Auto-detection**
```python
# Detect format from path extension
if ragmd_path.suffix == '.ragmd':
    backend = SingleFileRAGService(...)
else:
    backend = GenericRAGService(...)
```

---

## 7. Implementation

### 6.1 File I/O Operations

```python
class SingleFileRAGService(GenericRAGService):
    def load_from_ragmd(self, path: Path) -> None:
        """Load index, chunks, and documents from .ragmd file."""
        content = path.read_text(encoding='utf-8')

        # Parse frontmatter
        frontmatter, body = self._parse_frontmatter(content)
        self.model_name = frontmatter['model_name']
        self.embedding_dim = frontmatter['embedding_dim']

        # Extract sections
        sections = self._parse_markdown_sections(body)

        # Load chunks
        chunk_data = json.loads(sections['Chunks'])
        self.chunk_refs = [ChunkRef(**c) for c in chunk_data]

        # Load documents
        doc_data = json.loads(sections['Document Metadata'])
        self.documents = {
            doc_id: Document.from_dict(d)
            for doc_id, d in doc_data.items()
        }

        # Load FAISS index
        b64_index = sections['Vectors'].strip()
        self.index = self._deserialize_index(b64_index)

    def save_to_ragmd(self, path: Path) -> None:
        """Save index, chunks, and documents to .ragmd file."""
        # Build frontmatter
        frontmatter = {
            'format_version': '1.0',
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'vector_count': len(self.chunk_refs),
            'document_count': len(self.documents),
            'index_type': self.index.__class__.__name__,
            'updated_at': datetime.utcnow().isoformat() + 'Z'
        }

        # Serialize components
        chunks_json = json.dumps([
            {'faiss_id': i, 'document_id': r.document_id, 'chunk_key': r.chunk_key}
            for i, r in enumerate(self.chunk_refs)
        ], indent=2)

        docs_json = json.dumps(
            {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            indent=2
        )

        index_b64 = self._serialize_index(self.index)

        # Build markdown
        md_content = f"""---
{yaml.dump(frontmatter, default_flow_style=False)}---

## Documents

{self._build_document_table()}

## Chunks

```json
{chunks_json}
```

## Document Metadata

```json
{docs_json}
```

## Vectors

```base64
{index_b64}
```
"""
        path.write_text(md_content, encoding='utf-8')
```

### 6.2 Parsing Helpers

```python
def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter and body."""
    if not content.startswith('---\n'):
        raise ValueError("Missing frontmatter")

    parts = content.split('---\n', 2)
    frontmatter = yaml.safe_load(parts[1])
    body = parts[2]
    return frontmatter, body

def _parse_markdown_sections(self, body: str) -> dict[str, str]:
    """Extract content from ## headings."""
    sections = {}
    current_section = None
    current_content = []

    for line in body.split('\n'):
        if line.startswith('## '):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line[3:].strip()
            current_content = []
        else:
            current_content.append(line)

    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()

    return sections
```

---

## 8. Performance Characteristics

### 7.1 Time Complexity

| Operation | Multi-File | Single-File | Notes |
|-----------|------------|-------------|-------|
| **Load** | O(n) | O(n + b64) | Base64 decode adds ~30% overhead |
| **Search** | O(n·d) | O(n·d) | Identical (FAISS in-memory) |
| **Save** | O(n) | O(n + b64) | Base64 encode adds ~30% overhead |

Where:
- n = number of vectors
- d = embedding dimension
- b64 = base64 encoding/decoding time

### 7.2 Space Complexity

**Disk**:
- Multi-file: `FAISS_size + JSON_size`
- Single-file: `1.33 × FAISS_size + JSON_size` (base64 overhead)

**Memory** (during operation):
- Both approaches: Same (FAISS index + metadata in RAM)

### 7.3 Benchmarks (Estimated)

For 1000 documents × 10 chunks each × 768-dim embeddings:

| Format | File Size | Load Time | Save Time |
|--------|-----------|-----------|-----------|
| FAISS + JSON | ~30 MB | ~200 ms | ~150 ms |
| Single .ragmd | ~40 MB | ~260 ms | ~200 ms |

**Overhead**: ~30% space, ~30% I/O time

---

## 9. Limitations

### 8.1 Scalability

**Not recommended** for:
- >10,000 documents (file size >500 MB)
- High-frequency updates (slow save)
- Concurrent writes (no locking)

### 8.2 Performance

- Base64 encoding/decoding overhead (~30%)
- Full file rewrite on every save (no incremental updates)
- Git diff less effective on base64 blocks (binary data changes)

### 8.3 FAISS Index Types

Only works well with simple index types:
- ✓ `IndexFlatIP`, `IndexFlatL2` (full index serialization)
- ✓ `IndexIVFFlat` (small inverted indexes)
- ✗ Large quantized indexes (PQ, HNSW) → excessive file size

---

## 10. Migration Path

### 9.1 FAISS + JSON → Single File

```bash
# Convert existing backend
python -m chatboti.rag_cli convert \
  --from-faiss vectors-nomic.faiss \
  --from-json metadata-nomic.json \
  --to-ragmd embeddings-nomic.ragmd
```

### 9.2 Single File → FAISS + JSON

```bash
# Export to multi-file format
python -m chatboti.rag_cli convert \
  --from-ragmd embeddings-nomic.ragmd \
  --to-faiss vectors-nomic.faiss \
  --to-json metadata-nomic.json
```

### 9.3 Code Migration

```python
# Before (multi-file)
async with GenericRAGService("ollama", model="nomic-embed-text") as rag:
    results = await rag.search("query")

# After (single-file)
async with SingleFileRAGService(
    "ollama",
    model="nomic-embed-text",
    ragmd_path=Path("embeddings.ragmd")
) as rag:
    results = await rag.search("query")
```

---

## 11. Example File

### 10.1 Minimal Example

```markdown
---
format_version: "1.0"
model_name: "nomic-embed-text"
embedding_dim: 768
vector_count: 3
document_count: 1
index_type: "IndexFlatIP"
created_at: "2026-02-14T10:00:00Z"
updated_at: "2026-02-14T10:00:00Z"
---

## Documents

| Document ID | Source | Chunks | Title |
|-------------|--------|--------|-------|
| doc-001 | example.txt | 3 | Sample Document |

## Chunks

```json
[
  {"faiss_id": 0, "document_id": "doc-001", "chunk_key": "chunk-0"},
  {"faiss_id": 1, "document_id": "doc-001", "chunk_key": "chunk-1"},
  {"faiss_id": 2, "document_id": "doc-001", "chunk_key": "chunk-2"}
]
```

## Document Metadata

```json
{
  "doc-001": {
    "id": "doc-001",
    "source": "example.txt",
    "full_text": "This is a sample document with multiple chunks for testing.",
    "content": {},
    "chunks": {
      "chunk-0": {"i_start": 0, "i_end": 20, "faiss_id": 0},
      "chunk-1": {"i_start": 20, "i_end": 40, "faiss_id": 1},
      "chunk-2": {"i_start": 40, "i_end": 59, "faiss_id": 2}
    }
  }
}
```

## Vectors

```base64
aWRuZXhfZmxhdGlw... [base64-encoded FAISS index data] ...Qo=
```
```

---

## Conclusion

The single-file RAG backend format provides a **simple, portable, and version-controllable** alternative to multi-file storage for small-to-medium document collections. While it sacrifices some performance and scalability, it offers significant benefits for:

- **Prototyping**: Quick setup, easy inspection
- **Collaboration**: Single file sharing, git-friendly
- **Embedded systems**: No external database dependencies
- **Educational**: Transparent format for learning

**Recommendation**: Use for projects with <1000 documents where simplicity and portability outweigh performance requirements. Migrate to FAISS + SQLite for production scale.
