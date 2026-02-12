# Generic Document Storage Specification

## Executive Summary

This specification outlines the refactoring of the chatboti RAG system from a speaker-specific implementation to a generic document storage and retrieval system. The current system is tightly coupled to the speaker/agenda domain with hardcoded field names, single CSV source, and fixed embedding structure. This limits reusability for other document types like research papers, product catalogs, knowledge bases, or general text collections.

**Goal**: Create a flexible, domain-agnostic RAG architecture that maintains backwards compatibility with the existing speaker data while enabling new use cases.

**Recommendation**: Use FAISS as the primary vector storage format, with JSON for metadata initially (migrating to SQLite later for scale). Never use `List[float]` for embeddings - always use `ndarray[float32]`.

**Storage Evolution Strategy**: Start with JSON metadata (simple, readable, good for <1K documents), then migrate to SQLite when scaling beyond 1K documents (ACID transactions, indexed queries, better concurrency).

**Status**: Proposed Design
**Author**: Claude Sonnet 4.5
**Date**: 2026-02-12
**Last Updated**: 2026-02-12
**Related**: Consolidates content from STORAGE-EVOLUTION.md, faiss-multiple-documents-spec.md, metadata-storage-design.md, vector-storage-comparison.md

---

## Table of Contents

1. [Current System Analysis](#1-current-system-analysis)
2. [Problems with Current Approach](#2-problems-with-current-approach)
3. [Proposed Architecture](#3-proposed-architecture)
4. [Storage Evolution: JSON → SQLite](#4-storage-evolution-json--sqlite)
5. [Storage Format Comparison](#5-storage-format-comparison)
6. [Proposed Generic Document Model](#6-proposed-generic-document-model)
7. [API Design](#7-api-design)
8. [Backwards Compatibility](#8-backwards-compatibility)
9. [Integration Points](#9-integration-points)
10. [Example Use Cases](#10-example-use-cases)
11. [Implementation Approach](#11-implementation-approach)
12. [FAISS Integration Details](#12-faiss-integration-details)
13. [Migration Guide for Users](#13-migration-guide-for-users)
14. [Future Enhancements](#14-future-enhancements)
15. [Security Considerations](#15-security-considerations)
16. [Conclusion](#16-conclusion)

---

## 1. Current System Analysis

### 1.1 Architecture Overview

The current RAG system (`chatboti/rag.py`) implements a simple pipeline:

```
CSV Source → JSON with Embeddings → In-Memory Search → Speaker Results
```

**Key Components**:
- Single data source: `2025-09-02-speaker-bio.csv`
- Storage format: `embeddings-{model-name}.json` (JSON with inline vectors)
- Search: Linear cosine distance calculation across all embeddings
- Output: Speaker dictionaries with metadata

### 1.2 Speaker-Specific Hardcoding

The following elements are tightly coupled to the speaker domain:

#### **1.2.1 Hardcoded Field Names**

**In `_generate_speaker_embeddings()` (lines 122-126)**:
```python
for embed_key, field in [
    ("abstract_embedding", "final_abstract_max_150_words"),
    ("bio_embedding", "bio_max_120_words"),
]:
    speaker[embed_key] = await self.embed_client.embed(speaker[field])
```

**Problem**: Field names `final_abstract_max_150_words` and `bio_max_120_words` are speaker-specific. Other document types (research papers, product descriptions, etc.) have different field structures.

#### **1.2.2 Hardcoded CSV Source**

**In `_generate_speaker_embeddings()` (line 113)**:
```python
csv_text = self.read_text_file("2025-09-02-speaker-bio.csv")
```

**Problem**: Source file is hardcoded. Cannot load from:
- Different CSV files with different schemas
- JSON, XML, PDF, or plain text sources
- Multiple files simultaneously
- URLs or APIs

#### **1.2.3 Fixed Embedding Structure**

**In `get_speaker_distance()` (lines 161-163)**:
```python
if "abstract_embedding" in speaker and "bio_embedding" in speaker:
    d1 = self.cosine_distance(embedding, speaker["abstract_embedding"])
    d2 = self.cosine_distance(embedding, speaker["bio_embedding"])
    return (d1 + d2) / 2
```

**Problem**: Assumes exactly two embeddings per document (abstract + bio). Other use cases may need:
- Single embedding per document
- Multiple embeddings (title, summary, full text, metadata)
- Weighted embedding combinations
- Different similarity aggregation strategies

#### **1.2.4 Speaker-Specific Variable Names**

Throughout the codebase:
- `self.speakers_with_embeddings`
- `self.speakers`
- `get_best_speaker()`
- `get_speakers()`
- `_generate_speaker_embeddings()`
- `get_speaker_distance()`

**Problem**: All names assume "speaker" domain. Makes code harder to reuse and extend.

#### **1.2.5 MCP Tool Coupling**

**In `mcp_server.py` (lines 46-77)**:
```python
@mcp.tool()
async def get_best_speaker(query: str) -> Dict[str, Any]:
    """Find the most relevant speaker..."""
    best_speaker = await rag_service.get_best_speaker(query)
    return {
        "success": True,
        "speaker": best_speaker,
        "total_speakers_searched": len(rag_service.speakers_with_embeddings),
    }
```

**Problem**: MCP tools expose speaker-specific API. Need domain-agnostic tool layer.

### 1.3 Current Limitations Summary

| Limitation | Impact |
|------------|--------|
| Single CSV source | Cannot handle multiple file types or sources |
| Hardcoded field names | Cannot adapt to different document schemas |
| Fixed embedding pairs | Cannot support variable embedding strategies |
| Speaker-specific API | Cannot reuse for other domains without code duplication |
| No document type metadata | Cannot mix multiple document types in one index |
| No chunking support | Limited to pre-chunked CSV fields |
| No source tracking | Cannot identify document origin or update history |
| No multi-tenancy | Cannot isolate document collections per user/project |

---

## 2. Problems with Current Approach

### 2.1 Inefficient Embedding Storage: List[float] vs ndarray

**Current Problem**: Embeddings are stored as `List[float]` in JSON, which is extremely inefficient.

**Memory Waste Comparison** (1536-dim embedding):
```python
import sys
import numpy as np

# As Python list
python_list = [0.1] * 1536
size_list = sys.getsizeof(python_list)  # ~12 KB container
size_list += sum(sys.getsizeof(x) for x in python_list)  # ~36 KB floats
total_list = size_list  # ~48 KB

# As NumPy float32 array
numpy_array = np.array(python_list, dtype=np.float32)
total_numpy = numpy_array.nbytes  # 6 KB

print(f"Waste: {(total_list / total_numpy):.1f}x")  # 8x!
```

**Storage Size Comparison** (per 1536-dim embedding):
- JSON with `List[float]`: ~40 KB (text encoding)
- NumPy .npy binary: ~6 KB (4 bytes × 1536)
- FAISS index: ~6 KB (same as NumPy, but with indexing!)

**Performance Comparison** (cosine distance on 10K vectors):
- Python lists: ~5-10 seconds
- NumPy arrays: ~0.05 seconds (100x faster)
- FAISS index: ~0.001 seconds (5000x faster)

**Key Issues**:
- ❌ 8x memory waste with Python lists
- ❌ 6-8x larger storage size (JSON text encoding)
- ❌ Slow JSON parsing overhead
- ❌ No indexing for fast search
- ❌ No memory mapping (all loaded at once)

**Solution**: Always use `ndarray[float32]` and FAISS for storage.

### 2.2 Cannot Handle Different Document Types

**Example scenarios that fail**:

**Research Papers**:
```python
# Desired structure
{
    "id": "arxiv:2401.12345",
    "title": "Attention Is All You Need",
    "abstract": "...",
    "full_text": "...",
    "authors": ["Vaswani et al."],
    "embeddings": {
        "title_embedding": [...],
        "abstract_embedding": [...],
        "chunk_1_embedding": [...]
    }
}
```

**Product Catalog**:
```python
# Desired structure
{
    "id": "product-12345",
    "name": "Widget Pro",
    "description": "...",
    "specifications": "...",
    "category": "Electronics",
    "embeddings": {
        "description_embedding": [...],
        "specs_embedding": [...]
    }
}
```

**Knowledge Base Articles**:
```python
# Desired structure
{
    "id": "kb-67890",
    "title": "How to Reset Password",
    "content": "...",
    "tags": ["account", "security"],
    "embeddings": {
        "title_embedding": [...],
        "content_chunks": [
            {"chunk_id": 1, "text": "...", "embedding": [...]},
            {"chunk_id": 2, "text": "...", "embedding": [...]}
        ]
    }
}
```

**Current system**: Would require separate implementations for each type.

### 2.3 Hardcoded Field Names Limit Flexibility

**Example**: Loading a different speaker CSV with fields:
- `presenter_bio_summary` instead of `bio_max_120_words`
- `talk_description` instead of `final_abstract_max_150_words`

**Current system**: Requires code changes to `_generate_speaker_embeddings()`.

**Desired**: Configuration-driven field mapping.

### 2.4 Single CSV Source Limitation

**Cannot support**:
1. **Multi-file ingestion**: Load speakers from multiple CSVs
2. **Mixed formats**: Combine CSV speakers + PDF papers + JSON articles
3. **Incremental updates**: Add new documents without regenerating all embeddings
4. **Dynamic sources**: Fetch from APIs or databases
5. **Different schemas**: Documents with varying field structures

### 2.5 Difficult to Extend for New Use Cases

**Adding a new document type currently requires**:
1. Create new `RAGService` subclass or duplicate code
2. Hardcode new field names in embedding generation
3. Update distance calculation logic
4. Create new MCP tools
5. Duplicate testing and validation

**Result**: Code duplication, maintenance burden, fragile architecture.

---

## 3. Proposed Architecture

### 3.1 High-Level Overview

**Incremental Evolution Path**:

```
Phase 1 (Simple Start):        Phase 2 (Scaled Production):
├── vectors.faiss              ├── vectors.faiss
└── metadata.json              └── metadata.db (SQLite)
```

**Architecture Diagram**:
```
Multiple Sources → Document Ingestion → FAISS Index + Metadata → Hybrid Search
       ↓                  ↓                      ↓                    ↓
CSV/PDF/Web    Chunking + Embedding    vectors.faiss          Vector + Metadata
                                       metadata.json/db       retrieval
```

### 3.2 Why This Approach?

**Start Simple**: JSON metadata is human-readable, easy to debug, no SQL knowledge needed, works great for <1K documents

**Scale When Needed**: Migrate to SQLite when you hit:
- 1000+ documents
- Need for frequent updates
- Complex metadata queries
- Production deployment requirements

**FAISS for Vectors Always**: Use FAISS from the start for vector storage (10-100x faster search, memory-efficient, proven at scale)

## 4. Storage Evolution: JSON → SQLite

### 4.1 The Question

Can we use JSON for metadata initially, with a migration path to SQLite later?

### 4.2 Answer: YES! Incremental Approach is Better

This section documents the complete strategy for starting with simple JSON storage and evolving to SQLite for production scale.

### 4.3 Abstract Storage Interface

To enable seamless migration between JSON and SQLite, we define an abstract `MetadataStore` interface:

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class MetadataStore(ABC):
    """Abstract interface for metadata storage."""

    @abstractmethod
    def add_document(self, faiss_id: int, doc: Document) -> None:
        """Add document metadata."""
        pass

    @abstractmethod
    def get_document(self, faiss_id: int) -> Optional[Document]:
        """Get document by FAISS ID."""
        pass

    @abstractmethod
    def get_documents(self, faiss_ids: List[int]) -> List[Document]:
        """Get multiple documents by FAISS IDs."""
        pass

    @abstractmethod
    def update_document(self, faiss_id: int, doc: Document) -> None:
        """Update document metadata."""
        pass

    @abstractmethod
    def delete_document(self, faiss_id: int) -> None:
        """Delete document metadata."""
        pass

    @abstractmethod
    def search_metadata(self, query: dict) -> List[int]:
        """Search metadata, return FAISS IDs."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total documents."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close storage."""
        pass
```

### 4.4 JSON Implementation

#### When to Use JSON

- **Small datasets**: <1000 documents
- **Infrequent updates**: Batch loading, rare changes
- **Single-user**: No concurrent access
- **Development**: Prototyping and testing

#### Advantages

- ✅ **Simple**: No SQL knowledge needed
- ✅ **Readable**: Easy to inspect and debug
- ✅ **Portable**: Works everywhere
- ✅ **No dependencies**: No SQLite library needed
- ✅ **Fast start**: Quick prototyping
- ✅ **Version control friendly**: Can diff changes

#### Limitations

- ⚠️ **No transactions**: Risk of corruption on crash
- ⚠️ **Slow updates**: Requires full file rewrite
- ⚠️ **Memory intensive**: Loads entire file
- ⚠️ **No indexing**: Linear search for metadata queries
- ⚠️ **Poor concurrency**: File locking issues

#### Implementation

```python
import json
from pathlib import Path
from typing import List, Optional, Dict

class JSONMetadataStore(MetadataStore):
    """JSON file-based metadata storage."""

    def __init__(self, json_path: Path):
        self.json_path = json_path
        self.data: Dict[int, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load metadata from JSON file."""
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                raw_data = json.load(f)
                # Convert string keys to int (JSON keys are strings)
                self.data = {int(k): v for k, v in raw_data.items()}
        else:
            self.data = {}

    def _save(self) -> None:
        """Save metadata to JSON file."""
        # Write to temp file first, then rename (atomic on POSIX)
        temp_path = self.json_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        temp_path.replace(self.json_path)

    def add_document(self, faiss_id: int, doc: Document) -> None:
        """Add document metadata."""
        self.data[faiss_id] = doc.to_dict()
        self._save()

    def get_document(self, faiss_id: int) -> Optional[Document]:
        """Get document by FAISS ID."""
        if faiss_id in self.data:
            return Document.from_dict(self.data[faiss_id])
        return None

    def get_documents(self, faiss_ids: List[int]) -> List[Document]:
        """Get multiple documents by FAISS IDs."""
        docs = []
        for faiss_id in faiss_ids:
            doc = self.get_document(faiss_id)
            if doc:
                docs.append(doc)
        return docs

    def update_document(self, faiss_id: int, doc: Document) -> None:
        """Update document metadata."""
        if faiss_id in self.data:
            self.data[faiss_id] = doc.to_dict()
            self._save()

    def delete_document(self, faiss_id: int) -> None:
        """Delete document metadata."""
        if faiss_id in self.data:
            del self.data[faiss_id]
            self._save()

    def search_metadata(self, query: dict) -> List[int]:
        """Search metadata, return FAISS IDs."""
        results = []
        for faiss_id, doc_data in self.data.items():
            # Simple field matching (can be enhanced)
            matches = all(
                doc_data.get(key) == value
                for key, value in query.items()
            )
            if matches:
                results.append(faiss_id)
        return results

    def count(self) -> int:
        """Count total documents."""
        return len(self.data)

    def close(self) -> None:
        """Close storage (no-op for JSON)."""
        pass
```

#### JSON File Format

```json
{
  "0": {
    "doc_id": "doc-uuid-123",
    "content": "Full text content here...",
    "source": "papers.pdf",
    "doc_type": "research_paper",
    "metadata": {
      "title": "Attention Is All You Need",
      "authors": ["Vaswani", "Shazeer"],
      "year": 2017
    },
    "created_at": "2026-02-12T10:30:00Z"
  },
  "1": {
    "doc_id": "doc-uuid-456",
    "content": "Another document...",
    ...
  }
}
```

### 4.5 SQLite Implementation

#### When to Use SQLite

- **Large datasets**: >1000 documents
- **Frequent updates**: Real-time changes
- **Multi-user**: Concurrent access
- **Complex queries**: Filtering, sorting, joins
- **Production**: Deployed systems

#### Advantages

- ✅ **ACID transactions**: Safe concurrent access
- ✅ **Efficient updates**: Incremental changes
- ✅ **Indexing**: Fast metadata queries
- ✅ **Low memory**: Doesn't load everything
- ✅ **Reliable**: Battle-tested database
- ✅ **Standard**: SQL queries

#### Limitations

- ⚠️ **Complexity**: Requires SQL knowledge
- ⚠️ **Binary format**: Harder to inspect/diff
- ⚠️ **Schema management**: Migrations needed

#### Implementation

```python
import sqlite3
from pathlib import Path
from typing import List, Optional

class SQLiteMetadataStore(MetadataStore):
    """SQLite-based metadata storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                faiss_id INTEGER PRIMARY KEY,
                doc_id TEXT UNIQUE NOT NULL,
                content TEXT,
                source TEXT,
                doc_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS metadata (
                faiss_id INTEGER,
                key TEXT,
                value TEXT,
                FOREIGN KEY (faiss_id) REFERENCES documents(faiss_id),
                PRIMARY KEY (faiss_id, key)
            );

            CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id);
            CREATE INDEX IF NOT EXISTS idx_doc_type ON documents(doc_type);
            CREATE INDEX IF NOT EXISTS idx_metadata_key ON metadata(key);
        """)
        self.conn.commit()

    def add_document(self, faiss_id: int, doc: Document) -> None:
        """Add document metadata."""
        with self.conn:
            self.conn.execute(
                """INSERT INTO documents
                   (faiss_id, doc_id, content, source, doc_type)
                   VALUES (?, ?, ?, ?, ?)""",
                (faiss_id, doc.id, doc.content, doc.source, doc.doc_type)
            )
            # Store flexible metadata as key-value pairs
            for key, value in doc.metadata.items():
                self.conn.execute(
                    "INSERT INTO metadata (faiss_id, key, value) VALUES (?, ?, ?)",
                    (faiss_id, key, str(value))
                )

    def get_document(self, faiss_id: int) -> Optional[Document]:
        """Get document by FAISS ID."""
        row = self.conn.execute(
            "SELECT * FROM documents WHERE faiss_id = ?", (faiss_id,)
        ).fetchone()

        if not row:
            return None

        # Fetch metadata
        metadata_rows = self.conn.execute(
            "SELECT key, value FROM metadata WHERE faiss_id = ?", (faiss_id,)
        ).fetchall()
        metadata = {row['key']: row['value'] for row in metadata_rows}

        return Document(
            id=row['doc_id'],
            content=row['content'],
            source=row['source'],
            doc_type=row['doc_type'],
            metadata=metadata
        )

    def get_documents(self, faiss_ids: List[int]) -> List[Document]:
        """Get multiple documents by FAISS IDs."""
        placeholders = ','.join('?' * len(faiss_ids))
        rows = self.conn.execute(
            f"SELECT * FROM documents WHERE faiss_id IN ({placeholders})",
            faiss_ids
        ).fetchall()

        return [self._row_to_document(row) for row in rows]

    def search_metadata(self, query: dict) -> List[int]:
        """Search metadata, return FAISS IDs."""
        # Build dynamic query
        conditions = []
        params = []

        for key, value in query.items():
            conditions.append(
                "faiss_id IN (SELECT faiss_id FROM metadata WHERE key = ? AND value = ?)"
            )
            params.extend([key, str(value)])

        where_clause = " AND ".join(conditions)
        sql = f"SELECT faiss_id FROM documents WHERE {where_clause}"

        rows = self.conn.execute(sql, params).fetchall()
        return [row['faiss_id'] for row in rows]

    def count(self) -> int:
        """Count total documents."""
        return self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
```

### 4.6 GenericRAGService with Pluggable Storage

The RAG service automatically detects storage type by file extension:

```python
from pathlib import Path
from typing import Union

class GenericRAGService:
    """Generic RAG service with pluggable metadata storage."""

    def __init__(
        self,
        index_path: Path,
        metadata_store: Union[Path, MetadataStore],
        llm_service: str = "openai"
    ):
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        self.index_path = index_path

        # Initialize metadata storage
        if isinstance(metadata_store, MetadataStore):
            self.metadata = metadata_store
        elif isinstance(metadata_store, Path):
            # Auto-detect storage type by extension
            if metadata_store.suffix == '.json':
                self.metadata = JSONMetadataStore(metadata_store)
            elif metadata_store.suffix == '.db':
                self.metadata = SQLiteMetadataStore(metadata_store)
            else:
                raise ValueError(f"Unknown metadata format: {metadata_store.suffix}")

        # Initialize embedding client
        self.embed_client = get_llm_client(llm_service)

    async def add_document(self, doc: Document) -> int:
        """Add document to RAG system."""
        # Generate embedding
        embedding = await self.embed_client.embed(doc.content)
        embedding = np.array(embedding, dtype=np.float32).reshape(1, -1)

        # Add to FAISS
        faiss_id = self.index.ntotal
        self.index.add(embedding)

        # Save metadata
        self.metadata.add_document(faiss_id, doc)

        # Persist FAISS index
        faiss.write_index(self.index, str(self.index_path))

        return faiss_id

    async def search(self, query: str, k: int = 5) -> List[Document]:
        """Search documents by query."""
        # Get query embedding
        query_emb = await self.embed_client.embed(query)
        query_emb = np.array(query_emb, dtype=np.float32).reshape(1, -1)

        # FAISS search
        distances, faiss_ids = self.index.search(query_emb, k)

        # Fetch metadata
        return self.metadata.get_documents(faiss_ids[0].tolist())

    def close(self):
        """Close all resources."""
        self.metadata.close()
```

### 4.7 Automatic Migration: JSON → SQLite

The migration is simple and automatic:

```python
from pathlib import Path

def migrate_json_to_sqlite(json_path: Path, db_path: Path) -> None:
    """Migrate metadata from JSON to SQLite."""
    print(f"Migrating {json_path} → {db_path}")

    # Load JSON data
    json_store = JSONMetadataStore(json_path)

    # Create SQLite store
    sqlite_store = SQLiteMetadataStore(db_path)

    # Copy all documents
    count = 0
    for faiss_id in sorted(json_store.data.keys()):
        doc = json_store.get_document(faiss_id)
        if doc:
            sqlite_store.add_document(faiss_id, doc)
            count += 1

    print(f"✓ Migrated {count} documents")

    # Close stores
    json_store.close()
    sqlite_store.close()

    # Backup old JSON
    backup_path = json_path.with_suffix('.json.backup')
    json_path.rename(backup_path)
    print(f"✓ Backed up JSON to {backup_path}")
```

**Auto-Detection and Migration**:

```python
class GenericRAGService:
    def __init__(self, index_path: Path, metadata_path: Path, **kwargs):
        # Auto-migrate if JSON exists but SQLite doesn't
        json_path = metadata_path.with_suffix('.json')
        db_path = metadata_path.with_suffix('.db')

        if json_path.exists() and not db_path.exists():
            # Ask user or auto-migrate
            print("⚠️  JSON metadata detected. Migrating to SQLite for better performance...")
            migrate_json_to_sqlite(json_path, db_path)
            metadata_path = db_path

        # Continue with initialization...
```

### 4.8 Recommended Evolution Path

#### Phase 1: Start with JSON (Week 1-2)

```python
# Simple start
rag = GenericRAGService(
    index_path=Path("vectors.faiss"),
    metadata_store=Path("metadata.json")  # JSON storage
)
```

**Benefits**:
- Fast prototyping
- Easy debugging
- No SQL knowledge needed
- Good for <1000 documents

#### Phase 2: Migrate to SQLite (Week 3-4)

```python
# When scaling up or needing performance
from chatboti.storage import migrate_json_to_sqlite

migrate_json_to_sqlite(
    Path("metadata.json"),
    Path("metadata.db")
)

# Use SQLite
rag = GenericRAGService(
    index_path=Path("vectors.faiss"),
    metadata_store=Path("metadata.db")  # SQLite storage
)
```

**Triggers**:
- Dataset grows beyond 1000 documents
- Need frequent updates
- Want complex metadata queries
- Production deployment

### 4.9 JSON vs SQLite Comparison

| Feature | JSON | SQLite |
|---------|------|--------|
| **Simplicity** | ✅ Very simple | ⚠️ Moderate |
| **Readability** | ✅ Human readable | ❌ Binary |
| **Performance (<1K)** | ✅ Fast | ✅ Fast |
| **Performance (>1K)** | ⚠️ Slow | ✅ Fast |
| **Updates** | ❌ Full rewrite | ✅ Incremental |
| **Queries** | ❌ Linear scan | ✅ Indexed |
| **Transactions** | ❌ No ACID | ✅ ACID |
| **Memory** | ❌ Loads all | ✅ On-demand |
| **Concurrency** | ❌ File locks | ✅ Good |
| **Dependencies** | ✅ None | ✅ Built-in |

### 4.10 Conclusion on Storage Evolution

**Start with JSON, migrate to SQLite when needed.**

This incremental approach:
- ✅ Simplifies initial development
- ✅ Provides clear migration path
- ✅ Matches actual scaling needs
- ✅ Reduces upfront complexity
- ✅ Maintains flexibility

## 5. Storage Format Comparison

This section analyzes different storage formats for embedding vectors and recommends FAISS as the optimal solution.

### 3.1 Option 1: JSON with Lists (Current - DEPRECATED)

```json
{
  "id": "doc123",
  "content": "...",
  "embedding": [0.1, 0.2, 0.3, ...1536 floats...]
}
```

**Problems:**
- ❌ 6-8x larger than binary (text encoding)
- ❌ Slow to parse (JSON parsing overhead)
- ❌ Loads as Python lists (4-6x memory waste)
- ❌ No indexing for fast search
- ❌ No memory mapping (all loaded at once)

**Storage size**: ~40 KB per 1536-dim embedding

**Verdict**: Only keep for backwards compatibility.

### 3.2 Option 2: NumPy .npy Files

```python
# Save
embeddings = np.array(all_embeddings, dtype=np.float32)
np.save("embeddings.npy", embeddings)  # (N, D) array

# Load
embeddings = np.load("embeddings.npy", mmap_mode='r')  # Memory-mapped!
```

**Pros:**
- ✅ Efficient binary format (4 bytes per float)
- ✅ Memory mapping support (no load time)
- ✅ Native NumPy format
- ✅ Simple API

**Cons:**
- ❌ No indexing (still need linear search)
- ❌ Separate file from metadata
- ❌ No built-in compression
- ❌ Manual index management (row N = document ID?)

**Storage size**: 6 KB per 1536-dim embedding

**Verdict**: Good for simple cases, but lacks indexing.

### 3.3 Option 3: Parquet with Binary Columns

```python
import pyarrow as pa
import pyarrow.parquet as pq

# Save embeddings as binary blobs
df["embedding"] = df["embedding"].apply(
    lambda x: np.array(x, dtype=np.float32).tobytes()
)
df.to_parquet("documents.parquet", compression="zstd")

# Load
df = pd.read_parquet("documents.parquet")
df["embedding"] = df["embedding"].apply(
    lambda x: np.frombuffer(x, dtype=np.float32)
)
```

**Pros:**
- ✅ Vectors + metadata in one file
- ✅ Excellent compression (50-70%)
- ✅ Columnar format (efficient filtering)
- ✅ Industry standard

**Cons:**
- ❌ No indexing for vector search
- ❌ Must load entire column to search
- ❌ Complex API for updates
- ❌ Not optimized for vector operations

**Storage size**: 2-3 KB per embedding (with compression)

**Verdict**: Good for analytics, not optimized for vector search.

### 3.4 Option 4: SQLite with BLOB Columns

```sql
CREATE TABLE embeddings (
    doc_id TEXT PRIMARY KEY,
    embedding BLOB,  -- Raw float32 bytes
    dimensions INTEGER
);

-- Save
embedding_bytes = embedding.astype(np.float32).tobytes()
cursor.execute("INSERT INTO embeddings VALUES (?, ?, ?)",
               (doc_id, embedding_bytes, 1536))

-- Load
blob = cursor.execute("SELECT embedding FROM embeddings WHERE doc_id=?")
embedding = np.frombuffer(blob, dtype=np.float32)
```

**Pros:**
- ✅ Vectors + metadata in one database
- ✅ ACID transactions
- ✅ Efficient indexing for metadata
- ✅ Easy updates

**Cons:**
- ❌ No vector indexing (linear search)
- ❌ Must fetch all rows to search
- ❌ SQLite not optimized for large BLOBs
- ❌ Page size limitations

**Storage size**: 6-7 KB per embedding

**Verdict**: Good for metadata, not for vectors.

### 3.5 Option 5: FAISS as Primary Storage ✅ RECOMMENDED

```python
import faiss
import numpy as np

# Create index
dimension = 1536
index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

# Add vectors (automatically stores as float32)
embeddings = np.array(all_embeddings, dtype=np.float32)
index.add(embeddings)  # Returns IDs 0, 1, 2, ...

# Save to disk
faiss.write_index(index, "vectors.faiss")

# Load (memory-mapped, instant loading!)
index = faiss.read_index("vectors.faiss")

# Search (10-100x faster than linear)
query = np.array([...], dtype=np.float32)
distances, indices = index.search(query, k=5)  # Top 5 results
```

**Pros:**
- ✅ **Optimized for vector storage** (that's its purpose!)
- ✅ **Fast search** (10-100x faster with indices)
- ✅ **Memory mapping** (instant load, low memory)
- ✅ **Always uses float32** (no type confusion)
- ✅ **Compression options** (quantization)
- ✅ **Battle-tested** (Meta production)
- ✅ **Simple ID mapping** (0, 1, 2, ... → doc UUIDs in metadata DB)

**Cons:**
- ✅ Requires separate metadata storage (but that's good separation of concerns!)
- ✅ One more dependency (but only +15 MB)

**Storage size**: 6 KB per embedding (same as NumPy, but with indexing!)

**Verdict**: **Best choice for vector search at any scale.**

### 3.6 Package Size Analysis

A common concern is that FAISS might add significant overhead to the application. Surprisingly, **FAISS is actually smaller than NumPy itself!**

| Package | Download Size | Installed Size | Dependencies |
|---------|--------------|----------------|--------------|
| **NumPy** | 13-25 MB | **~20-40 MB** | None (standalone) |
| **faiss-cpu** | 8-15 MB | **~15-25 MB** | NumPy (required) |
| **Total with faiss-cpu** | - | **~35-65 MB** | Both packages |

**Platform-Specific Sizes:**

#### NumPy
- Linux x86_64: 35-45 MB installed
- macOS x86_64: 30-40 MB installed
- macOS arm64: 20-25 MB installed (current platform)
- Windows x86_64: 25-35 MB installed

#### faiss-cpu
- Linux x86_64: 20-25 MB installed
- macOS x86_64: 18-22 MB installed
- macOS arm64: 13-18 MB installed (current platform)
- Windows x86_64: 18-22 MB installed

**Why FAISS is Smaller Than Expected:**
1. **Highly optimized C++ code** - Compiled binary with minimal overhead
2. **Focused scope** - Only vector similarity search (NumPy does much more)
3. **Shared dependencies** - Leverages NumPy for array operations

**Docker Image Size Impact:**
```dockerfile
FROM python:3.13-slim
# Base image: ~120 MB

# Adding NumPy only
RUN pip install numpy==2.3.4
# Image size: ~140 MB

# Adding NumPy + faiss-cpu
RUN pip install numpy==2.3.4 faiss-cpu==1.9.0
# Image size: ~155 MB (+15 MB = +10% increase)

# For comparison: ChromaDB
RUN pip install chromadb
# Image size: ~420 MB (+300 MB = +250% increase!)
```

**Comparison with Other Vector DBs:**

| Package | Installed Size | Memory Overhead | Notes |
|---------|----------------|-----------------|-------|
| **NumPy** | 20-40 MB | Baseline | - |
| **faiss-cpu** | 15-25 MB | +0.3% | Minimal overhead |
| **chromadb** | 200-300 MB | +10-50% | Includes DuckDB, SQLite, HTTP server |
| **qdrant-client** | 30-50 MB | +5-10% | - |
| **weaviate-client** | 25-40 MB | +5-10% | - |
| **pinecone-client** | 15-25 MB | Cloud-based | Requires external service |

**Performance vs Size Trade-off:**

| Solution | Size | Search Speed (10K docs) | Complexity |
|----------|------|-------------------------|------------|
| **NumPy only** | 20 MB | 50-100ms (linear) | Simple |
| **NumPy + faiss-cpu** | 35 MB | **1-5ms** | Medium |
| **ChromaDB** | 300 MB | 5-10ms | High |

**Runtime Memory Overhead:**

FAISS adds less than 0.3% memory overhead for vector storage:

```python
import faiss

# 126 embeddings (63 speakers × 2 embeddings)
embeddings = np.random.rand(126, 1536).astype(np.float32)
# NumPy storage: 756 KB

index = faiss.IndexFlatIP(1536)
index.add(embeddings)
# FAISS overhead:
# - Index metadata: ~1-2 KB
# - Vector storage: 756 KB (same as NumPy)
# - Total: 757-758 KB (minimal overhead!)
```

**Verdict**: Adding faiss-cpu is a no-brainer. The size cost is minimal (+15 MB, +10%) compared to the performance gains (10-50x speedup).

### 3.7 Recommended Architecture

**Storage Layer (On-Disk):**
```
documents/
├── vectors.faiss          # FAISS index (all embeddings as float32)
└── metadata.db            # SQLite (document info, no embeddings)
```

**vectors.faiss:**
- FAISS index with N embeddings
- ID 0 → First document
- ID 1 → Second document
- Each ID maps to a row in metadata.db

**metadata.db (SQLite):**
```sql
CREATE TABLE documents (
    faiss_id INTEGER PRIMARY KEY,  -- Matches FAISS index position
    doc_id TEXT UNIQUE,             -- User-facing UUID
    content TEXT,
    source TEXT,
    created_at TIMESTAMP,
    -- NO embedding column!
);

CREATE TABLE chunks (
    faiss_id INTEGER PRIMARY KEY,  -- Matches FAISS index position
    chunk_id TEXT UNIQUE,
    doc_id TEXT,
    chunk_index INTEGER,
    text TEXT,
    start_pos INTEGER,
    end_pos INTEGER,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);
```

**Key Principles:**
1. **Never use `List[float]`** - Always `ndarray[float32]`
2. **Vectors in FAISS** - Optimized storage and search
3. **Metadata in SQLite** - Transactional, queryable
4. **Separation of concerns** - Each system does what it's best at

---

## 12. FAISS Integration Details

This section provides detailed information about integrating FAISS for vector similarity search, complementing the storage format comparison in Section 5.

### 12.1 Why FAISS for Chatboti?

From the analysis in earlier sections, FAISS becomes the recommended choice for this project because:

**Benefits over JSON storage**:
- **Performance**: 10-100x faster for large datasets (1-5ms vs 50-100ms for 10K docs)
- **Memory efficiency**: Memory-mapped indices (instant load, low memory)
- **Scalability**: Proven at billion-vector scale at Meta
- **Small overhead**: Only +15 MB dependency (+10% Docker image size)
- **Type safety**: Always uses float32 (no List[float] confusion)

**When to use FAISS**: Immediately for this project, even at current small scale
- Current: 63 speakers = 126 embeddings (acceptable with either approach)
- Future: Will support multiple document sources → 1000s-10000s of embeddings
- FAISS provides migration path from IndexFlatIP → IndexIVFFlat as dataset grows

### 12.2 Index Selection for Different Scales

#### For Current and Small Scale (<1000 documents, ~2000 embeddings)

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

### 12.3 Distance Metrics and Cosine Similarity

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

### 12.4 Vector Storage: FAISS Only vs Dual Storage

**Option A: Store in FAISS only** (Recommended)

```python
# Vectors only in FAISS index
index.add(vectors)
faiss.write_index(index, "embeddings.index")

# Metadata in JSON/SQLite (no vector column)
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

### 12.5 Index Persistence and Loading

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

### 12.6 ID Mapping: FAISS Indices to Document IDs

**Challenge**: FAISS uses integer indices (0, 1, 2, ...), but we need string chunk IDs (UUIDs).

**Solution 1: IndexIDMap wrapper**

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

**Solution 2: Maintain separate mapping** (Simpler, recommended)

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

### 12.7 Multiple Document Sources and FAISS

When supporting multiple document types (CSV, PDF, JSON, web):

**Single FAISS Index for All Documents**:

```python
# All embeddings in one index
index = faiss.IndexFlatIP(dimension)

# Add embeddings from different sources
index.add(speaker_embeddings)  # 0-125
index.add(pdf_embeddings)      # 126-500
index.add(web_embeddings)      # 501-1000

# Metadata store tracks document type
metadata[0] = {"doc_type": "speaker", "source": "speakers.csv", ...}
metadata[126] = {"doc_type": "pdf", "source": "paper.pdf", ...}
```

**Benefits**:
- Single index to manage
- Cross-domain search by default
- Simple architecture

**Filtering by document type**:

```python
# 1. Search FAISS (all documents)
distances, faiss_ids = index.search(query, k=20)

# 2. Filter by metadata
filtered = [
    id for id in faiss_ids[0]
    if metadata[id]["doc_type"] == "pdf"
][:5]
```

### 12.8 Document Chunking and FAISS

**Current System**: Each speaker = 2 chunks (abstract, bio) → 126 embeddings for 63 speakers

**With Multiple Documents**:
- PDF: 10 pages → 30 chunks (3 per page)
- Web page: 20 paragraphs → 20 chunks
- CSV row: 2 text fields → 2 chunks

**FAISS Mapping**:

```python
# Chunk to FAISS ID mapping
chunks = [
    {"id": "chunk-1", "text": "Bio text", "faiss_id": 0},
    {"id": "chunk-2", "text": "Abstract text", "faiss_id": 1},
    {"id": "chunk-3", "text": "PDF paragraph 1", "faiss_id": 2},
    ...
]

# Add all chunk embeddings to FAISS
embeddings = np.array([chunk["embedding"] for chunk in chunks])
index.add(embeddings)
```

### 12.9 Performance Benchmarks

**Expected Performance with FAISS**:

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

### 12.10 Memory Optimization

**Technique 1: Memory-Mapped Indices**

```python
# FAISS automatically memory-maps large indices
index = faiss.read_index("embeddings.index", faiss.IO_FLAG_MMAP)

# Only loads accessed vectors into RAM
# Reduces startup time for large indices
```

**Technique 2: Product Quantization** (lossy compression, future)

```python
# Reduce memory by 8-32x with minimal accuracy loss
pq = faiss.IndexPQ(dimension, M=8, nbits=8)
pq.train(training_vectors)
pq.add(vectors)

# Memory: dimension / M * nbits / 8 bytes per vector
# Example: 1536 / 8 * 1 = 192 bytes (vs 6144 bytes uncompressed)
```

**Recommendation**: Start with memory-mapped IndexFlatIP, add compression only if needed.

### 12.11 Package Size Impact

From the analysis in Section 5.6:

| Package | Installed Size | Notes |
|---------|----------------|-------|
| **NumPy** | 20-40 MB | Already required |
| **faiss-cpu** | 15-25 MB | **New dependency** |
| **Total** | 35-65 MB | Only +15 MB (+10%) |

**Docker Image Impact**:
- Base Python 3.13-slim: ~120 MB
- +NumPy: ~140 MB
- +NumPy + faiss-cpu: ~155 MB (+15 MB = +10% increase)

**Comparison with alternatives**:
- ChromaDB: +300 MB (+250% increase)
- Pinecone/Weaviate: Cloud-based (external service)

**Verdict**: faiss-cpu is lightweight and worth the small overhead.

### 12.12 Implementation Roadmap for FAISS

**Phase 1: FAISS Integration** (Week 1-2)

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

---

## 6. Proposed Generic Document Model

### 4.1 Core Abstractions

#### **4.1.1 Document Interface**

A document is the fundamental unit of storage with flexible structure:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class Document:
    """Generic document representation independent of domain."""

    id: str                                    # Unique identifier
    content: Dict[str, Any]                    # Original document fields
    metadata: Dict[str, Any] = field(default_factory=dict)  # Source, type, timestamps
    embeddings: Dict[str, List[float]] = field(default_factory=dict)  # Named embeddings
    chunks: Optional[List['DocumentChunk']] = None  # Chunked content

    def get_field(self, field_name: str, default: Any = None) -> Any:
        """Get field from content with fallback to default."""
        return self.content.get(field_name, default)

    def add_embedding(self, name: str, vector: List[float]) -> None:
        """Add a named embedding to the document."""
        self.embeddings[name] = vector

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "embeddings": self.embeddings,
            "chunks": [c.to_dict() for c in self.chunks] if self.chunks else []
        }
```

**Key features**:
- **Domain-agnostic**: No speaker-specific fields
- **Flexible content**: Arbitrary field structure in `content` dict
- **Named embeddings**: Multiple embeddings with descriptive names
- **Metadata separation**: Source info separate from content
- **Chunking support**: Optional chunked representation

#### **4.1.2 Document Chunk**

Represents a fragment of a larger document:

```python
@dataclass
class DocumentChunk:
    """Represents a chunk of a document for embedding."""

    chunk_id: str              # Unique chunk identifier
    document_id: str           # Parent document reference
    text: str                  # Chunk content
    chunk_index: int           # Position in document (0-based)
    chunk_type: str            # e.g., "paragraph", "section", "abstract"
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
            "embedding": self.embedding,
            "metadata": self.metadata
        }
```

**Metadata examples**:
- `start_char`, `end_char`: Character boundaries
- `token_count`: Approximate tokens
- `overlap_prev`, `overlap_next`: Chunk overlap sizes
- `section_title`: Semantic grouping

#### **4.1.3 Embedding Configuration**

Defines which fields to embed and how:

```python
@dataclass
class EmbeddingConfig:
    """Configuration for generating embeddings from document fields."""

    source_field: str          # Field name in document.content
    embedding_name: str        # Name to store embedding under
    weight: float = 1.0        # Weight for similarity aggregation
    chunk_strategy: Optional[str] = None  # "none", "fixed", "semantic"
    chunk_size: int = 512      # Tokens per chunk (if chunking)
    chunk_overlap: int = 50    # Token overlap between chunks

    def should_chunk(self) -> bool:
        return self.chunk_strategy is not None and self.chunk_strategy != "none"
```

**Example configurations**:

```python
# Speaker configuration (backwards compatible)
speaker_config = [
    EmbeddingConfig(
        source_field="final_abstract_max_150_words",
        embedding_name="abstract_embedding",
        weight=1.0
    ),
    EmbeddingConfig(
        source_field="bio_max_120_words",
        embedding_name="bio_embedding",
        weight=1.0
    )
]

# Research paper configuration
paper_config = [
    EmbeddingConfig(
        source_field="title",
        embedding_name="title_embedding",
        weight=2.0  # Higher weight for title matches
    ),
    EmbeddingConfig(
        source_field="abstract",
        embedding_name="abstract_embedding",
        weight=1.5
    ),
    EmbeddingConfig(
        source_field="full_text",
        embedding_name="content_chunks",
        weight=1.0,
        chunk_strategy="fixed",
        chunk_size=512,
        chunk_overlap=50
    )
]
```

### 4.2 Document Type Registry

Allows registering multiple document types in a single system:

```python
@dataclass
class DocumentType:
    """Metadata about a document type and how to process it."""

    type_name: str                     # e.g., "speaker", "paper", "product"
    embedding_configs: List[EmbeddingConfig]
    id_field: str = "id"               # Which field contains document ID
    display_fields: List[str] = field(default_factory=list)  # Fields for UI

class DocumentTypeRegistry:
    """Registry for managing multiple document types."""

    def __init__(self):
        self._types: Dict[str, DocumentType] = {}

    def register(self, doc_type: DocumentType) -> None:
        """Register a document type configuration."""
        self._types[doc_type.type_name] = doc_type

    def get(self, type_name: str) -> Optional[DocumentType]:
        """Retrieve document type configuration."""
        return self._types.get(type_name)

    def list_types(self) -> List[str]:
        """List all registered document types."""
        return list(self._types.keys())
```

**Usage example**:

```python
registry = DocumentTypeRegistry()

# Register speaker type
registry.register(DocumentType(
    type_name="speaker",
    embedding_configs=speaker_config,
    id_field="name",  # Use name as ID
    display_fields=["name", "bio_max_120_words", "final_abstract_max_150_words"]
))

# Register paper type
registry.register(DocumentType(
    type_name="paper",
    embedding_configs=paper_config,
    id_field="id",
    display_fields=["title", "authors", "abstract"]
))
```

---

## 5. API Design

### 5.1 Generic Document Ingestion Interface

Replace hardcoded CSV loading with pluggable loaders:

```python
from abc import ABC, abstractmethod

class DocumentLoader(ABC):
    """Abstract base for document loading strategies."""

    @abstractmethod
    async def load(self, source: str, doc_type: DocumentType) -> List[Document]:
        """Load documents from source and convert to Document objects."""
        pass

    @abstractmethod
    def supports(self, source: str) -> bool:
        """Check if this loader can handle the given source."""
        pass

class CSVDocumentLoader(DocumentLoader):
    """Loads documents from CSV files."""

    async def load(self, source: str, doc_type: DocumentType) -> List[Document]:
        csv_text = Path(source).read_text()
        reader = csv.DictReader(StringIO(csv_text))

        documents = []
        for row in reader:
            # Convert CSV row to Document
            content = {py_.snake_case(k): v for k, v in row.items()}
            doc_id = content.get(doc_type.id_field, str(uuid4()))

            doc = Document(
                id=doc_id,
                content=content,
                metadata={
                    "source": source,
                    "type": doc_type.type_name,
                    "imported_at": datetime.utcnow().isoformat()
                }
            )
            documents.append(doc)

        return documents

    def supports(self, source: str) -> bool:
        return source.endswith('.csv')

class JSONDocumentLoader(DocumentLoader):
    """Loads documents from JSON files."""

    async def load(self, source: str, doc_type: DocumentType) -> List[Document]:
        data = json.loads(Path(source).read_text())

        # Handle both single document and array
        if isinstance(data, dict):
            data = [data]

        documents = []
        for item in data:
            doc_id = item.get(doc_type.id_field, str(uuid4()))
            doc = Document(
                id=doc_id,
                content=item,
                metadata={
                    "source": source,
                    "type": doc_type.type_name,
                    "imported_at": datetime.utcnow().isoformat()
                }
            )
            documents.append(doc)

        return documents

    def supports(self, source: str) -> bool:
        return source.endswith('.json')
```

### 5.2 Generic RAG Service

Refactor `RAGService` to be domain-agnostic:

```python
import faiss
import numpy as np
from numpy.typing import NDArray

class GenericRAGService:
    """Domain-agnostic RAG service with FAISS vector storage."""

    def __init__(
        self,
        llm_service: Optional[str] = None,
        doc_type_registry: Optional[DocumentTypeRegistry] = None,
        data_dir: Optional[Path] = None
    ):
        self.llm_service = llm_service or os.getenv("LLM_SERVICE", "openai").lower()
        self.embed_client = self._initialize_embed_client()
        self.doc_type_registry = doc_type_registry or DocumentTypeRegistry()
        self.data_dir = data_dir or Path("chatboti/data")

        # FAISS index for vector storage
        self.dimension = 1536  # Configure based on embedding model
        self.faiss_index: Optional[faiss.Index] = None

        # SQLite for metadata
        self.db_path = self.data_dir / "metadata.db"
        self.db: Optional[sqlite3.Connection] = None

        # In-memory document cache (optional for small datasets)
        self.documents: Dict[str, Document] = {}  # doc_id -> Document

        # Document loaders
        self.document_loaders: List[DocumentLoader] = [
            CSVDocumentLoader(),
            JSONDocumentLoader(),
            # Add more loaders as needed
        ]

    async def load_documents(
        self,
        source: str,
        doc_type_name: str,
        force_regenerate: bool = False
    ) -> None:
        """
        Load documents from source and generate embeddings.

        Args:
            source: Path to data file (CSV, JSON, etc.)
            doc_type_name: Registered document type name
            force_regenerate: Regenerate even if embeddings exist
        """
        doc_type = self.doc_type_registry.get(doc_type_name)
        if not doc_type:
            raise ValueError(f"Unknown document type: {doc_type_name}")

        # Find appropriate loader
        loader = self._find_loader(source)
        if not loader:
            raise ValueError(f"No loader found for source: {source}")

        # Load documents
        logger.info(f"Loading {doc_type_name} documents from {source}")
        documents = await loader.load(source, doc_type)

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents")
        for doc in documents:
            await self._generate_embeddings(doc, doc_type)
            self.documents[doc.id] = doc

        logger.info(f"Loaded {len(documents)} documents of type {doc_type_name}")

    async def _generate_embeddings(
        self,
        document: Document,
        doc_type: DocumentType
    ) -> None:
        """Generate embeddings based on document type configuration."""
        for config in doc_type.embedding_configs:
            field_value = document.get_field(config.source_field)
            if not field_value:
                logger.warning(
                    f"Field '{config.source_field}' not found in document {document.id}"
                )
                continue

            if config.should_chunk():
                # Handle chunking for long text
                chunks = self._chunk_text(field_value, config)
                for chunk in chunks:
                    chunk.embedding = await self.embed_client.embed(chunk.text)
                    if not document.chunks:
                        document.chunks = []
                    document.chunks.append(chunk)
            else:
                # Single embedding for field
                embedding = await self.embed_client.embed(field_value)
                document.add_embedding(config.embedding_name, embedding)

    async def search(
        self,
        query: str,
        doc_type_name: Optional[str] = None,
        top_k: int = 1
    ) -> List[Document]:
        """
        Search for best matching documents.

        Args:
            query: Search query
            doc_type_name: Filter by document type (None = search all)
            top_k: Number of results to return

        Returns:
            List of best matching documents
        """
        query_embedding = await self.embed_client.embed(query)

        # Filter documents by type if specified
        docs = self.documents.values()
        if doc_type_name:
            docs = [d for d in docs if d.metadata.get("type") == doc_type_name]

        # Calculate distances
        distances = []
        for doc in docs:
            doc_type = self.doc_type_registry.get(doc.metadata.get("type"))
            if not doc_type:
                continue

            distance = self._calculate_document_distance(
                query_embedding, doc, doc_type
            )
            distances.append((distance, doc))

        # Sort and return top-k
        distances.sort(key=lambda x: x[0])
        return [doc for _, doc in distances[:top_k]]

    def _calculate_document_distance(
        self,
        query_embedding: List[float],
        document: Document,
        doc_type: DocumentType
    ) -> float:
        """Calculate weighted distance between query and document."""
        total_distance = 0.0
        total_weight = 0.0

        for config in doc_type.embedding_configs:
            if config.embedding_name in document.embeddings:
                # Use stored embedding
                doc_embedding = document.embeddings[config.embedding_name]
                distance = self.cosine_distance(query_embedding, doc_embedding)
                total_distance += distance * config.weight
                total_weight += config.weight
            elif document.chunks:
                # Search through chunks
                chunk_distances = [
                    self.cosine_distance(query_embedding, chunk.embedding)
                    for chunk in document.chunks
                    if chunk.embedding and chunk.chunk_type == config.embedding_name
                ]
                if chunk_distances:
                    # Use best chunk distance
                    distance = min(chunk_distances)
                    total_distance += distance * config.weight
                    total_weight += config.weight

        return total_distance / total_weight if total_weight > 0 else float('inf')

    def _find_loader(self, source: str) -> Optional[DocumentLoader]:
        """Find appropriate loader for source."""
        for loader in self.document_loaders:
            if loader.supports(source):
                return loader
        return None

    @staticmethod
    def cosine_distance(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine distance (1 - cosine similarity)."""
        a = np.asarray(vec1, dtype=np.float64)
        b = np.asarray(vec2, dtype=np.float64)

        if a.size != b.size:
            raise ValueError(f"Vector size mismatch: {a.size} vs {b.size}")

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 1.0  # Maximum distance

        return 1.0 - (dot_product / (norm_a * norm_b))
```

### 5.3 Configuration File Format

Allow users to define document types via configuration:

**`rag_config.yaml`**:
```yaml
document_types:
  speaker:
    id_field: name
    display_fields:
      - name
      - bio_max_120_words
      - final_abstract_max_150_words
    embeddings:
      - source_field: final_abstract_max_150_words
        embedding_name: abstract_embedding
        weight: 1.0
      - source_field: bio_max_120_words
        embedding_name: bio_embedding
        weight: 1.0

  research_paper:
    id_field: id
    display_fields:
      - title
      - authors
      - abstract
    embeddings:
      - source_field: title
        embedding_name: title_embedding
        weight: 2.0
      - source_field: abstract
        embedding_name: abstract_embedding
        weight: 1.5
      - source_field: full_text
        embedding_name: content_chunks
        weight: 1.0
        chunk_strategy: fixed
        chunk_size: 512
        chunk_overlap: 50

  product:
    id_field: product_id
    display_fields:
      - name
      - description
      - category
    embeddings:
      - source_field: description
        embedding_name: description_embedding
        weight: 1.5
      - source_field: specifications
        embedding_name: specs_embedding
        weight: 1.0

data_sources:
  speakers:
    type: speaker
    source: data/2025-09-02-speaker-bio.csv
    loader: csv

  papers:
    type: research_paper
    source: data/papers.json
    loader: json
```

**Configuration loader**:
```python
import yaml

def load_rag_config(config_path: str) -> DocumentTypeRegistry:
    """Load document type configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    registry = DocumentTypeRegistry()

    for type_name, type_config in config['document_types'].items():
        embedding_configs = []
        for emb_conf in type_config['embeddings']:
            embedding_configs.append(EmbeddingConfig(
                source_field=emb_conf['source_field'],
                embedding_name=emb_conf['embedding_name'],
                weight=emb_conf.get('weight', 1.0),
                chunk_strategy=emb_conf.get('chunk_strategy'),
                chunk_size=emb_conf.get('chunk_size', 512),
                chunk_overlap=emb_conf.get('chunk_overlap', 50)
            ))

        registry.register(DocumentType(
            type_name=type_name,
            embedding_configs=embedding_configs,
            id_field=type_config['id_field'],
            display_fields=type_config.get('display_fields', [])
        ))

    return registry
```

---

## 6. Backwards Compatibility

### 6.1 Migration Path for Existing Speaker Data

**Phase 1: Parallel Implementation**

Add generic system alongside existing code:

```python
# In rag.py
class RAGService:
    """Legacy speaker-specific service (DEPRECATED)."""
    # Keep existing implementation unchanged
    pass

class GenericRAGService:
    """New generic document service."""
    # New implementation
    pass

# Compatibility wrapper
class SpeakerRAGService(GenericRAGService):
    """Backwards-compatible wrapper for speaker data."""

    def __init__(self, llm_service: Optional[str] = None):
        registry = DocumentTypeRegistry()
        registry.register(DocumentType(
            type_name="speaker",
            embedding_configs=[
                EmbeddingConfig("final_abstract_max_150_words", "abstract_embedding"),
                EmbeddingConfig("bio_max_120_words", "bio_embedding")
            ],
            id_field="name",
            display_fields=["name", "bio_max_120_words", "final_abstract_max_150_words"]
        ))
        super().__init__(llm_service, registry)

    async def get_best_speaker(self, query: str) -> Optional[dict]:
        """Legacy API: Get best speaker."""
        results = await self.search(query, doc_type_name="speaker", top_k=1)
        if results:
            # Return in legacy format
            doc = results[0]
            result = doc.content.copy()
            result.pop("abstract_embedding", None)
            result.pop("bio_embedding", None)
            return result
        return None

    async def get_speakers(self) -> List[dict]:
        """Legacy API: Get all speakers."""
        speaker_docs = [
            d for d in self.documents.values()
            if d.metadata.get("type") == "speaker"
        ]
        return [self._strip_embeddings(d.content) for d in speaker_docs]

    @staticmethod
    def _strip_embeddings(content: dict) -> dict:
        return {k: v for k, v in content.items() if "embedding" not in k}
```

**Phase 2: Update MCP Server**

Modify `mcp_server.py` to use compatibility wrapper:

```python
# Old code
# from chatboti.rag import RAGService
# rag_service = RAGService(llm_service=embed_service)

# New code
from chatboti.rag import SpeakerRAGService
rag_service = SpeakerRAGService(llm_service=embed_service)

# MCP tools remain unchanged - they continue to work
@mcp.tool()
async def get_best_speaker(query: str) -> Dict[str, Any]:
    best_speaker = await rag_service.get_best_speaker(query)
    # ... rest unchanged
```

**Phase 3: Gradual Adoption**

New features use `GenericRAGService` directly:

```python
# New generic tools
@mcp.tool()
async def search_documents(
    query: str,
    doc_type: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """Generic document search across all types."""
    results = await generic_rag_service.search(query, doc_type, top_k)
    return {
        "success": True,
        "results": [doc.to_dict() for doc in results],
        "query": query
    }
```

### 6.2 Data Migration Strategy

**Step 1: Convert existing JSON embeddings to generic format**

```python
async def migrate_speaker_embeddings():
    """Convert legacy speaker embeddings to generic document format."""
    # Load legacy format
    legacy_data = json.loads(Path("embeddings-text-embedding-3-small.json").read_text())

    # Convert to generic documents
    documents = []
    for speaker_data in legacy_data:
        doc = Document(
            id=speaker_data["name"],
            content={k: v for k, v in speaker_data.items() if "embedding" not in k},
            metadata={
                "type": "speaker",
                "source": "2025-09-02-speaker-bio.csv",
                "migrated_at": datetime.utcnow().isoformat()
            },
            embeddings={
                "abstract_embedding": speaker_data["abstract_embedding"],
                "bio_embedding": speaker_data["bio_embedding"]
            }
        )
        documents.append(doc)

    # Save in generic format
    generic_data = [doc.to_dict() for doc in documents]
    Path("documents-speaker-text-embedding-3-small.json").write_text(
        json.dumps(generic_data, indent=2)
    )

    logger.info(f"Migrated {len(documents)} speaker documents")
```

**Step 2: Support loading both formats during transition**

```python
async def connect(self):
    """Load embeddings from either legacy or generic format."""
    generic_file = f"documents-speaker-{self.embed_model}.json"
    legacy_file = f"embeddings-{self.embed_model}.json"

    if Path(generic_file).exists():
        # Load generic format
        logger.info(f"Loading from generic format: {generic_file}")
        data = json.loads(Path(generic_file).read_text())
        for doc_data in data:
            doc = Document(**doc_data)
            self.documents[doc.id] = doc

    elif Path(legacy_file).exists():
        # Auto-migrate from legacy format
        logger.info(f"Auto-migrating from legacy format: {legacy_file}")
        await self.migrate_speaker_embeddings()
        await self.connect()  # Recursive call to load migrated data

    else:
        raise FileNotFoundError("No embeddings found")
```

### 6.3 Adapter Pattern for Legacy Code

Create adapter to make old code work with new system:

```python
class LegacySpeakerAdapter:
    """Adapter to make GenericRAGService look like old RAGService."""

    def __init__(self, generic_service: GenericRAGService):
        self._service = generic_service

    @property
    def speakers_with_embeddings(self) -> List[dict]:
        """Emulate old speakers_with_embeddings attribute."""
        speaker_docs = [
            d for d in self._service.documents.values()
            if d.metadata.get("type") == "speaker"
        ]
        # Convert to legacy format
        return [
            {
                **doc.content,
                **{k: v for k, v in doc.embeddings.items()}
            }
            for doc in speaker_docs
        ]

    @property
    def speakers(self) -> List[dict]:
        """Emulate old speakers attribute."""
        return [
            {k: v for k, v in s.items() if "embedding" not in k}
            for s in self.speakers_with_embeddings
        ]

    async def get_best_speaker(self, query: str) -> Optional[dict]:
        results = await self._service.search(query, "speaker", top_k=1)
        if results:
            return {k: v for k, v in results[0].content.items() if "embedding" not in k}
        return None

    async def get_speakers(self) -> List[dict]:
        return self.speakers
```

---

## 7. Integration Points

### 7.1 Metadata Storage Design Integration

This specification complements `metadata-storage-design.md`:

**Alignment**:
- Generic `Document` model maps to `documents` table
- `DocumentChunk` maps to `chunks` table
- Named embeddings map to `embeddings` table
- Metadata tracking aligns with SQLite schema

**Coordination**:
```python
# GenericRAGService can use SQLite backend
class SQLiteDocumentStore:
    """Store documents in SQLite instead of JSON."""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self._init_schema()

    def save_document(self, doc: Document) -> None:
        """Save document to SQLite using schema from metadata-storage-design."""
        # Insert into documents table
        self.db.execute("""
            INSERT INTO documents (id, source_path, source_type, metadata, raw_text)
            VALUES (?, ?, ?, ?, ?)
        """, (
            doc.id,
            doc.metadata.get("source"),
            doc.metadata.get("type"),
            json.dumps(doc.metadata),
            json.dumps(doc.content)
        ))

        # Insert chunks if present
        if doc.chunks:
            for chunk in doc.chunks:
                self.db.execute("""
                    INSERT INTO chunks (id, document_id, chunk_index, text, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    chunk.chunk_id,
                    doc.id,
                    chunk.chunk_index,
                    chunk.text,
                    json.dumps(chunk.metadata)
                ))

                # Insert embedding
                if chunk.embedding:
                    self.db.execute("""
                        INSERT INTO embeddings (id, chunk_id, model_name, vector)
                        VALUES (?, ?, ?, ?)
                    """, (
                        f"{chunk.chunk_id}-emb",
                        chunk.chunk_id,
                        self.embed_model,
                        np.array(chunk.embedding).tobytes()
                    ))

        self.db.commit()
```

### 7.2 Vector Storage Integration

This specification is storage-agnostic and works with both approaches from `vector-storage-comparison.md`:

**Parquet Backend**:
```python
class ParquetDocumentStore:
    """Store documents in Parquet format."""

    def save_documents(self, documents: List[Document], path: str) -> None:
        import polars as pl

        # Flatten documents to DataFrame
        data = []
        for doc in documents:
            row = {
                "id": doc.id,
                "type": doc.metadata.get("type"),
                **doc.content,
                **{f"emb_{k}": v for k, v in doc.embeddings.items()}
            }
            data.append(row)

        df = pl.DataFrame(data)
        df.write_parquet(path)

    def load_documents(self, path: str) -> List[Document]:
        import polars as pl

        df = pl.read_parquet(path)
        documents = []

        for row in df.iter_rows(named=True):
            # Separate embeddings from content
            embeddings = {
                k.replace("emb_", ""): v
                for k, v in row.items()
                if k.startswith("emb_")
            }
            content = {
                k: v for k, v in row.items()
                if not k.startswith("emb_") and k not in ["id", "type"]
            }

            doc = Document(
                id=row["id"],
                content=content,
                metadata={"type": row["type"]},
                embeddings=embeddings
            )
            documents.append(doc)

        return documents
```

**FAISS Backend**:
```python
import faiss

class FAISSDocumentStore:
    """Store embeddings in FAISS index, metadata in JSON/SQLite."""

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.id_to_doc: Dict[int, Document] = {}
        self.doc_id_to_idx: Dict[str, int] = {}

    def add_document(self, doc: Document) -> None:
        """Add document embeddings to FAISS index."""
        for emb_name, embedding in doc.embeddings.items():
            idx = len(self.id_to_doc)
            self.index.add(np.array([embedding], dtype=np.float32))
            self.id_to_doc[idx] = doc
            self.doc_id_to_idx[doc.id] = idx

    def search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Search for similar documents using FAISS."""
        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vec, k)

        return [self.id_to_doc[idx] for idx in indices[0] if idx in self.id_to_doc]
```

### 7.3 Plugin Architecture for Document Loaders

Allow third-party loaders via plugin system:

```python
class DocumentLoaderPlugin:
    """Base class for loader plugins."""

    @staticmethod
    def name() -> str:
        """Plugin identifier."""
        raise NotImplementedError

    @staticmethod
    def loader_class() -> Type[DocumentLoader]:
        """Return loader class."""
        raise NotImplementedError

# Example plugins
class PDFLoaderPlugin(DocumentLoaderPlugin):
    @staticmethod
    def name() -> str:
        return "pdf"

    @staticmethod
    def loader_class() -> Type[DocumentLoader]:
        from chatboti.loaders.pdf import PDFDocumentLoader
        return PDFDocumentLoader

class WebLoaderPlugin(DocumentLoaderPlugin):
    @staticmethod
    def name() -> str:
        return "web"

    @staticmethod
    def loader_class() -> Type[DocumentLoader]:
        from chatboti.loaders.web import WebDocumentLoader
        return WebDocumentLoader

# Plugin registry
class LoaderPluginRegistry:
    def __init__(self):
        self._plugins: Dict[str, Type[DocumentLoader]] = {}

    def register_plugin(self, plugin: DocumentLoaderPlugin) -> None:
        self._plugins[plugin.name()] = plugin.loader_class()

    def get_loader(self, name: str) -> Optional[Type[DocumentLoader]]:
        return self._plugins.get(name)

    def auto_discover_plugins(self) -> None:
        """Discover plugins in chatboti.loaders module."""
        # Use entry points or module scanning
        pass
```

---

## 8. Example Use Cases

### 8.1 Speaker Data (Current Use Case)

**Configuration**:
```python
speaker_type = DocumentType(
    type_name="speaker",
    embedding_configs=[
        EmbeddingConfig("final_abstract_max_150_words", "abstract_embedding", weight=1.0),
        EmbeddingConfig("bio_max_120_words", "bio_embedding", weight=1.0)
    ],
    id_field="name",
    display_fields=["name", "bio_max_120_words", "final_abstract_max_150_words"]
)
```

**Usage**:
```python
rag = GenericRAGService()
rag.doc_type_registry.register(speaker_type)
await rag.load_documents("data/2025-09-02-speaker-bio.csv", "speaker")

results = await rag.search("machine learning expert", doc_type_name="speaker")
best_speaker = results[0]
print(f"Best match: {best_speaker.content['name']}")
```

### 8.2 Research Papers

**Configuration**:
```python
paper_type = DocumentType(
    type_name="research_paper",
    embedding_configs=[
        EmbeddingConfig("title", "title_embedding", weight=2.0),
        EmbeddingConfig("abstract", "abstract_embedding", weight=1.5),
        EmbeddingConfig(
            "full_text",
            "content_chunks",
            weight=1.0,
            chunk_strategy="fixed",
            chunk_size=512,
            chunk_overlap=50
        )
    ],
    id_field="arxiv_id",
    display_fields=["title", "authors", "abstract", "year"]
)
```

**Data format** (`papers.json`):
```json
[
    {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": ["Vaswani et al."],
        "abstract": "The dominant sequence transduction models...",
        "full_text": "1 Introduction\nRecurrent neural networks...",
        "year": 2017,
        "citations": 98765
    }
]
```

**Usage**:
```python
rag.doc_type_registry.register(paper_type)
await rag.load_documents("data/papers.json", "research_paper")

results = await rag.search("transformer architecture for NLP")
for paper in results[:3]:
    print(f"{paper.content['title']} ({paper.content['year']})")
    print(f"  Authors: {', '.join(paper.content['authors'])}")
```

### 8.3 Product Catalog

**Configuration**:
```python
product_type = DocumentType(
    type_name="product",
    embedding_configs=[
        EmbeddingConfig("name", "name_embedding", weight=1.5),
        EmbeddingConfig("description", "description_embedding", weight=2.0),
        EmbeddingConfig("specifications", "specs_embedding", weight=1.0)
    ],
    id_field="product_id",
    display_fields=["name", "category", "price", "description"]
)
```

**Data format** (`products.csv`):
```csv
product_id,name,category,price,description,specifications
P-12345,Widget Pro,Electronics,99.99,"Advanced widget with AI capabilities","AI: Yes, Battery: 10h, Weight: 200g"
P-12346,Gadget Max,Electronics,149.99,"Professional-grade gadget","Power: 500W, Ports: 4x USB-C"
```

**Usage**:
```python
rag.doc_type_registry.register(product_type)
await rag.load_documents("data/products.csv", "product")

results = await rag.search("portable AI device with long battery")
for product in results[:5]:
    print(f"{product.content['name']} - ${product.content['price']}")
    print(f"  {product.content['description']}")
```

### 8.4 Knowledge Base Articles

**Configuration**:
```python
kb_type = DocumentType(
    type_name="knowledge_base",
    embedding_configs=[
        EmbeddingConfig("title", "title_embedding", weight=2.0),
        EmbeddingConfig(
            "content",
            "content_chunks",
            weight=1.0,
            chunk_strategy="semantic",  # Split on paragraphs
            chunk_size=256
        )
    ],
    id_field="article_id",
    display_fields=["title", "category", "tags", "updated_at"]
)
```

**Data format** (`kb_articles.json`):
```json
[
    {
        "article_id": "kb-001",
        "title": "How to Reset Your Password",
        "category": "Account Management",
        "tags": ["password", "security", "authentication"],
        "content": "To reset your password:\n\n1. Navigate to login page...",
        "created_at": "2025-01-15",
        "updated_at": "2025-02-10",
        "view_count": 1523
    }
]
```

**Usage**:
```python
rag.doc_type_registry.register(kb_type)
await rag.load_documents("data/kb_articles.json", "knowledge_base")

results = await rag.search("forgot password can't login")
for article in results[:3]:
    print(f"📄 {article.content['title']}")
    print(f"   Category: {article.content['category']}")
    print(f"   Tags: {', '.join(article.content['tags'])}")
```

### 8.5 Multi-Domain Search

**Unified search across all document types**:

```python
# Load multiple document types
await rag.load_documents("data/speakers.csv", "speaker")
await rag.load_documents("data/papers.json", "research_paper")
await rag.load_documents("data/products.csv", "product")
await rag.load_documents("data/kb_articles.json", "knowledge_base")

# Search across all types
results = await rag.search("machine learning", top_k=10)

for doc in results:
    doc_type = doc.metadata.get("type")
    if doc_type == "speaker":
        print(f"🎤 Speaker: {doc.content['name']}")
    elif doc_type == "research_paper":
        print(f"📄 Paper: {doc.content['title']}")
    elif doc_type == "product":
        print(f"🛒 Product: {doc.content['name']}")
    elif doc_type == "knowledge_base":
        print(f"📚 Article: {doc.content['title']}")

# Or filter by type
papers_only = await rag.search("neural networks", doc_type_name="research_paper")
```

---

## 9. Implementation Approach

### 9.1 Implementation Phases

**Phase 1: Core Abstractions (Week 1)**
- Create `Document`, `DocumentChunk`, `EmbeddingConfig` classes
- Implement `DocumentType` and `DocumentTypeRegistry`
- Write unit tests for core models

**Phase 2: Generic Service (Week 1-2)**
- Implement `GenericRAGService` with basic functionality
- Create `CSVDocumentLoader` and `JSONDocumentLoader`
- Add configuration file support (YAML)
- Write integration tests

**Phase 3: Backwards Compatibility (Week 2)**
- Create `SpeakerRAGService` wrapper
- Implement data migration utility
- Create adapter for legacy code
- Test with existing MCP server

**Phase 4: FAISS Integration (Week 3)**
- Migrate from JSON to FAISS vector storage
- Implement FAISS index creation and loading
- Add FAISS-based search methods
- Create migration utility from JSON to FAISS
- Update cosine_distance to use FAISS inner product

**Phase 5: Advanced Features (Week 3-4)**
- Implement text chunking strategies
- Add PDF and web loaders
- Optimize FAISS index types (IVF, PQ for large datasets)
- Add quantization for compression

**Phase 6: Documentation & Migration (Week 4)**
- Write migration guide
- Create configuration examples
- Update API documentation
- Create tutorial for new document types
- Document FAISS architecture and performance benchmarks

### 9.2 Refactoring Strategy

**Step 1: Add new code alongside existing**
```
chatboti/
├── rag.py                    # Legacy RAGService (keep as-is)
├── generic_rag.py            # New GenericRAGService
├── models/
│   ├── document.py           # Document, DocumentChunk
│   └── config.py             # EmbeddingConfig, DocumentType
├── loaders/
│   ├── base.py               # DocumentLoader ABC
│   ├── csv_loader.py         # CSVDocumentLoader
│   ├── json_loader.py        # JSONDocumentLoader
│   └── pdf_loader.py         # PDFDocumentLoader (future)
└── storage/
    ├── json_store.py         # JSON-based storage
    ├── parquet_store.py      # Parquet-based storage
    └── sqlite_store.py       # SQLite-based storage
```

**Step 2: Create compatibility layer**
```python
# In rag.py
from chatboti.generic_rag import GenericRAGService, SpeakerRAGService

# Export both old and new
__all__ = ['RAGService', 'GenericRAGService', 'SpeakerRAGService']

# Keep RAGService as deprecated alias
class RAGService(SpeakerRAGService):
    """Deprecated: Use SpeakerRAGService or GenericRAGService instead."""

    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "RAGService is deprecated. Use SpeakerRAGService or GenericRAGService.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)
```

**Step 3: Update dependent code incrementally**
```python
# mcp_server.py
# Phase 1: Use backwards-compatible wrapper
from chatboti.rag import SpeakerRAGService
rag_service = SpeakerRAGService(llm_service=embed_service)

# Phase 2: Migrate to generic service with speaker config
from chatboti.generic_rag import GenericRAGService
from chatboti.config import load_rag_config

registry = load_rag_config("rag_config.yaml")
rag_service = GenericRAGService(llm_service=embed_service, doc_type_registry=registry)
await rag_service.load_documents("data/speakers.csv", "speaker")
```

### 9.3 Testing Strategy

**Unit Tests**:
```python
# tests/test_document.py
def test_document_creation():
    doc = Document(
        id="test-1",
        content={"title": "Test", "text": "Content"},
        metadata={"type": "test"}
    )
    assert doc.id == "test-1"
    assert doc.get_field("title") == "Test"

def test_document_add_embedding():
    doc = Document(id="test-1", content={})
    doc.add_embedding("test_emb", [0.1, 0.2, 0.3])
    assert "test_emb" in doc.embeddings
    assert len(doc.embeddings["test_emb"]) == 3

# tests/test_loaders.py
async def test_csv_loader():
    loader = CSVDocumentLoader()
    doc_type = DocumentType(
        type_name="test",
        embedding_configs=[],
        id_field="id"
    )
    docs = await loader.load("test_data.csv", doc_type)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)

# tests/test_generic_rag.py
async def test_search_single_type():
    rag = GenericRAGService()
    # Load test documents
    # Perform search
    # Verify results
    pass

async def test_search_multi_type():
    # Test filtering by document type
    pass

async def test_weighted_embeddings():
    # Verify weight-based ranking
    pass
```

**Integration Tests**:
```python
# tests/integration/test_speaker_migration.py
async def test_legacy_speaker_compatibility():
    """Verify SpeakerRAGService works like old RAGService."""
    rag = SpeakerRAGService()
    await rag.load_documents("data/speakers.csv", "speaker")

    result = await rag.get_best_speaker("AI expert")
    assert result is not None
    assert "name" in result
    assert "abstract_embedding" not in result  # Stripped

async def test_data_migration():
    """Test migrating legacy JSON to generic format."""
    # Run migration utility
    # Load with GenericRAGService
    # Verify data integrity
    pass
```

**Performance Tests**:
```python
# tests/performance/test_search_performance.py
import time

async def test_search_latency():
    """Ensure search remains fast with generic implementation."""
    rag = GenericRAGService()
    await rag.load_documents("data/large_dataset.json", "test")

    start = time.time()
    results = await rag.search("test query", top_k=10)
    elapsed = time.time() - start

    assert elapsed < 0.1  # Sub-100ms for reasonable dataset
```

### 9.4 Configuration Management

**Environment-based configuration**:
```python
# config.py
import os
from pathlib import Path

class RAGConfig:
    """Configuration for RAG system."""

    def __init__(self):
        self.data_dir = Path(os.getenv("RAG_DATA_DIR", "chatboti/data"))
        self.config_file = os.getenv("RAG_CONFIG_FILE", "rag_config.yaml")
        self.storage_backend = os.getenv("RAG_STORAGE", "json")  # json, parquet, sqlite
        self.enable_chunking = os.getenv("RAG_ENABLE_CHUNKING", "false").lower() == "true"
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

    def get_storage_backend(self):
        if self.storage_backend == "parquet":
            from chatboti.storage.parquet_store import ParquetDocumentStore
            return ParquetDocumentStore
        elif self.storage_backend == "sqlite":
            from chatboti.storage.sqlite_store import SQLiteDocumentStore
            return SQLiteDocumentStore
        else:
            from chatboti.storage.json_store import JSONDocumentStore
            return JSONDocumentStore
```

---

## 10. Migration Guide for Users

### 10.1 For Existing Speaker-Based Applications

**Minimal Changes (Backwards Compatible)**:
```python
# Before (still works)
from chatboti.rag import RAGService
rag = RAGService()

# After (recommended)
from chatboti.rag import SpeakerRAGService
rag = SpeakerRAGService()  # Same API
```

**No code changes required** if using `SpeakerRAGService`.

### 10.2 For New Applications

**Use generic service from the start**:

1. Create configuration file:
```yaml
# my_rag_config.yaml
document_types:
  my_docs:
    id_field: id
    display_fields: [title, summary]
    embeddings:
      - source_field: content
        embedding_name: content_embedding
        weight: 1.0

data_sources:
  my_data:
    type: my_docs
    source: data/my_documents.json
    loader: json
```

2. Initialize service:
```python
from chatboti.generic_rag import GenericRAGService
from chatboti.config import load_rag_config

registry = load_rag_config("my_rag_config.yaml")
rag = GenericRAGService(doc_type_registry=registry)
await rag.load_documents("data/my_documents.json", "my_docs")
```

3. Search:
```python
results = await rag.search("my query", top_k=5)
for doc in results:
    print(doc.content["title"])
```

### 10.3 Adding Custom Document Loaders

**Create custom loader**:
```python
from chatboti.loaders.base import DocumentLoader
from chatboti.models.document import Document

class MyCustomLoader(DocumentLoader):
    """Load documents from custom source."""

    async def load(self, source: str, doc_type: DocumentType) -> List[Document]:
        # Your custom loading logic
        raw_data = self._fetch_from_api(source)

        documents = []
        for item in raw_data:
            doc = Document(
                id=item['id'],
                content=item,
                metadata={'source': source, 'type': doc_type.type_name}
            )
            documents.append(doc)

        return documents

    def supports(self, source: str) -> bool:
        return source.startswith("https://my-api.com/")
```

**Register loader**:
```python
rag = GenericRAGService()
rag.document_loaders.append(MyCustomLoader())
await rag.load_documents("https://my-api.com/docs", "my_type")
```

---

## 11. Future Enhancements

### 11.1 Advanced Chunking Strategies

**Semantic chunking** (respects paragraph/section boundaries):
```python
class SemanticChunker:
    """Chunk text based on semantic boundaries."""

    def chunk(self, text: str, max_tokens: int = 512) -> List[str]:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            if current_tokens + para_tokens > max_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks
```

**Hierarchical chunking** (nested chunks at different granularities):
```python
@dataclass
class HierarchicalChunk:
    """Multi-level chunk with parent-child relationships."""

    chunk_id: str
    level: int  # 0=document, 1=section, 2=paragraph
    parent_id: Optional[str]
    children_ids: List[str]
    text: str
    embedding: Optional[List[float]] = None
```

### 11.2 Multi-Modal Support

**Support images, audio, video**:
```python
@dataclass
class MultiModalDocument(Document):
    """Document with multiple modalities."""

    images: List[bytes] = field(default_factory=list)
    image_embeddings: Dict[str, List[float]] = field(default_factory=dict)
    audio_url: Optional[str] = None
    audio_transcript: Optional[str] = None

# Example: Research paper with figures
paper = MultiModalDocument(
    id="paper-123",
    content={"title": "...", "abstract": "..."},
    images=[fig1_bytes, fig2_bytes],
    embeddings={"text_embedding": [...], "abstract_embedding": [...]},
    image_embeddings={"fig1": [...], "fig2": [...]}
)
```

### 11.3 Hybrid Search (Dense + Sparse)

**Combine vector similarity with BM25 keyword matching**:
```python
class HybridSearchRAG(GenericRAGService):
    """RAG with hybrid dense + sparse retrieval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bm25_index = None  # BM25 index for keyword search

    async def search(
        self,
        query: str,
        doc_type_name: Optional[str] = None,
        top_k: int = 10,
        alpha: float = 0.5  # Weight: 0=pure BM25, 1=pure vector
    ) -> List[Document]:
        # Get dense results
        dense_results = await super().search(query, doc_type_name, top_k * 2)

        # Get sparse results
        sparse_results = self._bm25_search(query, doc_type_name, top_k * 2)

        # Combine with weighted scoring (Reciprocal Rank Fusion)
        combined = self._rank_fusion(dense_results, sparse_results, alpha)

        return combined[:top_k]
```

### 11.4 Query Expansion and Reranking

**Improve retrieval with query reformulation**:
```python
class QueryExpansionRAG(GenericRAGService):
    """RAG with automatic query expansion."""

    async def search(
        self,
        query: str,
        expand_query: bool = True,
        rerank: bool = True,
        **kwargs
    ) -> List[Document]:
        # Expand query with synonyms/related terms
        if expand_query:
            expanded = await self._expand_query(query)
            query = f"{query} {expanded}"

        # Initial retrieval
        results = await super().search(query, **kwargs)

        # Rerank with cross-encoder
        if rerank:
            results = await self._rerank(query, results)

        return results

    async def _expand_query(self, query: str) -> str:
        """Use LLM to expand query with related terms."""
        prompt = f"Given query '{query}', list 3-5 related search terms:"
        expansion = await self.llm_client.complete(prompt)
        return expansion

    async def _rerank(
        self,
        query: str,
        results: List[Document]
    ) -> List[Document]:
        """Rerank with cross-encoder for better relevance."""
        # Use cross-encoder model to score query-document pairs
        scores = []
        for doc in results:
            score = await self.cross_encoder.score(query, doc.content)
            scores.append((score, doc))

        scores.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scores]
```

### 11.5 Caching and Performance Optimization

**Add LRU cache for common queries**:
```python
from functools import lru_cache
import hashlib

class CachedRAG(GenericRAGService):
    """RAG with query result caching."""

    def __init__(self, *args, cache_size: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.cache_size = cache_size

    def _cache_key(self, query: str, doc_type: Optional[str], top_k: int) -> str:
        key_str = f"{query}:{doc_type}:{top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def search(self, query: str, doc_type_name=None, top_k=10) -> List[Document]:
        cache_key = self._cache_key(query, doc_type_name, top_k)

        if cache_key in self.cache:
            logger.debug(f"Cache hit for query: {query}")
            return self.cache[cache_key]

        results = await super().search(query, doc_type_name, top_k)

        # Evict oldest if cache full
        if len(self.cache) >= self.cache_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        self.cache[cache_key] = results
        return results
```

---

## 12. Security Considerations

### 12.1 Input Validation

**Validate document sources**:
```python
class SecureRAG(GenericRAGService):
    """RAG with security validations."""

    ALLOWED_PATHS = ["/data", "/documents"]
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

    async def load_documents(self, source: str, doc_type_name: str, **kwargs):
        # Prevent path traversal
        source_path = Path(source).resolve()
        if not any(str(source_path).startswith(p) for p in self.ALLOWED_PATHS):
            raise SecurityError(f"Source path not allowed: {source}")

        # Check file size
        if source_path.is_file() and source_path.stat().st_size > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {source_path}")

        return await super().load_documents(source, doc_type_name, **kwargs)
```

### 12.2 Data Sanitization

**Sanitize document content**:
```python
import html
import re

def sanitize_content(content: Dict[str, Any]) -> Dict[str, Any]:
    """Remove potentially dangerous content from documents."""
    sanitized = {}
    for key, value in content.items():
        if isinstance(value, str):
            # Remove HTML tags
            value = html.escape(value)
            # Remove control characters
            value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        sanitized[key] = value
    return sanitized
```

### 12.3 Access Control

**Document-level permissions**:
```python
@dataclass
class SecureDocument(Document):
    """Document with access control metadata."""

    owner: str = ""
    permissions: Dict[str, List[str]] = field(default_factory=dict)  # role -> actions

    def can_access(self, user: str, action: str = "read") -> bool:
        """Check if user can perform action on document."""
        user_role = self._get_user_role(user)
        return action in self.permissions.get(user_role, [])
```

---

## 13. Conclusion

This specification provides a comprehensive roadmap for refactoring the chatboti RAG system from a speaker-specific implementation to a generic, extensible document storage and retrieval platform.

**Key Benefits**:
- **Flexibility**: Support any document type with configuration
- **Extensibility**: Plugin architecture for loaders and storage backends
- **Backwards Compatibility**: Existing speaker code continues to work
- **Scalability**: FAISS-based vector storage scales from 100 to 100M+ documents
- **Performance**: 10-50x faster search with minimal overhead (+15 MB, +10% size)
- **Efficiency**: Never use `List[float]` - always `ndarray[float32]` (8x memory savings)
- **Maintainability**: Clean abstractions reduce code duplication

**Recommended Architecture**:
```
Vectors → FAISS index file (.faiss)    [Optimized for search]
Metadata → SQLite database (.db)       [Optimized for queries]
Runtime → NumPy arrays (float32)       [Optimized for memory]
```

**Next Steps**:
1. Review and approve this specification
2. Create implementation tasks in Beads
3. Begin Phase 1 implementation (core abstractions)
4. Phase 4: Migrate to FAISS vector storage
5. Iterate with feedback from stakeholders

---

**Document Version**: 1.0
**Status**: Proposed Design
**Author**: Claude Sonnet 4.5
**Date**: 2026-02-12
**Related Documents**:
- metadata-storage-design.md
- vector-storage-comparison.md

**Change Log**:
- 2026-02-12: Initial specification created
