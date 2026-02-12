# Storage Evolution Strategy: JSON → SQLite Migration

## The Question

Can we use JSON for metadata initially, with a migration path to SQLite later?

## Answer: YES! Incremental Approach is Better

### Evolution Path

```
Phase 1 (Simple):          Phase 2 (Scaled):
├── vectors.faiss          ├── vectors.faiss
└── metadata.json          └── metadata.db (SQLite)
```

## Storage Interface Design

### Abstract Storage Layer

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

## Implementation 1: JSON Storage

### Advantages
- ✅ **Simple**: No SQL knowledge needed
- ✅ **Readable**: Easy to inspect and debug
- ✅ **Portable**: Works everywhere
- ✅ **No dependencies**: No SQLite library needed
- ✅ **Fast start**: Quick prototyping
- ✅ **Version control friendly**: Can diff changes

### Limitations
- ⚠️ **No transactions**: Risk of corruption on crash
- ⚠️ **Slow updates**: Requires full file rewrite
- ⚠️ **Memory intensive**: Loads entire file
- ⚠️ **No indexing**: Linear search for metadata queries
- ⚠️ **Poor concurrency**: File locking issues

### When to Use
- **Small datasets**: <1000 documents
- **Infrequent updates**: Batch loading, rare changes
- **Single-user**: No concurrent access
- **Development**: Prototyping and testing

### Implementation

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

### JSON File Format

```json
{
  "0": {
    "doc_id": "doc-uuid-123",
    "content": "Full text content here...",
    "source": "papers.pdf",
    "doc_type": "research_paper",
    "metadata": {
      "title": "Attention Is All You Need",
      "authors": ["Vaswani", "Shazeer", ...],
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

## Implementation 2: SQLite Storage

### Advantages
- ✅ **ACID transactions**: Safe concurrent access
- ✅ **Efficient updates**: Incremental changes
- ✅ **Indexing**: Fast metadata queries
- ✅ **Low memory**: Doesn't load everything
- ✅ **Reliable**: Battle-tested database
- ✅ **Standard**: SQL queries

### Limitations
- ⚠️ **Complexity**: Requires SQL knowledge
- ⚠️ **Binary format**: Harder to inspect/diff
- ⚠️ **Schema management**: Migrations needed

### When to Use
- **Large datasets**: >1000 documents
- **Frequent updates**: Real-time changes
- **Multi-user**: Concurrent access
- **Complex queries**: Filtering, sorting, joins
- **Production**: Deployed systems

### Implementation

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

## GenericRAGService with Storage Abstraction

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

## Migration Strategy: JSON → SQLite

### Automatic Migration Utility

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

# Usage
migrate_json_to_sqlite(
    Path("metadata.json"),
    Path("metadata.db")
)
```

### Detection and Auto-Migration

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

## Recommended Evolution Path

### Phase 1: Start with JSON (Week 1-2)
```python
# Simple start
rag = GenericRAGService(
    index_path=Path("vectors.faiss"),
    metadata_store=Path("metadata.json")  # JSON storage
)
```

**Benefits:**
- Fast prototyping
- Easy debugging
- No SQL knowledge needed
- Good for <1000 documents

### Phase 2: Migrate to SQLite (Week 3-4)
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

**Triggers:**
- Dataset grows beyond 1000 documents
- Need frequent updates
- Want complex metadata queries
- Production deployment

## Comparison Summary

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

## Conclusion

**Start with JSON, migrate to SQLite when needed.**

This incremental approach:
- ✅ Simplifies initial development
- ✅ Provides clear migration path
- ✅ Matches actual scaling needs
- ✅ Reduces upfront complexity
- ✅ Maintains flexibility

**Implementation order:**
1. Week 1-2: JSON storage (simple, working system)
2. Week 3-4: SQLite storage + migration utility
3. Production: Use SQLite (performance + reliability)
