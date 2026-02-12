# Generic Document Storage Specification

## Executive Summary

This specification outlines the refactoring of the chatboti RAG system from a speaker-specific implementation to a generic document storage and retrieval system. The current system is tightly coupled to the speaker/agenda domain with hardcoded field names, single CSV source, and fixed embedding structure. This limits reusability for other document types like research papers, product catalogs, knowledge bases, or general text collections.

**Goal**: Create a flexible, domain-agnostic RAG architecture that maintains backwards compatibility with the existing speaker data while enabling new use cases.

**Status**: Proposed Design
**Author**: Claude Sonnet 4.5
**Date**: 2026-02-12
**Related**: metadata-storage-design.md, vector-storage-comparison.md

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

### 2.1 Cannot Handle Different Document Types

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

### 2.2 Hardcoded Field Names Limit Flexibility

**Example**: Loading a different speaker CSV with fields:
- `presenter_bio_summary` instead of `bio_max_120_words`
- `talk_description` instead of `final_abstract_max_150_words`

**Current system**: Requires code changes to `_generate_speaker_embeddings()`.

**Desired**: Configuration-driven field mapping.

### 2.3 Single CSV Source Limitation

**Cannot support**:
1. **Multi-file ingestion**: Load speakers from multiple CSVs
2. **Mixed formats**: Combine CSV speakers + PDF papers + JSON articles
3. **Incremental updates**: Add new documents without regenerating all embeddings
4. **Dynamic sources**: Fetch from APIs or databases
5. **Different schemas**: Documents with varying field structures

### 2.4 Difficult to Extend for New Use Cases

**Adding a new document type currently requires**:
1. Create new `RAGService` subclass or duplicate code
2. Hardcode new field names in embedding generation
3. Update distance calculation logic
4. Create new MCP tools
5. Duplicate testing and validation

**Result**: Code duplication, maintenance burden, fragile architecture.

---

## 3. Proposed Generic Document Model

### 3.1 Core Abstractions

#### **3.1.1 Document Interface**

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

#### **3.1.2 Document Chunk**

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

#### **3.1.3 Embedding Configuration**

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

### 3.2 Document Type Registry

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

## 4. API Design

### 4.1 Generic Document Ingestion Interface

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

### 4.2 Generic RAG Service

Refactor `RAGService` to be domain-agnostic:

```python
class GenericRAGService:
    """Domain-agnostic RAG service for any document type."""

    def __init__(
        self,
        llm_service: Optional[str] = None,
        doc_type_registry: Optional[DocumentTypeRegistry] = None
    ):
        self.llm_service = llm_service or os.getenv("LLM_SERVICE", "openai").lower()
        self.embed_client = self._initialize_embed_client()
        self.doc_type_registry = doc_type_registry or DocumentTypeRegistry()

        # Generic storage (replaces speakers_with_embeddings)
        self.documents: Dict[str, Document] = {}  # id -> Document
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

### 4.3 Configuration File Format

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

## 5. Backwards Compatibility

### 5.1 Migration Path for Existing Speaker Data

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

### 5.2 Data Migration Strategy

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

### 5.3 Adapter Pattern for Legacy Code

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

## 6. Integration Points

### 6.1 Metadata Storage Design Integration

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

### 6.2 Vector Storage (Parquet/FAISS) Integration

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

### 6.3 Plugin Architecture for Document Loaders

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

## 7. Example Use Cases

### 7.1 Speaker Data (Current Use Case)

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

### 7.2 Research Papers

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

### 7.3 Product Catalog

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

### 7.4 Knowledge Base Articles

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

### 7.5 Multi-Domain Search

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

## 8. Implementation Approach

### 8.1 Implementation Phases

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

**Phase 4: Advanced Features (Week 3)**
- Implement text chunking strategies
- Add PDF and web loaders
- Integrate with SQLite metadata store
- Add FAISS support for large datasets

**Phase 5: Documentation & Migration (Week 3-4)**
- Write migration guide
- Create configuration examples
- Update API documentation
- Create tutorial for new document types

### 8.2 Refactoring Strategy

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

### 8.3 Testing Strategy

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

### 8.4 Configuration Management

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

## 9. Migration Guide for Users

### 9.1 For Existing Speaker-Based Applications

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

### 9.2 For New Applications

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

### 9.3 Adding Custom Document Loaders

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

## 10. Future Enhancements

### 10.1 Advanced Chunking Strategies

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

### 10.2 Multi-Modal Support

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

### 10.3 Hybrid Search (Dense + Sparse)

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

### 10.4 Query Expansion and Reranking

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

### 10.5 Caching and Performance Optimization

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

## 11. Security Considerations

### 11.1 Input Validation

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

### 11.2 Data Sanitization

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

### 11.3 Access Control

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

## 12. Conclusion

This specification provides a comprehensive roadmap for refactoring the chatboti RAG system from a speaker-specific implementation to a generic, extensible document storage and retrieval platform.

**Key Benefits**:
- **Flexibility**: Support any document type with configuration
- **Extensibility**: Plugin architecture for loaders and storage backends
- **Backwards Compatibility**: Existing speaker code continues to work
- **Scalability**: Clear integration with advanced storage (SQLite, FAISS)
- **Maintainability**: Clean abstractions reduce code duplication

**Next Steps**:
1. Review and approve this specification
2. Create implementation tasks in Beads
3. Begin Phase 1 implementation (core abstractions)
4. Iterate with feedback from stakeholders

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
