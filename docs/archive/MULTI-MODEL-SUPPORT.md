# Multi-Model Support: Handling Different Embedding Models

## The Question

Should we create different document versions for different embedding models?

## Answer: No - Design One Flexible Architecture

Instead of multiple document versions, design a **model-agnostic architecture** that handles any embedding model dynamically.

## Why Different Models Matter

### Common Embedding Models

| Model | Provider | Dimensions | Use Case |
|-------|----------|------------|----------|
| `text-embedding-3-small` | OpenAI | 1536 | General purpose |
| `text-embedding-3-large` | OpenAI | 3072 | High quality |
| `text-embedding-ada-002` | OpenAI | 1536 | Legacy |
| `nomic-embed-text` | Nomic AI | 768 | Open source |
| `amazon.titan-embed-text-v1` | AWS Bedrock | 1536 | AWS native |
| `cohere.embed-english-v3` | Cohere | 1024 | Cohere |
| `all-MiniLM-L6-v2` | Sentence Transformers | 384 | Fast, local |
| `bge-large-en-v1.5` | BAAI | 1024 | SOTA open |

### Key Differences

1. **Dimensions**: 384 to 3072
2. **Performance**: Speed vs accuracy trade-offs
3. **Cost**: $0.00002 to $0.0004 per 1K tokens
4. **Availability**: Cloud vs local

## Flexible Architecture Design

### 1. Model Configuration

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding model."""
    provider: str           # "openai", "bedrock", "ollama", etc.
    model_name: str         # Full model identifier
    dimensions: int         # Vector dimensions
    max_tokens: int         # Context window
    cost_per_1k: float     # Pricing (optional)
    normalize: bool = True  # Whether to normalize vectors

# Define supported models
EMBEDDING_MODELS = {
    "openai/text-embedding-3-small": EmbeddingModelConfig(
        provider="openai",
        model_name="text-embedding-3-small",
        dimensions=1536,
        max_tokens=8191,
        cost_per_1k=0.00002
    ),
    "openai/text-embedding-3-large": EmbeddingModelConfig(
        provider="openai",
        model_name="text-embedding-3-large",
        dimensions=3072,
        max_tokens=8191,
        cost_per_1k=0.00013
    ),
    "nomic/nomic-embed-text": EmbeddingModelConfig(
        provider="ollama",
        model_name="nomic-embed-text",
        dimensions=768,
        max_tokens=8192,
        cost_per_1k=0.0  # Free, local
    ),
    "bedrock/amazon.titan-embed-text-v1": EmbeddingModelConfig(
        provider="bedrock",
        model_name="amazon.titan-embed-text-v1",
        dimensions=1536,
        max_tokens=8000,
        cost_per_1k=0.0001
    ),
}
```

### 2. Dynamic FAISS Index Creation

```python
import faiss
import numpy as np
from pathlib import Path

class MultiModelRAGService:
    """RAG service supporting multiple embedding models."""

    def __init__(
        self,
        data_dir: Path,
        model_id: str,  # e.g., "openai/text-embedding-3-small"
        llm_service: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Get model configuration
        if model_id not in EMBEDDING_MODELS:
            raise ValueError(f"Unknown model: {model_id}")
        self.model_config = EMBEDDING_MODELS[model_id]
        self.model_id = model_id

        # Model-specific file paths
        self.index_path = self.data_dir / f"vectors_{self._sanitize_model_id()}.faiss"
        self.metadata_path = self.data_dir / f"metadata_{self._sanitize_model_id()}.json"

        # Initialize or load FAISS index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            # Verify dimensions match
            if self.index.d != self.model_config.dimensions:
                raise ValueError(
                    f"Index dimension {self.index.d} doesn't match "
                    f"model dimension {self.model_config.dimensions}"
                )
        else:
            # Create new index with correct dimensions
            self.index = faiss.IndexFlatIP(self.model_config.dimensions)

        # Initialize embedding client
        self.embed_client = get_llm_client(
            self.model_config.provider,
            model=self.model_config.model_name
        )

        # Load metadata
        self.metadata = JSONMetadataStore(self.metadata_path)

    def _sanitize_model_id(self) -> str:
        """Convert model ID to filename-safe string."""
        return self.model_id.replace("/", "-").replace(":", "-")

    async def add_document(self, doc: Document) -> int:
        """Add document with current model's embeddings."""
        # Generate embedding with configured model
        embedding = await self.embed_client.embed(doc.content)
        embedding = np.array(embedding, dtype=np.float32)

        # Normalize if configured
        if self.model_config.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        embedding = embedding.reshape(1, -1)

        # Verify dimension
        if embedding.shape[1] != self.model_config.dimensions:
            raise ValueError(
                f"Embedding dimension {embedding.shape[1]} doesn't match "
                f"expected {self.model_config.dimensions}"
            )

        # Add to FAISS
        faiss_id = self.index.ntotal
        self.index.add(embedding)

        # Save metadata with model info
        doc_dict = doc.to_dict()
        doc_dict['embedding_model'] = self.model_id
        self.metadata.add_document(faiss_id, Document.from_dict(doc_dict))

        # Persist
        faiss.write_index(self.index, str(self.index_path))

        return faiss_id
```

### 3. Storage Structure (Multi-Model)

```
data/
├── vectors_openai-text-embedding-3-small.faiss    # 1536-dim index
├── metadata_openai-text-embedding-3-small.json
├── vectors_nomic-nomic-embed-text.faiss           # 768-dim index
├── metadata_nomic-nomic-embed-text.json
└── config.yaml                                     # Model registry
```

### 4. Model Migration Utility

```python
async def migrate_embeddings(
    old_model_id: str,
    new_model_id: str,
    data_dir: Path
) -> None:
    """Re-embed all documents with new model."""

    print(f"Migrating from {old_model_id} → {new_model_id}")

    # Load old service
    old_service = MultiModelRAGService(data_dir, old_model_id)

    # Create new service
    new_service = MultiModelRAGService(data_dir, new_model_id)

    # Re-embed all documents
    count = old_service.metadata.count()
    for faiss_id in range(count):
        doc = old_service.metadata.get_document(faiss_id)
        if doc:
            # Re-embed with new model
            await new_service.add_document(doc)
            print(f"  {faiss_id + 1}/{count}: {doc.id[:20]}...")

    print(f"✓ Migrated {count} documents")
    print(f"  Old: {old_model_id} ({old_service.model_config.dimensions} dims)")
    print(f"  New: {new_model_id} ({new_service.model_config.dimensions} dims)")
```

## Storage Size by Model

### Example: 1000 Documents

| Model | Dimensions | FAISS Size | Compression Ratio |
|-------|-----------|------------|-------------------|
| `all-MiniLM-L6-v2` | 384 | **1.5 MB** | 1x (baseline) |
| `nomic-embed-text` | 768 | **3.0 MB** | 2x |
| `cohere.embed-english-v3` | 1024 | **4.0 MB** | 2.7x |
| `text-embedding-3-small` | 1536 | **6.0 MB** | 4x |
| `text-embedding-3-large` | 3072 | **12.0 MB** | 8x |

**Formula**: `Storage = N_docs × dimensions × 4 bytes (float32)`

### With Quantization (Optional)

| Model | Original | PQ Quantized (m=8) | Savings |
|-------|----------|-------------------|---------|
| 384-dim | 1.5 MB | 96 KB | **94%** |
| 768-dim | 3.0 MB | 96 KB | **97%** |
| 1536-dim | 6.0 MB | 96 KB | **98.4%** |
| 3072-dim | 12.0 MB | 96 KB | **99.2%** |

## Configuration File Approach

### config.yaml

```yaml
# RAG Configuration
models:
  # Primary model
  primary: openai/text-embedding-3-small

  # Available models
  available:
    - id: openai/text-embedding-3-small
      provider: openai
      dimensions: 1536
      max_tokens: 8191
      cost_per_1k: 0.00002

    - id: nomic/nomic-embed-text
      provider: ollama
      dimensions: 768
      max_tokens: 8192
      cost_per_1k: 0.0

    - id: bedrock/amazon.titan-embed-text-v1
      provider: bedrock
      dimensions: 1536
      max_tokens: 8000
      cost_per_1k: 0.0001

# Storage settings
storage:
  base_dir: ./data
  metadata_format: json  # or sqlite

# FAISS settings
faiss:
  index_type: flat  # flat, ivf, pq
  normalize: true
  metric: ip  # inner product (for cosine)
```

### Usage

```python
from chatboti.config import load_config

config = load_config("config.yaml")

# Use primary model
rag = MultiModelRAGService(
    data_dir=config.storage.base_dir,
    model_id=config.models.primary
)

# Or specify different model
rag = MultiModelRAGService(
    data_dir=config.storage.base_dir,
    model_id="nomic/nomic-embed-text"  # Smaller, faster, free
)
```

## Multi-Model Search (Advanced)

### Cross-Model Search

```python
class MultiModelSearchService:
    """Search across multiple embedding models."""

    def __init__(self, data_dir: Path, model_ids: List[str]):
        self.services = {
            model_id: MultiModelRAGService(data_dir, model_id)
            for model_id in model_ids
        }

    async def search_all(self, query: str, k: int = 5) -> Dict[str, List[Document]]:
        """Search across all models, return results per model."""
        results = {}
        for model_id, service in self.services.items():
            results[model_id] = await service.search(query, k)
        return results

    async def ensemble_search(
        self,
        query: str,
        k: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Document]:
        """Ensemble search with weighted voting."""
        all_results = await self.search_all(query, k * 2)  # Get more candidates

        # Score documents by weighted rank
        doc_scores = {}
        for model_id, docs in all_results.items():
            weight = weights.get(model_id, 1.0) if weights else 1.0
            for rank, doc in enumerate(docs):
                score = weight / (rank + 1)  # 1/rank weighting
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = {'doc': doc, 'score': 0}
                doc_scores[doc.id]['score'] += score

        # Sort by ensemble score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        return [item['doc'] for item in sorted_docs[:k]]
```

## Performance Comparison

### Search Speed (10K documents)

| Model | Dimensions | Linear Search | FAISS Flat | FAISS IVF |
|-------|-----------|---------------|------------|-----------|
| 384-dim | 384 | 25ms | **0.5ms** | 0.2ms |
| 768-dim | 768 | 50ms | **1.0ms** | 0.4ms |
| 1536-dim | 1536 | 100ms | **2.0ms** | 0.8ms |
| 3072-dim | 3072 | 200ms | **4.0ms** | 1.6ms |

**Observation**: Larger dimensions = slower search, but FAISS keeps it fast.

## Model Selection Guide

### Choose Based on Use Case

**Small, Fast, Local (384-768 dims):**
- `all-MiniLM-L6-v2` (384)
- `nomic-embed-text` (768)
- **Use when**: Speed > quality, local deployment, cost-sensitive
- **Storage**: 1.5-3 MB per 1K docs

**Balanced (1024-1536 dims):**
- `text-embedding-3-small` (1536)
- `amazon.titan-embed-text-v1` (1536)
- `bge-large-en-v1.5` (1024)
- **Use when**: Good balance of quality and speed
- **Storage**: 4-6 MB per 1K docs

**High Quality (3072 dims):**
- `text-embedding-3-large` (3072)
- **Use when**: Quality is critical, cost acceptable
- **Storage**: 12 MB per 1K docs

### Migration Path

```
Start:    all-MiniLM-L6-v2 (384, fast prototyping)
  ↓
Improve:  nomic-embed-text (768, better quality)
  ↓
Scale:    text-embedding-3-small (1536, production)
  ↓
Optimize: text-embedding-3-large (3072, best quality)
```

## Documentation Update

### Update Generic Spec to Include

**Add Section: "Multi-Model Support"**

```markdown
## X. Multi-Model Support

### X.1 Model Configuration
- Support for any embedding model
- Dynamic dimension handling
- Model-specific file naming

### X.2 Storage Strategy
- Separate FAISS indices per model
- Metadata includes model identifier
- Migration utilities for model changes

### X.3 Model Selection
- Comparison table (dims, cost, speed)
- Selection criteria
- Migration path

### X.4 Implementation
- `EmbeddingModelConfig` class
- `MultiModelRAGService` implementation
- Configuration file format
```

## Conclusion

**Don't create separate document versions.**

Instead:
1. ✅ **Parameterize** the architecture by model dimensions
2. ✅ **Add a section** on multi-model support to the main spec
3. ✅ **Document** model selection criteria and trade-offs
4. ✅ **Provide** migration utilities for model changes
5. ✅ **Show** configuration examples for common models

This keeps documentation DRY while supporting any embedding model dynamically.
