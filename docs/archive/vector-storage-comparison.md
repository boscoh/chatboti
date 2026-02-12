# Vector Storage Comparison: Parquet vs FAISS-CPU

## Executive Summary

This document evaluates two approaches for storing and retrieving speaker embeddings in the chatboti project: Parquet columnar format and FAISS-CPU similarity search library. Based on the current use case (63 speakers, ~1MB embeddings file), **Parquet is recommended** for its simplicity, portability, and sufficient performance at this scale.

## Current Use Case Context

- **Dataset size**: 63 speakers
- **Storage format**: JSON (~750KB - 1.1MB)
- **Embeddings per speaker**: 2 (abstract, bio)
- **Query pattern**: Single nearest-neighbor search per user query
- **Current implementation**: In-memory numpy cosine distance calculation
- **Expected scale**: Hundreds to low thousands of embeddings

## Option 1: Apache Parquet

### Overview

Apache Parquet is a columnar storage format optimized for analytical workloads. Originally designed for big data processing, it provides efficient compression and column-oriented storage that can be used for embeddings alongside metadata.

### Performance Characteristics

**Read Performance:**
- Cold start latency for reading embeddings: ~10-50ms for small datasets
- Efficient column selection allows reading only needed fields
- Compression reduces I/O overhead by 33-50% compared to JSON

**Write Performance:**
- Slightly slower than JSON for single writes
- Optimized for batch writes
- For 63 speakers: <100ms to write entire dataset

**Query Performance:**
- No built-in similarity search
- Requires loading vectors into memory and computing distances (numpy/polars)
- Linear scan performance: O(n) where n = number of vectors
- For 63 speakers with 1536-dim embeddings: <1ms for similarity search after loading

### Storage Efficiency

- **Compression**: Reduces storage by 33-50% compared to raw JSON
- **File size estimate**: 500-800KB for current dataset (vs 750KB-1.1MB JSON)
- **Format overhead**: Minimal for datasets >100 vectors
- **Metadata storage**: Efficient columnar storage of speaker metadata alongside embeddings

### Integration Complexity

**Pros:**
- Simple integration via `pyarrow` or `polars` libraries
- Portable format readable by many tools (Pandas, DuckDB, Spark)
- No external dependencies beyond Python libraries
- Easy backup and version control
- Can store metadata and embeddings in single file

**Cons:**
- Requires additional libraries (pyarrow ~10MB, polars ~25MB)
- No built-in indexing for vector similarity
- Requires implementing similarity search logic

**Implementation effort**: 2-4 hours
- Convert JSON to Parquet schema
- Update load/save logic in `rag.py`
- Test with existing query patterns

### Code Example

```python
import polars as pl
import numpy as np

# Write embeddings
df = pl.DataFrame({
    'name': ['Speaker A', 'Speaker B'],
    'bio': ['Bio text...', 'Bio text...'],
    'abstract_embedding': [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    'bio_embedding': [[0.5, 0.6, ...], [0.7, 0.8, ...]]
})
df.write_parquet('embeddings.parquet')

# Read and search
df = pl.read_parquet('embeddings.parquet')
query_vec = np.array([0.15, 0.25, ...])

# Compute cosine similarity
embeddings = np.stack(df['abstract_embedding'].to_numpy())
similarities = embeddings @ query_vec / (
    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
)
best_idx = np.argmax(similarities)
```

## Option 2: FAISS-CPU

### Overview

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. Developed by Meta, it provides optimized algorithms for nearest-neighbor search at scale.

### Performance Characteristics

**Index Build Performance:**
- IndexFlatL2 (exact search): ~1-5ms for 63 vectors
- IndexIVFFlat (approximate search): 10-50ms for 63 vectors (overkill for this scale)
- GPU acceleration available but not needed at this scale

**Query Performance:**
- Exact search (IndexFlatL2): <1ms for single query on 63 vectors
- Approximate search: <0.1ms but not meaningful at this scale
- Built-in batch query support for multiple simultaneous searches
- At 1M vectors (768-dim): single-digit milliseconds with optimized indices

**Memory Performance:**
- IndexFlatL2: Stores full vectors in memory (~400KB for 63 x 1536-dim vectors)
- Compressed indices (Product Quantization): 50% memory reduction with minimal accuracy loss
- Memory efficiency techniques not beneficial below 10K vectors

### Storage Efficiency

**Index Types:**
- **IndexFlatL2**: Full precision, no compression, fastest exact search
  - Memory: 63 vectors x 1536 dims x 4 bytes = ~388KB per embedding field
- **IndexIVFFlat**: Partitioned index with compression
  - Memory reduction: ~20-30% but adds complexity
- **IndexPQ**: Product quantization for extreme compression
  - Memory reduction: Up to 97% (8 bytes per vector)
  - Accuracy trade-off: 95-99% precision

**On-disk storage:**
- FAISS indices can be serialized to disk
- File size similar to uncompressed vectors for IndexFlatL2
- Compressed indices smaller but lossy

### Integration Complexity

**Pros:**
- Purpose-built for vector similarity search
- Highly optimized C++ implementation with Python bindings
- Scales to billions of vectors when needed
- GPU acceleration available (faiss-gpu package)
- Rich set of index types for different trade-offs

**Cons:**
- Additional dependency (faiss-cpu ~15-20MB)
- Steep learning curve for advanced features
- Requires understanding of index types and parameters
- No built-in metadata storage (need separate data structure)
- More complex error handling and edge cases
- Overkill for small datasets (<10K vectors)

**Implementation effort**: 4-8 hours
- Choose and configure appropriate index type
- Implement index building and persistence
- Maintain separate metadata storage
- Handle index updates and rebuilds
- Test index parameters and accuracy

### Code Example

```python
import faiss
import numpy as np
import json

# Build index
embeddings = np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]], dtype=np.float32)
dimension = embeddings.shape[1]

# Exact search index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, 'embeddings.faiss')

# Load and search
index = faiss.read_index('embeddings.faiss')
query_vec = np.array([[0.15, 0.25, ...]], dtype=np.float32)

k = 1  # number of nearest neighbors
distances, indices = index.search(query_vec, k)
best_idx = indices[0][0]

# Note: Metadata must be stored separately in JSON/Parquet
metadata = json.load(open('speaker_metadata.json'))
best_speaker = metadata[best_idx]
```

## Detailed Comparison

### Performance Comparison

| Metric | Parquet + Numpy | FAISS-CPU (IndexFlatL2) | Winner |
|--------|----------------|------------------------|--------|
| Read latency (63 vectors) | 10-50ms | 1-5ms (index load) | FAISS |
| Query latency (single) | <1ms | <1ms | Tie |
| Query latency (batch 100) | ~50ms | ~5ms | FAISS |
| Write latency | <100ms | <10ms | FAISS |
| Scale to 1M vectors | 50-100ms query | <5ms query (optimized) | FAISS |

### Storage Comparison

| Metric | Parquet | FAISS-CPU | Winner |
|--------|---------|-----------|--------|
| File size (63 speakers) | 500-800KB | 400-500KB (flat) | FAISS |
| Compression available | Yes (33-50%) | Yes (50-97% lossy) | Parquet |
| Metadata included | Yes | No (separate file) | Parquet |
| Portability | Excellent | Fair (binary format) | Parquet |

### Integration Comparison

| Metric | Parquet | FAISS-CPU | Winner |
|--------|---------|-----------|--------|
| Implementation time | 2-4 hours | 4-8 hours | Parquet |
| Learning curve | Gentle | Steep | Parquet |
| Code complexity | Low | Medium-High | Parquet |
| Dependency size | ~25MB (polars) | ~20MB | Tie |
| Maintenance burden | Low | Medium | Parquet |

## Pros and Cons Summary

### Parquet

**Pros:**
- Simple integration with existing Python ecosystem
- Single file stores vectors and metadata together
- Excellent portability and tooling support
- No complex configuration or tuning required
- Easy to inspect and debug
- Sufficient performance for current scale
- Standard format for data engineering workflows

**Cons:**
- No built-in similarity search optimization
- Linear scan performance (acceptable at current scale)
- Requires loading data into memory for queries
- May need migration to specialized solution above 10K vectors

### FAISS-CPU

**Pros:**
- Purpose-built for vector similarity search
- Exceptional performance at large scale
- Rich set of optimization techniques
- Battle-tested at Meta scale (billions of vectors)
- Clear migration path to GPU acceleration
- Efficient batch query support

**Cons:**
- Significant complexity overhead for small datasets
- Requires separate metadata storage
- Steeper learning curve
- More complex error handling
- Binary format less portable
- Overkill for datasets under 10K vectors

## Recommendation

**For the current chatboti project: Use Parquet**

### Rationale

1. **Scale is small**: 63 speakers means ~126 embeddings total. Linear scan takes <1ms.

2. **Simplicity matters**: Parquet + polars is straightforward to implement and maintain. FAISS adds significant complexity for negligible benefit at this scale.

3. **Unified storage**: Parquet stores embeddings and metadata together, eliminating synchronization concerns.

4. **Portability**: Parquet files are readable by many tools, making data inspection and migration easier.

5. **Future-proof**: The format scales well to thousands of speakers. If growth reaches 10K+ speakers, migration to FAISS becomes worthwhile.

6. **Development velocity**: 2-4 hour implementation vs 4-8 hours, with lower maintenance burden.

### When to Reconsider

Switch to FAISS when:
- Speaker count exceeds 5,000-10,000 (query latency becomes noticeable)
- Batch queries become common (100+ queries per request)
- Real-time latency requirements demand <1ms queries
- Memory constraints require compression (Product Quantization)

### Migration Path

If switching to FAISS becomes necessary:
1. Keep metadata in Parquet (it's efficient for tabular data)
2. Move embeddings to FAISS index
3. Maintain consistent ordering between metadata and index
4. Start with IndexFlatL2 for exact search
5. Explore approximate indices (IVF, HNSW) only if needed for scale

## Implementation Recommendation

For immediate next steps with Parquet:

```python
# Use polars for efficient Parquet I/O
import polars as pl

class RAGService:
    def __init__(self, llm_service: Optional[str] = None):
        # ... existing init code ...
        self.embed_parquet = self.embed_json.replace('.json', '.parquet')

    async def connect(self):
        if self.speakers_with_embeddings and self.speakers:
            return
        elif self.is_exists(self.embed_parquet):
            df = pl.read_parquet(self._resolve_data_path(self.embed_parquet))
            self.speakers_with_embeddings = df.to_dicts()
        elif self.is_exists(self.embed_json):
            # Migrate from JSON to Parquet
            self.speakers_with_embeddings = json.loads(
                self.read_text_file(self.embed_json)
            )
            df = pl.DataFrame(self.speakers_with_embeddings)
            df.write_parquet(self._resolve_data_path(self.embed_parquet))
        else:
            # Generate embeddings and save as Parquet
            self.speakers_with_embeddings = await self._generate_speaker_embeddings()
            df = pl.DataFrame(self.speakers_with_embeddings)
            df.write_parquet(self._resolve_data_path(self.embed_parquet))

        self.speakers = py_.map(self.speakers_with_embeddings, self._strip_embeddings)
```

## References

### FAISS Resources
- [FAISS GitHub Repository](https://github.com/facebookresearch/faiss)
- [Meta Engineering: FAISS Library for Efficient Similarity Search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
- [FAISS Low-level Benchmarks](https://github.com/facebookresearch/faiss/wiki/Low-level-benchmarks)
- [FAISS Vector Codec Benchmarks](https://github.com/facebookresearch/faiss/wiki/Vector-codec-benchmarks)
- [FAISS Memory Footprint Optimization](https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint)
- [OpenSearch: Optimizing with FAISS FP16 Quantization](https://opensearch.org/blog/optimizing-opensearch-with-fp16-quantization/)
- [Enhancing GPU-Accelerated Vector Search in FAISS with NVIDIA cuVS](https://developer.nvidia.com/blog/enhancing-gpu-accelerated-vector-search-in-faiss-with-nvidia-cuvs/)
- [Top Vector Databases of 2026](https://rahulkolekar.com/top-vector-databases-of-2026-free-paid-and-performance-comparison/)

### Parquet Resources
- [The Best Way to Use Text Embeddings Portably with Parquet and Polars](https://minimaxir.com/2025/02/embeddings-parquet/)
- [Parquet Data Format: Pros and Cons for 2025](https://edgedelta.com/company/blog/parquet-data-format)
- [MotherDuck: Why Choose Parquet Table File Format](https://motherduck.com/learn-more/why-choose-parquet-table-file-format/)
- [Apache Parquet: Efficient Data Storage](https://www.databricks.com/glossary/what-is-parquet)
- [Is Parquet Becoming the Bottleneck? New Storage Formats in 2025](https://www.databend.com/blog/category-engineering/2025-09-15-storage-format)

### Comparison Resources
- [ChromaDB vs FAISS: Comprehensive Guide](https://mohamedbakrey094.medium.com/chromadb-vs-faiss-a-comprehensive-guide-for-vector-search-and-ai-applications-39762ed1326f)
- [Understanding FAISS Vector Store and Advantages](https://medium.com/@amrita.thakur/understanding-faiss-vector-store-and-its-advantages-cdc7b54afe47)
- [Choosing the Right Vector Database: Architectures and Trade-offs](https://medium.com/@soniakashyap001/choosing-the-right-vector-database-architectures-storage-trade-offs-and-real-world-fit-84193b7de5df)
- [Best Vector Databases in 2025: Complete Comparison](https://www.firecrawl.dev/blog/best-vector-databases-2025)
- [A Benchmark of FAISS and Annoy](https://arxiv.org/pdf/2412.01555)

---

**Document Version**: 1.0
**Date**: 2026-02-12
**Author**: Claude Sonnet 4.5
