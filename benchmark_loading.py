"""Benchmark loading performance: FaissRAGService vs HDF5RAGService."""

import asyncio
import time
from pathlib import Path
import json

from microeval.llm import get_llm_client
from chatboti.faiss_rag import FaissRAGService
from chatboti.hdf5_rag import HDF5RAGService


async def benchmark_faiss_load(embed_client, index_path: Path, metadata_path: Path, n_runs: int = 5):
    """Benchmark FaissRAGService loading time."""
    times = []

    for i in range(n_runs):
        start = time.perf_counter()
        async with FaissRAGService(
            embed_client=embed_client,
            index_path=index_path,
            metadata_path=metadata_path
        ) as rag:
            # Service is loaded in __aenter__
            pass
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Run {i+1}/{n_runs}: {times[-1]*1000:.1f} ms")

    return times


async def benchmark_hdf5_load(embed_client, hdf5_path: Path, n_runs: int = 5):
    """Benchmark HDF5RAGService loading time."""
    times = []

    for i in range(n_runs):
        start = time.perf_counter()
        async with HDF5RAGService(
            embed_client=embed_client,
            hdf5_path=hdf5_path
        ) as rag:
            # Service is loaded in __aenter__
            pass
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Run {i+1}/{n_runs}: {times[-1]*1000:.1f} ms")

    return times


def get_file_stats(index_path: Path, metadata_path: Path):
    """Get file size statistics."""
    index_size = index_path.stat().st_size if index_path.exists() else 0
    metadata_size = metadata_path.stat().st_size if metadata_path.exists() else 0

    # Get vector and document counts from metadata
    with open(metadata_path) as f:
        data = json.load(f)
        n_vectors = len(data['chunk_refs'])
        n_docs = len(data['documents'])

    return {
        'index_size': index_size,
        'metadata_size': metadata_size,
        'total_size': index_size + metadata_size,
        'n_vectors': n_vectors,
        'n_docs': n_docs
    }


async def main():
    """Run benchmarks."""
    data_dir = Path("chatboti/data")

    # FAISS + JSON files
    faiss_index = data_dir / "vectors-nomic-embed-text.faiss"
    faiss_metadata = data_dir / "metadata-nomic-embed-text.json"

    # HDF5 file (will create if doesn't exist)
    hdf5_file = data_dir / "embeddings-nomic-embed-text.h5"

    print("=" * 70)
    print("RAG Service Loading Performance Benchmark")
    print("=" * 70)

    # Create and connect embed client (reused for all operations)
    print("\n⚙ Connecting to embed service...")
    embed_client = get_llm_client("ollama", model="nomic-embed-text")
    await embed_client.connect()

    # Get file stats
    stats = get_file_stats(faiss_index, faiss_metadata)
    print(f"\nDataset Statistics:")
    print(f"  Vectors: {stats['n_vectors']:,}")
    print(f"  Documents: {stats['n_docs']:,}")
    print(f"  FAISS index size: {stats['index_size']:,} bytes ({stats['index_size']/1024:.1f} KB)")
    print(f"  JSON metadata size: {stats['metadata_size']:,} bytes ({stats['metadata_size']/1024:.1f} KB)")
    print(f"  Total size: {stats['total_size']:,} bytes ({stats['total_size']/1024:.1f} KB)")

    # Create HDF5 file if it doesn't exist
    if not hdf5_file.exists():
        print(f"\n⚙ Creating HDF5 file from FAISS+JSON...")
        start = time.perf_counter()

        # Load from FAISS+JSON
        async with FaissRAGService(
            embed_client=embed_client,
            index_path=faiss_index,
            metadata_path=faiss_metadata
        ) as rag:
            # Create new HDF5 service (will be empty initially)
            h5_rag = HDF5RAGService.__new__(HDF5RAGService)
            h5_rag.embed_client = embed_client
            h5_rag.hdf5_path = hdf5_file
            h5_rag.data_dir = data_dir
            h5_rag.embedding_dim = rag.embedding_dim
            h5_rag.model_name = rag.model_name
            h5_rag._initialized = True

            # Copy data
            h5_rag.index = rag.index
            h5_rag.chunk_refs = rag.chunk_refs
            h5_rag.documents = rag.documents

            # Save to HDF5
            h5_rag.save()

        end = time.perf_counter()
        print(f"  Conversion time: {(end-start)*1000:.1f} ms")

    # Get HDF5 file size
    h5_size = hdf5_file.stat().st_size
    print(f"  HDF5 file size: {h5_size:,} bytes ({h5_size/1024:.1f} KB)")
    print(f"  Size ratio (HDF5/FAISS+JSON): {h5_size/stats['total_size']:.2f}x")

    # Benchmark FaissRAGService (FAISS+JSON)
    print(f"\n{'─' * 70}")
    print("Benchmark 1: FaissRAGService (FAISS + JSON)")
    print(f"{'─' * 70}")
    faiss_times = await benchmark_faiss_load(embed_client, faiss_index, faiss_metadata, n_runs=5)
    faiss_avg = sum(faiss_times) / len(faiss_times)
    faiss_min = min(faiss_times)
    faiss_max = max(faiss_times)

    # Benchmark HDF5RAGService
    print(f"\n{'─' * 70}")
    print("Benchmark 2: HDF5RAGService (HDF5)")
    print(f"{'─' * 70}")
    hdf5_times = await benchmark_hdf5_load(embed_client, hdf5_file, n_runs=5)
    hdf5_avg = sum(hdf5_times) / len(hdf5_times)
    hdf5_min = min(hdf5_times)
    hdf5_max = max(hdf5_times)

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"\nFaissRAGService (FAISS + JSON):")
    print(f"  Average: {faiss_avg*1000:.1f} ms")
    print(f"  Min:     {faiss_min*1000:.1f} ms")
    print(f"  Max:     {faiss_max*1000:.1f} ms")

    print(f"\nHDF5RAGService (HDF5):")
    print(f"  Average: {hdf5_avg*1000:.1f} ms")
    print(f"  Min:     {hdf5_min*1000:.1f} ms")
    print(f"  Max:     {hdf5_max*1000:.1f} ms")

    print(f"\nPerformance Comparison:")
    speedup = faiss_avg / hdf5_avg
    if speedup > 1:
        print(f"  HDF5 is {speedup:.2f}x FASTER than FAISS+JSON")
    else:
        print(f"  HDF5 is {1/speedup:.2f}x SLOWER than FAISS+JSON")

    print(f"\nFile Size Comparison:")
    print(f"  FAISS+JSON: {stats['total_size']/1024:.1f} KB (2 files)")
    print(f"  HDF5:       {h5_size/1024:.1f} KB (1 file)")
    size_ratio = h5_size / stats['total_size']
    if size_ratio > 1:
        print(f"  HDF5 is {size_ratio:.2f}x LARGER")
    else:
        print(f"  HDF5 is {1/size_ratio:.2f}x SMALLER")

    print(f"\n{'=' * 70}")

    # Cleanup
    await embed_client.close()


if __name__ == "__main__":
    asyncio.run(main())
