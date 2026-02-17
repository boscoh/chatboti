"""CLI commands for RAG services."""

import logging
from pathlib import Path
from typing import AsyncGenerator, Union

import h5py
from chatboti.llm import SimpleLLMClient
from rich.pretty import pprint

from chatboti.config import get_embed_client
from chatboti.faiss_rag import FaissRAGService
from chatboti.hdf5_rag import HDF5RAGService

logger = logging.getLogger(__name__)


async def create_rag_service(
    index_path: Union[str, Path] = None,
    metadata_path: Union[str, Path] = None,
    data_dir: Union[str, Path] = None,
    embed_client: SimpleLLMClient = None,
    verbose: bool = False,
) -> AsyncGenerator[Union[FaissRAGService, HDF5RAGService], None]:
    """Create and initialize a RAG service based on file extension.

    :param index_path: Path to index file (.faiss or .h5)
    :param metadata_path: Path to metadata JSON (for FAISS only)
    :param data_dir: Data directory path
    :param embed_client: Embedding client (will be created if None)
    :param verbose: Print service info when created
    :return: Async generator yielding the RAG service
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(data_dir)

    # Track if we created the client (so we know to close it)
    created_client = embed_client is None

    if created_client:
        embed_client = await get_embed_client()

    try:
        use_hdf5 = index_path and Path(index_path).suffix == ".h5"

        if verbose:
            print(f"• Format: {'HDF5' if use_hdf5 else 'FAISS'}")

        if use_hdf5:
            async with HDF5RAGService(
                embed_client=embed_client,
                data_dir=data_dir,
                hdf5_path=Path(index_path) if index_path else None,
            ) as rag:
                if verbose:
                    print(f"• HDF5 file: {rag.hdf5_path}")
                    print(f"• Embedding dim: {rag.embedding_dim}")
                yield rag
        else:
            async with FaissRAGService(
                embed_client=embed_client,
                data_dir=data_dir,
                index_path=Path(index_path) if index_path else None,
                metadata_path=Path(metadata_path) if metadata_path else None,
            ) as rag:
                if verbose:
                    print(f"• Index: {rag.index_path}")
                    print(f"• Metadata: {rag.metadata_path}")
                    print(f"• Embedding dim: {rag.embedding_dim}")
                yield rag
    finally:
        # Close the client if we created it
        if created_client and embed_client:
            await embed_client.close()
            # Give asyncio time to clean up aiohttp connections
            import asyncio

            await asyncio.sleep(0.1)


async def build_embeddings(
    csv_path: str = None, index_path: str = None, metadata_path: str = None
):
    """Build RAG embeddings from speaker CSV data.

    :param csv_path: Path to CSV file (default: chatboti/data/2025-09-02-speaker-bio.csv)
    :param index_path: Path to save FAISS index (or .h5 for HDF5 format)
    :param metadata_path: Path to save metadata JSON (not used if index_path is .h5)
    """
    data_dir = Path(__file__).parent / "data"
    csv_path = Path(csv_path) if csv_path else data_dir / "2025-09-02-speaker-bio.csv"

    if not csv_path.exists():
        logger.error(f"CSV file not found at {csv_path}")
        return 1

    print(f"• CSV: {csv_path}")

    async for rag in create_rag_service(
        index_path=index_path,
        metadata_path=metadata_path,
        data_dir=data_dir,
        verbose=True,
    ):
        print()
        print(f"→ Building embeddings from {csv_path}...")
        await rag.build_embeddings_from_documents(str(csv_path))

        print("\n✓ RAG embeddings built successfully!")
        print(f"  ├─ Documents: {len(rag.documents)}")
        print(f"  ├─ Chunks: {len(rag.chunk_refs)}")
        print(f"  └─ Vectors: {rag.index.ntotal}")


async def search_rag(
    query: str, k: int = 5, index_path: str = None, metadata_path: str = None
):
    """Search RAG index for relevant documents.

    :param query: Search query
    :param k: Number of results
    :param index_path: Path to index file (.faiss or .h5)
    :param metadata_path: Path to metadata JSON (for FAISS only)
    """
    data_dir = Path(__file__).parent / "data"

    async for rag in create_rag_service(
        index_path=index_path,
        metadata_path=metadata_path,
        data_dir=data_dir,
        verbose=True,
    ):
        print(f"✓ Loaded: {len(rag.documents)} documents, {rag.index.ntotal} vectors\n")
        print(f"→ Searching for: '{query}'")
        results = await rag.search(query, k=k, include_documents=True)

        print(f"• Found {len(results)} results:\n")
        for i, result in enumerate(results, 1):
            print(f"{'=' * 70}")
            print(f"Result {i}:")
            print(f"{'=' * 70}")
            if result.content:
                pprint(result.content)
            elif result.document_text:
                print(result.document_text)
            else:
                print("(no content)")
            print()


async def convert_to_hdf5(index_path: str, metadata_path: str, output_path: str):
    """Convert FAISS+JSON to HDF5 format.

    :param index_path: Path to FAISS index file
    :param metadata_path: Path to metadata JSON file
    :param output_path: Path to output HDF5 file
    """
    index_path_obj = Path(index_path)
    metadata_path_obj = Path(metadata_path)
    output_path_obj = Path(output_path)

    if not index_path_obj.exists():
        logger.error(f"Index file not found: {index_path_obj}")
        return 1

    if not metadata_path_obj.exists():
        logger.error(f"Metadata file not found: {metadata_path_obj}")
        return 1

    print(f"• Input index: {index_path_obj}")
    print(f"• Input metadata: {metadata_path_obj}")
    print(f"• Output HDF5: {output_path_obj}\n")

    data_dir = Path(__file__).parent / "data"

    async for rag_faiss in create_rag_service(
        index_path=index_path, metadata_path=metadata_path, data_dir=data_dir
    ):
        print(
            f"✓ Loaded: {len(rag_faiss.documents)} documents, {rag_faiss.index.ntotal} vectors\n"
        )

        print("→ Converting to HDF5 format...")
        async for rag_hdf5 in create_rag_service(
            index_path=output_path, data_dir=data_dir
        ):
            rag_hdf5.documents = rag_faiss.documents
            rag_hdf5.chunk_refs = rag_faiss.chunk_refs
            rag_hdf5.index = rag_faiss.index
            rag_hdf5.embedding_dim = rag_faiss.embedding_dim
            rag_hdf5.model_name = rag_faiss.model_name

            await rag_hdf5.save()
            print(f"✓ Saved to: {rag_hdf5.hdf5_path}")


async def convert_from_hdf5(
    input_path: str, index_path: str = None, metadata_path: str = None
):
    """Convert HDF5 to FAISS+JSON format.

    :param input_path: Path to HDF5 file
    :param index_path: Path to output FAISS index file
    :param metadata_path: Path to output metadata JSON file
    """
    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        logger.error(f"HDF5 file not found: {input_path_obj}")
        return 1

    print(f"• Input HDF5: {input_path_obj}\n")

    data_dir = Path(__file__).parent / "data"

    async for rag_hdf5 in create_rag_service(index_path=input_path, data_dir=data_dir):
        print(
            f"✓ Loaded: {len(rag_hdf5.documents)} documents, {rag_hdf5.index.ntotal} vectors\n"
        )

        print("→ Converting to FAISS+JSON format...")
        async for rag_faiss in create_rag_service(
            index_path=index_path, metadata_path=metadata_path, data_dir=data_dir
        ):
            rag_faiss.documents = rag_hdf5.documents
            rag_faiss.chunk_refs = rag_hdf5.chunk_refs
            rag_faiss.index = rag_hdf5.index
            rag_faiss.embedding_dim = rag_hdf5.embedding_dim
            rag_faiss.model_name = rag_hdf5.model_name

            await rag_faiss.save()
            print(f"✓ Saved index to: {rag_faiss.index_path}")
            print(f"✓ Saved metadata to: {rag_faiss.metadata_path}")


async def show_hdf5_info(hdf5_path: str):
    """Display HDF5 file metadata and statistics.

    :param hdf5_path: Path to HDF5 file
    """
    hdf5_path_obj = Path(hdf5_path)

    if not hdf5_path_obj.exists():
        logger.error(f"HDF5 file not found: {hdf5_path_obj}")
        return 1

    with h5py.File(hdf5_path_obj, "r") as f:
        print(f"HDF5 File: {hdf5_path_obj}")
        print(f"File size: {hdf5_path_obj.stat().st_size / 1024 / 1024:.2f} MB\n")

        print("Metadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")

        print("\nDatasets:")
        for name in f.keys():
            dataset = f[name]
            print(f"  {name}:")
            print(f"    Shape: {dataset.shape}")
            print(f"    Dtype: {dataset.dtype}")
            print(f"    Size: {dataset.nbytes / 1024 / 1024:.2f} MB")
