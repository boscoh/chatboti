"""CLI commands for GenericRAGService."""

# Standard library
import json
import os
from pathlib import Path

# Third-party
from dotenv import load_dotenv
from microeval.llm import get_llm_client, load_config
from rich.pretty import pprint

# Local
from chatboti.generic_rag import GenericRAGService

# Optional HDF5 support
try:
    from chatboti.hdf5_rag import HDF5RAGService
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    HDF5RAGService = None


def get_default_model(models_dict: dict, service: str) -> str:
    """Get the default model for a service (first in list or string value)."""
    models = models_dict.get(service, [])
    if isinstance(models, list) and models:
        return models[0]
    elif isinstance(models, str):
        return models
    return ""


async def build_embeddings(
    csv_path: str = None,
    index_path: str = None,
    metadata_path: str = None
):
    """Build RAG embeddings from speaker CSV data.

    :param csv_path: Path to CSV file (default: chatboti/data/2025-09-02-speaker-bio.csv)
    :param index_path: Path to save FAISS index (or .h5 for HDF5 format)
    :param metadata_path: Path to save metadata JSON (not used if index_path is .h5)
    """
    load_dotenv()

    # Get service and model from config
    model_config = load_config()
    embed_models = model_config.get("embed_models", {})

    service = os.getenv("EMBED_SERVICE") or os.getenv("CHAT_SERVICE")
    model = os.getenv("EMBED_MODEL") or get_default_model(embed_models, service)

    if not service:
        print("✗ Error: EMBED_SERVICE or CHAT_SERVICE must be set")
        return 1

    if not model:
        print(f"✗ Error: EMBED_MODEL not set for service '{service}'")
        print(f"   Available models in config: {embed_models}")
        return 1

    # Default CSV path
    data_dir = Path(__file__).parent / "data"
    if not csv_path:
        csv_path = str(data_dir / "2025-09-02-speaker-bio.csv")
    csv_path = Path(csv_path)

    # Check if CSV exists
    if not csv_path.exists():
        print(f"✗ Error: CSV file not found at {csv_path}")
        return 1

    # Detect if using HDF5 format
    use_hdf5 = index_path and Path(index_path).suffix == '.h5'

    if use_hdf5:
        if not HDF5_AVAILABLE:
            print("✗ Error: HDF5 support not available")
            print("   Install h5py: pip install h5py")
            return 1

        print(f"• Service: {service}")
        print(f"• Model: {model}")
        print(f"• CSV: {csv_path}")
        print(f"• Format: HDF5")
        print()

        # Create and connect embed client
        print(f"→ Connecting to {service}...")
        embed_client = get_llm_client(service, model=model)
        await embed_client.connect()

        # Create HDF5 RAG service
        async with HDF5RAGService(
            embed_client=embed_client,
            data_dir=data_dir,
            hdf5_path=Path(index_path)
        ) as rag:
            print(f"✓ Connected to {service}:{model}")
            print(f"• HDF5 file: {rag.hdf5_path}")
            print(f"• Embedding dim: {rag.embedding_dim}")
            print()

            # Build embeddings
            print(f"→ Building embeddings from {csv_path}...")
            await rag.build_embeddings_from_documents(str(csv_path))

            # Summary
            print()
            print("✓ RAG embeddings built successfully!")
            print(f"  ├─ Documents: {len(rag.documents)}")
            print(f"  ├─ Chunks: {len(rag.chunk_refs)}")
            print(f"  ├─ Vectors: {rag.index.ntotal}")
            print(f"  └─ Saved to: {rag.hdf5_path}")

        # Close embed client
        await embed_client.close()
    else:
        # Convert string paths to Path objects for factory method
        index_path_obj = Path(index_path) if index_path else None
        metadata_path_obj = Path(metadata_path) if metadata_path else None

        print(f"• Service: {service}")
        print(f"• Model: {model}")
        print(f"• CSV: {csv_path}")
        print()

        # Create and connect embed client
        print(f"→ Connecting to {service}...")
        embed_client = get_llm_client(service, model=model)
        await embed_client.connect()

        # Create RAG service using context manager
        async with GenericRAGService(
            embed_client=embed_client,
            data_dir=data_dir,
            index_path=index_path_obj,
            metadata_path=metadata_path_obj
        ) as rag:
            print(f"✓ Connected to {service}:{model}")
            print(f"• Index: {rag.index_path}")
            print(f"• Metadata: {rag.metadata_path}")
            print(f"• Embedding dim: {rag.embedding_dim}")
            print()

            # Build embeddings
            print(f"→ Building embeddings from {csv_path}...")
            await rag.build_embeddings_from_documents(str(csv_path))

            # Summary
            print()
            print("✓ RAG embeddings built successfully!")
            print(f"  ├─ Documents: {len(rag.documents)}")
            print(f"  ├─ Chunks: {len(rag.chunk_refs)}")
            print(f"  ├─ Vectors: {rag.index.ntotal}")
            print(f"  └─ Saved to: {rag.index_path} and {rag.metadata_path}")

        # Close embed client
        await embed_client.close()

    return 0


async def search_rag(
    query: str,
    k: int = 5,
    index_path: str = None,
    metadata_path: str = None
):
    """Search the RAG index.

    :param query: Search query
    :param k: Number of results to return
    :param index_path: Path to FAISS index (or .h5 for HDF5 format)
    :param metadata_path: Path to metadata JSON (not used if index_path is .h5)
    """
    load_dotenv()

    # Get service and model from config
    model_config = load_config()
    embed_models = model_config.get("embed_models", {})

    service = os.getenv("EMBED_SERVICE") or os.getenv("CHAT_SERVICE")
    model = os.getenv("EMBED_MODEL") or get_default_model(embed_models, service)

    if not service or not model:
        print("✗ Error: EMBED_SERVICE and EMBED_MODEL must be set")
        return 1

    # Detect if using HDF5 format
    use_hdf5 = index_path and Path(index_path).suffix == '.h5'

    if use_hdf5:
        if not HDF5_AVAILABLE:
            print("✗ Error: HDF5 support not available")
            print("   Install h5py: pip install h5py")
            return 1

        # Create and connect embed client
        data_dir = Path(__file__).parent / "data"
        embed_client = get_llm_client(service, model=model)
        await embed_client.connect()

        # Create HDF5 RAG service
        async with HDF5RAGService(
            embed_client=embed_client,
            data_dir=data_dir,
            hdf5_path=Path(index_path)
        ) as rag:
            print(f"✓ Loaded HDF5 RAG: {len(rag.documents)} documents, {rag.index.ntotal} vectors\n")

            # Search with documents included
            print(f"→ Searching for: '{query}'")
            results = await rag.search(query, k=k, include_documents=True)

            print(f"• Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"{'='*70}")
                print(f"Result {i}:")
                print(f"{'='*70}")
                if result.document_text:
                    try:
                        doc_json = json.loads(result.document_text)
                        pprint(doc_json)
                    except (json.JSONDecodeError, TypeError):
                        print(result.document_text)
                else:
                    print("(no document_text)")
                print()

        # Close embed client
        await embed_client.close()
    else:
        # Convert string paths to Path objects
        data_dir = Path(__file__).parent / "data"
        index_path_obj = Path(index_path) if index_path else None
        metadata_path_obj = Path(metadata_path) if metadata_path else None

        # Create and connect embed client
        embed_client = get_llm_client(service, model=model)
        await embed_client.connect()

        # Create RAG service using context manager
        async with GenericRAGService(
            embed_client=embed_client,
            data_dir=data_dir,
            index_path=index_path_obj,
            metadata_path=metadata_path_obj
        ) as rag:
            print(f"✓ Loaded RAG: {len(rag.documents)} documents, {rag.index.ntotal} vectors\n")

            # Search with documents included
            print(f"→ Searching for: '{query}'")
            results = await rag.search(query, k=k, include_documents=True)

            print(f"• Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"{'='*70}")
                print(f"Result {i}:")
                print(f"{'='*70}")
                if result.document_text:
                    try:
                        doc_json = json.loads(result.document_text)
                        pprint(doc_json)
                    except (json.JSONDecodeError, TypeError):
                        print(result.document_text)
                else:
                    print("(no document_text)")
                print()

        # Close embed client
        await embed_client.close()

    return 0


async def convert_to_hdf5(
    index_path: str,
    metadata_path: str,
    output_path: str
):
    """Convert FAISS+JSON format to HDF5 format.

    :param index_path: Path to FAISS index file (.faiss)
    :param metadata_path: Path to metadata JSON file
    :param output_path: Path to output HDF5 file (.h5)
    """
    if not HDF5_AVAILABLE:
        print("✗ Error: HDF5 support not available")
        print("   Install h5py: pip install h5py")
        return 1

    load_dotenv()

    # Get service and model from config
    model_config = load_config()
    embed_models = model_config.get("embed_models", {})

    service = os.getenv("EMBED_SERVICE") or os.getenv("CHAT_SERVICE")
    model = os.getenv("EMBED_MODEL") or get_default_model(embed_models, service)

    if not service or not model:
        print("✗ Error: EMBED_SERVICE and EMBED_MODEL must be set")
        return 1

    # Check input files exist
    index_path_obj = Path(index_path)
    metadata_path_obj = Path(metadata_path)
    output_path_obj = Path(output_path)

    if not index_path_obj.exists():
        print(f"✗ Error: FAISS index not found at {index_path_obj}")
        return 1

    if not metadata_path_obj.exists():
        print(f"✗ Error: Metadata JSON not found at {metadata_path_obj}")
        return 1

    print(f"→ Converting FAISS+JSON to HDF5...")
    print(f"  ├─ FAISS: {index_path_obj}")
    print(f"  ├─ JSON: {metadata_path_obj}")
    print(f"  └─ Output: {output_path_obj}")
    print()

    data_dir = Path(__file__).parent / "data"

    # Create and connect embed client
    embed_client = get_llm_client(service, model=model)
    await embed_client.connect()

    # Load from FAISS+JSON
    print("→ Loading FAISS+JSON format...")
    async with GenericRAGService(
        embed_client=embed_client,
        data_dir=data_dir,
        index_path=index_path_obj,
        metadata_path=metadata_path_obj
    ) as rag_faiss:
        print(f"✓ Loaded {len(rag_faiss.documents)} documents, {rag_faiss.index.ntotal} vectors")

        # Save to HDF5
        print(f"→ Converting to HDF5 format...")
        async with HDF5RAGService(
            embed_client=embed_client,
            data_dir=data_dir,
            hdf5_path=output_path_obj
        ) as rag_hdf5:
            # Transfer data
            rag_hdf5.documents = rag_faiss.documents
            rag_hdf5.chunk_refs = rag_faiss.chunk_refs
            rag_hdf5.index = rag_faiss.index
            rag_hdf5.embedding_dim = rag_faiss.embedding_dim

            # Save to HDF5
            rag_hdf5.save()
            print(f"✓ Saved to {output_path_obj}")

    # Close embed client
    await embed_client.close()

    print()
    print("✓ Conversion complete!")
    print(f"  ├─ Documents: {len(rag_faiss.documents)}")
    print(f"  ├─ Chunks: {len(rag_faiss.chunk_refs)}")
    print(f"  └─ Vectors: {rag_faiss.index.ntotal}")

    return 0


async def convert_from_hdf5(
    input_path: str,
    index_path: str = None,
    metadata_path: str = None
):
    """Convert HDF5 format to FAISS+JSON format.

    :param input_path: Path to HDF5 file (.h5)
    :param index_path: Path to output FAISS index file (.faiss)
    :param metadata_path: Path to output metadata JSON file
    """
    if not HDF5_AVAILABLE:
        print("✗ Error: HDF5 support not available")
        print("   Install h5py: pip install h5py")
        return 1

    load_dotenv()

    # Get service and model from config
    model_config = load_config()
    embed_models = model_config.get("embed_models", {})

    service = os.getenv("EMBED_SERVICE") or os.getenv("CHAT_SERVICE")
    model = os.getenv("EMBED_MODEL") or get_default_model(embed_models, service)

    if not service or not model:
        print("✗ Error: EMBED_SERVICE and EMBED_MODEL must be set")
        return 1

    # Check input file exists
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        print(f"✗ Error: HDF5 file not found at {input_path_obj}")
        return 1

    # Set default output paths if not provided
    data_dir = Path(__file__).parent / "data"
    if not index_path:
        index_path = str(data_dir / f"vectors-{model}.faiss")
    if not metadata_path:
        metadata_path = str(data_dir / f"metadata-{model}.json")

    index_path_obj = Path(index_path)
    metadata_path_obj = Path(metadata_path)

    print(f"→ Converting HDF5 to FAISS+JSON...")
    print(f"  ├─ Input: {input_path_obj}")
    print(f"  ├─ FAISS: {index_path_obj}")
    print(f"  └─ JSON: {metadata_path_obj}")
    print()

    # Create and connect embed client
    embed_client = get_llm_client(service, model=model)
    await embed_client.connect()

    # Load from HDF5
    print("→ Loading HDF5 format...")
    async with HDF5RAGService(
        embed_client=embed_client,
        data_dir=data_dir,
        hdf5_path=input_path_obj
    ) as rag_hdf5:
        print(f"✓ Loaded {len(rag_hdf5.documents)} documents, {rag_hdf5.index.ntotal} vectors")

        # Save to FAISS+JSON
        print(f"→ Converting to FAISS+JSON format...")
        async with GenericRAGService(
            embed_client=embed_client,
            data_dir=data_dir,
            index_path=index_path_obj,
            metadata_path=metadata_path_obj
        ) as rag_faiss:
            # Transfer data
            rag_faiss.documents = rag_hdf5.documents
            rag_faiss.chunk_refs = rag_hdf5.chunk_refs
            rag_faiss.index = rag_hdf5.index
            rag_faiss.embedding_dim = rag_hdf5.embedding_dim

            # Save to FAISS+JSON
            rag_faiss.save()
            print(f"✓ Saved to {index_path_obj} and {metadata_path_obj}")

    # Close embed client
    await embed_client.close()

    print()
    print("✓ Conversion complete!")
    print(f"  ├─ Documents: {len(rag_hdf5.documents)}")
    print(f"  ├─ Chunks: {len(rag_hdf5.chunk_refs)}")
    print(f"  └─ Vectors: {rag_hdf5.index.ntotal}")

    return 0


async def show_hdf5_info(hdf5_path: str):
    """Display HDF5 file metadata and statistics.

    :param hdf5_path: Path to HDF5 file (.h5)
    """
    if not HDF5_AVAILABLE:
        print("✗ Error: HDF5 support not available")
        print("   Install h5py: pip install h5py")
        return 1

    try:
        import h5py
    except ImportError:
        print("✗ Error: h5py not installed")
        print("   Install h5py: pip install h5py")
        return 1

    hdf5_path_obj = Path(hdf5_path)
    if not hdf5_path_obj.exists():
        print(f"✗ Error: HDF5 file not found at {hdf5_path_obj}")
        return 1

    print(f"→ Reading HDF5 file: {hdf5_path_obj}")
    print()

    with h5py.File(hdf5_path_obj, 'r') as f:
        # Read metadata attributes
        print("═" * 70)
        print("HDF5 File Metadata")
        print("═" * 70)

        if 'model_name' in f.attrs:
            print(f"Model name:      {f.attrs['model_name']}")
        if 'embedding_dim' in f.attrs:
            print(f"Embedding dim:   {f.attrs['embedding_dim']}")
        if 'created_at' in f.attrs:
            print(f"Created at:      {f.attrs['created_at']}")
        if 'updated_at' in f.attrs:
            print(f"Updated at:      {f.attrs['updated_at']}")

        print()
        print("═" * 70)
        print("Dataset Statistics")
        print("═" * 70)

        # Vector count
        if 'vectors' in f:
            vectors_shape = f['vectors'].shape
            print(f"Vectors:         {vectors_shape[0]:,} × {vectors_shape[1]}")
            print(f"Vector dtype:    {f['vectors'].dtype}")

            # Calculate size
            vector_bytes = f['vectors'].size * f['vectors'].dtype.itemsize
            vector_mb = vector_bytes / (1024 * 1024)
            print(f"Vector size:     {vector_mb:.2f} MB")

        # Chunk count
        if 'chunks' in f:
            chunk_count = len(f['chunks'])
            print(f"Chunks:          {chunk_count:,}")

        # Document count
        if 'documents' in f:
            doc_count = len(f['documents'])
            print(f"Documents:       {doc_count:,}")

            # List document IDs
            if doc_count <= 20:
                print()
                print("Document IDs:")
                for doc_id in f['documents'].keys():
                    doc_group = f['documents'][doc_id]
                    source = doc_group.attrs.get('source', 'N/A')
                    chunk_count = len(doc_group['chunks']) if 'chunks' in doc_group else 0
                    print(f"  • {doc_id} ({source}) - {chunk_count} chunks")
            else:
                print(f"  (showing first 10 of {doc_count})")
                print()
                print("Document IDs:")
                for i, doc_id in enumerate(list(f['documents'].keys())[:10]):
                    doc_group = f['documents'][doc_id]
                    source = doc_group.attrs.get('source', 'N/A')
                    chunk_count = len(doc_group['chunks']) if 'chunks' in doc_group else 0
                    print(f"  • {doc_id} ({source}) - {chunk_count} chunks")

        # File size
        print()
        print("═" * 70)
        print("File Information")
        print("═" * 70)
        file_size = hdf5_path_obj.stat().st_size
        file_mb = file_size / (1024 * 1024)
        print(f"File size:       {file_mb:.2f} MB")
        print(f"File path:       {hdf5_path_obj.absolute()}")

    print()
    print("✓ Info display complete")

    return 0
