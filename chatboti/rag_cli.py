"""CLI commands for GenericRAGService."""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from microeval.llm import get_llm_client, load_config


def make_model_slug(model_name: str) -> str:
    """Convert model name to filesystem-safe slug.

    :param model_name: Model name (e.g., 'nomic-embed-text', 'text-embedding-3-small')
    :return: Slug (e.g., 'nomic-embed-text', 'text-embedding-3-small')
    """
    # Replace non-alphanumeric with hyphens, remove :latest suffix
    slug = re.sub(r':latest$', '', model_name)
    slug = re.sub(r'[^a-z0-9]+', '-', slug.lower())
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug


async def detect_embedding_dim(embed_client) -> int:
    """Detect embedding dimension by running a test query.

    :param embed_client: Embedding client
    :return: Embedding dimension
    """
    print("→ Detecting embedding dimension...")
    test_embedding = await embed_client.embed("test")
    dim = len(test_embedding)
    print(f"✓ Detected dimension: {dim}")
    return dim


async def build_embeddings(
    csv_path: str = None,
    index_path: str = None,
    metadata_path: str = None
):
    """Build RAG embeddings from speaker CSV data.

    :param csv_path: Path to CSV file (default: chatboti/data/2025-09-02-speaker-bio.csv)
    :param index_path: Path to save FAISS index
    :param metadata_path: Path to save metadata JSON
    """
    from chatboti.generic_rag import GenericRAGService

    load_dotenv()

    # Get service and model from config first (needed for default paths)
    model_config = load_config()
    embed_models = model_config.get("embed_models", {})

    service = os.getenv("EMBED_SERVICE") or os.getenv("CHAT_SERVICE")
    model = os.getenv("EMBED_MODEL") or embed_models.get(service)

    if not service:
        print("✗ Error: EMBED_SERVICE or CHAT_SERVICE must be set")
        return 1

    if not model:
        print(f"✗ Error: EMBED_MODEL not set for service '{service}'")
        print(f"   Available models in config: {embed_models}")
        return 1

    # Create model-specific filenames
    model_slug = make_model_slug(model)

    # Default paths in chatboti/data directory with model slug
    data_dir = Path(__file__).parent / "data"
    if not csv_path:
        csv_path = str(data_dir / "2025-09-02-speaker-bio.csv")
    if not index_path:
        index_path = str(data_dir / f"vectors-{model_slug}.faiss")
    if not metadata_path:
        metadata_path = str(data_dir / f"metadata-{model_slug}.json")

    csv_path = Path(csv_path)
    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    # Check if CSV exists
    if not csv_path.exists():
        print(f"✗ Error: CSV file not found at {csv_path}")
        return 1

    print(f"• Service: {service}")
    print(f"• Model: {model}")
    print(f"• CSV: {csv_path}")
    print(f"• Index: {index_path}")
    print(f"• Metadata: {metadata_path}")
    print()

    # Create embed client using microeval
    print(f"→ Connecting to {service}...")
    embed_client = get_llm_client(service, model=model)
    await embed_client.connect()
    print(f"✓ Connected to {service}:{model}")
    print()

    # Detect embedding dimension
    embedding_dim = await detect_embedding_dim(embed_client)
    print()

    # Initialize RAG service
    print("→ Initializing GenericRAGService...")
    rag = GenericRAGService(
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_dim=embedding_dim,
        embed_client=embed_client
    )

    # Build embeddings
    print(f"→ Building embeddings from {csv_path}...")
    await rag.build_embeddings_from_documents(str(csv_path), "speaker")

    # Close connection
    await embed_client.close()

    # Summary
    print()
    print("✓ RAG embeddings built successfully!")
    print(f"  ├─ Documents: {len(rag.documents)}")
    print(f"  ├─ Chunks: {len(rag.chunk_refs)}")
    print(f"  ├─ Vectors: {rag.index.ntotal}")
    print(f"  └─ Saved to: {index_path} and {metadata_path}")

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
    :param index_path: Path to FAISS index
    :param metadata_path: Path to metadata JSON
    """
    from chatboti.generic_rag import GenericRAGService

    load_dotenv()

    # Get service and model from config
    model_config = load_config()
    embed_models = model_config.get("embed_models", {})

    service = os.getenv("EMBED_SERVICE") or os.getenv("CHAT_SERVICE")
    model = os.getenv("EMBED_MODEL") or embed_models.get(service)

    if not service or not model:
        print("✗ Error: EMBED_SERVICE and EMBED_MODEL must be set")
        return 1

    # Create model-specific filenames
    model_slug = make_model_slug(model)

    # Default paths in chatboti/data directory with model slug
    data_dir = Path(__file__).parent / "data"
    if not index_path:
        index_path = str(data_dir / f"vectors-{model_slug}.faiss")
    if not metadata_path:
        metadata_path = str(data_dir / f"metadata-{model_slug}.json")

    index_path = Path(index_path)
    metadata_path = Path(metadata_path)

    # Check if files exist
    if not index_path.exists() or not metadata_path.exists():
        print(f"✗ Error: RAG not found at {index_path}")
        print(f"   Run 'chatboti build-rag' first with EMBED_SERVICE={service} EMBED_MODEL={model}")
        return 1

    # Create embed client
    embed_client = get_llm_client(service, model=model)
    await embed_client.connect()

    # Detect dimension
    embedding_dim = await detect_embedding_dim(embed_client)

    # Load RAG
    rag = GenericRAGService(
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_dim=embedding_dim,
        embed_client=embed_client
    )

    print(f"✓ Loaded RAG: {len(rag.documents)} documents, {rag.index.ntotal} vectors\n")

    # Search
    print(f"→ Searching for: '{query}'")
    results = await rag.search(query, k=k)

    print(f"• Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Document: {result.document_id}")
        print(f"   Field: {result.chunk_key}")
        print(f"   Text: {result.text[:200]}...")
        print()

    await embed_client.close()
    return 0
