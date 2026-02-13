"""CLI commands for GenericRAGService."""

import os
import json
from pathlib import Path
from rich.pretty import pprint
from dotenv import load_dotenv
from microeval.llm import load_config


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
    :param index_path: Path to save FAISS index
    :param metadata_path: Path to save metadata JSON
    """
    from chatboti.generic_rag import GenericRAGService

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

    # Convert string paths to Path objects for factory method
    index_path_obj = Path(index_path) if index_path else None
    metadata_path_obj = Path(metadata_path) if metadata_path else None

    print(f"• Service: {service}")
    print(f"• Model: {model}")
    print(f"• CSV: {csv_path}")
    print()

    # Create RAG service using context manager
    print(f"→ Connecting to {service}...")
    async with GenericRAGService(
        service_name=service,
        model=model,
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
    model = os.getenv("EMBED_MODEL") or get_default_model(embed_models, service)

    if not service or not model:
        print("✗ Error: EMBED_SERVICE and EMBED_MODEL must be set")
        return 1

    # Convert string paths to Path objects
    data_dir = Path(__file__).parent / "data"
    index_path_obj = Path(index_path) if index_path else None
    metadata_path_obj = Path(metadata_path) if metadata_path else None

    # Create RAG service using context manager
    async with GenericRAGService(
        service_name=service,
        model=model,
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

    return 0
