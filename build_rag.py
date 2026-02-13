#!/usr/bin/env python3
"""Build RAG embeddings from speaker CSV data."""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from microeval.llm import get_llm_client, load_config

from chatboti.generic_rag import GenericRAGService


async def detect_embedding_dim(embed_client) -> int:
    """Detect embedding dimension by running a test query.

    :param embed_client: Embedding client
    :return: Embedding dimension
    """
    print("🔍 Detecting embedding dimension...")
    test_embedding = await embed_client.embed("test")
    dim = len(test_embedding)
    print(f"✅ Detected dimension: {dim}")
    return dim


async def main():
    """Build RAG embeddings from speakers CSV."""
    load_dotenv()

    # Configuration
    csv_path = Path("chatboti/data/2025-09-02-speaker-bio.csv")
    index_path = Path("vectors.faiss")
    metadata_path = Path("metadata.json")

    # Check if CSV exists
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        return 1

    # Get service and model from config
    model_config = load_config()
    embed_models = model_config.get("embed_models", {})

    service = os.getenv("EMBED_SERVICE") or os.getenv("CHAT_SERVICE")
    model = os.getenv("EMBED_MODEL") or embed_models.get(service)

    if not service:
        print("❌ Error: EMBED_SERVICE or CHAT_SERVICE must be set")
        return 1

    if not model:
        print(f"❌ Error: EMBED_MODEL not set for service '{service}'")
        print(f"   Available models in config: {embed_models}")
        return 1

    print(f"🔧 Service: {service}")
    print(f"🔧 Model: {model}")
    print(f"📄 CSV: {csv_path}")
    print(f"💾 Index: {index_path}")
    print(f"💾 Metadata: {metadata_path}")
    print()

    # Create embed client using microeval
    print(f"🔌 Connecting to {service}...")
    embed_client = get_llm_client(service, model=model)
    await embed_client.connect()
    print(f"✅ Connected to {service}:{model}")
    print()

    # Detect embedding dimension
    embedding_dim = await detect_embedding_dim(embed_client)
    print()

    # Initialize RAG service
    print("🚀 Initializing GenericRAGService...")
    rag = GenericRAGService(
        index_path=index_path,
        metadata_path=metadata_path,
        embedding_dim=embedding_dim,
        embed_client=embed_client
    )

    # Build embeddings
    print(f"🔨 Building embeddings from {csv_path}...")
    await rag.build_embeddings_from_documents(str(csv_path), "speaker")

    # Close connection
    await embed_client.close()

    # Summary
    print()
    print("✅ RAG embeddings built successfully!")
    print(f"   📊 Documents: {len(rag.documents)}")
    print(f"   📊 Chunks: {len(rag.chunk_refs)}")
    print(f"   📊 Vectors: {rag.index.ntotal}")
    print(f"   💾 Saved to: {index_path} and {metadata_path}")

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
