#!/usr/bin/env python3
"""Test RAG search."""

import asyncio
from pathlib import Path
from microeval.llm import get_llm_client
from chatboti.generic_rag import GenericRAGService


async def main():
    # Initialize embed client
    embed_client = get_llm_client("ollama", model="nomic-embed-text")
    await embed_client.connect()

    # Load RAG
    rag = GenericRAGService(
        index_path=Path("vectors.faiss"),
        metadata_path=Path("metadata.json"),
        embedding_dim=768,
        embed_client=embed_client
    )

    print(f"✅ Loaded RAG: {len(rag.documents)} documents, {rag.index.ntotal} vectors\n")

    # Search
    query = "quantum computing"
    print(f"🔍 Searching for: '{query}'")
    results = await rag.search(query, k=3)

    print(f"📊 Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Document: {result.document_id}")
        print(f"   Field: {result.chunk_key}")
        print(f"   Text: {result.text[:100]}...")
        print()

    await embed_client.close()


if __name__ == "__main__":
    asyncio.run(main())
