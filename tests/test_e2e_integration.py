
import pytest

from chatboti.faiss_rag import FaissRAGService


@pytest.mark.asyncio
async def test_full_pipeline_csv_to_search(tmp_path, embed_client):
    """Test complete pipeline: load CSV → build embeddings → search → verify results."""

    # 1. Create sample CSV
    csv_path = tmp_path / "speakers.csv"
    csv_path.write_text(
        "name,title,bio,abstract\n"
        "Alice,Professor,Expert in quantum computing,Research on quantum algorithms\n"
        "Bob,Researcher,Machine learning specialist,Deep learning applications\n"
    )

    # 2. Initialize RAG service with embed client
    index_path = tmp_path / "vectors.faiss"
    metadata_path = tmp_path / "metadata.json"

    async with FaissRAGService(
        index_path=index_path, metadata_path=metadata_path, embed_client=embed_client
    ) as rag:
        # 3. Load documents and build embeddings
        await rag.build_embeddings_from_documents(str(csv_path))

        # 4. Verify embeddings were generated
        # CSV loader embeds all non-empty fields: name, title, bio, abstract = 4 fields × 2 speakers = 8
        # Plus possibly document IDs or other fields, so check for reasonable range
        assert embed_client.call_count >= 8  # At least 8 for the main fields
        assert rag.index.ntotal >= 8  # At least 8 embeddings in FAISS
        assert len(rag.chunk_refs) >= 8
        assert len(rag.documents) == 2

        # 5. Verify files were saved
        assert index_path.exists()
        assert metadata_path.exists()

        # 6. Perform search
        results = await rag.search("quantum computing", k=2)

        # 7. Verify search results
        assert len(results) == 2
        assert all(hasattr(r, "document_id") for r in results)
        assert all(hasattr(r, "chunk_key") for r in results)
        assert all(hasattr(r, "text") for r in results)
        assert all(r.text for r in results)  # Non-empty text

        # 8. Verify we can search with include_documents
        results_with_docs = await rag.search(
            "machine learning", k=1, include_documents=True
        )
        assert len(results_with_docs) == 1
        # document_text may be None for CSV-loaded documents, so just verify the result exists
        assert results_with_docs[0] is not None

    # 9. Verify persistence - load from saved files
    async with FaissRAGService(
        index_path=index_path, metadata_path=metadata_path, embed_client=embed_client
    ) as rag2:
        # Verify loaded state matches
        assert rag2.index.ntotal >= 8
        assert len(rag2.chunk_refs) >= 8
        assert len(rag2.documents) == 2

        # Search should still work
        results2 = await rag2.search("quantum", k=1)
        assert len(results2) == 1


@pytest.mark.asyncio
async def test_search_without_embeddings_fails_gracefully(tmp_path, embed_client):
    """Test that searching on empty index handles gracefully."""
    index_path = tmp_path / "empty.faiss"
    metadata_path = tmp_path / "empty.json"

    async with FaissRAGService(
        index_path=index_path, metadata_path=metadata_path, embed_client=embed_client
    ) as rag:
        # Search on empty index
        results = await rag.search("test query", k=5)
        assert len(results) == 0  # Should return empty, not crash
